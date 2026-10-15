# pyright: reportImplicitRelativeImport=false
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from pivot import compose, config, loaders, types
from pivot.storage import cache
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


@runtime_checkable
class Store(Protocol):
    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def prepare_output(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def commit(self, ref: types.ArtifactRef, path: pathlib.Path) -> str: ...

    def hash_artifact(
        self, ref: types.ArtifactRef, state_db: state_mod.StateDB | None = None
    ) -> types.HashInfo: ...

    def exists(self, ref: types.ArtifactRef) -> bool: ...


class StoreSpec(TypedDict):
    kind: str
    cache_dir: str
    project_root: str
    pipeline_name: str
    input_bindings: dict[str, str]


class CacheStore:
    _cache_dir: pathlib.Path
    _state_dir: pathlib.Path | None
    _ref_dir: pathlib.Path

    def __init__(self, cache_dir: pathlib.Path, state_dir: pathlib.Path | None) -> None:
        self._cache_dir = cache_dir
        self._state_dir = state_dir
        self._ref_dir = cache_dir / "refs"

    @contextlib.contextmanager
    def _state_db(self) -> Generator[state_mod.StateDB | None]:
        if self._state_dir is None:
            yield None
            return
        with state_mod.StateDB(self._state_dir) as db:
            yield db

    def _ref_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        key = ref.identity.key or compose.SINGLE_OUTPUT_KEY
        return self._ref_dir / ref.identity.producer / key

    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            raise FileNotFoundError(ref_path)

        with self._state_db() as db:
            if ref_path.is_dir():
                ref_hash, _ = cache.hash_directory(ref_path, db)
            else:
                ref_hash, _ = cache.hash_file(ref_path, db)

        cache_path = cache.get_cache_path(self._cache_dir, ref_hash)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        return cache_path

    def prepare_output(self, ref: types.ArtifactRef) -> pathlib.Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if ref.tag == types.ArtifactTag.DIRECTORY:
            return pathlib.Path(tempfile.mkdtemp(prefix="pivot_cache_out_", dir=self._cache_dir))

        fd, tmp_path = tempfile.mkstemp(prefix="pivot_cache_out_", dir=self._cache_dir)
        os.close(fd)
        return pathlib.Path(tmp_path)

    def commit(self, ref: types.ArtifactRef, path: pathlib.Path) -> str:
        with self._state_db() as db:
            output_hash = cache.save_to_cache(
                path,
                self._cache_dir,
                state_db=db,
                checkout_mode=config.CheckoutMode.SYMLINK,
            )

        cache_path = cache.get_cache_path(self._cache_dir, output_hash["hash"])
        ref_path = self._ref_path(ref)
        cache.remove_output(ref_path)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.symlink_to(cache_path.resolve())
        return output_hash["hash"]

    def hash_artifact(
        self, ref: types.ArtifactRef, state_db: state_mod.StateDB | None = None
    ) -> types.HashInfo:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            raise FileNotFoundError(ref_path)

        if state_db is not None:
            if ref_path.is_dir():
                tree_hash, manifest = cache.hash_directory(ref_path, state_db)
                return types.DirHash(hash=tree_hash, manifest=manifest)
            file_hash, _ = cache.hash_file(ref_path, state_db)
            return types.FileHash(hash=file_hash)

        with self._state_db() as db:
            if ref_path.is_dir():
                tree_hash, manifest = cache.hash_directory(ref_path, db)
                return types.DirHash(hash=tree_hash, manifest=manifest)

            file_hash, _ = cache.hash_file(ref_path, db)
            return types.FileHash(hash=file_hash)

    def exists(self, ref: types.ArtifactRef) -> bool:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            return False
        try:
            cache_path = self.checkout(ref)
        except FileNotFoundError:
            return False
        return cache_path.exists()


class WorkspaceStore:
    _project_root: pathlib.Path
    _pipeline_name: str
    _input_bindings: dict[str, str]
    _path_map: dict[str, types.ArtifactIdentity]
    _output_producers: set[str]
    _cache_dir: pathlib.Path | None

    def __init__(
        self,
        project_root: pathlib.Path,
        pipeline_name: str,
        input_bindings: dict[str, str],
        cache_dir: pathlib.Path | None = None,
    ) -> None:
        self._project_root = project_root.resolve()
        self._pipeline_name = pipeline_name
        self._input_bindings = input_bindings.copy()
        self._path_map = {}
        self._output_producers = set[str]()
        self._cache_dir = cache_dir.resolve() if cache_dir is not None else None

    def _is_input_ref(self, ref: types.ArtifactRef) -> bool:
        if ref.tag != types.ArtifactTag.DATA:
            return False
        if ref.identity.key is not None:
            return False
        if ref.identity.producer in self._output_producers:
            return False
        output_path = self._resolve_output_path(ref)
        return not (output_path.exists() or output_path.is_symlink())

    def _resolve_input_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        name = ref.identity.producer
        if name in self._input_bindings:
            binding = pathlib.Path(self._input_bindings[name])
            if binding.is_absolute():
                return binding
            return self._project_root / binding
        return self._project_root / "data" / "raw" / name

    def _resolve_output_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        return self._project_root / relative_output_path(ref)

    def _resolve_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        if self._is_input_ref(ref):
            path = self._resolve_input_path(ref)
        else:
            path = self._resolve_output_path(ref)

        resolved = path.resolve()
        existing = self._path_map.get(str(resolved))
        if existing is not None and existing != ref.identity:
            raise ValueError(
                f"Collision detected: {ref.identity} maps to {resolved} already used by {existing}"
            )
        self._path_map[str(resolved)] = ref.identity
        return path

    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path:
        return self._resolve_path(ref)

    def prepare_output(self, ref: types.ArtifactRef) -> pathlib.Path:
        self._output_producers.add(ref.identity.producer)
        path = self._resolve_path(ref)
        if ref.tag == types.ArtifactTag.DIRECTORY:
            cache.remove_output(path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.remove_output(path)
        return path

    def commit(self, ref: types.ArtifactRef, path: pathlib.Path) -> str:
        self._output_producers.add(ref.identity.producer)
        if path.is_dir():
            tree_hash, _ = cache.hash_directory(path, None)
            return tree_hash
        file_hash, _ = cache.hash_file(path, None)
        return file_hash

    def backup_to_cache(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        *,
        state_dir: pathlib.Path | None = None,
    ) -> None:
        if self._cache_dir is None:
            return
        if ref.tag is types.ArtifactTag.METRIC:
            return
        _ = cache.save_to_cache(
            path,
            self._cache_dir,
            state_db=None,
            checkout_mode=config.CheckoutMode.COPY,
            state_dir=state_dir,
        )

    def restore_from_cache(
        self,
        ref: types.ArtifactRef,
        output_hash: types.HashInfo,
        *,
        state_dir: pathlib.Path | None = None,
    ) -> bool:
        if self._cache_dir is None:
            return False
        if ref.tag is types.ArtifactTag.METRIC:
            return False
        output_path = self._resolve_output_path(ref)
        return cache.restore_from_cache(
            output_path,
            output_hash,
            self._cache_dir,
            checkout_mode=config.CheckoutMode.COPY,
            state_dir=state_dir,
        )

    def hash_artifact(
        self, ref: types.ArtifactRef, state_db: state_mod.StateDB | None = None
    ) -> types.HashInfo:
        path = self._resolve_path(ref)
        if ref.tag == types.ArtifactTag.DIRECTORY or path.is_dir():
            tree_hash, manifest = cache.hash_directory(path, state_db)
            return types.DirHash(hash=tree_hash, manifest=manifest)
        file_hash, _ = cache.hash_file(path, state_db)
        return types.FileHash(hash=file_hash)

    def cache_contains(self, ref: types.ArtifactRef, output_hash: types.HashInfo) -> bool:
        """Return whether this output can be restored from the local cache."""
        if self._cache_dir is None or ref.tag is types.ArtifactTag.METRIC:
            return False
        if types.is_dir_hash(output_hash):
            return all(
                cache.get_cache_path(self._cache_dir, entry["hash"]).exists()
                for entry in output_hash["manifest"]
            )
        return cache.get_cache_path(self._cache_dir, output_hash["hash"]).exists()

    def resolve_display_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        """Resolve the workspace path for an artifact, for display/CLI use.

        Unlike ``checkout``/``prepare_output``, this does not mutate internal
        state (collision map, output producers set).  It always treats the ref
        as an output, which is correct for stage-produced artifacts shown in
        CLI commands.
        """
        return self._resolve_output_path(ref)

    def output_identities_by_path(
        self, outs: Iterable[types.ArtifactRef]
    ) -> dict[str, types.ArtifactIdentity]:
        """Map each output's workspace path (relative to the project root) to its identity.

        The inverse of ``resolve_display_path`` over stage outputs, for resolving a
        user-typed file path to the artifact it belongs to.
        """
        return {
            str(self.resolve_display_path(out).relative_to(self._project_root)): out.identity
            for out in outs
        }

    def exists(self, ref: types.ArtifactRef) -> bool:
        path = self._resolve_path(ref)
        return path.exists()


def relative_output_path(ref: types.ArtifactRef) -> pathlib.Path:
    """Return an output's workspace path relative to the project root.

    This is the one rule that places outputs: ``<prefix>/<stage>.<ext>`` for a single output,
    ``<prefix>/<stage>/<key>.<ext>`` for a keyed one, and no extension for a directory. The
    prefix is ``metrics`` or ``plots`` for tagged outputs and ``data`` otherwise.
    """
    stage_name = ref.identity.producer
    prefix = "data"
    if ref.tag == types.ArtifactTag.METRIC:
        prefix = "metrics"
    elif ref.tag == types.ArtifactTag.PLOT:
        prefix = "plots"

    key = ref.identity.key
    if ref.tag == types.ArtifactTag.DIRECTORY:
        if key is None:
            return pathlib.Path(prefix) / stage_name
        return pathlib.Path(prefix) / stage_name / key

    ext = loaders.format_extension(ref.format)
    if key is None:
        return pathlib.Path(prefix) / f"{stage_name}.{ext}"
    return pathlib.Path(prefix) / stage_name / f"{key}.{ext}"


def store_from_spec(spec: StoreSpec) -> Store:
    kind = spec["kind"]
    if kind == "cache":
        return CacheStore(
            cache_dir=pathlib.Path(spec["cache_dir"]),
            state_dir=pathlib.Path(spec["project_root"]) / ".pivot",
        )
    if kind == "workspace":
        cache_dir_str = spec.get("cache_dir")
        return WorkspaceStore(
            project_root=pathlib.Path(spec["project_root"]),
            pipeline_name=spec["pipeline_name"],
            input_bindings=spec["input_bindings"],
            cache_dir=pathlib.Path(cache_dir_str) if cache_dir_str else None,
        )
    raise ValueError(f"Unknown store kind: {kind}")
