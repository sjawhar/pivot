# pyright: reportImplicitRelativeImport=false
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from pivot import config, loaders, stage_def, types
from pivot.storage import cache
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    from collections.abc import Generator


@runtime_checkable
class Store(Protocol):
    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def prepare_output(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def commit(self, ref: types.ArtifactRef, path: pathlib.Path) -> str: ...

    def hash_artifact(self, ref: types.ArtifactRef) -> types.HashInfo: ...

    def exists(self, ref: types.ArtifactRef) -> bool: ...


class StoreSpec(TypedDict):
    kind: str
    cache_dir: str
    project_root: str
    pipeline_name: str
    input_bindings: dict[str, str]


class CacheStore:
    _cache_dir: pathlib.Path
    _state_db_path: pathlib.Path | None
    _ref_dir: pathlib.Path

    def __init__(self, cache_dir: pathlib.Path, state_db_path: pathlib.Path | None) -> None:
        self._cache_dir = cache_dir
        self._state_db_path = state_db_path
        self._ref_dir = cache_dir / "refs"

    @contextlib.contextmanager
    def _state_db(self) -> Generator[state_mod.StateDB | None]:
        if self._state_db_path is None:
            yield None
            return
        with state_mod.StateDB(self._state_db_path) as db:
            yield db

    def _ref_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        key = ref.identity.key or stage_def.SINGLE_OUTPUT_KEY
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

    def hash_artifact(self, ref: types.ArtifactRef) -> types.HashInfo:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            raise FileNotFoundError(ref_path)

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

    def __init__(
        self,
        project_root: pathlib.Path,
        pipeline_name: str,
        input_bindings: dict[str, str],
    ) -> None:
        self._project_root = project_root.resolve()
        self._pipeline_name = pipeline_name
        self._input_bindings = input_bindings.copy()
        self._path_map = {}
        self._output_producers = set[str]()

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

    def _format_extension(self, fmt: object) -> str:
        match fmt:
            case loaders.DataFrameJSONL():
                return "jsonl"
            case loaders.CSV():
                return "csv"
            case loaders.YAML():
                return "yaml"
            case loaders.JSON():
                return "json"
            case loaders.Text():
                return "txt"
            case loaders.Pickle():
                return "pkl"
            case loaders.MatplotlibFigure():
                return "png"
            case _:
                return "dat"

    def _resolve_output_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        stage_name = ref.identity.producer
        prefix = "data"
        if ref.tag == types.ArtifactTag.METRIC:
            prefix = "metrics"
        elif ref.tag == types.ArtifactTag.PLOT:
            prefix = "plots"

        key = ref.identity.key
        if ref.tag == types.ArtifactTag.DIRECTORY:
            if key is None:
                rel = pathlib.Path(prefix) / self._pipeline_name / stage_name
            else:
                rel = pathlib.Path(prefix) / self._pipeline_name / stage_name / key
            return self._project_root / rel

        ext = self._format_extension(ref.format)
        if key is None:
            rel = pathlib.Path(prefix) / self._pipeline_name / f"{stage_name}.{ext}"
        else:
            rel = pathlib.Path(prefix) / self._pipeline_name / stage_name / f"{key}.{ext}"
        return self._project_root / rel

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
            path.mkdir(parents=True, exist_ok=True)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commit(self, ref: types.ArtifactRef, path: pathlib.Path) -> str:
        self._output_producers.add(ref.identity.producer)
        if ref.tag == types.ArtifactTag.DIRECTORY:
            tree_hash, _ = cache.hash_directory(path, None)
            return tree_hash
        file_hash, _ = cache.hash_file(path, None)
        return file_hash

    def hash_artifact(self, ref: types.ArtifactRef) -> types.HashInfo:
        path = self._resolve_path(ref)
        if ref.tag == types.ArtifactTag.DIRECTORY:
            tree_hash, manifest = cache.hash_directory(path, None)
            return types.DirHash(hash=tree_hash, manifest=manifest)
        file_hash, _ = cache.hash_file(path, None)
        return types.FileHash(hash=file_hash)

    def resolve_display_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        """Resolve the workspace path for an artifact, for display/CLI use.

        Unlike ``checkout``/``prepare_output``, this does not mutate internal
        state (collision map, output producers set).  It always treats the ref
        as an output, which is correct for stage-produced artifacts shown in
        CLI commands.
        """
        return self._resolve_output_path(ref)

    def exists(self, ref: types.ArtifactRef) -> bool:
        path = self._resolve_path(ref)
        return path.exists()


def store_from_spec(spec: StoreSpec) -> Store:
    kind = spec["kind"]
    if kind == "cache":
        project_root = pathlib.Path(spec["project_root"])
        state_db_path = project_root / ".pivot"
        return CacheStore(
            cache_dir=pathlib.Path(spec["cache_dir"]),
            state_db_path=state_db_path,
        )
    if kind == "workspace":
        return WorkspaceStore(
            project_root=pathlib.Path(spec["project_root"]),
            pipeline_name=spec["pipeline_name"],
            input_bindings=spec["input_bindings"],
        )
    raise ValueError(f"Unknown store kind: {kind}")
