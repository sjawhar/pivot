from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from pivot import config, stage_def, types
from pivot.storage import cache
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


@runtime_checkable
class Store(Protocol):
    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def prepare_output(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def commit(self, ref: types.ArtifactRef, path: pathlib.Path) -> types.HashInfo: ...

    def hash_artifact(
        self,
        ref: types.ArtifactRef,
        state_db: state_mod.StateDB | None = None,
        *,
        pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    ) -> types.HashInfo: ...

    def exists(self, ref: types.ArtifactRef) -> bool: ...


class StoreSpec(TypedDict):
    kind: str
    cache_dir: str
    project_root: str
    pipeline_name: str
    input_bindings: dict[str, str]
    checkout_modes: list[config.CheckoutMode]


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
        key = ref.identity.key or stage_def.SINGLE_OUTPUT_KEY
        return self._ref_dir / ref.identity.producer / key

    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            raise FileNotFoundError(ref_path)

        with self._state_db() as db:
            if ref_path.is_dir():
                ref_hash = cache.hash_directory(ref_path, db)["hash"]
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

    def commit(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        *,
        state_dir: pathlib.Path | None = None,
    ) -> types.HashInfo:
        _ = state_dir
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
        return output_hash

    def hash_artifact(
        self,
        ref: types.ArtifactRef,
        state_db: state_mod.StateDB | None = None,
        *,
        pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    ) -> types.HashInfo:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            raise FileNotFoundError(ref_path)

        if state_db is not None:
            if ref_path.is_dir():
                return cache.hash_directory(ref_path, state_db, pending=pending)
            file_hash, _ = cache.hash_file(ref_path, state_db, pending=pending)
            return types.FileHash(hash=file_hash)

        with self._state_db() as db:
            if ref_path.is_dir():
                return cache.hash_directory(ref_path, db, pending=pending)

            file_hash, _ = cache.hash_file(ref_path, db, pending=pending)
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
    _cache_dir: pathlib.Path | None
    _checkout_modes: list[config.CheckoutMode] | None

    def __init__(
        self,
        project_root: pathlib.Path,
        pipeline_name: str,
        input_bindings: dict[str, str],
        cache_dir: pathlib.Path | None = None,
        checkout_modes: list[config.CheckoutMode] | None = None,
    ) -> None:
        """``checkout_modes`` orders how outputs are linked to the cache; None means the default."""
        self._project_root = project_root.resolve()
        self._pipeline_name = pipeline_name
        self._input_bindings = input_bindings.copy()
        self._path_map = {}
        self._cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self._checkout_modes = checkout_modes

    @property
    def cache_dir(self) -> pathlib.Path | None:
        """The cache object root (``<cache.dir>/files``), or None when this store has no cache."""
        return self._cache_dir

    def _resolve_input_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        name = ref.identity.producer
        if name in self._input_bindings:
            return self._project_root / self._input_bindings[name]
        return self._project_root / "data" / "raw" / name

    def _resolve_output_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        path = self._project_root / relative_output_path(ref)
        logical_path = pathlib.Path(os.path.normpath(path))
        if not logical_path.is_relative_to(self._project_root):
            raise ValueError(
                f"Output {types.identity_key(ref.identity)!r} maps outside project root: {path}"
            )
        return path

    def _resolve_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        if ref.tag is types.ArtifactTag.INPUT:
            path = self._resolve_input_path(ref)
        else:
            path = self._resolve_output_path(ref)

        # Logical path, not resolve(): two declared paths that are symlinks to one file (a
        # symlink checkout, two symlinked inputs) are distinct artifacts.
        logical = os.path.normpath(path)
        existing = self._path_map.get(logical)
        if existing is not None and existing != ref.identity:
            raise ValueError(
                f"Collision detected: {ref.identity} maps to {logical} already used by {existing}"
            )
        self._path_map[logical] = ref.identity
        return path

    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path:
        return self._resolve_path(ref)

    def prepare_output(self, ref: types.ArtifactRef) -> pathlib.Path:
        path = self._resolve_path(ref)
        if ref.tag == types.ArtifactTag.DIRECTORY:
            cache.remove_output(path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.remove_output(path)
        return path

    def commit(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        *,
        state_dir: pathlib.Path | None = None,
        warn_left_out_symlinks: bool = True,
    ) -> types.HashInfo:
        """Hash a written output once and, unless it is a metric, place it in the cache.

        ``warn_left_out_symlinks`` logs the warning naming the symlinks a directory output
        leaves out; a quiet command turns it off.
        """
        output_hash: types.HashInfo
        if path.is_dir():
            output_hash = cache.hash_directory(
                path,
                None,
                cache_dir=self._cache_dir,
                warn_left_out_symlinks=warn_left_out_symlinks,
            )
        else:
            file_hash, _ = cache.hash_file(path, None)
            output_hash = types.FileHash(hash=file_hash)
        if self._cache_dir is not None and ref.tag is not types.ArtifactTag.METRIC:
            cache.store_hashed_output(
                path,
                output_hash,
                self._cache_dir,
                self._checkout_modes,
                state_dir=state_dir,
            )
        return output_hash

    def restore_from_cache(
        self,
        ref: types.ArtifactRef,
        output_hash: types.HashInfo,
        *,
        state_dir: pathlib.Path | None = None,
        checkout_modes: list[config.CheckoutMode] | None = None,
        recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    ) -> bool:
        """Restore ``ref`` from the cache; ``recorded`` is as ``cache.restore_from_cache``
        says."""
        if self._cache_dir is None:
            return False
        if ref.tag is types.ArtifactTag.METRIC:
            return False
        output_path = self._resolve_output_path(ref)
        return cache.restore_from_cache(
            output_path,
            output_hash,
            self._cache_dir,
            checkout_modes=checkout_modes if checkout_modes is not None else self._checkout_modes,
            state_dir=state_dir,
            recorded=recorded,
        )

    def hash_artifact(
        self,
        ref: types.ArtifactRef,
        state_db: state_mod.StateDB | None = None,
        *,
        pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    ) -> types.HashInfo:
        path = self._resolve_path(ref)
        if ref.directory is None and (ref.tag == types.ArtifactTag.DIRECTORY or path.is_dir()):
            return cache.hash_directory(path, state_db, cache_dir=self._cache_dir, pending=pending)
        file_hash, _ = cache.hash_file(path, state_db, pending=pending)
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
    prefix is ``metrics`` or ``plots`` for tagged outputs and ``data`` otherwise. One file of a
    directory output is placed at its path inside the directory (its key ends with that path).
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

    from pivot import (
        loaders,
    )  # loaders imports pandas; keep it off CLI paths that never place outputs

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
            checkout_modes=spec["checkout_modes"],
        )
    raise ValueError(f"Unknown store kind: {kind}")
