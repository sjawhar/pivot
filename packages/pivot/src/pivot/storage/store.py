from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from pivot import config, ignore, stage_def, types
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing, ownership
from pivot.storage import state as state_mod

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping


def _record(
    hashed: listing.HashedDirectory, hashed_dirs: dict[str, listing.HashedDirectory] | None
) -> types.DirHash:
    """A hashed directory's record, the directory kept in ``hashed_dirs`` (by hash) if given."""
    if hashed_dirs is not None:
        hashed_dirs[hashed["record"]["hash"]] = hashed
    return hashed["record"]


@runtime_checkable
class Store(Protocol):
    def checkout(self, ref: types.ArtifactRef) -> pathlib.Path: ...

    def prepare_output(
        self,
        ref: types.ArtifactRef,
        claims: ownership.StageClaims | None = None,
        returned: ownership.StageMembers | None = None,
        *,
        state_dir: pathlib.Path | None = None,
    ) -> pathlib.Path: ...

    def commit(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        *,
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
    ) -> types.HashInfo: ...

    def hash_artifact(
        self,
        ref: types.ArtifactRef,
        state_db: state_mod.StateDB | None = None,
        *,
        pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
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
            is_directory = ref_path.is_dir()
            if is_directory:
                ref_hash = cache.hash_directory(ref_path, db)["record"]["hash"]
            else:
                ref_hash, _ = cache.hash_file(ref_path, db)

        cache_path = (
            cache.resolve_cache_tree_path(self._cache_dir, ref_hash)
            if is_directory
            else cache.resolve_cache_object_path(self._cache_dir, ref_hash)
        )
        if cache_path is None:
            raise FileNotFoundError(ref_hash)
        return cache_path

    def prepare_output(
        self,
        ref: types.ArtifactRef,
        claims: ownership.StageClaims | None = None,
        returned: ownership.StageMembers | None = None,
        *,
        state_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """A fresh path in the cache for the output. Nothing in the workspace is replaced, so
        ``claims`` and ``returned`` (the stage's ``*.pvt`` files there) and ``state_dir`` (for
        a workspace path's restore lock) do not apply."""
        _ = claims, returned, state_dir
        state_dir_mod.make_dirs(self._cache_dir)
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
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
    ) -> types.HashInfo:
        _ = state_dir
        with self._state_db() as db:
            output_hash = cache.save_to_cache(
                path,
                self._cache_dir,
                state_db=db,
                checkout_mode=config.CheckoutMode.SYMLINK,
                hashed_dirs=hashed_dirs,
            )

        cache_path = (
            cache.tree_path(self._cache_dir, output_hash["hash"])
            if types.is_dir_hash(output_hash)
            else cache.get_cache_path(self._cache_dir, output_hash["hash"])
        )
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
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
    ) -> types.HashInfo:
        ref_path = self._ref_path(ref)
        if not ref_path.exists() and not ref_path.is_symlink():
            raise FileNotFoundError(ref_path)

        with contextlib.ExitStack() as stack:
            db = state_db if state_db is not None else stack.enter_context(self._state_db())
            if ref_path.is_dir():
                return _record(cache.hash_directory(ref_path, db, pending=pending), hashed_dirs)
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
    _output_paths: dict[types.ArtifactRef, pathlib.Path]
    _cache_dir: pathlib.Path | None
    _checkout_modes: list[config.CheckoutMode] | None
    _ignore_filter: ignore.IgnoreFilter

    def __init__(
        self,
        project_root: pathlib.Path,
        pipeline_name: str,
        input_bindings: dict[str, str],
        cache_dir: pathlib.Path | None = None,
        checkout_modes: list[config.CheckoutMode] | None = None,
        ignore_filter: ignore.IgnoreFilter | None = None,
    ) -> None:
        """``checkout_modes`` orders how outputs are linked to the cache; None means the default.

        ``ignore_filter`` names the files of a directory input that its hash leaves out; pass
        the command's one filter where it has one. None makes one for this project's
        ``.pivotignore``, read when a directory input is first hashed.
        """
        self._project_root = project_root.resolve()
        self._pipeline_name = pipeline_name
        self._input_bindings = input_bindings.copy()
        self._path_map = {}
        self._output_paths = {}
        self._cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self._checkout_modes = checkout_modes
        self._ignore_filter = (
            ignore_filter if ignore_filter is not None else ignore.IgnoreFilter(self._project_root)
        )

    @property
    def cache_dir(self) -> pathlib.Path | None:
        """The cache object root (``<cache.dir>/files``), or None when this store has no cache."""
        return self._cache_dir

    def _input_binding(self, identity: types.ArtifactIdentity) -> str:
        """An input's path relative to the project root."""
        name = identity.producer
        if name in self._input_bindings:
            return self._input_bindings[name]
        return f"data/raw/{name}"

    def _resolve_input_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        return self._project_root / self._input_binding(ref.identity)

    def input_member_filter(self, identity: types.ArtifactIdentity) -> ignore.MemberFilter | None:
        """The ``.pivotignore`` rule for the files of input directory ``identity``.

        Each file is matched on its path inside the project (the input's binding plus its
        relpath), never on a symlink target. None when no file of it can be left out
        (``ignore.IgnoreFilter.member_filter``).
        """
        return self._ignore_filter.member_filter(os.path.normpath(self._input_binding(identity)))

    def _input_member_filter(self, ref: types.ArtifactRef) -> ignore.MemberFilter | None:
        """The ``.pivotignore`` rule for the files of directory dependency ``ref``.

        Only a pipeline input has one. A stage output has none: its dependency hash is the
        hash its producer records, which is cached and restored whole.
        """
        if ref.tag is not types.ArtifactTag.INPUT:
            return None
        return self.input_member_filter(ref.identity)

    def _resolve_output_path(self, ref: types.ArtifactRef) -> pathlib.Path:
        # A stage asks for each output's path several times (its write, each ownership check,
        # its commit), and the placement depends on the ref alone: compute it once per ref
        cached = self._output_paths.get(ref)
        if cached is not None:
            return cached
        path = self._project_root / relative_output_path(ref)
        logical_path = pathlib.Path(os.path.normpath(path))
        if not logical_path.is_relative_to(self._project_root):
            raise ValueError(
                f"Output {types.identity_key(ref.identity)!r} maps outside project root: {path}"
            )
        self._output_paths[ref] = path
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

    def refuse_output_replacements(
        self,
        refs: Iterable[types.ArtifactRef],
        claims: ownership.StageClaims | None = None,
        returned: Mapping[types.ArtifactRef, ownership.StageMembers] | None = None,
    ) -> None:
        """Fail before replacing these outputs removes a user's tracked path or `.pvt` sidecar.

        Each output's own ``*.pvt`` files are ``ownership.stage_members`` of ``claims`` (the
        stage's) and of what this run writes there (``returned``).
        """
        replacements = list[ownership.Replacement]()
        written = dict[pathlib.Path, ownership.StageMembers]()
        for ref in refs:
            path = ownership.normalized(self._resolve_output_path(ref))
            replacements.append(ownership.stage_output(path, ref.identity.producer))
            if returned is not None and ref in returned:
                written[path] = returned[ref]
        ownership.refuse_output_replacements(
            self._project_root,
            replacements,
            lambda path: ownership.stage_members(path, claims, returned=written.get(path)),
        )

    def _write_target(
        self,
        ref: types.ArtifactRef,
        claims: ownership.StageClaims | None,
        returned: ownership.StageMembers | None,
    ) -> ownership.ForcedRestore:
        """The forced-restore target for a stage write of ``ref``: its own ``*.pvt`` files are
        ``ownership.stage_members`` of ``claims`` and of what this run writes (``returned``)."""
        output_path = ownership.normalized(self._resolve_output_path(ref))
        return ownership.ForcedRestore(
            replacement=ownership.stage_output(output_path, ref.identity.producer),
            restored=None,
            claims=None if claims is None else lambda: claims,
            written={} if returned is None else returned,
        )

    def refuse_output_replacement_at_write(
        self,
        ref: types.ArtifactRef,
        claims: ownership.StageClaims | None = None,
        returned: ownership.StageMembers | None = None,
    ) -> None:
        """Run the forced-restore refusal after an output is prepared and before its writer
        writes, so a user file that landed in that window is named instead of becoming a loader
        error."""
        target = self._write_target(ref, claims, returned)
        refusal = ownership.forced_restore_refusal(self._project_root, [target])
        if refusal is not None:
            raise refusal

    def prepare_output(
        self,
        ref: types.ArtifactRef,
        claims: ownership.StageClaims | None = None,
        returned: ownership.StageMembers | None = None,
        *,
        state_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Clear the output's path for the stage to write, refusing first if that would remove
        a user's file (``refuse_output_replacements``).

        The old output is checked again just before it is removed (``cache.remove_output_checked``,
        ``ownership.swap_refusal``): a directory renamed aside, so a user's file copied into it
        after the first check is named and put back with the output, never deleted; a file in
        place, just before it is unlinked. Only the first check sees a sidecar that names a path
        holding nothing. With ``state_dir`` the removal holds the path's restore lock and first
        removes, once checked the same way, the backup a killed stage write of the path left.
        """
        self.refuse_output_replacements(
            [ref], claims, None if returned is None else {ref: returned}
        )
        path = self._resolve_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = self._write_target(ref, claims, returned)

        def refuse_moved(entry: pathlib.Path) -> None:
            refusal = ownership.swap_refusal(self._project_root, target, entry)
            if refusal is not None:
                raise refusal

        cache.remove_output_checked(path, refuse_moved, state_dir)
        if ref.tag is types.ArtifactTag.DIRECTORY:
            path.mkdir(exist_ok=True)
        return path

    def prepare_commit(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        *,
        state_dir: pathlib.Path | None = None,
        warn_left_out_symlinks: bool = True,
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
        claims: ownership.StageClaims | None = None,
        written: ownership.StageMembers | None = None,
    ) -> types.FileHash | listing.HashedDirectory:
        """Hash a written output and publish its cache objects, outside any naming-lock unit.

        ``warn_left_out_symlinks`` logs the warning naming the symlinks a directory output
        leaves out; a quiet command turns it off. ``hashed_dirs`` receives a hashed directory by
        its hash, so ``finish_commit`` and the lock writer have its listing. The stage's own
        ``*.pvt`` files in the output are ``ownership.stage_members`` of ``claims`` and of the
        ones this run wrote there (``written``); every other ``*.pvt`` there is a user's, and the
        commit refuses here, before anything is published, since linking the output to the cache
        would replace it. ``finish_commit`` must follow inside the caller's unit before a record
        names the returned hash.
        """
        hashed: types.FileHash | listing.HashedDirectory
        if path.is_dir():
            hashed = cache.hash_directory(
                path,
                None,
                cache_dir=self._cache_dir,
                warn_left_out_symlinks=warn_left_out_symlinks,
            )
        else:
            file_hash, _ = cache.hash_file(path, None)
            hashed = types.FileHash(hash=file_hash)
        if self._cache_dir is not None and ref.tag is not types.ArtifactTag.METRIC:
            # Linking to the cache replaces each workspace file (symlink mode: the directory)
            self.refuse_output_replacements(
                [ref], claims, None if written is None else {ref: written}
            )
            cache.publish_hashed_output(
                path,
                hashed,
                self._cache_dir,
                self._checkout_modes,
                state_dir=state_dir,
            )
        if "listing" in hashed:
            _ = _record(hashed, hashed_dirs)
        return hashed

    def finish_commit(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        hashed: types.FileHash | listing.HashedDirectory,
        *,
        state_dir: pathlib.Path | None = None,
        claims: ownership.StageClaims | None = None,
        written: ownership.StageMembers | None = None,
    ) -> types.HashInfo:
        """Inside the caller's naming-lock unit: re-check, publish any missing object, link, and
        return the record a lock keeps for ``hashed``.

        The objects ``prepare_commit`` published may have been removed by a collection run
        before the unit was taken. ``store_hashed_output`` finds each object that remains with a
        cheap existence check and publishes any that is gone again from ``path`` before it links
        the workspace output, so the record that follows names only objects in the cache. A
        directory that symlink mode replaces by a link to its cache tree is renamed aside and
        checked again there (``claims`` and ``written`` as ``prepare_commit`` says), as
        ``prepare_output`` does, and goes back when that check refuses.
        """
        if self._cache_dir is not None and ref.tag is not types.ArtifactTag.METRIC:
            target = self._write_target(ref, claims, written)

            def refuse_moved(entry: pathlib.Path) -> None:
                refusal = ownership.swap_refusal(self._project_root, target, entry)
                if refusal is not None:
                    raise refusal

            cache.store_hashed_output(
                path,
                hashed,
                self._cache_dir,
                self._checkout_modes,
                state_dir=state_dir,
                verify=True,
                before_remove=refuse_moved,
            )
        return hashed["record"] if "listing" in hashed else hashed

    def commit(
        self,
        ref: types.ArtifactRef,
        path: pathlib.Path,
        *,
        state_dir: pathlib.Path | None = None,
        warn_left_out_symlinks: bool = True,
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
        claims: ownership.StageClaims | None = None,
        written: ownership.StageMembers | None = None,
    ) -> types.HashInfo:
        """Hash a written output once and, unless it is a metric, place it in the cache.

        For a caller that holds no naming-lock unit. A pivot commit site calls
        ``prepare_commit`` before its unit and ``finish_commit`` inside it. ``hashed_dirs``,
        ``claims`` and ``written`` are as ``prepare_commit`` says.
        """
        hashed = self.prepare_commit(
            ref,
            path,
            state_dir=state_dir,
            warn_left_out_symlinks=warn_left_out_symlinks,
            hashed_dirs=hashed_dirs,
            claims=claims,
            written=written,
        )
        return self.finish_commit(
            ref, path, hashed, state_dir=state_dir, claims=claims, written=written
        )

    def cached_listing(self, output_hash: types.DirHash) -> listing.DirListing | None:
        """The listing this store's cache holds for a directory, or None."""
        if self._cache_dir is None:
            return None
        return cache.cached_listing(self._cache_dir, output_hash)

    def restore_from_cache(
        self,
        ref: types.ArtifactRef,
        output_hash: types.HashInfo,
        *,
        state_dir: pathlib.Path | None = None,
        checkout_modes: list[config.CheckoutMode] | None = None,
        members: listing.DirListing | None = None,
        claims: ownership.StageClaims | None = None,
        recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    ) -> bool:
        """Restore ``ref`` from the cache, refusing a user file at the initial and swap checks.

        The stage's own ``*.pvt`` files are ``ownership.stage_members`` of ``claims`` and the
        restored hash, whose files a directory's listing names: ``members``, else the cache's
        listing, read once here for the checks and the restore. ``ownership.record_restored``
        records the ones ``output_hash`` writes that the stage's lock does not list before the
        restore writes them. ``recorded`` is as ``cache.restore_from_cache`` says.
        """
        if self._cache_dir is None:
            return False
        if ref.tag is types.ArtifactTag.METRIC:
            return False
        if members is None and types.is_dir_hash(output_hash):
            members = self.cached_listing(output_hash)
            if members is None:
                return False
        output_path = self._resolve_output_path(ref)
        replacement = ownership.stage_output(
            ownership.normalized(output_path), ref.identity.producer
        )
        target = ownership.ForcedRestore(
            replacement=replacement,
            restored=output_hash,
            claims=None if claims is None else lambda: claims,
        )
        if members is not None:
            target["restored_members"] = members
        ownership.refuse_output_replacements(
            self._project_root,
            [replacement],
            lambda path: ownership.stage_members(
                path, claims, restored=output_hash, restored_members=members
            ),
        )
        if claims is not None:
            ownership.record_restored(lambda: claims, output_path, output_hash, members)

        def before_swap(entry: pathlib.Path) -> None:
            refusal = ownership.swap_refusal(self._project_root, target, entry)
            if refusal is not None:
                raise refusal

        return cache.restore_from_cache(
            output_path,
            output_hash,
            self._cache_dir,
            checkout_modes=checkout_modes if checkout_modes is not None else self._checkout_modes,
            state_dir=state_dir,
            members=members,
            before_swap=before_swap,
            recorded=recorded,
        )

    def hash_artifact(
        self,
        ref: types.ArtifactRef,
        state_db: state_mod.StateDB | None = None,
        *,
        pending: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
        hashed_dirs: dict[str, listing.HashedDirectory] | None = None,
    ) -> types.HashInfo:
        """Hash dependency ``ref`` as it is in the workspace.

        A directory input leaves out the files ``.pivotignore`` names (``_input_member_filter``).
        """
        path = self._resolve_path(ref)
        if ref.directory is None and (ref.tag == types.ArtifactTag.DIRECTORY or path.is_dir()):
            hashed = cache.hash_directory(
                path,
                state_db,
                cache_dir=self._cache_dir,
                pending=pending,
                ignored=self._input_member_filter(ref),
            )
            return _record(hashed, hashed_dirs)
        file_hash, _ = cache.hash_file(path, state_db, pending=pending)
        return types.FileHash(hash=file_hash)

    def recorded_input_hash(
        self,
        ref: types.ArtifactRef,
        recorded: types.DirHash,
        read_members: Callable[[], listing.DirListing],
    ) -> types.DirHash:
        """The dependency hash ``hash_artifact`` gives directory input ``ref`` when it holds the
        directory ``recorded``.

        ``recorded`` keeps every file, as a ``.pvt`` records a tracked directory; when
        ``.pivotignore`` names files of the input, the files of its listing (``read_members``,
        read only then) are taken out of it (``cache.filter_listing``).
        """
        ignored = self._input_member_filter(ref)
        if ignored is None:
            return recorded
        return cache.filter_listing(read_members(), ignored)["record"]

    def input_relpath(self, ref: types.ArtifactRef) -> str:
        """A pipeline input's path relative to the project root, POSIX form: the name of its
        listing file when it is a directory."""
        relpath = os.path.relpath(self._resolve_input_path(ref), self._project_root)
        return pathlib.PurePath(relpath).as_posix()

    def cache_contains(
        self,
        ref: types.ArtifactRef,
        output_hash: types.HashInfo,
        members: listing.DirListing | None = None,
    ) -> bool:
        """Return whether this output can be restored from the local cache.

        ``members`` is a directory's listing; None reads the cache's listing object.
        """
        if self._cache_dir is None or ref.tag is types.ArtifactTag.METRIC:
            return False
        if types.is_dir_hash(output_hash):
            if members is None:
                members = self.cached_listing(output_hash)
            return members is not None and all(
                cache.resolve_cache_object_path(self._cache_dir, entry["hash"]) is not None
                for entry in members["files"]
            )
        return cache.resolve_cache_object_path(self._cache_dir, output_hash["hash"]) is not None

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


def store_from_spec(spec: StoreSpec, *, ignore_filter: ignore.IgnoreFilter | None = None) -> Store:
    """Build the store ``spec`` describes; ``ignore_filter`` is the command's ``.pivotignore``
    filter for a workspace store, as ``WorkspaceStore`` takes it."""
    kind = spec["kind"]
    if kind == "cache":
        project_root = pathlib.Path(spec["project_root"])
        return CacheStore(
            cache_dir=pathlib.Path(spec["cache_dir"]),
            state_dir=state_dir_mod.ensure_state_dir(project_root),
        )
    if kind == "workspace":
        cache_dir_str = spec.get("cache_dir")
        return WorkspaceStore(
            project_root=pathlib.Path(spec["project_root"]),
            pipeline_name=spec["pipeline_name"],
            input_bindings=spec["input_bindings"],
            cache_dir=pathlib.Path(cache_dir_str) if cache_dir_str else None,
            checkout_modes=spec["checkout_modes"],
            ignore_filter=ignore_filter,
        )
    raise ValueError(f"Unknown store kind: {kind}")
