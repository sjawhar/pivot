from __future__ import annotations

import asyncio
import enum
import functools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, NotRequired, TypedDict

import xxhash
import yaml

from pivot import config, exceptions, git, project, types, yaml_config
from pivot import state_dir as state_dir_mod
from pivot.storage import cache, listing, lock, ownership, state, track
from pivot.types import HashInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pivot.remote.protocol import RemoteStorage
    from pivot.storage.track import PvtData
    from pivot.types import DirHash, TransferResult

logger = logging.getLogger(__name__)


class TargetType(enum.StrEnum):
    """Type of target being retrieved."""

    FILE = "file"
    STAGE = "stage"


class RestoreStatus(enum.StrEnum):
    """Status of a file restoration attempt."""

    RESTORED = "restored"
    SKIPPED = "skipped"
    ERROR = "error"


class RestoreResult(TypedDict):
    """Result of restoring a single file."""

    status: RestoreStatus
    path: str
    message: str
    unreachable_remote: NotRequired[str]
    """The URL of the remote this restore could not reach, so the rest of the get skips it."""


class TargetInfo(TypedDict):
    """Information about a target to retrieve."""

    target_type: TargetType
    original_target: str
    paths: list[str]
    hashes: dict[str, HashInfo | None]
    members: dict[str, listing.DirListing]  # each directory's listing at the revision, by path
    error: NotRequired[str]
    """Why the target cannot be restored (its .pvt is not valid), reported for each path."""


class StageTarget(TypedDict):
    """A CLI target naming a stage's outputs, resolved against the current pipeline."""

    stage_name: str
    state_dir: pathlib.Path
    paths_by_key: dict[str | None, str]
    """Fallback lock output key -> workspace path for older locks with no recorded path."""


def _parse_lock_data_from_bytes(content: bytes) -> lock.CommittedLock | None:
    """Parse committed lock file content from bytes, an older pivot's format included."""
    try:
        data: object = yaml.load(content, Loader=yaml_config.Loader)
    except yaml.YAMLError:
        return None
    parsed = lock.parse_committed(data)
    return parsed if isinstance(parsed, lock.CommittedLock) else None


def _parse_pvt_data_from_bytes(
    content: bytes,
) -> tuple[PvtData, listing.DirListing | None] | str | None:
    """Parse .pvt file content from bytes: None if it is not a valid .pvt file, else its data
    with an older pivot's inline directory listing, or why a restore cannot use it
    (``track.pvt_field_error``)."""
    try:
        data: object = yaml.load(content, Loader=yaml_config.Loader)
    except yaml.YAMLError:
        return None
    return track.parse_pvt(data)


def get_lock_data_from_revision(
    stage_name: str, rev: str, state_dir: pathlib.Path
) -> lock.CommittedLock | None:
    """Read a lock at a revision, falling back to the legacy state directory when needed."""
    root = project.get_project_root()
    state_dirs = [state_dir]
    if (other_state_dir := state_dir_mod.other_state_dir_for(state_dir)) is not None:
        state_dirs.append(other_state_dir)
    for candidate in state_dirs:
        try:
            rel_path = str(lock.get_stages_dir(candidate).relative_to(root) / f"{stage_name}.lock")
        except ValueError:
            continue
        content = git.read_file_from_revision(rel_path, rev)
        if content is not None:
            if candidate != state_dir:
                state_dir_mod.report_legacy_state_dir(state_dir)
            return _parse_lock_data_from_bytes(content)
    return None


def get_pvt_data_from_revision(
    pvt_rel_path: str, rev: str
) -> tuple[PvtData, listing.DirListing | None] | str | None:
    """Read and parse .pvt file from a git revision: None when the revision has no valid .pvt
    file there, else its data with an older pivot's inline listing, or why a restore cannot use
    it."""
    content = git.read_file_from_revision(pvt_rel_path, rev)
    if content is None:
        return None
    return _parse_pvt_data_from_bytes(content)


def _listing_from_revision(
    rel_path: str, rev: str, record: DirHash, cache_dir: pathlib.Path | None
) -> listing.DirListing | None:
    """A directory's listing by its committed file at a git revision, else the cache's object
    in ``cache_dir`` (``listing.read_found``); None when neither holds it."""
    content = git.read_file_from_revision(rel_path, rev)
    try:
        return listing.read_found(content, f"{rel_path} at revision {rev}", record, cache_dir)
    except exceptions.ListingMissingError:
        return None


def _normalize_target_path(target: str, proj_root: pathlib.Path) -> str:
    """Normalize target to relative path, validating it's within project."""
    # Security check: reject path traversal in user input
    if track.has_path_traversal(target):
        raise exceptions.TargetNotFoundError(f"Path traversal not allowed in target: {target!r}")

    target_path = pathlib.Path(target)
    if not target_path.is_absolute():
        target_path = proj_root / target_path

    try:
        return str(target_path.relative_to(proj_root))
    except ValueError:
        raise exceptions.TargetNotFoundError(
            f"Target path is outside project root: {target!r}"
        ) from None


def resolve_targets(
    targets: Sequence[str],
    rev: str,
    stage_targets: Mapping[str, StageTarget],
    cache_dir: pathlib.Path | None = None,
) -> list[TargetInfo]:
    """Resolve targets to TargetInfo: a target in ``stage_targets`` is a stage, others are files.

    A directory's listing is read from the revision, else from the cache object in
    ``cache_dir`` (``<cache.dir>/files``)."""
    proj_root = project.get_project_root()
    results = list[TargetInfo]()

    for target in targets:
        if target in stage_targets:
            stage_target = stage_targets[target]
            stage_name = stage_target["stage_name"]
            committed = get_lock_data_from_revision(stage_name, rev, stage_target["state_dir"])
            if committed is None:
                raise exceptions.TargetNotFoundError(
                    f"Stage '{stage_name}' has no lock file at revision '{rev}'"
                )
            paths_by_key = stage_target["paths_by_key"]
            selected_key = types.identity_from_key(target).key
            hashes = dict[str, HashInfo | None]()
            members = dict[str, listing.DirListing]()
            for entry in committed.data["outs"]:
                key = entry["key"]
                if selected_key is not None and key != selected_key:
                    continue
                if "path" in entry:
                    path = entry["path"]
                elif key in paths_by_key:
                    path = paths_by_key[key]
                else:
                    raise exceptions.TargetNotFoundError(
                        f"Target '{target}' output key {key!r} at revision '{rev}' has no "
                        + "recorded path and is not defined by the current pipeline"
                    )
                hashes[path] = hash_info = lock.entry_hash_info(entry)
                if is_dir_hash(hash_info):
                    if key in committed.listings:
                        members[path] = committed.listings[key]
                        continue
                    listing_state_dir = stage_target["state_dir"]
                    try:
                        stages_rel = lock.get_stages_dir(listing_state_dir).relative_to(proj_root)
                    except ValueError:
                        legacy_state_dir = state_dir_mod.other_state_dir_for(listing_state_dir)
                        if legacy_state_dir is None:
                            raise exceptions.TargetNotFoundError(
                                f"Stage '{stage_name}' has no git-tracked state directory"
                            ) from None
                        stages_rel = lock.get_stages_dir(legacy_state_dir).relative_to(proj_root)
                    listing_rel = listing.output_listing_path(
                        stages_rel, types.ArtifactIdentity(stage_name, key)
                    ).as_posix()
                    found = _listing_from_revision(listing_rel, rev, hash_info, cache_dir)
                    if found is not None:
                        members[path] = found
            if not hashes:
                raise exceptions.TargetNotFoundError(
                    f"Target '{target}' recorded no outputs at revision '{rev}'"
                )
            results.append(
                TargetInfo(
                    target_type=TargetType.STAGE,
                    original_target=target,
                    paths=list(hashes),
                    hashes=hashes,
                    members=members,
                )
            )
            continue

        # Normalize path (validates traversal and project bounds)
        rel_target = _normalize_target_path(target, proj_root)
        pvt_rel_path = rel_target + ".pvt"

        # Try as a .pvt tracked file. One that is not valid is this target's error, so the other
        # targets are still restored and reported.
        parsed_pvt = get_pvt_data_from_revision(pvt_rel_path, rev)
        if isinstance(parsed_pvt, str):
            results.append(
                TargetInfo(
                    target_type=TargetType.FILE,
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: None},
                    members={},
                    error=f"{pvt_rel_path} at {rev} is not a valid .pvt file: {parsed_pvt}",
                )
            )
            continue
        if parsed_pvt is not None:
            pvt_data, inline_members = parsed_pvt
            pvt_hash = track.pvt_to_hash_info(pvt_data)
            pvt_members = dict[str, listing.DirListing]()
            if is_dir_hash(pvt_hash):
                found = inline_members or _listing_from_revision(
                    listing.tracked_listing_path(pathlib.Path(pvt_rel_path)).as_posix(),
                    rev,
                    pvt_hash,
                    cache_dir,
                )
                if found is not None:
                    pvt_members[rel_target] = found
            results.append(
                TargetInfo(
                    target_type=TargetType.FILE,
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: pvt_hash},
                    members=pvt_members,
                )
            )
            continue

        # Try as a git-tracked file
        content = git.read_file_from_revision(rel_target, rev)
        if content is not None:
            results.append(
                TargetInfo(
                    target_type=TargetType.FILE,
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: None},  # None means must use git
                    members={},
                )
            )
            continue

        raise exceptions.TargetNotFoundError(
            f"Target '{target}' not found at revision '{rev}' (not a stage name, tracked file, or git-tracked file)"
        )

    return results


def _write_fallback_content(
    dest_path: pathlib.Path,
    content: bytes,
    lock_token: str | None,
    before_swap: Callable[[pathlib.Path], None],
) -> os.stat_result:
    """Atomically replace a cache-miss target; return the staged file's stat before its swap."""

    def write_content(fd: int) -> None:
        with os.fdopen(fd, "wb") as file:
            file.write(content)

    staging_path = (
        cache.restore_staging_path(dest_path, lock_token) if lock_token is not None else None
    )
    return cache.atomic_write_file(
        dest_path, write_content, staging_path=staging_path, before_swap=before_swap
    )


def _refuse_swap(target: ownership.ForcedRestore, entry: pathlib.Path) -> None:
    """Raise if the entry the swap of ``target`` removes holds a user's file."""
    refusal = ownership.swap_refusal(project.get_project_root(), target, entry)
    if refusal is not None:
        raise refusal


def _replace_checked[T](
    dest_path: pathlib.Path,
    target: ownership.ForcedRestore,
    state_dir: pathlib.Path,
    replace: Callable[[str | None, Callable[[pathlib.Path], None]], T],
) -> T:
    """Run ``replace`` under the target state directory's restore lock and ownership checks."""
    with cache.restore_lock(dest_path, state_dir) as lock_token:
        refusal = ownership.forced_restore_refusal(project.get_project_root(), [target])
        if refusal is not None:
            raise refusal
        return replace(lock_token, functools.partial(_refuse_swap, target))


def _reason(error: exceptions.PivotError | OSError) -> str:
    """``error``'s message, followed on the next line by the tip a pivot error carries."""
    if isinstance(error, exceptions.PivotError) and (tip := error.get_suggestion()):
        return f"{error}\n  Tip: {tip}"
    return str(error)


def _download_error(failed: list[TransferResult]) -> exceptions.RemoteError:
    """The error of a download whose ``failed`` files did not arrive: a file's failure to reach
    the remote if there is one, since it ends the remote step for the rest of the get, else the
    first file's error."""
    unreachable = [result for result in failed if "connection_failed" in result]
    result = (unreachable or failed)[0]
    message = result["error"] if "error" in result else "unknown error"
    if unreachable:
        return exceptions.RemoteUnreachableError(message)
    return exceptions.RemoteError(message)


def _download(remote: RemoteStorage, files_dir: pathlib.Path, hashes: list[str]) -> None:
    """Download the cache objects ``hashes`` names into ``files_dir``, raising the error of a
    download whose files did not arrive (``_download_error``)."""
    results = asyncio.run(
        remote.download_batch(
            [(file_hash, cache.get_cache_path(files_dir, file_hash)) for file_hash in hashes],
            readonly=True,
        )
    )
    if failed := [result for result in results if not result["success"]]:
        raise _download_error(failed)


def _skipped(dest_path: pathlib.Path, force: bool) -> RestoreResult | None:
    """The result for a target already in the workspace that ``get`` keeps, or None."""
    if dest_path.exists() and not force:
        return RestoreResult(
            status=RestoreStatus.SKIPPED,
            path=str(dest_path),
            message=f"Skipped: {dest_path} (already exists, use --force to overwrite)",
        )
    return None


def restore_file(
    rel_path: str,
    output_hash: HashInfo | None,
    rev: str,
    dest_path: pathlib.Path,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
    target: ownership.ForcedRestore,
    *,
    remote_factory: Callable[[], RemoteStorage],
    state_dir: pathlib.Path | None = None,
    recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None = None,
    members: listing.DirListing | None = None,
) -> RestoreResult:
    """Restore a single file from revision and return its result.

    Tries the local cache, then git, then downloads the cache objects the output still misses
    from the remote ``remote_factory`` opens; it is called only when that download is needed.
    A remote that raised ``RemoteUnreachableError`` is named in the result's
    ``unreachable_remote``.
    ``target`` is what the restore replaces (``_restore_targets``). Each write checks it again
    under the restore lock, just before it builds the replacement, and once more at the swap; a
    refusal is this target's error. For a stage output restored to its own path, the ``*.pvt``
    members ``output_hash`` writes are recorded as the stage's before anything is written.

    ``members`` is a directory's listing at the revision; None reads the cache's object, once,
    here, for the ``*.pvt`` record and the restore. A directory with neither is not restored: no
    listing names the files to restore or download.

    A restore appends ``(path, stat, hash)`` for each file it writes to ``recorded``, for the
    caller to batch by state directory, so the next status, explanation or run looks the file up
    instead of reading it. Each stat is of what was staged, taken before the rename put it in
    place. ``state_dir`` is the stage's own state directory for a stage target, otherwise the
    configured directory from ``config.locate_state_dir()``.
    """
    if (skipped := _skipped(dest_path, force)) is not None:
        return skipped

    if members is None and output_hash is not None and is_dir_hash(output_hash):
        try:
            members = cache.cached_listing(cache_dir / "files", output_hash)
        except (exceptions.PivotError, OSError) as error:
            # With no usable listing the cache strategy is skipped; the result names the cause
            logger.debug("Cannot read the cached listing for %s: %s", rel_path, error)
    if members is not None and "restored_members" not in target:
        # The ownership checks read the restored directory's files from the same listing
        target = target.copy()
        target["restored_members"] = members
    if target["claims"] is not None and output_hash is not None:
        ownership.record_restored(target["claims"], dest_path, output_hash, members)

    state_dir = state_dir or config.locate_state_dir()
    try:
        return _restore_from_sources(
            rel_path,
            output_hash,
            rev,
            dest_path,
            cache_dir,
            checkout_modes,
            target,
            remote_factory,
            state_dir,
            recorded,
            members,
        )
    except exceptions.OutputOwnershipError as error:
        return RestoreResult(
            status=RestoreStatus.ERROR, path=str(dest_path), message=f"Error: {dest_path} - {error}"
        )


def _restore_from_sources(
    rel_path: str,
    output_hash: HashInfo | None,
    rev: str,
    dest_path: pathlib.Path,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    target: ownership.ForcedRestore,
    remote_factory: Callable[[], RemoteStorage],
    state_dir: pathlib.Path,
    recorded: list[tuple[pathlib.Path, os.stat_result, str]] | None,
    members: listing.DirListing | None,
) -> RestoreResult:
    """``restore_file``'s strategies, each write under ``_replace_checked``."""
    path_str = str(dest_path)
    files_dir = cache_dir / "files"

    # Ensure parent directory exists before any restore strategy
    try:
        state_dir_mod.make_dirs(dest_path.parent)
    except OSError as e:
        return RestoreResult(
            status=RestoreStatus.ERROR,
            path=path_str,
            message=f"Error: {dest_path} - cannot create its directory: {e}",
        )

    # Strategy 1: Try local cache (if hash available). A directory needs its listing: the one
    # committed at the revision, else the cache's object (``restore_file`` read it).
    if output_hash is not None:
        cached_hash = output_hash
        restored = list[tuple[pathlib.Path, os.stat_result, str]]()
        success = False
        try:
            if is_dir_hash(output_hash):
                should_try_cache = members is not None
            else:
                should_try_cache = (
                    cache.resolve_cache_object_path(files_dir, output_hash["hash"]) is not None
                )
            if should_try_cache:
                success = _replace_checked(
                    dest_path,
                    target,
                    state_dir,
                    lambda token, before_swap: cache.restore_from_cache(
                        dest_path,
                        cached_hash,
                        files_dir,
                        checkout_modes=checkout_modes,
                        state_dir=state_dir,
                        lock_token=token,
                        members=members,
                        before_swap=before_swap,
                        recorded=restored,
                    ),
                )
        except exceptions.OutputOwnershipError:
            raise
        except (exceptions.PivotError, OSError) as error:
            # Strategy 3 retries this restore, and reports the error if it happens again.
            logger.debug("Cache restore failed for %s: %s", rel_path, error)
        if success:
            if recorded is not None:
                recorded.extend(restored)
            return RestoreResult(
                status=RestoreStatus.RESTORED,
                path=path_str,
                message=f"Restored: {dest_path} (from cache)",
            )
    # Strategy 2: Try git fallback
    content = git.read_file_from_revision(rel_path, rev)
    if content is not None:
        try:
            written = _replace_checked(
                dest_path,
                target,
                state_dir,
                lambda token, before_swap: _write_fallback_content(
                    dest_path, content, token, before_swap
                ),
            )
        except exceptions.OutputOwnershipError:
            raise
        except OSError as error:
            return RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to write from git: {error}",
            )
        if recorded is not None:
            recorded.append((dest_path, written, xxhash.xxh64(content).hexdigest()))
        return RestoreResult(
            status=RestoreStatus.RESTORED,
            path=path_str,
            message=f"Restored: {dest_path} (from git)",
        )

    # Strategy 3: Download the cache objects the output still misses from the remote, then
    # restore from the cache. Any failure is this target's error, so the targets already
    # restored are still reported.
    if output_hash is not None:
        if is_dir_hash(output_hash) and members is None:
            # A directory is restored from its listing, which names its files; with none
            # committed or cached, its members are unknown and nothing can be downloaded.
            return RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - not in local cache or git, and no committed or "
                + "cached listing names its files to download from the remote",
            )
        cached_hash = output_hash
        source = "cache"
        remote: RemoteStorage | None = None
        downloaded = list[tuple[pathlib.Path, os.stat_result, str]]()
        try:
            missing_hashes = cache.missing_cache_objects(
                output_hash, members, files_dir, checkout_modes
            )
            if missing_hashes:
                source = "remote"
                remote = remote_factory()
                _download(remote, files_dir, missing_hashes)
            success = _replace_checked(
                dest_path,
                target,
                state_dir,
                lambda token, before_swap: cache.restore_from_cache(
                    dest_path,
                    cached_hash,
                    files_dir,
                    checkout_modes=checkout_modes,
                    state_dir=state_dir,
                    lock_token=token,
                    members=members,
                    before_swap=before_swap,
                    recorded=downloaded,
                ),
            )
        except exceptions.OutputOwnershipError:
            raise
        except exceptions.RemoteError as e:
            failure = RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to download from remote: {_reason(e)}",
            )
            if remote is not None and isinstance(e, exceptions.RemoteUnreachableError):
                failure["unreachable_remote"] = remote.url
            return failure
        except (exceptions.PivotError, OSError) as e:
            return RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to restore from {source}: {_reason(e)}",
            )
        if success:
            if recorded is not None:
                recorded.extend(downloaded)
            return RestoreResult(
                status=RestoreStatus.RESTORED,
                path=path_str,
                message=f"Restored: {dest_path} (from {source})",
            )
        return RestoreResult(
            status=RestoreStatus.ERROR,
            path=path_str,
            message=f"Error: {dest_path} - failed to restore from {source}",
        )

    return RestoreResult(
        status=RestoreStatus.ERROR,
        path=path_str,
        message=f"Error: {dest_path} - not in git at {rev}, and no hash is recorded for it",
    )


def _read_claims(
    proj_root: pathlib.Path, stage_target: StageTarget, files_dir: pathlib.Path
) -> ownership.StageClaims:
    """The stage's claims (``ownership.StageClaims``): the hashes its current lock records, each
    directory's listing committed beside that lock, its ``nocommit:`` record and the cache object
    root ``files_dir``, with a recorder for the ``*.pvt`` members ``get`` writes."""
    stage_lock = lock.StageLock(
        stage_target["stage_name"], lock.get_stages_dir(stage_target["state_dir"])
    )
    lock_data = stage_lock.read(cache_dir=files_dir)
    paths = dict[types.ArtifactIdentity, pathlib.Path]()
    if lock_data is not None:
        recorded_paths = (
            lock_data["output_paths"]
            if "output_paths" in lock_data
            else dict[types.ArtifactIdentity, str]()
        )
        for identity in lock_data["output_hashes"]:
            path = recorded_paths.get(identity) or stage_target["paths_by_key"].get(identity.key)
            if path is not None:
                paths[identity] = proj_root / path
    locked = (
        {path: lock_data["output_hashes"][identity] for identity, path in paths.items()}
        if lock_data is not None
        else {}
    )
    return ownership.read_stage_claims(
        proj_root,
        stage_target["stage_name"],
        stage_target["state_dir"],
        locked,
        cache_dir=files_dir,
        recorder=functools.partial(
            ownership.record_members,
            proj_root,
            stage_target["state_dir"],
            stage_target["stage_name"],
        ),
        listing_readers=ownership.lock_listing_readers(stage_lock, lock_data, paths, files_dir),
    )


def _restore_targets(
    proj_root: pathlib.Path,
    target_infos: list[TargetInfo],
    output: pathlib.Path | None,
    stage_targets: Mapping[str, StageTarget],
    files_dir: pathlib.Path,
) -> dict[tuple[int, str], ownership.ForcedRestore]:
    """What restoring each path of each target replaces, keyed by (target index, path), for
    ``ownership.forced_restore_refusal``. A target whose ``.pvt`` is not valid is only reported,
    never restored, so it is left out.

    A tracked or git file restored to its own path is its own restore: it may replace its own
    content, never another tracked file, import or sidecar beneath it. A stage output restored to
    its own path is that stage's output, checked with the stage's claims and the revision's hash.
    An ``--output`` path is a destination the user named.
    """
    restores = dict[tuple[int, str], ownership.ForcedRestore]()
    for index, target_info in enumerate(target_infos):
        if "error" in target_info:
            continue
        stage_target = (
            stage_targets[target_info["original_target"]]
            if target_info["target_type"] is TargetType.STAGE
            else None
        )
        claims = (
            functools.cache(functools.partial(_read_claims, proj_root, stage_target, files_dir))
            if stage_target is not None
            else None
        )
        for rel_path in target_info["paths"]:
            own_path = proj_root / rel_path
            dest_path = output if output is not None else own_path
            restored = target_info["hashes"].get(rel_path)
            if dest_path != own_path:
                restore = ownership.ForcedRestore(
                    replacement=ownership.destination(dest_path), restored=restored, claims=None
                )
            elif stage_target is not None:
                restore = ownership.ForcedRestore(
                    replacement=ownership.stage_output(dest_path, stage_target["stage_name"]),
                    restored=restored,
                    claims=claims,
                )
            else:
                restore = ownership.ForcedRestore(
                    replacement=ownership.tracked_restore(dest_path), restored=restored, claims=None
                )
            # The listing at the revision names a restored directory's files
            members = target_info["members"].get(rel_path)
            if members is not None:
                restore["restored_members"] = members
            restores[(index, rel_path)] = restore
    return restores


def restore_targets_from_revision(
    targets: Sequence[str],
    rev: str,
    output: pathlib.Path | None,
    cache_dir: pathlib.Path,
    stage_targets: Mapping[str, StageTarget],
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
    *,
    remote_factory: Callable[[], RemoteStorage],
) -> tuple[list[str], bool]:
    """Restore targets from a git revision.

    A relative ``output`` is relative to the working directory.

    Returns:
        Tuple of (messages, success) where success is False if any files failed.
    """
    proj_root = project.get_project_root()
    output_path = output if output is None or output.is_absolute() else pathlib.Path.cwd() / output

    # Validate revision exists
    commit_sha = git.resolve_revision(rev)
    if commit_sha is None:
        raise exceptions.RevisionNotFoundError(f"Cannot resolve revision: '{rev}'")

    # Resolve targets
    target_infos = resolve_targets(targets, rev, stage_targets, cache_dir / "files")

    # Validate -o usage
    if output is not None:
        if len(target_infos) != 1:
            raise exceptions.GetError("--output/-o can only be used with a single target")
        if len(target_infos[0]["paths"]) != 1:
            raise exceptions.GetError("--output/-o can only be used with a single output")

    restores = _restore_targets(
        proj_root, target_infos, output_path, stage_targets, cache_dir / "files"
    )
    # Only --force replaces an existing path; check every one before any is replaced. Each is
    # checked again just before it is replaced (`restore_file`).
    if force:
        refusal = ownership.forced_restore_refusal(proj_root, restores.values())
        if refusal is not None:
            raise refusal

    results = list[RestoreResult]()
    # Resolves the remote and builds its backend once per call, not once per target that misses
    # the cache.
    open_remote = functools.cache(remote_factory)
    # A remote this call could not reach is not tried again, so an unreachable remote costs one
    # connection timeout per call, not one per target. Only RemoteUnreachableError sets it: a
    # failure of one object leaves the remote in use for the other targets.
    unreachable: str | None = None

    def reachable_remote() -> RemoteStorage:
        if unreachable is not None:
            raise exceptions.RemoteUnreachableError(
                f"{unreachable} could not be reached for an earlier target, "
                + "so it is not tried again"
            )
        return open_remote()

    # Every target's file hashes are saved once per state directory when the loop ends,
    # best-effort. A stage output uses its stage state directory; other files use the configured
    # project state directory.
    batch = state.FileHashBatch()
    try:
        for index, target_info in enumerate(target_infos):
            hash_state_dir = (
                stage_targets[target_info["original_target"]]["state_dir"]
                if target_info["target_type"] is TargetType.STAGE
                else config.locate_state_dir()
            )
            for rel_path in target_info["paths"]:
                output_hash = target_info["hashes"].get(rel_path)
                dest_path = output_path if output_path is not None else proj_root / rel_path
                if "error" in target_info:
                    # A target already in the workspace is kept, as for a valid .pvt, so the
                    # check refuses nothing that get skipped before it.
                    results.append(
                        _skipped(dest_path, force)
                        or RestoreResult(
                            status=RestoreStatus.ERROR,
                            path=str(dest_path),
                            message=f"Error: {dest_path} - {target_info['error']}",
                        )
                    )
                    continue

                recorded = list[tuple[pathlib.Path, os.stat_result, str]]()
                result = restore_file(
                    rel_path=rel_path,
                    output_hash=output_hash,
                    rev=rev,
                    dest_path=dest_path,
                    cache_dir=cache_dir,
                    checkout_modes=checkout_modes,
                    force=force,
                    target=restores[(index, rel_path)],
                    remote_factory=reachable_remote,
                    state_dir=hash_state_dir,
                    recorded=recorded,
                    members=target_info["members"].get(rel_path),
                )
                batch.add(hash_state_dir, recorded)
                if "unreachable_remote" in result:
                    unreachable = result["unreachable_remote"]
                results.append(result)
    finally:
        batch.save()

    # Format output: show all messages, with errors summarized at the end
    messages = [r["message"] for r in results]
    errors = [r for r in results if r["status"] == RestoreStatus.ERROR]

    if errors:
        messages.append("")
        messages.append(f"Failed to restore {len(errors)} file(s). See errors above.")

    return (messages, len(errors) == 0)
