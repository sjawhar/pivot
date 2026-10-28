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
from pivot.storage import cache, listing, lock, state, track
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
    dest_path: pathlib.Path, content: bytes, state_dir: pathlib.Path
) -> os.stat_result:
    """Atomically replace a cache-miss target under the resolved state directory's restore lock;
    return the written file's stat, taken before it replaced the target."""

    def write_content(fd: int) -> None:
        with os.fdopen(fd, "wb") as file:
            file.write(content)

    with cache.restore_lock(dest_path, state_dir) as lock_token:
        staging_path = (
            cache.restore_staging_path(dest_path, lock_token) if lock_token is not None else None
        )
        return cache.atomic_write_file(dest_path, write_content, staging_path=staging_path)


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

    ``members`` is a directory's listing at the revision; None reads the cache's object. A
    directory with neither is not restored: no listing names the files to restore or download.

    A restore appends ``(path, stat, hash)`` for each file it writes to ``recorded``, for the
    caller to batch by state directory, so the next status, explanation or run looks the file up
    instead of reading it. Each stat is of what was staged, taken before the rename put it in
    place. ``state_dir`` is the stage's own state directory for a stage target, otherwise the
    configured directory from ``config.locate_state_dir()``.
    """
    state_dir = state_dir or config.locate_state_dir()
    path_str = str(dest_path)
    files_dir = cache_dir / "files"
    if (skipped := _skipped(dest_path, force)) is not None:
        return skipped

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
    # committed at the revision, else the cache's object.
    if output_hash is not None:
        restored = list[tuple[pathlib.Path, os.stat_result, str]]()
        success = False
        try:
            if is_dir_hash(output_hash):
                if members is None:
                    members = cache.cached_listing(files_dir, output_hash)
                should_try_cache = members is not None
            else:
                should_try_cache = (
                    cache.resolve_cache_object_path(files_dir, output_hash["hash"]) is not None
                )
            if should_try_cache:
                success = cache.restore_from_cache(
                    dest_path,
                    output_hash,
                    files_dir,
                    checkout_modes=checkout_modes,
                    state_dir=state_dir,
                    members=members,
                    recorded=restored,
                )
        except (exceptions.PivotError, OSError) as e:
            # Strategy 3 retries this restore, and reports the error if it happens again
            logger.debug(f"Cache restore failed for {rel_path}: {e}")
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
            written = _write_fallback_content(dest_path, content, state_dir)
        except OSError as e:
            return RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to write from git: {e}",
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
            success = cache.restore_from_cache(
                dest_path,
                output_hash,
                files_dir,
                checkout_modes=checkout_modes,
                state_dir=state_dir,
                members=members,
                recorded=downloaded,
            )
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

    Returns:
        Tuple of (messages, success) where success is False if any files failed.
    """
    proj_root = project.get_project_root()

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
        for target_info in target_infos:
            hash_state_dir = (
                stage_targets[target_info["original_target"]]["state_dir"]
                if target_info["target_type"] is TargetType.STAGE
                else config.locate_state_dir()
            )
            for rel_path in target_info["paths"]:
                output_hash = target_info["hashes"].get(rel_path)
                dest_path = output if output is not None else proj_root / rel_path
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
