from __future__ import annotations

import asyncio
import enum
import functools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeGuard

import yaml

from pivot import config, exceptions, git, project, types, yaml_config
from pivot.storage import cache, lock, track
from pivot.types import HashInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pivot.remote.protocol import RemoteStorage
    from pivot.storage.track import PvtData
    from pivot.types import StorageLockData, TransferResult

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
    error: NotRequired[str]
    """Why the target cannot be restored (its .pvt is not valid), reported for each path."""


class StageTarget(TypedDict):
    """A CLI target naming a stage's outputs, resolved against the current pipeline."""

    stage_name: str
    state_dir: pathlib.Path
    paths_by_key: dict[str | None, str]
    """Fallback lock output key -> workspace path for older locks with no recorded path."""


def _parse_yaml_bytes[T](
    content: bytes,
    validator: Callable[[object], TypeGuard[T]],
) -> T | None:
    """Parse YAML bytes and validate. Returns None on any failure."""
    try:
        data: object = yaml.load(content, Loader=yaml_config.Loader)
        if validator(data):
            return data
        return None
    except yaml.YAMLError:
        return None


def _parse_lock_data_from_bytes(content: bytes) -> StorageLockData | None:
    """Parse lock file content from bytes."""
    return _parse_yaml_bytes(content, lock.is_lock_data)


def _parse_pvt_data_from_bytes(content: bytes) -> PvtData | str | None:
    """Parse .pvt file content from bytes: None if it is not a valid .pvt file, else its data, or
    why a restore cannot use it (``track.pvt_field_error``)."""
    data = _parse_yaml_bytes(content, track.is_pvt_data)
    if data is None:
        return None
    if (error := track.pvt_field_error(data)) is not None:
        return error
    # Security check: no path traversal in stored path
    if track.has_path_traversal(data["path"]):
        return None
    return data


def get_lock_data_from_revision(
    stage_name: str, rev: str, state_dir: pathlib.Path
) -> StorageLockData | None:
    """Read and parse lock file for a stage from a git revision."""
    stages_dir = lock.get_stages_dir(state_dir)
    rel_path = str(stages_dir.relative_to(project.get_project_root()) / f"{stage_name}.lock")
    content = git.read_file_from_revision(rel_path, rev)
    if content is None:
        return None
    return _parse_lock_data_from_bytes(content)


def get_pvt_data_from_revision(pvt_rel_path: str, rev: str) -> PvtData | str | None:
    """Read and parse .pvt file from a git revision: None when the revision has no valid .pvt
    file there, else its data, or why a restore cannot use it."""
    content = git.read_file_from_revision(pvt_rel_path, rev)
    if content is None:
        return None
    return _parse_pvt_data_from_bytes(content)


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
) -> list[TargetInfo]:
    """Resolve targets to TargetInfo: a target in ``stage_targets`` is a stage, others are files."""
    proj_root = project.get_project_root()
    results = list[TargetInfo]()

    for target in targets:
        if target in stage_targets:
            stage_target = stage_targets[target]
            stage_name = stage_target["stage_name"]
            lock_data = get_lock_data_from_revision(stage_name, rev, stage_target["state_dir"])
            if lock_data is None:
                raise exceptions.TargetNotFoundError(
                    f"Stage '{stage_name}' has no lock file at revision '{rev}'"
                )
            paths_by_key = stage_target["paths_by_key"]
            selected_key = types.identity_from_key(target).key
            hashes = dict[str, HashInfo | None]()
            for entry in lock_data["outs"]:
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
                hashes[path] = lock.entry_hash_info(entry)
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
                )
            )
            continue

        # Normalize path (validates traversal and project bounds)
        rel_target = _normalize_target_path(target, proj_root)
        pvt_rel_path = rel_target + ".pvt"

        # Try as a .pvt tracked file. One that is not valid is this target's error, so the other
        # targets are still restored and reported.
        pvt_data = get_pvt_data_from_revision(pvt_rel_path, rev)
        if isinstance(pvt_data, str):
            results.append(
                TargetInfo(
                    target_type=TargetType.FILE,
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: None},
                    error=f"{pvt_rel_path} at {rev} is not a valid .pvt file: {pvt_data}",
                )
            )
            continue
        if pvt_data is not None:
            results.append(
                TargetInfo(
                    target_type=TargetType.FILE,
                    original_target=target,
                    paths=[rel_target],
                    hashes={rel_target: track.pvt_to_hash_info(pvt_data)},
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
                )
            )
            continue

        raise exceptions.TargetNotFoundError(
            f"Target '{target}' not found at revision '{rev}' (not a stage name, tracked file, or git-tracked file)"
        )

    return results


def _write_fallback_content(dest_path: pathlib.Path, content: bytes) -> None:
    """Atomically replace a cache-miss target under the cache restore lock."""

    def write_content(fd: int) -> None:
        with os.fdopen(fd, "wb") as file:
            file.write(content)

    with cache.restore_lock(dest_path, config.get_state_dir()) as lock_token:
        staging_path = (
            cache.restore_staging_path(dest_path, lock_token) if lock_token is not None else None
        )
        cache.atomic_write_file(dest_path, write_content, staging_path=staging_path)


def _reason(error: exceptions.PivotError | OSError) -> str:
    """``error``'s message, followed on the next line by the tip a pivot error carries."""
    if isinstance(error, exceptions.PivotError) and (tip := error.get_suggestion()):
        return f"{error}\n  Tip: {tip}"
    return str(error)


def _download_error(failed: list[TransferResult]) -> exceptions.RemoteError:
    """The error of a download whose ``failed`` files did not arrive: a file's connection failure
    if there is one, since it ends the remote step for the rest of the get, else the first file's
    error."""
    unreachable = [result for result in failed if "connection_failed" in result]
    result = (unreachable or failed)[0]
    message = result["error"] if "error" in result else "unknown error"
    if unreachable:
        return exceptions.RemoteConnectionError(message)
    return exceptions.RemoteError(message)


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
) -> RestoreResult:
    """Restore a single file from revision and return its result.

    Tries the local cache, then git, then downloads the cache objects the output still misses
    from the remote ``remote_factory`` opens; it is called only when that download is needed.
    A remote that raised ``RemoteConnectionError`` is named in the result's
    ``unreachable_remote``.
    """
    path_str = str(dest_path)
    files_dir = cache_dir / "files"

    if dest_path.exists() and not force:
        return RestoreResult(
            status=RestoreStatus.SKIPPED,
            path=path_str,
            message=f"Skipped: {dest_path} (already exists, use --force to overwrite)",
        )

    # Ensure parent directory exists before any restore strategy
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return RestoreResult(
            status=RestoreStatus.ERROR,
            path=path_str,
            message=f"Error: {dest_path} - cannot create its directory: {e}",
        )

    # Strategy 1: Try local cache (if hash available)
    if output_hash is not None:
        try:
            # For directories with manifest, restore_from_cache checks each file in manifest.
            # For files, we check if the single cached file exists first.
            should_try_cache = True
            if "manifest" not in output_hash:
                should_try_cache = cache.get_cache_path(files_dir, output_hash["hash"]).exists()
            if should_try_cache:
                success = cache.restore_from_cache(
                    dest_path,
                    output_hash,
                    files_dir,
                    checkout_modes=checkout_modes,
                    state_dir=config.get_state_dir(),
                )
                if success:
                    return RestoreResult(
                        status=RestoreStatus.RESTORED,
                        path=path_str,
                        message=f"Restored: {dest_path} (from cache)",
                    )
        except (exceptions.PivotError, OSError) as e:
            # Strategy 3 retries this restore, and reports the error if it happens again
            logger.debug(f"Cache restore failed for {rel_path}: {e}")

    # Strategy 2: Try git fallback
    content = git.read_file_from_revision(rel_path, rev)
    if content is not None:
        try:
            _write_fallback_content(dest_path, content)
            return RestoreResult(
                status=RestoreStatus.RESTORED,
                path=path_str,
                message=f"Restored: {dest_path} (from git)",
            )
        except OSError as e:
            return RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to write from git: {e}",
            )

    # Strategy 3: Download the cache objects the output still misses from the remote, then
    # restore from the cache. Any failure is this target's error, so the targets already
    # restored are still reported.
    if output_hash is not None:
        source = "cache"
        remote: RemoteStorage | None = None
        try:
            missing_hashes = cache.missing_cache_objects(output_hash, files_dir, checkout_modes)
            if missing_hashes:
                source = "remote"
                remote = remote_factory()
                download_results = asyncio.run(
                    remote.download_batch(
                        [
                            (file_hash, cache.get_cache_path(files_dir, file_hash))
                            for file_hash in missing_hashes
                        ],
                        readonly=True,
                    )
                )
                if failed := [result for result in download_results if not result["success"]]:
                    raise _download_error(failed)
            success = cache.restore_from_cache(
                dest_path,
                output_hash,
                files_dir,
                checkout_modes=checkout_modes,
                state_dir=config.get_state_dir(),
            )
        except exceptions.RemoteError as e:
            failure = RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to download from remote: {_reason(e)}",
            )
            if remote is not None and isinstance(e, exceptions.RemoteConnectionError):
                failure["unreachable_remote"] = remote.url
            return failure
        except (exceptions.PivotError, OSError) as e:
            return RestoreResult(
                status=RestoreStatus.ERROR,
                path=path_str,
                message=f"Error: {dest_path} - failed to restore from {source}: {_reason(e)}",
            )
        if success:
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
    target_infos = resolve_targets(targets, rev, stage_targets)

    # Validate -o usage
    if output is not None:
        if len(target_infos) != 1:
            raise exceptions.GetError("--output/-o can only be used with a single target")
        if len(target_infos[0]["paths"]) != 1:
            raise exceptions.GetError("--output/-o can only be used with a single output")

    results = list[RestoreResult]()
    # Resolves the remote and builds its backend once per call, not once per target that misses
    # the cache
    open_remote = functools.cache(remote_factory)
    # A remote this call could not reach is not tried again, so an unreachable remote costs one
    # connection timeout per call, not one per target
    unreachable: str | None = None

    def reachable_remote() -> RemoteStorage:
        if unreachable is not None:
            raise exceptions.RemoteConnectionError(
                f"{unreachable} could not be reached for an earlier target, "
                + "so it is not tried again"
            )
        return open_remote()

    for target_info in target_infos:
        for rel_path in target_info["paths"]:
            output_hash = target_info["hashes"].get(rel_path)
            dest_path = output if output is not None else proj_root / rel_path
            if "error" in target_info:
                results.append(
                    RestoreResult(
                        status=RestoreStatus.ERROR,
                        path=str(dest_path),
                        message=f"Error: {dest_path} - {target_info['error']}",
                    )
                )
                continue

            result = restore_file(
                rel_path=rel_path,
                output_hash=output_hash,
                rev=rev,
                dest_path=dest_path,
                cache_dir=cache_dir,
                checkout_modes=checkout_modes,
                force=force,
                remote_factory=reachable_remote,
            )
            if "unreachable_remote" in result:
                unreachable = result["unreachable_remote"]
            results.append(result)

    # Format output: show all messages, with errors summarized at the end
    messages = [r["message"] for r in results]
    errors = [r for r in results if r["status"] == RestoreStatus.ERROR]

    if errors:
        messages.append("")
        messages.append(f"Failed to restore {len(errors)} file(s). See errors above.")

    return (messages, len(errors) == 0)
