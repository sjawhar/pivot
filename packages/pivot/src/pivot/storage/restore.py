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
from pivot.storage import cache, lock, ownership, track
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


def _write_fallback_content(
    dest_path: pathlib.Path, content: bytes, lock_token: str | None
) -> bool:
    """Atomically replace a cache-miss target; the caller holds its restore lock."""

    def write_content(fd: int) -> None:
        with os.fdopen(fd, "wb") as file:
            file.write(content)

    staging_path = (
        cache.restore_staging_path(dest_path, lock_token) if lock_token is not None else None
    )
    cache.atomic_write_file(dest_path, write_content, staging_path=staging_path)
    return True


def _replace_checked(
    dest_path: pathlib.Path,
    target: ownership.ForcedRestore | None,
    replace: Callable[[str | None], bool],
) -> bool:
    """Run ``replace`` under ``dest_path``'s restore lock, right after checking ``target`` again
    there (``ownership.forced_restore_refusal``): a user file that appeared since the check of
    every target refuses this target alone.

    Raises:
        exceptions.OutputOwnershipError: Replacing ``dest_path`` would remove a user's file.
    """
    with cache.restore_lock(dest_path, config.get_state_dir()) as lock_token:
        if target is not None:
            refusal = ownership.forced_restore_refusal(project.get_project_root(), [target])
            if refusal is not None:
                raise refusal
        return replace(lock_token)


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
    target: ownership.ForcedRestore | None = None,
    *,
    remote_factory: Callable[[], RemoteStorage],
) -> RestoreResult:
    """Restore a single file from revision and return its result.

    Tries the local cache, then git, then downloads the cache objects the output still misses
    from the remote ``remote_factory`` opens; it is called only when that download is needed.
    A remote that raised ``RemoteUnreachableError`` is named in the result's
    ``unreachable_remote``.

    ``target`` is what the restore replaces (``_restore_targets``). Each write checks it again
    under the restore lock, just before it replaces ``dest_path``, and a refusal is this target's
    error. For a stage output restored to its own path, the ``*.pvt`` members ``output_hash``
    writes are recorded as the stage's before anything is written.
    """
    if (skipped := _skipped(dest_path, force)) is not None:
        return skipped

    if target is not None and target["claims"] is not None and output_hash is not None:
        ownership.record_restored(target["claims"], dest_path, output_hash)

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
        )
    except exceptions.OutputOwnershipError as e:
        return RestoreResult(
            status=RestoreStatus.ERROR, path=str(dest_path), message=f"Error: {dest_path} - {e}"
        )


def _restore_from_sources(
    rel_path: str,
    output_hash: HashInfo | None,
    rev: str,
    dest_path: pathlib.Path,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    target: ownership.ForcedRestore | None,
    remote_factory: Callable[[], RemoteStorage],
) -> RestoreResult:
    """``restore_file``'s strategies, each write under ``_replace_checked``.

    Raises:
        exceptions.OutputOwnershipError: The check under the restore lock refused the target.
    """
    path_str = str(dest_path)
    files_dir = cache_dir / "files"
    state_dir = config.get_state_dir()

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
        cached_hash = output_hash
        try:
            # For directories with manifest, restore_from_cache checks each file in manifest.
            # For files, we check if the single cached file exists first.
            should_try_cache = True
            if "manifest" not in output_hash:
                should_try_cache = cache.get_cache_path(files_dir, output_hash["hash"]).exists()
            if should_try_cache and _replace_checked(
                dest_path,
                target,
                lambda token: cache.restore_from_cache(
                    dest_path,
                    cached_hash,
                    files_dir,
                    checkout_modes=checkout_modes,
                    state_dir=state_dir,
                    lock_token=token,
                ),
            ):
                return RestoreResult(
                    status=RestoreStatus.RESTORED,
                    path=path_str,
                    message=f"Restored: {dest_path} (from cache)",
                )
        except exceptions.OutputOwnershipError:
            raise
        except (exceptions.PivotError, OSError) as e:
            # Strategy 3 retries this restore, and reports the error if it happens again
            logger.debug(f"Cache restore failed for {rel_path}: {e}")

    # Strategy 2: Try git fallback
    content = git.read_file_from_revision(rel_path, rev)
    if content is not None:
        try:
            _replace_checked(
                dest_path, target, lambda token: _write_fallback_content(dest_path, content, token)
            )
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
        cached_hash = output_hash
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
            success = _replace_checked(
                dest_path,
                target,
                lambda token: cache.restore_from_cache(
                    dest_path,
                    cached_hash,
                    files_dir,
                    checkout_modes=checkout_modes,
                    state_dir=state_dir,
                    lock_token=token,
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


def _current_output_hashes(stage_target: StageTarget) -> dict[str, HashInfo]:
    """The hashes the stage's current lock records, by workspace path relative to the root."""
    lock_data = lock.StageLock(
        stage_target["stage_name"], lock.get_stages_dir(stage_target["state_dir"])
    ).read()
    if lock_data is None:
        return {}
    recorded_paths = (
        lock_data["output_paths"]
        if "output_paths" in lock_data
        else dict[types.ArtifactIdentity, str]()
    )
    hashes = dict[str, HashInfo]()
    for identity, output_hash in lock_data["output_hashes"].items():
        path = recorded_paths.get(identity) or stage_target["paths_by_key"].get(identity.key)
        if path is not None:
            hashes[path] = output_hash
    return hashes


def _read_claims(
    proj_root: pathlib.Path, stage_target: StageTarget, files_dir: pathlib.Path
) -> ownership.StageClaims:
    """The stage's claims (``ownership.StageClaims``): the hashes its current lock records, its
    ``nocommit:`` record and the cache object root ``files_dir``, with a recorder for the
    ``*.pvt`` members ``get`` writes."""
    return ownership.read_stage_claims(
        proj_root,
        stage_target["stage_name"],
        stage_target["state_dir"],
        {
            proj_root / path: output_hash
            for path, output_hash in _current_output_hashes(stage_target).items()
        },
        cache_dir=files_dir,
        recorder=functools.partial(
            ownership.record_members,
            proj_root,
            stage_target["state_dir"],
            stage_target["stage_name"],
        ),
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
                restores[(index, rel_path)] = ownership.ForcedRestore(
                    replacement=ownership.destination(dest_path), restored=restored, claims=None
                )
            elif stage_target is not None:
                restores[(index, rel_path)] = ownership.ForcedRestore(
                    replacement=ownership.stage_output(dest_path, stage_target["stage_name"]),
                    restored=restored,
                    claims=claims,
                )
            else:
                restores[(index, rel_path)] = ownership.ForcedRestore(
                    replacement=ownership.tracked_restore(dest_path), restored=restored, claims=None
                )
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
    output_path = pathlib.Path(os.path.abspath(output)) if output is not None else None

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
    # the cache
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

    for index, target_info in enumerate(target_infos):
        for rel_path in target_info["paths"]:
            output_hash = target_info["hashes"].get(rel_path)
            dest_path = output_path if output_path is not None else proj_root / rel_path
            if "error" in target_info:
                # A target already in the workspace is kept, as for a valid .pvt
                results.append(
                    _skipped(dest_path, force)
                    or RestoreResult(
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
                target=restores[(index, rel_path)],
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
