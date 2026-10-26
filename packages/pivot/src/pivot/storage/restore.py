from __future__ import annotations

import enum
import logging
import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

import xxhash
import yaml

from pivot import config, exceptions, git, project, types, yaml_config
from pivot.remote import storage as remote
from pivot.storage import cache, listing, lock, state, track
from pivot.types import HashInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pivot.storage.track import PvtData
    from pivot.types import DirHash

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


class TargetInfo(TypedDict):
    """Information about a target to retrieve."""

    target_type: TargetType
    original_target: str
    paths: list[str]
    hashes: dict[str, HashInfo | None]
    members: dict[str, listing.DirListing]  # each directory's listing at the revision, by path


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


def _parse_pvt_data_from_bytes(content: bytes) -> tuple[PvtData, listing.DirListing | None] | None:
    """Parse .pvt file content from bytes, with an older pivot's inline directory listing."""
    try:
        data: object = yaml.load(content, Loader=yaml_config.Loader)
    except yaml.YAMLError:
        return None
    return track.parse_pvt(data)


def get_lock_data_from_revision(
    stage_name: str, rev: str, state_dir: pathlib.Path
) -> lock.CommittedLock | None:
    """Read and parse lock file for a stage from a git revision."""
    stages_dir = lock.get_stages_dir(state_dir)
    rel_path = str(stages_dir.relative_to(project.get_project_root()) / f"{stage_name}.lock")
    content = git.read_file_from_revision(rel_path, rev)
    if content is None:
        return None
    return _parse_lock_data_from_bytes(content)


def get_pvt_data_from_revision(
    pvt_rel_path: str, rev: str
) -> tuple[PvtData, listing.DirListing | None] | None:
    """Read and parse .pvt file from a git revision."""
    content = git.read_file_from_revision(pvt_rel_path, rev)
    if content is None:
        return None
    return _parse_pvt_data_from_bytes(content)


def _listing_from_revision(rel_path: str, rev: str, record: DirHash) -> listing.DirListing | None:
    """A directory's committed listing file at a git revision, checked against ``record``;
    None when the revision has no such file."""
    content = git.read_file_from_revision(rel_path, rev)
    if content is None:
        return None
    source = f"{rel_path} at revision {rev}"
    return listing.parse(listing.check(content, record, source), record, source)


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
            committed = get_lock_data_from_revision(stage_name, rev, stage_target["state_dir"])
            if committed is None:
                raise exceptions.TargetNotFoundError(
                    f"Stage '{stage_name}' has no lock file at revision '{rev}'"
                )
            paths_by_key = stage_target["paths_by_key"]
            selected_key = types.identity_from_key(target).key
            hashes = dict[str, HashInfo | None]()
            members = dict[str, listing.DirListing]()
            stages_rel = lock.get_stages_dir(stage_target["state_dir"]).relative_to(proj_root)
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
                    listing_rel = listing.output_listing_path(
                        stages_rel, types.ArtifactIdentity(stage_name, key)
                    ).as_posix()
                    found = _listing_from_revision(listing_rel, rev, hash_info)
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

        # Try as a .pvt tracked file
        parsed_pvt = get_pvt_data_from_revision(pvt_rel_path, rev)
        if parsed_pvt is not None:
            pvt_data, inline_members = parsed_pvt
            pvt_hash = track.pvt_to_hash_info(pvt_data)
            pvt_members = dict[str, listing.DirListing]()
            if is_dir_hash(pvt_hash):
                found = inline_members or _listing_from_revision(
                    pvt_rel_path + listing.SUFFIX, rev, pvt_hash
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


def _verify_content_hash(content: bytes, expected_hash: str) -> bool:
    """Verify content matches expected xxhash64 hash."""
    actual_hash = xxhash.xxh64(content).hexdigest()
    return actual_hash == expected_hash


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


def restore_file(
    rel_path: str,
    output_hash: HashInfo | None,
    rev: str,
    dest_path: pathlib.Path,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
    hash_state_dir: pathlib.Path | None = None,
    members: listing.DirListing | None = None,
) -> RestoreResult:
    """Restore a single file from revision. Returns result with status.

    A restore whose content has a known hash records each file's hash in the StateDB of
    ``hash_state_dir`` (the configured state directory when None; a stage output's own state
    directory), so the next status, explanation or run looks it up instead of reading it.
    ``members`` is a directory's listing at the revision; None reads the cache's object.
    """
    path_str = str(dest_path)

    if dest_path.exists() and not force:
        return RestoreResult(
            status=RestoreStatus.SKIPPED,
            path=path_str,
            message=f"Skipped: {dest_path} (already exists, use --force to overwrite)",
        )

    # Ensure parent directory exists before any restore strategy
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Try local cache (if hash available). A directory needs its listing: the one
    # committed at the revision, else the cache's object.
    if output_hash is not None:
        if is_dir_hash(output_hash):
            if members is None:
                members = cache.cached_listing(cache_dir / "files", output_hash)
            should_try_cache = members is not None
        else:
            cached_path = cache.get_cache_path(cache_dir / "files", output_hash["hash"])
            should_try_cache = cached_path.exists()

        if should_try_cache:
            try:
                success = cache.restore_from_cache(
                    dest_path,
                    output_hash,
                    cache_dir / "files",
                    checkout_modes=checkout_modes,
                    state_dir=config.get_state_dir(),
                    members=members,
                )
            except OSError as e:
                logger.debug(f"Cache restore failed for {rel_path}: {e}")
                success = False
            if success:
                with state.StateDB(hash_state_dir or config.get_state_dir()) as state_db:
                    state_db.save_many(cache.known_file_hashes(dest_path, output_hash, members))
                return RestoreResult(
                    status=RestoreStatus.RESTORED,
                    path=path_str,
                    message=f"Restored: {dest_path} (from cache)",
                )

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

    # Strategy 3: Try remote fallback (if hash available)
    if output_hash is not None and not is_dir_hash(output_hash):
        file_hash = output_hash["hash"]
        remote_content = remote.fetch_from_remote(file_hash)
        if remote_content is not None:
            # Verify hash before trusting remote content
            if not _verify_content_hash(remote_content, file_hash):
                return RestoreResult(
                    status=RestoreStatus.ERROR,
                    path=path_str,
                    message=(
                        f"Error: {dest_path} - remote content corrupted (hash mismatch). "
                        f"Expected {file_hash[:8]}..., try re-pushing from source."
                    ),
                )
            try:
                _write_fallback_content(dest_path, remote_content)
            except OSError as e:
                return RestoreResult(
                    status=RestoreStatus.ERROR,
                    path=path_str,
                    message=f"Error: {dest_path} - failed to write from remote: {e}",
                )
            # Cache the fetched content for future use (best effort). An object already there
            # is kept: another process may be linking it (cache.publish_cache_object).
            try:
                cached_path = cache.get_cache_path(cache_dir / "files", file_hash)
                with cache.staged_cache_file(cached_path) as staged:
                    staged.write_bytes(remote_content)
                    os.chmod(staged, 0o444)
                    _ = cache.publish_cache_object(staged, cached_path)
            except OSError:
                pass  # Caching failure is non-fatal
            with state.StateDB(hash_state_dir or config.get_state_dir()) as state_db:
                state_db.save_many([(dest_path, dest_path.stat(), file_hash)])
            return RestoreResult(
                status=RestoreStatus.RESTORED,
                path=path_str,
                message=f"Restored: {dest_path} (from remote)",
            )

    return RestoreResult(
        status=RestoreStatus.ERROR,
        path=path_str,
        message=f"Error: {dest_path} - not in local cache, git, or remote",
    )


def restore_targets_from_revision(
    targets: Sequence[str],
    rev: str,
    output: pathlib.Path | None,
    cache_dir: pathlib.Path,
    stage_targets: Mapping[str, StageTarget],
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
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

    for target_info in target_infos:
        hash_state_dir = (
            stage_targets[target_info["original_target"]]["state_dir"]
            if target_info["target_type"] is TargetType.STAGE
            else None
        )
        for rel_path in target_info["paths"]:
            output_hash = target_info["hashes"].get(rel_path)
            dest_path = output if output is not None else proj_root / rel_path

            result = restore_file(
                rel_path=rel_path,
                output_hash=output_hash,
                rev=rev,
                dest_path=dest_path,
                cache_dir=cache_dir,
                checkout_modes=checkout_modes,
                force=force,
                hash_state_dir=hash_state_dir,
                members=target_info["members"].get(rel_path),
            )
            results.append(result)

    # Format output: show all messages, with errors summarized at the end
    messages = [r["message"] for r in results]
    errors = [r for r in results if r["status"] == RestoreStatus.ERROR]

    if errors:
        messages.append("")
        messages.append(f"Failed to restore {len(errors)} file(s). See errors above.")

    return (messages, len(errors) == 0)
