from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from pivot import config, exceptions, metrics, project
from pivot.remote import config as remote_config
from pivot.remote import storage as remote_mod
from pivot.storage import cache, lock, track
from pivot.types import DirHash, FileHash, HashInfo, RemoteStatus, TransferSummary

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.storage import state as state_mod

logger = logging.getLogger(__name__)


def _get_cache_files_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    """Get the files subdirectory of the cache."""
    return cache_dir / "files"


def get_local_cache_hashes(cache_dir: pathlib.Path) -> set[str]:
    """Scan local cache and return all content hashes.

    Uses os.scandir for efficiency - DirEntry caches stat results, eliminating
    redundant syscalls compared to pathlib.iterdir + is_file/is_dir.
    """
    _t = metrics.start()
    files_dir = _get_cache_files_dir(cache_dir)
    if not files_dir.exists():
        metrics.end("sync.get_local_cache_hashes", _t)
        return set()

    hashes = set[str]()
    with os.scandir(files_dir) as prefix_entries:
        for prefix_entry in prefix_entries:
            # DirEntry.is_dir() uses cached stat from scandir
            if not prefix_entry.is_dir() or len(prefix_entry.name) != 2:
                continue
            with os.scandir(prefix_entry.path) as hash_entries:
                for hash_entry in hash_entries:
                    # DirEntry.is_file() uses cached stat from scandir
                    if hash_entry.is_file():
                        full_hash = prefix_entry.name + hash_entry.name
                        if len(full_hash) == cache.XXHASH64_HEX_LENGTH:
                            hashes.add(full_hash)

    metrics.end("sync.get_local_cache_hashes", _t)
    return hashes


def get_stage_output_hashes(state_dir: pathlib.Path, stage_names: list[str]) -> set[str]:
    """Extract output hashes from lock files for specific stages."""
    hashes = set[str]()

    for stage_name in stage_names:
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data is None:
            logger.warning(f"No lock file for stage '{stage_name}'")
            continue

        for output_hash in lock_data["output_hashes"].values():
            if output_hash is None:
                continue
            hashes.add(output_hash["hash"])
            if "manifest" in output_hash:
                for entry in output_hash["manifest"]:
                    hashes.add(entry["hash"])

    return hashes


def get_stage_dep_hashes(state_dir: pathlib.Path, stage_names: list[str]) -> set[str]:
    """Extract dependency hashes from lock files for specific stages."""
    hashes = set[str]()

    for stage_name in stage_names:
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data is None:
            continue

        for dep_hash in lock_data["dep_hashes"].values():
            hashes.add(dep_hash["hash"])
            if "manifest" in dep_hash:
                for entry in dep_hash["manifest"]:
                    hashes.add(entry["hash"])

    return hashes


def _extract_hashes_from_output(output_hash: HashInfo) -> set[str]:
    """Extract all hashes from a HashInfo (file or directory)."""
    hashes = set[str]()
    hashes.add(output_hash["hash"])
    if "manifest" in output_hash:
        for entry in output_hash["manifest"]:
            hashes.add(entry["hash"])
    return hashes


def _get_file_hash_from_stages(rel_path: str, state_dir: pathlib.Path) -> HashInfo | None:
    """Look up a file's hash from stage lock files."""
    stages_dir = lock.get_stages_dir(state_dir)
    if not stages_dir.exists():
        return None

    for lock_file in stages_dir.glob("*.lock"):
        stage_name = lock_file.stem
        stage_lock = lock.StageLock(stage_name, stages_dir)
        lock_data = stage_lock.read()
        if lock_data is None:
            continue

        for out_path, out_hash in lock_data["output_hashes"].items():
            if out_hash is not None and out_path == rel_path:
                return out_hash

    return None


def _get_file_hash_from_pvt(rel_path: str, proj_root: pathlib.Path) -> HashInfo | None:
    """Look up a file's hash from .pvt tracking file."""
    pvt_path = proj_root / (rel_path + ".pvt")
    if not pvt_path.exists():
        return None

    track_data = track.read_pvt_file(pvt_path)
    if track_data is None:
        return None

    if "manifest" in track_data:
        return DirHash(hash=track_data["hash"], manifest=track_data["manifest"])
    return FileHash(hash=track_data["hash"])


def get_target_hashes(
    targets: list[str], state_dir: pathlib.Path, include_deps: bool = False
) -> set[str]:
    """Resolve targets (stage names or file paths) to cache hashes.

    Args:
        targets: List of stage names or file paths
        state_dir: State directory path (.pivot)
        include_deps: If True, also include dependency hashes for stages

    Returns:
        Set of content hashes for all resolved targets
    """
    _t = metrics.start()
    proj_root = project.get_project_root()
    hashes = set[str]()
    unresolved = list[str]()

    for target in targets:
        # Try as stage name first (no path separators)
        if "/" not in target and "\\" not in target:
            stage_lock = lock.StageLock(target, lock.get_stages_dir(state_dir))
            lock_data = stage_lock.read()
            if lock_data is not None:
                for out_hash in lock_data["output_hashes"].values():
                    if out_hash is not None:
                        hashes |= _extract_hashes_from_output(out_hash)
                if include_deps:
                    for dep_hash in lock_data["dep_hashes"].values():
                        hashes |= _extract_hashes_from_output(dep_hash)
                continue

        # Try as file path
        rel_path = project.to_relative_path(project.normalize_path(target), proj_root)

        # Check stage outputs
        out_hash = _get_file_hash_from_stages(rel_path, state_dir)
        if out_hash is not None:
            hashes |= _extract_hashes_from_output(out_hash)
            continue

        # Check .pvt tracked files
        pvt_hash = _get_file_hash_from_pvt(rel_path, proj_root)
        if pvt_hash is not None:
            hashes |= _extract_hashes_from_output(pvt_hash)
            continue

        unresolved.append(target)

    if unresolved:
        logger.warning(f"Could not resolve targets: {', '.join(unresolved)}")

    metrics.end("sync.get_target_hashes", _t)
    return hashes


async def compare_status(
    local_hashes: set[str],
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    jobs: int | None = None,
) -> RemoteStatus:
    """Compare local cache against remote, using index to minimize HEAD requests."""
    _t = metrics.start()
    if not local_hashes:
        metrics.end("sync.compare_status", _t)
        return RemoteStatus(local_only=set(), remote_only=set(), common=set())

    jobs = jobs if jobs is not None else config.get_remote_jobs()
    known_on_remote = state_db.remote_hashes_intersection(remote_name, local_hashes)
    unknown_hashes = local_hashes - known_on_remote

    if unknown_hashes:
        existence = await remote.bulk_exists(list(unknown_hashes), concurrency=jobs)
        newly_found = {h for h, exists in existence.items() if exists}
        state_db.remote_hashes_add(remote_name, newly_found)
        known_on_remote = known_on_remote | newly_found

    local_only = local_hashes - known_on_remote
    common = local_hashes & known_on_remote

    metrics.end("sync.compare_status", _t)
    return RemoteStatus(local_only=local_only, remote_only=set(), common=common)


async def _push_async(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Push cache files to remote (async implementation)."""
    _t = metrics.start()
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    if targets:
        local_hashes = get_target_hashes(targets, state_dir, include_deps=False)
    else:
        local_hashes = get_local_cache_hashes(cache_dir)

    if not local_hashes:
        metrics.end("sync.push_async", _t)
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])

    status = await compare_status(local_hashes, remote, state_db, remote_name, jobs)

    if not status["local_only"]:
        metrics.end("sync.push_async", _t)
        return TransferSummary(transferred=0, skipped=len(status["common"]), failed=0, errors=[])

    files_dir = _get_cache_files_dir(cache_dir)
    items = list[tuple[pathlib.Path, str]]()
    for hash_ in status["local_only"]:
        cache_path = cache.get_cache_path(files_dir, hash_)
        if cache_path.exists():
            items.append((cache_path, hash_))

    results = await remote.upload_batch(items, concurrency=jobs, callback=callback)

    transferred = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    state_db.remote_hashes_add(remote_name, [r["hash"] for r in transferred])

    errors = [r["error"] for r in failed if "error" in r]

    metrics.end("sync.push_async", _t)
    return TransferSummary(
        transferred=len(transferred),
        skipped=len(status["common"]),
        failed=len(failed),
        errors=errors,
    )


def push(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Push cache files to remote storage."""
    return asyncio.run(
        _push_async(cache_dir, state_dir, remote, state_db, remote_name, targets, jobs, callback)
    )


async def _pull_async(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Pull cache files from remote (async implementation)."""
    _t = metrics.start()
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    if targets:
        needed_hashes = get_target_hashes(targets, state_dir, include_deps=True)
    else:
        needed_hashes = await remote.list_hashes()

    if not needed_hashes:
        metrics.end("sync.pull_async", _t)
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])

    local_hashes = get_local_cache_hashes(cache_dir)
    missing_locally = needed_hashes - local_hashes

    if not missing_locally:
        metrics.end("sync.pull_async", _t)
        return TransferSummary(transferred=0, skipped=len(needed_hashes), failed=0, errors=[])

    files_dir = _get_cache_files_dir(cache_dir)
    items = list[tuple[str, pathlib.Path]]()
    for hash_ in missing_locally:
        cache_path = cache.get_cache_path(files_dir, hash_)
        items.append((hash_, cache_path))

    results = await remote.download_batch(items, concurrency=jobs, callback=callback)

    transferred = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    state_db.remote_hashes_add(remote_name, [r["hash"] for r in transferred])

    errors = [r["error"] for r in failed if "error" in r]

    metrics.end("sync.pull_async", _t)
    return TransferSummary(
        transferred=len(transferred),
        skipped=len(needed_hashes) - len(missing_locally),
        failed=len(failed),
        errors=errors,
    )


def pull(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Pull cache files from remote storage."""
    return asyncio.run(
        _pull_async(cache_dir, state_dir, remote, state_db, remote_name, targets, jobs, callback)
    )


def create_remote_from_name(name: str | None = None) -> tuple[remote_mod.S3Remote, str]:
    """Create S3Remote from configured remote name. Returns (remote, name)."""
    url = remote_config.get_remote_url(name)
    resolved_name = name
    if resolved_name is None:
        resolved_name = remote_config.get_default_remote()
        if resolved_name is None:
            remotes = remote_config.list_remotes()
            if len(remotes) == 1:
                resolved_name = next(iter(remotes.keys()))
            else:
                raise exceptions.RemoteNotFoundError("Could not determine remote name")

    return remote_mod.S3Remote(url), resolved_name
