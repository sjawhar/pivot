from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import TYPE_CHECKING

from pivot import cache, exceptions, lock, project, remote_config
from pivot import remote as remote_mod
from pivot.types import RemoteStatus, TransferSummary

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot import state as state_mod

logger = logging.getLogger(__name__)


def _get_cache_files_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    """Get the files subdirectory of the cache."""
    return cache_dir / "files"


def get_local_cache_hashes(cache_dir: pathlib.Path) -> set[str]:
    """Scan local cache and return all content hashes."""
    files_dir = _get_cache_files_dir(cache_dir)
    if not files_dir.exists():
        return set()

    hashes = set[str]()
    for prefix_dir in files_dir.iterdir():
        if not prefix_dir.is_dir() or len(prefix_dir.name) != 2:
            continue
        for hash_file in prefix_dir.iterdir():
            if hash_file.is_file():
                full_hash = prefix_dir.name + hash_file.name
                if len(full_hash) == cache.XXHASH64_HEX_LENGTH:
                    hashes.add(full_hash)

    return hashes


def get_stage_output_hashes(cache_dir: pathlib.Path, stage_names: list[str]) -> set[str]:
    """Extract output hashes from lock files for specific stages."""
    hashes = set[str]()

    for stage_name in stage_names:
        stage_lock = lock.StageLock(stage_name, cache_dir)
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


def get_stage_dep_hashes(cache_dir: pathlib.Path, stage_names: list[str]) -> set[str]:
    """Extract dependency hashes from lock files for specific stages."""
    hashes = set[str]()

    for stage_name in stage_names:
        stage_lock = lock.StageLock(stage_name, cache_dir)
        lock_data = stage_lock.read()
        if lock_data is None:
            continue

        for dep_hash in lock_data["dep_hashes"].values():
            hashes.add(dep_hash["hash"])
            if "manifest" in dep_hash:
                for entry in dep_hash["manifest"]:
                    hashes.add(entry["hash"])

    return hashes


async def compare_status(
    local_hashes: set[str],
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    jobs: int = 20,
) -> RemoteStatus:
    """Compare local cache against remote, using index to minimize HEAD requests."""
    if not local_hashes:
        return RemoteStatus(local_only=set(), remote_only=set(), common=set())

    known_on_remote = state_db.remote_hashes_intersection(remote_name, local_hashes)
    unknown_hashes = local_hashes - known_on_remote

    if unknown_hashes:
        existence = await remote.bulk_exists(list(unknown_hashes), concurrency=jobs)
        newly_found = {h for h, exists in existence.items() if exists}
        state_db.remote_hashes_add(remote_name, newly_found)
        known_on_remote = known_on_remote | newly_found

    local_only = local_hashes - known_on_remote
    common = local_hashes & known_on_remote

    return RemoteStatus(local_only=local_only, remote_only=set(), common=common)


async def _push_async(
    cache_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    stages: list[str] | None = None,
    jobs: int = 20,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Push cache files to remote (async implementation)."""
    if stages:
        local_hashes = get_stage_output_hashes(cache_dir, stages)
    else:
        local_hashes = get_local_cache_hashes(cache_dir)

    if not local_hashes:
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])

    status = await compare_status(local_hashes, remote, state_db, remote_name, jobs)

    if not status["local_only"]:
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

    return TransferSummary(
        transferred=len(transferred),
        skipped=len(status["common"]),
        failed=len(failed),
        errors=errors,
    )


def push(
    cache_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    stages: list[str] | None = None,
    jobs: int = 20,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Push cache files to remote storage."""
    return asyncio.run(
        _push_async(cache_dir, remote, state_db, remote_name, stages, jobs, callback)
    )


async def _pull_async(
    cache_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    stages: list[str] | None = None,
    jobs: int = 20,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Pull cache files from remote (async implementation)."""
    if stages:
        needed_hashes = get_stage_output_hashes(cache_dir, stages)
        needed_hashes |= get_stage_dep_hashes(cache_dir, stages)
    else:
        needed_hashes = await remote.list_hashes()

    if not needed_hashes:
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])

    local_hashes = get_local_cache_hashes(cache_dir)
    missing_locally = needed_hashes - local_hashes

    if not missing_locally:
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

    return TransferSummary(
        transferred=len(transferred),
        skipped=len(needed_hashes) - len(missing_locally),
        failed=len(failed),
        errors=errors,
    )


def pull(
    cache_dir: pathlib.Path,
    remote: remote_mod.S3Remote,
    state_db: state_mod.StateDB,
    remote_name: str,
    stages: list[str] | None = None,
    jobs: int = 20,
    callback: Callable[[int], None] | None = None,
) -> TransferSummary:
    """Pull cache files from remote storage."""
    return asyncio.run(
        _pull_async(cache_dir, remote, state_db, remote_name, stages, jobs, callback)
    )


def get_default_cache_dir() -> pathlib.Path:
    """Get the default cache directory for the current project."""
    return project.get_project_root() / ".pivot" / "cache"


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
