from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from pivot import config, exceptions, metrics, names, project, registry, types
from pivot.remote import config as remote_config
from pivot.remote import protocol
from pivot.storage import cache, lock, track
from pivot.storage import store as store_mod
from pivot.types import HashInfo, RemoteStatus, TransferSummary, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.registry import RegistryStageInfo
    from pivot.storage import state as state_mod

logger = logging.getLogger(__name__)


def get_local_cache_hashes(cache_dir: pathlib.Path) -> set[str]:
    """Scan current and legacy default caches for all local content hashes."""
    _t = metrics.start()
    files_dirs = [cache_dir / "files"]
    legacy_files_dir = project.legacy_cache_files_dir_for(files_dirs[0])
    if legacy_files_dir is not None:
        files_dirs.append(legacy_files_dir)

    hashes = set[str]()
    for files_dir in files_dirs:
        if not files_dir.exists():
            continue
        is_legacy = files_dir == legacy_files_dir
        with os.scandir(files_dir) as prefix_entries:
            for prefix_entry in prefix_entries:
                if not prefix_entry.is_dir() or len(prefix_entry.name) != 2:
                    continue
                with os.scandir(prefix_entry.path) as hash_entries:
                    for hash_entry in hash_entries:
                        full_hash = prefix_entry.name + hash_entry.name
                        object_path = pathlib.Path(hash_entry.path)
                        if cache.is_valid_hash(full_hash) and cache.is_cache_object_path(
                            object_path
                        ):
                            hashes.add(full_hash)
                            if is_legacy:
                                project.report_legacy_state_dir(cache_dir.parent)

    metrics.end("sync.get_local_cache_hashes", _t)
    return hashes


def _extract_file_hashes_from_hash_info(hash_info: HashInfo) -> set[str]:
    """Extract hashes that correspond to cached file blobs only.

    For directories, returns only manifest entry hashes (not the tree hash).
    """
    if is_dir_hash(hash_info):
        return {entry["hash"] for entry in hash_info["manifest"]}
    return {hash_info["hash"]}


def get_stage_output_hashes(state_dir: pathlib.Path, stage_names: list[str]) -> set[str]:
    """Extract output hashes from lock files for specific stages.

    Only includes hashes for outputs with cache=True.
    """
    hashes = set[str]()
    from pivot.cli import helpers as cli_helpers

    for stage_name in stage_names:
        try:
            stage_info = cli_helpers.get_stage(stage_name)
        except KeyError:
            logger.warning(f"Stage {stage_name} not found in registry, skipping")
            continue
        non_cached_paths = {
            types.identity_key(out.identity)
            for out in stage_info["outs"]
            if out.tag is types.ArtifactTag.METRIC
        }

        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data is None:
            logger.warning(f"No lock file for stage '{stage_name}'")
            continue

        for out_path, output_hash in lock_data["output_hashes"].items():
            if types.identity_key(out_path) not in non_cached_paths:
                hashes |= _extract_file_hashes_from_hash_info(output_hash)

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
            hashes |= _extract_file_hashes_from_hash_info(dep_hash)

    return hashes


def _get_file_hash_from_stages(
    target_identity_key: str,
    state_dir: pathlib.Path,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> HashInfo | None:
    """Look up a file's hash from stage lock files."""
    if all_stages is not None:
        for stage_name, stage_info in all_stages.items():
            stage_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
            stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
            lock_data = stage_lock.read()
            if lock_data is None:
                continue
            non_cached_paths = {
                types.identity_key(out.identity)
                for out in stage_info["outs"]
                if out.tag is types.ArtifactTag.METRIC
            }
            for out_path, out_hash in lock_data["output_hashes"].items():
                identity_key = types.identity_key(out_path)
                if identity_key == target_identity_key and identity_key not in non_cached_paths:
                    return out_hash
        return None

    from pivot.cli import helpers as cli_helpers

    # Stage names come from the configured stages directory and its 0.5.2 counterpart; each lock
    # is read through StageLock, which finds a stage's lock where it is, one stage at a time.
    stages_dir = lock.get_stages_dir(state_dir)
    stage_names = set[str]()
    for scanned_dir in (stages_dir, project.legacy_stages_dir_for(stages_dir)):
        if scanned_dir is not None and scanned_dir.is_dir():
            stage_names.update(
                lock_file.relative_to(scanned_dir).with_suffix("").as_posix()
                for lock_file in scanned_dir.rglob("*.lock")
            )

    for stage_name in sorted(stage_names):
        try:
            stage_info = cli_helpers.get_stage(stage_name)
            non_cached_paths = {
                types.identity_key(out.identity)
                for out in stage_info["outs"]
                if out.tag is types.ArtifactTag.METRIC
            }
        except (KeyError, exceptions.PivotError) as exc:
            logger.debug("Stage %s not available for cache filtering: %s", stage_name, exc)
            non_cached_paths = set[str]()

        stage_lock = lock.StageLock(stage_name, stages_dir)
        lock_data = stage_lock.read()
        if lock_data is None:
            continue

        for out_path, out_hash in lock_data["output_hashes"].items():
            identity_key = types.identity_key(out_path)
            if identity_key == target_identity_key and identity_key not in non_cached_paths:
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

    return track.pvt_to_hash_info(track_data)


def _get_tracked_hashes(proj_root: pathlib.Path) -> set[str]:
    """Blob hashes of every `pivot track`-ed file or directory in the project.

    Imports are left out: `pivot import` writes the file straight to the workspace, so its
    blob never enters the local cache and push never sends it to this project's remote.
    """
    hashes = set[str]()
    for pvt_data in track.discover_pvt_files(proj_root).values():
        if track.is_import(pvt_data):
            continue
        hashes |= _extract_file_hashes_from_hash_info(track.pvt_to_hash_info(pvt_data))
    return hashes


def _is_cached_stage_output(
    identity: types.ArtifactIdentity, producer_info: RegistryStageInfo
) -> bool:
    """Whether a stage output is stored in the cache (Metrics are git-tracked, not cached)."""
    return not any(
        out.identity == identity and out.tag is types.ArtifactTag.METRIC
        for out in producer_info["outs"]
    )


def get_target_hashes(
    targets: list[str],
    state_dir: pathlib.Path,
    include_deps: bool = False,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> set[str]:
    """Resolve targets (stage names or file paths) to cache hashes."""
    _t = metrics.start()
    proj_root = project.get_project_root()
    hashes = set[str]()
    output_identities: dict[str, types.ArtifactIdentity] | None = None
    unresolved = list[str]()
    tracked_hashes: set[str] | None = None

    for target in targets:
        original_target = target
        target_identity = types.identity_from_key(target)
        stage_name: str | None = None
        if all_stages is not None:
            resolved_name = names.resolve_stage_name(target_identity.producer, all_stages)
            if resolved_name in all_stages:
                stage_name = resolved_name

        if stage_name is not None and all_stages is not None:
            if target_identity.key is None:
                stage_info = all_stages[stage_name]
                target_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
                non_cached_paths = {
                    types.identity_key(out.identity)
                    for out in stage_info["outs"]
                    if out.tag is types.ArtifactTag.METRIC
                }

                try:
                    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(target_state_dir))
                except ValueError:
                    stage_lock = None

                if stage_lock is not None:
                    lock_data = stage_lock.read()
                    if lock_data is not None:
                        for out_path, out_hash in lock_data["output_hashes"].items():
                            if types.identity_key(out_path) not in non_cached_paths:
                                hashes |= _extract_file_hashes_from_hash_info(out_hash)
                        if include_deps:
                            # Only deps whose blobs pivot caches (and so pushes) can be fetched:
                            # upstream stage outputs and `pivot track` inputs. Raw inputs
                            # committed to git and `pivot import` inputs are never in the remote.
                            for dep_identity, dep_hash in lock_data["dep_hashes"].items():
                                dep_blobs = _extract_file_hashes_from_hash_info(dep_hash)
                                producer_info = all_stages.get(dep_identity.producer)
                                if producer_info is not None:
                                    if _is_cached_stage_output(dep_identity, producer_info):
                                        hashes |= dep_blobs
                                    continue
                                if tracked_hashes is None:
                                    tracked_hashes = _get_tracked_hashes(proj_root)
                                hashes |= dep_blobs & tracked_hashes
                        continue
            else:
                resolved_identity = types.ArtifactIdentity(stage_name, target_identity.key)
                out_hash = _get_file_hash_from_stages(
                    types.identity_key(resolved_identity),
                    state_dir,
                    all_stages,
                )
                if out_hash is not None:
                    hashes |= _extract_file_hashes_from_hash_info(out_hash)
                    continue

            unresolved.append(original_target)
            continue

        # Strip .pvt suffix if present (CLI normalizes these, but be defensive)
        target_p = pathlib.Path(target)
        if target_p.suffix == ".pvt":
            target = str(track.get_data_path(target_p))
            target_identity = types.identity_from_key(target)

        abs_path = str(project.normalize_path(target))
        rel_path = project.to_relative_path(abs_path, proj_root)

        # A workspace path of a stage output stands for that output's identity.
        if all_stages is not None:
            if output_identities is None:
                # Display paths depend only on qualified stage names, not the pipeline name.
                workspace = store_mod.WorkspaceStore(proj_root, "", {})
                output_identities = workspace.output_identities_by_path(
                    out for stage_info in all_stages.values() for out in stage_info["outs"]
                )
            target_identity = output_identities.get(rel_path, target_identity)

        out_hash = _get_file_hash_from_stages(
            types.identity_key(target_identity), state_dir, all_stages
        )
        if out_hash is not None:
            hashes |= _extract_file_hashes_from_hash_info(out_hash)
            continue

        pvt_hash = _get_file_hash_from_pvt(rel_path, proj_root)
        if pvt_hash is not None:
            hashes |= _extract_file_hashes_from_hash_info(pvt_hash)
            continue

        unresolved.append(target)

    if unresolved:
        logger.warning(f"Could not resolve targets: {', '.join(unresolved)}")

    metrics.end("sync.get_target_hashes", _t)
    return hashes


def check_remote_url(
    state_db: state_mod.StateDB,
    remote_name: str,
    remote: protocol.RemoteStorage,
) -> None:
    """Detect remote URL change and clear stale index."""
    current_url = remote.url
    stored_url = state_db.remote_get_url(remote_name)
    if stored_url is None:
        # First use — record URL, don't clear (fail-open)
        state_db.remote_set_url(remote_name, current_url)
    elif stored_url != current_url:
        logger.warning(
            "Remote '%s' URL changed (%s → %s), clearing cached index",
            remote_name,
            stored_url,
            current_url,
        )
        state_db.remote_index_clear(remote_name)
        state_db.remote_set_url(remote_name, current_url)
    else:
        logger.debug("Remote '%s' URL unchanged: %s", remote_name, current_url)


async def compare_status(
    local_hashes: set[str],
    remote: protocol.RemoteStorage,
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
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int, int, str], None] | None = None,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> TransferSummary:
    """Push cache files to remote (async implementation)."""
    _t = metrics.start()
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    check_remote_url(state_db, remote_name, remote)

    if targets:
        local_hashes = get_target_hashes(
            targets, state_dir, include_deps=False, all_stages=all_stages
        )
    else:
        local_hashes = get_local_cache_hashes(cache_dir)

    if not local_hashes:
        metrics.end("sync.push_async", _t)
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])

    status = await compare_status(local_hashes, remote, state_db, remote_name, jobs)

    if not status["local_only"]:
        metrics.end("sync.push_async", _t)
        return TransferSummary(transferred=0, skipped=len(status["common"]), failed=0, errors=[])

    files_dir = cache_dir / "files"
    items = list[tuple[pathlib.Path, str]]()
    skipped_non_file = 0
    for hash_ in status["local_only"]:
        cache_path = cache.resolve_cache_object_path(files_dir, hash_)
        if cache_path is not None and cache_path.is_file():
            items.append((cache_path, hash_))
        else:
            skipped_non_file += 1
    if skipped_non_file:
        logger.debug("Skipped %d non-file cache entries during push", skipped_non_file)

    results = await remote.upload_batch(items, concurrency=jobs, callback=callback)

    transferred = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    state_db.remote_hashes_add(remote_name, [r["hash"] for r in transferred])
    errors = [r["error"] for r in failed if "error" in r]

    metrics.end("sync.push_async", _t)
    return TransferSummary(
        transferred=len(transferred),
        skipped=len(status["common"]) + skipped_non_file,
        failed=len(failed),
        errors=errors,
    )


def push(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int, int, str], None] | None = None,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> TransferSummary:
    """Push cache files to remote storage."""
    return asyncio.run(
        _push_async(
            cache_dir,
            state_dir,
            remote,
            state_db,
            remote_name,
            targets,
            jobs,
            callback,
            all_stages,
        )
    )


async def _pull_async(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int, int, str], None] | None = None,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> TransferSummary:
    """Pull cache files from remote (async implementation)."""
    _t = metrics.start()
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    check_remote_url(state_db, remote_name, remote)

    if targets:
        needed_hashes = get_target_hashes(
            targets, state_dir, include_deps=True, all_stages=all_stages
        )
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

    files_dir = cache_dir / "files"
    items = list[tuple[str, pathlib.Path]]()
    for hash_ in missing_locally:
        cache_path = cache.get_cache_path(files_dir, hash_)
        items.append((hash_, cache_path))

    results = await remote.download_batch(items, concurrency=jobs, callback=callback, readonly=True)

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
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    callback: Callable[[int, int, str], None] | None = None,
    all_stages: dict[str, RegistryStageInfo] | None = None,
) -> TransferSummary:
    """Pull cache files from remote storage."""
    return asyncio.run(
        _pull_async(
            cache_dir,
            state_dir,
            remote,
            state_db,
            remote_name,
            targets,
            jobs,
            callback,
            all_stages,
        )
    )


def create_remote_from_name(name: str | None = None) -> tuple[protocol.RemoteStorage, str]:
    """Open the configured remote by name (default if None). Returns (remote, name)."""
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

    return protocol.open_remote(url), resolved_name
