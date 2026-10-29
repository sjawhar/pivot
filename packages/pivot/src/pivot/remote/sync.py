from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from pivot import config, exceptions, metrics, names, project, registry, types
from pivot import state_dir as state_dir_mod
from pivot.remote import config as remote_config
from pivot.remote import protocol
from pivot.storage import cache, listing, lock, member_skip, track
from pivot.storage import store as store_mod
from pivot.types import HashInfo, LockData, RemoteStatus, TransferSummary, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.registry import RegistryStageInfo
    from pivot.storage import state as state_mod

logger = logging.getLogger(__name__)


def get_local_cache_hashes(cache_dir: pathlib.Path) -> set[str]:
    """Scan configured and legacy cache roots for locally available content hashes."""
    _t = metrics.start()
    files_dirs = [cache_dir / "files"]
    if (other_files_dir := state_dir_mod.other_cache_files_dir_for(files_dirs[0])) is not None:
        files_dirs.append(other_files_dir)

    hashes = set[str]()
    for index, files_dir in enumerate(files_dirs):
        if not files_dir.is_dir():
            continue
        with os.scandir(files_dir) as prefix_entries:
            for prefix_entry in prefix_entries:
                if not prefix_entry.is_dir() or len(prefix_entry.name) != 2:
                    continue
                with os.scandir(prefix_entry.path) as hash_entries:
                    for hash_entry in hash_entries:
                        if not hash_entry.is_file(follow_symlinks=False):
                            continue
                        full_hash = prefix_entry.name + hash_entry.name
                        if cache.is_valid_hash(full_hash):
                            hashes.add(full_hash)
                            if index:
                                state_dir_mod.report_legacy_state_dir(cache_dir.parent)

    metrics.end("sync.get_local_cache_hashes", _t)
    return hashes


def _blob_hashes(
    hash_info: HashInfo,
    read_members: Callable[[types.DirHash], listing.DirListing],
    files_dir: pathlib.Path,
    *,
    require_members: bool = False,
) -> set[str]:
    """The cache objects that store an output: a file's object, or a directory's listing object
    and its files' objects.

    ``read_members`` reads a directory's listing (its committed listing file, else the cache's
    object). A listing neither the configured cache nor the legacy one holds is stored from the
    committed file, so push sends it. A directory whose listing neither holds, or whose committed
    listing cannot be used and the cache lacks, gives its listing object alone: a pull fetches it
    first, then names the files.

    With ``require_members`` the caller (push) needs the member objects it would transfer, so a
    directory whose listing can be read from neither the committed file nor the cache raises
    (``CorruptListingError``/``ListingMissingError``, naming the file) rather than naming a
    listing object that is not there — there is nothing to send for it, and reporting it
    skipped would read as success.
    """
    if not is_dir_hash(hash_info):
        return {hash_info["hash"]}
    try:
        members = read_members(hash_info)
    except exceptions.ListingMissingError:
        if require_members:
            raise
        return {hash_info["hash"]}
    except exceptions.CorruptListingError as e:
        if require_members:
            raise
        logger.warning("%s; only the cache's copy of that listing is transferred", e)
        return {hash_info["hash"]}
    if cache.resolve_cache_object_path(files_dir, hash_info["hash"]) is None:
        listing.store_object(files_dir, hash_info["hash"], listing.encode(members))
    return {hash_info["hash"], *(entry["hash"] for entry in members["files"])}


def _output_blobs(
    hash_info: HashInfo,
    stages_dir: pathlib.Path,
    identity: types.ArtifactIdentity,
    files_dir: pathlib.Path,
    lock_data: LockData | None,
    *,
    require_members: bool = False,
) -> set[str]:
    """The cache objects of a stage output, a directory's listing read beside its lock."""
    committed = listing.output_listing_path(stages_dir, identity)
    if lock_data is not None:
        committed = lock.committed_listing(lock_data, committed)
    if committed is not None and not committed.is_file():
        other_stages_dir = state_dir_mod.other_stages_dir_for(stages_dir)
        if other_stages_dir is not None:
            legacy = listing.output_listing_path(other_stages_dir, identity)
            if legacy.is_file():
                state_dir_mod.report_legacy_state_dir(stages_dir.parent)
                committed = legacy
    return _blob_hashes(
        hash_info,
        lambda record: listing.read(committed, record, files_dir),
        files_dir,
        require_members=require_members,
    )


def _tracked_blobs(
    data_path: pathlib.Path,
    pvt_data: track.PvtData,
    files_dir: pathlib.Path,
    *,
    require_members: bool = False,
) -> set[str]:
    """The cache objects of a tracked file or directory."""
    return _blob_hashes(
        track.pvt_to_hash_info(pvt_data),
        lambda _record: track.tracked_members(data_path, pvt_data, files_dir),
        files_dir,
        require_members=require_members,
    )


def _get_file_hash_from_stages(
    target_identity_key: str,
    state_dir: pathlib.Path,
    files_dir: pathlib.Path,
    all_stages: dict[str, RegistryStageInfo] | None = None,
    *,
    require_members: bool = False,
) -> set[str] | None:
    """The cache objects of a stage output named by its identity key, from the lock files."""
    if all_stages is not None:
        for stage_name, stage_info in all_stages.items():
            stage_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
            stages_dir = lock.get_stages_dir(stage_state_dir)
            lock_data = lock.StageLock(stage_name, stages_dir).read(cache_dir=files_dir)
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
                    return _output_blobs(
                        out_hash,
                        stages_dir,
                        out_path,
                        files_dir,
                        lock_data,
                        require_members=require_members,
                    )
        return None

    from pivot.cli import helpers as cli_helpers

    stages_dir = lock.get_stages_dir(state_dir)
    stage_names = set[str]()
    for scanned_dir in (stages_dir, state_dir_mod.other_stages_dir_for(stages_dir)):
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
        lock_data = stage_lock.read(cache_dir=files_dir)
        if lock_data is None:
            continue

        for out_path, out_hash in lock_data["output_hashes"].items():
            identity_key = types.identity_key(out_path)
            if identity_key == target_identity_key and identity_key not in non_cached_paths:
                return _output_blobs(
                    out_hash,
                    stages_dir,
                    out_path,
                    files_dir,
                    lock_data,
                    require_members=require_members,
                )

    return None


def _get_file_hash_from_pvt(
    rel_path: str,
    proj_root: pathlib.Path,
    files_dir: pathlib.Path,
    *,
    skip: track.PvtSkip | None = None,
    require_members: bool = False,
) -> set[str] | None:
    """The cache objects of a path tracked by a .pvt file; None when ``skip`` leaves that .pvt
    out (a stage's own member of its output, which tracks nothing)."""
    pvt_path = proj_root / (rel_path + ".pvt")
    if not pvt_path.exists():
        return None

    track_data = track.read_pvt_file(pvt_path)
    if track_data is None:
        return None
    if skip is not None and skip(pvt_path, track_data):
        return None

    return _tracked_blobs(
        proj_root / rel_path, track_data, files_dir, require_members=require_members
    )


def _get_tracked_blobs(
    proj_root: pathlib.Path,
    files_dir: pathlib.Path,
    *,
    skip: track.PvtSkip | None = None,
    require_members: bool = False,
) -> dict[str, set[str]]:
    """The cache objects of every `pivot track`-ed file or directory, by its recorded hash.

    Imports are left out: `pivot import` writes the file straight to the workspace, so its
    blob never enters the local cache and push never sends it to this project's remote. So is
    a stage's own ``*.pvt`` member of its output (``skip``): it records no tracked file, so no
    dependency may resolve through it.
    """
    blobs = dict[str, set[str]]()
    for data_path, pvt_data in track.discover_pvt_files(proj_root, skip=skip).items():
        if track.is_import(pvt_data):
            continue
        blobs[pvt_data["hash"]] = _tracked_blobs(
            pathlib.Path(data_path), pvt_data, files_dir, require_members=require_members
        )
    return blobs


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
    cache_dir: pathlib.Path | None = None,
    *,
    require_members: bool = False,
) -> set[str]:
    """Resolve targets (stage names or file paths) to cache hashes.

    A directory resolves to its listing object and its files' objects. ``cache_dir`` is the
    cache root (``config.locate_cache_dir()`` when None). With ``require_members`` (push), a
    directory whose committed listing cannot be read and whose listing object the cache lacks
    raises rather than naming an object that is not there (``_blob_hashes``). A ``*.pvt`` file a
    stage wrote as its own member resolves nothing (``member_skip.own_member_skip``: the stages
    of ``all_stages`` by the registry, any other by its lock in ``state_dir``).
    """
    _t = metrics.start()
    proj_root = project.get_project_root()
    files_dir = (cache_dir if cache_dir is not None else config.locate_cache_dir()) / "files"
    hashes = set[str]()
    output_identities: dict[str, types.ArtifactIdentity] | None = None
    unresolved = list[str]()
    tracked_blobs: dict[str, set[str]] | None = None
    skip = member_skip.own_member_skip(proj_root, all_stages or {}, state_dir, files_dir)

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
                    lock_data = stage_lock.read(cache_dir=files_dir)
                    if lock_data is not None:
                        for out_path, out_hash in lock_data["output_hashes"].items():
                            if types.identity_key(out_path) not in non_cached_paths:
                                hashes |= _output_blobs(
                                    out_hash,
                                    stage_lock.stages_dir,
                                    out_path,
                                    files_dir,
                                    lock_data,
                                    require_members=require_members,
                                )
                        if include_deps:
                            # Only deps whose blobs pivot caches (and so pushes) can be fetched:
                            # upstream stage outputs and `pivot track` inputs. Raw inputs
                            # committed to git and `pivot import` inputs are never in the remote.
                            # A producer's listing is read beside its own lock (_output_blobs).
                            producer_locks = dict[str, LockData | None]()
                            for dep_identity, dep_hash in lock_data["dep_hashes"].items():
                                producer_info = all_stages.get(dep_identity.producer)
                                if producer_info is not None:
                                    if _is_cached_stage_output(dep_identity, producer_info):
                                        producer = dep_identity.producer
                                        producer_stages_dir = lock.get_stages_dir(
                                            registry.get_stage_state_dir(producer_info, state_dir)
                                        )
                                        if (
                                            types.is_dir_hash(dep_hash)
                                            and producer not in producer_locks
                                        ):
                                            producer_locks[producer] = lock.StageLock(
                                                producer, producer_stages_dir
                                            ).read(cache_dir=files_dir)
                                        hashes |= _output_blobs(
                                            dep_hash,
                                            producer_stages_dir,
                                            dep_identity,
                                            files_dir,
                                            producer_locks.get(producer),
                                            require_members=require_members,
                                        )
                                    continue
                                if tracked_blobs is None:
                                    tracked_blobs = _get_tracked_blobs(
                                        proj_root,
                                        files_dir,
                                        skip=skip,
                                        require_members=require_members,
                                    )
                                hashes |= tracked_blobs.get(dep_hash["hash"], set[str]())
                        continue
            else:
                resolved_identity = types.ArtifactIdentity(stage_name, target_identity.key)
                out_blobs = _get_file_hash_from_stages(
                    types.identity_key(resolved_identity),
                    state_dir,
                    files_dir,
                    all_stages,
                    require_members=require_members,
                )
                if out_blobs is not None:
                    hashes |= out_blobs
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

        out_blobs = _get_file_hash_from_stages(
            types.identity_key(target_identity),
            state_dir,
            files_dir,
            all_stages,
            require_members=require_members,
        )
        if out_blobs is not None:
            hashes |= out_blobs
            continue

        pvt_blobs = _get_file_hash_from_pvt(
            rel_path, proj_root, files_dir, skip=skip, require_members=require_members
        )
        if pvt_blobs is not None:
            hashes |= pvt_blobs
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
        try:
            local_hashes = get_target_hashes(
                targets,
                state_dir,
                include_deps=False,
                all_stages=all_stages,
                cache_dir=cache_dir,
                require_members=True,
            )
        except (exceptions.CorruptListingError, exceptions.ListingMissingError) as e:
            # A directory this push was asked to transfer has no readable listing and no cache
            # object for it, so pivot holds none of its members: fail it, naming the file, as
            # `pivot verify` does, rather than count it skipped and read as success.
            metrics.end("sync.push_async", _t)
            return TransferSummary(
                transferred=0, skipped=0, failed=1, errors=[f"{e}. {e.get_suggestion()}"]
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
        cache_path = cache.get_cache_path(files_dir, hash_)
        if not cache_path.is_file():
            other_files_dir = state_dir_mod.other_cache_files_dir_for(files_dir)
            if other_files_dir is not None:
                other_path = cache.get_cache_path(other_files_dir, hash_)
                if other_path.is_file():
                    state_dir_mod.report_legacy_state_dir(cache_dir)
                    cache_path = other_path
        if cache_path.is_file():
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
            targets, state_dir, include_deps=True, all_stages=all_stages, cache_dir=cache_dir
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
    downloaded = set(missing_locally)
    if targets and any(r["success"] for r in results):
        # A directory whose listing was neither committed nor cached named only its listing
        # object; with the listing now cached, its files can be named and fetched.
        named_now = get_target_hashes(
            targets, state_dir, include_deps=True, all_stages=all_stages, cache_dir=cache_dir
        )
        second = sorted(named_now - needed_hashes - get_local_cache_hashes(cache_dir))
        if second:
            needed_hashes |= named_now
            downloaded.update(second)
            results += await remote.download_batch(
                [(hash_, cache.get_cache_path(files_dir, hash_)) for hash_ in second],
                concurrency=jobs,
                callback=callback,
                readonly=True,
            )

    transferred = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    state_db.remote_hashes_add(remote_name, [r["hash"] for r in transferred])
    errors = [r["error"] for r in failed if "error" in r]

    metrics.end("sync.pull_async", _t)
    return TransferSummary(
        transferred=len(transferred),
        skipped=len(needed_hashes - downloaded),
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
