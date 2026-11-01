from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, TypedDict

from pivot import config, exceptions, expansion, metrics, names, project, registry, types
from pivot import state_dir as state_dir_mod
from pivot.remote import config as remote_config
from pivot.remote import protocol
from pivot.storage import cache, listing, lock, member_skip, track
from pivot.storage import store as store_mod
from pivot.types import HashInfo, LockData, RemoteStatus, TransferSummary, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pivot.registry import MapTemplate, RegistryStageInfo
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
    unlisted: dict[str, types.DirHash] | None = None,
) -> set[str]:
    """The cache objects that store an output: a file's object, or a directory's listing object
    and its files' objects.

    ``read_members`` reads a directory's listing (its committed listing file, else the cache's
    object). A listing neither the configured cache nor the legacy one holds is stored from the
    committed file, so push sends it. A directory whose listing neither holds, or whose committed
    listing cannot be used and the cache lacks, gives its listing object alone, and is added to
    ``unlisted`` by its hash: a pull fetches the listing first, then names the files.

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
        if unlisted is not None:
            unlisted[hash_info["hash"]] = hash_info
        return {hash_info["hash"]}
    except exceptions.CorruptListingError as e:
        if require_members:
            raise
        logger.warning("%s; only the cache's copy of that listing is transferred", e)
        if unlisted is not None:
            unlisted[hash_info["hash"]] = hash_info
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
    unlisted: dict[str, types.DirHash] | None = None,
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
        unlisted=unlisted,
    )


def _tracked_blobs(
    data_path: pathlib.Path,
    pvt_data: track.PvtData,
    files_dir: pathlib.Path,
    *,
    require_members: bool = False,
    unlisted: dict[str, types.DirHash] | None = None,
) -> set[str]:
    """The cache objects of a tracked file or directory."""
    return _blob_hashes(
        track.pvt_to_hash_info(pvt_data),
        lambda _record: track.tracked_members(data_path, pvt_data, files_dir),
        files_dir,
        require_members=require_members,
        unlisted=unlisted,
    )


def _get_file_hash_from_stages(
    target_identity_key: str,
    state_dir: pathlib.Path,
    files_dir: pathlib.Path,
    all_stages: dict[str, RegistryStageInfo] | None = None,
    *,
    require_members: bool = False,
    unlisted: dict[str, types.DirHash] | None = None,
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
                        unlisted=unlisted,
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
                    unlisted=unlisted,
                )

    return None


def _get_file_hash_from_pvt(
    rel_path: str,
    proj_root: pathlib.Path,
    files_dir: pathlib.Path,
    *,
    skip: track.PvtSkip | None,
    require_members: bool = False,
    unlisted: dict[str, types.DirHash] | None = None,
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
        proj_root / rel_path,
        track_data,
        files_dir,
        require_members=require_members,
        unlisted=unlisted,
    )


def _mapped_source_listing(
    stage_name: str,
    templates: Mapping[str, MapTemplate],
    all_stages: dict[str, RegistryStageInfo],
    state_dir: pathlib.Path,
    files_dir: pathlib.Path,
    *,
    require_members: bool,
) -> set[str]:
    """The listing object of the directory output that ``stage_name``, an instance of a mapped
    stage (``Pipeline.map``), is mapped over; empty for any other stage.

    A clone without the producer's committed listing fetches this object to expand the mapped
    stage before it can pull or check out the instance, so the instance's set holds it. It is
    stored from the committed listing when the cache lacks it, so push can send it.
    """
    template = registry.template_of(stage_name, templates)
    if template is None:
        return set()
    source = templates[template]["source"]
    if source.producer not in all_stages:
        return set()
    stages_dir = lock.get_stages_dir(
        registry.get_stage_state_dir(all_stages[source.producer], state_dir)
    )
    lock_data = lock.StageLock(source.producer, stages_dir).read(cache_dir=files_dir)
    record = expansion.recorded_directory(source, lock_data)
    if record is None:
        return set()
    _output_blobs(record, stages_dir, source, files_dir, lock_data, require_members=require_members)
    return {record["hash"]}


def get_target_hashes(
    targets: list[str],
    state_dir: pathlib.Path,
    all_stages: dict[str, RegistryStageInfo] | None = None,
    cache_dir: pathlib.Path | None = None,
    *,
    require_members: bool = False,
    templates: Mapping[str, MapTemplate] | None = None,
    unlisted: dict[str, types.DirHash] | None = None,
) -> set[str]:
    """The cache objects of ``targets`` (stage names or file paths): the objects that
    ``pivot pull`` of them and then ``pivot checkout`` and a rerun of them use in a fresh clone.
    ``pivot push``, ``fetch`` and ``pull`` of targets all transfer this set, so a push of a
    target sends every object a pull of it asks for.

    A stage names its cached outputs (metrics are git-tracked) and the objects its lock records
    under ``reads`` (``pivot.read_by_hash``), which a rerun reads; a stage output or tracked
    path names its own objects; a directory names its listing object and its files' objects. No
    target names its stage's dependencies, which are targets of their own; an untargeted pull
    fetches every object on the remote. An instance of a mapped stage of ``templates`` also
    names the listing object of the directory it is mapped over, which a clone fetches to
    expand it (``_mapped_source_listing``). A registered stage, or a declared output, with no
    recorded hash names nothing.

    ``cache_dir`` is the cache root (``config.locate_cache_dir()`` when None). With
    ``require_members`` (push), a directory whose committed listing cannot be read and whose
    listing object the cache lacks raises rather than naming an object that is not there
    (``_blob_hashes``); without it, such a directory names its listing object alone and is added
    to ``unlisted``. A ``*.pvt`` file a stage wrote as its own member resolves nothing
    (``member_skip.own_member_skip``: the stages of ``all_stages`` by the registry, any other by
    its lock in ``state_dir``).

    Raises:
        UnknownTargetError: Naming each target that is not a stage, stage output or tracked
            path, project-relative for a path.
    """
    _t = metrics.start()
    proj_root = project.get_project_root()
    files_dir = (cache_dir if cache_dir is not None else config.locate_cache_dir()) / "files"
    hashes = set[str]()
    output_identities: dict[str, types.ArtifactIdentity] | None = None
    unresolved = list[str]()
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
            stage_info = all_stages[stage_name]
            if templates is not None:
                hashes |= _mapped_source_listing(
                    stage_name,
                    templates,
                    all_stages,
                    state_dir,
                    files_dir,
                    require_members=require_members,
                )
            if target_identity.key is None:
                target_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
                non_cached_paths = {
                    types.identity_key(out.identity)
                    for out in stage_info["outs"]
                    if out.tag is types.ArtifactTag.METRIC
                }
                stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(target_state_dir))
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
                                unlisted=unlisted,
                            )
                    # The objects the stage read by hash are always cache objects.
                    if "reads" in lock_data:
                        hashes.update(lock_data["reads"])
                continue
            resolved_identity = types.ArtifactIdentity(stage_name, target_identity.key)
            out_blobs = _get_file_hash_from_stages(
                types.identity_key(resolved_identity),
                state_dir,
                files_dir,
                all_stages,
                require_members=require_members,
                unlisted=unlisted,
            )
            if out_blobs is not None:
                hashes |= out_blobs
            elif not any(out.identity == resolved_identity for out in stage_info["outs"]):
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
            unlisted=unlisted,
        )
        if out_blobs is not None:
            hashes |= out_blobs
            if templates is not None and all_stages is not None:
                hashes |= _mapped_source_listing(
                    target_identity.producer,
                    templates,
                    all_stages,
                    state_dir,
                    files_dir,
                    require_members=require_members,
                )
            continue

        pvt_blobs = _get_file_hash_from_pvt(
            rel_path,
            proj_root,
            files_dir,
            skip=skip,
            require_members=require_members,
            unlisted=unlisted,
        )
        if pvt_blobs is not None:
            hashes |= pvt_blobs
            continue

        if output_identities is None or rel_path not in output_identities:
            unresolved.append(rel_path)

    metrics.end("sync.get_target_hashes", _t)
    if unresolved:
        raise exceptions.UnknownTargetError(unresolved)
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
) -> RemoteStatus:
    """Compare the local cache with the remote, listing the remote's objects.

    ``local_only`` holds what an untargeted push uploads: the cached objects neither the remote
    index nor the listing holds. ``remote_only`` holds what an untargeted pull downloads: the
    remote's objects the cache lacks. Each cached object the listing shows on the remote is
    recorded in the index, as a push records it.
    """
    _t = metrics.start()
    indexed = state_db.remote_hashes_intersection(remote_name, local_hashes)
    remote_hashes = await remote.list_hashes()
    listed = local_hashes & remote_hashes
    state_db.remote_hashes_add(remote_name, listed - indexed)
    common = indexed | listed
    metrics.end("sync.compare_status", _t)
    return RemoteStatus(
        local_only=local_hashes - common,
        remote_only=remote_hashes - local_hashes,
        common=common,
    )


class _PushPlan(TypedDict):
    """What a push of a set of cache objects does."""

    upload: list[tuple[pathlib.Path, str]]  # (cache file, hash) of each object it uploads
    on_remote: set[str]  # the objects the remote holds already
    unavailable: list[str]  # the objects the remote lacks and the cache does not hold
    found: set[str]  # the objects on the remote that the index did not record


def _push_hashes(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    targets: list[str] | None,
    all_stages: dict[str, RegistryStageInfo] | None,
    templates: Mapping[str, MapTemplate] | None,
) -> set[str]:
    """The objects a push of ``targets`` sends when the remote lacks them: their objects
    (``get_target_hashes``), or every cached object when there are no targets.

    Raises:
        UnknownTargetError: A target is not a stage, stage output or tracked path.
        CorruptListingError, ListingMissingError: A directory to push has no readable listing and
            no cache object for it, so the cache holds none of its files to send.
    """
    if targets:
        return get_target_hashes(
            targets,
            state_dir,
            all_stages=all_stages,
            cache_dir=cache_dir,
            require_members=True,
            templates=templates,
        )
    return get_local_cache_hashes(cache_dir)


async def _plan_push(
    cache_dir: pathlib.Path,
    hashes: set[str],
    remote: protocol.RemoteStorage,
    indexed: set[str],
    jobs: int,
) -> _PushPlan:
    """What pushing ``hashes`` does: the remote holds the ``indexed`` ones and those it reports
    holding; each other one is uploaded from the cache, or is unavailable when the configured
    and legacy caches hold no regular file for it."""
    unknown = hashes - indexed
    found = set[str]()
    if unknown:
        existence = await remote.bulk_exists(sorted(unknown), concurrency=jobs)
        found = {hash_ for hash_, exists in existence.items() if exists}
    on_remote = (hashes & indexed) | found

    files_dir = cache_dir / "files"
    other_files_dir = state_dir_mod.other_cache_files_dir_for(files_dir)
    upload = list[tuple[pathlib.Path, str]]()
    unavailable = list[str]()
    for hash_ in sorted(hashes - on_remote):
        cache_path = cache.get_cache_path(files_dir, hash_)
        if not cache_path.is_file() and other_files_dir is not None:
            other_path = cache.get_cache_path(other_files_dir, hash_)
            if other_path.is_file():
                state_dir_mod.report_legacy_state_dir(cache_dir.parent)
                cache_path = other_path
        if cache_path.is_file():
            upload.append((cache_path, hash_))
        else:
            unavailable.append(hash_)
    return _PushPlan(upload=upload, on_remote=on_remote, unavailable=unavailable, found=found)


def _unavailable_errors(remote_name: str, unavailable: list[str]) -> list[str]:
    return [
        f"Cannot push {hash_}: remote '{remote_name}' lacks it and the local cache does not hold it"
        for hash_ in unavailable
    ]


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
    *,
    templates: Mapping[str, MapTemplate] | None = None,
) -> TransferSummary:
    """Push cache files to remote (async implementation)."""
    _t = metrics.start()
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    hashes = _push_hashes(cache_dir, state_dir, targets, all_stages, templates)
    if not hashes:
        metrics.end("sync.push_async", _t)
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])

    check_remote_url(state_db, remote_name, remote)
    plan = await _plan_push(
        cache_dir, hashes, remote, state_db.remote_hashes_intersection(remote_name, hashes), jobs
    )
    state_db.remote_hashes_add(remote_name, plan["found"])
    results = (
        await remote.upload_batch(plan["upload"], concurrency=jobs, callback=callback)
        if plan["upload"]
        else []
    )

    transferred = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    state_db.remote_hashes_add(remote_name, [r["hash"] for r in transferred])
    errors = _unavailable_errors(remote_name, plan["unavailable"])
    errors += [r["error"] for r in failed if "error" in r]

    metrics.end("sync.push_async", _t)
    return TransferSummary(
        transferred=len(transferred),
        skipped=len(plan["on_remote"]),
        failed=len(failed) + len(plan["unavailable"]),
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
    *,
    templates: Mapping[str, MapTemplate] | None = None,
) -> TransferSummary:
    """Push the objects of ``targets``, or every cached object, that the remote lacks. An
    object the remote lacks and the cache does not hold fails, naming it. ``templates`` are
    the pipeline's mapped stages (``get_target_hashes``)."""
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
            templates=templates,
        )
    )


def push_dry_run(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB | None,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    all_stages: dict[str, RegistryStageInfo] | None = None,
    *,
    templates: Mapping[str, MapTemplate] | None = None,
) -> TransferSummary:
    """What ``push`` would do, uploading and recording nothing: ``transferred`` counts the
    objects it would upload, ``failed`` and ``errors`` the objects it could not send.

    ``state_db`` is read only, None when the project has none. Its index answers for the
    remote as it does for ``push``: unless it records another URL for this remote, which
    ``push`` would clear first.
    """
    hashes = _push_hashes(cache_dir, state_dir, targets, all_stages, templates)
    if not hashes:
        return TransferSummary(transferred=0, skipped=0, failed=0, errors=[])
    indexed = set[str]()
    if state_db is not None and state_db.remote_get_url(remote_name) in (None, remote.url):
        indexed = state_db.remote_hashes_intersection(remote_name, hashes)
    plan = asyncio.run(
        _plan_push(
            cache_dir,
            hashes,
            remote,
            indexed,
            jobs if jobs is not None else config.get_remote_jobs(),
        )
    )
    return TransferSummary(
        transferred=len(plan["upload"]),
        skipped=len(plan["on_remote"]),
        failed=len(plan["unavailable"]),
        errors=_unavailable_errors(remote_name, plan["unavailable"]),
    )


async def _pull_hashes(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    targets: list[str] | None,
    all_stages: dict[str, RegistryStageInfo] | None,
    templates: Mapping[str, MapTemplate] | None,
    unlisted: dict[str, types.DirHash] | None = None,
) -> set[str]:
    """The objects a pull of ``targets`` fetches when the cache lacks them: their objects
    (``get_target_hashes``, which adds each directory whose listing is not local to
    ``unlisted``), or every object on the remote when there are no targets.

    Raises:
        UnknownTargetError: A target is not a stage, stage output or tracked path.
    """
    if targets:
        return get_target_hashes(
            targets,
            state_dir,
            all_stages=all_stages,
            cache_dir=cache_dir,
            templates=templates,
            unlisted=unlisted,
        )
    return await remote.list_hashes()


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
    *,
    templates: Mapping[str, MapTemplate] | None = None,
) -> TransferSummary:
    """Pull cache files from remote (async implementation)."""
    _t = metrics.start()
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    check_remote_url(state_db, remote_name, remote)
    needed_hashes = await _pull_hashes(cache_dir, state_dir, remote, targets, all_stages, templates)

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
            targets, state_dir, all_stages=all_stages, cache_dir=cache_dir, templates=templates
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
    *,
    templates: Mapping[str, MapTemplate] | None = None,
) -> TransferSummary:
    """Pull cache files from remote storage. ``templates`` are the pipeline's mapped stages
    (``get_target_hashes``)."""
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
            templates=templates,
        )
    )


async def _held_by(remote: protocol.RemoteStorage, hashes: set[str], jobs: int) -> set[str]:
    """The objects of ``hashes`` the remote holds."""
    if not hashes:
        return set()
    existence = await remote.bulk_exists(sorted(hashes), concurrency=jobs)
    return {hash_ for hash_, exists in existence.items() if exists}


async def _listed_files(
    remote: protocol.RemoteStorage, records: dict[str, types.DirHash], jobs: int
) -> set[str]:
    """The file objects the directory listings at ``records`` name, read from the remote into a
    temporary directory and dropped: the files a pull fetches once it has those listings. A
    listing that cannot be downloaded or read names none."""
    if not records:
        return set()
    with tempfile.TemporaryDirectory() as scratch:
        files_dir = pathlib.Path(scratch)
        _ = await remote.download_batch(
            [(hash_, cache.get_cache_path(files_dir, hash_)) for hash_ in sorted(records)],
            concurrency=jobs,
        )
        files = set[str]()
        for record in records.values():
            try:
                members = listing.read(None, record, files_dir)
            except (exceptions.ListingMissingError, exceptions.CorruptListingError):
                continue
            files.update(entry["hash"] for entry in members["files"])
        return files


async def _pull_dry_run_async(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    remote_name: str,
    targets: list[str] | None,
    jobs: int,
    all_stages: dict[str, RegistryStageInfo] | None,
    templates: Mapping[str, MapTemplate] | None,
) -> TransferSummary:
    unlisted = dict[str, types.DirHash]()
    needed = await _pull_hashes(
        cache_dir, state_dir, remote, targets, all_stages, templates, unlisted
    )
    local = get_local_cache_hashes(cache_dir)
    missing = needed - local
    on_remote = await _held_by(remote, missing, jobs) if targets else missing
    # A directory whose listing is neither committed nor cached names its listing object alone;
    # the pull fetches that listing, then the files it names, so they are counted too.
    files = await _listed_files(
        remote, {hash_: record for hash_, record in unlisted.items() if hash_ in on_remote}, jobs
    )
    extra = files - needed - local
    needed |= extra
    missing |= extra
    on_remote |= await _held_by(remote, extra, jobs)
    absent = sorted(missing - on_remote)
    return TransferSummary(
        transferred=len(on_remote),
        skipped=len(needed - missing),
        failed=len(absent),
        errors=[f"Cannot fetch {hash_}: remote '{remote_name}' lacks it" for hash_ in absent],
    )


def pull_dry_run(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    remote_name: str,
    targets: list[str] | None = None,
    jobs: int | None = None,
    all_stages: dict[str, RegistryStageInfo] | None = None,
    *,
    templates: Mapping[str, MapTemplate] | None = None,
) -> TransferSummary:
    """What ``pull`` would download, downloading nothing into the cache and recording nothing:
    ``transferred`` counts the objects the cache lacks and the remote holds, the files of a
    listing only the remote holds included (read into a temporary directory and dropped), and
    ``failed`` and ``errors`` the objects the remote lacks too."""
    return asyncio.run(
        _pull_dry_run_async(
            cache_dir,
            state_dir,
            remote,
            remote_name,
            targets,
            jobs if jobs is not None else config.get_remote_jobs(),
            all_stages,
            templates,
        )
    )


async def _fetch_objects_async(
    cache_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB | None,
    remote_name: str,
    hashes: set[str],
    jobs: int | None,
) -> TransferSummary:
    if state_db is not None:
        check_remote_url(state_db, remote_name, remote)
    files_dir = cache_dir / "files"
    missing = sorted(hashes - get_local_cache_hashes(cache_dir))
    if not missing:
        return TransferSummary(transferred=0, skipped=len(hashes), failed=0, errors=[])
    results = await remote.download_batch(
        [(hash_, cache.get_cache_path(files_dir, hash_)) for hash_ in missing],
        concurrency=jobs if jobs is not None else config.get_remote_jobs(),
        readonly=True,
    )
    transferred = [r for r in results if r["success"]]
    if state_db is not None:
        state_db.remote_hashes_add(remote_name, [r["hash"] for r in transferred])
    return TransferSummary(
        transferred=len(transferred),
        skipped=len(hashes) - len(missing),
        failed=len(results) - len(transferred),
        errors=[r["error"] for r in results if not r["success"] and "error" in r],
    )


def fetch_objects(
    cache_dir: pathlib.Path,
    remote: protocol.RemoteStorage,
    state_db: state_mod.StateDB | None,
    remote_name: str,
    hashes: set[str],
    jobs: int | None = None,
) -> TransferSummary:
    """Download the cache objects at ``hashes`` the cache lacks, and nothing else: a
    directory's listing, so a pull can name its files (and expand a mapped stage over them)
    before it fetches any. With no ``state_db`` (a dry run) nothing is recorded in the remote
    index."""
    return asyncio.run(_fetch_objects_async(cache_dir, remote, state_db, remote_name, hashes, jobs))


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
