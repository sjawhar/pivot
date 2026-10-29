"""Detailed explanations for stage change detection.

Compares current state against lock files to explain WHY stages would run,
showing specific code, param, and dependency changes.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pydantic

from pivot import exceptions, parameters, project, registry, skip, types
from pivot.executor import worker
from pivot.storage import listing, lock, state, track
from pivot.storage import store as store_mod
from pivot.types import (
    ArtifactIdentity,
    FileHashEntry,
    HashInfo,
    LockData,
    StageExplanation,
    identity_from_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    import pygtrie

OUTPUTS_MISSING_REASON = "Outputs missing from cache"
"""Reason for a stage whose lock is current but whose cacheable outputs are absent locally.

``pivot status`` counts stages with this reason as restorable by ``pivot pull``, so it is
never used when the only unavailable outputs are metrics, which are not cached.
"""


def _unavailable_outputs_reason(
    store: store_mod.WorkspaceStore, unavailable: list[types.ArtifactRef]
) -> str:
    """Say why a stage whose lock is current must still run to regenerate ``unavailable``."""
    if any(ref.tag is not types.ArtifactTag.METRIC for ref in unavailable):
        return OUTPUTS_MISSING_REASON
    changed = list[str]()
    missing = list[str]()
    for ref in unavailable:
        key = types.identity_key(ref.identity)
        (changed if store.resolve_display_path(ref).exists() else missing).append(key)
    reasons = list[str]()
    if changed:
        reasons.append(f"Metric output changed: {', '.join(changed)}")
    if missing:
        reasons.append(f"Metric output missing: {', '.join(missing)}")
    return "; ".join(reasons)


def read_locks(
    stage_names: Iterable[str],
    all_stages: dict[str, registry.RegistryStageInfo],
    default_state_dir: pathlib.Path,
    cache_dir: pathlib.Path | None,
) -> dict[str, LockData | None]:
    """Read each named stage's lock once; None for a stage that has not run. ``cache_dir``
    receives the listings a lock an older pivot wrote holds (``StageLock.read``)."""
    locks = dict[str, LockData | None]()
    for stage_name in stage_names:
        locks[stage_name] = lock.StageLock(
            stage_name, stages_dir(all_stages[stage_name], default_state_dir)
        ).read(cache_dir=cache_dir)
    return locks


def stages_dir(
    stage_info: registry.RegistryStageInfo, default_state_dir: pathlib.Path
) -> pathlib.Path:
    """The directory that holds a stage's lock and its outputs' listings."""
    return lock.get_stages_dir(registry.get_stage_state_dir(stage_info, default_state_dir))


def producer_output_hashes(
    locks: Mapping[str, LockData | None],
    deps: Iterable[types.ArtifactRef],
    stages_dirs: Mapping[str, pathlib.Path],
    cache_dir: pathlib.Path | None,
    unreadable: dict[ArtifactIdentity, str] | None = None,
) -> dict[ArtifactIdentity, HashInfo]:
    """Merge the locked output hashes of already-read producer locks (by stage name).

    A dependency in ``deps`` on one file of a directory output finds that file's hash in the
    directory's listing: its producer's committed listing file (in ``stages_dirs``), else the
    cache's object. A listing is read only for such a dependency, once per directory, so a
    fresh clone needs no pull. A directory whose listing neither holds, or whose committed
    listing cannot be used (``listing.read``), leaves its files to be hashed on disk; for the
    latter, ``unreadable`` receives why, naming the file, by each file dependency on it: it is
    the consumer's problem, not every stage's.
    """
    hashes = dict[ArtifactIdentity, HashInfo]()
    for lock_data in locks.values():
        if lock_data is not None:
            hashes.update(lock_data["output_hashes"])
    wanted = dict[ArtifactIdentity, set[ArtifactIdentity]]()
    for ref in deps:
        if ref.directory is not None:
            wanted.setdefault(ref.directory, set[ArtifactIdentity]()).add(ref.identity)
    for directory, files in wanted.items():
        record = hashes.get(directory)
        if record is None or not types.is_dir_hash(record) or directory.producer not in stages_dirs:
            continue
        committed = listing.output_listing_path(stages_dirs[directory.producer], directory)
        producer_lock = locks.get(directory.producer)
        if producer_lock is not None:
            committed = lock.committed_listing(producer_lock, committed)
        try:
            members = listing.read(committed, record, cache_dir)
        except exceptions.ListingMissingError:
            continue
        except exceptions.CorruptListingError as e:
            if unreadable is not None:
                unreadable.update(dict.fromkeys(files, str(e)))
            continue
        for entry in members["files"]:
            file_identity = types.directory_file_identity(directory, entry["relpath"])
            if file_identity in files:
                hashes[file_identity] = types.FileHash(hash=entry["hash"])
    return hashes


def _to_identity_keyed(str_hashes: dict[str, HashInfo]) -> dict[ArtifactIdentity, HashInfo]:
    return {identity_from_key(k): v for k, v in str_hashes.items()}


def _find_tracked_ancestor(dep: Path, tracked_trie: pygtrie.Trie[str]) -> Path | None:
    """Find the tracked path that contains dep (exact match or ancestor)."""
    dep_key = dep.parts

    # Exact match
    if dep_key in tracked_trie:
        return pathlib.Path(tracked_trie[dep_key])

    # Dependency is inside a tracked directory
    prefix_item = tracked_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        return pathlib.Path(prefix_item.value)

    return None


def _find_tracked_hash(
    dep: Path,
    tracked_files: dict[str, track.PvtData],
    tracked_trie: pygtrie.Trie[str],
    cache_dir: pathlib.Path | None,
) -> HashInfo | None:
    """Find hash for dep from tracked files data.

    Returns HashInfo if dep is tracked (exact match or inside tracked directory, found in its
    listing), None otherwise.
    """
    tracked_path = _find_tracked_ancestor(dep, tracked_trie)
    if not tracked_path:
        return None

    pvt_data = tracked_files[str(tracked_path)]

    # Exact match - use top-level hash
    if dep == tracked_path:
        return track.pvt_to_hash_info(pvt_data)

    # Nested path - find in the directory's listing
    if "nfiles" not in pvt_data:
        return None  # Single file .pvt can't contain nested paths
    try:
        members = track.tracked_members(tracked_path, pvt_data, cache_dir)
    except exceptions.ListingMissingError:
        return None

    relpath = str(dep.relative_to(tracked_path))
    for entry in members["files"]:
        if entry["relpath"] == relpath:
            return types.FileHash(hash=entry["hash"])

    return None  # Path not found in the listing


def get_stage_explanation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: dict[str, types.ArtifactRef],
    outs: list[types.ArtifactRef],
    params_instance: pydantic.BaseModel | None,
    overrides: parameters.ParamsOverrides | None,
    state_dir: pathlib.Path,
    store: store_mod.WorkspaceStore,
    *,
    lock_data: LockData | None,
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, track.PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
    producer_hashes: dict[ArtifactIdentity, HashInfo] | None = None,
    file_hash_entries: list[FileHashEntry] | None = None,
    unreadable_listings: Mapping[ArtifactIdentity, str] | None = None,
) -> StageExplanation:
    """Compute detailed explanation of why a stage would run.

    ``lock_data`` is the stage's lock, already read by the caller (None when the stage has
    not run). A stage is stale when repro will rewrite its lock by running it or restoring
    from the run cache; it is cached when its lock already matches and repro at most
    restores outputs. ``unreadable_listings`` says, by file dependency, why the listing that
    would give its hash cannot be used (``producer_output_hashes``); a missing dependency's
    reason names it.

    The explanation reads the state database and writes nothing. ``file_hash_entries``, when
    given, receives the file-hash entry of every file whose content it read, for the caller to
    save with ``state.save_file_hashes``.
    """
    if lock_data is None:
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason="forced" if force else "No previous run",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )
    if not force:
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        restarted = lock.restart_proofs(stage_lock, lock_data, outs)
        if restarted:
            return StageExplanation(
                stage_name=stage_name,
                will_run=True,
                is_forced=False,
                reason=f"Refused: {lock.restarted_reason(stage_lock, restarted)}",
                code_changes=[],
                param_changes=[],
                dep_changes=[],
                upstream_stale=[],
            )

    try:
        current_params = parameters.get_effective_params(params_instance, stage_name, overrides)
    except pydantic.ValidationError as error:
        assert params_instance is not None, "only params validation raises here"
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason="Invalid params.yaml:\n"
            + parameters.describe_validation_error(type(params_instance), error),
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    with state.open_readonly(state_dir, create=False) as state_db:
        known_hashes = producer_hashes or {}
        pending_deps = {name: ref for name, ref in deps.items() if ref.identity not in known_hashes}
        str_hashes, missing_deps, unreadable_deps, dep_entries = worker.hash_dependencies(
            pending_deps,
            store,
            state_db,
        )
        if file_hash_entries is not None:
            file_hash_entries.extend(dep_entries)
        dep_hashes = _to_identity_keyed(str_hashes)
        for ref in deps.values():
            if ref.identity in known_hashes:
                dep_hashes[ref.identity] = known_hashes[ref.identity]

        if allow_missing:
            unresolved = list[str]()
            for dep_key in missing_deps:
                ref = next(
                    ref for ref in deps.values() if types.identity_key(ref.identity) == dep_key
                )
                tracked_hash = None
                if tracked_files is not None and tracked_trie is not None:
                    tracked_hash = _find_tracked_hash(
                        project.normalize_path(store.checkout(ref)),
                        tracked_files,
                        tracked_trie,
                        store.cache_dir,
                    )
                fallback_hash = tracked_hash or lock_data["dep_hashes"].get(ref.identity)
                if fallback_hash is None:
                    unresolved.append(dep_key)
                else:
                    dep_hashes[ref.identity] = fallback_hash
            missing_deps = unresolved

        if missing_deps or unreadable_deps:
            code_changes = skip.diff_code_manifests(lock_data["code_manifest"], fingerprint)
            param_changes = skip.diff_params(lock_data["params"], current_params)
            reasons = list[str]()
            if code_changes:
                reasons.append("Code changed")
            if param_changes:
                reasons.append("Params changed")
            if missing_deps:
                unreadable_by_key = {
                    types.identity_key(identity): problem
                    for identity, problem in (unreadable_listings or {}).items()
                }
                shown = [
                    f"{dep_key} ({unreadable_by_key[dep_key]})"
                    if dep_key in unreadable_by_key
                    else dep_key
                    for dep_key in missing_deps
                ]
                reasons.append(f"Missing deps: {', '.join(shown)}")
            if unreadable_deps:
                reasons.append(f"Unreadable deps: {', '.join(unreadable_deps)}")
            return StageExplanation(
                stage_name=stage_name,
                will_run=True,
                is_forced=force,
                reason="; ".join(reasons),
                code_changes=code_changes,
                param_changes=param_changes,
                dep_changes=[],
                upstream_stale=[],
            )

        decision = skip.check_stage(
            lock_data=lock_data,
            fingerprint=fingerprint,
            params=current_params,
            dep_hashes=dep_hashes,
            out_paths=[out.identity for out in outs],
            explain=True,
            force=force,
        )
        unavailable = list[types.ArtifactRef]()
        empty_dirs_unrecorded = list[types.ArtifactRef]()
        if not allow_missing:
            try:
                restoration = worker.restore_outputs_to_hashes(
                    store,
                    outs,
                    lock_data["output_hashes"],
                    state_db,
                    state_dir=state_dir,
                    listing_lock=None
                    if "older_format" in lock_data
                    else lock.StageLock(stage_name, lock.get_stages_dir(state_dir)),
                    restore=False,
                    report_changes=True,
                    file_hash_entries=file_hash_entries,
                )
            except exceptions.UnreadableLinkError as error:
                # An output holds a link the check cannot read: its message, which names the
                # link, is the stage's reason and the one report of it. Raised, it would also be
                # logged as a warning by the caller (status, verify, repro --dry-run).
                return StageExplanation(
                    stage_name=stage_name,
                    will_run=True,
                    is_forced=force,
                    reason=str(error),
                    code_changes=[],
                    param_changes=[],
                    dep_changes=[],
                    upstream_stale=[],
                )
            unavailable = restoration.unavailable
            empty_dirs_unrecorded = restoration.empty_dirs_unrecorded
        if not decision["changed"] and (unavailable or empty_dirs_unrecorded):
            return StageExplanation(
                stage_name=stage_name,
                will_run=True,
                is_forced=force,
                reason=(
                    _unavailable_outputs_reason(store, unavailable)
                    if unavailable
                    else worker.empty_dirs_unrecorded_reason(empty_dirs_unrecorded)
                ),
                code_changes=[],
                param_changes=[],
                dep_changes=[],
                upstream_stale=[],
            )
        return StageExplanation(
            stage_name=stage_name,
            will_run=decision["changed"],
            is_forced=force,
            reason=decision["reason"],
            code_changes=decision.get("code_changes", []),
            param_changes=decision.get("param_changes", []),
            dep_changes=decision.get("dep_changes", []),
            upstream_stale=[],
        )
