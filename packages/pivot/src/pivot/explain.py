"""Detailed explanations for stage change detection.

Compares current state against lock files to explain WHY stages would run,
showing specific code, param, and dependency changes.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pydantic

from pivot import parameters, project, registry, skip, types
from pivot.executor import worker
from pivot.storage import lock, state, track
from pivot.storage import store as store_mod
from pivot.types import (
    ArtifactIdentity,
    HashInfo,
    LockData,
    StageExplanation,
    identity_from_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
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
) -> dict[str, LockData | None]:
    """Read each named stage's lock once; None for a stage that has not run."""
    locks = dict[str, LockData | None]()
    for stage_name in stage_names:
        state_dir = registry.get_stage_state_dir(all_stages[stage_name], default_state_dir)
        locks[stage_name] = lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).read()
    return locks


def producer_output_hashes(
    locks: Iterable[LockData | None],
) -> dict[ArtifactIdentity, HashInfo]:
    """Merge the locked output hashes of already-read producer locks.

    Each file of a directory output is also listed under its own identity (the directory's
    key with the file's path appended), so a stage that reads one file finds its hash.
    """
    hashes = dict[ArtifactIdentity, HashInfo]()
    for lock_data in locks:
        if lock_data is None:
            continue
        for identity, hash_info in lock_data["output_hashes"].items():
            hashes[identity] = hash_info
            if not types.is_dir_hash(hash_info):
                continue
            for entry in hash_info["manifest"]:
                file_identity = types.directory_file_identity(identity, entry["relpath"])
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
) -> HashInfo | None:
    """Find hash for dep from tracked files data.

    Returns HashInfo if dep is tracked (exact match or inside tracked directory),
    None otherwise.
    """
    tracked_path = _find_tracked_ancestor(dep, tracked_trie)
    if not tracked_path:
        return None

    pvt_data = tracked_files[str(tracked_path)]

    # Exact match - use top-level hash
    if dep == tracked_path:
        return track.pvt_to_hash_info(pvt_data)

    # Nested path - find in manifest
    if "manifest" not in pvt_data:
        return None  # Single file .pvt can't contain nested paths

    relpath = str(dep.relative_to(tracked_path))
    for entry in pvt_data["manifest"]:
        if entry["relpath"] == relpath:
            return {"hash": entry["hash"]}

    return None  # Path not found in manifest


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
) -> StageExplanation:
    """Compute detailed explanation of why a stage would run.

    ``lock_data`` is the stage's lock, already read by the caller (None when the stage has
    not run). A stage is stale when repro will rewrite its lock by running it or restoring
    from the run cache; it is cached when its lock already matches and repro at most
    restores outputs.
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
        str_hashes, missing_deps, unreadable_deps, _ = worker.hash_dependencies(
            pending_deps,
            store,
            state_db,
        )
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
                reasons.append(f"Missing deps: {', '.join(missing_deps)}")
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
            restoration = worker.restore_outputs_to_hashes(
                store,
                outs,
                lock_data["output_hashes"],
                state_db,
                state_dir=state_dir,
                restore=False,
                report_changes=True,
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
