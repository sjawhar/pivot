"""Detailed explanations for stage change detection.

Compares current state against lock files to explain WHY stages would run,
showing specific code, param, and dependency changes.
"""

from __future__ import annotations

import contextlib
import pathlib
from typing import TYPE_CHECKING

import pydantic

from pivot import parameters, project, registry, skip, types
from pivot.executor import worker
from pivot.storage import lock, state
from pivot.storage import store as store_mod
from pivot.types import (
    ArtifactIdentity,
    HashInfo,
    StageExplanation,
    identity_from_key,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pygtrie

    from pivot.storage.track import PvtData

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


def producer_output_hashes(
    stage_names: list[str],
    all_stages: dict[str, registry.RegistryStageInfo],
    default_state_dir: pathlib.Path,
) -> dict[ArtifactIdentity, HashInfo]:
    """Read locked output hashes for producer stages in an execution order."""
    hashes = dict[ArtifactIdentity, HashInfo]()
    for stage_name in stage_names:
        stage_info = all_stages[stage_name]
        state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data is not None:
            hashes.update(lock_data["output_hashes"])
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
    tracked_files: dict[str, PvtData],
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
        if "manifest" in pvt_data:
            return {"hash": pvt_data["hash"], "manifest": pvt_data["manifest"]}
        return {"hash": pvt_data["hash"]}

    # Nested path - find in manifest
    if "manifest" not in pvt_data:
        return None  # Single file .pvt can't contain nested paths

    relpath = str(dep.relative_to(tracked_path))
    for entry in pvt_data["manifest"]:
        if entry["relpath"] == relpath:
            return {"hash": entry["hash"]}

    return None  # Path not found in manifest


def _state_db_exists(state_dir: pathlib.Path) -> bool:
    """Return True when the LMDB directory already exists."""
    return (state_dir / "state.lmdb").exists()


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
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
    producer_hashes: dict[ArtifactIdentity, HashInfo] | None = None,
) -> StageExplanation:
    """Compute detailed explanation of why a stage would run.

    A stage is stale when repro will rewrite its lock by running it or restoring
    from the run cache; it is cached when its lock already matches and repro at
    most restores outputs.
    """
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
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
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Invalid params.yaml:\n{error}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    state_context: contextlib.AbstractContextManager[state.StateDB | None]
    if _state_db_exists(state_dir):
        state_context = state.StateDB(state_dir, readonly=True)
    else:
        state_context = contextlib.nullcontext(None)
    with state_context as state_db:
        known_hashes = producer_hashes or {}
        pending_deps = {name: ref for name, ref in deps.items() if ref.identity not in known_hashes}
        str_hashes, missing_deps, unreadable_deps, _ = worker.hash_dependencies(
            pending_deps,
            store,
            state_db,
        )
        dep_hashes = _to_identity_keyed(str_hashes)
        for identity, hash_info in known_hashes.items():
            if any(ref.identity == identity for ref in deps.values()):
                dep_hashes[identity] = hash_info

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
        if not allow_missing:
            unavailable = worker.restore_outputs_to_hashes(
                store,
                outs,
                lock_data["output_hashes"],
                state_db,
                state_dir=state_dir,
                restore=False,
                report_changes=True,
            ).unavailable
        if not decision["changed"] and unavailable:
            return StageExplanation(
                stage_name=stage_name,
                will_run=True,
                is_forced=force,
                reason=_unavailable_outputs_reason(store, unavailable),
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
