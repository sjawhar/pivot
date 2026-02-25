"""Detailed explanations for stage change detection.

Compares current state against lock files to explain WHY stages would run,
showing specific code, param, and dependency changes.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pydantic

from pivot import parameters, project, skip
from pivot.executor import worker
from pivot.storage import lock, state
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


def get_stage_explanation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: list[str],
    outs_paths: list[str],
    params_instance: pydantic.BaseModel | None,
    overrides: parameters.ParamsOverrides | None,
    state_dir: Path,
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
) -> StageExplanation:
    """Compute detailed explanation of why a stage would run.

    Args:
        allow_missing: If True and a dep file is missing, try to use hash from
            tracked_files (.pvt data) first, then fall back to the lock file's
            recorded hash for that dep (enabling remote verification).
        tracked_files: Dict of absolute path -> PvtData from .pvt files.
        tracked_trie: Trie of tracked paths for efficient lookup.
    """
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()

    if not lock_data:
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
    except pydantic.ValidationError as e:
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Invalid params.yaml:\n{e}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    # Check generation tracking first (O(1) skip detection)
    # Use verify_files=False since status predicts run behavior after restoration
    state_db_path = state_dir / "state.db"
    if state_db_path.exists():
        with state.StateDB(state_db_path, readonly=True) as state_db:
            if not force and worker.can_skip_via_generation(
                stage_name=stage_name,
                fingerprint=fingerprint,
                deps=deps,
                outs_paths=outs_paths,
                current_params=current_params,
                lock_data=lock_data,
                state_db=state_db,
                verify_files=False,
            ):
                return StageExplanation(
                    stage_name=stage_name,
                    will_run=False,
                    is_forced=False,
                    reason="",
                    code_changes=[],
                    param_changes=[],
                    dep_changes=[],
                    upstream_stale=[],
                )

    # Separate deps into filesystem paths (hashable) and identity keys (lock-data only).
    # Identity-keyed deps (stage-to-stage) can't be independently hashed without a Store,
    # so we use the lock file's recorded hashes as the baseline for comparison.
    file_deps = list[str]()
    identity_fallback = dict[ArtifactIdentity, HashInfo]()
    missing_deps = list[str]()

    for dep in deps:
        dep_path = pathlib.Path(dep)
        if dep_path.exists():
            file_deps.append(dep)
        else:
            # Not a filesystem path — treat as identity key
            dep_id = identity_from_key(dep)
            hash_info: HashInfo | None = None
            if allow_missing and tracked_files is not None and tracked_trie is not None:
                hash_info = _find_tracked_hash(dep_path, tracked_files, tracked_trie)
            if hash_info is None:
                hash_info = lock_data["dep_hashes"].get(dep_id)
            if hash_info:
                identity_fallback[dep_id] = hash_info
            else:
                missing_deps.append(dep)

    str_hashes, more_missing, unreadable_deps, _ = worker.hash_dependencies(file_deps)
    dep_hashes = _to_identity_keyed(str_hashes)
    dep_hashes.update(identity_fallback)
    missing_deps.extend(more_missing)

    if missing_deps or unreadable_deps:
        # fingerprint is pre-computed by the caller; diffing two dicts is cheap
        code_changes = skip.diff_code_manifests(lock_data["code_manifest"], fingerprint)
        param_changes = skip.diff_params(lock_data["params"], current_params)
        reasons = list[str]()
        if code_changes:
            reasons.append("Code changed")
        if param_changes:
            reasons.append("Params changed")
        if missing_deps:
            rel_missing = [project.to_relative_path(p) for p in missing_deps]
            reasons.append(f"Missing deps: {', '.join(rel_missing)}")
        if unreadable_deps:
            rel_unreadable = [project.to_relative_path(p) for p in unreadable_deps]
            reasons.append(f"Unreadable deps: {', '.join(rel_unreadable)}")
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

    out_identities = [identity_from_key(p) for p in outs_paths]

    decision = skip.check_stage(
        lock_data=lock_data,
        fingerprint=fingerprint,
        params=current_params,
        dep_hashes=dep_hashes,
        out_paths=out_identities,
        explain=True,
        force=force,
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
