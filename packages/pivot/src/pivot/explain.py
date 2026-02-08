"""Detailed explanations for stage change detection.

Compares current state against lock files to explain WHY stages would run,
showing specific code, param, and dependency changes.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, TypeVar

import pydantic

from pivot import parameters, project
from pivot.executor import worker
from pivot.storage import lock, state
from pivot.types import (
    ChangeType,
    CodeChange,
    DepChange,
    HashInfo,
    ParamChange,
    StageExplanation,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pygtrie

    from pivot.storage.track import PvtData


T = TypeVar("T")
C = TypeVar("C")


def _diff_dicts(
    old: dict[str, T],
    new: dict[str, T],
    make_change: Callable[[str, T | None, T | None, ChangeType], C],
) -> list[C]:
    """Generic dict differ that produces typed change objects."""
    changes = list[C]()
    all_keys = set(old.keys()) | set(new.keys())

    for key in sorted(all_keys):
        in_old = key in old
        in_new = key in new

        if not in_old:
            changes.append(make_change(key, None, new[key], ChangeType.ADDED))
        elif not in_new:
            changes.append(make_change(key, old[key], None, ChangeType.REMOVED))
        elif old[key] != new[key]:
            changes.append(make_change(key, old[key], new[key], ChangeType.MODIFIED))

    return changes


def diff_code_manifests(old: dict[str, str], new: dict[str, str]) -> list[CodeChange]:
    """Diff two code manifests, returning list of changes."""
    return _diff_dicts(
        old,
        new,
        lambda k, o, n, t: CodeChange(key=k, old_hash=o, new_hash=n, change_type=t),
    )


def diff_params(old: dict[str, Any], new: dict[str, Any]) -> list[ParamChange]:
    """Diff two param dicts, returning list of changes."""
    return _diff_dicts(
        old,
        new,
        lambda k, o, n, t: ParamChange(key=k, old_value=o, new_value=n, change_type=t),
    )


def _extract_hash(info: HashInfo) -> str:
    """Extract hash from HashInfo (FileHash or DirHash)."""
    return info["hash"]


def diff_dep_hashes(old: dict[str, HashInfo], new: dict[str, HashInfo]) -> list[DepChange]:
    """Diff two dep_hashes dicts, returning list of changes.

    Paths in the result are relative to project root for user-facing display.
    """

    def make_dep_change(
        path: str,
        old_info: HashInfo | None,
        new_info: HashInfo | None,
        change_type: ChangeType,
    ) -> DepChange:
        old_hash = _extract_hash(old_info) if old_info else None
        new_hash = _extract_hash(new_info) if new_info else None
        # Convert absolute paths to relative for user-facing output
        rel_path = project.to_relative_path(path)
        return DepChange(
            path=rel_path, old_hash=old_hash, new_hash=new_hash, change_type=change_type
        )

    return _diff_dicts(old, new, make_dep_change)


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

    # Hash dependencies - with optional fallback for missing files
    if allow_missing:
        deps_to_hash = list[str]()
        fallback_hashes = dict[str, HashInfo]()
        missing_deps = list[str]()

        for dep in deps:
            dep_path = pathlib.Path(dep)
            if dep_path.exists():
                deps_to_hash.append(dep)
            else:
                # Try .pvt file first
                hash_info = None
                if tracked_files is not None and tracked_trie is not None:
                    hash_info = _find_tracked_hash(dep_path, tracked_files, tracked_trie)
                # Fall back to lock file hash (for remote verification)
                normalized = str(project.normalize_path(dep))
                if hash_info is None:
                    hash_info = lock_data["dep_hashes"].get(normalized)
                if hash_info:
                    fallback_hashes[normalized] = hash_info
                else:
                    missing_deps.append(dep)

        file_hashes, more_missing, unreadable_deps = worker.hash_dependencies(deps_to_hash)
        dep_hashes = {**file_hashes, **fallback_hashes}
        missing_deps.extend(more_missing)
    else:
        dep_hashes, missing_deps, unreadable_deps = worker.hash_dependencies(deps)

    if missing_deps:
        # Convert to relative paths for user-facing message
        rel_missing = [project.to_relative_path(p) for p in missing_deps]
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Missing deps: {', '.join(rel_missing)}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    if unreadable_deps:
        # Convert to relative paths for user-facing message
        rel_unreadable = [project.to_relative_path(p) for p in unreadable_deps]
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Unreadable deps: {', '.join(rel_unreadable)}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
            upstream_stale=[],
        )

    # Extract lock data fields (LockData has all required keys after _convert_from_storage_format)
    old_manifest = lock_data["code_manifest"]
    old_params = lock_data["params"]
    old_dep_hashes = lock_data["dep_hashes"]

    # lock.StageLock.read() already converts paths to absolute
    code_changes = diff_code_manifests(old_manifest, fingerprint)
    param_changes = diff_params(old_params, current_params)
    dep_changes = diff_dep_hashes(old_dep_hashes, dep_hashes)

    has_changes = bool(code_changes or param_changes or dep_changes)
    will_run = has_changes or force

    if force and not has_changes:
        reason = "forced"
    elif code_changes:
        reason = "Code changed"
    elif param_changes:
        reason = "Params changed"
    elif dep_changes:
        reason = "Input dependencies changed"
    else:
        reason = ""

    return StageExplanation(
        stage_name=stage_name,
        will_run=will_run,
        is_forced=force,
        reason=reason,
        code_changes=code_changes,
        param_changes=param_changes,
        dep_changes=dep_changes,
        upstream_stale=[],
    )
