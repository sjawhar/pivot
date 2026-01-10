"""Detailed explanations for stage change detection.

Compares current state against lock files to explain WHY stages would run,
showing specific code, param, and dependency changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import pydantic

from pivot import parameters
from pivot.executor import worker
from pivot.storage import lock
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
    """Diff two dep_hashes dicts, returning list of changes."""

    def make_dep_change(
        path: str,
        old_info: HashInfo | None,
        new_info: HashInfo | None,
        change_type: ChangeType,
    ) -> DepChange:
        old_hash = _extract_hash(old_info) if old_info else None
        new_hash = _extract_hash(new_info) if new_info else None
        return DepChange(path=path, old_hash=old_hash, new_hash=new_hash, change_type=change_type)

    return _diff_dicts(old, new, make_dep_change)


def get_stage_explanation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: list[str],
    params_instance: pydantic.BaseModel | None,
    overrides: parameters.ParamsOverrides | None,
    cache_dir: Path,
    force: bool = False,
) -> StageExplanation:
    """Compute detailed explanation of why a stage would run."""
    stage_lock = lock.StageLock(stage_name, cache_dir)
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
        )

    dep_hashes, missing_deps, unreadable_deps = worker.hash_dependencies(deps)

    if missing_deps:
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Missing deps: {', '.join(missing_deps)}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
        )

    if unreadable_deps:
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Unreadable deps: {', '.join(unreadable_deps)}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
        )

    try:
        current_params = parameters.get_effective_params(params_instance, stage_name, overrides)
    except pydantic.ValidationError as e:
        return StageExplanation(
            stage_name=stage_name,
            will_run=True,
            is_forced=force,
            reason=f"Invalid params.yaml: {e}",
            code_changes=[],
            param_changes=[],
            dep_changes=[],
        )

    # Extract lock data fields (LockData uses total=False, so check membership)
    old_manifest = lock_data["code_manifest"] if "code_manifest" in lock_data else {}  # noqa: SIM401
    old_params = lock_data["params"] if "params" in lock_data else {}  # noqa: SIM401
    old_dep_hashes = lock_data["dep_hashes"] if "dep_hashes" in lock_data else {}  # noqa: SIM401

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
    )
