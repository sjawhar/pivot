"""Unified skip detection — single source of truth for 'will this stage run?'

Covers Tier 2 (lock file comparison). Tier 1 (generation check) is invoked by
callers before this function when a StateDB is available. Tier 3 (run cache)
is handled by callers after this function when changed=True.
"""

from __future__ import annotations

from typing import Any

from pivot import project
from pivot.types import (
    ChangeDecision,
    ChangeType,
    CodeChange,
    DepChange,
    HashInfo,
    LockData,
    ParamChange,
)


def check_stage(
    lock_data: LockData | None,
    fingerprint: dict[str, str],
    params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
    out_paths: list[str],
    *,
    explain: bool = False,
    force: bool = False,
) -> ChangeDecision:
    """Single source of truth for Tier 2 skip detection.

    When explain=False (engine/repro), short-circuits at first detected change.
    When explain=True (status/explain), evaluates all comparisons exhaustively.

    Callers are responsible for:
    - Tier 1 (generation check via can_skip_via_generation) before calling this
    - Tier 3 (run cache) after this returns changed=True
    """
    if force and not explain:
        return ChangeDecision(changed=True, reason="forced")

    if lock_data is None:
        return ChangeDecision(changed=True, reason="No previous run")

    # Fast path: cheap dict equality, no detailed diffs or path conversion.
    if not explain:
        if lock_data["code_manifest"] != fingerprint:
            return ChangeDecision(changed=True, reason="Code changed")
        if lock_data["params"] != params:
            return ChangeDecision(changed=True, reason="Params changed")
        if lock_data["dep_hashes"] != dep_hashes:
            return ChangeDecision(changed=True, reason="Input dependencies changed")
        if sorted(lock_data["output_hashes"].keys()) != sorted(out_paths):
            return ChangeDecision(changed=True, reason="Output paths changed")
        return ChangeDecision(changed=False, reason="")

    # Explain path: full diffs for all categories.
    code_changes = diff_code_manifests(lock_data["code_manifest"], fingerprint)
    param_changes = diff_params(lock_data["params"], params)
    dep_changes = diff_dep_hashes(lock_data["dep_hashes"], dep_hashes)

    locked_out_paths = sorted(lock_data["output_hashes"].keys())
    out_changed = sorted(out_paths) != locked_out_paths

    changed = force or bool(code_changes or param_changes or dep_changes or out_changed)
    reason = _first_reason(code_changes, param_changes, dep_changes, out_changed, force)

    return ChangeDecision(
        changed=changed,
        reason=reason,
        code_changes=code_changes,
        param_changes=param_changes,
        dep_changes=dep_changes,
    )


def _first_reason(
    code_changes: list[CodeChange],
    param_changes: list[ParamChange],
    dep_changes: list[DepChange],
    out_changed: bool,
    force: bool,
) -> str:
    # Actual changes take precedence over "forced" — force is the fallback
    # reason when nothing else changed.
    if code_changes:
        return "Code changed"
    if param_changes:
        return "Params changed"
    if dep_changes:
        return "Input dependencies changed"
    if out_changed:
        return "Output paths changed"
    if force:
        return "forced"
    return ""


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def diff_code_manifests(old: dict[str, str], new: dict[str, str]) -> list[CodeChange]:
    changes = list[CodeChange]()
    all_keys = sorted(set(old.keys()) | set(new.keys()))
    for key in all_keys:
        in_old = key in old
        in_new = key in new
        if not in_old:
            changes.append(
                CodeChange(key=key, old_hash=None, new_hash=new[key], change_type=ChangeType.ADDED)
            )
        elif not in_new:
            changes.append(
                CodeChange(
                    key=key, old_hash=old[key], new_hash=None, change_type=ChangeType.REMOVED
                )
            )
        elif old[key] != new[key]:
            changes.append(
                CodeChange(
                    key=key, old_hash=old[key], new_hash=new[key], change_type=ChangeType.MODIFIED
                )
            )
    return changes


def diff_params(old: dict[str, Any], new: dict[str, Any]) -> list[ParamChange]:
    changes = list[ParamChange]()
    all_keys = sorted(set(old.keys()) | set(new.keys()))
    for key in all_keys:
        in_old = key in old
        in_new = key in new
        if not in_old:
            changes.append(
                ParamChange(
                    key=key, old_value=None, new_value=new[key], change_type=ChangeType.ADDED
                )
            )
        elif not in_new:
            changes.append(
                ParamChange(
                    key=key, old_value=old[key], new_value=None, change_type=ChangeType.REMOVED
                )
            )
        elif old[key] != new[key]:
            changes.append(
                ParamChange(
                    key=key, old_value=old[key], new_value=new[key], change_type=ChangeType.MODIFIED
                )
            )
    return changes


def diff_dep_hashes(old: dict[str, HashInfo], new: dict[str, HashInfo]) -> list[DepChange]:
    changes = list[DepChange]()
    all_keys = sorted(set(old.keys()) | set(new.keys()))
    for key in all_keys:
        in_old = key in old
        in_new = key in new
        if not in_old:
            rel_path = project.to_relative_path(key)
            changes.append(
                DepChange(
                    path=rel_path,
                    old_hash=None,
                    new_hash=new[key]["hash"],
                    change_type=ChangeType.ADDED,
                )
            )
        elif not in_new:
            rel_path = project.to_relative_path(key)
            changes.append(
                DepChange(
                    path=rel_path,
                    old_hash=old[key]["hash"],
                    new_hash=None,
                    change_type=ChangeType.REMOVED,
                )
            )
        elif old[key] != new[key]:
            rel_path = project.to_relative_path(key)
            changes.append(
                DepChange(
                    path=rel_path,
                    old_hash=old[key]["hash"],
                    new_hash=new[key]["hash"],
                    change_type=ChangeType.MODIFIED,
                )
            )
    return changes
