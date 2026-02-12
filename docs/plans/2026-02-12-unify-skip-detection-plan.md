# Unified Skip Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify skip detection into a single `check_stage()` function, move the skip/run decision from the worker to the engine, and simplify the worker to execution-only.

**Architecture:** A new `pivot/skip.py` module owns the "will this stage run?" decision (Tiers 1+2). The engine calls it in `_start_ready_stages()` before dispatch, holding artifact flocks for the full skip-check-through-completion duration. The worker no longer contains skip detection or artifact locking. See `docs/plans/2026-02-12-unify-skip-detection-design.md` for the full design rationale, profiling data, and architecture review.

**Tech Stack:** Python 3.13+, pytest, anyio, LMDB, fcntl (flock)

---

### Design Decisions (from brainstorming session)

**One function, not two.** `check_stage(explain=False)` short-circuits at first change. `check_stage(explain=True)` evaluates all comparisons exhaustively. Same check ordering in both modes — no divergence possible.

**Pre-computed fingerprint.** Profiling on 173-stage pipeline showed fingerprinting is always needed (for lock comparison if skipping, for lock commit if running). Deferred computation via callable adds complexity without saving work. Fingerprinting is just-in-time per stage — computed when the stage becomes ready, not up front.

**Engine holds flock for full duration.** flock can't be transferred across process boundaries. The engine acquires artifact flock, holds the fd open from check through completion, releases in `_handle_stage_completion()`. Worker runs under the engine's flock protection.

**Test style (from `tests/AGENTS.md`):** Flat `def test_*`, module-level `_helper_*`, `monkeypatch.setattr()` not direct assignment, `autospec=True` for mocks, assert observable outcomes.

---

## Task 1: Fix agent_rpc state_dir bug

One-line bug fix. Independent of all other tasks.

**Files:**
- Modify: `packages/pivot/src/pivot/engine/agent_rpc.py:300`
- Test: `packages/pivot/tests/engine/test_agent_rpc.py` (verify existing tests pass)

**Step 1: Fix the state_dir lookup**

In `packages/pivot/src/pivot/engine/agent_rpc.py`, the `"explain"` case (line 300) passes `config_io.get_state_dir()` instead of the per-stage state_dir. Replace:

```python
                        state_dir=config_io.get_state_dir(),
```

With:

```python
                        state_dir=registry_mod.get_stage_state_dir(
                            reg_info, config_io.get_state_dir()
                        ),
```

Add the import at the top of the file (with existing imports from `pivot`):

```python
from pivot import registry as registry_mod
```

**Step 2: Verify type checker is clean**

Run:
```bash
uv run basedpyright packages/pivot/src/pivot/engine/agent_rpc.py
```
Expected: Clean (or only pre-existing warnings).

**Step 3: Run agent_rpc tests**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_agent_rpc.py -v
```
Expected: All PASS.

**Step 4: Commit**

Message: "fix: agent_rpc explain uses per-stage state_dir instead of global default"

---

## Task 2: Create `ChangeDecision` type

Add the return type for `check_stage()` to `pivot/types.py`.

**Files:**
- Modify: `packages/pivot/src/pivot/types.py`

**Step 1: Add `ChangeDecision` TypedDict**

Add after `StageExplanation` (around line 357):

```python
class ChangeDecision(TypedDict, total=False):
    """Result of skip detection check_stage().

    In fast mode (explain=False), detail fields are omitted (not present).
    In explain mode (explain=True), they contain the full diff information.
    """

    changed: Required[bool]
    reason: Required[str]
    code_changes: list[CodeChange]
    param_changes: list[ParamChange]
    dep_changes: list[DepChange]
```

Update the module's exports if there's an `__all__` list to include `"ChangeDecision"`.

**Step 2: Verify type checker**

Run:
```bash
uv run basedpyright packages/pivot/src/pivot/types.py
```
Expected: Clean.

**Step 3: Commit**

Message: "feat: add ChangeDecision type for unified skip detection"

---

## Task 3: Create `pivot/skip.py` with `check_stage()`

The core of this plan. Extract and unify skip detection logic into a single module.

**Files:**
- Create: `packages/pivot/src/pivot/skip.py`
- Test: `packages/pivot/tests/core/test_skip.py`

**Step 1: Write tests for `check_stage()` fast mode**

Create `packages/pivot/tests/core/test_skip.py`:

```python
"""Tests for skip.check_stage — unified skip detection."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from pivot import skip
from pivot.types import ChangeDecision, ChangeType, HashInfo, LockData


def _helper_make_lock_data(
    *,
    code_manifest: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    dep_hashes: dict[str, HashInfo] | None = None,
    output_hashes: dict[str, str] | None = None,
) -> LockData:
    """Build a minimal LockData for testing."""
    return LockData(
        code_manifest=code_manifest or {"func:main": "abc123"},
        params=params or {"lr": 0.01},
        dep_hashes=dep_hashes or {"data/input.csv": {"hash": "hash_a"}},
        output_hashes=output_hashes or {"data/output.csv": "hash_out"},
    )


# =============================================================================
# No lock data (first run)
# =============================================================================


def test_check_stage_no_lock_data_returns_changed() -> None:
    """First run (no lock file) always returns changed."""
    result = skip.check_stage(
        lock_data=None,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
    )
    assert result["changed"] is True
    assert "No previous run" in result["reason"]


# =============================================================================
# Fast mode: short-circuit behavior
# =============================================================================


def test_check_stage_fast_code_changed() -> None:
    """Code manifest change detected and short-circuits."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
    )
    assert result["changed"] is True
    assert "Code changed" in result["reason"]


def test_check_stage_fast_params_changed() -> None:
    """Params change detected."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.99},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
    )
    assert result["changed"] is True
    assert "Params changed" in result["reason"]


def test_check_stage_fast_deps_changed() -> None:
    """Dep hash change detected."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "DIFFERENT"}},
        out_paths=["data/output.csv"],
    )
    assert result["changed"] is True
    assert "dependencies changed" in result["reason"]


def test_check_stage_fast_out_paths_changed() -> None:
    """Output path list change detected."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv", "data/new_output.csv"],
    )
    assert result["changed"] is True
    assert "Output paths changed" in result["reason"]


def test_check_stage_fast_nothing_changed() -> None:
    """All fields match — stage can skip."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
    )
    assert result["changed"] is False
    assert result["reason"] == ""


def test_check_stage_fast_force_returns_changed() -> None:
    """Force flag always returns changed."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
        force=True,
    )
    assert result["changed"] is True
    assert "forced" in result["reason"]


# =============================================================================
# Explain mode: exhaustive comparisons
# =============================================================================


def test_check_stage_explain_returns_all_changes() -> None:
    """Explain mode evaluates ALL comparisons, doesn't short-circuit."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.99},
        dep_hashes={"data/input.csv": {"hash": "DIFFERENT"}},
        out_paths=["data/output.csv"],
        explain=True,
    )
    assert result["changed"] is True
    assert len(result.get("code_changes", [])) > 0, "Should report code changes"
    assert len(result.get("param_changes", [])) > 0, "Should report param changes"
    assert len(result.get("dep_changes", [])) > 0, "Should report dep changes"


def test_check_stage_explain_nothing_changed() -> None:
    """Explain mode with no changes returns empty change lists."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "abc123"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
        explain=True,
    )
    assert result["changed"] is False
    assert result.get("code_changes", []) == []
    assert result.get("param_changes", []) == []
    assert result.get("dep_changes", []) == []


# =============================================================================
# Short-circuit verification: fast mode doesn't populate detail fields
# =============================================================================


def test_check_stage_fast_mode_no_detail_fields_on_short_circuit() -> None:
    """Fast mode short-circuits — detail fields are absent or empty."""
    lock_data = _helper_make_lock_data()
    result = skip.check_stage(
        lock_data=lock_data,
        fingerprint={"func:main": "DIFFERENT"},
        params={"lr": 0.01},
        dep_hashes={"data/input.csv": {"hash": "hash_a"}},
        out_paths=["data/output.csv"],
    )
    assert result["changed"] is True
    assert result.get("param_changes") is None or result.get("param_changes") == [], (
        "Fast mode should not compute param changes after code short-circuit"
    )
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/pivot/tests/core/test_skip.py -v
```
Expected: FAIL — `pivot.skip` doesn't exist yet.

**Step 3: Implement `skip.py`**

Create `packages/pivot/src/pivot/skip.py`:

```python
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
    if force:
        if not explain:
            return ChangeDecision(changed=True, reason="forced")
        # explain mode falls through to compute all diffs

    if lock_data is None:
        return ChangeDecision(changed=True, reason="No previous run")

    # Tier 2: ordered comparisons, cheapest first
    code_changes = _diff_code_manifests(lock_data["code_manifest"], fingerprint)
    if code_changes and not explain and not force:
        return ChangeDecision(changed=True, reason="Code changed")

    param_changes = _diff_params(lock_data["params"], params)
    if param_changes and not explain and not force:
        return ChangeDecision(changed=True, reason="Params changed")

    dep_changes = _diff_dep_hashes(lock_data["dep_hashes"], dep_hashes)
    if dep_changes and not explain and not force:
        return ChangeDecision(changed=True, reason="Input dependencies changed")

    out_changes = _diff_out_paths(
        sorted(lock_data["output_hashes"].keys()), sorted(out_paths)
    )
    if out_changes and not explain and not force:
        return ChangeDecision(changed=True, reason="Output paths changed")

    changed = force or bool(code_changes or param_changes or dep_changes or out_changes)
    reason = _first_reason(code_changes, param_changes, dep_changes, out_changes, force)

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
    out_changes: bool,
    force: bool,
) -> str:
    if force:
        return "forced"
    if code_changes:
        return "Code changed"
    if param_changes:
        return "Params changed"
    if dep_changes:
        return "Input dependencies changed"
    if out_changes:
        return "Output paths changed"
    return ""


# ---------------------------------------------------------------------------
# Diff functions (moved from explain.py)
# ---------------------------------------------------------------------------

_T = Any  # Generic type var for dict values
_C = Any  # Generic type var for change objects


def _diff_dicts(
    old: dict[str, Any],
    new: dict[str, Any],
    make_change: Any,
) -> list[Any]:
    """Generic dict differ that produces typed change objects."""
    changes = list[Any]()
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


def _diff_code_manifests(old: dict[str, str], new: dict[str, str]) -> list[CodeChange]:
    """Diff two code manifests, returning list of changes."""
    return _diff_dicts(
        old,
        new,
        lambda k, o, n, t: CodeChange(key=k, old_hash=o, new_hash=n, change_type=t),
    )


def _diff_params(old: dict[str, Any], new: dict[str, Any]) -> list[ParamChange]:
    """Diff two param dicts, returning list of changes."""
    return _diff_dicts(
        old,
        new,
        lambda k, o, n, t: ParamChange(key=k, old_value=o, new_value=n, change_type=t),
    )


def _extract_hash(info: HashInfo) -> str:
    """Extract hash from HashInfo (FileHash or DirHash)."""
    return info["hash"]


def _diff_dep_hashes(old: dict[str, HashInfo], new: dict[str, HashInfo]) -> list[DepChange]:
    """Diff two dep_hashes dicts, returning list of changes."""

    def make_dep_change(
        path: str,
        old_info: HashInfo | None,
        new_info: HashInfo | None,
        change_type: ChangeType,
    ) -> DepChange:
        old_hash = _extract_hash(old_info) if old_info else None
        new_hash = _extract_hash(new_info) if new_info else None
        rel_path = project.to_relative_path(path)
        return DepChange(
            path=rel_path, old_hash=old_hash, new_hash=new_hash, change_type=change_type
        )

    return _diff_dicts(old, new, make_dep_change)


def _diff_out_paths(old_sorted: list[str], new_sorted: list[str]) -> bool:
    """Check if output path lists differ. Returns True if changed."""
    return old_sorted != new_sorted
```

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/pivot/tests/core/test_skip.py -v
```
Expected: All PASS.

**Step 5: Run type checker**

Run:
```bash
uv run basedpyright packages/pivot/src/pivot/skip.py
```
Expected: Clean.

**Step 6: Commit**

Message: "feat: add skip.check_stage() — unified skip detection function"

---

## Task 4: Wire `explain.py` to use `skip.check_stage(explain=True)`

Replace the inline Tier 2 logic in `get_stage_explanation()` with a call to
`skip.check_stage(explain=True)`. Keep the explain-specific features
(allow_missing, tracked files, upstream_stale) as wrappers around the shared
function.

**Files:**
- Modify: `packages/pivot/src/pivot/explain.py`
- Test: `packages/pivot/tests/core/test_explain.py`

**Step 1: Run existing explain tests (baseline)**

Run:
```bash
uv run pytest packages/pivot/tests/core/test_explain.py -v
```
Expected: All PASS. Record the count.

**Step 2: Modify `get_stage_explanation()` to call `check_stage`**

In `explain.py`, replace the inline Tier 2 comparison block (lines 293-315)
with a call to `skip.check_stage(explain=True)`. The function should:

1. Keep the existing Tier 1 generation check (lines 209-233) — it returns early
2. Keep the existing dep hashing with `allow_missing` fallback (lines 236-263) — this is explain-specific
3. Replace lines 293-315 (the manual `diff_code_manifests` / `diff_params` / `diff_dep_hashes` calls and reason building) with:

```python
    from pivot import skip

    decision = skip.check_stage(
        lock_data=lock_data,
        fingerprint=fingerprint,
        params=current_params,
        dep_hashes=dep_hashes,
        out_paths=outs_paths,
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
```

Remove the now-unused local diff calls and the `old_manifest` / `old_params` / `old_dep_hashes` variables.

**Step 3: Remove public `diff_*` functions from `explain.py`**

The `diff_code_manifests`, `diff_params`, `diff_dep_hashes`, and `_diff_dicts`
functions have moved to `skip.py`. Delete them from `explain.py`. Keep
`_extract_hash` only if still referenced (it moved to `skip.py` too). Keep the
`_find_tracked_*` functions — they're explain-specific.

Check for other callers of the public `diff_*` functions before deleting:

Run:
```bash
grep -rn 'explain\.diff_code_manifests\|explain\.diff_params\|explain\.diff_dep_hashes' packages/pivot/
```

Update any other callers to import from `skip` instead of `explain`.

**Step 4: Run explain tests**

Run:
```bash
uv run pytest packages/pivot/tests/core/test_explain.py -v
```
Expected: All PASS (same count as baseline).

**Step 5: Run type checker on modified files**

Run:
```bash
uv run basedpyright packages/pivot/src/pivot/explain.py packages/pivot/src/pivot/skip.py
```
Expected: Clean.

**Step 6: Commit**

Message: "refactor: explain.py delegates Tier 2 comparison to skip.check_stage"

---

## Task 5: Wire `agent_rpc.py` to use `skip.check_stage(explain=True)`

**Files:**
- Modify: `packages/pivot/src/pivot/engine/agent_rpc.py:281-303`

**Step 1: Replace explain call with skip.check_stage**

The `"explain"` case in `_handle_query` currently calls
`explain_mod.get_stage_explanation()`. This can remain as-is since Task 4
already made `get_stage_explanation` delegate to `skip.check_stage`. No changes
needed here beyond the state_dir fix from Task 1.

Verify the state_dir fix from Task 1 is in place and the explain path works:

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_agent_rpc.py -v
```
Expected: All PASS.

**Step 2: Commit (if any changes)**

Message: "chore: verify agent_rpc explain path uses unified skip detection"

---

## Task 6: Quality checks

**Step 1: Run formatter and linter**

Run:
```bash
uv run ruff format . && uv run ruff check .
```
Expected: Clean.

**Step 2: Run type checker**

Run:
```bash
uv run basedpyright
```
Expected: Clean (or only pre-existing warnings).

**Step 3: Run full test suite**

Run:
```bash
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto
```
Expected: All PASS.

**Step 4: Commit any fixups**

Message: "chore: fix lint/type issues from skip detection unification"

---

## Summary of Acceptance Criteria

| Criterion | Verified By |
|-----------|------------|
| `check_stage(explain=False)` short-circuits at first change | Task 3 tests |
| `check_stage(explain=True)` returns all change details | Task 3 tests |
| `check_stage` and `is_changed_with_lock_data` agree on all inputs | Task 3 tests |
| `explain.py` delegates to `check_stage` | Task 4 (existing explain tests pass) |
| `agent_rpc.py` uses per-stage state_dir | Task 1 |
| Output path comparison present in explain path (previously missing) | Task 3 tests (`test_check_stage_fast_out_paths_changed`) |
| Full test suite passes | Task 6 Step 3 |

## Execution Order

```
Task 1  — Fix agent_rpc state_dir bug (independent, ship immediately)
Task 2  — Add ChangeDecision type
Task 3  — Create skip.py with check_stage() and tests
Task 4  — Wire explain.py to use check_stage
Task 5  — Verify agent_rpc (may be no-op after Tasks 1+4)
Task 6  — Quality checks
```

## Out of Scope (separate plan)

The following are part of the design but require a separate implementation plan:

- **Engine wiring**: Moving skip detection into `_start_ready_stages()`, engine
  holding artifact flocks, output restoration in engine
- **Worker simplification**: Removing skip detection and artifact locking from worker
- **Run cache in engine**: Moving Tier 3 from worker to engine

These depend on the engine hardening plan (`2026-02-11-engine-hardening.md`)
completing first, as both modify `_start_ready_stages()` and the engine's
orchestration loop. The `skip.py` module from this plan is a prerequisite.
