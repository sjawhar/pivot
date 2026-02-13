# Split StageStatus.SKIPPED Into Semantic Statuses — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the ambiguous `StageStatus.SKIPPED` with three specific statuses (`CACHED`, `BLOCKED`, `CANCELLED`) so every consumer gets the correct semantic status without calling `categorize_stage_result()`.

**Architecture:** Add `CACHED`, `BLOCKED`, `CANCELLED` to `StageStatus`. Remove `SKIPPED`. Update all 6 creation sites (4 in worker, 2 in engine) to set the correct status. Simplify `categorize_stage_result()` to a trivial 1:1 mapping. Update all consumers.

**Tech Stack:** Python 3.13+ StrEnum, TypedDict, Literal types

---

## Scope

**Source files:** 10 files, ~18 references to update
**Test files:** ~25 files, ~50 references to update
**Breaking change:** Yes — `StageStatus.SKIPPED` removed, lock files with `"skipped"` won't load. Pre-alpha, acceptable per AGENTS.md.

---

### Task 1: Update StageStatus Enum and Type Aliases

**Files:**
- Modify: `packages/pivot/src/pivot/types.py`
- Test: `packages/pivot/tests/test_types.py`

**Step 1: Write the failing test**

Update `test_completion_type_includes_skipped` → rename to `test_completion_type_includes_all_terminal_statuses`:

```python
def test_completion_type_includes_all_terminal_statuses() -> None:
    """CompletionType includes all terminal status values."""
    assert StageStatus.RAN in get_args(CompletionType)
    assert StageStatus.CACHED in get_args(CompletionType)
    assert StageStatus.BLOCKED in get_args(CompletionType)
    assert StageStatus.CANCELLED in get_args(CompletionType)
    assert StageStatus.FAILED in get_args(CompletionType)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/test_types.py -k "completion_type" -v`
Expected: FAIL — `StageStatus.CACHED` doesn't exist

**Step 3: Update the enum and type aliases**

In `types.py`:

```python
class StageStatus(enum.StrEnum):
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CACHED = "cached"         # Was SKIPPED — stage up-to-date, outputs restored from cache
    BLOCKED = "blocked"       # Was SKIPPED — upstream dependency failed
    CANCELLED = "cancelled"   # Was SKIPPED — run cancelled by user
    FAILED = "failed"
    RAN = "ran"
    UNKNOWN = "unknown"

CompletionType = Literal[
    StageStatus.RAN, StageStatus.CACHED, StageStatus.BLOCKED,
    StageStatus.CANCELLED, StageStatus.FAILED,
]
```

Update `categorize_stage_result` — it no longer needs reason-parsing for SKIPPED:

```python
def categorize_stage_result(status: StageStatus, reason: str) -> DisplayCategory:
    """Map status to display category for consistent UI."""
    match status:
        case StageStatus.READY:
            return DisplayCategory.PENDING
        case StageStatus.IN_PROGRESS:
            return DisplayCategory.RUNNING
        case StageStatus.COMPLETED | StageStatus.RAN:
            return DisplayCategory.SUCCESS
        case StageStatus.FAILED:
            return DisplayCategory.FAILED
        case StageStatus.CACHED:
            return DisplayCategory.CACHED
        case StageStatus.BLOCKED:
            return DisplayCategory.BLOCKED
        case StageStatus.CANCELLED:
            return DisplayCategory.CANCELLED
        case StageStatus.UNKNOWN:
            return DisplayCategory.UNKNOWN
```

Note: `categorize_stage_result` is now trivial (1:1 mapping). It can stay for backward compatibility with existing call sites — removing all callers is a separate cleanup. The important thing is it no longer parses `reason` strings.

Update `StageRunRecord` type annotation (line ~736):
```python
status: Literal[StageStatus.RAN, StageStatus.CACHED, StageStatus.BLOCKED, StageStatus.CANCELLED, StageStatus.FAILED]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/test_types.py -k "completion_type" -v`
Expected: PASS

**Step 5: Commit**

---

### Task 2: Update Worker — All SKIPPED Returns → CACHED

**Files:**
- Modify: `packages/pivot/src/pivot/executor/worker.py`
- Test: `packages/pivot/tests/execution/test_executor_worker.py`

Workers only return SKIPPED when a stage is up-to-date (cache hit). All 4 sites become CACHED.

**Step 1: Update `_make_result` type annotation (line 157)**

```python
status: Literal[StageStatus.RAN, StageStatus.CACHED, StageStatus.FAILED],
```

**Step 2: Replace all 4 SKIPPED returns**

- Line 276: `StageStatus.SKIPPED` → `StageStatus.CACHED` (generation-based skip)
- Line 329: `StageStatus.SKIPPED` → `StageStatus.CACHED` (lock-based skip)
- Line 350: `StageStatus.SKIPPED` → `StageStatus.CACHED` (run cache, no-commit mode)
- Line 372: `StageStatus.SKIPPED` → `StageStatus.CACHED` (run cache, with deferred writes)

**Step 3: Run worker tests**

Run: `uv run pytest packages/pivot/tests/execution/test_executor_worker.py -x --tb=short`

Many tests will fail because they assert `status == "skipped"`. Update all assertions:
- Grep for `== "skipped"` in `test_executor_worker.py` (~12 locations)
- All should become `== "cached"` (workers only produce cache-hit skips)

**Step 4: Run full worker test suite**

Run: `uv run pytest packages/pivot/tests/execution/test_executor_worker.py -v`
Expected: PASS

**Step 5: Commit**

---

### Task 3: Update Engine — Blocked and Cancelled Stages

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Test: `packages/pivot/tests/engine/test_engine.py`

The engine creates SKIPPED events for three distinct scenarios:
- Cached (worker returned skip) — already CACHED from Task 2
- Blocked (upstream failed) — needs BLOCKED
- Cancelled (user cancelled) — needs CANCELLED

**Step 1: Rename `_emit_skipped_stage` → `_emit_terminal_stage`**

Update the method to accept a `status` parameter instead of always using SKIPPED:

```python
async def _emit_terminal_stage(
    self,
    stage_name: str,
    status: StageStatus,
    reason: str,
    results: dict[str, executor_core.ExecutionSummary],
    run_id: str = "",
) -> None:
    """Record and emit a non-executed stage completion (cached/blocked/cancelled)."""
    results[stage_name] = executor_core.ExecutionSummary(
        status=status,
        reason=reason,
        input_hash=None,
    )
    stage_index, total_stages = self._get_stage_index(stage_name)
    await self.emit(
        StageCompleted(
            type="stage_completed",
            stage=stage_name,
            status=status,
            reason=reason,
            duration_ms=0.0,
            index=stage_index,
            total=total_stages,
            run_id=run_id,
            input_hash=None,
            output_summary=None,
        )
    )
```

**Step 2: Update all 3 call sites**

- Line 1047 (blocked by failed upstream):
  ```python
  await self._emit_terminal_stage(name, StageStatus.BLOCKED, f"upstream '{first_failed}' failed", results, run_id=run_id)
  ```

- Line 1066 (cancelled):
  ```python
  await self._emit_terminal_stage(name, StageStatus.CANCELLED, "cancelled", results, run_id=run_id)
  ```

- Line 1109 (blocked by failed upstream, post-loop cleanup):
  ```python
  await self._emit_terminal_stage(name, StageStatus.BLOCKED, f"upstream '{failed_upstream}' failed", results, run_id=run_id)
  ```

**Step 3: Update line 973 — deferred writes check**

```python
# Before:
result["status"] in (StageStatus.RAN, StageStatus.SKIPPED)
# After:
result["status"] in (StageStatus.RAN, StageStatus.CACHED)
```

Only CACHED stages have deferred writes (blocked/cancelled don't execute at all).

**Step 4: Update `_record_skipped_stage` (line ~1660)**

This method handles worker-returned skips (all CACHED). Rename to `_record_cached_stage` and use `StageStatus.CACHED`.

**Step 5: Run engine tests**

Run: `uv run pytest packages/pivot/tests/engine/test_engine.py -x --tb=short`

Update assertions: grep for `"skipped"` (~3 locations). Determine whether each should become `"cached"`, `"blocked"`, or `"cancelled"` based on the test scenario.

**Step 6: Commit**

---

### Task 4: Update Type Annotations

**Files:**
- Modify: `packages/pivot/src/pivot/executor/core.py` (line 120)
- Modify: `packages/pivot/src/pivot/engine/types.py` (StageCompleted TypedDict)

**Step 1: Update ExecutionSummary**

```python
status: Literal[StageStatus.RAN, StageStatus.CACHED, StageStatus.BLOCKED, StageStatus.CANCELLED, StageStatus.FAILED, StageStatus.UNKNOWN]
```

**Step 2: Update `count_results` in executor/core.py**

The function currently uses `categorize_stage_result`. Now it can match directly:

```python
match result["status"]:
    case StageStatus.RAN:
        ran += 1
    case StageStatus.FAILED:
        failed += 1
    case StageStatus.BLOCKED:
        blocked += 1
    case StageStatus.CACHED | StageStatus.CANCELLED:
        cached += 1
    case _:
        pass
```

**Step 3: Run affected tests**

Run: `uv run pytest packages/pivot/tests/execution/ packages/pivot/tests/engine/test_types.py -x --tb=short`

**Step 4: Commit**

---

### Task 5: Update CLI Consumers

**Files:**
- Modify: `packages/pivot/src/pivot/cli/repro.py`
- Modify: `packages/pivot/src/pivot/cli/run.py`
- Modify: `packages/pivot/src/pivot/cli/history.py`
- Modify: `packages/pivot/src/pivot/cli/console.py`
- Modify: `packages/pivot/src/pivot/engine/sinks.py`

**Step 1: Update repro.py and run.py — SKIPPED count**

Both have `sum(1 for r in results.values() if r["status"] == StageStatus.SKIPPED)`. Replace:

```python
skipped = sum(1 for r in results.values() if r["status"] in (StageStatus.CACHED, StageStatus.BLOCKED, StageStatus.CANCELLED))
```

Or better — use `count_results()` which already handles this.

**Step 2: Update history.py**

- Line 54: Replace `StageStatus.SKIPPED` count with CACHED + BLOCKED + CANCELLED
- Line 127: Update match case:
  ```python
  case StageStatus.CACHED | StageStatus.BLOCKED | StageStatus.CANCELLED:
      icon = "•"
  ```

**Step 3: Update console.py**

The `stage_result` method (line 149) already calls `categorize_stage_result()` which now does a trivial mapping. No changes needed — it still works. Optionally, replace with direct status matching later (cleanup task, not blocking).

**Step 4: Update sinks.py**

The sinks already use `DisplayCategory` via `_categorize()`. Since `categorize_stage_result()` still works (trivial mapping now), the sinks work as-is. But clean up:
- The `_SKIP_CATEGORIES` set and `_categorize()` helper can match on `StageStatus` directly instead of going through `DisplayCategory`
- The `_CATEGORY_SYMBOL` and `_CATEGORY_WORD` dicts can be keyed on `StageStatus` instead of `DisplayCategory`

This is optional cleanup — sinks work correctly either way since `categorize_stage_result()` is now a trivial 1:1 mapping.

**Step 5: Run CLI tests**

Run: `uv run pytest packages/pivot/tests/cli/ -x --tb=short`

Update failing tests that check for `"skipped"` string in output:
- `test_cli.py` line 791
- `test_cli_commit.py` line 263
- Various keep-going tests (already updated for "blocked")

**Step 6: Commit**

---

### Task 6: Update TUI

**Files:**
- Modify: `packages/pivot-tui/src/pivot_tui/run.py`
- Modify: `packages/pivot-tui/src/pivot_tui/widgets/status.py`
- Modify: `packages/pivot-tui/src/pivot_tui/widgets/logs.py`

**Step 1: Update run.py**

- Line 550: Replace `StageStatus.SKIPPED` with all three new statuses in the completion check tuple
- Line 572: Same — update progress counting
- Line 666: Replace `status == StageStatus.SKIPPED` with `status in (StageStatus.CACHED, StageStatus.BLOCKED, StageStatus.CANCELLED)`

**Step 2: Update widgets/status.py**

The status widget functions already use `categorize_stage_result()` which still works. No changes required — `categorize_stage_result(StageStatus.CACHED, "")` returns `DisplayCategory.CACHED` just like before. 

Optionally, switch to matching `StageStatus` directly (cleanup task).

**Step 3: Update widgets/logs.py**

- Line 109: Replace SKIPPED match:
  ```python
  case StageStatus.CACHED | StageStatus.BLOCKED | StageStatus.CANCELLED:
      self.write("[dim]Stage was skipped[/]")
  ```

**Step 4: Run TUI tests**

Run: `uv run pytest packages/pivot-tui/tests/ -x --tb=short`

Update: `test_status.py` parametrized tests to use `StageStatus.CACHED` (with "cache hit" reason) and `StageStatus.BLOCKED` (with "upstream failed" reason) instead of `StageStatus.SKIPPED`.

**Step 5: Commit**

---

### Task 7: Update Remaining Test Files

**Files:** ~15 test files with `== "skipped"` assertions

This is the mechanical bulk of the work. Each `"skipped"` assertion needs to become `"cached"`, `"blocked"`, or `"cancelled"` depending on the test scenario.

**Pattern:**
- Tests where a stage is unchanged/up-to-date: `"skipped"` → `"cached"`
- Tests where a stage is blocked by upstream failure: `"skipped"` → `"blocked"`
- Tests where a run is cancelled: `"skipped"` → `"cancelled"`

**Files to update (by expected new status):**

| File | Assertions | Expected new status |
|------|-----------|---------------------|
| `test_executor_worker.py` | 12 | `"cached"` (all cache-hit scenarios) |
| `test_executor.py` | 7 | `"cached"` (most), `"blocked"` (upstream failure tests) |
| `test_skip_detection_integration.py` | 5 | `"cached"` |
| `test_run_cache_lock_update.py` | 4 | `"cached"` |
| `test_no_fingerprint.py` | 4 | `"cached"` |
| `test_engine.py` | 3 | `"cached"` or `"blocked"` depending on test |
| `test_incremental_out.py` | 2 | `"cached"` |
| `test_execution_modes.py` | 1 | `"cached"` |
| `test_executor_pvt.py` | 1 | `"cached"` |
| `test_cli_commit.py` | 1 | `"cached"` |
| `test_cli.py` | 1 | `"cached"` |
| `test_run_history.py` | 2 | `"cached"` |
| `test_jsonl_sink.py` | 1 | `"cached"` |
| `test_cli_run_common.py` | 2 | `"cached"` |
| `test_engine_shutdown.py` | 1 | Add CACHED, BLOCKED, CANCELLED to acceptable statuses |
| `test_unified_execution.py` | 1 | Add CACHED, BLOCKED, CANCELLED to acceptable statuses |

**Note:** `test_sync.py` and `test_transfer.py` use `result["skipped"]` for sync operation counts — NOT stage status. Leave these unchanged.

**Step 1: Update each test file** (can be done in batches by directory)

**Step 2: Run full test suite**

Run: `uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto`
Expected: PASS

**Step 3: Run quality checks**

Run: `uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: Clean

**Step 4: Commit**
