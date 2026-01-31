# Unified Watch/Non-Watch Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify watch and non-watch execution paths so non-watch mode uses the event system, fixing `--quiet` flag and stale stage list bugs.

**Architecture:** Non-watch mode becomes "watch mode with restrictions" — same event-driven execution, but with `OneShotSource` only and `exit_on_completion=True`. Single CLI code path configures sinks identically for both modes.

**Tech Stack:** Python 3.13+, TypedDict, loky, watchfiles

**GitHub Issue:** #305

---

## Task 1: Add `CompletionType` Alias

**Files:**
- Modify: `src/pivot/types.py:35-45`
- Test: `tests/test_types.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_types.py
"""Tests for pivot.types module."""

from pivot.types import CompletionType, StageStatus


def test_completion_type_includes_ran() -> None:
    """CompletionType should include RAN."""
    status: CompletionType = StageStatus.RAN
    assert status == StageStatus.RAN


def test_completion_type_includes_skipped() -> None:
    """CompletionType should include SKIPPED."""
    status: CompletionType = StageStatus.SKIPPED
    assert status == StageStatus.SKIPPED


def test_completion_type_includes_failed() -> None:
    """CompletionType should include FAILED."""
    status: CompletionType = StageStatus.FAILED
    assert status == StageStatus.FAILED
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ImportError: cannot import name 'CompletionType'`

**Step 3: Write minimal implementation**

In `src/pivot/types.py`, after the `StageStatus` class (around line 45), add:

```python
CompletionType = Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED]
"""Status values for stages that have finished execution."""
```

Add `Literal` to the imports at the top of the file if not present.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS

**Step 5: Run type checker**

Run: `uv run basedpyright src/pivot/types.py`
Expected: No errors

**Step 6: Commit**

```bash
jj describe -m "refactor(types): add CompletionType alias for terminal stage statuses"
```

---

## Task 2: Update `StageCompleted` to Use `CompletionType`

**Files:**
- Modify: `src/pivot/engine/types.py:135-144`
- Test: Type checker validates (no runtime test needed)

**Step 1: Write the failing test**

This is a type-level change. Create a type test file:

```python
# tests/engine/test_types_static.py
"""Static type tests for engine types (validated by type checker, not pytest)."""

from pivot.engine.types import StageCompleted
from pivot.types import CompletionType, StageStatus


def test_stage_completed_status_is_completion_type() -> None:
    """StageCompleted.status should only accept CompletionType values."""
    event: StageCompleted = {
        "type": "stage_completed",
        "stage": "test",
        "status": StageStatus.RAN,  # Valid
        "reason": "success",
        "duration_ms": 100.0,
        "index": 1,
        "total": 1,
    }
    # This assignment validates the type
    _status: CompletionType = event["status"]
    assert _status == StageStatus.RAN
```

**Step 2: Run type checker to verify it fails**

Run: `uv run basedpyright tests/engine/test_types_static.py`
Expected: Error because `StageStatus` is not assignable to `CompletionType`

**Step 3: Write minimal implementation**

In `src/pivot/engine/types.py`, update the import and `StageCompleted`:

```python
# Add to imports
from pivot.types import CompletionType

# Update StageCompleted class (around line 135)
class StageCompleted(TypedDict):
    """A stage finished (ran, skipped, or failed)."""

    type: Literal["stage_completed"]
    stage: str
    status: CompletionType  # Changed from StageStatus
    reason: str
    duration_ms: float
    index: int
    total: int
```

**Step 4: Run type checker to verify it passes**

Run: `uv run basedpyright tests/engine/test_types_static.py`
Expected: No errors

**Step 5: Run full type check**

Run: `uv run basedpyright .`
Expected: No new errors (fix any that appear from this change)

**Step 6: Commit**

```bash
jj describe -m "refactor(engine): use CompletionType for StageCompleted.status"
```

---

## Task 3: Update `ExecutionSummary` to Use `CompletionType`

**Files:**
- Modify: `src/pivot/executor/core.py:118-123`

**Step 1: Verify current state**

Run: `uv run basedpyright src/pivot/executor/core.py`
Expected: No errors (baseline)

**Step 2: Write minimal implementation**

In `src/pivot/executor/core.py`, update `ExecutionSummary`:

```python
# Add to imports
from pivot.types import CompletionType

# Update ExecutionSummary (around line 118)
class ExecutionSummary(TypedDict):
    """Summary result for a single stage after execution (returned by executor.run)."""

    status: CompletionType  # Changed from Literal[...] with UNKNOWN
    reason: str
```

**Step 3: Run type checker**

Run: `uv run basedpyright .`
Expected: No errors (or fix any that appear — UNKNOWN usage sites need updating)

**Step 4: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass

**Step 5: Commit**

```bash
jj describe -m "refactor(executor): use CompletionType for ExecutionSummary.status"
```

---

## Task 4: Update `StageResult` to Use `CompletionType`

**Files:**
- Modify: `src/pivot/types.py` (StageResult class)

**Step 1: Find and update StageResult**

Run: `grep -n "class StageResult" src/pivot/types.py`

Update the `status` field to use `CompletionType`.

**Step 2: Run type checker**

Run: `uv run basedpyright .`
Expected: No errors

**Step 3: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass

**Step 4: Commit**

```bash
jj describe -m "refactor(types): use CompletionType for StageResult.status"
```

---

## Task 5: Add `ResultCollectorSink`

**Files:**
- Modify: `src/pivot/engine/sinks.py`
- Test: `tests/engine/test_sinks.py` (new or existing)

**Step 1: Write the failing test**

```python
# tests/engine/test_sinks.py
"""Tests for engine sinks."""

import threading

from pivot.engine.sinks import ResultCollectorSink
from pivot.engine.types import StageCompleted
from pivot.types import StageStatus


def test_result_collector_collects_stage_completed() -> None:
    """ResultCollectorSink should collect StageCompleted events."""
    sink = ResultCollectorSink()

    event: StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "success",
        "duration_ms": 1500.0,
        "index": 1,
        "total": 2,
    }
    sink.handle(event)

    results = sink.get_results()
    assert "train" in results
    assert results["train"]["status"] == StageStatus.RAN


def test_result_collector_ignores_other_events() -> None:
    """ResultCollectorSink should ignore non-StageCompleted events."""
    sink = ResultCollectorSink()

    # StageStarted event (different type)
    event = {"type": "stage_started", "stage": "train", "index": 1, "total": 2}
    sink.handle(event)  # type: ignore[arg-type]

    results = sink.get_results()
    assert results == {}


def test_result_collector_overwrites_on_rerun() -> None:
    """ResultCollectorSink should overwrite results if stage runs again."""
    sink = ResultCollectorSink()

    event1: StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.FAILED,
        "reason": "error",
        "duration_ms": 100.0,
        "index": 1,
        "total": 1,
    }
    sink.handle(event1)

    event2: StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "success",
        "duration_ms": 200.0,
        "index": 1,
        "total": 1,
    }
    sink.handle(event2)

    results = sink.get_results()
    assert results["train"]["status"] == StageStatus.RAN


def test_result_collector_thread_safe() -> None:
    """ResultCollectorSink should be thread-safe."""
    sink = ResultCollectorSink()
    errors = []

    def add_events(stage_prefix: str) -> None:
        try:
            for i in range(100):
                event: StageCompleted = {
                    "type": "stage_completed",
                    "stage": f"{stage_prefix}_{i}",
                    "status": StageStatus.RAN,
                    "reason": "success",
                    "duration_ms": 1.0,
                    "index": 1,
                    "total": 1,
                }
                sink.handle(event)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_events, args=(f"t{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    results = sink.get_results()
    assert len(results) == 500  # 5 threads * 100 events each
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_sinks.py -v`
Expected: FAIL with `ImportError: cannot import name 'ResultCollectorSink'`

**Step 3: Write minimal implementation**

In `src/pivot/engine/sinks.py`, add:

```python
import threading

# ... existing code ...

class ResultCollectorSink:
    """Collects StageCompleted events for programmatic access to results."""

    def __init__(self) -> None:
        self._results: dict[str, StageCompleted] = {}
        self._lock = threading.Lock()

    def handle(self, event: OutputEvent) -> None:
        """Collect StageCompleted events, ignore others."""
        if event["type"] != "stage_completed":
            return
        with self._lock:
            self._results[event["stage"]] = event

    def get_results(self) -> dict[str, StageCompleted]:
        """Return collected results. Call after run() completes."""
        with self._lock:
            return dict(self._results)

    def close(self) -> None:
        """No-op for interface compatibility."""
        pass
```

Update `__all__` to include `"ResultCollectorSink"`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_sinks.py -v`
Expected: PASS

**Step 5: Run type checker**

Run: `uv run basedpyright src/pivot/engine/sinks.py`
Expected: No errors

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add ResultCollectorSink for collecting execution results"
```

---

## Task 6: Add `PipelineReloaded.stages` Field

**Files:**
- Modify: `src/pivot/engine/types.py:116-124`
- Test: `tests/engine/test_types_static.py`

**Step 1: Write the failing test**

```python
# Add to tests/engine/test_types_static.py
def test_pipeline_reloaded_has_stages_field() -> None:
    """PipelineReloaded should have a stages field with sorted stage list."""
    from pivot.engine.types import PipelineReloaded

    event: PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages": ["stage_a", "stage_b"],  # Topologically sorted
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": None,
    }
    assert event["stages"] == ["stage_a", "stage_b"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_types_static.py::test_pipeline_reloaded_has_stages_field -v`
Expected: FAIL with `TypedDict "PipelineReloaded" has no key "stages"`

**Step 3: Write minimal implementation**

In `src/pivot/engine/types.py`, update `PipelineReloaded`:

```python
class PipelineReloaded(TypedDict):
    """Registry was reloaded, DAG structure may have changed."""

    type: Literal["pipeline_reloaded"]
    stages: list[str]  # All stages in topological order
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_types_static.py::test_pipeline_reloaded_has_stages_field -v`
Expected: PASS

**Step 5: Run type checker to find emission sites**

Run: `uv run basedpyright .`
Expected: Errors at sites that emit `PipelineReloaded` without `stages` field

**Step 6: Update emission sites**

Find all places that create `PipelineReloaded` events and add the `stages` field with topologically sorted stage list. Use `get_stage_dag()` and `nx.topological_sort()`.

**Step 7: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass

**Step 8: Commit**

```bash
jj describe -m "feat(engine): add stages field to PipelineReloaded event"
```

---

## Task 7: Add Completion Tracking to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Test: `tests/engine/test_engine.py`

**Step 1: Write the failing test**

```python
# Add to tests/engine/test_engine.py
def test_engine_tracks_completion_stages(tmp_path: pathlib.Path) -> None:
    """Engine should track which stages were requested for completion."""
    # Setup minimal pipeline
    create_minimal_pipeline(tmp_path)  # Use existing test helper

    with engine.Engine() as eng:
        eng.add_source(sources.OneShotSource(stages=["train"], force=True, reason="test"))

        # Before run, no completion stages
        assert eng._completion_stages is None

        # This will be tested more fully when run() is implemented
```

**Step 2: Add completion tracking fields**

In `src/pivot/engine/engine.py`, add to `__init__`:

```python
# Completion tracking for exit_on_completion mode
self._completion_stages: set[str] | None = None
self._exit_on_completion: bool = False
```

**Step 3: Update `_handle_run_requested` to track stages**

When `exit_on_completion` is True, store the requested stages:

```python
def _handle_run_requested(self, event: RunRequested) -> None:
    # ... existing code ...
    if self._exit_on_completion and self._completion_stages is None:
        # Track stages from first RunRequested for completion detection
        self._completion_stages = set(stages_to_run)
```

**Step 4: Add completion check method**

```python
def _all_completion_stages_done(self) -> bool:
    """Check if all stages tracked for completion are done."""
    if self._completion_stages is None:
        return False
    with self._stage_states_lock:
        for stage in self._completion_stages:
            if stage not in self._stage_states:
                return False
            state = self._stage_states[stage]
            if state != StageExecutionState.COMPLETED:
                return False
    return True
```

**Step 5: Run tests**

Run: `uv run pytest tests/engine/test_engine.py -v -k completion`
Expected: Tests pass

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add completion stage tracking"
```

---

## Task 8: Implement Unified `run()` Method

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Test: `tests/engine/test_engine.py`

**Step 1: Write the failing test**

```python
# Add to tests/engine/test_engine.py
def test_engine_run_exits_on_completion(tmp_path: pathlib.Path) -> None:
    """Engine.run(exit_on_completion=True) should exit when stages complete."""
    create_minimal_pipeline(tmp_path)

    with engine.Engine() as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)
        eng.add_source(sources.OneShotSource(stages=None, force=True, reason="test"))

        eng.run(exit_on_completion=True)

        results = collector.get_results()
        assert len(results) > 0  # At least one stage ran


def test_engine_run_without_exit_on_completion_blocks(tmp_path: pathlib.Path) -> None:
    """Engine.run(exit_on_completion=False) should block until shutdown."""
    create_minimal_pipeline(tmp_path)

    with engine.Engine() as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)
        eng.add_source(sources.OneShotSource(stages=None, force=True, reason="test"))

        # Run in thread, call shutdown after brief delay
        import threading
        import time

        def delayed_shutdown() -> None:
            time.sleep(0.5)
            eng.shutdown()

        shutdown_thread = threading.Thread(target=delayed_shutdown)
        shutdown_thread.start()

        eng.run(exit_on_completion=False)  # Should block until shutdown

        shutdown_thread.join()
        results = collector.get_results()
        assert len(results) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_engine.py::test_engine_run_exits_on_completion -v`
Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'run'`

**Step 3: Implement `run()` method**

```python
def run(self, exit_on_completion: bool = False) -> None:
    """Run the event loop, processing events from sources.

    Args:
        exit_on_completion: If True, exit when all requested stages complete.
                           If False, run until shutdown() is called.
    """
    self._exit_on_completion = exit_on_completion
    self._completion_stages = None  # Reset, will be set by first RunRequested

    # Start all sources
    for source in self._sources:
        source.start(self.submit)

    try:
        while not self._shutdown_event.is_set():
            try:
                event = self._event_queue.get(timeout=0.1)
                self._handle_input_event(event)
            except queue.Empty:
                pass

            # Check completion condition
            if exit_on_completion and self._all_completion_stages_done():
                break
    finally:
        # Stop all sources
        for source in self._sources:
            source.stop()
```

**Step 4: Run tests**

Run: `uv run pytest tests/engine/test_engine.py -v -k "run_exits or run_without"`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add unified run() method with exit_on_completion"
```

---

## Task 9: Delete `run_once()` and Update Callers

**Files:**
- Modify: `src/pivot/engine/engine.py` (delete `run_once`)
- Modify: `src/pivot/executor/core.py` (update `run()`)
- Modify: `src/pivot/cli/run.py` (update CLI)
- Modify: `tests/engine/test_engine.py`
- Modify: `tests/engine/test_run_history.py`
- Modify: `tests/tui/test_run.py`

**Step 1: Find all `run_once` usages**

Run: `grep -rn "run_once" src/ tests/`

**Step 2: Update each caller**

For each caller, replace:
```python
results = eng.run_once(stages=..., force=..., ...)
```

With:
```python
collector = ResultCollectorSink()
eng.add_sink(collector)
eng.add_source(OneShotSource(stages=..., force=..., reason="..."))
eng.run(exit_on_completion=True)
results = collector.get_results()
```

**Step 3: Delete `run_once` from Engine**

Remove the entire `run_once` method from `src/pivot/engine/engine.py`.

**Step 4: Run type checker**

Run: `uv run basedpyright .`
Expected: No errors (all usages updated)

**Step 5: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass (may need fixes)

**Step 6: Commit**

```bash
jj describe -m "refactor(engine): delete run_once, migrate callers to run()"
```

---

## Task 10: Refactor CLI to Single `_run_pipeline()` Function

**Files:**
- Modify: `src/pivot/cli/run.py`
- Test: Existing CLI tests + manual verification

**Step 1: Create shared helpers**

```python
def _configure_sinks(
    eng: engine.Engine,
    quiet: bool,
    tui: bool,
    console: Console | None,
    tui_queue: TuiQueue | None,
    jsonl_callback: Callable[[RunJsonEvent], None] | None,
) -> ResultCollectorSink:
    """Configure output sinks for the engine. Returns the result collector."""
    collector = sinks.ResultCollectorSink()
    eng.add_sink(collector)

    if jsonl_callback:
        eng.add_sink(sinks.JsonlSink(jsonl_callback))
    elif not quiet:
        if tui and tui_queue:
            eng.add_sink(sinks.TuiSink(tui_queue))
        elif console:
            eng.add_sink(sinks.ConsoleSink(console))

    return collector


def _configure_watch_sources(
    eng: engine.Engine,
    watch_paths: list[pathlib.Path],
    force: bool,
    stages: list[str] | None,
    debounce: int,
) -> None:
    """Configure sources for watch mode."""
    eng.add_source(sources.FilesystemSource(watch_paths, debounce=debounce))
    if force:
        eng.add_source(sources.OneShotSource(stages=stages, force=True, reason="watch:initial"))


def _configure_oneshot_source(
    eng: engine.Engine,
    stages: list[str] | None,
    force: bool,
) -> None:
    """Configure source for non-watch mode."""
    eng.add_source(sources.OneShotSource(stages=stages, force=force, reason="cli"))
```

**Step 2: Create unified `_run_pipeline()`**

```python
def _run_pipeline(
    stages: list[str] | None,
    watch: bool,
    force: bool,
    quiet: bool,
    tui: bool,
    # ... other params
) -> int:
    """Run pipeline with unified watch/non-watch execution."""
    with engine.Engine() as eng:
        collector = _configure_sinks(eng, quiet=quiet, tui=tui, ...)

        if watch:
            _configure_watch_sources(eng, force=force, ...)
        else:
            _configure_oneshot_source(eng, stages=stages, force=force)

        eng.run(exit_on_completion=not watch)

        results = collector.get_results()
        return _compute_exit_code(results)
```

**Step 3: Update Click command to use `_run_pipeline()`**

Replace the existing watch/non-watch branching with a call to `_run_pipeline()`.

**Step 4: Delete old functions**

Remove `_run_watch_plain()`, `_run_watch_tui()`, and other duplicated functions that are no longer needed.

**Step 5: Run tests**

Run: `uv run pytest tests/cli/ -v`
Expected: All tests pass

**Step 6: Manual verification**

```bash
# Test non-watch mode
uv run pivot run --quiet
uv run pivot run

# Test watch mode
uv run pivot run --watch --quiet  # Should now work!
uv run pivot run --watch
```

**Step 7: Commit**

```bash
jj describe -m "refactor(cli): unify watch/non-watch execution paths"
```

---

## Task 11: Add Integration Tests for Bug Fixes

**Files:**
- Create: `tests/integration/test_unified_execution.py`

**Step 1: Write test for `--quiet` in watch mode (original bug #1)**

```python
# tests/integration/test_unified_execution.py
"""Integration tests for unified watch/non-watch execution."""

import pathlib
import subprocess
import time


def test_quiet_flag_works_in_watch_mode(tmp_path: pathlib.Path) -> None:
    """The --quiet flag should suppress output in watch mode (bug fix)."""
    # Create minimal pipeline
    (tmp_path / "pivot.yaml").write_text("""
stages:
  - name: hello
    func: pipeline:hello
""")
    (tmp_path / "pipeline.py").write_text("""
def hello() -> None:
    print("Hello!")
""")

    # Run watch mode with --quiet, force initial run, timeout after 2s
    proc = subprocess.Popen(
        ["uv", "run", "pivot", "run", "--watch", "--quiet", "--force"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    proc.terminate()
    stdout, stderr = proc.communicate()

    # With --quiet, should have minimal/no output
    assert b"Hello!" not in stdout  # Stage output suppressed
    assert b"Running" not in stdout  # Progress suppressed


def test_stage_list_updates_on_pipeline_reload(tmp_path: pathlib.Path) -> None:
    """Stage list should update when pipeline is modified in watch mode (bug fix)."""
    # This test verifies the PipelineReloaded event includes updated stages
    # Implementation depends on TUI internals - may need adjustment
    pass  # TODO: Implement based on actual TUI test patterns
```

**Step 2: Run tests**

Run: `uv run pytest tests/integration/test_unified_execution.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj describe -m "test: add integration tests for unified execution bug fixes"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `docs/architecture/engine.md`
- Modify: `docs/architecture/watch.md`

**Step 1: Update engine.md**

Document the unified execution model:
- `run(exit_on_completion=True)` for single runs
- `run(exit_on_completion=False)` for watch mode
- `ResultCollectorSink` for getting results

**Step 2: Update watch.md**

Explain that watch and non-watch share the same execution path, with the difference being:
- Sources: `OneShotSource` only vs `FilesystemSource` + optional `OneShotSource`
- Exit behavior: `exit_on_completion=True` vs `False`

**Step 3: Commit**

```bash
jj describe -m "docs: update architecture docs for unified execution"
```

---

## Task 13: Final Verification

**Step 1: Run full quality checks**

```bash
uv run ruff format .
uv run ruff check .
uv run basedpyright .
uv run pytest tests/ -n auto
```

**Step 2: Verify both original bugs are fixed**

1. `--quiet` in watch mode: `uv run pivot run --watch --quiet --force`
2. Stage list on reload: Modify `pivot.yaml` while watch mode is running, verify TUI updates

**Step 3: Final commit if any cleanup needed**

```bash
jj describe -m "chore: final cleanup for unified execution"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Add `CompletionType` alias | `src/pivot/types.py` |
| 2 | Update `StageCompleted` | `src/pivot/engine/types.py` |
| 3 | Update `ExecutionSummary` | `src/pivot/executor/core.py` |
| 4 | Update `StageResult` | `src/pivot/types.py` |
| 5 | Add `ResultCollectorSink` | `src/pivot/engine/sinks.py` |
| 6 | Add `PipelineReloaded.stages` | `src/pivot/engine/types.py` |
| 7 | Add completion tracking | `src/pivot/engine/engine.py` |
| 8 | Implement `run()` method | `src/pivot/engine/engine.py` |
| 9 | Delete `run_once()` | Multiple files |
| 10 | Refactor CLI | `src/pivot/cli/run.py` |
| 11 | Integration tests | `tests/integration/` |
| 12 | Update docs | `docs/architecture/` |
| 13 | Final verification | All |
