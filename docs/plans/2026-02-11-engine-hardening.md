# Engine Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the Engine↔Scheduler boundary, fix deferred-event semantics, and extract watch-mode policy into a dedicated coordinator.

**Architecture:** Three improvements to the engine subsystem: (1) add Scheduler invariants so programming errors fail loud instead of silently clamping, replace test helpers that poke private attributes with `initialize()`, (2) fix `_process_deferred_events` to drain until empty with a max-iterations guard that emits a diagnostic event on trip, (3) extract a `WatchCoordinator` that owns watch-mode **policy and planning** (affected-stage computation, path filtering, worker restart decisions) while Engine retains **execution and mutation** (run state machine, pipeline reload mechanics, event emission).

**Tech Stack:** Python 3.13+, pytest, anyio, NetworkX

**Issues:** #418, #419, #420 (umbrella: #414)

---

### Design Decisions (from gap analysis + Oracle consultation)

**#419 Guard behavior:** The guard must NOT raise. `_process_deferred_events` is called from `_handle_stage_completion`, which lives inside `_orchestrate_execution`'s `try/except Exception` block. An exception there would be treated as a stage failure, causing double `_handle_stage_completion` calls and corrupting scheduler state (double `release_mutexes`, double `on_stage_completed`). If the exception escapes further, it kills the entire anyio task group and terminates watch mode. Instead: log ERROR + emit a diagnostic `EngineDiagnostic` OutputEvent + drop remaining events. Watch mode continues; user gets notified.

**#420 Extraction boundary:** The coordinator owns **policy + planning** — it decides *what* should happen (which stages are affected, should this path be filtered, should workers restart). Engine owns **execution + mutation** — it performs the actual state changes (reload pipeline, request runs, emit events). The coordinator calls Engine via a small callback interface. This is a middle ground: more useful than pure graph queries, but doesn't try to move the deeply-coupled reload/run-state-machine logic. Registry reload, `sys.modules` invalidation, and rerun orchestration stay in Engine.

**#418 Test rewrite ordering:** Task 2 (public-API rewrite) MUST happen before Tasks 3–4 (new invariants). Otherwise the new invariants cause cascading failures in tests that still use private field mutation.

**Existing Engine tests after coordinator wiring:** Engine's `_should_filter_path` must lazily create the coordinator from `self._graph` when called (not only when `_graph` is explicitly set via `_orchestrate_execution`). This prevents existing tests that set `engine._graph = g` directly from breaking.

**Test style rules (from `tests/AGENTS.md`):**
- Use `monkeypatch.setattr(...)` not direct assignment for mocks
- Module-level `_helper_*` functions, not inline lambdas in tests
- No `--timeout` flag (pytest-timeout not confirmed in deps)
- Assert observable outcomes, not internal call counts
- `autospec=True` when mocking functions/methods

---

## Task 1: Audit Engine→Scheduler Private Field Access (#418 scope check)

PR #434 already removed delegation property pairs. Verify no Engine code writes to scheduler private fields.

**Files:**
- Read: `packages/pivot/src/pivot/engine/engine.py`
- Read: `packages/pivot/tests/engine/test_scheduler_characterization.py`

**Step 1: Grep for Engine→Scheduler private access**

Run:
```bash
grep -n '_scheduler\._' packages/pivot/src/pivot/engine/engine.py
```
Expected: No matches (already clean from PR #434).

**Step 2: Grep for test private field access on Scheduler**

Run:
```bash
grep -n '\._stage_states\|\._upstream_unfinished\|\._downstream\|\._stage_mutex\|\._mutex_counts\|\._stop_starting_new' packages/pivot/tests/engine/test_scheduler_characterization.py
```
Expected: Multiple matches in `_helper_make_scheduler` and inline test mutations. These will be replaced in Task 2.

**Step 3: Done — no commit, just verification**

---

## Task 2: Replace `_helper_make_scheduler` with `initialize()` (#418)

The characterization tests bypass the public API by directly setting `scheduler._stage_states` etc. Replace with a helper that calls `Scheduler.initialize()` and drives state through public methods.

**Files:**
- Modify: `packages/pivot/tests/engine/test_scheduler_characterization.py`

**Step 1: Write new helper `_helper_init_scheduler` using public API**

Replace `_helper_make_scheduler` with:

```python
import networkx as nx

from pivot.engine import graph as engine_graph
from pivot.engine.types import NodeType


def _helper_init_scheduler(
    *,
    execution_order: list[str],
    edges: list[tuple[str, str]] | None = None,
    stage_mutex: dict[str, list[str]] | None = None,
) -> Scheduler:
    """Create a Scheduler using only the public API.

    Args:
        execution_order: Stage names in topological order.
        edges: (upstream, downstream) stage dependency pairs.
        stage_mutex: Per-stage mutex list. Defaults to empty list per stage.
    """
    if stage_mutex is None:
        stage_mutex = {name: list[str]() for name in execution_order}

    if edges:
        # Build a minimal bipartite graph so Scheduler.initialize() can derive
        # upstream/downstream from it.  Each edge A->B is modeled as:
        #   stage:A -> artifact:A__B -> stage:B
        g: nx.DiGraph[str] = nx.DiGraph()
        for name in execution_order:
            g.add_node(engine_graph.stage_node(name), type=NodeType.STAGE)
        for src, dst in edges:
            art = f"artifact:{src}__{dst}"
            g.add_node(art, type=NodeType.ARTIFACT)
            g.add_edge(engine_graph.stage_node(src), art)
            g.add_edge(art, engine_graph.stage_node(dst))
    else:
        g = None  # type: ignore[assignment]

    scheduler = Scheduler()
    scheduler.initialize(execution_order, g, stage_mutex=stage_mutex)
    return scheduler
```

Delete `_helper_make_scheduler` entirely.

**Step 2: Update `_helper_startable_in_order` to use public property**

```python
def _helper_startable_in_order(scheduler: Scheduler, running_count: int) -> list[str]:
    startable: list[str] = []
    for name in list(scheduler.stage_states.keys()):
        if scheduler.can_start(name, running_count=running_count):
            startable.append(name)
    return startable
```

**Step 3: Rewrite each test to use `_helper_init_scheduler` and drive state via public methods**

Key patterns for reaching needed states:
- **READY with no upstream:** `_helper_init_scheduler(execution_order=["stage"])` → stage starts READY
- **PENDING with upstream:** `edges=[("A", "B")]` → B starts PENDING
- **RUNNING:** call `scheduler.set_state("stage", StageExecutionState.RUNNING)` (public)
- **COMPLETED:** call `scheduler.set_state("stage", StageExecutionState.COMPLETED)` (public)
- **BLOCKED:** complete upstream with `failed=True` via `scheduler.on_stage_completed()`
- **Held mutex:** call `scheduler.acquire_mutexes("holder_stage")` (public)
- **Released mutex:** call `scheduler.release_mutexes("holder_stage")` (public, after acquire)

Test-by-test conversion guidance:

- `test_can_start_requires_ready_and_no_upstream`: Use `edges=[("upstream", "stage")]`. Stage starts PENDING. Complete upstream via `set_state` + `on_stage_completed`. Then verify stage becomes startable.

- `test_can_start_respects_named_mutex`: Use two stages with same mutex. Acquire mutex for one via `acquire_mutexes`. Verify the other can't start. Release. Verify it can.

- `test_can_start_respects_exclusive_mutex_and_running`: Use `stage_mutex={"exclusive": [EXCLUSIVE_MUTEX], "normal": []}`. Test `running_count` parameter. For the "normal blocked by exclusive" case, call `acquire_mutexes("exclusive")` to hold the exclusive mutex.

- `test_ready_selection_respects_stage_state_insertion_order`: Independent stages (no edges). All start READY. Verify iteration order matches execution_order.

- `test_chain_updates_downstream_after_completion`: Use `edges=[("A","B"),("B","C")]`.

- `test_fan_out_sets_ready_after_upstream_completion`: Use `edges=[("A","B"),("A","C")]`.

- `test_fan_in_waits_for_all_upstream`: Use `edges=[("A","C"),("B","C")]`.

- `test_on_stage_completed_cascades_failure`: Use `edges=[("A","B")]` with `stage_mutex`. Call `acquire_mutexes("A")` before `release_mutexes("A")` to avoid underflow.

- `test_apply_fail_fast_blocks_ready_and_pending`: Create 4 stages: "failed" (independent, no edges), "ready" (independent), "pending" (has upstream via edge), "blocked" (set up as downstream of failed — but since apply_fail_fast is called *after* failure cascade, set up so "ready"/"pending" are in their natural states, then call `apply_fail_fast`). Drive "failed" to COMPLETED first.

- `test_apply_cancel_marks_ready_and_pending_completed`: Create 3 stages. Drive "running" to RUNNING via `set_state`. Leave "ready" and "pending" in natural states.

- `test_state_enum_comparison_for_monotonicity_guard`: Drive a stage to COMPLETED via `set_state`. No scheduler construction changes needed beyond using `_helper_init_scheduler`.

**Step 4: Run tests to verify**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_scheduler_characterization.py -v
```
Expected: All tests PASS.

**Step 5: Verify no private access remains**

Run:
```bash
grep -rn 'scheduler\._' packages/pivot/tests/engine/test_scheduler_characterization.py
```
Expected: No matches.

**Step 6: Commit**

Message: "refactor: replace scheduler test helper with public API (initialize)"

---

## Task 3: Fail-loud mutex underflow in Scheduler (#418)

Currently `release_mutexes` silently clamps negative counts to 0. In pre-alpha this should raise.

**Prerequisite:** Task 2 must be complete (tests no longer rely on synthetic `_mutex_counts`).

**Files:**
- Modify: `packages/pivot/src/pivot/engine/scheduler.py:152-157`
- Test: `packages/pivot/tests/engine/test_scheduler_characterization.py`

**Step 1: Write failing test for mutex underflow**

Add to `test_scheduler_characterization.py`:

```python
def test_release_mutexes_raises_on_underflow() -> None:
    """Releasing a mutex that was never acquired should raise ValueError."""
    scheduler = _helper_init_scheduler(
        execution_order=["stage"],
        stage_mutex={"stage": ["my_mutex"]},
    )
    # Release without acquire — should fail loud
    with pytest.raises(ValueError, match="Mutex.*released when not held"):
        scheduler.release_mutexes("stage")
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_scheduler_characterization.py::test_release_mutexes_raises_on_underflow -v
```
Expected: FAIL — currently logs error and clamps to 0.

**Step 3: Make mutex underflow raise ValueError**

In `packages/pivot/src/pivot/engine/scheduler.py`, replace the silent clamp:

```python
    def release_mutexes(self, stage: str) -> None:
        for mutex in self._stage_mutex.get(stage, []):
            self._mutex_counts[mutex] -= 1
            if self._mutex_counts[mutex] < 0:
                _logger.error("Mutex '%s' released when not held", mutex)
                self._mutex_counts[mutex] = 0
```

With:

```python
    def release_mutexes(self, stage: str) -> None:
        for mutex in self._stage_mutex.get(stage, []):
            if self._mutex_counts[mutex] <= 0:
                msg = f"Mutex '{mutex}' released when not held (stage '{stage}')"
                raise ValueError(msg)
            self._mutex_counts[mutex] -= 1
```

**Safety note:** In Engine, `acquire_mutexes` happens immediately before submission (`engine.py:1296-1297`) and `release_mutexes` in `_handle_stage_completion` for that same stage (`engine.py:1370-1371`). No other Engine call sites exist, so a correct Engine run will never underflow. This only catches programmer errors.

**Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_scheduler_characterization.py::test_release_mutexes_raises_on_underflow -v
```
Expected: PASS.

**Step 5: Run full scheduler test suite**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_scheduler_characterization.py -v
```
Expected: All PASS. If any test fails because it calls `release_mutexes` before `acquire_mutexes`, fix it by adding the acquire call first — this is a test bug that was previously hidden by the silent clamp.

**Step 6: Commit**

Message: "fix: fail loud on mutex underflow in Scheduler"

---

## Task 4: Validate `initialize()` inputs (#418)

Add cheap validation: execution_order stages must match stage_mutex keys.

**Files:**
- Modify: `packages/pivot/src/pivot/engine/scheduler.py:72-114`
- Test: `packages/pivot/tests/engine/test_scheduler_characterization.py`

**Step 1: Write failing test**

```python
def test_initialize_validates_stage_mutex_consistency() -> None:
    """initialize() raises if stage_mutex keys don't match execution_order."""
    scheduler = Scheduler()
    with pytest.raises(ValueError, match="stage_mutex"):
        scheduler.initialize(
            execution_order=["A", "B"],
            graph=None,
            stage_mutex={"A": [], "C": []},  # Missing B, has extra C
        )
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_scheduler_characterization.py::test_initialize_validates_stage_mutex_consistency -v
```
Expected: FAIL — no validation currently.

**Step 3: Add validation to `initialize()`**

At the top of `Scheduler.initialize()`, after `stages_set = set(execution_order)`:

```python
        mutex_keys = set(stage_mutex.keys())
        missing = stages_set - mutex_keys
        extra = mutex_keys - stages_set
        if missing or extra:
            parts = list[str]()
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"unknown: {sorted(extra)}")
            msg = f"stage_mutex inconsistency — {', '.join(parts)}"
            raise ValueError(msg)
```

**Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_scheduler_characterization.py::test_initialize_validates_stage_mutex_consistency -v
```
Expected: PASS.

**Step 5: Run full test suite to check nothing breaks**

Run:
```bash
uv run pytest packages/pivot/tests/engine/ -v
```
Expected: All PASS.

**Step 6: Commit**

Message: "feat: validate initialize() inputs in Scheduler"

---

## Task 5: Document determinism tie-breaker (#418)

Add docstring to `initialize()` documenting the tie-breaker and contract.

**Files:**
- Modify: `packages/pivot/src/pivot/engine/scheduler.py`

**Step 1: Add docstring to `initialize()`**

```python
    def initialize(
        self,
        execution_order: list[str],
        graph: nx.DiGraph[str] | None,
        *,
        stage_mutex: dict[str, list[str]],
    ) -> None:
        """Reset and configure scheduler for a new execution.

        Args:
            execution_order: Stage names in topological order. This order
                determines the determinism tie-breaker: when multiple stages
                are eligible to start simultaneously, they are considered in
                ``execution_order`` sequence (i.e., dict insertion order of
                ``_stage_states``).
            graph: Bipartite artifact-stage graph for deriving upstream/downstream
                relationships. Pass None for single-stage or no-dependency runs.
            stage_mutex: Mapping of stage name to mutex group names. Must contain
                exactly the same keys as ``execution_order``.

        Raises:
            ValueError: If ``stage_mutex`` keys don't match ``execution_order``.
        """
```

**Step 2: Commit**

Message: "docs: document Scheduler determinism tie-breaker and initialize() contract"

---

## Task 6: Add `EngineDiagnostic` output event type (#419 prerequisite)

The deferred-event guard needs to emit a diagnostic event. Add the type first.

**Files:**
- Modify: `packages/pivot/src/pivot/engine/types.py`

**Step 1: Add `EngineDiagnostic` TypedDict**

Add after `SinkStateChanged` and before the `OutputEvent` union:

```python
class EngineDiagnostic(TypedDict):
    """Engine-level diagnostic for non-fatal operational issues.

    Emitted when the engine detects an anomaly that doesn't warrant stopping
    execution but should be visible to operators (e.g., deferred event loop
    guard tripped, unexpected state transitions).
    """

    type: Literal["engine_diagnostic"]
    seq: NotRequired[int]
    run_id: NotRequired[str]
    message: str
    detail: str
```

Update the `OutputEvent` union to include it:

```python
OutputEvent = (
    EngineStateChanged
    | PipelineReloaded
    | StageStarted
    | StageCompleted
    | StageStateChanged
    | LogLine
    | SinkStateChanged
    | EngineDiagnostic
)
```

Update `__all__` to include `"EngineDiagnostic"`.

**Step 2: Run type checker to verify**

Run:
```bash
uv run basedpyright packages/pivot/src/pivot/engine/types.py
```
Expected: Clean.

**Step 3: Commit**

Message: "feat: add EngineDiagnostic output event type"

---

## Task 7: Fix deferred-event drain semantics (#419)

`_process_deferred_events` pops a list once. If `_handle_input_event` defers new events for the same stage, they're lost. Fix with a while-loop and max-iterations guard.

**Design decision:** Guard trips emit `EngineDiagnostic` + drop remaining events (never raise). See "Design Decisions" section at top for rationale.

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py:1653-1664`
- Test: `packages/pivot/tests/engine/test_engine.py`

**Step 1: Write failing test — events deferred during processing are not lost**

Add a module-level helper and test to `test_engine.py`. Import `InputEvent` from types.

```python
_deferred_event_call_log: list[str] = []


async def _helper_tracking_handle_with_redeferral(
    engine: Engine, event: InputEvent
) -> None:
    """Module-level handler that defers one additional event on first call."""
    _deferred_event_call_log.append(event["type"])
    if len(_deferred_event_call_log) == 1:
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["nested.csv"]),
        )


@pytest.mark.anyio
async def test_deferred_events_during_processing_are_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If handling a deferred event defers another for the same stage, both are processed."""
    _deferred_event_call_log.clear()

    async with Engine() as engine:
        monkeypatch.setattr(
            engine,
            "_handle_input_event",
            lambda event: _helper_tracking_handle_with_redeferral(engine, event),
        )

        # Seed one deferred event
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["initial.csv"]),
        )

        await engine._process_deferred_events("stage_a")

        assert len(_deferred_event_call_log) == 2, (
            f"Expected 2 events processed, got {len(_deferred_event_call_log)}"
        )
        assert "stage_a" not in engine._deferred_events, "No leftover deferred events"
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_engine.py::test_deferred_events_during_processing_are_not_lost -v
```
Expected: FAIL — only 1 event processed, second is lost.

**Step 3: Write test for max-iterations guard — asserts diagnostic event emitted**

```python
_infinite_defer_count: int = 0


async def _helper_infinite_defer_handle(engine: Engine, event: InputEvent) -> None:
    """Module-level handler that always defers another event — simulates infinite loop."""
    global _infinite_defer_count
    _infinite_defer_count += 1
    engine._defer_event_for_stage(
        "stage_a",
        DataArtifactChanged(type="data_artifact_changed", paths=["loop.csv"]),
    )


@pytest.mark.anyio
async def test_deferred_events_max_iterations_guard_emits_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infinite deferral loop is caught by guard; diagnostic event emitted, events dropped."""
    global _infinite_defer_count
    _infinite_defer_count = 0

    sink = _MockAsyncSink()

    async with Engine() as engine:
        engine.add_sink(sink)

        monkeypatch.setattr(
            engine,
            "_handle_input_event",
            lambda event: _helper_infinite_defer_handle(engine, event),
        )

        # Seed one event
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["seed.csv"]),
        )

        # Should not hang — guard trips
        await engine._process_deferred_events("stage_a")

        # Guard stopped iteration at max
        assert _infinite_defer_count == engine._DEFERRED_MAX_ITERATIONS, (
            f"Expected {engine._DEFERRED_MAX_ITERATIONS} iterations, got {_infinite_defer_count}"
        )

        # Diagnostic event emitted (via emit() which goes through output channel)
        diagnostics = [e for e in sink.events if e["type"] == "engine_diagnostic"]
        assert len(diagnostics) == 1, f"Expected 1 diagnostic event, got {len(diagnostics)}"
        diag = diagnostics[0]
        assert "stage_a" in diag["message"], "Diagnostic should name the stage"
        assert "stage_a" not in engine._deferred_events, "Remaining events should be dropped"
```

**Note:** The `_MockAsyncSink` may not receive events from `emit()` without the full dispatch pipeline running. If this is the case, the implementer should instead check via a `monkeypatch` on `engine.emit` that captures the diagnostic event, or use a simpler assertion (check log output via `caplog`). Adapt the test to what works with Engine's internal plumbing.

**Step 4: Implement while-loop with max-iterations guard and diagnostic emission**

Replace the current `_process_deferred_events` in `engine.py`. Also add `EngineDiagnostic` to the imports from `pivot.engine.types`.

```python
    _DEFERRED_MAX_ITERATIONS: int = 100

    async def _process_deferred_events(self, stage: str) -> None:
        """Process deferred events for a completed stage, draining until empty.

        If event handlers defer new events for the same stage during processing,
        those are picked up in subsequent iterations. A max-iterations guard
        prevents infinite loops — on trip, remaining events are dropped and a
        diagnostic event is emitted.
        """
        for _ in range(self._DEFERRED_MAX_ITERATIONS):
            events = self._deferred_events.pop(stage, [])
            if not events:
                return
            for event in events:
                try:
                    await self._handle_input_event(event)
                except Exception:
                    _logger.exception(
                        "Error processing deferred event for stage %s: %s", stage, event
                    )

        # Guard tripped — drop remaining events and emit diagnostic
        remaining = self._deferred_events.pop(stage, [])
        remaining_count = len(remaining)
        message = (
            f"Deferred event loop for stage '{stage}' hit max iterations "
            f"({self._DEFERRED_MAX_ITERATIONS})"
        )
        detail = f"Dropped {remaining_count} remaining event(s)" if remaining_count else ""
        _logger.error("%s. %s", message, detail)
        await self.emit(
            EngineDiagnostic(
                type="engine_diagnostic",
                message=message,
                detail=detail,
            )
        )
```

**Step 5: Run deferred event tests**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_engine.py -k "deferred_events" -v
```
Expected: All deferred event tests PASS (including pre-existing ones).

**Step 6: Run full engine tests**

Run:
```bash
uv run pytest packages/pivot/tests/engine/ -v
```
Expected: All PASS.

**Step 7: Commit**

Message: "fix: drain deferred events until empty with max-iterations guard (#419)"

---

## Task 8: Create `WatchCoordinator` with policy + planning (#420)

Extract watch-mode policy into a coordinator that owns affected-stage computation, path filtering, and worker restart decisions. Engine retains execution and mutation (run state machine, pipeline reload mechanics, event emission).

**Architecture:**

```
WatchCoordinator (policy brain)         Engine (execution + mutation)
├── on_data_changed(paths)         →    host.request_run(affected)
│   - filter executing outputs          host.get_stage_state(stage)
│   - compute affected stages           host.defer_event(stage, event)
│   - return affected list
├── on_code_changed(paths)         →    host.reload_pipeline(paths) -> ok?
│   - decide: reload? restart?          host.restart_workers(n, max)
│   - decide: which stages?             host.request_run(stages)
└── should_filter_path(path)            host.get_stage_state(stage)
```

**Files:**
- Create: `packages/pivot/src/pivot/engine/watch.py`
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Create: `packages/pivot/tests/engine/test_watch.py`

### Step 1: Write WatchCoordinator unit tests

Create `packages/pivot/tests/engine/test_watch.py`:

```python
"""Tests for the WatchCoordinator."""

from __future__ import annotations

import pathlib

import networkx as nx

from pivot.engine import graph as engine_graph
from pivot.engine.types import NodeType, StageExecutionState
from pivot.engine.watch import WatchCoordinator


def _helper_build_graph(
    stages: dict[str, dict[str, list[str]]],
) -> nx.DiGraph[str]:
    """Build a minimal bipartite graph for testing.

    Args:
        stages: Mapping of stage_name -> {"deps": [paths], "outs": [paths]}
    """
    g: nx.DiGraph[str] = nx.DiGraph()
    for name, info in stages.items():
        stage = engine_graph.stage_node(name)
        g.add_node(stage, type=NodeType.STAGE)
        for dep in info.get("deps", []):
            art = engine_graph.artifact_node(pathlib.Path(dep))
            g.add_node(art, type=NodeType.ARTIFACT)
            g.add_edge(art, stage)
        for out in info.get("outs", []):
            art = engine_graph.artifact_node(pathlib.Path(out))
            g.add_node(art, type=NodeType.ARTIFACT)
            g.add_edge(stage, art)
    return g


def _helper_state_map(
    state: StageExecutionState,
) -> dict[str, StageExecutionState]:
    """Create a defaultdict-like lookup that returns a fixed state for any stage."""

    class _FixedState(dict[str, StageExecutionState]):
        def __missing__(self, key: str) -> StageExecutionState:
            return state

    return _FixedState()


# =============================================================================
# Affected Stage Computation
# =============================================================================


def test_affected_stages_returns_consumers_and_downstream() -> None:
    """Changed input affects its consumer and all transitive downstream stages."""
    g = _helper_build_graph({
        "extract": {"deps": ["/data/raw.csv"], "outs": ["/data/clean.csv"]},
        "train": {"deps": ["/data/clean.csv"], "outs": ["/models/model.pkl"]},
        "evaluate": {"deps": ["/models/model.pkl"], "outs": ["/results/metrics.json"]},
    })
    coord = WatchCoordinator(graph=g)

    affected = coord.get_affected_stages([pathlib.Path("/data/raw.csv")])
    assert set(affected) == {"extract", "train", "evaluate"}, (
        "all downstream stages should be affected"
    )


def test_affected_stages_unknown_path_returns_empty() -> None:
    """Path not in graph returns no affected stages."""
    g = _helper_build_graph({
        "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
    })
    coord = WatchCoordinator(graph=g)

    assert coord.get_affected_stages([pathlib.Path("/unknown.csv")]) == []


def test_affected_stages_deduplicates() -> None:
    """Multiple paths affecting the same stage are deduplicated."""
    g = _helper_build_graph({
        "stage_a": {"deps": ["/input1.csv", "/input2.csv"], "outs": ["/output.csv"]},
    })
    coord = WatchCoordinator(graph=g)

    affected = coord.get_affected_stages([
        pathlib.Path("/input1.csv"),
        pathlib.Path("/input2.csv"),
    ])
    assert affected == ["stage_a"], "should deduplicate"


# =============================================================================
# Path Filtering
# =============================================================================


def test_should_filter_path_true_for_preparing_and_running_producer() -> None:
    """Paths produced by a PREPARING, WAITING_ON_LOCK, or RUNNING stage are filtered."""
    g = _helper_build_graph({
        "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
    })
    coord = WatchCoordinator(graph=g)
    output = pathlib.Path("/output.csv")

    for state in (
        StageExecutionState.PREPARING,
        StageExecutionState.WAITING_ON_LOCK,
        StageExecutionState.RUNNING,
    ):
        state_map = _helper_state_map(state)
        assert coord.should_filter_path(output, get_stage_state=state_map.__getitem__) is True, (
            f"should filter when producer is {state.name}"
        )


def test_should_filter_path_false_for_non_executing_producer() -> None:
    """Paths produced by PENDING, BLOCKED, READY, or COMPLETED stages are NOT filtered."""
    g = _helper_build_graph({
        "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
    })
    coord = WatchCoordinator(graph=g)
    output = pathlib.Path("/output.csv")

    for state in (
        StageExecutionState.PENDING,
        StageExecutionState.BLOCKED,
        StageExecutionState.READY,
        StageExecutionState.COMPLETED,
    ):
        state_map = _helper_state_map(state)
        assert coord.should_filter_path(output, get_stage_state=state_map.__getitem__) is False, (
            f"should NOT filter when producer is {state.name}"
        )


def test_should_filter_path_false_for_input_artifact() -> None:
    """Input artifacts (no producer) should never be filtered."""
    g = _helper_build_graph({
        "stage_a": {"deps": ["/input.csv"], "outs": ["/output.csv"]},
    })
    coord = WatchCoordinator(graph=g)

    state_map = _helper_state_map(StageExecutionState.RUNNING)
    assert coord.should_filter_path(
        pathlib.Path("/input.csv"), get_stage_state=state_map.__getitem__
    ) is False


# =============================================================================
# Worker Restart Policy
# =============================================================================


def test_should_restart_workers_true_when_parallel() -> None:
    """Worker restart recommended when parallel mode is enabled."""
    coord = WatchCoordinator(graph=nx.DiGraph())
    assert coord.should_restart_workers(parallel=True) is True


def test_should_restart_workers_false_when_not_parallel() -> None:
    """Worker restart not recommended in sequential mode."""
    coord = WatchCoordinator(graph=nx.DiGraph())
    assert coord.should_restart_workers(parallel=False) is False
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_watch.py -v
```
Expected: FAIL — `WatchCoordinator` doesn't exist yet.

### Step 3: Implement WatchCoordinator

Create `packages/pivot/src/pivot/engine/watch.py`:

```python
"""Watch-mode coordinator: owns policy and planning for watch-triggered actions.

The WatchCoordinator decides *what* should happen in response to file changes.
Engine performs the actual *execution* (state mutations, event emission, run lifecycle).

Responsibilities owned by WatchCoordinator:
- Affected-stage computation (which stages to run after a file change)
- Path filtering (should events for this path be deferred/ignored?)
- Worker restart policy (should workers restart after code reload?)

Responsibilities retained by Engine:
- Pipeline reload mechanics (sys.modules, discovery, fingerprint caches)
- Run state machine (cancel/coalesce, run_id generation, task groups)
- Event emission (OutputEvents to sinks)
- Deferred event storage/processing (tied to execution lifecycle)
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Callable

from pivot.engine import graph as engine_graph
from pivot.engine.types import StageExecutionState

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["WatchCoordinator"]


class WatchCoordinator:
    """Policy and planning coordinator for watch-mode file change handling.

    Stateless with respect to execution — all execution state is accessed
    via callbacks (get_stage_state) rather than owned directly. This enables
    unit testing with synthetic graphs and state maps.
    """

    _graph: nx.DiGraph[str]

    def __init__(self, graph: nx.DiGraph[str]) -> None:
        self._graph = graph

    @property
    def graph(self) -> nx.DiGraph[str]:
        return self._graph

    @graph.setter
    def graph(self, g: nx.DiGraph[str]) -> None:
        self._graph = g

    def should_filter_path(
        self,
        path: pathlib.Path,
        *,
        get_stage_state: Callable[[str], StageExecutionState],
    ) -> bool:
        """Check if a path change should be filtered (produced by an executing stage).

        Returns True if the path's producer stage is currently between
        PREPARING and COMPLETED (exclusive) — i.e., PREPARING, WAITING_ON_LOCK,
        or RUNNING.
        """
        producer = engine_graph.get_producer(self._graph, path)
        if producer is None:
            return False
        state = get_stage_state(producer)
        return StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED

    def get_affected_stages(self, paths: list[pathlib.Path]) -> list[str]:
        """Get all stages affected by the given path changes (including downstream).

        Deduplicates across paths. Returns a list (order is not guaranteed).
        """
        affected = set[str]()
        for path in paths:
            consumers = engine_graph.get_consumers(self._graph, path)
            affected.update(consumers)
            for stage in consumers:
                downstream = engine_graph.get_downstream_stages(self._graph, stage)
                affected.update(downstream)
        return list(affected)

    def get_producer(self, path: pathlib.Path) -> str | None:
        """Get the stage that produces a given artifact path."""
        return engine_graph.get_producer(self._graph, path)

    def should_restart_workers(self, *, parallel: bool) -> bool:
        """Decide whether worker pool should restart after code/config change.

        Workers should restart to pick up reloaded code, but only when
        running in parallel mode (sequential mode has no persistent pool).
        """
        return parallel
```

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest packages/pivot/tests/engine/test_watch.py -v
```
Expected: All PASS.

**Step 5: Commit**

Message: "feat: add WatchCoordinator for watch-mode policy (#420)"

### Step 6: Wire WatchCoordinator into Engine

Modify `engine.py` to delegate path filtering and affected-stage computation to `WatchCoordinator`. Use **lazy initialization** so existing tests that set `engine._graph` directly still work.

**Add import:**

```python
from pivot.engine import watch as watch_mod
```

**Add lazy accessor method (new method on Engine):**

```python
    def _get_watch_coordinator(self) -> watch_mod.WatchCoordinator | None:
        """Lazily create/update WatchCoordinator from current graph.

        This ensures existing tests that set engine._graph directly still work
        without needing to also set up the coordinator.
        """
        if self._graph is None:
            return None
        if not hasattr(self, "_watch_coordinator") or self._watch_coordinator is None:
            self._watch_coordinator: watch_mod.WatchCoordinator = watch_mod.WatchCoordinator(self._graph)
        elif self._watch_coordinator.graph is not self._graph:
            self._watch_coordinator.graph = self._graph
        return self._watch_coordinator
```

**Replace `_should_filter_path`:**

```python
    def _should_filter_path(self, path: pathlib.Path) -> bool:
        """Check if path should be filtered (output of executing stage)."""
        coordinator = self._get_watch_coordinator()
        if coordinator is None:
            return False
        return coordinator.should_filter_path(path, get_stage_state=self._get_stage_state)
```

**Replace `_get_affected_stages_for_path`:**

```python
    def _get_affected_stages_for_path(self, path: pathlib.Path) -> list[str]:
        """Get stages affected by a path change using bipartite graph."""
        coordinator = self._get_watch_coordinator()
        if coordinator is None:
            return []
        return coordinator.get_affected_stages([path])
```

**Replace `_get_affected_stages_for_paths`:**

```python
    def _get_affected_stages_for_paths(self, paths: list[pathlib.Path]) -> list[str]:
        """Get all stages affected by multiple path changes (including downstream)."""
        coordinator = self._get_watch_coordinator()
        if coordinator is None:
            return []

        filtered = list[pathlib.Path]()
        for path in paths:
            if self._should_filter_path(path):
                _logger.debug("Filtering event for %s (output of executing stage)", path)
                continue
            filtered.append(path)

        if not filtered:
            return []
        return coordinator.get_affected_stages(filtered)
```

**Update `_handle_data_artifact_changed`** — replace `engine_graph.get_producer(self._graph, path)` with coordinator:

```python
        for path in paths:
            if self._should_filter_path(path):
                coordinator = self._get_watch_coordinator()
                producer = coordinator.get_producer(path) if coordinator else None
                if producer:
                    deferred_paths.append((producer, path))
                    continue
            filtered_paths.append(path)
```

**Update `_handle_code_or_config_changed`** — use coordinator for worker restart decision:

Replace the `if self._stored_parallel:` block with:

```python
            coordinator = self._get_watch_coordinator()
            if coordinator is not None and coordinator.should_restart_workers(parallel=self._stored_parallel):
                # ... (rest of restart logic stays the same)
```

**Update `_invalidate_caches`** — reset coordinator:

```python
    def _invalidate_caches(self) -> None:
        """Invalidate all caches when code changes."""
        linecache.clearcache()
        importlib.invalidate_caches()
        self._graph = None
        if hasattr(self, "_watch_coordinator"):
            self._watch_coordinator = None
        if self._pipeline is not None:
            self._pipeline.invalidate_dag_cache()
```

**Step 7: Run all engine tests**

Run:
```bash
uv run pytest packages/pivot/tests/engine/ -v
```
Expected: All PASS (existing tests work because lazy init creates coordinator from `engine._graph`).

**Step 8: Verify acceptance criteria test patterns**

Run:
```bash
uv run pytest packages/pivot/tests/engine/ -k "watch or affected" -v
```
Expected: Both new `test_watch.py` tests and existing Engine watch/affected tests PASS.

**Step 9: Commit**

Message: "refactor: delegate watch-mode policy to WatchCoordinator (#420)"

---

## Task 9: Quality checks

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
uv run pytest packages/pivot/tests -n auto
```
Expected: All PASS.

**Step 4: Commit any fixups**

Message: "chore: fix lint/type issues from engine hardening"

---

## Summary of Acceptance Criteria

| Criterion | Verified By |
|-----------|------------|
| No Engine code writes to `self._scheduler._*` private fields | Task 1 grep (already clean) |
| Scheduler invariant violations caught by tests (fail loud) | Task 3 (mutex underflow), Task 4 (initialize validation) |
| Characterization tests use public APIs only | Task 2 |
| `test_scheduler_characterization.py` → PASS | Task 2 Step 4 |
| Deferred event during processing not lost | Task 7 Step 1 test |
| Max-iterations guard emits diagnostic, drops events | Task 7 Step 3 test |
| `test_engine -k "deferred_events"` → PASS | Task 7 Step 5 |
| WatchCoordinator unit tests pass | Task 8 Step 4 |
| Existing Engine watch/affected tests still pass | Task 8 Step 7 |
| `test_engine -k "watch or affected"` → PASS | Task 8 Step 8 |
| Full test suite passes | Task 9 Step 3 |

## Execution Order

```
Task 1  — Audit (verification only, no code changes)
Task 2  — Rewrite characterization tests to public API  [MUST be before 3-4]
Task 3  — Mutex underflow fail-loud
Task 4  — Validate initialize() inputs
Task 5  — Document determinism tie-breaker
Task 6  — Add EngineDiagnostic event type  [MUST be before 7]
Task 7  — Fix deferred-event drain (#419)
Task 8  — WatchCoordinator extraction (#420)
Task 9  — Quality checks
```

---

## Related Work: Skip Detection Unification

This plan intentionally does NOT address the skip detection divergence between
`worker.execute_stage()` and `explain.get_stage_explanation()`. That's a separate
workstream tracked in `2026-02-11-unify-skip-detection-design.md`. Here's why
they're adjacent but separate, and what the end-state architecture should be.

### How the layers relate

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: "What's affected?"  (graph traversal)              │
│   WatchCoordinator.get_affected_stages()                    │  ← this plan (#420)
│   Input: changed file paths                                 │
│   Output: set of stage names                                │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: "Will it actually run?"  (hash comparison)         │
│   Shared: is_changed()                                      │  ← skip detection plan
│   Input: stage name + current fingerprint/params/deps/outs  │
│   Output: (changed: bool, reason: str, details: ChangeSet)  │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: "Execute or restore"  (under lock)                 │
│   worker.execute_stage()                                    │  ← stays in worker
│   Input: WorkerStageInfo + Layer 2 decision                 │
│   Output: StageResult                                       │
└─────────────────────────────────────────────────────────────┘
```

Layer 1 (this plan) feeds into Layer 2 (skip detection), which feeds into Layer 3
(execution). They're compositional: WatchCoordinator says "these stages might need
to run," skip detection says "this one actually does," worker says "running it now."

### Current problem

Layer 2 is implemented twice with different code:

| | Worker path | Explain path |
|---|---|---|
| Tier 1 (generation) | `can_skip_via_generation()` | Same function ✓ |
| Tier 2 (hash compare) | `StageLock.is_changed_with_lock_data()` | Independent `diff_*` functions ✗ |
| Tier 3 (run cache) | `_try_skip_via_run_cache()` | Not implemented ✗ |
| Output paths check | Yes (via `is_changed_with_lock_data`) | Missing ✗ |

The worker and explain paths can give different answers for the same stage.

### End-state architecture

One shared function answers "is this stage changed?" with both a boolean decision
AND optional detailed diffs:

```python
# In a new module, e.g. pivot/skip.py or pivot/change_detection.py

class ChangeDecision:
    """Result of comparing current stage state against lock file."""
    changed: bool
    reason: str  # "", "Code changed", "Params changed", etc.

class DetailedChangeDecision(ChangeDecision):
    """Extended result with human-readable diffs for explain/status."""
    code_changes: list[CodeChange]
    param_changes: list[ParamChange]
    dep_changes: list[DepChange]

def is_stage_changed(
    stage_lock: StageLock,
    lock_data: LockData | None,
    fingerprint: dict[str, str],
    params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
    out_paths: list[str],
    *,
    detailed: bool = False,
) -> ChangeDecision | DetailedChangeDecision:
    """Single source of truth for Tier 2 skip detection.

    Used by:
    - worker._check_skip_or_run() with detailed=False
    - explain.get_stage_explanation() with detailed=True
    - (future) WatchCoordinator skip prediction with detailed=False
    """
    # 1. Use is_changed_with_lock_data for the boolean decision (includes out_paths)
    # 2. If detailed=True, also run diff_* functions for human-readable output
    # 3. Both paths guaranteed to agree because the boolean comes from one place
```

**Key invariant:** The `changed` boolean always comes from `is_changed_with_lock_data()`.
The `diff_*` functions are only for human-readable detail — they never override the
boolean decision. This eliminates the drift.

### What needs to happen (separate plan)

1. Fix `agent_rpc.py` state_dir bug (use per-stage lookup)
2. Create shared `is_stage_changed()` function
3. Worker's `_check_skip_or_run()` calls `is_stage_changed(detailed=False)`
4. Explain's `get_stage_explanation()` calls `is_stage_changed(detailed=True)`
5. Add output path comparison to explain (free — comes from shared function)
6. Decide on Tier 3 (run cache) for explain: document as "Tier 1+2 only" or extend
7. Regression test: worker and explain agree on `will_run` for all change types

### Why this plan doesn't include it

- Different files (`explain.py`, `lock.py`, `worker.py` vs `engine.py`, `scheduler.py`)
- Different testing strategy (needs real lock files/StateDB vs Engine mocking)
- No dependency on WatchCoordinator or Scheduler changes
- Can be done in parallel with engine hardening
