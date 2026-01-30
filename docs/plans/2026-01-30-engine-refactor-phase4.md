# Engine Refactor Phase 4: Execution Orchestration & Watch Mode Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move execution orchestration from executor to Engine, add per-stage state tracking and output filtering, integrate watch mode, and delete WatchEngine.

**Architecture:** Engine becomes the execution orchestrator. The `_execute_greedy()` loop moves from `executor/core.py` to `engine/engine.py`. Executor is simplified to "run this single stage, return result". Engine uses the bipartite graph from `engine/graph.py` for change detection via `get_consumers()`. Per-stage execution states (PENDING→BLOCKED→READY→PREPARING→RUNNING→COMPLETED) enable output filtering during execution. FilesystemSource events that hit executing stage outputs are deferred until completion.

**Tech Stack:** Python 3.13+, watchfiles, threading, loky, networkx, Protocol, TypedDict

---

## Prerequisites

Before starting, verify Phase 3 is complete:
```bash
uv run pytest tests/engine/ -v
```

Expected: All tests pass. FilesystemSource and OneShotSource implemented.

---

## Task 1: Add Per-Stage Execution State Types

**Files:**
- Modify: `src/pivot/engine/types.py`
- Test: `tests/engine/test_engine.py`

Add the stage execution state enum and event types needed for per-stage tracking.

**Step 1: Add StageExecutionState enum to types.py**

Add to `src/pivot/engine/types.py` after existing enums:

```python
from enum import IntEnum

class StageExecutionState(IntEnum):
    """Per-stage execution state for fine-grained tracking.

    Uses IntEnum for ordered comparisons (e.g., state >= PREPARING).

    State transitions:
    - PENDING: Initial state, waiting for upstream dependencies
    - BLOCKED: Upstream failed, will not run
    - READY: All upstream complete, eligible to start
    - PREPARING: Preparing to run (acquiring mutex, setting up)
    - RUNNING: Actively executing in worker
    - COMPLETED: Finished (check StageStatus for success/failure/skipped)
    """

    PENDING = 0
    BLOCKED = 1
    READY = 2
    PREPARING = 3
    RUNNING = 4
    COMPLETED = 5
```

**Step 2: Add StageStateChanged event type**

```python
class StageStateChanged(TypedDict):
    """Emitted when a stage's execution state changes."""

    type: Literal["stage_state_changed"]
    stage: str
    state: StageExecutionState
    previous_state: StageExecutionState
```

**Step 3: Update OutputEvent union**

```python
OutputEvent = (
    EngineStateChanged
    | PipelineReloaded
    | StageStarted
    | StageCompleted
    | StageStateChanged
    | LogLine
)
```

**Step 4: Update `__all__`**

```python
__all__ = [
    # ... existing ...
    "StageExecutionState",
    "StageStateChanged",
]
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine/types.py && uv run ruff check src/pivot/engine/types.py && uv run basedpyright src/pivot/engine/types.py
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add StageExecutionState and StageStateChanged types

Per-stage execution states: PENDING→BLOCKED→READY→PREPARING→RUNNING→COMPLETED"
```

---

## Task 2: Add Stage State Tracking to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

Engine needs to track per-stage execution state for output filtering and progress reporting.

**Step 1: Write failing test for stage state tracking**

Add to `tests/engine/test_engine.py`:

```python
def test_engine_tracks_stage_states() -> None:
    """Engine emits StageStateChanged events during execution."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        # Submit run request
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": ["stage_a"],
            "force": False,
            "reason": "test",
        }
        eng.submit(event)

        _helper_run_loop_with_delayed_shutdown(eng, delay=0.1)

        # Should have stage state changes
        state_changes = [e for e in sink.events if e["type"] == "stage_state_changed"]
        # At minimum: PENDING -> READY -> RUNNING -> COMPLETED
        assert len(state_changes) >= 2
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_tracks_stage_states -v
```

Expected: FAIL (no stage_state_changed events emitted)

**Step 3: Add stage state tracking to Engine**

Add imports to `src/pivot/engine/engine.py`:

```python
from pivot.engine.types import (
    # ... existing ...
    StageExecutionState,
    StageStateChanged,
)
```

Add to `Engine.__init__`:

```python
        # Per-stage execution state tracking
        self._stage_states: dict[str, StageExecutionState] = dict[str, StageExecutionState]()
        self._stage_states_lock: threading.Lock = threading.Lock()
```

Add state transition method:

```python
    def _set_stage_state(self, stage: str, new_state: StageExecutionState) -> None:
        """Update stage execution state and emit event."""
        with self._stage_states_lock:
            old_state = self._stage_states.get(stage, StageExecutionState.PENDING)
            if old_state == new_state:
                return
            self._stage_states[stage] = new_state

        self.emit(StageStateChanged(
            type="stage_state_changed",
            stage=stage,
            state=new_state,
            previous_state=old_state,
        ))

    def get_stage_state(self, stage: str) -> StageExecutionState:
        """Get current execution state for a stage. Thread-safe."""
        with self._stage_states_lock:
            return self._stage_states.get(stage, StageExecutionState.PENDING)

    def get_executing_stages(self) -> list[str]:
        """Get stages currently in PREPARING or RUNNING state. Thread-safe.

        Uses IntEnum ordering: PREPARING=3, RUNNING=4, COMPLETED=5
        """
        with self._stage_states_lock:
            return [
                name for name, state in self._stage_states.items()
                if StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED
            ]
```

**Step 4: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add per-stage execution state tracking

_stage_states dict with thread-safe access via _stage_states_lock.
Methods: _set_stage_state(), get_stage_state(), get_executing_stages()."
```

---

## Task 3: Add Output Filtering Based on Stage States

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `src/pivot/engine/sources.py`
- Modify: `tests/engine/test_engine.py`

Filter filesystem events for outputs of executing stages. Defer events until stage completes.

**Step 1: Write failing test for output filtering**

Add to `tests/engine/test_engine.py`:

```python
def test_engine_filters_executing_stage_outputs(tmp_path: pathlib.Path) -> None:
    """Engine filters filesystem events for outputs of executing stages."""
    eng = engine.Engine()

    # Register stage with known output
    output_path = tmp_path / "output.csv"

    # Simulate stage in RUNNING state
    eng._stage_states["process_data"] = types.StageExecutionState.RUNNING

    # Build graph with the stage's output
    from pivot.engine import graph as engine_graph
    g = nx.DiGraph()
    stage_node = engine_graph.stage_node("process_data")
    artifact_node = engine_graph.artifact_node(output_path)
    g.add_node(stage_node, type=engine_graph.NodeType.STAGE)
    g.add_node(artifact_node, type=engine_graph.NodeType.ARTIFACT)
    g.add_edge(stage_node, artifact_node)  # stage produces artifact
    eng._graph = g

    # Check if path should be filtered
    assert eng._should_filter_path(output_path) is True

    # Non-output paths should NOT be filtered
    other_path = tmp_path / "input.csv"
    assert eng._should_filter_path(other_path) is False


def test_engine_filters_incremental_out_during_preparing(tmp_path: pathlib.Path) -> None:
    """Engine filters IncrementalOut paths during PREPARING state (restoration phase)."""
    eng = engine.Engine()

    # IncrementalOut directory being restored
    incremental_dir = tmp_path / "incremental_output"
    incremental_dir.mkdir()

    # Simulate stage in PREPARING state (when IncrementalOut restoration happens)
    eng._stage_states["incremental_stage"] = types.StageExecutionState.PREPARING

    # Build graph with the stage's IncrementalOut
    from pivot.engine import graph as engine_graph
    g = nx.DiGraph()
    stage_node = engine_graph.stage_node("incremental_stage")
    artifact_node = engine_graph.artifact_node(incremental_dir)
    g.add_node(stage_node, type=engine_graph.NodeType.STAGE)
    g.add_node(artifact_node, type=engine_graph.NodeType.ARTIFACT)
    g.add_edge(stage_node, artifact_node)  # stage produces artifact
    eng._graph = g

    # Check that IncrementalOut directory is filtered during PREPARING
    assert eng._should_filter_path(incremental_dir) is True

    # Files within the IncrementalOut should also be filtered
    file_in_incremental = incremental_dir / "restored_file.parquet"
    # Need to check parent path containment
    producer = engine_graph.get_producer(g, incremental_dir)
    assert producer == "incremental_stage"

    # Verify PREPARING state triggers filtering (not just RUNNING)
    eng._stage_states["incremental_stage"] = types.StageExecutionState.READY
    assert eng._should_filter_path(incremental_dir) is False

    eng._stage_states["incremental_stage"] = types.StageExecutionState.PREPARING
    assert eng._should_filter_path(incremental_dir) is True
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_filters_executing_stage_outputs -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute '_should_filter_path'`

**Step 3: Add output filtering to Engine**

Add to `src/pivot/engine/engine.py`:

```python
from pivot.engine import graph as engine_graph
```

Add to `Engine.__init__`:

```python
        # Deferred events for filtered paths (stage -> list of events)
        self._deferred_events: dict[str, list[InputEvent]] = dict[str, list[InputEvent]]()
```

Add filtering methods:

```python
    def _should_filter_path(self, path: pathlib.Path) -> bool:
        """Check if path should be filtered (output of executing stage).

        Uses IntEnum ordering for comparison: filter if state >= PREPARING and < COMPLETED.
        """
        if self._graph is None:
            return False

        # Get the stage that produces this artifact
        producer = engine_graph.get_producer(self._graph, path)
        if producer is None:
            return False

        # Filter if producer is currently executing (PREPARING or RUNNING)
        state = self.get_stage_state(producer)
        return StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED

    def _get_output_paths_for_stage(self, stage: str) -> list[pathlib.Path]:
        """Get all output paths for a stage from the graph."""
        if self._graph is None:
            return []

        node = engine_graph.stage_node(stage)
        if node not in self._graph:
            return []

        paths = list[pathlib.Path]()
        for successor in self._graph.successors(node):
            if self._graph.nodes[successor]["type"] == engine_graph.NodeType.ARTIFACT:
                _, path_str = engine_graph.parse_node(successor)
                paths.append(pathlib.Path(path_str))
        return paths

    def _defer_event_for_stage(self, stage: str, event: InputEvent) -> None:
        """Defer an event until the stage completes."""
        if stage not in self._deferred_events:
            self._deferred_events[stage] = list[InputEvent]()
        self._deferred_events[stage].append(event)

    def _process_deferred_events(self, stage: str) -> None:
        """Process any deferred events for a completed stage."""
        events = self._deferred_events.pop(stage, [])
        for event in events:
            self._handle_input_event(event)
```

**Step 4: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add output filtering for executing stages

_should_filter_path() checks if path is output of PREPARING/RUNNING stage.
Deferred events processed when stage completes via _process_deferred_events()."
```

---

## Task 4: Add Change Detection Using Bipartite Graph

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

Use `engine/graph.py` functions for change detection instead of building a separate file index.

**Step 1: Write failing test for change detection**

Add to `tests/engine/test_engine.py`:

```python
def test_engine_uses_bipartite_graph_for_change_detection(tmp_path: pathlib.Path) -> None:
    """Engine uses get_consumers() from bipartite graph for change detection."""
    eng = engine.Engine()

    # Build a graph with dependencies
    from pivot.engine import graph as engine_graph
    g = nx.DiGraph()

    input_path = tmp_path / "input.csv"
    stage_node = engine_graph.stage_node("process_data")
    artifact_node = engine_graph.artifact_node(input_path)

    g.add_node(artifact_node, type=engine_graph.NodeType.ARTIFACT)
    g.add_node(stage_node, type=engine_graph.NodeType.STAGE)
    g.add_edge(artifact_node, stage_node)  # artifact consumed by stage

    eng._graph = g

    # Get affected stages for a path change
    affected = eng._get_affected_stages_for_path(input_path)

    assert "process_data" in affected
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_uses_bipartite_graph_for_change_detection -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute '_get_affected_stages_for_path'`

**Step 3: Add change detection using bipartite graph**

Add to `src/pivot/engine/engine.py`:

```python
    def _get_affected_stages_for_path(self, path: pathlib.Path) -> list[str]:
        """Get stages affected by a path change using bipartite graph."""
        if self._graph is None:
            return []

        # Use get_consumers() from engine/graph.py
        consumers = engine_graph.get_consumers(self._graph, path)
        if not consumers:
            return []

        # Add downstream stages
        all_affected = set(consumers)
        for stage in consumers:
            downstream = engine_graph.get_downstream_stages(self._graph, stage)
            all_affected.update(downstream)

        return list(all_affected)

    def _get_affected_stages_for_paths(
        self, paths: list[pathlib.Path], *, include_downstream: bool = True
    ) -> list[str]:
        """Get all stages affected by multiple path changes."""
        affected = set[str]()

        for path in paths:
            # Skip if this is an output of an executing stage
            if self._should_filter_path(path):
                producer = engine_graph.get_producer(self._graph, path) if self._graph else None
                if producer:
                    # Defer this event - create a synthetic event for later processing
                    _logger.debug(f"Deferring event for {path} (output of {producer})")
                    continue

            stage_affected = self._get_affected_stages_for_path(path)
            if include_downstream:
                affected.update(stage_affected)
            else:
                # Only direct consumers, not downstream
                if self._graph:
                    affected.update(engine_graph.get_consumers(self._graph, path))

        return list(affected)
```

**Step 4: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): use bipartite graph for change detection

_get_affected_stages_for_path() uses engine_graph.get_consumers().
_get_affected_stages_for_paths() handles multiple paths with output filtering."
```

---

## Task 5: Add Single-Stage Execution to Executor

**Files:**
- Modify: `src/pivot/executor/core.py`
- Create: `tests/executor/test_single_stage.py`

Add a simpler function that runs ONE stage and returns the result. This is what Engine will call.

**Step 1: Write failing test for single-stage execution**

Create `tests/executor/test_single_stage.py`:

```python
"""Tests for single-stage execution."""

from __future__ import annotations

import pathlib

import pytest

from pivot import registry
from pivot.executor import core as executor
from pivot.types import StageStatus


def _helper_stage_func(params: None) -> dict[str, str]:
    return {"result": "success"}


@pytest.fixture
def registered_stage(tmp_path: pathlib.Path) -> str:
    """Register a simple stage for testing."""
    registry.REGISTRY.register(
        name="test_single",
        func=_helper_stage_func,
        deps=[],
        outs=[],
        deps_paths=[],
        mutex=[],
    )
    return "test_single"


def test_execute_single_stage(registered_stage: str, tmp_path: pathlib.Path) -> None:
    """execute_single_stage() runs one stage and returns result."""
    result = executor.execute_single_stage(
        stage_name=registered_stage,
        cache_dir=tmp_path / "cache",
        force=False,
    )

    assert result["status"] in (StageStatus.RAN, StageStatus.SKIPPED)


def test_execute_single_stage_with_force(registered_stage: str, tmp_path: pathlib.Path) -> None:
    """execute_single_stage() with force=True always runs."""
    result = executor.execute_single_stage(
        stage_name=registered_stage,
        cache_dir=tmp_path / "cache",
        force=True,
    )

    assert result["status"] == StageStatus.RAN
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/executor/test_single_stage.py::test_execute_single_stage -v
```

Expected: FAIL with `AttributeError: module 'pivot.executor.core' has no attribute 'execute_single_stage'`

**Step 3: Add execute_single_stage() to executor/core.py**

Add to `src/pivot/executor/core.py` after existing imports:

```python
def execute_single_stage(
    stage_name: str,
    cache_dir: pathlib.Path,
    force: bool = False,
    no_commit: bool = False,
    no_cache: bool = False,
    overrides: parameters.ParamsOverrides | None = None,
    checkout_modes: list[cache.CheckoutMode] | None = None,
    output_queue: mp.Queue[OutputMessage] | None = None,
) -> StageResult:
    """Execute a single stage and return its result.

    This is the simplified executor interface for Engine orchestration.
    The Engine handles scheduling, mutex coordination, and parallel execution.
    This function just runs the stage.

    Args:
        stage_name: Name of stage to execute.
        cache_dir: Directory for lock files.
        force: If True, bypass cache and force re-execution.
        no_commit: If True, defer lock files to pending dir.
        no_cache: If True, skip caching outputs entirely.
        overrides: Parameter overrides from params.yaml.
        checkout_modes: Checkout modes for cache restoration.
        output_queue: Queue for worker stdout/stderr.

    Returns:
        StageResult with status, reason, and output_lines.
    """
    overrides = overrides or {}
    checkout_modes = checkout_modes or config.get_checkout_mode_order()

    # Get stage info
    stage_info = registry.REGISTRY.get(stage_name)

    # Set up state DB
    state_db_path = config.get_state_db_path()
    state_dir = config.get_state_dir()
    project_root = project.get_project_root()

    # Create temporary output queue if not provided
    spawn_ctx = mp.get_context("spawn")
    local_manager = spawn_ctx.Manager() if output_queue is None else None
    if output_queue is None:
        output_queue = local_manager.Queue()  # pyright: ignore[reportOptionalMemberAccess, reportAssignmentType]

    try:
        with state_mod.StateDB(state_db_path) as state_db:
            result = worker.run_stage(
                stage_name=stage_name,
                stage_info=stage_info,
                cache_dir=cache_dir,
                output_queue=output_queue,
                state_db=state_db,
                project_root=project_root,
                state_dir=state_dir,
                overrides=overrides,
                checkout_modes=checkout_modes,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
            )

            # Apply deferred writes for successful stages (only in commit mode)
            if result["status"] == StageStatus.RAN and not no_commit:
                output_paths = [str(out.path) for out in stage_info["outs"]]
                _apply_deferred_writes(stage_name, output_paths, result, state_db)

            return result
    finally:
        if local_manager is not None:
            local_manager.shutdown()
```

**Step 4: Run tests**

```bash
uv run pytest tests/executor/test_single_stage.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/executor && uv run ruff check src/pivot/executor && uv run basedpyright src/pivot/executor
```

**Step 6: Commit**

```bash
jj describe -m "feat(executor): add execute_single_stage() for Engine orchestration

Simplified interface: run one stage, return result.
Engine handles scheduling, mutex, parallelism."
```

---

## Task 6: Move Execution Orchestration to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

Move the `_execute_greedy()` loop logic from executor to Engine. This is the core architectural change.

**Step 1: Write failing test for Engine execution orchestration**

Add to `tests/engine/test_engine.py`:

```python
def test_engine_orchestrates_parallel_execution() -> None:
    """Engine orchestrates parallel stage execution with mutex handling."""
    # This test verifies Engine handles:
    # 1. Stage state transitions (PENDING -> READY -> RUNNING -> COMPLETED)
    # 2. Dependency tracking (upstream_unfinished)
    # 3. Mutex coordination
    # 4. Output filtering during execution

    eng = engine.Engine()
    sink = _MockSink()
    eng.add_sink(sink)

    # Create a mock graph and stage states
    # For now, verify the orchestration methods exist
    assert hasattr(eng, '_orchestrate_execution')
    assert hasattr(eng, '_start_ready_stages')
```

**Step 2: Add orchestration data structures to Engine**

Add to `Engine.__init__`:

```python
        # Execution orchestration state
        self._futures: dict[concurrent.futures.Future[StageResult], str] = dict[concurrent.futures.Future[StageResult], str]()
        self._mutex_counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        self._stage_upstream_unfinished: dict[str, set[str]] = dict[str, set[str]]()
        self._stage_downstream: dict[str, list[str]] = dict[str, list[str]]()
        self._stage_mutex: dict[str, list[str]] = dict[str, list[str]]()
        self._executor: concurrent.futures.Executor | None = None
        self._max_workers: int = 1
        self._error_mode: OnError = OnError.FAIL
        self._stop_starting_new: bool = False
```

Add imports:

```python
import collections
import concurrent.futures
import contextlib
import dataclasses
import multiprocessing as mp
import queue
import time

import loky

from pivot import config, dag, parameters, project, registry
from pivot.executor import core as executor_core
from pivot.storage import cache
from pivot.types import OutputMessage, StageResult, StageStatus

from pivot.engine.types import (
    # ... existing ...
    LogLine,
    StageStarted,
    StageCompleted,
)
```

**Step 3: Add orchestration methods**

Add to `src/pivot/engine/engine.py`:

```python
    def _initialize_orchestration(
        self,
        execution_order: list[str],
        max_workers: int,
        error_mode: OnError,
    ) -> None:
        """Initialize orchestration state for a new execution.

        Uses bipartite graph (self._graph) to derive stage dependencies via
        engine_graph.get_upstream_stages() and get_downstream_stages().
        """
        self._futures.clear()
        self._mutex_counts.clear()
        self._stage_upstream_unfinished.clear()
        self._stage_downstream.clear()
        self._stage_mutex.clear()
        self._stage_states.clear()
        self._deferred_events.clear()
        self._stop_starting_new = False

        self._max_workers = max_workers
        self._error_mode = error_mode

        stages_set = set(execution_order)

        for stage_name in execution_order:
            stage_info = registry.REGISTRY.get(stage_name)

            # Upstream stages that must complete first (uses bipartite graph)
            upstream = [
                u for u in engine_graph.get_upstream_stages(self._graph, stage_name)
                if u in stages_set
            ]
            self._stage_upstream_unfinished[stage_name] = set(upstream)

            # Downstream stages that depend on this one (uses bipartite graph)
            downstream = [
                d for d in engine_graph.get_downstream_stages(self._graph, stage_name)
                if d in stages_set
            ]
            self._stage_downstream[stage_name] = downstream

            # Mutex groups
            self._stage_mutex[stage_name] = stage_info["mutex"]

            # Initial state: READY if no upstream, else PENDING
            initial_state = (
                StageExecutionState.READY
                if not upstream
                else StageExecutionState.PENDING
            )
            self._set_stage_state(stage_name, initial_state)

    def _can_start_stage(self, stage_name: str) -> bool:
        """Check if stage is eligible to start (ready and mutex available)."""
        if self.get_stage_state(stage_name) != StageExecutionState.READY:
            return False

        # Check mutex availability
        for mutex in self._stage_mutex.get(stage_name, []):
            if mutex == executor_core.EXCLUSIVE_MUTEX:
                # Exclusive mutex: no other stages can run
                if self._mutex_counts[mutex] > 0 or len(self._futures) > 0:
                    return False
            elif self._mutex_counts[mutex] > 0:
                return False

        return True

    def _start_ready_stages(
        self,
        cache_dir: pathlib.Path,
        output_queue: mp.Queue[OutputMessage],
        overrides: parameters.ParamsOverrides,
        checkout_modes: list[cache.CheckoutMode],
        force: bool,
        no_commit: bool,
        no_cache: bool,
        stage_start_times: dict[str, float],
    ) -> None:
        """Start all eligible stages up to max_workers."""
        if self._executor is None or self._stop_starting_new:
            return

        slots_available = self._max_workers - len(self._futures)
        if slots_available <= 0:
            return

        # Find stages that can start
        ready_stages = [
            name for name in self._stage_states
            if self._can_start_stage(name)
        ]

        for stage_name in ready_stages[:slots_available]:
            # Acquire mutex locks
            for mutex in self._stage_mutex.get(stage_name, []):
                self._mutex_counts[mutex] += 1

            # Transition to PREPARING
            self._set_stage_state(stage_name, StageExecutionState.PREPARING)

            # Submit to executor
            future = self._executor.submit(
                executor_core.execute_single_stage,
                stage_name=stage_name,
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                overrides=overrides,
                checkout_modes=checkout_modes,
                output_queue=output_queue,
            )
            self._futures[future] = stage_name

            # Record start time for duration calculation
            stage_start_times[stage_name] = time.perf_counter()

            # Transition to RUNNING and emit StageStarted
            self._set_stage_state(stage_name, StageExecutionState.RUNNING)

            # Get stage index for event
            stage_index = list(self._stage_states.keys()).index(stage_name) + 1
            total_stages = len(self._stage_states)

            # Emit StageStarted event (sinks use this for display)
            self.emit(StageStarted(
                type="stage_started",
                stage=stage_name,
                index=stage_index,
                total=total_stages,
            ))

    def _handle_stage_completion(
        self,
        stage_name: str,
        result: StageResult,
        start_time: float,
    ) -> None:
        """Handle a stage completing execution."""
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Transition to COMPLETED
        self._set_stage_state(stage_name, StageExecutionState.COMPLETED)

        # Get stage index for event
        stage_index = list(self._stage_states.keys()).index(stage_name) + 1
        total_stages = len(self._stage_states)

        # Emit StageCompleted event (sinks use this for display)
        self.emit(StageCompleted(
            type="stage_completed",
            stage=stage_name,
            status=result["status"],
            reason=result["reason"],
            duration_ms=duration_ms,
            index=stage_index,
            total=total_stages,
        ))

        # Release mutex locks
        for mutex in self._stage_mutex.get(stage_name, []):
            self._mutex_counts[mutex] -= 1
            if self._mutex_counts[mutex] < 0:
                _logger.error(f"Mutex '{mutex}' released when not held")
                self._mutex_counts[mutex] = 0

        # Update downstream stages' upstream_unfinished
        for downstream_name in self._stage_downstream.get(stage_name, []):
            unfinished = self._stage_upstream_unfinished.get(downstream_name)
            if unfinished:
                unfinished.discard(stage_name)
                # If all upstream complete, mark as READY
                if not unfinished and self.get_stage_state(downstream_name) == StageExecutionState.PENDING:
                    self._set_stage_state(downstream_name, StageExecutionState.READY)

        # Handle failure cascading
        if result["status"] == StageStatus.FAILED:
            self._cascade_failure(stage_name)

        # Process deferred events for this stage's outputs
        self._process_deferred_events(stage_name)

    def _cascade_failure(self, failed_stage: str) -> None:
        """Mark downstream stages as blocked due to upstream failure."""
        for downstream_name in self._stage_downstream.get(failed_stage, []):
            state = self.get_stage_state(downstream_name)
            if state in (StageExecutionState.PENDING, StageExecutionState.READY):
                self._set_stage_state(downstream_name, StageExecutionState.BLOCKED)
                # Recursively cascade
                self._cascade_failure(downstream_name)
```

**Step 4: Add helper function to engine/graph.py**

First, add `get_upstream_stages()` and `get_stage_dag()` to `src/pivot/engine/graph.py`:

```python
def get_upstream_stages(g: nx.DiGraph[str], stage_name: str) -> list[str]:
    """Get stages whose outputs are consumed by this stage.

    Args:
        g: The bipartite graph.
        stage_name: Name of the stage.

    Returns:
        List of stage names that produce artifacts consumed by this stage.
    """
    node = stage_node(stage_name)
    if node not in g:
        return []

    upstream = list[str]()
    # Find artifacts consumed by this stage (predecessors of stage node)
    for artifact in g.predecessors(node):
        if g.nodes[artifact]["type"] != NodeType.ARTIFACT:
            continue
        # Find stage that produces this artifact
        for producer in g.predecessors(artifact):
            if g.nodes[producer]["type"] == NodeType.STAGE:
                upstream.append(parse_node(producer)[1])
    return upstream


def get_stage_dag(g: nx.DiGraph[str]) -> nx.DiGraph[str]:
    """Extract stage-only DAG from bipartite graph.

    Creates a new graph where nodes are stage names (not "stage:name") and
    edges represent direct dependencies (stage A -> stage B means A must
    complete before B can start).

    Args:
        g: The bipartite graph.

    Returns:
        Stage-only DAG compatible with dag.get_execution_order().
    """
    stage_dag: nx.DiGraph[str] = nx.DiGraph()

    # Add all stage nodes
    for node in g.nodes():
        if g.nodes[node]["type"] == NodeType.STAGE:
            stage_name = parse_node(node)[1]
            stage_dag.add_node(stage_name)

    # Add edges: if stage A produces artifact X and stage B consumes X,
    # then A -> B in stage DAG (B depends on A)
    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.STAGE:
            continue
        stage_name = parse_node(node)[1]

        # For each artifact this stage produces
        for artifact in g.successors(node):
            if g.nodes[artifact]["type"] != NodeType.ARTIFACT:
                continue
            # For each stage that consumes this artifact
            for consumer in g.successors(artifact):
                if g.nodes[consumer]["type"] == NodeType.STAGE:
                    consumer_name = parse_node(consumer)[1]
                    # Edge: producer -> consumer (consumer depends on producer)
                    stage_dag.add_edge(stage_name, consumer_name)

    return stage_dag
```

Update `__all__` in `engine/graph.py`:
```python
__all__ = [
    # ... existing ...
    "get_upstream_stages",
    "get_stage_dag",
]
```

**Step 5: Add orchestrated execution method**

```python
    def _orchestrate_execution(
        self,
        stages: list[str] | None,
        force: bool,
        single_stage: bool,
        parallel: bool,
        max_workers: int | None,
        no_commit: bool,
        no_cache: bool,
        on_error: OnError,
        cache_dir: pathlib.Path | None,
    ) -> dict[str, executor_core.ExecutionSummary]:
        """Orchestrate parallel stage execution with the Engine's event loop."""
        import multiprocessing as mp

        from pivot.storage import project_lock

        if cache_dir is None:
            cache_dir = config.get_cache_dir()

        # Build bipartite graph (single source of truth)
        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        self._graph = engine_graph.build_graph(all_stages)

        # Extract stage-only DAG for execution order
        stage_dag = engine_graph.get_stage_dag(self._graph)

        if stages:
            registered = set(stage_dag.nodes())
            unknown = [s for s in stages if s not in registered]
            if unknown:
                from pivot import exceptions
                raise exceptions.StageNotFoundError(unknown, available_stages=list(registered))

        execution_order = dag.get_execution_order(stage_dag, stages, single_stage=single_stage)

        if not execution_order:
            return {}

        # Compute max workers
        effective_max_workers = 1 if not parallel else executor_core._compute_max_workers(
            len(execution_order), max_workers
        )

        # Load config
        overrides = parameters.load_params_yaml()
        checkout_modes = config.get_checkout_mode_order()

        # Initialize orchestration state (uses bipartite graph for dependencies)
        self._initialize_orchestration(execution_order, effective_max_workers, on_error)

        # Create executor
        self._executor = executor_core._create_executor(effective_max_workers)

        # Create output queue
        spawn_ctx = mp.get_context("spawn")
        local_manager = spawn_ctx.Manager()
        output_queue: mp.Queue[OutputMessage] = local_manager.Queue()  # pyright: ignore[reportAssignmentType]

        # Track results and start times
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_start_times: dict[str, float] = {}  # Track when each stage started

        lock_context = project_lock.pending_state_lock() if no_commit else contextlib.nullcontext()

        # Start output reader thread for LogLine events
        output_thread: threading.Thread | None = None
        output_stop_event = threading.Event()

        try:
            # Start output drain thread to emit LogLine events
            output_thread = threading.Thread(
                target=self._drain_output_queue,
                args=(output_queue, output_stop_event),
                daemon=True,
            )
            output_thread.start()

            with lock_context:
                # Start initial ready stages
                self._start_ready_stages(
                    cache_dir=cache_dir,
                    output_queue=output_queue,
                    overrides=overrides,
                    checkout_modes=checkout_modes,
                    force=force,
                    no_commit=no_commit,
                    no_cache=no_cache,
                    stage_start_times=stage_start_times,
                )

                # Main execution loop
                while self._futures:
                    done, _ = concurrent.futures.wait(
                        self._futures.keys(),
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        stage_name = self._futures.pop(future)
                        start_time = stage_start_times.get(stage_name, time.perf_counter())

                        try:
                            result = future.result()
                            self._handle_stage_completion(stage_name, result, start_time)

                            # Record result
                            results[stage_name] = executor_core.ExecutionSummary(
                                status=result["status"],
                                reason=result["reason"],
                            )

                        except Exception as e:
                            _logger.exception(f"Stage {stage_name} failed with exception")
                            failed_result = StageResult(
                                status=StageStatus.FAILED,
                                reason=str(e),
                                output_lines=[],
                            )
                            self._handle_stage_completion(stage_name, failed_result, start_time)
                            results[stage_name] = executor_core.ExecutionSummary(
                                status=StageStatus.FAILED,
                                reason=str(e),
                            )

                    # Check error mode
                    if on_error == OnError.FAIL:
                        failed = [n for n, s in self._stage_states.items()
                                  if s == StageExecutionState.COMPLETED
                                  and results.get(n, {}).get("status") == StageStatus.FAILED]
                        if failed:
                            self._stop_starting_new = True
                            # Mark remaining READY/PENDING as blocked
                            for name, state in self._stage_states.items():
                                if state in (StageExecutionState.READY, StageExecutionState.PENDING):
                                    self._set_stage_state(name, StageExecutionState.BLOCKED)
                                    results[name] = executor_core.ExecutionSummary(
                                        status=StageStatus.SKIPPED,
                                        reason=f"upstream '{failed[0]}' failed",
                                    )

                    # Check cancellation
                    if self._cancel_event.is_set():
                        self._stop_starting_new = True
                        for name, state in self._stage_states.items():
                            if state in (StageExecutionState.READY, StageExecutionState.PENDING):
                                self._set_stage_state(name, StageExecutionState.COMPLETED)
                                results[name] = executor_core.ExecutionSummary(
                                    status=StageStatus.SKIPPED,
                                    reason="cancelled",
                                )

                    # Start more stages if slots available
                    if not self._stop_starting_new:
                        self._start_ready_stages(
                            cache_dir=cache_dir,
                            output_queue=output_queue,
                            overrides=overrides,
                            checkout_modes=checkout_modes,
                            force=force,
                            no_commit=no_commit,
                            no_cache=no_cache,
                            stage_start_times=stage_start_times,
                        )
        finally:
            # Signal output thread to stop
            output_stop_event.set()
            if output_thread:
                output_thread.join(timeout=1.0)

            self._executor = None
            local_manager.shutdown()

        return results

    def _drain_output_queue(
        self,
        output_queue: mp.Queue[OutputMessage],
        stop_event: threading.Event,
    ) -> None:
        """Drain output messages from worker processes and emit LogLine events."""
        while not stop_event.is_set():
            try:
                msg = output_queue.get(timeout=0.02)
                if msg is None:
                    break

                # Emit LogLine event for each output message
                # OutputMessage is tuple[str, str, bool]: (stage_name, line, is_stderr)
                stage_name, line, is_stderr = msg
                self.emit(LogLine(
                    type="log_line",
                    stage=stage_name,
                    line=line,
                    is_stderr=is_stderr,
                ))
            except queue.Empty:
                continue
            except (EOFError, OSError, BrokenPipeError):
                _logger.debug("Output queue drain exiting: queue closed or broken")
                break
```

**Step 5: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 6: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 7: Commit**

```bash
jj describe -m "feat(engine): move execution orchestration from executor

Engine now handles:
- Stage state transitions (PENDING→READY→RUNNING→COMPLETED)
- Parallel execution with loky ProcessPoolExecutor
- Mutex coordination
- Failure cascading
- Output filtering during execution

Executor simplified to execute_single_stage()."
```

---

## Task 7: Update run_once() to Use Engine Orchestration

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

Update `run_once()` to use the new `_orchestrate_execution()` method.

**Step 1: Update run_once() implementation**

The signature must retain all existing parameters for CLI compatibility. Parameters like
`show_output`, `tui_queue`, and `output_queue` are now handled via sinks, but the parameters
remain for backwards compatibility and CLI integration.

Replace the `run_once()` method body with:

```python
    def run_once(
        self,
        stages: list[str] | None = None,
        force: bool = False,
        single_stage: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        no_commit: bool = False,
        no_cache: bool = False,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
        on_error: OnError = OnError.FAIL,
        cache_dir: pathlib.Path | None = None,
        progress_callback: Callable[[RunJsonEvent], None] | None = None,
        explain_mode: bool = False,
        # Retained for CLI compatibility - handled via sinks
        show_output: bool = False,
        tui_queue: queue.Queue[object] | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
    ) -> dict[str, executor_core.ExecutionSummary]:
        """Execute stages once and return.

        This is the primary entry point for 'pivot run' without --watch.
        Uses Engine orchestration for parallel execution.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run all stages.
            single_stage: If True, run only the specified stages (no downstream).
            parallel: If True, run stages in parallel.
            max_workers: Maximum worker processes.
            no_commit: If True, don't update lockfiles.
            no_cache: If True, disable run cache.
            allow_uncached_incremental: Allow incremental outputs without cache.
            checkout_missing: Checkout missing dependency files from cache.
            on_error: Error handling mode ('fail' or 'keep_going').
            cache_dir: Directory for lock files (defaults to .pivot/cache).
            progress_callback: Callback for JSONL progress events.
            explain_mode: If True, explain why stages run/skip (for CLI --explain).
            show_output: If True, display stage output (for CLI compatibility).
            tui_queue: Queue for TUI messages (sinks preferred, but retained for compatibility).
            output_queue: Queue for worker output messages (Engine manages internally now).
        """
        if self._state == EngineState.ACTIVE:
            msg = "Engine is already active - concurrent run_once() calls are not allowed"
            raise RuntimeError(msg)

        self._cancel_event.clear()
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))
        self._stage_indices.clear()

        # If progress_callback provided, wrap it to also call our internal handler
        original_progress_callback = progress_callback

        def progress_adapter(event: RunJsonEvent) -> None:
            self._handle_progress_event(event)
            if original_progress_callback is not None:
                original_progress_callback(event)

        # Store progress callback for orchestration to use
        self._progress_callback = progress_adapter if progress_callback else None

        try:
            # Use Engine orchestration
            result = self._orchestrate_execution(
                stages=stages,
                force=force,
                single_stage=single_stage,
                parallel=parallel,
                max_workers=max_workers,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                cache_dir=cache_dir,
            )
            return result
        finally:
            self._state = EngineState.IDLE
            self._progress_callback = None
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
```

**Note:** The `tui_queue` and `output_queue` parameters are retained for backwards compatibility.
Callers should migrate to using sinks:
- `TuiSink` for TUI integration
- LogLine events replace direct output_queue reads

**Step 2: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
uv run pytest tests/ -n auto -k "not watch"
```

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 4: Commit**

```bash
jj describe -m "refactor(engine): run_once() uses Engine orchestration

Replaces delegation to executor.run() with _orchestrate_execution()."
```

---

## Task 8: Add Change Handlers for Watch Mode

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

Implement handlers for DataArtifactChanged and CodeOrConfigChanged events.

**Step 1: Implement _handle_data_artifact_changed**

Update the `_handle_input_event` method and add handlers:

```python
    def _handle_input_event(self, event: InputEvent) -> None:
        """Process a single input event."""
        match event["type"]:
            case "run_requested":
                self._handle_run_requested(event)
            case "cancel_requested":
                self._handle_cancel_requested()
            case "data_artifact_changed":
                self._handle_data_artifact_changed(event)
            case "code_or_config_changed":
                self._handle_code_or_config_changed(event)

    def _handle_data_artifact_changed(self, event: DataArtifactChanged) -> None:
        """Handle data artifact changes by running affected stages."""
        from pivot.engine.types import DataArtifactChanged

        paths = [pathlib.Path(p) for p in event["paths"]]

        # Filter out paths that are outputs of executing stages
        filtered_paths = list[pathlib.Path]()
        deferred_paths = list[tuple[str, pathlib.Path]]()

        for path in paths:
            if self._should_filter_path(path):
                producer = engine_graph.get_producer(self._graph, path) if self._graph else None
                if producer:
                    deferred_paths.append((producer, path))
                    continue
            filtered_paths.append(path)

        # Defer events for filtered paths
        for producer, path in deferred_paths:
            self._defer_event_for_stage(
                producer,
                DataArtifactChanged(type="data_artifact_changed", paths=[str(path)])
            )

        if not filtered_paths:
            return

        # Get affected stages
        affected = self._get_affected_stages_for_paths(filtered_paths)

        if not affected:
            return

        _logger.info(f"Data changed: {len(filtered_paths)} file(s) affect {len(affected)} stage(s)")

        # Execute affected stages
        self._execute_affected_stages(affected)

    def _handle_code_or_config_changed(self, event: CodeOrConfigChanged) -> None:
        """Handle code/config changes by reloading registry and re-running."""
        _logger.info("Code/config changed - reloading pipeline")

        # Reload registry (implementation in Task 9)
        # For now, just re-run all stages
        stages = list(registry.REGISTRY.list_stages())

        if stages:
            self._execute_affected_stages(stages)

    def _execute_affected_stages(self, stages: list[str]) -> None:
        """Execute the specified stages."""
        self._cancel_event.clear()

        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            self._orchestrate_execution(
                stages=stages,
                force=False,
                single_stage=False,
                parallel=True,
                max_workers=None,
                no_commit=False,
                no_cache=False,
                on_error=OnError.KEEP_GOING,  # Watch mode: continue on error
                cache_dir=None,
            )
        finally:
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
```

**Step 2: Add import for CodeOrConfigChanged and DataArtifactChanged**

```python
from pivot.engine.types import (
    # ... existing ...
    CodeOrConfigChanged,
    DataArtifactChanged,
)
```

**Step 3: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 4: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 5: Commit**

```bash
jj describe -m "feat(engine): add change handlers for watch mode

_handle_data_artifact_changed: uses bipartite graph for affected stages.
_handle_code_or_config_changed: reloads registry and re-runs.
Output filtering defers events for executing stage outputs."
```

---

## Task 9: Add Registry Reload to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

When code/config changes, Engine must reload the registry before executing.

**Step 1: Add registry reload methods**

Add to `src/pivot/engine/engine.py`:

```python
import importlib
import linecache
import runpy
import sys

import yaml

from pivot.engine.types import PipelineReloaded
```

Add methods:

```python
    def _invalidate_caches(self) -> None:
        """Invalidate all caches when code changes."""
        linecache.clearcache()
        importlib.invalidate_caches()
        self._graph = None
        registry.REGISTRY.invalidate_dag_cache()

    def _reload_registry(self) -> bool:
        """Reload the registry by re-importing pipeline definition.

        Returns True if reload succeeded, False if pipeline is invalid.
        """
        old_stages = registry.REGISTRY.snapshot()
        root = project.get_project_root()

        # Clear project modules from sys.modules
        self._clear_project_modules(root)

        # Try pivot.yaml first
        for name in ("pivot.yaml", "pivot.yml"):
            path = root / name
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        yaml_config = yaml.safe_load(f)
                    if isinstance(yaml_config, dict) and "stages" in yaml_config:
                        return self._reload_from_pipeline_file(path, old_stages)
                except Exception:
                    continue

        # Try pipeline.py
        pipeline_py = root / "pipeline.py"
        if pipeline_py.exists():
            return self._reload_from_pipeline_py(pipeline_py, old_stages)

        # Fallback: reimport stage modules
        return self._reload_from_decorators(old_stages)

    def _clear_project_modules(self, root: pathlib.Path) -> None:
        """Remove project modules from sys.modules."""
        root_str = str(root)
        to_remove = list[str]()

        for name, module in list(sys.modules.items()):
            if module is None:
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            try:
                if module_file.startswith(root_str):
                    to_remove.append(name)
            except (TypeError, AttributeError):
                continue

        for name in to_remove:
            del sys.modules[name]

    def _reload_from_pipeline_file(
        self, path: pathlib.Path, old_stages: dict[str, registry.RegistryStageInfo]
    ) -> bool:
        """Reload from pivot.yaml file."""
        from pivot.pipeline import yaml as pipeline_yaml

        registry.REGISTRY.clear()
        try:
            pipeline_yaml.register_from_pipeline_file(path)
            self._emit_reload_event(old_stages)
            return True
        except Exception as e:
            _logger.warning(f"Pipeline invalid: {e}")
            registry.REGISTRY.restore(old_stages)
            return False

    def _reload_from_pipeline_py(
        self, path: pathlib.Path, old_stages: dict[str, registry.RegistryStageInfo]
    ) -> bool:
        """Reload from pipeline.py file."""
        registry.REGISTRY.clear()
        try:
            runpy.run_path(str(path), run_name="_pivot_pipeline")
            self._emit_reload_event(old_stages)
            return True
        except Exception as e:
            _logger.warning(f"Pipeline invalid: {e}")
            registry.REGISTRY.restore(old_stages)
            return False

    def _reload_from_decorators(
        self, old_stages: dict[str, registry.RegistryStageInfo]
    ) -> bool:
        """Reload by reimporting stage modules."""
        modules: set[str] = set()
        for info in old_stages.values():
            func = info["func"]
            module_name = getattr(func, "__module__", None)
            if module_name and module_name in sys.modules:
                modules.add(module_name)

        if not modules:
            return True

        registry.REGISTRY.clear()
        errors = list[str]()

        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                errors.append(f"{module_name}: {e}")

        if errors:
            registry.REGISTRY.restore(old_stages)
            return False

        self._emit_reload_event(old_stages)
        return True

    def _emit_reload_event(self, old_stages: dict[str, registry.RegistryStageInfo]) -> None:
        """Emit PipelineReloaded event with diff information."""
        new_stages = set(registry.REGISTRY.list_stages())
        old_stage_names = set(old_stages.keys())

        added = list(new_stages - old_stage_names)
        removed = list(old_stage_names - new_stages)
        # TODO: detect modified stages by comparing fingerprints
        modified = list[str]()

        self.emit(PipelineReloaded(
            type="pipeline_reloaded",
            stages_added=added,
            stages_removed=removed,
            stages_modified=modified,
            error=None,
        ))
```

**Step 2: Update _handle_code_or_config_changed**

```python
    def _handle_code_or_config_changed(self, event: CodeOrConfigChanged) -> None:
        """Handle code/config changes by reloading registry and re-running."""
        _logger.info("Code/config changed - reloading pipeline")

        # Invalidate caches
        self._invalidate_caches()

        # Reload registry
        reload_ok = self._reload_registry()

        if not reload_ok:
            _logger.error("Pipeline invalid - waiting for fix")
            return

        # Rebuild graph
        self._graph = registry.REGISTRY.build_dag(validate=True)

        # Update watch paths if we have a FilesystemSource
        from pivot.engine.sources import FilesystemSource
        watch_paths = engine_graph.get_watch_paths(self._graph)
        for source in self._sources:
            if isinstance(source, FilesystemSource):
                source.set_watch_paths(watch_paths)

        # Re-run all stages
        stages = list(registry.REGISTRY.list_stages())

        if stages:
            self._execute_affected_stages(stages)
```

**Step 3: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 4: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 5: Commit**

```bash
jj describe -m "feat(engine): add registry reload on code changes

_reload_registry() handles pivot.yaml/pipeline.py/decorator-based pipelines.
Preserves old registry on failure. Updates watch paths after reload."
```

---

## Task 10: Add Agent RPC Methods to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `src/pivot/types.py` (update existing Agent types)
- Modify: `tests/engine/test_engine.py`

Add methods for Agent RPC: `try_start_run()`, `get_execution_status()`, `request_cancel()`.

**Step 1: Simplify Agent types in pivot/types.py**

The existing `AgentState` enum is unnecessary - we can use `EngineState` directly. Update `src/pivot/types.py`:

1. **Remove `AgentState` enum** (lines 719-727)

2. **Update `AgentStatusResult` to use `EngineState`**:

```python
from pivot.engine.types import EngineState

class AgentStatusResult(TypedDict, total=False):
    """Result of status() RPC method."""

    state: Required[EngineState]  # Changed from AgentState
    run_id: str
    stages_completed: list[str]
    stages_pending: list[str]
    ran: int
    skipped: int
    failed: int
```

Note: This creates a dependency from `pivot.types` to `pivot.engine.types`. If this circular import is problematic, move `EngineState` to `pivot.types` instead.

**Existing types to reuse** (no changes needed):
- `AgentRunStartResult` - already has correct fields
- `AgentRunRejection` - already has correct fields
- `AgentCancelResult` - already has correct fields

**Step 2: Add agent methods to Engine**

Add to `Engine.__init__`:

```python
        # Agent RPC state
        self._run_lock: threading.Lock = threading.Lock()
        self._current_run_id: str | None = None
```

Add methods (using existing types from `pivot.types`):

```python
    def try_start_run(
        self,
        run_id: str,
        stages: list[str] | None,
        force: bool,
    ) -> AgentRunStartResult | AgentRunRejection:
        """Atomically try to start a run.

        Returns AgentRunStartResult if started, AgentRunRejection if rejected.
        Thread-safe - can be called from asyncio thread.
        """
        from pivot.types import AgentRunRejection, AgentRunStartResult

        stages_to_run = stages or list(registry.REGISTRY.list_stages())

        with self._run_lock:
            if self._state != EngineState.IDLE:
                return AgentRunRejection(
                    reason="busy",
                    current_state=self._state.value,
                    current_run_id=self._current_run_id,
                )

            # Set up run state
            self._current_run_id = run_id

            # Submit run request
            self.submit(RunRequested(
                type="run_requested",
                stages=stages_to_run,
                force=force,
                reason=f"agent:{run_id}",
            ))

            return AgentRunStartResult(
                run_id=run_id,
                status="started",
                stages_queued=stages_to_run,
            )

    def get_execution_status(self, run_id: str | None = None) -> AgentStatusResult:
        """Query current execution state.

        Thread-safe - can be called from asyncio thread.
        """
        from pivot.types import AgentStatusResult

        with self._run_lock:
            status = AgentStatusResult(state=self._state)

            if run_id is not None and run_id != self._current_run_id:
                return status

            if self._current_run_id is not None:
                status["run_id"] = self._current_run_id

            # Get stages by state
            completed = [
                name for name, state in self._stage_states.items()
                if state == StageExecutionState.COMPLETED
            ]
            pending = [
                name for name, state in self._stage_states.items()
                if state in (StageExecutionState.PENDING, StageExecutionState.READY,
                            StageExecutionState.PREPARING, StageExecutionState.RUNNING)
            ]

            if completed:
                status["stages_completed"] = completed
            if pending:
                status["stages_pending"] = pending

            return status

    def request_cancel(self) -> AgentCancelResult:
        """Request cancellation of current execution.

        Thread-safe - can be called from asyncio thread.
        """
        from pivot.types import AgentCancelResult

        with self._run_lock:
            if self._state == EngineState.ACTIVE:
                self._cancel_event.set()
                return AgentCancelResult(cancelled=True)
            return AgentCancelResult(cancelled=False)
```

**Step 3: Run tests**

```bash
uv run pytest tests/engine/test_engine.py -v
```

**Step 4: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 5: Commit**

```bash
jj describe -m "feat(engine): add agent RPC methods

try_start_run(), get_execution_status(), request_cancel().
Thread-safe with run_lock for asyncio integration.
Reuses existing Agent* types from pivot.types."
```

---

## Task 11: Verify Run History

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Create: `tests/engine/test_run_history.py`

Verify that run history (RunManifest, RunCacheEntry) is correctly written after Engine-orchestrated execution.

**Step 1: Write test for run history**

Create `tests/engine/test_run_history.py`:

```python
"""Tests for run history after Engine execution."""

from __future__ import annotations

import pathlib

import pytest

from pivot import config, registry, run_history
from pivot.engine import engine


def _helper_stage_func(params: None) -> dict[str, str]:
    return {"result": "success"}


@pytest.fixture
def registered_stage() -> str:
    """Register a simple stage for testing."""
    registry.REGISTRY.register(
        name="history_test",
        func=_helper_stage_func,
        deps=[],
        outs=[],
        deps_paths=[],
        mutex=[],
    )
    return "history_test"


def test_engine_writes_run_history(
    registered_stage: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine writes run history after execution."""
    # Set up paths
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)

    eng = engine.Engine()

    # Run the stage
    results = eng.run_once(
        stages=[registered_stage],
        cache_dir=cache_dir,
    )

    assert registered_stage in results

    # Verify run history was written
    runs = run_history.list_runs(limit=1)
    assert len(runs) >= 1

    latest = runs[0]
    assert "run_id" in latest
    assert "started_at" in latest
    assert "ended_at" in latest


def test_engine_writes_run_cache_entry(
    registered_stage: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine writes run cache entries for successful stages."""
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(config, "get_state_dir", lambda: state_dir)

    eng = engine.Engine()

    # Run twice - second should be cached
    eng.run_once(stages=[registered_stage], cache_dir=cache_dir)
    results = eng.run_once(stages=[registered_stage], cache_dir=cache_dir)

    # Should be skipped due to cache
    assert results[registered_stage]["reason"] != ""  # Has a reason (e.g., "cached" or "up to date")
```

**Step 2: Run tests**

```bash
uv run pytest tests/engine/test_run_history.py -v
```

**Step 3: Add run history writing to _orchestrate_execution**

The existing `_write_run_history()` function is in `executor/core.py`. We'll call it from Engine after orchestration completes.

Update `_orchestrate_execution` to include run history:

```python
        # At the start of _orchestrate_execution, after execution_order:
        import datetime
        from pivot import run_history as run_history_mod

        started_at = datetime.datetime.now(datetime.UTC).isoformat()
        run_id = run_history_mod.generate_run_id()
        targeted_stages = stages if stages else execution_order

        # ... existing execution loop ...

        # At the end, before returning results:
        ended_at = datetime.datetime.now(datetime.UTC).isoformat()

        # Build stage states for history (convert ExecutionSummary to StageState)
        from pivot.executor.core import StageState, _write_run_history

        stage_states: dict[str, StageState] = {}
        for name, summary in results.items():
            stage_states[name] = StageState(
                status=summary["status"],
                reason=summary["reason"],
                result=StageResult(
                    status=summary["status"],
                    reason=summary["reason"],
                    output_lines=[],
                ),
            )

        _write_run_history(
            run_id=run_id,
            stage_states=stage_states,
            targeted_stages=targeted_stages,
            execution_order=execution_order,
            started_at=started_at,
            ended_at=ended_at,
        )
```

Note: `_write_run_history()` is currently a private function in `executor/core.py`. Task 14 will move it to be a public function when we clean up the executor module.

**Step 4: Run tests**

```bash
uv run pytest tests/engine/test_run_history.py -v
```

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): write run history after orchestrated execution

Writes RunManifest and RunCacheEntry via run_history module.
Verified: run history appears in list_runs() after Engine.run_once()."
```

---

## Task 12: Refactor AgentServer to Use Engine

**Files:**
- Modify: `src/pivot/tui/agent_server.py`
- Modify: `tests/tui/test_agent_server.py`

**Step 1: Update AgentServer to accept Engine**

Update `src/pivot/tui/agent_server.py`:

```python
if TYPE_CHECKING:
    from pathlib import Path

    from pivot.engine.engine import Engine
```

Update class:

```python
class AgentServer:
    """JSON-RPC server for agent control of pipeline execution.

    This is a stateless RPC facade - all execution state is managed by Engine.
    """

    _engine: Engine
    _socket_path: Path

    def __init__(self, engine: Engine, socket_path: Path) -> None:
        self._engine = engine
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None
```

Update `_handle_run`:

```python
    async def _handle_run(self, params: AgentRunParams) -> AgentRunStartResult:
        """Handle run() RPC method."""
        stages = params["stages"] if "stages" in params else None
        if stages:
            all_stages = set(registry.REGISTRY.list_stages())
            for stage in stages:
                if stage not in all_stages:
                    suggestions = difflib.get_close_matches(stage, all_stages, n=3, cutoff=0.6)
                    raise _StageNotFoundError(stage, suggestions)

        run_id = str(uuid.uuid4())[:12]
        force = params["force"] if "force" in params else False

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._engine.try_start_run,
            run_id,
            stages,
            force,
        )

        if "reason" in result:
            raise _ExecutionInProgressError

        return result
```

Update `_handle_status`:

```python
    async def _handle_status(self, params: dict[str, Any]) -> AgentStatusResult:
        """Handle status() RPC method."""
        run_id = params["run_id"] if "run_id" in params else None
        return self._engine.get_execution_status(run_id)
```

Update `_handle_cancel`:

```python
    async def _handle_cancel(self) -> AgentCancelResult:
        """Handle cancel() RPC method."""
        return self._engine.request_cancel()
```

Note: The `_connected_clients` tracking and `connected` property remain unchanged - they're part of AgentServer's socket management, not execution state.

**Step 2: Update tests**

Update `tests/tui/test_agent_server.py` to use Engine instead of WatchEngine.

**Step 3: Run tests**

```bash
uv run pytest tests/tui/test_agent_server.py -v
```

**Step 4: Run quality checks**

```bash
uv run ruff format src/pivot/tui && uv run ruff check src/pivot/tui && uv run basedpyright src/pivot/tui
```

**Step 5: Commit**

```bash
jj describe -m "refactor(agent): use Engine instead of WatchEngine

AgentServer delegates to Engine.try_start_run/get_execution_status/request_cancel."
```

---

## Task 13: Update CLI Watch Mode to Use Engine

**Files:**
- Modify: `src/pivot/cli/run.py`
- Modify: `src/pivot/tui/run.py`

**Step 1: Update _run_watch_with_tui to use Engine**

See the detailed implementation in the original plan. Key changes:
- Create Engine instead of WatchEngine
- Add FilesystemSource and OneShotSource
- Add TuiSink and WatchSink
- Run Engine.run_loop() in background thread

**Step 2: Add run_watch_tui_engine to tui/run.py**

Add a new function that runs TUI with Engine (see original plan for full implementation).

**Step 3: Run tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "watch"
uv run pytest tests/watch/ -v
```

**Step 4: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 5: Commit**

```bash
jj describe -m "feat(cli): route watch mode through Engine

_run_watch_with_tui uses Engine + FilesystemSource.
Adds run_watch_tui_engine() to tui/run.py."
```

---

## Task 14: Delete Redundant Executor Code

**Files:**
- Modify: `src/pivot/executor/core.py`
- Modify: `src/pivot/executor/__init__.py`
- Delete or update: `tests/executor/test_core.py` (tests for deleted functions)

Now that Engine handles orchestration, delete the redundant code from executor.

**Step 1: Delete `_execute_greedy()` and related helpers**

Delete from `src/pivot/executor/core.py`:

```python
# DELETE these functions:
def _execute_greedy(...) -> ...:
    """Execute stages in parallel using greedy scheduling."""
    ...

def _start_stages(...) -> ...:
    """Start stages that are ready to execute."""
    ...

def _await_completions(...) -> ...:
    """Wait for any stage to complete and process results."""
    ...
```

These functions are now replaced by Engine's `_orchestrate_execution()`, `_start_ready_stages()`, and the main execution loop.

**Step 2: Delete `StageLifecycle` class**

Delete from `src/pivot/executor/core.py`:

```python
# DELETE this class:
@dataclasses.dataclass
class StageLifecycle:
    """Manages stage execution state and notifications."""
    ...
```

Stage lifecycle is now managed by Engine via `StageExecutionState` and `_set_stage_state()`.

**Step 3: Delete `run()` function**

Delete from `src/pivot/executor/core.py`:

```python
# DELETE this function:
def run(
    stages: list[str] | None = None,
    force: bool = False,
    ...
) -> dict[str, ExecutionSummary]:
    """Execute pipeline stages."""
    ...
```

All callers now use `Engine.run_once()` instead.

**Step 4: Make `_write_run_history()` public**

Rename to `write_run_history()` (remove leading underscore) since Engine now calls it:

```python
def write_run_history(
    run_id: str,
    stage_states: dict[str, StageState],
    targeted_stages: list[str],
    execution_order: list[str],
    started_at: str,
    ended_at: str,
) -> None:
    """Write run history to StateDB."""
    ...
```

Update `src/pivot/executor/__init__.py` to export it:

```python
from pivot.executor.core import (
    ExecutionSummary,
    execute_single_stage,
    write_run_history,
    # Remove: run, StageLifecycle
)
```

**Step 5: Update Engine to use renamed function**

In `src/pivot/engine/engine.py`, update the import:

```python
from pivot.executor.core import write_run_history  # was: _write_run_history
```

**Step 6: Delete obsolete tests**

Remove tests for deleted functions from `tests/executor/test_core.py`:
- Tests for `run()`
- Tests for `_execute_greedy()`
- Tests for `StageLifecycle`

Keep tests for:
- `execute_single_stage()`
- `write_run_history()`
- Worker-related functions

**Step 7: Run tests**

```bash
uv run pytest tests/ -n auto
```

**Step 8: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 9: Commit**

```bash
jj describe -m "refactor(executor): delete redundant orchestration code

Delete _execute_greedy(), StageLifecycle, and run().
Engine now owns orchestration via _orchestrate_execution().
Rename _write_run_history() to write_run_history() (public API)."
```

---

## Task 15: Delete WatchEngine

**Files:**
- Delete: `src/pivot/watch/engine.py`
- Modify: `src/pivot/watch/__init__.py`
- Update: `tests/watch/test_engine.py` → move useful tests to `tests/engine/`

**Step 1: Verify all tests pass with Engine**

```bash
uv run pytest tests/ -n auto
```

Expected: PASS

**Step 2: Remove WatchEngine imports**

Update `src/pivot/watch/__init__.py` to remove WatchEngine export.

**Step 3: Delete src/pivot/watch/engine.py**

```bash
rm src/pivot/watch/engine.py
```

**Step 4: Move/update tests**

Move relevant tests from `tests/watch/test_engine.py` to `tests/engine/test_engine.py`.

**Step 5: Run all tests**

```bash
uv run pytest tests/ -n auto
```

**Step 6: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 7: Commit**

```bash
jj describe -m "refactor: delete WatchEngine

All watch functionality now handled by unified Engine.
Moved/updated tests to tests/engine/."
```

---

## Task 16: Final Verification and Cleanup

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -n auto --cov=pivot --cov-report=term-missing
```

Expected: PASS with coverage ≥90%

**Step 2: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 3: Manual integration test**

```bash
cd /tmp && mkdir test-pipeline && cd test-pipeline
echo "stages:" > pivot.yaml
echo "  hello:" >> pivot.yaml
echo "    cmd: echo hello" >> pivot.yaml

pivot run --watch --display
```

Verify:
- Initial execution runs
- File changes trigger re-execution
- Ctrl+C exits cleanly

**Step 4: Final commit**

```bash
jj describe -m "feat(engine): complete Phase 4 - Execution Orchestration

- Engine orchestrates parallel execution (moved from executor)
- Per-stage state tracking: PENDING→BLOCKED→READY→PREPARING→RUNNING→COMPLETED
- Output filtering: defers events for executing stage outputs
- Bipartite graph for change detection via get_consumers()
- Registry reload on code changes
- Agent RPC methods (try_start_run, get_execution_status, request_cancel)
- Run history verified (RunManifest, RunCacheEntry)
- CLI watch mode routed through Engine
- Executor simplified: deleted run(), _execute_greedy(), StageLifecycle
- WatchEngine deleted - unified architecture complete"
```

---

## Summary

After completing Phase 4 (16 tasks):

1. **Engine owns execution orchestration** - The `_execute_greedy()` loop logic moved from `executor/core.py` to `engine/engine.py`.

2. **Executor simplified** - Now just `execute_single_stage()` and `write_run_history()`. Deleted `run()`, `_execute_greedy()`, and `StageLifecycle`.

3. **Per-stage execution states** - `StageExecutionState` enum: PENDING→BLOCKED→READY→PREPARING→RUNNING→COMPLETED.

4. **Output filtering** - `_should_filter_path()` checks if path is output of executing stage (including IncrementalOut during PREPARING). Deferred events processed on completion.

5. **Bipartite graph for change detection** - Uses `engine/graph.py` functions (`get_consumers()`, `get_producer()`, `get_downstream_stages()`) instead of duplicate file index.

6. **Run history verified** - RunManifest and RunCacheEntry written after Engine-orchestrated execution via `write_run_history()`.

7. **Simplified Agent types** - Removed `AgentState` enum. Engine uses `EngineState` directly and returns existing `Agent*` types from `pivot.types`.

8. **WatchEngine deleted** - All functionality absorbed into unified Engine.

**Files changed:**
| File | Action |
|------|--------|
| `src/pivot/engine/types.py` | Add StageExecutionState, StageStateChanged |
| `src/pivot/engine/engine.py` | Add orchestration, state tracking, output filtering, change handlers, agent methods |
| `src/pivot/executor/core.py` | Add execute_single_stage(); delete run(), _execute_greedy(), StageLifecycle; rename _write_run_history → write_run_history |
| `src/pivot/executor/__init__.py` | Update exports (remove run, add write_run_history) |
| `src/pivot/types.py` | Update AgentStatusResult to use EngineState; remove AgentState |
| `src/pivot/tui/agent_server.py` | Use Engine instead of WatchEngine |
| `src/pivot/tui/run.py` | Add run_watch_tui_engine() |
| `src/pivot/cli/run.py` | Route watch mode through Engine |
| `src/pivot/watch/engine.py` | Delete |
| `tests/engine/` | Add comprehensive tests |
| `tests/executor/test_single_stage.py` | New tests for execute_single_stage() |
| `tests/engine/test_run_history.py` | New tests for run history |
| `tests/executor/test_core.py` | Delete tests for removed functions |
| `tests/watch/test_engine.py` | Delete (moved to tests/engine/) |
