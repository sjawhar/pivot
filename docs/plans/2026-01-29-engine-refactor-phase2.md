# Engine Refactor Phase 2: Engine Core

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the Engine class that coordinates execution, initially delegating to the existing executor while establishing the event-based notification system.

**Architecture:** Engine wraps the existing `executor.run()`, translating its notifications to the new event system. Event sinks subscribe to output events. This is a shim layer—behavior is unchanged but notifications flow through the new architecture.

**Tech Stack:** Python 3.13+, TypedDict, Protocol, threading (for event emission)

---

## Task 1: Add EventSource and EventSink Protocols

**Files:**
- Modify: `src/pivot/engine/types.py`
- Modify: `tests/engine/test_types.py`

**Step 1: Write failing tests for protocols**

Add to `tests/engine/test_types.py`:
```python
from collections.abc import Callable


def test_event_source_protocol() -> None:
    """EventSource protocol defines start/stop interface."""
    from pivot.engine.types import EventSource, InputEvent

    class MockSource:
        def start(self, submit: Callable[[InputEvent], None]) -> None:
            pass

        def stop(self) -> None:
            pass

    # Protocol should accept this implementation
    source: EventSource = MockSource()
    assert hasattr(source, "start")
    assert hasattr(source, "stop")


def test_event_sink_protocol() -> None:
    """EventSink protocol defines handle/close interface."""
    from pivot.engine.types import EventSink, OutputEvent

    class MockSink:
        def handle(self, event: OutputEvent) -> None:
            pass

        def close(self) -> None:
            pass

    # Protocol should accept this implementation
    sink: EventSink = MockSink()
    assert hasattr(sink, "handle")
    assert hasattr(sink, "close")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_types.py::test_event_source_protocol -v
```

Expected: FAIL with `ImportError: cannot import name 'EventSource'`

**Step 3: Write minimal implementation**

Add to `src/pivot/engine/types.py` imports:
```python
from collections.abc import Callable
from typing import Protocol
```

Update `__all__`:
```python
__all__ = [
    # ... existing exports ...
    # Protocols
    "EventSource",
    "EventSink",
]
```

Add after OutputEvent definition:
```python
# =============================================================================
# Protocols
# =============================================================================


class EventSource(Protocol):
    """Source that produces input events."""

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Begin producing events. Call submit() for each event."""
        ...

    def stop(self) -> None:
        """Stop producing events."""
        ...


class EventSink(Protocol):
    """Sink that consumes output events."""

    def handle(self, event: OutputEvent) -> None:
        """Process an event. Must be non-blocking."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_types.py -v
```

Expected: PASS (all tests including the 2 new ones)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add EventSource and EventSink protocols"
```

---

## Task 2: Create Engine Module Structure

**Files:**
- Create: `src/pivot/engine/engine.py`
- Create: `tests/engine/test_engine.py`
- Modify: `src/pivot/engine/__init__.py`

**Step 1: Write failing test for Engine instantiation**

Create `tests/engine/test_engine.py`:
```python
"""Tests for the Engine class."""

from __future__ import annotations

from pivot.engine import engine, types


def test_engine_initial_state_is_idle() -> None:
    """Engine starts in IDLE state."""
    eng = engine.Engine()
    assert eng.state == types.EngineState.IDLE


def test_engine_has_empty_sinks_initially() -> None:
    """Engine has no sinks until registered."""
    eng = engine.Engine()
    assert eng.sinks == []


def test_engine_graph_is_none_initially() -> None:
    """Engine graph is None until built."""
    eng = engine.Engine()
    assert eng.graph is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_initial_state_is_idle -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pivot.engine.engine'`

**Step 3: Write minimal implementation**

Create `src/pivot/engine/engine.py`:
```python
"""Engine: the central coordinator for event-driven pipeline execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.engine.types import EngineState, EventSink

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["Engine"]


class Engine:
    """Central coordinator for pipeline execution.

    The Engine is the single source of truth for execution state. All code paths
    (CLI, watch mode, agent RPC) go through the Engine.
    """

    def __init__(self) -> None:
        """Initialize the engine in IDLE state."""
        self._state = EngineState.IDLE
        self._sinks = list[EventSink]()
        self._graph: nx.DiGraph[str] | None = None

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        return self._state

    @property
    def sinks(self) -> list[EventSink]:
        """Registered event sinks."""
        return self._sinks

    @property
    def graph(self) -> nx.DiGraph[str] | None:
        """Current bipartite artifact-stage graph.

        Returns None until the graph is built (typically on first run or
        after registry reload). Status/verify code can query this graph
        to understand artifact-stage relationships.
        """
        return self._graph
```

Update `src/pivot/engine/__init__.py`:
```python
"""Engine module for event-driven pipeline execution."""

from __future__ import annotations

from pivot.engine import engine, graph, types

__all__ = ["engine", "graph", "types"]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_engine.py -v
```

Expected: PASS (all 3 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add Engine class skeleton

Initial state tracking, sink registration, graph property for queries."
```

---

## Task 3: Add Sink Registration and Event Emission

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing tests for sink management**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_add_sink() -> None:
    """Engine can register event sinks."""

    class MockSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    eng = engine.Engine()
    sink = MockSink()
    eng.add_sink(sink)

    assert len(eng.sinks) == 1
    assert eng.sinks[0] is sink


def test_engine_emit_sends_to_all_sinks() -> None:
    """Engine.emit() sends event to all registered sinks."""

    class MockSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    eng = engine.Engine()
    sink1 = MockSink()
    sink2 = MockSink()
    eng.add_sink(sink1)
    eng.add_sink(sink2)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    eng.emit(event)

    assert sink1.events == [event]
    assert sink2.events == [event]


def test_engine_close_closes_all_sinks() -> None:
    """Engine.close() calls close() on all sinks."""

    class MockSink:
        def __init__(self) -> None:
            self.closed = False

        def handle(self, event: types.OutputEvent) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    eng = engine.Engine()
    sink1 = MockSink()
    sink2 = MockSink()
    eng.add_sink(sink1)
    eng.add_sink(sink2)

    eng.close()

    assert sink1.closed
    assert sink2.closed
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_add_sink -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'add_sink'`

**Step 3: Write minimal implementation**

Add to `src/pivot/engine/engine.py` imports:
```python
from pivot.engine.types import EngineState, EventSink, OutputEvent
```

Add methods to the Engine class:
```python
    def add_sink(self, sink: EventSink) -> None:
        """Register an event sink to receive output events."""
        self._sinks.append(sink)

    def emit(self, event: OutputEvent) -> None:
        """Emit an event to all registered sinks."""
        for sink in self._sinks:
            sink.handle(event)

    def close(self) -> None:
        """Close all sinks and clean up resources."""
        for sink in self._sinks:
            sink.close()
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_engine.py -v
```

Expected: PASS (all 5 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add sink registration and event emission

add_sink(), emit(), close() methods."
```

---

## Task 4: Add run_once() Method Delegating to Executor

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for run_once signature**

Add to `tests/engine/test_engine.py`:
```python
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pivot.registry import REGISTRY


def test_engine_run_once_returns_execution_summary(tmp_path: Path) -> None:
    """run_once() returns dict mapping stage names to ExecutionSummary."""
    # Create minimal pipeline
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.write_text("a,b\n1,2\n")

    # Register a stage
    REGISTRY.clear()

    def _stage_func() -> None:
        output_file.write_text("processed")

    # Patch executor.run to avoid actual execution
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {"test_stage": {"status": "ran", "reason": ""}}

        eng = engine.Engine()
        result = eng.run_once()

        assert isinstance(result, dict)
        mock_executor.run.assert_called_once()


def test_engine_run_once_passes_stages_parameter() -> None:
    """run_once() passes stages parameter to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once(stages=["stage_a", "stage_b"])

        mock_executor.run.assert_called_once()
        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["stages"] == ["stage_a", "stage_b"]


def test_engine_run_once_passes_force_parameter() -> None:
    """run_once() passes force parameter to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once(force=True)

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["force"] is True


def test_engine_run_once_emits_state_changed_events() -> None:
    """run_once() emits engine state changed events."""

    class MockSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        sink = MockSink()
        eng.add_sink(sink)
        eng.run_once()

        # Should emit ACTIVE at start and IDLE at end
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 2
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[1]["state"] == types.EngineState.IDLE
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_once_returns_execution_summary -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'run_once'`

**Step 3: Write minimal implementation**

Add to `src/pivot/engine/engine.py` imports:
```python
from pivot import executor
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    EventSink,
    OutputEvent,
)
from pivot.executor.core import ExecutionSummary
```

Add the run_once method:
```python
    def run_once(
        self,
        stages: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, ExecutionSummary]:
        """Execute stages once and return.

        This is the primary entry point for 'pivot run' without --watch.
        Delegates to the existing executor while emitting events to sinks.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run all stages.

        Returns:
            Dict mapping stage name to ExecutionSummary.
        """
        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            # Delegate to existing executor
            result = executor.run(
                stages=stages,
                force=force,
            )
            return result
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_engine.py -v
```

Expected: PASS (all 9 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add run_once() delegating to executor

Wraps executor.run() with state transitions and event emission."
```

---

## Task 5: Add Executor Parameters to run_once()

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write tests for additional parameters**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_run_once_passes_all_executor_params() -> None:
    """run_once() passes through all relevant executor parameters."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once(
            stages=["stage_a"],
            force=True,
            single_stage=True,
            parallel=False,
            max_workers=4,
            no_commit=True,
            no_cache=True,
            show_output=True,
            allow_uncached_incremental=True,
            checkout_missing=True,
        )

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["stages"] == ["stage_a"]
        assert call_kwargs["force"] is True
        assert call_kwargs["single_stage"] is True
        assert call_kwargs["parallel"] is False
        assert call_kwargs["max_workers"] == 4
        assert call_kwargs["no_commit"] is True
        assert call_kwargs["no_cache"] is True
        assert call_kwargs["show_output"] is True
        assert call_kwargs["allow_uncached_incremental"] is True
        assert call_kwargs["checkout_missing"] is True
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_once_passes_all_executor_params -v
```

Expected: FAIL with `TypeError: Engine.run_once() got an unexpected keyword argument 'single_stage'`

**Step 3: Write minimal implementation**

Update the `run_once` method signature and body:
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
        show_output: bool = True,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
    ) -> dict[str, ExecutionSummary]:
        """Execute stages once and return.

        This is the primary entry point for 'pivot run' without --watch.
        Delegates to the existing executor while emitting events to sinks.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run all stages.
            single_stage: If True, run only the specified stages (no downstream).
            parallel: If True, run stages in parallel.
            max_workers: Maximum worker processes.
            no_commit: If True, don't update lockfiles.
            no_cache: If True, disable run cache.
            show_output: If True, show stage output to console.
            allow_uncached_incremental: Allow incremental outputs without cache.
            checkout_missing: Checkout missing dependency files from cache.

        Returns:
            Dict mapping stage name to ExecutionSummary.
        """
        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            # Delegate to existing executor
            result = executor.run(
                stages=stages,
                force=force,
                single_stage=single_stage,
                parallel=parallel,
                max_workers=max_workers,
                no_commit=no_commit,
                no_cache=no_cache,
                show_output=show_output,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
            return result
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_engine.py -v
```

Expected: PASS (all 10 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add all executor parameters to run_once()

single_stage, parallel, max_workers, no_commit, no_cache, etc."
```

---

## Task 6: Create ConsoleSink Adapter

**Files:**
- Create: `src/pivot/engine/sinks.py`
- Create: `tests/engine/test_sinks.py`
- Modify: `src/pivot/engine/__init__.py`

**Step 1: Write failing test for ConsoleSink**

Create `tests/engine/test_sinks.py`:
```python
"""Tests for event sinks."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from pivot.engine import sinks, types
from pivot.types import StageStatus


def test_console_sink_handles_stage_started() -> None:
    """ConsoleSink prints stage start message."""
    stream = StringIO()
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    console_mock.stage_start.assert_called_once()
    call_args = console_mock.stage_start.call_args
    assert call_args.kwargs["name"] == "train"
    assert call_args.kwargs["index"] == 2
    assert call_args.kwargs["total"] == 5


def test_console_sink_handles_stage_completed() -> None:
    """ConsoleSink prints stage result message."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
    }
    sink.handle(event)

    console_mock.stage_result.assert_called_once()
    call_args = console_mock.stage_result.call_args
    assert call_args.kwargs["name"] == "train"
    assert call_args.kwargs["status"] == StageStatus.RAN
    assert call_args.kwargs["reason"] == "inputs changed"
    # Duration should be converted from ms to seconds
    assert call_args.kwargs["duration"] == 1.2345


def test_console_sink_ignores_other_events() -> None:
    """ConsoleSink ignores events it doesn't handle."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    sink.handle(event)

    # Should not call any console methods
    console_mock.stage_start.assert_not_called()
    console_mock.stage_result.assert_not_called()


def test_console_sink_close_closes_console() -> None:
    """ConsoleSink.close() calls console.close()."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    sink.close()

    console_mock.close.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sinks.py::test_console_sink_handles_stage_started -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pivot.engine.sinks'`

**Step 3: Write minimal implementation**

Create `src/pivot/engine/sinks.py`:
```python
"""Event sink implementations for the engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.engine.types import OutputEvent, StageCompleted, StageStarted
from pivot.types import StageDisplayStatus

if TYPE_CHECKING:
    from pivot.tui.console import Console

__all__ = ["ConsoleSink"]


class ConsoleSink:
    """Event sink that outputs to console with colors and progress tracking."""

    def __init__(self, console: Console) -> None:
        """Initialize with a Console instance.

        Args:
            console: The Console to write to.
        """
        self._console = console

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by writing to console."""
        match event["type"]:
            case "stage_started":
                self._handle_stage_started(event)
            case "stage_completed":
                self._handle_stage_completed(event)
            case _:
                pass  # Ignore other event types

    def _handle_stage_started(self, event: StageStarted) -> None:
        """Handle stage started event."""
        self._console.stage_start(
            name=event["stage"],
            index=event["index"],
            total=event["total"],
            status=StageDisplayStatus.RUNNING,
        )

    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        self._console.stage_result(
            name=event["stage"],
            index=event["index"] if "index" in event else 0,
            total=event["total"] if "total" in event else 0,
            status=event["status"],
            reason=event["reason"],
            duration=event["duration_ms"] / 1000.0,  # Convert ms to seconds
        )

    def close(self) -> None:
        """Close the underlying console."""
        self._console.close()
```

Update `src/pivot/engine/__init__.py`:
```python
"""Engine module for event-driven pipeline execution."""

from __future__ import annotations

from pivot.engine import engine, graph, sinks, types

__all__ = ["engine", "graph", "sinks", "types"]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sinks.py -v
```

Expected: PASS (all 4 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add ConsoleSink for console output

Translates StageStarted/StageCompleted events to Console methods."
```

---

## Task 7: Add Index/Total to StageCompleted Event

The current `StageCompleted` event doesn't have `index` and `total` fields, but the console needs them. We need to add these optional fields or track them in the sink.

**Files:**
- Modify: `src/pivot/engine/types.py`
- Modify: `src/pivot/engine/sinks.py`
- Modify: `tests/engine/test_types.py`

**Step 1: Update StageCompleted to include index/total**

Update `src/pivot/engine/types.py`, modify the `StageCompleted` class:
```python
class StageCompleted(TypedDict):
    """A stage finished (ran, skipped, or failed)."""

    type: Literal["stage_completed"]
    stage: str
    status: StageStatus
    reason: str
    duration_ms: float
    index: int
    total: int
```

**Step 2: Update tests that create StageCompleted events**

Update `tests/engine/test_types.py`:
```python
def test_stage_completed_event() -> None:
    """StageCompleted event has required fields."""
    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
        "index": 3,
        "total": 5,
    }
    assert event["type"] == "stage_completed"
    assert event["status"] == StageStatus.RAN
    assert event["index"] == 3
    assert event["total"] == 5

    # Skipped stage
    event_skip: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "evaluate",
        "status": StageStatus.SKIPPED,
        "reason": "unchanged",
        "duration_ms": 0.0,
        "index": 4,
        "total": 5,
    }
    assert event_skip["status"] == StageStatus.SKIPPED


def test_output_event_union() -> None:
    """OutputEvent is a union of all output event types."""
    events: list[types.OutputEvent] = [
        {"type": "engine_state_changed", "state": types.EngineState.IDLE},
        {
            "type": "pipeline_reloaded",
            "stages_added": [],
            "stages_removed": [],
            "stages_modified": [],
            "error": None,
        },
        {"type": "stage_started", "stage": "x", "index": 1, "total": 1},
        {
            "type": "stage_completed",
            "stage": "x",
            "status": StageStatus.RAN,
            "reason": "",
            "duration_ms": 0,
            "index": 1,
            "total": 1,
        },
        {"type": "log_line", "stage": "x", "line": "", "is_stderr": False},
    ]
    assert len(events) == 5
```

**Step 3: Update ConsoleSink to use the new fields**

Update `src/pivot/engine/sinks.py`:
```python
    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        self._console.stage_result(
            name=event["stage"],
            index=event["index"],
            total=event["total"],
            status=event["status"],
            reason=event["reason"],
            duration=event["duration_ms"] / 1000.0,  # Convert ms to seconds
        )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/ -v
```

Expected: PASS (all tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "fix(engine): add index/total to StageCompleted event

Required for console output formatting."
```

---

## Task 8: Create TuiSink Adapter

**Files:**
- Modify: `src/pivot/engine/sinks.py`
- Modify: `tests/engine/test_sinks.py`

**Step 1: Write failing test for TuiSink**

Add to `tests/engine/test_sinks.py`:
```python
import queue

from pivot.types import TuiQueue, TuiMessageType


def test_tui_sink_handles_stage_started() -> None:
    """TuiSink sends status message for stage started."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "train"
    assert msg["index"] == 2
    assert msg["total"] == 5


def test_tui_sink_handles_stage_completed() -> None:
    """TuiSink sends status message for stage completed."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "train"
    assert msg["status"] == StageStatus.RAN


def test_tui_sink_handles_log_line() -> None:
    """TuiSink sends log message for log line."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    event: types.LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Epoch 1/10",
        "is_stderr": False,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.LOG
    assert msg["stage"] == "train"
    assert msg["line"] == "Epoch 1/10"
    assert msg["is_stderr"] is False


def test_tui_sink_close_sends_none() -> None:
    """TuiSink.close() sends None to signal termination."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    sink.close()

    msg = tui_queue.get_nowait()
    assert msg is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sinks.py::test_tui_sink_handles_stage_started -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.sinks' has no attribute 'TuiSink'`

**Step 3: Write minimal implementation**

Add imports to `src/pivot/engine/sinks.py`:
```python
import time

from pivot.engine.types import LogLine, OutputEvent, StageCompleted, StageStarted
from pivot.types import (
    StageDisplayStatus,
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiQueue,
    TuiStatusMessage,
)
```

Update `__all__`:
```python
__all__ = ["ConsoleSink", "TuiSink"]
```

Add the TuiSink class:
```python
class TuiSink:
    """Event sink that forwards events to TUI via queue."""

    def __init__(self, tui_queue: TuiQueue, run_id: str) -> None:
        """Initialize with a TUI message queue.

        Args:
            tui_queue: Queue to send TUI messages to.
            run_id: Unique identifier for this run.
        """
        self._queue = tui_queue
        self._run_id = run_id

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by sending to TUI queue."""
        match event["type"]:
            case "stage_started":
                self._handle_stage_started(event)
            case "stage_completed":
                self._handle_stage_completed(event)
            case "log_line":
                self._handle_log_line(event)
            case _:
                pass  # Ignore other event types

    def _handle_stage_started(self, event: StageStarted) -> None:
        """Handle stage started event."""
        msg = TuiStatusMessage(
            type=TuiMessageType.STATUS,
            stage=event["stage"],
            index=event["index"],
            total=event["total"],
            status=StageStatus.READY,  # Starting = not yet complete
            reason="",
            elapsed=None,
            run_id=self._run_id,
        )
        self._queue.put(msg)

    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        msg = TuiStatusMessage(
            type=TuiMessageType.STATUS,
            stage=event["stage"],
            index=event["index"],
            total=event["total"],
            status=event["status"],
            reason=event["reason"],
            elapsed=event["duration_ms"] / 1000.0,
            run_id=self._run_id,
        )
        self._queue.put(msg)

    def _handle_log_line(self, event: LogLine) -> None:
        """Handle log line event."""
        msg = TuiLogMessage(
            type=TuiMessageType.LOG,
            stage=event["stage"],
            line=event["line"],
            is_stderr=event["is_stderr"],
            timestamp=time.time(),
        )
        self._queue.put(msg)

    def close(self) -> None:
        """Signal TUI termination by sending None."""
        self._queue.put(None)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sinks.py -v
```

Expected: PASS (all 8 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add TuiSink for TUI integration

Translates events to TuiMessage and forwards via queue."
```

---

## Task 9: Create JsonlSink Adapter

**Files:**
- Modify: `src/pivot/engine/sinks.py`
- Modify: `tests/engine/test_sinks.py`

**Step 1: Write failing test for JsonlSink**

Add to `tests/engine/test_sinks.py`:
```python
import json


def test_jsonl_sink_handles_stage_started() -> None:
    """JsonlSink calls callback with StageStartEvent dict."""
    events_received = list[dict[str, object]]()

    def callback(event: dict[str, object]) -> None:
        events_received.append(event)

    sink = sinks.JsonlSink(callback=callback)

    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    assert len(events_received) == 1
    assert events_received[0]["type"] == "stage_start"
    assert events_received[0]["stage"] == "train"
    assert events_received[0]["index"] == 2
    assert events_received[0]["total"] == 5


def test_jsonl_sink_handles_stage_completed() -> None:
    """JsonlSink calls callback with StageCompleteEvent dict."""
    events_received = list[dict[str, object]]()

    def callback(event: dict[str, object]) -> None:
        events_received.append(event)

    sink = sinks.JsonlSink(callback=callback)

    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    assert len(events_received) == 1
    assert events_received[0]["type"] == "stage_complete"
    assert events_received[0]["stage"] == "train"
    assert events_received[0]["status"] == "ran"


def test_jsonl_sink_close_is_noop() -> None:
    """JsonlSink.close() does nothing."""
    sink = sinks.JsonlSink(callback=lambda _: None)
    sink.close()  # Should not raise
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sinks.py::test_jsonl_sink_handles_stage_started -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.sinks' has no attribute 'JsonlSink'`

**Step 3: Write minimal implementation**

Add imports to `src/pivot/engine/sinks.py`:
```python
from collections.abc import Callable
```

Update `__all__`:
```python
__all__ = ["ConsoleSink", "JsonlSink", "TuiSink"]
```

Add the JsonlSink class:
```python
class JsonlSink:
    """Event sink that emits JSONL events via callback.

    Translates engine events to the existing RunJsonEvent format for
    backwards compatibility with --json output.
    """

    def __init__(self, callback: Callable[[dict[str, object]], None]) -> None:
        """Initialize with a callback that receives event dicts.

        Args:
            callback: Function called with each event dict (for JSON serialization).
        """
        self._callback = callback

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by converting and calling callback."""
        match event["type"]:
            case "stage_started":
                self._handle_stage_started(event)
            case "stage_completed":
                self._handle_stage_completed(event)
            case _:
                pass  # Ignore other event types

    def _handle_stage_started(self, event: StageStarted) -> None:
        """Handle stage started event."""
        json_event: dict[str, object] = {
            "type": "stage_start",
            "stage": event["stage"],
            "index": event["index"],
            "total": event["total"],
        }
        self._callback(json_event)

    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        json_event: dict[str, object] = {
            "type": "stage_complete",
            "stage": event["stage"],
            "status": event["status"].value,  # Convert enum to string
            "reason": event["reason"],
            "duration_ms": event["duration_ms"],
            "index": event["index"],
            "total": event["total"],
        }
        self._callback(json_event)

    def close(self) -> None:
        """No cleanup needed for callback-based sink."""
        pass
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sinks.py -v
```

Expected: PASS (all 11 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add JsonlSink for JSON streaming output

Translates events to RunJsonEvent format for --json compatibility."
```

---

## Task 10: Final Integration Test

**Files:**
- Modify: `tests/engine/test_engine.py`

**Step 1: Write integration test for Engine with sinks**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_integration_with_sinks() -> None:
    """Engine correctly routes events to multiple sinks."""

    class RecordingSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()
            self.closed = False

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            self.closed = True

    with patch("pivot.engine.engine.executor") as mock_executor:
        # Simulate executor returning results
        mock_executor.run.return_value = {
            "stage_a": {"status": "ran", "reason": "inputs changed"},
        }

        eng = engine.Engine()
        sink1 = RecordingSink()
        sink2 = RecordingSink()
        eng.add_sink(sink1)
        eng.add_sink(sink2)

        result = eng.run_once(stages=["stage_a"])

        # Verify results returned
        assert "stage_a" in result

        # Verify both sinks received events
        assert len(sink1.events) >= 2  # At least ACTIVE and IDLE state changes
        assert len(sink2.events) >= 2
        assert sink1.events == sink2.events  # Same events to both

        # Verify state change events
        state_events = [e for e in sink1.events if e["type"] == "engine_state_changed"]
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[1]["state"] == types.EngineState.IDLE

        # Close and verify
        eng.close()
        assert sink1.closed
        assert sink2.closed
```

**Step 2: Run test to verify it passes**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_integration_with_sinks -v
```

Expected: PASS

**Step 3: Run full engine test suite**

```bash
uv run pytest tests/engine/ -v
```

Expected: PASS (all tests)

**Step 4: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

Expected: No errors

**Step 5: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

Expected: All existing tests still pass

**Step 6: Final commit**

```bash
jj describe -m "test(engine): add integration test for Engine with sinks

Verifies event routing and sink lifecycle."
```

---

## Task 11: Add Missing Parameters to Engine.run_once()

**Goal:** Extend Engine.run_once() to support all parameters needed by CLI (on_error, cache_dir, tui_queue, output_queue, progress_callback, etc.).

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for missing parameters**

Add to `tests/engine/test_engine.py`:
```python
import pathlib
from pivot.types import OnError


def test_engine_run_once_passes_on_error_parameter() -> None:
    """run_once() passes on_error parameter to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once(on_error=OnError.KEEP_GOING)

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["on_error"] == OnError.KEEP_GOING


def test_engine_run_once_passes_cache_dir_parameter() -> None:
    """run_once() passes cache_dir parameter to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        cache_dir = pathlib.Path("/tmp/test-cache")
        eng.run_once(cache_dir=cache_dir)

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["cache_dir"] == cache_dir


def test_engine_run_once_passes_tui_queue_parameter() -> None:
    """run_once() passes tui_queue parameter to executor."""
    import queue as thread_queue

    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        tui_queue: thread_queue.Queue[object] = thread_queue.Queue()
        eng.run_once(tui_queue=tui_queue)

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["tui_queue"] is tui_queue


def test_engine_run_once_passes_progress_callback_parameter() -> None:
    """run_once() passes progress_callback parameter to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        def callback(event: object) -> None:
            pass

        eng = engine.Engine()
        eng.run_once(progress_callback=callback)

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["progress_callback"] is callback
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_once_passes_on_error_parameter -v
```

Expected: FAIL with `TypeError: Engine.run_once() got an unexpected keyword argument 'on_error'`

**Step 3: Write minimal implementation**

Update `src/pivot/engine/engine.py` imports:
```python
import multiprocessing as mp
import pathlib

from pivot.types import OnError, OutputMessage, RunJsonEvent, TuiQueue
```

Update the `run_once` method signature:
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
        show_output: bool = True,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
        on_error: OnError = OnError.FAIL,
        cache_dir: pathlib.Path | None = None,
        tui_queue: TuiQueue | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
        progress_callback: Callable[[RunJsonEvent], None] | None = None,
        explain_mode: bool = False,
    ) -> dict[str, ExecutionSummary]:
```

Update the executor.run() call inside run_once:
```python
            result = executor.run(
                stages=stages,
                force=force,
                single_stage=single_stage,
                parallel=parallel,
                max_workers=max_workers,
                no_commit=no_commit,
                no_cache=no_cache,
                show_output=show_output,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
                on_error=on_error,
                cache_dir=cache_dir,
                tui_queue=tui_queue,
                output_queue=output_queue,
                progress_callback=progress_callback,
                explain_mode=explain_mode,
                cancel_event=self._cancel_event,
            )
```

Also add import for Callable in TYPE_CHECKING block:
```python
if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.executor.core import ExecutionSummary
    from pivot.types import OnError, OutputMessage, RunJsonEvent, TuiQueue
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_engine.py -v
```

Expected: PASS (all tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add CLI parameters to run_once()

on_error, cache_dir, tui_queue, output_queue, progress_callback, explain_mode."
```

---

## Task 12: Route Plain Mode Through Engine

**Goal:** Change CLI plain mode (non-TUI, non-JSON) to use Engine.run_once() instead of calling executor.run() directly.

**Files:**
- Modify: `src/pivot/cli/run.py`
- Modify: `tests/cli/test_run.py` (if needed)

**Step 1: Update plain mode execution in CLI**

In `src/pivot/cli/run.py`, find the plain mode execution block (around line 541-554):
```python
    else:
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=explain,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            show_output=not quiet,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
```

Replace with:
```python
    else:
        from pivot.engine import engine as engine_mod

        eng = engine_mod.Engine()
        try:
            results = eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                explain_mode=explain,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                show_output=not quiet,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
```

**Step 2: Run CLI integration tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "not tui and not json and not watch"
```

Expected: PASS (all non-TUI, non-JSON, non-watch tests)

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/cli && uv run ruff check src/pivot/cli && uv run basedpyright src/pivot/cli
```

**Step 4: Commit**

```bash
jj describe -m "feat(cli): route plain mode through Engine

First step in CLI migration - plain mode uses Engine.run_once()."
```

---

## Task 13: Route JSON Mode Through Engine

**Goal:** Change CLI JSON mode to use Engine.run_once() instead of calling executor.run() directly.

**Files:**
- Modify: `src/pivot/cli/run.py`

**Step 1: Update JSON mode execution in CLI**

In `src/pivot/cli/run.py`, find the JSON mode execution block (around line 503-540):
```python
    elif as_json:
        # JSONL streaming mode
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

        start_time = time.perf_counter()
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=False,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            show_output=False,
            progress_callback=cli_helpers.emit_jsonl,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
        ...
```

Replace with:
```python
    elif as_json:
        from pivot.engine import engine as engine_mod

        # JSONL streaming mode
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

        start_time = time.perf_counter()
        eng = engine_mod.Engine()
        try:
            results = eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                explain_mode=False,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                show_output=False,
                progress_callback=cli_helpers.emit_jsonl,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
        ...
```

**Step 2: Run CLI JSON tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "json"
```

Expected: PASS (all JSON tests)

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/cli && uv run ruff check src/pivot/cli && uv run basedpyright src/pivot/cli
```

**Step 4: Commit**

```bash
jj describe -m "feat(cli): route JSON mode through Engine

JSONL streaming now uses Engine.run_once()."
```

---

## Task 14: Route TUI Mode Through Engine

**Goal:** Change CLI TUI mode to use Engine instead of calling executor.run() directly.

**Files:**
- Modify: `src/pivot/cli/run.py`

**Step 1: Update TUI mode execution in CLI**

In `src/pivot/cli/run.py`, find the `_run_with_tui` function. The key change is to replace the `executor_func` that calls `executor.run()` with one that uses Engine:

Update the `executor_func` inside `_run_with_tui` (around line 185-199):
```python
    # Create executor function that passes the TUI queue and cancel event
    def executor_func() -> dict[str, ExecutionSummary]:
        return executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=resolved_cache_dir,
            show_output=False,
            tui_queue=tui_queue,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
            cancel_event=cancel_event,
        )
```

Replace with:
```python
    from pivot.engine import engine as engine_mod

    # Create executor function using Engine
    def executor_func() -> dict[str, ExecutionSummary]:
        eng = engine_mod.Engine()
        # Use TUI's cancel_event directly
        eng._cancel_event = cancel_event
        try:
            return eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=resolved_cache_dir,
                show_output=False,
                tui_queue=tui_queue,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
```

**Step 2: Run TUI tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "tui"
uv run pytest tests/tui/ -v
```

Expected: PASS (all TUI tests)

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

Expected: PASS (all tests)

**Step 4: Run quality checks**

```bash
uv run ruff format src/pivot/cli && uv run ruff check src/pivot/cli && uv run basedpyright src/pivot/cli
```

**Step 5: Commit**

```bash
jj describe -m "feat(cli): route TUI mode through Engine

All non-watch CLI paths now use Engine."
```

---

## Task 15: Final Verification and Cleanup

**Goal:** Verify all CLI execution paths use Engine and update documentation.

**Files:**
- Modify: `src/pivot/cli/run.py` (cleanup imports)

**Step 1: Remove unused executor import from CLI if possible**

Check if `executor` is still needed in `cli/run.py`. It may still be needed for:
- `executor.prepare_workers()` - keep this
- Watch mode still uses WatchEngine (not migrated yet)

If the only remaining use is `prepare_workers`, update import to be more specific:
```python
from pivot.executor import prepare_workers
```

Then update calls from `executor.prepare_workers(...)` to `prepare_workers(...)`.

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

Expected: PASS (all tests)

**Step 3: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

Expected: No errors

**Step 4: Commit**

```bash
jj describe -m "refactor(cli): cleanup after Engine migration

Remove unused imports, update documentation."
```

---

## Summary

After completing Phase 2, you will have:

1. **`src/pivot/engine/types.py`** - Updated with:
   - `EventSource` protocol (start, stop)
   - `EventSink` protocol (handle, close)
   - `StageCompleted` with index/total fields

2. **`src/pivot/engine/engine.py`** - Engine class with:
   - State tracking (IDLE, ACTIVE, SHUTDOWN)
   - Graph property for status/verify queries (`graph`)
   - Sink registration (`add_sink`)
   - Event emission (`emit`)
   - `run_once()` with full parameter support (matching executor.run())

3. **`src/pivot/engine/sinks.py`** - Event sink implementations:
   - `ConsoleSink` - wraps `Console` for plain text output
   - `TuiSink` - forwards to TUI via queue
   - `JsonlSink` - emits JSONL via callback

4. **`src/pivot/cli/run.py`** - Updated to route through Engine:
   - Plain mode uses Engine.run_once()
   - JSON mode uses Engine.run_once()
   - TUI mode uses Engine.run_once()
   - Watch mode still uses WatchEngine (Phase 3)

5. **Tests** - Comprehensive coverage for all new code

**Behavior is unchanged.** All CLI execution paths (except watch mode) now go through the Engine, which currently delegates to executor.run(). This establishes the single entry point for future enhancements.

**Next (Phase 3):**
1. Unify watch mode to use Engine.run_loop()
2. Have Engine emit StageStarted/StageCompleted events during execution
3. Eventually move execution logic from executor into Engine
