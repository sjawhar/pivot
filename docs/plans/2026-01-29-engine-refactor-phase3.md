# Engine Refactor Phase 3: Watch Mode Unification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify watch mode with the Engine by creating FilesystemSource, implementing Engine.run_loop(), and routing watch mode through the Engine while preserving all existing behavior.

**Architecture:** FilesystemSource wraps watchfiles and produces DataArtifactChanged/CodeOrConfigChanged events. Engine.run_loop() processes these events in a continuous loop. The existing WatchEngine is refactored to delegate execution to Engine while retaining watch-specific logic (debouncing, registry reload, watcher management).

**Tech Stack:** Python 3.13+, watchfiles, threading, Protocol

---

## Task 1: Create FilesystemSource Skeleton

**Files:**
- Create: `src/pivot/engine/sources.py`
- Create: `tests/engine/test_sources.py`
- Modify: `src/pivot/engine/__init__.py`

**Step 1: Write failing test for FilesystemSource instantiation**

Create `tests/engine/test_sources.py`:
```python
"""Tests for event sources."""

from __future__ import annotations

from pathlib import Path

from pivot.engine import sources, types


def test_filesystem_source_instantiation() -> None:
    """FilesystemSource can be instantiated with watch paths."""
    source = sources.FilesystemSource(watch_paths=[Path("/tmp/test")])
    assert hasattr(source, "start")
    assert hasattr(source, "stop")


def test_filesystem_source_conforms_to_protocol() -> None:
    """FilesystemSource conforms to EventSource protocol."""
    source = sources.FilesystemSource(watch_paths=[])
    # Protocol conformance: has start(submit) and stop()
    _source: types.EventSource = source
    assert _source is source
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sources.py::test_filesystem_source_instantiation -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pivot.engine.sources'`

**Step 3: Write minimal implementation**

Create `src/pivot/engine/sources.py`:
```python
"""Event source implementations for the engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pivot.engine.types import InputEvent

__all__ = ["FilesystemSource"]


class FilesystemSource:
    """Event source that watches filesystem for changes.

    Wraps watchfiles to detect file changes and emit DataArtifactChanged
    or CodeOrConfigChanged events.
    """

    _watch_paths: list[Path]
    _submit: Callable[[InputEvent], None] | None
    _running: bool

    def __init__(self, watch_paths: list[Path]) -> None:
        """Initialize with paths to watch.

        Args:
            watch_paths: List of paths to watch for changes.
        """
        self._watch_paths = list(watch_paths)
        self._submit = None
        self._running = False

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Begin producing events. Call submit() for each event."""
        self._submit = submit
        self._running = True

    def stop(self) -> None:
        """Stop producing events."""
        self._running = False
        self._submit = None
```

Update `src/pivot/engine/__init__.py`:
```python
"""Engine module for event-driven pipeline execution."""

from __future__ import annotations

from pivot.engine import engine, graph, sinks, sources, types

__all__ = ["engine", "graph", "sinks", "sources", "types"]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sources.py -v
```

Expected: PASS (both tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add FilesystemSource skeleton

Implements EventSource protocol for filesystem watching."
```

---

## Task 2: Add set_watch_paths Method to FilesystemSource

**Files:**
- Modify: `src/pivot/engine/sources.py`
- Modify: `tests/engine/test_sources.py`

**Step 1: Write failing test for set_watch_paths**

Add to `tests/engine/test_sources.py`:
```python
def test_filesystem_source_set_watch_paths() -> None:
    """FilesystemSource.set_watch_paths() updates watched paths."""
    source = sources.FilesystemSource(watch_paths=[Path("/tmp/a")])

    new_paths = [Path("/tmp/b"), Path("/tmp/c")]
    source.set_watch_paths(new_paths)

    assert source.watch_paths == new_paths


def test_filesystem_source_watch_paths_property() -> None:
    """FilesystemSource.watch_paths returns current paths."""
    paths = [Path("/tmp/test1"), Path("/tmp/test2")]
    source = sources.FilesystemSource(watch_paths=paths)

    assert source.watch_paths == paths
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sources.py::test_filesystem_source_set_watch_paths -v
```

Expected: FAIL with `AttributeError: 'FilesystemSource' object has no attribute 'set_watch_paths'`

**Step 3: Write minimal implementation**

Add to `src/pivot/engine/sources.py` in the FilesystemSource class:
```python
    @property
    def watch_paths(self) -> list[Path]:
        """Current paths being watched."""
        return list(self._watch_paths)

    def set_watch_paths(self, paths: list[Path]) -> None:
        """Update watched paths.

        Called by engine when DAG changes and watch scope needs updating.
        If watcher is running, it will be restarted with new paths.

        Args:
            paths: New list of paths to watch.
        """
        self._watch_paths = list(paths)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sources.py -v
```

Expected: PASS (all 4 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add set_watch_paths to FilesystemSource

Enables dynamic update of watched paths when DAG changes."
```

---

## Task 3: Create OneShotSource for Non-Watch Mode

**Files:**
- Modify: `src/pivot/engine/sources.py`
- Modify: `tests/engine/test_sources.py`

**Step 1: Write failing test for OneShotSource**

Add to `tests/engine/test_sources.py`:
```python
def test_oneshot_source_emits_run_requested() -> None:
    """OneShotSource emits a single RunRequested event then stops."""
    events_received = list[types.InputEvent]()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)

    source = sources.OneShotSource(
        stages=["train", "evaluate"],
        force=True,
        reason="cli",
    )
    source.start(submit)

    assert len(events_received) == 1
    event = events_received[0]
    assert event["type"] == "run_requested"
    assert event["stages"] == ["train", "evaluate"]
    assert event["force"] is True
    assert event["reason"] == "cli"


def test_oneshot_source_with_none_stages() -> None:
    """OneShotSource with stages=None emits event with stages=None."""
    events_received = list[types.InputEvent]()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)

    source = sources.OneShotSource(stages=None, force=False, reason="test")
    source.start(submit)

    assert len(events_received) == 1
    assert events_received[0]["stages"] is None


def test_oneshot_source_stop_is_noop() -> None:
    """OneShotSource.stop() is safe to call multiple times."""
    source = sources.OneShotSource(stages=None, force=False, reason="test")
    source.stop()  # Should not raise
    source.stop()  # Should not raise
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sources.py::test_oneshot_source_emits_run_requested -v
```

Expected: FAIL with `AttributeError: module 'pivot.engine.sources' has no attribute 'OneShotSource'`

**Step 3: Write minimal implementation**

Update `__all__` in `src/pivot/engine/sources.py`:
```python
__all__ = ["FilesystemSource", "OneShotSource"]
```

Add the OneShotSource class:
```python
class OneShotSource:
    """Event source that emits a single RunRequested event.

    Used for 'pivot run' without --watch. Emits the run request
    immediately when start() is called, then becomes inactive.
    """

    _stages: list[str] | None
    _force: bool
    _reason: str
    _emitted: bool

    def __init__(
        self,
        stages: list[str] | None,
        force: bool,
        reason: str,
    ) -> None:
        """Initialize with run parameters.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run.
            reason: Description of why this run was requested.
        """
        self._stages = stages
        self._force = force
        self._reason = reason
        self._emitted = False

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Emit a single RunRequested event."""
        if self._emitted:
            return

        from pivot.engine.types import RunRequested

        event = RunRequested(
            type="run_requested",
            stages=self._stages,
            force=self._force,
            reason=self._reason,
        )
        submit(event)
        self._emitted = True

    def stop(self) -> None:
        """No-op for one-shot source."""
        pass
```

Add import at top of file:
```python
if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pivot.engine.types import InputEvent
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sources.py -v
```

Expected: PASS (all 7 tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): add OneShotSource for single run requests

Emits one RunRequested event for 'pivot run' without --watch."
```

---

## Task 4: Add Source Registration to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for add_source**

Add to `tests/engine/test_engine.py`:
```python
from pivot.engine import sources


def test_engine_has_empty_sources_initially() -> None:
    """Engine has no sources until registered."""
    eng = engine.Engine()
    assert eng.sources == []


def test_engine_add_source() -> None:
    """Engine can register event sources."""
    eng = engine.Engine()
    source = sources.OneShotSource(stages=None, force=False, reason="test")
    eng.add_source(source)

    assert len(eng.sources) == 1
    assert eng.sources[0] is source
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_has_empty_sources_initially -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'sources'`

**Step 3: Write minimal implementation**

Add imports to `src/pivot/engine/engine.py`:
```python
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    OutputEvent,
)
```

Update Engine.__init__:
```python
    def __init__(self) -> None:
        """Initialize the engine in IDLE state."""
        self._state: EngineState = EngineState.IDLE
        self._sinks: list[EventSink] = list[EventSink]()
        self._sources: list[EventSource] = list[EventSource]()
```

Add sources property and add_source method:
```python
    @property
    def sources(self) -> list[EventSource]:
        """Registered event sources."""
        return self._sources

    def add_source(self, source: EventSource) -> None:
        """Register an event source to produce input events."""
        self._sources.append(source)
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
jj describe -m "feat(engine): add source registration to Engine

add_source() method and sources property."
```

---

## Task 5: Add submit() Method to Engine

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for submit**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_submit_adds_to_event_queue() -> None:
    """Engine.submit() queues an input event for processing."""
    eng = engine.Engine()

    event: types.RunRequested = {
        "type": "run_requested",
        "stages": ["train"],
        "force": False,
        "reason": "test",
    }
    eng.submit(event)

    # Event should be in the queue (we can't easily inspect, but submit should not raise)
    assert eng.state == types.EngineState.IDLE  # Still idle until run_loop starts


def test_engine_submit_is_thread_safe() -> None:
    """Engine.submit() is safe to call from multiple threads."""
    import threading

    eng = engine.Engine()
    events_submitted = list[bool]()

    def submit_event() -> None:
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": None,
            "force": False,
            "reason": "thread",
        }
        eng.submit(event)
        events_submitted.append(True)

    threads = [threading.Thread(target=submit_event) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(events_submitted) == 10
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_submit_adds_to_event_queue -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'submit'`

**Step 3: Write minimal implementation**

Add import at top of `src/pivot/engine/engine.py`:
```python
import queue
```

Update Engine.__init__:
```python
    def __init__(self) -> None:
        """Initialize the engine in IDLE state."""
        self._state: EngineState = EngineState.IDLE
        self._sinks: list[EventSink] = list[EventSink]()
        self._sources: list[EventSource] = list[EventSource]()
        self._event_queue: queue.Queue[InputEvent] = queue.Queue()
```

Add the InputEvent import:
```python
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    InputEvent,
    OutputEvent,
)
```

Add submit method:
```python
    def submit(self, event: InputEvent) -> None:
        """Submit an event for processing. Thread-safe.

        Events are queued and processed by run_loop().

        Args:
            event: Input event to process.
        """
        self._event_queue.put(event)
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
jj describe -m "feat(engine): add submit() method for event queueing

Thread-safe event submission via queue.Queue."
```

---

## Task 6: Add Engine.run_loop() Skeleton

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for run_loop**

Add to `tests/engine/test_engine.py`:
```python
import threading
import time


def test_engine_run_loop_processes_run_requested() -> None:
    """run_loop() processes RunRequested events."""

    class RecordingSink:
        events: list[types.OutputEvent]

        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {"stage_a": {"status": "ran", "reason": ""}}

        eng = engine.Engine()
        sink = RecordingSink()
        eng.add_sink(sink)

        # Submit a run request
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": ["stage_a"],
            "force": False,
            "reason": "test",
        }
        eng.submit(event)

        # Run loop in a thread, stop after processing
        def run_and_stop() -> None:
            # Give loop time to process one event
            time.sleep(0.1)
            eng.shutdown()

        stopper = threading.Thread(target=run_and_stop)
        stopper.start()

        eng.run_loop()
        stopper.join()

        # Verify executor was called
        mock_executor.run.assert_called_once()
        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["stages"] == ["stage_a"]


def test_engine_run_loop_emits_state_changes() -> None:
    """run_loop() emits ACTIVE/IDLE state changes around execution."""

    class RecordingSink:
        events: list[types.OutputEvent]

        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        sink = RecordingSink()
        eng.add_sink(sink)

        # Submit and immediately request shutdown
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": None,
            "force": False,
            "reason": "test",
        }
        eng.submit(event)

        def delayed_shutdown() -> None:
            time.sleep(0.1)
            eng.shutdown()

        stopper = threading.Thread(target=delayed_shutdown)
        stopper.start()

        eng.run_loop()
        stopper.join()

        # Check state changes: ACTIVE when processing, IDLE when done
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) >= 2
        # First should be ACTIVE (start processing), last should be IDLE (done)
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[-1]["state"] == types.EngineState.IDLE
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_loop_processes_run_requested -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute 'run_loop'`

**Step 3: Write minimal implementation**

Add import at top:
```python
import threading
```

Add to Engine.__init__:
```python
        self._shutdown_event: threading.Event = threading.Event()
```

Add shutdown method and run_loop method:
```python
    def shutdown(self) -> None:
        """Signal the engine to stop processing events."""
        self._shutdown_event.set()

    def run_loop(self) -> None:
        """Process events until shutdown. For 'pivot run --watch'.

        Blocks until shutdown() is called. Processes events from the
        queue and from registered sources.
        """
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for an event with timeout to check shutdown
                    event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._handle_input_event(event)
        finally:
            # Ensure we're in IDLE state on exit
            if self._state != EngineState.IDLE:
                self._state = EngineState.IDLE
                self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _handle_input_event(self, event: InputEvent) -> None:
        """Process a single input event."""
        match event["type"]:
            case "run_requested":
                self._handle_run_requested(event)
            case "cancel_requested":
                pass  # TODO: implement cancellation
            case "data_artifact_changed":
                pass  # TODO: implement in Phase 4
            case "code_or_config_changed":
                pass  # TODO: implement in Phase 4

    def _handle_run_requested(self, event: RunRequested) -> None:
        """Handle a RunRequested event by executing stages."""
        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            executor.run(
                stages=event["stages"],
                force=event["force"],
            )
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
```

Add RunRequested import:
```python
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    InputEvent,
    OutputEvent,
    RunRequested,
)
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
jj describe -m "feat(engine): add run_loop() for continuous event processing

Processes RunRequested events, emits state changes."
```

---

## Task 7: Add CancelRequested Handling

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for cancel handling**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_cancel_requested_sets_cancel_event() -> None:
    """CancelRequested sets internal cancel event."""
    eng = engine.Engine()

    # Verify cancel event is initially clear
    assert not eng._cancel_event.is_set()

    # Submit cancel request
    cancel_event: types.CancelRequested = {"type": "cancel_requested"}
    eng.submit(cancel_event)

    # Process the event
    def process_one() -> None:
        time.sleep(0.05)
        eng.shutdown()

    stopper = threading.Thread(target=process_one)
    stopper.start()
    eng.run_loop()
    stopper.join()

    # Cancel event should be set
    assert eng._cancel_event.is_set()


def test_engine_cancel_event_property() -> None:
    """Engine exposes cancel_event for executor integration."""
    eng = engine.Engine()

    # cancel_event should be a threading.Event
    assert hasattr(eng.cancel_event, "is_set")
    assert hasattr(eng.cancel_event, "set")
    assert hasattr(eng.cancel_event, "clear")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_cancel_requested_sets_cancel_event -v
```

Expected: FAIL with `AttributeError: 'Engine' object has no attribute '_cancel_event'`

**Step 3: Write minimal implementation**

Add to Engine.__init__:
```python
        self._cancel_event: threading.Event = threading.Event()
```

Add cancel_event property:
```python
    @property
    def cancel_event(self) -> threading.Event:
        """Cancel event for stopping execution.

        Can be passed to executor to enable cancellation.
        """
        return self._cancel_event
```

Update _handle_input_event:
```python
    def _handle_input_event(self, event: InputEvent) -> None:
        """Process a single input event."""
        match event["type"]:
            case "run_requested":
                self._handle_run_requested(event)
            case "cancel_requested":
                self._handle_cancel_requested()
            case "data_artifact_changed":
                pass  # TODO: implement in Phase 4
            case "code_or_config_changed":
                pass  # TODO: implement in Phase 4

    def _handle_cancel_requested(self) -> None:
        """Handle a CancelRequested event by setting cancel flag."""
        self._cancel_event.set()
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
jj describe -m "feat(engine): add CancelRequested event handling

Sets cancel_event for graceful stage-level cancellation."
```

---

## Task 8: Pass cancel_event to Executor in run_loop

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write test verifying cancel_event is passed**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_run_loop_passes_cancel_event_to_executor() -> None:
    """run_loop() passes cancel_event to executor.run()."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()

        event: types.RunRequested = {
            "type": "run_requested",
            "stages": None,
            "force": False,
            "reason": "test",
        }
        eng.submit(event)

        def delayed_shutdown() -> None:
            time.sleep(0.05)
            eng.shutdown()

        stopper = threading.Thread(target=delayed_shutdown)
        stopper.start()
        eng.run_loop()
        stopper.join()

        # Verify cancel_event was passed
        call_kwargs = mock_executor.run.call_args.kwargs
        assert "cancel_event" in call_kwargs
        assert call_kwargs["cancel_event"] is eng.cancel_event
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_loop_passes_cancel_event_to_executor -v
```

Expected: FAIL (cancel_event not in call_kwargs)

**Step 3: Write minimal implementation**

Update _handle_run_requested to pass cancel_event:
```python
    def _handle_run_requested(self, event: RunRequested) -> None:
        """Handle a RunRequested event by executing stages."""
        # Clear cancel event before starting new execution
        self._cancel_event.clear()

        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            executor.run(
                stages=event["stages"],
                force=event["force"],
                cancel_event=self._cancel_event,
            )
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
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
jj describe -m "feat(engine): pass cancel_event to executor in run_loop

Enables stage-level cancellation during watch mode."
```

---

## Task 9: Start Sources When run_loop Begins

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write test for source lifecycle**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_run_loop_starts_sources() -> None:
    """run_loop() starts all registered sources."""
    started = list[bool]()

    class MockSource:
        def start(self, submit: Callable[[types.InputEvent], None]) -> None:
            started.append(True)

        def stop(self) -> None:
            pass

    eng = engine.Engine()
    eng.add_source(MockSource())
    eng.add_source(MockSource())

    def delayed_shutdown() -> None:
        time.sleep(0.05)
        eng.shutdown()

    stopper = threading.Thread(target=delayed_shutdown)
    stopper.start()
    eng.run_loop()
    stopper.join()

    assert len(started) == 2


def test_engine_run_loop_stops_sources_on_shutdown() -> None:
    """run_loop() stops all sources when shutting down."""
    stopped = list[bool]()

    class MockSource:
        def start(self, submit: Callable[[types.InputEvent], None]) -> None:
            pass

        def stop(self) -> None:
            stopped.append(True)

    eng = engine.Engine()
    eng.add_source(MockSource())
    eng.add_source(MockSource())

    def delayed_shutdown() -> None:
        time.sleep(0.05)
        eng.shutdown()

    stopper = threading.Thread(target=delayed_shutdown)
    stopper.start()
    eng.run_loop()
    stopper.join()

    assert len(stopped) == 2
```

Add import at top of test file:
```python
from collections.abc import Callable
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_loop_starts_sources -v
```

Expected: FAIL (started list is empty)

**Step 3: Write minimal implementation**

Update run_loop:
```python
    def run_loop(self) -> None:
        """Process events until shutdown. For 'pivot run --watch'.

        Blocks until shutdown() is called. Processes events from the
        queue and from registered sources.
        """
        # Start all sources
        for source in self._sources:
            source.start(self.submit)

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for an event with timeout to check shutdown
                    event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._handle_input_event(event)
        finally:
            # Stop all sources
            for source in self._sources:
                source.stop()

            # Ensure we're in IDLE state on exit
            if self._state != EngineState.IDLE:
                self._state = EngineState.IDLE
                self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
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
jj describe -m "feat(engine): start/stop sources in run_loop

Sources are started when loop begins, stopped on shutdown."
```

---

## Task 10: Implement FilesystemSource Watcher Thread

**Files:**
- Modify: `src/pivot/engine/sources.py`
- Modify: `tests/engine/test_sources.py`

**Step 1: Write test for watcher thread**

Add to `tests/engine/test_sources.py`:
```python
import threading
import time


def test_filesystem_source_starts_watcher_thread(tmp_path: Path) -> None:
    """FilesystemSource.start() spawns a watcher thread."""
    watch_file = tmp_path / "data.csv"
    watch_file.touch()

    events_received = list[types.InputEvent]()
    event_received = threading.Event()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)
        event_received.set()

    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(submit)

    # Give watcher time to start
    time.sleep(0.2)

    # Modify file to trigger event
    watch_file.write_text("new content")

    # Wait for event with timeout
    assert event_received.wait(timeout=2.0), "Timed out waiting for file change event"

    source.stop()

    # Should have received at least one event
    assert len(events_received) >= 1
    assert events_received[0]["type"] == "data_artifact_changed"


def test_filesystem_source_stop_terminates_watcher(tmp_path: Path) -> None:
    """FilesystemSource.stop() terminates the watcher thread."""
    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(lambda _: None)

    # Give watcher time to start
    time.sleep(0.1)

    source.stop()

    # Give thread time to terminate
    time.sleep(0.2)

    # Watcher should no longer be running
    assert not source._running
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/engine/test_sources.py::test_filesystem_source_starts_watcher_thread -v
```

Expected: FAIL (no event received - watcher not actually watching)

**Step 3: Write minimal implementation**

Update `src/pivot/engine/sources.py` with the full FilesystemSource implementation:
```python
"""Event source implementations for the engine."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import watchfiles

from pivot.engine.types import CodeOrConfigChanged, DataArtifactChanged, RunRequested

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pivot.engine.types import InputEvent

__all__ = ["FilesystemSource", "OneShotSource"]

# File patterns that trigger code reload (same as watch/engine.py)
_CODE_FILE_SUFFIXES = (".py",)
_CONFIG_FILE_NAMES = (
    "pivot.yaml",
    "pivot.yml",
    "pipeline.py",
    "params.yaml",
    "params.yml",
    ".pivotignore",
)


def _is_code_or_config(path: str) -> bool:
    """Check if a path is a code or config file."""
    from pathlib import Path as PathClass

    p = PathClass(path)
    return p.suffix in _CODE_FILE_SUFFIXES or p.name in _CONFIG_FILE_NAMES


class FilesystemSource:
    """Event source that watches filesystem for changes.

    Wraps watchfiles to detect file changes and emit DataArtifactChanged
    or CodeOrConfigChanged events.
    """

    _watch_paths: list[Path]
    _submit: Callable[[InputEvent], None] | None
    _running: bool
    _shutdown_event: threading.Event
    _watcher_thread: threading.Thread | None

    def __init__(self, watch_paths: list[Path]) -> None:
        """Initialize with paths to watch.

        Args:
            watch_paths: List of paths to watch for changes.
        """
        self._watch_paths = list(watch_paths)
        self._submit = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._watcher_thread = None

    @property
    def watch_paths(self) -> list[Path]:
        """Current paths being watched."""
        return list(self._watch_paths)

    def set_watch_paths(self, paths: list[Path]) -> None:
        """Update watched paths.

        Called by engine when DAG changes and watch scope needs updating.
        If watcher is running, it will be restarted with new paths.

        Args:
            paths: New list of paths to watch.
        """
        self._watch_paths = list(paths)

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Begin producing events. Call submit() for each event."""
        self._submit = submit
        self._running = True
        self._shutdown_event.clear()

        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
        )
        self._watcher_thread.start()

    def stop(self) -> None:
        """Stop producing events."""
        self._running = False
        self._shutdown_event.set()

        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=3.0)
            self._watcher_thread = None

        self._submit = None

    def _watch_loop(self) -> None:
        """Watch for file changes and submit events."""
        if not self._watch_paths:
            return

        for changes in watchfiles.watch(
            *self._watch_paths,
            stop_event=self._shutdown_event,
        ):
            if self._submit is None:
                break

            # Classify changes as code/config or data
            code_paths = list[str]()
            data_paths = list[str]()

            for _change_type, path in changes:
                if _is_code_or_config(path):
                    code_paths.append(path)
                else:
                    data_paths.append(path)

            # Emit appropriate events
            if code_paths:
                self._submit(
                    CodeOrConfigChanged(
                        type="code_or_config_changed",
                        paths=code_paths,
                    )
                )

            if data_paths:
                self._submit(
                    DataArtifactChanged(
                        type="data_artifact_changed",
                        paths=data_paths,
                    )
                )


class OneShotSource:
    """Event source that emits a single RunRequested event.

    Used for 'pivot run' without --watch. Emits the run request
    immediately when start() is called, then becomes inactive.
    """

    _stages: list[str] | None
    _force: bool
    _reason: str
    _emitted: bool

    def __init__(
        self,
        stages: list[str] | None,
        force: bool,
        reason: str,
    ) -> None:
        """Initialize with run parameters.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run.
            reason: Description of why this run was requested.
        """
        self._stages = stages
        self._force = force
        self._reason = reason
        self._emitted = False

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Emit a single RunRequested event."""
        if self._emitted:
            return

        event = RunRequested(
            type="run_requested",
            stages=self._stages,
            force=self._force,
            reason=self._reason,
        )
        submit(event)
        self._emitted = True

    def stop(self) -> None:
        """No-op for one-shot source."""
        pass
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sources.py -v
```

Expected: PASS (all tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): implement FilesystemSource watcher thread

Uses watchfiles to detect changes and emit events."
```

---

## Task 11: Add Code/Config Change Detection Test

**Files:**
- Modify: `tests/engine/test_sources.py`

**Step 1: Write test for code/config detection**

Add to `tests/engine/test_sources.py`:
```python
def test_filesystem_source_emits_code_changed_for_py_files(tmp_path: Path) -> None:
    """FilesystemSource emits CodeOrConfigChanged for .py files."""
    py_file = tmp_path / "stages.py"
    py_file.write_text("# initial")

    events_received = list[types.InputEvent]()
    event_received = threading.Event()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)
        event_received.set()

    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(submit)

    # Give watcher time to start
    time.sleep(0.2)

    # Modify Python file
    py_file.write_text("# modified")

    # Wait for event
    assert event_received.wait(timeout=2.0)
    source.stop()

    # Should be code_or_config_changed
    code_events = [e for e in events_received if e["type"] == "code_or_config_changed"]
    assert len(code_events) >= 1
    assert any(str(py_file) in str(e["paths"]) for e in code_events)


def test_filesystem_source_emits_code_changed_for_config_files(tmp_path: Path) -> None:
    """FilesystemSource emits CodeOrConfigChanged for pivot.yaml."""
    config_file = tmp_path / "pivot.yaml"
    config_file.write_text("# initial")

    events_received = list[types.InputEvent]()
    event_received = threading.Event()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)
        event_received.set()

    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(submit)

    time.sleep(0.2)
    config_file.write_text("# modified")

    assert event_received.wait(timeout=2.0)
    source.stop()

    code_events = [e for e in events_received if e["type"] == "code_or_config_changed"]
    assert len(code_events) >= 1
```

**Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_sources.py -v
```

Expected: PASS (all tests)

**Step 3: Commit**

```bash
jj describe -m "test(engine): add code/config change detection tests

Verifies FilesystemSource classifies .py and config files correctly."
```

---

## Task 12: Run Full Test Suite and Quality Checks

**Files:**
- None (verification only)

**Step 1: Run engine tests**

```bash
uv run pytest tests/engine/ -v
```

Expected: PASS (all tests)

**Step 2: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

Expected: No errors

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

Expected: All existing tests still pass

**Step 4: Final commit**

```bash
jj describe -m "feat(engine): complete Phase 3 - Watch Mode Unification

- FilesystemSource wraps watchfiles for file change detection
- OneShotSource for single run requests
- Engine.run_loop() for continuous event processing
- Engine.submit() for thread-safe event queueing
- CancelRequested handling with cancel_event
- Source lifecycle management (start/stop)"
```

---

## Summary

After completing Phase 3, you will have:

1. **`src/pivot/engine/sources.py`** - Event sources:
   - `FilesystemSource` - wraps watchfiles, emits DataArtifactChanged/CodeOrConfigChanged
   - `OneShotSource` - emits single RunRequested for non-watch mode

2. **`src/pivot/engine/engine.py`** - Updated Engine with:
   - Source registration (`add_source`, `sources` property)
   - Event submission (`submit()`) with thread-safe queue
   - Continuous event processing (`run_loop()`)
   - Source lifecycle management (start/stop in run_loop)
   - Cancel event handling

3. **Tests** - Comprehensive coverage for all new code

**What's NOT in this phase:**
- CLI integration (routing `pivot run` through Engine) - Phase 4
- DataArtifactChanged/CodeOrConfigChanged handling in Engine - Phase 4
- Replacing WatchEngine with Engine - Phase 4

**Phase 4 will:**
1. Add DataArtifactChanged/CodeOrConfigChanged handlers to Engine
2. Route `pivot run` CLI through Engine
3. Route `pivot run --watch` through Engine
4. Delete the old WatchEngine

---

## Appendix: File Modifications Summary

| File | Action | Description |
|------|--------|-------------|
| `src/pivot/engine/sources.py` | Create | FilesystemSource, OneShotSource |
| `src/pivot/engine/engine.py` | Modify | add_source, submit, run_loop, shutdown |
| `src/pivot/engine/__init__.py` | Modify | Export sources module |
| `tests/engine/test_sources.py` | Create | Tests for sources |
| `tests/engine/test_engine.py` | Modify | Tests for new Engine methods |
