"""Tests for the Engine class."""

from __future__ import annotations

import contextlib
import pathlib
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pivot.engine import engine, sources, types
from pivot.types import (
    OnError,
    RunEventType,
    RunJsonEvent,
    StageCompleteEvent,
    StageStartEvent,
    StageStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Module-level test helpers (per tests/CLAUDE.md)
# =============================================================================


class _MockSink:
    """Test sink for capturing engine events."""

    events: list[types.OutputEvent]
    closed: bool

    def __init__(self) -> None:
        self.events = list[types.OutputEvent]()
        self.closed = False

    def handle(self, event: types.OutputEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def _helper_run_loop_with_delayed_shutdown(eng: engine.Engine, delay: float = 0.05) -> None:
    """Run engine loop and shut it down after a delay."""

    def delayed_shutdown() -> None:
        time.sleep(delay)
        eng.shutdown()

    stopper = threading.Thread(target=delayed_shutdown)
    stopper.start()
    eng.run_loop()
    stopper.join()


class _FailingSink:
    """Test sink that raises exceptions on handle or close."""

    fail_on_handle: bool
    fail_on_close: bool
    closed: bool

    def __init__(self, fail_on_handle: bool = False, fail_on_close: bool = False) -> None:
        self.fail_on_handle = fail_on_handle
        self.fail_on_close = fail_on_close
        self.closed = False

    def handle(self, event: types.OutputEvent) -> None:
        if self.fail_on_handle:
            raise RuntimeError("handle failed")

    def close(self) -> None:
        if self.fail_on_close:
            raise RuntimeError("close failed")
        self.closed = True


# =============================================================================
# Basic Engine Tests
# =============================================================================


def test_engine_initial_state_is_idle() -> None:
    """Engine starts in IDLE state."""
    eng = engine.Engine()
    assert eng.state == types.EngineState.IDLE


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


def test_engine_has_empty_sinks_initially() -> None:
    """Engine has no sinks until registered."""
    eng = engine.Engine()
    assert eng.sinks == []


def test_engine_graph_is_none_initially() -> None:
    """Engine graph is None until built."""
    eng = engine.Engine()
    assert eng.graph is None


def test_engine_add_sink() -> None:
    """Engine can register event sinks."""
    eng = engine.Engine()
    sink = _MockSink()
    eng.add_sink(sink)

    assert len(eng.sinks) == 1
    assert eng.sinks[0] is sink


def test_engine_emit_sends_to_all_sinks() -> None:
    """Engine.emit() sends event to all registered sinks."""
    eng = engine.Engine()
    sink1 = _MockSink()
    sink2 = _MockSink()
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
    eng = engine.Engine()
    sink1 = _MockSink()
    sink2 = _MockSink()
    eng.add_sink(sink1)
    eng.add_sink(sink2)

    eng.close()

    assert sink1.closed
    assert sink2.closed


def test_engine_run_once_returns_execution_summary() -> None:
    """run_once() returns dict mapping stage names to ExecutionSummary."""
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
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)
        eng.run_once()

        # Should emit ACTIVE at start and IDLE at end
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 2
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[1]["state"] == types.EngineState.IDLE


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
        assert call_kwargs["allow_uncached_incremental"] is True
        assert call_kwargs["checkout_missing"] is True


def test_engine_integration_with_sinks() -> None:
    """Engine correctly routes events to multiple sinks."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        # Simulate executor returning results
        mock_executor.run.return_value = {
            "stage_a": {"status": "ran", "reason": "inputs changed"},
        }

        eng = engine.Engine()
        sink1 = _MockSink()
        sink2 = _MockSink()
        eng.add_sink(sink1)
        eng.add_sink(sink2)

        result = eng.run_once(stages=["stage_a"])

        # Verify results returned
        assert "stage_a" in result

        # Verify both sinks received events
        assert len(sink1.events) >= 2, "At least ACTIVE and IDLE state changes"
        assert len(sink2.events) >= 2, "At least ACTIVE and IDLE state changes"
        assert sink1.events == sink2.events, "Same events to both sinks"

        # Verify state change events
        state_events = [e for e in sink1.events if e["type"] == "engine_state_changed"]
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[1]["state"] == types.EngineState.IDLE

        # Close and verify
        eng.close()
        assert sink1.closed
        assert sink2.closed


def test_engine_run_once_emits_idle_on_exception() -> None:
    """run_once() emits IDLE state even when executor raises."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.side_effect = RuntimeError("execution failed")

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        with contextlib.suppress(RuntimeError):
            eng.run_once()

        # Should still emit IDLE at end
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 2
        assert state_events[1]["state"] == types.EngineState.IDLE
        assert eng.state == types.EngineState.IDLE


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


def test_engine_run_loop_processes_run_requested() -> None:
    """run_loop() processes RunRequested events."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {"stage_a": {"status": "ran", "reason": ""}}

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        # Submit a run request
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": ["stage_a"],
            "force": False,
            "reason": "test",
        }
        eng.submit(event)

        _helper_run_loop_with_delayed_shutdown(eng, delay=0.1)

        # Verify executor was called
        mock_executor.run.assert_called_once()
        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["stages"] == ["stage_a"]


def test_engine_run_loop_emits_state_changes() -> None:
    """run_loop() emits ACTIVE/IDLE state changes around execution."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        # Submit and immediately request shutdown
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": None,
            "force": False,
            "reason": "test",
        }
        eng.submit(event)

        _helper_run_loop_with_delayed_shutdown(eng, delay=0.1)

        # Check state changes: ACTIVE when processing, IDLE when done
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) >= 2
        # First should be ACTIVE (start processing), last should be IDLE (done)
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[-1]["state"] == types.EngineState.IDLE


def test_engine_cancel_requested_sets_cancel_event() -> None:
    """CancelRequested sets internal cancel event."""
    eng = engine.Engine()

    # Verify cancel event is initially clear
    assert not eng._cancel_event.is_set()

    # Submit cancel request
    cancel_event: types.CancelRequested = {"type": "cancel_requested"}
    eng.submit(cancel_event)

    # Process the event
    _helper_run_loop_with_delayed_shutdown(eng)

    # Cancel event should be set
    assert eng._cancel_event.is_set()


def test_engine_cancel_event_property() -> None:
    """Engine exposes cancel_event for executor integration."""
    eng = engine.Engine()

    # cancel_event should be a threading.Event
    assert hasattr(eng.cancel_event, "is_set")
    assert hasattr(eng.cancel_event, "set")
    assert hasattr(eng.cancel_event, "clear")


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

        _helper_run_loop_with_delayed_shutdown(eng)

        # Verify cancel_event was passed
        call_kwargs = mock_executor.run.call_args.kwargs
        assert "cancel_event" in call_kwargs
        assert call_kwargs["cancel_event"] is eng.cancel_event


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

    _helper_run_loop_with_delayed_shutdown(eng)

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

    _helper_run_loop_with_delayed_shutdown(eng)

    assert len(stopped) == 2


# =============================================================================
# Tests for CLI parameter passthrough (Task 11)
# =============================================================================


def test_engine_run_once_passes_progress_callback_parameter() -> None:
    """run_once() passes a progress_callback adapter to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        def callback(event: RunJsonEvent) -> None:
            pass

        eng = engine.Engine()
        eng.run_once(progress_callback=callback)

        call_kwargs = mock_executor.run.call_args.kwargs
        # Engine wraps the user callback in an adapter for stage event emission
        assert call_kwargs["progress_callback"] is not None
        assert callable(call_kwargs["progress_callback"])


def test_engine_run_once_passes_cancel_event() -> None:
    """run_once() passes internal cancel_event to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once()

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["cancel_event"] is eng.cancel_event


# =============================================================================
# Tests for Stage Event Emission (Task 1)
# =============================================================================


def test_engine_run_once_emits_stage_started_events() -> None:
    """run_once() emits StageStarted events to sinks."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {"train": {"status": "ran", "reason": ""}}

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        # Manually call the progress adapter to simulate executor behavior
        eng._handle_progress_event(
            StageStartEvent(
                type=RunEventType.STAGE_START,
                stage="train",
                index=1,
                total=2,
                timestamp="2026-01-30T00:00:00Z",
            )
        )

        stage_events = [e for e in sink.events if e["type"] == "stage_started"]
        assert len(stage_events) == 1
        assert stage_events[0]["stage"] == "train"
        assert stage_events[0]["index"] == 1
        assert stage_events[0]["total"] == 2


def test_engine_run_once_emits_stage_completed_events() -> None:
    """run_once() emits StageCompleted events to sinks."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {"train": {"status": "ran", "reason": ""}}

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        # Simulate stage start first (to populate index/total tracking)
        eng._handle_progress_event(
            StageStartEvent(
                type=RunEventType.STAGE_START,
                stage="train",
                index=1,
                total=2,
                timestamp="2026-01-30T00:00:00Z",
            )
        )

        # Then stage complete
        eng._handle_progress_event(
            StageCompleteEvent(
                type=RunEventType.STAGE_COMPLETE,
                stage="train",
                status=StageStatus.RAN,
                reason="inputs changed",
                duration_ms=1234.5,
                timestamp="2026-01-30T00:00:00Z",
            )
        )

        stage_events = [e for e in sink.events if e["type"] == "stage_completed"]
        assert len(stage_events) == 1
        assert stage_events[0]["stage"] == "train"
        assert stage_events[0]["status"] == StageStatus.RAN
        assert stage_events[0]["duration_ms"] == 1234.5


def test_engine_run_once_forwards_to_user_progress_callback() -> None:
    """run_once() still forwards events to user's progress_callback."""
    user_events = list[RunJsonEvent]()

    def user_callback(event: RunJsonEvent) -> None:
        user_events.append(event)

    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once(progress_callback=user_callback)

        # The progress_adapter wraps the user callback, so we need to verify
        # by checking that executor.run was called with a progress_callback
        call_kwargs = mock_executor.run.call_args.kwargs
        assert "progress_callback" in call_kwargs
        assert call_kwargs["progress_callback"] is not None

        # Call the adapter to verify forwarding works
        adapter = call_kwargs["progress_callback"]
        test_event = StageStartEvent(
            type=RunEventType.STAGE_START,
            stage="train",
            index=1,
            total=1,
            timestamp="2026-01-30T00:00:00Z",
        )
        adapter(test_event)

        assert len(user_events) == 1
        assert user_events[0]["type"] == RunEventType.STAGE_START


# =============================================================================
# Integration Test for Full Event Flow (Task 6)
# =============================================================================


def test_engine_full_event_flow_integration() -> None:
    """Integration test verifying complete event flow through Engine.

    Verifies:
    - Event sequence: state changes + stage events
    - Stage details preserved correctly
    - Multiple stages handled in order
    """

    def executor_side_effect(**kwargs: object) -> dict[str, dict[str, str]]:
        """Simulate executor calling progress_callback during execution."""
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None and callable(progress_callback):
            # Simulate executor calling progress_callback for each stage
            progress_callback(
                StageStartEvent(
                    type=RunEventType.STAGE_START,
                    stage="stage_a",
                    index=1,
                    total=2,
                    timestamp="2026-01-30T00:00:00Z",
                )
            )
            progress_callback(
                StageCompleteEvent(
                    type=RunEventType.STAGE_COMPLETE,
                    stage="stage_a",
                    status=StageStatus.RAN,
                    reason="inputs changed",
                    duration_ms=100.0,
                    timestamp="2026-01-30T00:00:01Z",
                )
            )
            progress_callback(
                StageStartEvent(
                    type=RunEventType.STAGE_START,
                    stage="stage_b",
                    index=2,
                    total=2,
                    timestamp="2026-01-30T00:00:02Z",
                )
            )
            progress_callback(
                StageCompleteEvent(
                    type=RunEventType.STAGE_COMPLETE,
                    stage="stage_b",
                    status=StageStatus.SKIPPED,
                    reason="up to date",
                    duration_ms=5.0,
                    timestamp="2026-01-30T00:00:02Z",
                )
            )
        return {
            "stage_a": {"status": "ran", "reason": "inputs changed"},
            "stage_b": {"status": "skipped", "reason": "up to date"},
        }

    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.side_effect = executor_side_effect

        eng = engine.Engine()
        sink = _MockSink()
        eng.add_sink(sink)

        # Run stages - executor_side_effect will call progress_callback during run
        result = eng.run_once(stages=["stage_a", "stage_b"])

        # Verify executor was called and returned expected result
        assert "stage_a" in result
        assert "stage_b" in result

        # Verify event sequence
        # State events: ACTIVE at start, IDLE at end (emitted by run_once)
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 2
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[1]["state"] == types.EngineState.IDLE

        # Stage started events
        started_events = [e for e in sink.events if e["type"] == "stage_started"]
        assert len(started_events) == 2
        assert started_events[0]["stage"] == "stage_a"
        assert started_events[0]["index"] == 1
        assert started_events[0]["total"] == 2
        assert started_events[1]["stage"] == "stage_b"
        assert started_events[1]["index"] == 2
        assert started_events[1]["total"] == 2

        # Stage completed events
        completed_events = [e for e in sink.events if e["type"] == "stage_completed"]
        assert len(completed_events) == 2

        # Verify stage_a completed event
        assert completed_events[0]["stage"] == "stage_a"
        assert completed_events[0]["status"] == StageStatus.RAN
        assert completed_events[0]["reason"] == "inputs changed"
        assert completed_events[0]["duration_ms"] == 100.0
        assert completed_events[0]["index"] == 1
        assert completed_events[0]["total"] == 2

        # Verify stage_b completed event
        assert completed_events[1]["stage"] == "stage_b"
        assert completed_events[1]["status"] == StageStatus.SKIPPED
        assert completed_events[1]["reason"] == "up to date"
        assert completed_events[1]["duration_ms"] == 5.0
        assert completed_events[1]["index"] == 2
        assert completed_events[1]["total"] == 2

        # Verify event ordering: ACTIVE -> stage events -> IDLE
        event_types = [e["type"] for e in sink.events]
        active_idx = event_types.index("engine_state_changed")
        idle_idx = len(event_types) - 1 - event_types[::-1].index("engine_state_changed")
        stage_started_indices = [i for i, t in enumerate(event_types) if t == "stage_started"]
        stage_completed_indices = [i for i, t in enumerate(event_types) if t == "stage_completed"]

        # All stage events should be between ACTIVE and IDLE
        for idx in stage_started_indices + stage_completed_indices:
            assert active_idx < idx < idle_idx, "Stage events should be between state changes"


# =============================================================================
# Tests for Engine Robustness Features
# =============================================================================


def test_engine_set_cancel_event() -> None:
    """set_cancel_event() replaces the internal cancel event."""
    eng = engine.Engine()
    original_event = eng.cancel_event

    new_event = threading.Event()
    eng.set_cancel_event(new_event)

    assert eng.cancel_event is new_event
    assert eng.cancel_event is not original_event


def test_engine_context_manager() -> None:
    """Engine can be used as a context manager."""
    sink = _MockSink()

    with engine.Engine() as eng:
        eng.add_sink(sink)
        assert eng.state == types.EngineState.IDLE

    # Sink should be closed after exiting context
    assert sink.closed


def test_engine_context_manager_closes_on_exception() -> None:
    """Engine context manager closes sinks even when exception raised."""
    sink = _MockSink()

    with contextlib.suppress(RuntimeError), engine.Engine() as eng:
        eng.add_sink(sink)
        raise RuntimeError("test exception")

    # Sink should still be closed
    assert sink.closed


def test_engine_run_once_clears_cancel_event() -> None:
    """run_once() clears cancel event before execution."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.cancel_event.set()  # Simulate previous cancellation

        eng.run_once()

        # Cancel event should have been cleared before execution
        # We verify by checking it's not set after a successful run
        assert not eng.cancel_event.is_set()


def test_engine_run_once_rejects_concurrent_calls() -> None:
    """run_once() raises if called while engine is already active."""
    eng = engine.Engine()

    # Manually set state to ACTIVE to simulate concurrent call
    eng._state = types.EngineState.ACTIVE

    with pytest.raises(RuntimeError, match="already active"):
        eng.run_once()


def test_engine_emit_continues_on_sink_failure() -> None:
    """emit() continues to other sinks if one fails."""
    eng = engine.Engine()

    failing_sink = _FailingSink(fail_on_handle=True)
    working_sink = _MockSink()

    eng.add_sink(failing_sink)
    eng.add_sink(working_sink)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    eng.emit(event)  # Should not raise

    # Working sink should still receive the event
    assert len(working_sink.events) == 1


def test_engine_close_continues_on_sink_failure() -> None:
    """close() continues to other sinks if one fails."""
    eng = engine.Engine()

    failing_sink = _FailingSink(fail_on_close=True)
    working_sink = _MockSink()

    eng.add_sink(failing_sink)
    eng.add_sink(working_sink)

    eng.close()  # Should not raise

    # Working sink should still be closed
    assert working_sink.closed


# =============================================================================
# Parameterized Tests for run_once() Parameter Passthrough
# =============================================================================


@pytest.mark.parametrize(
    ("param_name", "param_value"),
    [
        pytest.param("on_error", OnError.KEEP_GOING, id="on_error"),
        pytest.param("cache_dir", pathlib.Path("/tmp/test-cache"), id="cache_dir"),
        pytest.param("single_stage", True, id="single_stage"),
        pytest.param("parallel", False, id="parallel"),
        pytest.param("max_workers", 4, id="max_workers"),
        pytest.param("no_commit", True, id="no_commit"),
        pytest.param("no_cache", True, id="no_cache"),
        pytest.param("allow_uncached_incremental", True, id="allow_uncached_incremental"),
        pytest.param("checkout_missing", True, id="checkout_missing"),
    ],
)
def test_engine_run_once_passes_parameter(param_name: str, param_value: object) -> None:
    """run_once() passes parameters through to executor."""
    with patch("pivot.engine.engine.executor") as mock_executor:
        mock_executor.run.return_value = {}

        eng = engine.Engine()
        eng.run_once(**{param_name: param_value})  # pyright: ignore[reportArgumentType]

        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs[param_name] == param_value
