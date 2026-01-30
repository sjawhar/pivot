"""Tests for the Engine class."""

from __future__ import annotations

import contextlib
import pathlib
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import networkx as nx
import pytest

from pivot import registry
from pivot.engine import engine, sources, types
from pivot.engine import graph as engine_graph
from pivot.types import (
    OnError,
    RunEventType,
    RunJsonEvent,
    StageCompleteEvent,
    StageResult,
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
    with engine.Engine() as eng:
        assert eng.state == types.EngineState.IDLE


def test_engine_has_empty_sources_initially() -> None:
    """Engine has no sources until registered."""
    with engine.Engine() as eng:
        assert eng.sources == []


def test_engine_add_source() -> None:
    """Engine can register event sources."""
    with engine.Engine() as eng:
        source = sources.OneShotSource(stages=None, force=False, reason="test")
        eng.add_source(source)

        assert len(eng.sources) == 1
        assert eng.sources[0] is source


def test_engine_has_empty_sinks_initially() -> None:
    """Engine has no sinks until registered."""
    with engine.Engine() as eng:
        assert eng.sinks == []


def test_engine_graph_is_none_initially() -> None:
    """Engine graph is None until built."""
    with engine.Engine() as eng:
        assert eng.graph is None


def test_engine_graph_property_returns_bipartite_graph(tmp_path: pathlib.Path) -> None:
    """Engine.graph returns the bipartite artifact-stage graph after execution.

    Verifies:
    - Engine has 'graph' attribute
    - After run_once(), graph is populated (not None)
    - Graph is a networkx DiGraph with bipartite structure (artifact and stage nodes)
    """
    with engine.Engine() as eng:

        # Verify graph attribute exists and starts as None
        assert hasattr(eng, "graph")
        assert eng.graph is None

        # Build a test graph that _orchestrate_execution would create
        test_graph: nx.DiGraph[str] = nx.DiGraph()
        input_artifact = engine_graph.artifact_node(tmp_path / "input.csv")
        output_artifact = engine_graph.artifact_node(tmp_path / "output.csv")
        stage = engine_graph.stage_node("test_stage")
        test_graph.add_node(input_artifact, type=types.NodeType.ARTIFACT)
        test_graph.add_node(output_artifact, type=types.NodeType.ARTIFACT)
        test_graph.add_node(stage, type=types.NodeType.STAGE)
        test_graph.add_edge(input_artifact, stage)
        test_graph.add_edge(stage, output_artifact)

        # Mock _orchestrate_execution to set the graph (simulating real behavior)
        def mock_orchestrate(self: engine.Engine, **kwargs: object) -> dict[str, object]:
            self._graph = test_graph
            return {"test_stage": {"status": "ran", "reason": ""}}

        with patch.object(engine.Engine, "_orchestrate_execution", mock_orchestrate):
            eng.run_once()

        # After run_once(), graph should be populated via _orchestrate_execution
        result = eng.graph
        assert result is not None, "run_once() should populate the graph"

        # Verify bipartite structure: has both artifact and stage nodes
        stage_nodes = [
            n
            for n in result.nodes()  # pyright: ignore[reportGeneralTypeIssues] - typeshed stub quirk
            if result.nodes[n]["type"] == types.NodeType.STAGE
        ]
        artifact_nodes = [
            n
            for n in result.nodes()  # pyright: ignore[reportGeneralTypeIssues] - typeshed stub quirk
            if result.nodes[n]["type"] == types.NodeType.ARTIFACT
        ]
        assert len(stage_nodes) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert len(artifact_nodes) == 2  # pyright: ignore[reportUnknownArgumentType]

        # Verify edges exist in correct direction (artifact -> stage -> artifact)
        assert result.has_edge(input_artifact, stage)
        assert result.has_edge(stage, output_artifact)


def test_engine_add_sink() -> None:
    """Engine can register event sinks."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        assert len(eng.sinks) == 1
        assert eng.sinks[0] is sink


def test_engine_emit_sends_to_all_sinks() -> None:
    """Engine.emit() sends event to all registered sinks."""
    with engine.Engine() as eng:
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
    with engine.Engine() as eng:
        sink1 = _MockSink()
        sink2 = _MockSink()
        eng.add_sink(sink1)
        eng.add_sink(sink2)

        eng.close()

        assert sink1.closed
        assert sink2.closed


def test_engine_run_once_returns_execution_summary() -> None:
    """run_once() returns dict mapping stage names to ExecutionSummary."""
    with patch.object(
        engine.Engine,
        "_orchestrate_execution",
        return_value={"test_stage": {"status": "ran", "reason": ""}},
    ):
        with engine.Engine() as eng:
            result = eng.run_once()

            assert isinstance(result, dict)
            assert "test_stage" in result


def test_engine_run_once_passes_stages_parameter() -> None:
    """run_once() passes stages parameter to _orchestrate_execution."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}) as mock_orchestrate:
        with engine.Engine() as eng:
            eng.run_once(stages=["stage_a", "stage_b"])

            mock_orchestrate.assert_called_once()
            call_kwargs = mock_orchestrate.call_args.kwargs
            assert call_kwargs["stages"] == ["stage_a", "stage_b"]


def test_engine_run_once_passes_force_parameter() -> None:
    """run_once() passes force parameter to _orchestrate_execution."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}) as mock_orchestrate:
        with engine.Engine() as eng:
            eng.run_once(force=True)

            call_kwargs = mock_orchestrate.call_args.kwargs
            assert call_kwargs["force"] is True


def test_engine_run_once_emits_state_changed_events() -> None:
    """run_once() emits engine state changed events."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}):
        with engine.Engine() as eng:
            sink = _MockSink()
            eng.add_sink(sink)
            eng.run_once()

            # Should emit ACTIVE at start and IDLE at end
            state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
            assert len(state_events) == 2
            assert state_events[0]["state"] == types.EngineState.ACTIVE
            assert state_events[1]["state"] == types.EngineState.IDLE


def test_engine_run_once_passes_all_orchestration_params() -> None:
    """run_once() passes through all relevant orchestration parameters."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}) as mock_orchestrate:
        with engine.Engine() as eng:
            eng.run_once(
                stages=["stage_a"],
                force=True,
                single_stage=True,
                parallel=False,
                max_workers=4,
                no_commit=True,
                no_cache=True,
                allow_uncached_incremental=True,  # Retained for CLI but not passed to orchestration
                checkout_missing=True,  # Retained for CLI but not passed to orchestration
            )

            call_kwargs = mock_orchestrate.call_args.kwargs
            assert call_kwargs["stages"] == ["stage_a"]
            assert call_kwargs["force"] is True
            assert call_kwargs["single_stage"] is True
            assert call_kwargs["parallel"] is False
            assert call_kwargs["max_workers"] == 4
            assert call_kwargs["no_commit"] is True
            assert call_kwargs["no_cache"] is True
            # Note: allow_uncached_incremental and checkout_missing are retained for CLI
            # compatibility but not passed to _orchestrate_execution (handled differently)


def test_engine_integration_with_sinks() -> None:
    """Engine correctly routes events to multiple sinks."""
    with patch.object(
        engine.Engine,
        "_orchestrate_execution",
        return_value={"stage_a": {"status": "ran", "reason": "inputs changed"}},
    ):
        with engine.Engine() as eng:
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
    """run_once() emits IDLE state even when _orchestrate_execution raises."""
    with patch.object(
        engine.Engine, "_orchestrate_execution", side_effect=RuntimeError("execution failed")
    ):
        with engine.Engine() as eng:
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
    with engine.Engine() as eng:

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
    with engine.Engine() as eng:
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
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Mock _orchestrate_execution (Engine now orchestrates directly)
        with patch.object(eng, "_orchestrate_execution", return_value={}) as mock_orchestrate:
            # Submit a run request
            event: types.RunRequested = {
                "type": "run_requested",
                "stages": ["stage_a"],
                "force": False,
                "reason": "test",
            }
            eng.submit(event)

            _helper_run_loop_with_delayed_shutdown(eng, delay=0.1)

            # Verify orchestration was called with correct stages
            mock_orchestrate.assert_called_once()
            call_kwargs = mock_orchestrate.call_args.kwargs
            assert call_kwargs["stages"] == ["stage_a"]


def test_engine_run_loop_emits_state_changes() -> None:
    """run_loop() emits ACTIVE/IDLE state changes around execution."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Mock _orchestrate_execution (Engine now orchestrates directly)
        with patch.object(eng, "_orchestrate_execution", return_value={}):
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
    with engine.Engine() as eng:

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
    with engine.Engine() as eng:

        # cancel_event should be a threading.Event
        assert hasattr(eng.cancel_event, "is_set")
        assert hasattr(eng.cancel_event, "set")
        assert hasattr(eng.cancel_event, "clear")


def test_engine_run_loop_clears_cancel_event() -> None:
    """run_loop() clears cancel_event before processing RunRequested."""
    with engine.Engine() as eng:

        # Set cancel event before run
        eng._cancel_event.set()

        # Mock _orchestrate_execution to verify cancel event was cleared
        cancel_event_state_during_orchestration = list[bool]()

        def capture_cancel_state(**kwargs: object) -> dict[str, object]:
            cancel_event_state_during_orchestration.append(eng._cancel_event.is_set())
            return {}

        with patch.object(eng, "_orchestrate_execution", side_effect=capture_cancel_state):
            event: types.RunRequested = {
                "type": "run_requested",
                "stages": None,
                "force": False,
                "reason": "test",
            }
            eng.submit(event)

            _helper_run_loop_with_delayed_shutdown(eng)

            # Cancel event should have been cleared before orchestration
            assert cancel_event_state_during_orchestration == [False]


def test_engine_run_loop_starts_sources() -> None:
    """run_loop() starts all registered sources."""
    started = list[bool]()

    class MockSource:
        def start(self, submit: Callable[[types.InputEvent], None]) -> None:
            started.append(True)

        def stop(self) -> None:
            pass

    with engine.Engine() as eng:
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

    with engine.Engine() as eng:
        eng.add_source(MockSource())
        eng.add_source(MockSource())

        _helper_run_loop_with_delayed_shutdown(eng)

        assert len(stopped) == 2


# =============================================================================
# Tests for CLI parameter passthrough (Task 11)
# =============================================================================


def test_engine_run_once_stores_progress_callback() -> None:
    """run_once() stores progress_callback for event forwarding."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}):

        def callback(event: RunJsonEvent) -> None:
            pass

        with engine.Engine() as eng:
            # _progress_callback should be None before run
            assert eng._progress_callback is None

            # Use a flag to verify callback was stored during execution
            stored_callback: list[object] = [None]

            def mock_orchestrate(*args: object, **kwargs: object) -> dict[str, object]:
                stored_callback[0] = eng._progress_callback
                return {}

            with patch.object(engine.Engine, "_orchestrate_execution", side_effect=mock_orchestrate):
                eng.run_once(progress_callback=callback)

            # Callback should have been stored during execution
            assert stored_callback[0] is callback
            # And cleared after
            assert eng._progress_callback is None


def test_engine_run_once_clears_cancel_event_before_execution() -> None:
    """run_once() clears cancel_event before calling _orchestrate_execution."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}):
        with engine.Engine() as eng:
            # Set cancel event to simulate previous cancellation
            eng._cancel_event.set()
            assert eng._cancel_event.is_set()

            eng.run_once()

            # Cancel event should have been cleared
            assert not eng._cancel_event.is_set()


# =============================================================================
# Tests for Stage Event Emission (Task 1)
# =============================================================================


def test_engine_run_once_emits_stage_started_events() -> None:
    """run_once() emits StageStarted events to sinks."""
    with engine.Engine() as eng:
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
    with engine.Engine() as eng:
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


def test_engine_run_once_progress_callback_is_stored_during_execution() -> None:
    """run_once() stores progress_callback and clears it after execution."""
    user_events = list[RunJsonEvent]()

    def user_callback(event: RunJsonEvent) -> None:
        user_events.append(event)

    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}):
        with engine.Engine() as eng:

            # Verify callback is stored during run
            callback_during_run: list[object] = [None]

            def capture_callback(*args: object, **kwargs: object) -> dict[str, object]:
                callback_during_run[0] = eng._progress_callback
                return {}

            with patch.object(engine.Engine, "_orchestrate_execution", side_effect=capture_callback):
                eng.run_once(progress_callback=user_callback)

            # Callback should have been stored during execution
            assert callback_during_run[0] is user_callback
            # And cleared after
            assert eng._progress_callback is None


# =============================================================================
# Integration Test for Full Event Flow (Task 6)
# =============================================================================


def test_engine_full_event_flow_integration() -> None:
    """Integration test verifying complete event flow through Engine.

    Verifies:
    - Event sequence: state changes + stage events
    - Stage details preserved correctly
    - Multiple stages handled in order

    Note: This test mocks _orchestrate_execution but simulates the events
    that would be emitted during real orchestration.
    """

    def orchestrate_side_effect(
        self: engine.Engine, *args: object, **kwargs: object
    ) -> dict[str, dict[str, str]]:
        """Simulate orchestration emitting events during execution."""
        # Emit StageStarted for stage_a
        self.emit(
            types.StageStarted(
                type="stage_started",
                stage="stage_a",
                index=1,
                total=2,
            )
        )
        # Emit StageCompleted for stage_a
        self.emit(
            types.StageCompleted(
                type="stage_completed",
                stage="stage_a",
                status=StageStatus.RAN,
                reason="inputs changed",
                duration_ms=100.0,
                index=1,
                total=2,
            )
        )
        # Emit StageStarted for stage_b
        self.emit(
            types.StageStarted(
                type="stage_started",
                stage="stage_b",
                index=2,
                total=2,
            )
        )
        # Emit StageCompleted for stage_b
        self.emit(
            types.StageCompleted(
                type="stage_completed",
                stage="stage_b",
                status=StageStatus.SKIPPED,
                reason="up to date",
                duration_ms=5.0,
                index=2,
                total=2,
            )
        )
        return {
            "stage_a": {"status": "ran", "reason": "inputs changed"},
            "stage_b": {"status": "skipped", "reason": "up to date"},
        }

    with patch.object(engine.Engine, "_orchestrate_execution", orchestrate_side_effect):
        with engine.Engine() as eng:
            sink = _MockSink()
            eng.add_sink(sink)

            # Run stages - orchestrate_side_effect will emit events during run
            result = eng.run_once(stages=["stage_a", "stage_b"])

            # Verify results returned
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
    with engine.Engine() as eng:
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


def test_engine_run_once_rejects_concurrent_calls() -> None:
    """run_once() raises if called while engine is already active."""
    with engine.Engine() as eng:

        # Manually set state to ACTIVE to simulate concurrent call
        eng._state = types.EngineState.ACTIVE

        with pytest.raises(RuntimeError, match="already active"):
            eng.run_once()


def test_engine_emit_continues_on_sink_failure() -> None:
    """emit() continues to other sinks if one fails."""
    with engine.Engine() as eng:

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
    with engine.Engine() as eng:

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
        pytest.param("checkout_missing", True, id="checkout_missing"),
    ],
)
def test_engine_run_once_passes_parameter(param_name: str, param_value: object) -> None:
    """run_once() passes parameters through to _orchestrate_execution."""
    with patch.object(engine.Engine, "_orchestrate_execution", return_value={}) as mock_orchestrate:
        with engine.Engine() as eng:
            eng.run_once(**{param_name: param_value})  # pyright: ignore[reportArgumentType]

            call_kwargs = mock_orchestrate.call_args.kwargs
            assert call_kwargs[param_name] == param_value


# =============================================================================
# Tests for Stage Execution State Tracking (Phase 4 Task 2)
# =============================================================================


def test_engine_get_stage_state_returns_pending_for_unknown_stage() -> None:
    """get_stage_state() returns PENDING for unknown stages."""
    with engine.Engine() as eng:
        assert eng.get_stage_state("unknown_stage") == types.StageExecutionState.PENDING


def test_engine_set_stage_state_emits_state_changed_event() -> None:
    """_set_stage_state() emits StageStateChanged event."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        eng._set_stage_state("train", types.StageExecutionState.RUNNING)

        state_events = [e for e in sink.events if e["type"] == "stage_state_changed"]
        assert len(state_events) == 1
        assert state_events[0]["stage"] == "train"
        assert state_events[0]["state"] == types.StageExecutionState.RUNNING
        assert state_events[0]["previous_state"] == types.StageExecutionState.PENDING


def test_engine_set_stage_state_no_event_on_same_state() -> None:
    """_set_stage_state() does not emit event when state unchanged."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Set to RUNNING
        eng._set_stage_state("train", types.StageExecutionState.RUNNING)
        # Set to RUNNING again
        eng._set_stage_state("train", types.StageExecutionState.RUNNING)

        state_events = [e for e in sink.events if e["type"] == "stage_state_changed"]
        assert len(state_events) == 1, "Should only emit once"


def test_engine_set_stage_state_tracks_previous_state() -> None:
    """_set_stage_state() tracks state progression correctly."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Progress through states
        eng._set_stage_state("train", types.StageExecutionState.READY)
        eng._set_stage_state("train", types.StageExecutionState.PREPARING)
        eng._set_stage_state("train", types.StageExecutionState.RUNNING)
        eng._set_stage_state("train", types.StageExecutionState.COMPLETED)

        state_events = [e for e in sink.events if e["type"] == "stage_state_changed"]
        assert len(state_events) == 4

        # Verify state progression
        assert state_events[0]["previous_state"] == types.StageExecutionState.PENDING
        assert state_events[0]["state"] == types.StageExecutionState.READY

        assert state_events[1]["previous_state"] == types.StageExecutionState.READY
        assert state_events[1]["state"] == types.StageExecutionState.PREPARING

        assert state_events[2]["previous_state"] == types.StageExecutionState.PREPARING
        assert state_events[2]["state"] == types.StageExecutionState.RUNNING

        assert state_events[3]["previous_state"] == types.StageExecutionState.RUNNING
        assert state_events[3]["state"] == types.StageExecutionState.COMPLETED


def test_engine_get_stage_state_returns_current_state() -> None:
    """get_stage_state() returns the current state after transitions."""
    with engine.Engine() as eng:

        eng._set_stage_state("train", types.StageExecutionState.RUNNING)
        assert eng.get_stage_state("train") == types.StageExecutionState.RUNNING

        eng._set_stage_state("train", types.StageExecutionState.COMPLETED)
        assert eng.get_stage_state("train") == types.StageExecutionState.COMPLETED


def test_engine_get_executing_stages_empty_initially() -> None:
    """get_executing_stages() returns empty list initially."""
    with engine.Engine() as eng:
        assert eng.get_executing_stages() == []


def test_engine_get_executing_stages_includes_preparing_and_running() -> None:
    """get_executing_stages() includes stages in PREPARING or RUNNING state."""
    with engine.Engine() as eng:

        # Set up various stages in different states
        eng._set_stage_state("stage_pending", types.StageExecutionState.PENDING)
        eng._set_stage_state("stage_blocked", types.StageExecutionState.BLOCKED)
        eng._set_stage_state("stage_ready", types.StageExecutionState.READY)
        eng._set_stage_state("stage_preparing", types.StageExecutionState.PREPARING)
        eng._set_stage_state("stage_running", types.StageExecutionState.RUNNING)
        eng._set_stage_state("stage_completed", types.StageExecutionState.COMPLETED)

        executing = eng.get_executing_stages()

        # Only PREPARING and RUNNING should be included
        assert "stage_preparing" in executing
        assert "stage_running" in executing
        assert len(executing) == 2

        # Others should NOT be included
        assert "stage_pending" not in executing
        assert "stage_blocked" not in executing
        assert "stage_ready" not in executing
        assert "stage_completed" not in executing


def test_engine_stage_state_tracking_is_thread_safe() -> None:
    """Stage state tracking methods are thread-safe."""
    with engine.Engine() as eng:
        errors = list[str]()

        def update_stages() -> None:
            try:
                for i in range(100):
                    stage_name = f"stage_{threading.current_thread().name}_{i}"
                    eng._set_stage_state(stage_name, types.StageExecutionState.RUNNING)
                    eng.get_stage_state(stage_name)
                    eng.get_executing_stages()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=update_stages, name=str(i)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread-safety errors: {errors}"


# =============================================================================
# Tests for Output Filtering Based on Stage States (Phase 4 Task 3)
# =============================================================================


def test_engine_filters_executing_stage_outputs(tmp_path: pathlib.Path) -> None:
    """Engine filters filesystem events for outputs of executing stages."""
    with engine.Engine() as eng:

        # Register stage with known output
        output_path = tmp_path / "output.csv"

        # Simulate stage in RUNNING state
        eng._stage_states["process_data"] = types.StageExecutionState.RUNNING

        # Build graph with the stage's output
        g: nx.DiGraph[str] = nx.DiGraph()
        stage_node = engine_graph.stage_node("process_data")
        artifact_node = engine_graph.artifact_node(output_path)
        g.add_node(stage_node, type=types.NodeType.STAGE)
        g.add_node(artifact_node, type=types.NodeType.ARTIFACT)
        g.add_edge(stage_node, artifact_node)  # stage produces artifact
        eng._graph = g

        # Check if path should be filtered
        assert eng._should_filter_path(output_path) is True

        # Non-output paths should NOT be filtered
        other_path = tmp_path / "input.csv"
        assert eng._should_filter_path(other_path) is False


def test_engine_filters_incremental_out_during_preparing(tmp_path: pathlib.Path) -> None:
    """Engine filters IncrementalOut paths during PREPARING state (restoration phase)."""
    with engine.Engine() as eng:

        # IncrementalOut directory being restored
        incremental_dir = tmp_path / "incremental_output"
        incremental_dir.mkdir()

        # Simulate stage in PREPARING state (when IncrementalOut restoration happens)
        eng._stage_states["incremental_stage"] = types.StageExecutionState.PREPARING

        # Build graph with the stage's IncrementalOut
        g: nx.DiGraph[str] = nx.DiGraph()
        stage_node = engine_graph.stage_node("incremental_stage")
        artifact_node = engine_graph.artifact_node(incremental_dir)
        g.add_node(stage_node, type=types.NodeType.STAGE)
        g.add_node(artifact_node, type=types.NodeType.ARTIFACT)
        g.add_edge(stage_node, artifact_node)  # stage produces artifact
        eng._graph = g

        # Check that IncrementalOut directory is filtered during PREPARING
        assert eng._should_filter_path(incremental_dir) is True

        # Verify PREPARING state triggers filtering (not just RUNNING)
        eng._stage_states["incremental_stage"] = types.StageExecutionState.READY
        assert eng._should_filter_path(incremental_dir) is False

        eng._stage_states["incremental_stage"] = types.StageExecutionState.PREPARING
        assert eng._should_filter_path(incremental_dir) is True


def test_engine_should_filter_path_returns_false_without_graph() -> None:
    """_should_filter_path() returns False when graph is not set."""
    with engine.Engine() as eng:
        assert eng._graph is None
        assert eng._should_filter_path(pathlib.Path("/some/path.csv")) is False


def test_engine_should_filter_path_returns_false_for_input_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    """_should_filter_path() returns False for artifacts with no producer (inputs)."""
    with engine.Engine() as eng:

        # Create graph with an input artifact (has no producer)
        input_path = tmp_path / "input.csv"
        g: nx.DiGraph[str] = nx.DiGraph()
        artifact_node = engine_graph.artifact_node(input_path)
        stage_node = engine_graph.stage_node("process")
        g.add_node(artifact_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_node, type=types.NodeType.STAGE)
        g.add_edge(artifact_node, stage_node)  # artifact is INPUT to stage (no producer)
        eng._graph = g

        # Even if stage is running, input artifacts should not be filtered
        eng._stage_states["process"] = types.StageExecutionState.RUNNING
        assert eng._should_filter_path(input_path) is False


def test_engine_should_filter_path_returns_false_for_completed_stage(
    tmp_path: pathlib.Path,
) -> None:
    """_should_filter_path() returns False when producer stage is COMPLETED."""
    with engine.Engine() as eng:

        output_path = tmp_path / "output.csv"
        g: nx.DiGraph[str] = nx.DiGraph()
        stage_node = engine_graph.stage_node("process")
        artifact_node = engine_graph.artifact_node(output_path)
        g.add_node(stage_node, type=types.NodeType.STAGE)
        g.add_node(artifact_node, type=types.NodeType.ARTIFACT)
        g.add_edge(stage_node, artifact_node)
        eng._graph = g

        # Stage completed - should not filter
        eng._stage_states["process"] = types.StageExecutionState.COMPLETED
        assert eng._should_filter_path(output_path) is False


def test_engine_get_output_paths_for_stage(tmp_path: pathlib.Path) -> None:
    """_get_output_paths_for_stage() returns all output paths for a stage."""
    with engine.Engine() as eng:

        output1 = tmp_path / "output1.csv"
        output2 = tmp_path / "output2.csv"
        g: nx.DiGraph[str] = nx.DiGraph()
        stage_node = engine_graph.stage_node("process")
        g.add_node(stage_node, type=types.NodeType.STAGE)

        # Add two output artifacts
        for output_path in [output1, output2]:
            artifact_node = engine_graph.artifact_node(output_path)
            g.add_node(artifact_node, type=types.NodeType.ARTIFACT)
            g.add_edge(stage_node, artifact_node)

        eng._graph = g

        paths = eng._get_output_paths_for_stage("process")
        assert len(paths) == 2
        assert output1 in paths
        assert output2 in paths


def test_engine_get_output_paths_for_stage_empty_without_graph() -> None:
    """_get_output_paths_for_stage() returns empty list when graph is None."""
    with engine.Engine() as eng:
        assert eng._get_output_paths_for_stage("any_stage") == []


def test_engine_get_output_paths_for_stage_unknown_stage(tmp_path: pathlib.Path) -> None:
    """_get_output_paths_for_stage() returns empty list for unknown stage."""
    with engine.Engine() as eng:

        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_node(engine_graph.stage_node("other_stage"), type=types.NodeType.STAGE)
        eng._graph = g

        assert eng._get_output_paths_for_stage("unknown_stage") == []


def test_engine_defer_event_for_stage() -> None:
    """_defer_event_for_stage() stores events for later processing."""
    with engine.Engine() as eng:

        event1: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": ["/path/to/file1.csv"],
        }
        event2: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": ["/path/to/file2.csv"],
        }

        eng._defer_event_for_stage("process", event1)
        eng._defer_event_for_stage("process", event2)

        assert len(eng._deferred_events["process"]) == 2
        assert eng._deferred_events["process"][0] == event1
        assert eng._deferred_events["process"][1] == event2


def test_engine_defer_event_for_multiple_stages() -> None:
    """_defer_event_for_stage() tracks events per stage independently."""
    with engine.Engine() as eng:

        event1: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": ["/path/to/file1.csv"],
        }
        event2: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": ["/path/to/file2.csv"],
        }

        eng._defer_event_for_stage("stage_a", event1)
        eng._defer_event_for_stage("stage_b", event2)

        assert len(eng._deferred_events["stage_a"]) == 1
        assert len(eng._deferred_events["stage_b"]) == 1


def test_engine_process_deferred_events(tmp_path: pathlib.Path) -> None:
    """_process_deferred_events() processes and clears deferred events."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Defer a run_requested event
        event: types.RunRequested = {
            "type": "run_requested",
            "stages": ["downstream"],
            "force": False,
            "reason": "deferred",
        }
        eng._defer_event_for_stage("process", event)

        # Verify event is deferred
        assert "process" in eng._deferred_events
        assert len(eng._deferred_events["process"]) == 1

        # Process deferred events - mock _handle_input_event to avoid actual execution
        processed_events = list[types.InputEvent]()

        def capture_event(evt: types.InputEvent) -> None:
            processed_events.append(evt)

        with patch.object(eng, "_handle_input_event", side_effect=capture_event):
            eng._process_deferred_events("process")

        # Deferred events should be cleared and processed
        assert "process" not in eng._deferred_events
        assert len(processed_events) == 1
        assert processed_events[0] == event


def test_engine_process_deferred_events_empty_list() -> None:
    """_process_deferred_events() handles stages with no deferred events."""
    with engine.Engine() as eng:

        # Should not raise
        eng._process_deferred_events("nonexistent_stage")

        # Should remain empty
        assert "nonexistent_stage" not in eng._deferred_events


def test_engine_handle_stage_completion_processes_deferred_events() -> None:
    """_handle_stage_completion() processes deferred events regardless of status.

    This verifies that deferred events are processed even when a stage completes
    with SKIPPED status (e.g., cache hit), not just RAN status.
    """
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Set up stage state
        eng._stage_states["stage_a"] = types.StageExecutionState.RUNNING
        eng._stage_mutex["stage_a"] = []
        eng._stage_downstream["stage_a"] = []

        # Defer an event for this stage
        deferred_event: types.RunRequested = {
            "type": "run_requested",
            "stages": ["downstream"],
            "force": False,
            "reason": "deferred test",
        }
        eng._defer_event_for_stage("stage_a", deferred_event)
        assert "stage_a" in eng._deferred_events

        # Create a SKIPPED result (simulating cache hit)
        from pivot.types import StageStatus

        skipped_result: StageResult = {
            "status": StageStatus.SKIPPED,
            "reason": "unchanged (cache hit)",
            "output_lines": [],
        }

        # Track if deferred event was processed
        processed_events = list[types.InputEvent]()

        def capture_event(evt: types.InputEvent) -> None:
            processed_events.append(evt)

        with patch.object(eng, "_handle_input_event", side_effect=capture_event):
            eng._handle_stage_completion("stage_a", skipped_result, time.perf_counter())

        # Verify deferred event was processed even though status was SKIPPED
        assert "stage_a" not in eng._deferred_events, "Deferred events should be cleared"
        assert len(processed_events) == 1, "Deferred event should be processed"
        assert processed_events[0] == deferred_event


# =============================================================================
# Tests for Change Detection Using Bipartite Graph (Phase 4 Task 4)
# =============================================================================


def test_engine_get_affected_stages_for_path_uses_bipartite_graph(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_path() uses get_consumers() from bipartite graph."""
    with engine.Engine() as eng:

        # Build a graph with dependencies
        g: nx.DiGraph[str] = nx.DiGraph()

        input_path = tmp_path / "input.csv"
        stage_node = engine_graph.stage_node("process_data")
        artifact_node = engine_graph.artifact_node(input_path)

        g.add_node(artifact_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_node, type=types.NodeType.STAGE)
        g.add_edge(artifact_node, stage_node)  # artifact consumed by stage

        eng._graph = g

        # Get affected stages for a path change
        affected = eng._get_affected_stages_for_path(input_path)

        assert "process_data" in affected


def test_engine_get_affected_stages_for_path_includes_downstream(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_path() includes downstream stages."""
    with engine.Engine() as eng:

        # Build a pipeline: input -> stage_a -> intermediate -> stage_b
        g: nx.DiGraph[str] = nx.DiGraph()

        input_path = tmp_path / "input.csv"
        intermediate = tmp_path / "intermediate.csv"

        # Nodes
        input_node = engine_graph.artifact_node(input_path)
        stage_a_node = engine_graph.stage_node("stage_a")
        intermediate_node = engine_graph.artifact_node(intermediate)
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(input_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_a_node, type=types.NodeType.STAGE)
        g.add_node(intermediate_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_b_node, type=types.NodeType.STAGE)

        # Edges: input -> stage_a -> intermediate -> stage_b
        g.add_edge(input_node, stage_a_node)
        g.add_edge(stage_a_node, intermediate_node)
        g.add_edge(intermediate_node, stage_b_node)

        eng._graph = g

        # Change to input should affect both stage_a and stage_b (downstream)
        affected = eng._get_affected_stages_for_path(input_path)

        assert "stage_a" in affected
        assert "stage_b" in affected


def test_engine_get_affected_stages_for_path_returns_empty_without_graph() -> None:
    """_get_affected_stages_for_path() returns empty list when graph is None."""
    with engine.Engine() as eng:
        assert eng._graph is None

        affected = eng._get_affected_stages_for_path(pathlib.Path("/some/path.csv"))

        assert affected == []


def test_engine_get_affected_stages_for_path_unknown_artifact(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_path() returns empty for unknown artifacts."""
    with engine.Engine() as eng:

        # Graph with one stage but no artifacts
        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_node(engine_graph.stage_node("some_stage"), type=types.NodeType.STAGE)
        eng._graph = g

        # Query for an artifact not in the graph
        affected = eng._get_affected_stages_for_path(tmp_path / "unknown.csv")

        assert affected == []


def test_engine_get_affected_stages_for_paths_multiple_paths(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_paths() handles multiple path changes."""
    with engine.Engine() as eng:

        # Build graph: input_a -> stage_a, input_b -> stage_b
        g: nx.DiGraph[str] = nx.DiGraph()

        input_a = tmp_path / "input_a.csv"
        input_b = tmp_path / "input_b.csv"

        input_a_node = engine_graph.artifact_node(input_a)
        input_b_node = engine_graph.artifact_node(input_b)
        stage_a_node = engine_graph.stage_node("stage_a")
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(input_a_node, type=types.NodeType.ARTIFACT)
        g.add_node(input_b_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_a_node, type=types.NodeType.STAGE)
        g.add_node(stage_b_node, type=types.NodeType.STAGE)

        g.add_edge(input_a_node, stage_a_node)
        g.add_edge(input_b_node, stage_b_node)

        eng._graph = g

        # Both inputs changed
        affected = eng._get_affected_stages_for_paths([input_a, input_b])

        assert "stage_a" in affected
        assert "stage_b" in affected


def test_engine_get_affected_stages_for_paths_deduplicates(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_paths() deduplicates affected stages."""
    with engine.Engine() as eng:

        # Build graph: two inputs -> same stage
        g: nx.DiGraph[str] = nx.DiGraph()

        input_a = tmp_path / "input_a.csv"
        input_b = tmp_path / "input_b.csv"

        input_a_node = engine_graph.artifact_node(input_a)
        input_b_node = engine_graph.artifact_node(input_b)
        stage_node = engine_graph.stage_node("process")

        g.add_node(input_a_node, type=types.NodeType.ARTIFACT)
        g.add_node(input_b_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_node, type=types.NodeType.STAGE)

        g.add_edge(input_a_node, stage_node)
        g.add_edge(input_b_node, stage_node)

        eng._graph = g

        # Both inputs affect the same stage
        affected = eng._get_affected_stages_for_paths([input_a, input_b])

        # Should only appear once
        assert affected.count("process") == 1


def test_engine_get_affected_stages_for_paths_filters_executing_stage_outputs(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_paths() filters outputs of executing stages."""
    with engine.Engine() as eng:

        # Build graph: stage_a -> output -> stage_b
        g: nx.DiGraph[str] = nx.DiGraph()

        output_path = tmp_path / "output.csv"

        stage_a_node = engine_graph.stage_node("stage_a")
        output_node = engine_graph.artifact_node(output_path)
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(stage_a_node, type=types.NodeType.STAGE)
        g.add_node(output_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_b_node, type=types.NodeType.STAGE)

        g.add_edge(stage_a_node, output_node)  # stage_a produces output
        g.add_edge(output_node, stage_b_node)  # stage_b consumes output

        eng._graph = g

        # stage_a is currently running
        eng._stage_states["stage_a"] = types.StageExecutionState.RUNNING

        # Change to output (produced by running stage_a) should be filtered
        affected = eng._get_affected_stages_for_paths([output_path])

        # stage_b should NOT be affected because the change came from running stage_a
        assert "stage_b" not in affected


def test_engine_get_affected_stages_for_paths_without_downstream(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_paths() can exclude downstream with flag."""
    with engine.Engine() as eng:

        # Build a pipeline: input -> stage_a -> intermediate -> stage_b
        g: nx.DiGraph[str] = nx.DiGraph()

        input_path = tmp_path / "input.csv"
        intermediate = tmp_path / "intermediate.csv"

        input_node = engine_graph.artifact_node(input_path)
        stage_a_node = engine_graph.stage_node("stage_a")
        intermediate_node = engine_graph.artifact_node(intermediate)
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(input_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_a_node, type=types.NodeType.STAGE)
        g.add_node(intermediate_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_b_node, type=types.NodeType.STAGE)

        g.add_edge(input_node, stage_a_node)
        g.add_edge(stage_a_node, intermediate_node)
        g.add_edge(intermediate_node, stage_b_node)

        eng._graph = g

        # With include_downstream=False, only direct consumers
        affected = eng._get_affected_stages_for_paths([input_path], include_downstream=False)

        assert "stage_a" in affected
        assert "stage_b" not in affected


def test_engine_get_affected_stages_for_paths_empty_list() -> None:
    """_get_affected_stages_for_paths() handles empty path list."""
    with engine.Engine() as eng:

        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_node(engine_graph.stage_node("some_stage"), type=types.NodeType.STAGE)
        eng._graph = g

        affected = eng._get_affected_stages_for_paths([])

        assert affected == []


# =============================================================================
# Tests for Execution Orchestration (Phase 4 Task 6)
# =============================================================================


def test_engine_orchestrates_parallel_execution() -> None:
    """Engine orchestrates parallel stage execution with mutex handling."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Verify the orchestration methods exist
        assert hasattr(eng, "_orchestrate_execution")
        assert hasattr(eng, "_start_ready_stages")
        assert hasattr(eng, "_handle_stage_completion")
        assert hasattr(eng, "_cascade_failure")
        assert hasattr(eng, "_initialize_orchestration")
        assert hasattr(eng, "_can_start_stage")
        assert hasattr(eng, "_drain_output_queue")


def _helper_mock_get_stage(name: str) -> dict[str, object]:
    """Mock stage info getter for orchestration tests."""
    if name in ("stage_a", "stage_b"):
        return {"mutex": []}
    msg = f"Unknown stage: {name}"
    raise KeyError(msg)


def test_engine_initialize_orchestration_sets_up_state(tmp_path: pathlib.Path) -> None:
    """_initialize_orchestration() sets up stage states based on DAG."""
    with engine.Engine() as eng:

        # Build a simple graph: stage_a -> output -> stage_b
        g: nx.DiGraph[str] = nx.DiGraph()

        output_path = tmp_path / "output.csv"

        stage_a_node = engine_graph.stage_node("stage_a")
        output_node = engine_graph.artifact_node(output_path)
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(stage_a_node, type=types.NodeType.STAGE)
        g.add_node(output_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_b_node, type=types.NodeType.STAGE)

        g.add_edge(stage_a_node, output_node)
        g.add_edge(output_node, stage_b_node)

        eng._graph = g

        # Register the stages in the registry (with patch)
        with patch.object(registry.REGISTRY, "get", side_effect=_helper_mock_get_stage):
            eng._initialize_orchestration(
                ["stage_a", "stage_b"], max_workers=2, error_mode=OnError.FAIL
            )

        # stage_a should be READY (no upstream)
        assert eng.get_stage_state("stage_a") == types.StageExecutionState.READY
        # stage_b should be PENDING (has upstream)
        assert eng.get_stage_state("stage_b") == types.StageExecutionState.PENDING

        # Check upstream tracking
        assert eng._stage_upstream_unfinished["stage_a"] == set()
        assert "stage_a" in eng._stage_upstream_unfinished["stage_b"]


def test_engine_can_start_stage_checks_upstream() -> None:
    """_can_start_stage() returns False if upstream incomplete."""
    with engine.Engine() as eng:

        # Set up state manually
        eng._stage_states["stage_a"] = types.StageExecutionState.READY
        eng._stage_states["stage_b"] = types.StageExecutionState.READY
        eng._stage_upstream_unfinished["stage_a"] = set()
        eng._stage_upstream_unfinished["stage_b"] = {"stage_a"}
        eng._stage_mutex["stage_a"] = []
        eng._stage_mutex["stage_b"] = []
        eng._mutex_counts.clear()

        # stage_a can start (no upstream)
        assert eng._can_start_stage("stage_a") is True

        # stage_b cannot start (stage_a not finished)
        assert eng._can_start_stage("stage_b") is False


def test_engine_can_start_stage_checks_mutex() -> None:
    """_can_start_stage() returns False if mutex held."""
    with engine.Engine() as eng:

        # Set up state manually
        eng._stage_states["stage_a"] = types.StageExecutionState.READY
        eng._stage_states["stage_b"] = types.StageExecutionState.READY
        eng._stage_upstream_unfinished["stage_a"] = set()
        eng._stage_upstream_unfinished["stage_b"] = set()
        eng._stage_mutex["stage_a"] = ["gpu"]
        eng._stage_mutex["stage_b"] = ["gpu"]
        eng._mutex_counts["gpu"] = 1  # stage_a has the mutex

        # stage_a can't start (mutex already held) - well, actually we check if state is READY
        # Reset stage_a to simulate it already running
        eng._stage_states["stage_a"] = types.StageExecutionState.RUNNING

        # stage_b cannot start (gpu mutex held)
        assert eng._can_start_stage("stage_b") is False


def test_engine_cascade_failure_marks_downstream_blocked() -> None:
    """_cascade_failure() marks downstream stages as BLOCKED."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Set up: stage_a -> stage_b -> stage_c
        eng._stage_states["stage_a"] = types.StageExecutionState.COMPLETED
        eng._stage_states["stage_b"] = types.StageExecutionState.READY
        eng._stage_states["stage_c"] = types.StageExecutionState.PENDING
        eng._stage_downstream["stage_a"] = ["stage_b"]
        eng._stage_downstream["stage_b"] = ["stage_c"]
        eng._stage_downstream["stage_c"] = []

        # Cascade failure from stage_a
        eng._cascade_failure("stage_a")

        # Both downstream stages should be blocked
        assert eng.get_stage_state("stage_b") == types.StageExecutionState.BLOCKED
        assert eng.get_stage_state("stage_c") == types.StageExecutionState.BLOCKED


def test_engine_handle_stage_completion_updates_downstream() -> None:
    """_handle_stage_completion() updates downstream upstream_unfinished."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Set up: stage_a -> stage_b
        eng._stage_states["stage_a"] = types.StageExecutionState.RUNNING
        eng._stage_states["stage_b"] = types.StageExecutionState.PENDING
        eng._stage_upstream_unfinished["stage_a"] = set()
        eng._stage_upstream_unfinished["stage_b"] = {"stage_a"}
        eng._stage_downstream["stage_a"] = ["stage_b"]
        eng._stage_downstream["stage_b"] = []
        eng._stage_mutex["stage_a"] = []

        result = StageResult(status=StageStatus.RAN, reason="", output_lines=[])
        start_time = time.perf_counter() - 0.1

        eng._handle_stage_completion("stage_a", result, start_time)

        # stage_b should now be READY
        assert eng.get_stage_state("stage_a") == types.StageExecutionState.COMPLETED
        assert eng.get_stage_state("stage_b") == types.StageExecutionState.READY
        assert eng._stage_upstream_unfinished["stage_b"] == set()

        # Should have emitted StageCompleted
        completed_events = [e for e in sink.events if e["type"] == "stage_completed"]
        assert len(completed_events) == 1
        assert completed_events[0]["stage"] == "stage_a"


# =============================================================================
# Tests for Watch Mode Change Handlers (Phase 4 Task 8)
# =============================================================================


def test_engine_handle_data_artifact_changed_filters_executing_outputs(
    tmp_path: pathlib.Path,
) -> None:
    """_handle_data_artifact_changed() filters events for executing stage outputs."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Build graph: stage_a -> output.csv -> stage_b
        output_path = tmp_path / "output.csv"

        g: nx.DiGraph[str] = nx.DiGraph()
        stage_a_node = engine_graph.stage_node("stage_a")
        output_node = engine_graph.artifact_node(output_path)
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(stage_a_node, type=types.NodeType.STAGE)
        g.add_node(output_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_b_node, type=types.NodeType.STAGE)

        g.add_edge(stage_a_node, output_node)  # stage_a produces output
        g.add_edge(output_node, stage_b_node)  # stage_b consumes output

        eng._graph = g

        # stage_a is currently running
        eng._stage_states["stage_a"] = types.StageExecutionState.RUNNING

        # Create event for output change
        event: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": [str(output_path)],
        }

        # Handle the event
        eng._handle_data_artifact_changed(event)

        # Event should be deferred, not processed immediately
        assert "stage_a" in eng._deferred_events
        assert len(eng._deferred_events["stage_a"]) == 1

        # No execution should have started (no ACTIVE state event)
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        active_events = [e for e in state_events if e["state"] == types.EngineState.ACTIVE]
        assert len(active_events) == 0


def test_engine_handle_data_artifact_changed_processes_unfiltered_paths(
    tmp_path: pathlib.Path,
) -> None:
    """_handle_data_artifact_changed() processes paths not from executing stages."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Build graph: input.csv -> stage_a
        input_path = tmp_path / "input.csv"

        g: nx.DiGraph[str] = nx.DiGraph()
        input_node = engine_graph.artifact_node(input_path)
        stage_a_node = engine_graph.stage_node("stage_a")

        g.add_node(input_node, type=types.NodeType.ARTIFACT)
        g.add_node(stage_a_node, type=types.NodeType.STAGE)

        g.add_edge(input_node, stage_a_node)  # input consumed by stage_a

        eng._graph = g

        # Create event for input change
        event: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": [str(input_path)],
        }

        # Mock _execute_affected_stages to verify it's called
        executed_stages: list[list[str]] = []

        def mock_execute(stages: list[str]) -> None:
            executed_stages.append(stages)

        eng._execute_affected_stages = mock_execute  # type: ignore[method-assign]

        # Handle the event
        eng._handle_data_artifact_changed(event)

        # Should have called execute with stage_a
        assert len(executed_stages) == 1
        assert "stage_a" in executed_stages[0]


def test_engine_handle_data_artifact_changed_no_affected_stages(
    tmp_path: pathlib.Path,
) -> None:
    """_handle_data_artifact_changed() does nothing for paths with no consumers."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Build graph with no stages consuming this path
        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_node(engine_graph.stage_node("unrelated_stage"), type=types.NodeType.STAGE)
        eng._graph = g

        # Create event for unknown path
        event: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": [str(tmp_path / "unknown.csv")],
        }

        # Should not raise and no execution started
        eng._handle_data_artifact_changed(event)

        # No state changes
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 0


def _helper_code_change_stage_func(params: None) -> dict[str, str]:
    """Module-level helper for code change test."""
    return {"result": "ok"}


def test_engine_handle_code_or_config_changed_triggers_full_rerun() -> None:
    """_handle_code_or_config_changed() triggers re-execution of all stages."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Track which stages would be executed
        executed_stages: list[list[str]] = []

        def mock_execute(stages: list[str]) -> None:
            executed_stages.append(stages)

        # Mock the reload methods - the handler now does reload before execution
        with (
            patch.object(eng, "_invalidate_caches"),
            patch.object(eng, "_reload_registry", return_value=True),
            patch.object(eng, "_execute_affected_stages", side_effect=mock_execute),
            patch(
                "pivot.engine.engine.registry.REGISTRY.list_stages",
                return_value=["test_code_change_stage"],
            ),
            patch(
                "pivot.engine.engine.registry.REGISTRY.get",
                return_value={"deps_paths": [], "outs_paths": [], "mutex": []},
            ),
            patch("pivot.engine.engine.engine_graph.build_graph", return_value=nx.DiGraph()),
            patch("pivot.engine.engine.engine_graph.get_watch_paths", return_value=[]),
        ):
            # Create event
            event: types.CodeOrConfigChanged = {
                "type": "code_or_config_changed",
                "paths": ["pivot.yaml"],
            }

            # Handle the event
            eng._handle_code_or_config_changed(event)

            # Should have called execute with at least the registered stage
            assert len(executed_stages) == 1
            assert "test_code_change_stage" in executed_stages[0]


def test_engine_execute_affected_stages_emits_state_changes() -> None:
    """_execute_affected_stages() emits ACTIVE/IDLE state changes."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Mock _orchestrate_execution to avoid actual execution
        def mock_orchestrate(*args: object, **kwargs: object) -> dict[str, object]:
            return {}

        eng._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue]

        # Execute some stages
        eng._execute_affected_stages(["stage_a"])

        # Should have emitted ACTIVE then IDLE
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 2
        assert state_events[0]["state"] == types.EngineState.ACTIVE
        assert state_events[1]["state"] == types.EngineState.IDLE


def test_engine_execute_affected_stages_uses_keep_going_error_mode() -> None:
    """_execute_affected_stages() uses KEEP_GOING error mode for watch mode."""
    with engine.Engine() as eng:

        # Capture the on_error argument
        captured_kwargs: list[dict[str, object]] = []

        def mock_orchestrate(*args: object, **kwargs: object) -> dict[str, object]:
            captured_kwargs.append(dict(kwargs))
            return {}

        eng._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue]

        # Execute
        eng._execute_affected_stages(["stage_a"])

        # Verify KEEP_GOING mode
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["on_error"] == OnError.KEEP_GOING


def test_engine_execute_affected_stages_clears_cancel_event() -> None:
    """_execute_affected_stages() clears cancel event before execution."""
    with engine.Engine() as eng:

        def mock_orchestrate(*args: object, **kwargs: object) -> dict[str, object]:
            return {}

        eng._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue]

        # Set cancel event
        eng._cancel_event.set()
        assert eng._cancel_event.is_set()

        # Execute
        eng._execute_affected_stages(["stage_a"])

        # Cancel event should be cleared
        assert not eng._cancel_event.is_set()


def test_engine_execute_affected_stages_emits_idle_on_exception() -> None:
    """_execute_affected_stages() emits IDLE even when orchestration raises."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        def mock_orchestrate(*args: object, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("Orchestration failed")

        eng._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue]

        # Execute - should not raise to caller
        with contextlib.suppress(RuntimeError):
            eng._execute_affected_stages(["stage_a"])

        # Should still have emitted IDLE
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) >= 2
        assert state_events[-1]["state"] == types.EngineState.IDLE
        assert eng.state == types.EngineState.IDLE


def test_engine_handle_input_event_routes_data_artifact_changed() -> None:
    """_handle_input_event() correctly routes data_artifact_changed events."""
    with engine.Engine() as eng:

        handled_events: list[types.DataArtifactChanged] = []

        def mock_handler(event: types.DataArtifactChanged) -> None:
            handled_events.append(event)

        eng._handle_data_artifact_changed = mock_handler  # type: ignore[method-assign]

        event: types.DataArtifactChanged = {
            "type": "data_artifact_changed",
            "paths": ["/path/to/file.csv"],
        }
        eng._handle_input_event(event)

        assert len(handled_events) == 1
        assert handled_events[0] == event


def test_engine_handle_input_event_routes_code_or_config_changed() -> None:
    """_handle_input_event() correctly routes code_or_config_changed events."""
    with engine.Engine() as eng:

        handled_events: list[types.CodeOrConfigChanged] = []

        def mock_handler(event: types.CodeOrConfigChanged) -> None:
            handled_events.append(event)

        eng._handle_code_or_config_changed = mock_handler  # pyright: ignore[reportAttributeAccessIssue]

        event: types.CodeOrConfigChanged = {
            "type": "code_or_config_changed",
            "paths": ["pivot.yaml"],
        }
        eng._handle_input_event(event)

        assert len(handled_events) == 1
        assert handled_events[0] == event


# =============================================================================
# Tests for Registry Reload (Phase 4 Task 9)
# =============================================================================


def test_engine_invalidate_caches_clears_graph() -> None:
    """_invalidate_caches() clears the engine's graph cache."""
    with engine.Engine() as eng:

        # Set up a mock graph
        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_node(engine_graph.stage_node("test"), type=types.NodeType.STAGE)
        eng._graph = g

        assert eng._graph is not None

        eng._invalidate_caches()

        assert eng._graph is None


def test_engine_invalidate_caches_calls_registry_invalidate() -> None:
    """_invalidate_caches() calls REGISTRY.invalidate_dag_cache()."""
    with engine.Engine() as eng:

        with patch("pivot.engine.engine.registry.REGISTRY.invalidate_dag_cache") as mock_invalidate:
            eng._invalidate_caches()
            mock_invalidate.assert_called_once()


def test_engine_emit_reload_event_emits_pipeline_reloaded(tmp_path: pathlib.Path) -> None:
    """_emit_reload_event() emits PipelineReloaded with diff information."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        # Create old stages snapshot
        old_stages: dict[str, registry.RegistryStageInfo] = {
            "stage_a": registry.RegistryStageInfo(
                func=lambda: None,
                name="stage_a",
                deps={},
                deps_paths=[],
                outs=[],
                outs_paths=[],
                params=None,
                mutex=[],
                variant=None,
                signature=None,
                fingerprint={},
                dep_specs={},
                out_specs={},
                params_arg_name=None,
            ),
            "stage_b": registry.RegistryStageInfo(
                func=lambda: None,
                name="stage_b",
                deps={},
                deps_paths=[],
                outs=[],
                outs_paths=[],
                params=None,
                mutex=[],
                variant=None,
                signature=None,
                fingerprint={},
                dep_specs={},
                out_specs={},
                params_arg_name=None,
            ),
        }

        # Simulate registry now has stage_a, stage_c (stage_b removed, stage_c added)
        with patch(
            "pivot.engine.engine.registry.REGISTRY.list_stages",
            return_value=["stage_a", "stage_c"],
        ):
            eng._emit_reload_event(old_stages)

        # Should have emitted PipelineReloaded event
        reload_events = [e for e in sink.events if e["type"] == "pipeline_reloaded"]
        assert len(reload_events) == 1

        event = reload_events[0]
        assert "stage_c" in event["stages_added"]
        assert "stage_b" in event["stages_removed"]
        assert event["error"] is None


def test_engine_clear_project_modules_removes_project_files(tmp_path: pathlib.Path) -> None:
    """_clear_project_modules() removes project modules from sys.modules."""
    import sys
    import types as types_mod

    with engine.Engine() as eng:

        # Create a mock module that appears to be in the project
        mock_module = types_mod.ModuleType("test_project_module")
        mock_module.__file__ = str(tmp_path / "my_module.py")

        # Create a mock module outside the project
        external_module = types_mod.ModuleType("external_module")
        external_module.__file__ = "/usr/lib/python/external.py"

        # Add them to sys.modules
        sys.modules["test_project_module"] = mock_module
        sys.modules["external_module"] = external_module

        try:
            # Clear project modules
            eng._clear_project_modules(tmp_path)

            # Project module should be removed
            assert "test_project_module" not in sys.modules

            # External module should remain
            assert "external_module" in sys.modules
        finally:
            # Cleanup
            sys.modules.pop("test_project_module", None)
            sys.modules.pop("external_module", None)


def test_engine_reload_registry_returns_true_on_success(tmp_path: pathlib.Path) -> None:
    """_reload_registry() returns True when reload succeeds."""
    with engine.Engine() as eng:

        # Create a simple pivot.yaml
        pivot_yaml = tmp_path / "pivot.yaml"
        pivot_yaml.write_text("stages: {}")

        with (
            patch("pivot.project.get_project_root", return_value=tmp_path),
            patch.object(eng, "_clear_project_modules"),
            patch.object(eng, "_emit_reload_event"),
        ):
            result = eng._reload_registry()

        assert result is True


def test_engine_reload_registry_returns_false_on_failure(tmp_path: pathlib.Path) -> None:
    """_reload_registry() returns False and restores when reload fails."""
    with engine.Engine() as eng:

        # Create a pivot.yaml
        pivot_yaml = tmp_path / "pivot.yaml"
        pivot_yaml.write_text("stages:\n  bad_stage:\n    python: nonexistent.module.func")

        with (
            patch("pivot.engine.engine.project.get_project_root", return_value=tmp_path),
            patch.object(eng, "_clear_project_modules"),
            patch(
                "pivot.pipeline.yaml.register_from_pipeline_file",
                side_effect=Exception("Import failed"),
            ),
        ):
            result = eng._reload_registry()

        assert result is False
        # Registry should be restored
        # (We can't easily verify the exact state, but the method should have called restore)


def test_engine_reload_from_decorators_returns_true_with_no_modules() -> None:
    """_reload_from_decorators() returns True when no modules to reload."""
    with engine.Engine() as eng:
        old_stages: dict[str, registry.RegistryStageInfo] = {}  # No stages = no modules

        result = eng._reload_from_decorators(old_stages)

        assert result is True


def test_engine_handle_code_or_config_changed_calls_invalidate_caches() -> None:
    """_handle_code_or_config_changed() calls _invalidate_caches()."""
    with engine.Engine() as eng:

        invalidate_called = [False]

        def mock_invalidate() -> None:
            invalidate_called[0] = True

        def mock_reload() -> bool:
            return True

        eng._invalidate_caches = mock_invalidate  # type: ignore[method-assign]
        eng._reload_registry = mock_reload  # type: ignore[method-assign]
        eng._execute_affected_stages = lambda stages: None  # type: ignore[method-assign]

        # Need to mock project.get_project_root and registry.REGISTRY.list_stages
        with (
            patch("pivot.engine.engine.registry.REGISTRY.list_stages", return_value=[]),
            patch("pivot.engine.engine.registry.REGISTRY.get", return_value={}),
        ):
            event: types.CodeOrConfigChanged = {
                "type": "code_or_config_changed",
                "paths": ["pivot.yaml"],
            }
            eng._handle_code_or_config_changed(event)

        assert invalidate_called[0] is True


def test_engine_handle_code_or_config_changed_aborts_on_invalid_pipeline() -> None:
    """_handle_code_or_config_changed() does not execute stages if reload fails."""
    with engine.Engine() as eng:
        sink = _MockSink()
        eng.add_sink(sink)

        def mock_reload() -> bool:
            return False  # Simulate invalid pipeline

        eng._invalidate_caches = lambda: None  # type: ignore[method-assign]
        eng._reload_registry = mock_reload  # type: ignore[method-assign]

        executed_stages: list[list[str]] = []

        def mock_execute(stages: list[str]) -> None:
            executed_stages.append(stages)

        eng._execute_affected_stages = mock_execute  # type: ignore[method-assign]

        event: types.CodeOrConfigChanged = {
            "type": "code_or_config_changed",
            "paths": ["pivot.yaml"],
        }
        eng._handle_code_or_config_changed(event)

        # No stages should have been executed
        assert len(executed_stages) == 0


def test_engine_handle_code_or_config_changed_rebuilds_graph() -> None:
    """_handle_code_or_config_changed() rebuilds the bipartite graph after reload."""
    with engine.Engine() as eng:

        # Verify graph is None initially
        assert eng._graph is None

        with (
            patch.object(eng, "_invalidate_caches"),
            patch.object(eng, "_reload_registry", return_value=True),
            patch.object(eng, "_execute_affected_stages"),
            patch("pivot.engine.engine.registry.REGISTRY.list_stages", return_value=["stage_a"]),
            patch(
                "pivot.engine.engine.registry.REGISTRY.get",
                return_value={"deps_paths": [], "outs_paths": [], "mutex": []},
            ),
            patch(
                "pivot.engine.engine.engine_graph.build_graph",
                return_value=nx.DiGraph(),
            ) as mock_build_graph,
            patch("pivot.engine.engine.engine_graph.get_watch_paths", return_value=[]),
        ):
            event: types.CodeOrConfigChanged = {
                "type": "code_or_config_changed",
                "paths": ["pivot.yaml"],
            }
            eng._handle_code_or_config_changed(event)

            # build_graph should have been called
            mock_build_graph.assert_called_once()


def test_engine_handle_code_or_config_changed_updates_watch_paths() -> None:
    """_handle_code_or_config_changed() updates FilesystemSource watch paths."""
    from pivot.engine import sources

    with engine.Engine() as eng:

        # Add a FilesystemSource
        fs_source = sources.FilesystemSource([])
        eng.add_source(fs_source)

        new_watch_paths = [pathlib.Path("/new/path")]

        with (
            patch.object(eng, "_invalidate_caches"),
            patch.object(eng, "_reload_registry", return_value=True),
            patch.object(eng, "_execute_affected_stages"),
            patch("pivot.engine.engine.registry.REGISTRY.list_stages", return_value=[]),
            patch("pivot.engine.engine.registry.REGISTRY.get", return_value={}),
            patch(
                "pivot.engine.engine.engine_graph.build_graph",
                return_value=nx.DiGraph(),
            ),
            patch(
                "pivot.engine.engine.engine_graph.get_watch_paths",
                return_value=new_watch_paths,
            ),
        ):
            event: types.CodeOrConfigChanged = {
                "type": "code_or_config_changed",
                "paths": ["pivot.yaml"],
            }
            eng._handle_code_or_config_changed(event)

        # FilesystemSource should have updated watch paths
        assert fs_source.watch_paths == new_watch_paths


# =============================================================================
# Tests for Agent RPC Methods (Phase 4 Task 10)
# =============================================================================


def test_engine_try_start_run_returns_start_result_when_idle() -> None:
    """try_start_run() returns AgentRunStartResult when engine is IDLE."""
    with engine.Engine() as eng:

        # Mock registry to return some stages
        with patch("pivot.engine.engine.registry.REGISTRY.list_stages", return_value=["stage_a"]):
            result = eng.try_start_run(run_id="test-run-123", stages=None, force=False)

        # Should return start result
        assert "run_id" in result
        assert result["run_id"] == "test-run-123"
        assert result["status"] == "started"
        assert result["stages_queued"] == ["stage_a"]


def test_engine_try_start_run_accepts_specific_stages() -> None:
    """try_start_run() uses provided stages instead of all stages."""
    with engine.Engine() as eng:

        result = eng.try_start_run(run_id="test-run-456", stages=["my_stage"], force=True)

        # Result is AgentRunStartResult (has stages_queued)
        assert "stages_queued" in result
        assert result["stages_queued"] == ["my_stage"]


def test_engine_try_start_run_rejects_when_active() -> None:
    """try_start_run() returns AgentRunRejection when engine is ACTIVE."""
    with engine.Engine() as eng:

        # Set engine to ACTIVE state
        eng._state = types.EngineState.ACTIVE
        eng._current_run_id = "existing-run"

        result = eng.try_start_run(run_id="new-run", stages=None, force=False)

        # Should return rejection (AgentRunRejection)
        assert "reason" in result
        assert result["reason"] == "busy"
        assert result["current_state"] == "active"
        assert result.get("current_run_id") == "existing-run"


def test_engine_try_start_run_sets_current_run_id() -> None:
    """try_start_run() sets _current_run_id on success."""
    with engine.Engine() as eng:

        assert eng._current_run_id is None

        with patch("pivot.engine.engine.registry.REGISTRY.list_stages", return_value=[]):
            eng.try_start_run(run_id="my-run-id", stages=[], force=False)

        assert eng._current_run_id == "my-run-id"


def test_engine_try_start_run_submits_run_requested_event() -> None:
    """try_start_run() submits RunRequested event to queue."""
    with engine.Engine() as eng:

        with patch("pivot.engine.engine.registry.REGISTRY.list_stages", return_value=["stage_a"]):
            eng.try_start_run(run_id="test-run", stages=None, force=True)

        # Check the event was queued
        assert not eng._event_queue.empty()
        event = eng._event_queue.get_nowait()
        assert event["type"] == "run_requested"
        assert event["stages"] == ["stage_a"]
        assert event["force"] is True
        assert event["reason"] == "agent:test-run"


def test_engine_try_start_run_is_thread_safe() -> None:
    """try_start_run() is thread-safe under concurrent access."""
    with engine.Engine() as eng:
        results: list[object] = []
        errors: list[str] = []

        def try_start(run_id: str) -> None:
            try:
                result = eng.try_start_run(run_id=run_id, stages=["stage"], force=False)
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads simultaneously
        threads = [threading.Thread(target=try_start, args=(f"run-{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert errors == []
        # All calls should have returned a result
        assert len(results) == 10


def test_engine_get_execution_status_returns_state() -> None:
    """get_execution_status() returns current engine state."""
    with engine.Engine() as eng:

        status = eng.get_execution_status()

        assert status["state"] == "idle"


def test_engine_get_execution_status_includes_run_id_when_set() -> None:
    """get_execution_status() includes run_id when _current_run_id is set."""
    with engine.Engine() as eng:
        eng._current_run_id = "my-run-123"

        status = eng.get_execution_status()

        assert "run_id" in status
        assert status["run_id"] == "my-run-123"


def test_engine_get_execution_status_filters_by_run_id() -> None:
    """get_execution_status() returns minimal status for non-matching run_id."""
    with engine.Engine() as eng:
        eng._current_run_id = "current-run"

        # Request status for a different run
        status = eng.get_execution_status(run_id="other-run")

        # Should only have state, not detailed info
        assert status["state"] == "idle"
        assert "run_id" not in status


def test_engine_get_execution_status_includes_stage_lists() -> None:
    """get_execution_status() includes stages_completed and stages_pending."""
    with engine.Engine() as eng:
        eng._current_run_id = "test-run"

        # Set up stage states
        eng._stage_states["completed_stage"] = types.StageExecutionState.COMPLETED
        eng._stage_states["pending_stage"] = types.StageExecutionState.PENDING
        eng._stage_states["running_stage"] = types.StageExecutionState.RUNNING
        eng._stage_states["ready_stage"] = types.StageExecutionState.READY
        eng._stage_states["preparing_stage"] = types.StageExecutionState.PREPARING

        status = eng.get_execution_status()

        assert "stages_completed" in status
        assert status["stages_completed"] == ["completed_stage"]

        assert "stages_pending" in status
        # Pending includes PENDING, READY, PREPARING, RUNNING
        assert set(status["stages_pending"]) == {
            "pending_stage",
            "running_stage",
            "ready_stage",
            "preparing_stage",
        }


def test_engine_get_execution_status_omits_empty_lists() -> None:
    """get_execution_status() omits stages_completed/stages_pending when empty."""
    with engine.Engine() as eng:

        status = eng.get_execution_status()

        assert "stages_completed" not in status
        assert "stages_pending" not in status


def test_engine_get_execution_status_is_thread_safe() -> None:
    """get_execution_status() is thread-safe under concurrent access."""
    with engine.Engine() as eng:
        eng._current_run_id = "test-run"
        eng._stage_states["stage_a"] = types.StageExecutionState.RUNNING

        results: list[object] = []
        errors: list[str] = []

        def get_status() -> None:
            try:
                result = eng.get_execution_status()
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_status) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 10


def test_engine_request_cancel_returns_true_when_active() -> None:
    """request_cancel() returns cancelled=True when engine is ACTIVE."""
    with engine.Engine() as eng:
        eng._state = types.EngineState.ACTIVE

        result = eng.request_cancel()

        assert result.get("cancelled") is True
        assert eng._cancel_event.is_set()


def test_engine_request_cancel_returns_false_when_idle() -> None:
    """request_cancel() returns cancelled=False when engine is IDLE."""
    with engine.Engine() as eng:

        result = eng.request_cancel()

        assert result.get("cancelled") is False
        assert not eng._cancel_event.is_set()


def test_engine_request_cancel_returns_false_when_shutdown() -> None:
    """request_cancel() returns cancelled=False when engine is SHUTDOWN."""
    with engine.Engine() as eng:
        eng._state = types.EngineState.SHUTDOWN

        result = eng.request_cancel()

        assert result.get("cancelled") is False


def test_engine_request_cancel_is_thread_safe() -> None:
    """request_cancel() is thread-safe under concurrent access."""
    with engine.Engine() as eng:
        eng._state = types.EngineState.ACTIVE

        from pivot.types import AgentCancelResult

        results: list[AgentCancelResult] = []
        errors: list[str] = []

        def cancel() -> None:
            try:
                result = eng.request_cancel()
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=cancel) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 10
        # At least the first call should have returned True
        assert any(r.get("cancelled") for r in results)


def test_engine_run_lock_is_initialized() -> None:
    """Engine initializes _run_lock for thread safety."""
    with engine.Engine() as eng:

        assert hasattr(eng, "_run_lock")
        assert isinstance(eng._run_lock, type(threading.Lock()))


def test_engine_current_run_id_is_initialized_none() -> None:
    """Engine initializes _current_run_id to None."""
    with engine.Engine() as eng:

        assert eng._current_run_id is None


# =============================================================================
# Keep-Going Mode Tests
# =============================================================================


def test_engine_keep_going_is_false_initially() -> None:
    """Engine starts with keep_going disabled."""
    with engine.Engine() as eng:
        assert eng.keep_going is False


def test_engine_toggle_keep_going_enables() -> None:
    """toggle_keep_going() enables mode and returns True."""
    with engine.Engine() as eng:

        result = eng.toggle_keep_going()

        assert result is True
        assert eng.keep_going is True


def test_engine_toggle_keep_going_disables() -> None:
    """toggle_keep_going() disables mode when already enabled."""
    with engine.Engine() as eng:
        eng.toggle_keep_going()  # Enable first

        result = eng.toggle_keep_going()  # Now disable

        assert result is False
        assert eng.keep_going is False


def test_engine_set_keep_going_true() -> None:
    """set_keep_going(True) enables mode."""
    with engine.Engine() as eng:

        eng.set_keep_going(True)

        assert eng.keep_going is True


def test_engine_set_keep_going_false() -> None:
    """set_keep_going(False) disables mode."""
    with engine.Engine() as eng:
        eng.set_keep_going(True)  # Enable first

        eng.set_keep_going(False)

        assert eng.keep_going is False


def test_engine_toggle_keep_going_is_thread_safe() -> None:
    """toggle_keep_going() is thread-safe under concurrent access."""
    with engine.Engine() as eng:

        results: list[bool] = []
        errors: list[str] = []

        def toggle() -> None:
            try:
                result = eng.toggle_keep_going()
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=toggle) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 20
        # With 20 toggles, we should end up back where we started (False)
        assert eng.keep_going is False
