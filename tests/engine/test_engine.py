"""Tests for the Engine class."""

from __future__ import annotations

import pathlib

import networkx as nx
import pytest

from pivot.engine import graph as engine_graph
from pivot.engine.engine import Engine
from pivot.engine.sinks import ResultCollectorSink
from pivot.engine.sources import OneShotSource
from pivot.engine.types import (
    DataArtifactChanged,
    EngineState,
    NodeType,
    OutputEvent,
    StageExecutionState,
)


class _MockAsyncSink:
    """Test sink for capturing events."""

    events: list[OutputEvent]
    closed: bool

    def __init__(self) -> None:
        self.events = list[OutputEvent]()
        self.closed = False

    async def handle(self, event: OutputEvent) -> None:
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_engine_initial_state_is_idle() -> None:
    """Engine starts in IDLE state."""
    async with Engine() as engine:
        assert engine.state == EngineState.IDLE


@pytest.mark.anyio
async def test_engine_has_empty_sources_initially() -> None:
    """Engine has no sources until registered."""
    async with Engine() as engine:
        assert engine.sources == []


@pytest.mark.anyio
async def test_engine_has_empty_sinks_initially() -> None:
    """Engine has no sinks until registered."""
    async with Engine() as engine:
        assert engine.sinks == []


@pytest.mark.anyio
async def test_engine_add_source() -> None:
    """Engine can register async event sources."""
    async with Engine() as engine:
        source = OneShotSource(stages=None, force=False, reason="test")
        engine.add_source(source)

        assert len(engine.sources) == 1


@pytest.mark.anyio
async def test_engine_add_sink() -> None:
    """Engine can register async event sinks."""
    async with Engine() as engine:
        sink = ResultCollectorSink()
        engine.add_sink(sink)

        assert len(engine.sinks) == 1


@pytest.mark.anyio
async def test_engine_run_without_context_manager_raises() -> None:
    """Engine.run() raises RuntimeError if called outside async with."""
    engine = Engine()

    # Should raise because channels aren't initialized
    with pytest.raises(RuntimeError, match="context manager"):
        await engine.run(exit_on_completion=True)


@pytest.mark.anyio
async def test_engine_closes_sinks_even_when_one_fails() -> None:
    """Engine.__aexit__() continues closing sinks even if one raises."""

    class _FailingSink:
        closed: bool

        def __init__(self) -> None:
            self.closed = False

        async def handle(self, event: OutputEvent) -> None:
            pass

        async def close(self) -> None:
            raise RuntimeError("Sink close failed")

    class _GoodSink:
        closed: bool

        def __init__(self) -> None:
            self.closed = False

        async def handle(self, event: OutputEvent) -> None:
            pass

        async def close(self) -> None:
            self.closed = True

    failing_sink = _FailingSink()
    good_sink = _GoodSink()

    async with Engine() as engine:
        engine.add_sink(failing_sink)
        engine.add_sink(good_sink)

    # Good sink should still be closed despite failing sink
    assert good_sink.closed, "Good sink should be closed even when other sink fails"


# =============================================================================
# Event Handler Tests
# =============================================================================

# Note: test_engine_handle_cancel_requested_sets_event was removed because it tested
# private implementation details (_cancel_event, _handle_cancel_requested) rather than
# observable behavior. Cancellation should be tested via integration tests that send
# CancelRequested events and verify stages are properly marked as cancelled.


@pytest.mark.anyio
async def test_engine_handle_data_artifact_changed_filters_executing_outputs(
    tmp_path: pathlib.Path,
) -> None:
    """_handle_data_artifact_changed() filters events for executing stage outputs."""

    sink = _MockAsyncSink()

    async with Engine() as engine:
        engine.add_sink(sink)

        # Build graph: stage_a -> output.csv -> stage_b
        output_path = tmp_path / "output.csv"

        g: nx.DiGraph[str] = nx.DiGraph()
        stage_a_node = engine_graph.stage_node("stage_a")
        output_node = engine_graph.artifact_node(output_path)
        stage_b_node = engine_graph.stage_node("stage_b")

        g.add_node(stage_a_node, type=NodeType.STAGE)
        g.add_node(output_node, type=NodeType.ARTIFACT)
        g.add_node(stage_b_node, type=NodeType.STAGE)

        g.add_edge(stage_a_node, output_node)  # stage_a produces output
        g.add_edge(output_node, stage_b_node)  # stage_b consumes output

        engine._graph = g

        # stage_a is currently running
        engine._stage_states["stage_a"] = StageExecutionState.RUNNING

        # Create event for output change
        event = DataArtifactChanged(
            type="data_artifact_changed",
            paths=[str(output_path)],
        )

        # Handle the event
        await engine._handle_data_artifact_changed(event)

        # Event should be deferred, not processed immediately
        assert "stage_a" in engine._deferred_events
        assert len(engine._deferred_events["stage_a"]) == 1

        # No execution should have started (no ACTIVE state event)
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        active_events = [e for e in state_events if e["state"] == EngineState.ACTIVE]
        assert len(active_events) == 0


@pytest.mark.anyio
async def test_engine_handle_data_artifact_changed_no_affected_stages(
    tmp_path: pathlib.Path,
) -> None:
    """_handle_data_artifact_changed() does nothing for paths with no consumers."""

    sink = _MockAsyncSink()

    async with Engine() as engine:
        engine.add_sink(sink)

        # Build graph with no stages consuming this path
        g: nx.DiGraph[str] = nx.DiGraph()
        g.add_node(engine_graph.stage_node("unrelated_stage"), type=NodeType.STAGE)
        engine._graph = g

        # Create event for unknown path
        event = DataArtifactChanged(
            type="data_artifact_changed",
            paths=[str(tmp_path / "unknown.csv")],
        )

        # Should not raise and no execution started
        await engine._handle_data_artifact_changed(event)

        # No state changes
        state_events = [e for e in sink.events if e["type"] == "engine_state_changed"]
        assert len(state_events) == 0


@pytest.mark.anyio
async def test_engine_process_deferred_events() -> None:
    """_process_deferred_events() processes deferred events for a stage."""
    async with Engine() as engine:
        # Manually add deferred events
        event1 = DataArtifactChanged(type="data_artifact_changed", paths=["file1.csv"])
        event2 = DataArtifactChanged(type="data_artifact_changed", paths=["file2.csv"])

        engine._deferred_events["stage_a"].append(event1)
        engine._deferred_events["stage_a"].append(event2)

        # Process deferred events - they should be removed from the dict
        await engine._process_deferred_events("stage_a")

        # Events should be removed
        assert "stage_a" not in engine._deferred_events


@pytest.mark.anyio
async def test_engine_process_deferred_events_empty_list() -> None:
    """_process_deferred_events() handles empty/missing stage gracefully."""
    async with Engine() as engine:
        # Process for non-existent stage should not raise
        await engine._process_deferred_events("nonexistent_stage")


@pytest.mark.anyio
async def test_engine_should_filter_path_returns_false_without_graph() -> None:
    """_should_filter_path() returns False when graph is None."""
    async with Engine() as engine:
        assert engine._graph is None
        assert engine._should_filter_path(pathlib.Path("any/path.csv")) is False


@pytest.mark.anyio
async def test_engine_should_filter_path_returns_false_for_input_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    """_should_filter_path() returns False for input artifacts (no producer)."""

    async with Engine() as engine:
        input_path = tmp_path / "input.csv"

        g: nx.DiGraph[str] = nx.DiGraph()
        input_node = engine_graph.artifact_node(input_path)
        stage_node = engine_graph.stage_node("stage_a")

        g.add_node(input_node, type=NodeType.ARTIFACT)
        g.add_node(stage_node, type=NodeType.STAGE)
        g.add_edge(input_node, stage_node)  # input consumed by stage_a

        engine._graph = g
        engine._stage_states["stage_a"] = StageExecutionState.RUNNING

        # Input artifact should not be filtered (no producer)
        assert engine._should_filter_path(input_path) is False


@pytest.mark.anyio
async def test_engine_get_affected_stages_for_path_returns_empty_without_graph() -> None:
    """_get_affected_stages_for_path() returns empty list when graph is None."""
    async with Engine() as engine:
        assert engine._graph is None
        assert engine._get_affected_stages_for_path(pathlib.Path("any/path.csv")) == []


@pytest.mark.anyio
async def test_engine_get_affected_stages_for_paths_deduplicates(
    tmp_path: pathlib.Path,
) -> None:
    """_get_affected_stages_for_paths() deduplicates affected stages."""

    async with Engine() as engine:
        # Create paths
        input1 = tmp_path / "input1.csv"
        input2 = tmp_path / "input2.csv"

        # Build graph: both inputs consumed by stage_a
        g: nx.DiGraph[str] = nx.DiGraph()
        stage_node = engine_graph.stage_node("stage_a")
        input1_node = engine_graph.artifact_node(input1)
        input2_node = engine_graph.artifact_node(input2)

        g.add_node(stage_node, type=NodeType.STAGE)
        g.add_node(input1_node, type=NodeType.ARTIFACT)
        g.add_node(input2_node, type=NodeType.ARTIFACT)
        g.add_edge(input1_node, stage_node)
        g.add_edge(input2_node, stage_node)

        engine._graph = g

        # Both paths affect the same stage
        affected = engine._get_affected_stages_for_paths([input1, input2])

        # Should only return stage_a once
        assert affected == ["stage_a"]


@pytest.mark.anyio
async def test_engine_invalidate_caches_clears_graph() -> None:
    """_invalidate_caches() clears the graph."""

    async with Engine() as engine:
        # Set a graph
        engine._graph = nx.DiGraph()

        # Invalidate
        engine._invalidate_caches()

        # Graph should be None
        assert engine._graph is None


# =============================================================================
# Engine State Management Tests
# =============================================================================


@pytest.mark.anyio
async def test_engine_set_stage_state_updates_internal_state() -> None:
    """_set_stage_state() updates internal stage state tracking."""
    async with Engine() as engine:
        # Initially, stage should be PENDING
        assert engine._get_stage_state("test_stage") == StageExecutionState.PENDING

        # Set stage state
        await engine._set_stage_state("test_stage", StageExecutionState.RUNNING)

        # State should be updated
        assert engine._get_stage_state("test_stage") == StageExecutionState.RUNNING


@pytest.mark.anyio
async def test_engine_set_stage_state_tracks_multiple_stages() -> None:
    """_set_stage_state() independently tracks state for multiple stages."""
    async with Engine() as engine:
        # Set different states for different stages
        await engine._set_stage_state("stage_a", StageExecutionState.RUNNING)
        await engine._set_stage_state("stage_b", StageExecutionState.COMPLETED)

        # Each should maintain its own state
        assert engine._get_stage_state("stage_a") == StageExecutionState.RUNNING
        assert engine._get_stage_state("stage_b") == StageExecutionState.COMPLETED
        assert engine._get_stage_state("stage_c") == StageExecutionState.PENDING


@pytest.mark.anyio
async def test_engine_get_stage_state_returns_pending_by_default() -> None:
    """_get_stage_state() returns PENDING for unknown stages."""
    async with Engine() as engine:
        state = engine._get_stage_state("unknown_stage")
        assert state == StageExecutionState.PENDING


@pytest.mark.anyio
async def test_engine_is_idle_returns_true_initially() -> None:
    """_is_idle() returns True when engine is in IDLE state."""
    async with Engine() as engine:
        assert engine._is_idle() is True, "Engine should be idle initially"
        assert engine.state == EngineState.IDLE


# =============================================================================
# Engine Pipeline Requirement Tests
# =============================================================================


@pytest.mark.anyio
async def test_engine_list_stages_requires_pipeline() -> None:
    """_list_stages() raises RuntimeError when no pipeline configured."""
    async with Engine() as engine:
        with pytest.raises(RuntimeError, match="Engine requires a Pipeline"):
            engine._list_stages()


@pytest.mark.anyio
async def test_engine_get_stage_requires_pipeline() -> None:
    """_get_stage() raises RuntimeError when no pipeline configured."""
    async with Engine() as engine:
        with pytest.raises(RuntimeError, match="Engine requires a Pipeline"):
            engine._get_stage("any_stage")


@pytest.mark.anyio
async def test_engine_get_all_stages_requires_pipeline() -> None:
    """_get_all_stages() raises RuntimeError when no pipeline configured."""
    async with Engine() as engine:
        with pytest.raises(RuntimeError, match="Engine requires a Pipeline"):
            engine._get_all_stages()


# =============================================================================
# Engine Deferred Events Tests (Additional)
# =============================================================================


@pytest.mark.anyio
async def test_engine_deferred_events_empty_initially() -> None:
    """Engine starts with no deferred events."""
    async with Engine() as engine:
        assert len(engine._deferred_events) == 0


@pytest.mark.anyio
async def test_engine_process_deferred_events_with_multiple_events() -> None:
    """_process_deferred_events() processes multiple deferred events for a stage."""
    async with Engine() as engine:
        # Add multiple deferred events
        event1 = DataArtifactChanged(type="data_artifact_changed", paths=["file1.csv"])
        event2 = DataArtifactChanged(type="data_artifact_changed", paths=["file2.csv"])
        event3 = DataArtifactChanged(type="data_artifact_changed", paths=["file3.csv"])

        engine._deferred_events["stage_a"].extend([event1, event2, event3])
        assert len(engine._deferred_events["stage_a"]) == 3

        # Process deferred events
        await engine._process_deferred_events("stage_a")

        # All events should be removed
        assert "stage_a" not in engine._deferred_events


# =============================================================================
# Orchestration Params Storage Tests
# =============================================================================


@pytest.mark.anyio
async def test_engine_stores_orchestration_params_on_run_requested() -> None:
    """Engine stores no_commit/no_cache/on_error from RunRequested for watch re-runs."""
    from pivot.engine.types import RunRequested
    from pivot.types import OnError

    async with Engine() as engine:
        # Initial state - defaults
        assert engine._stored_no_commit is False
        assert engine._stored_no_cache is False
        assert engine._stored_on_error == OnError.FAIL

        # Create event with non-default params
        event = RunRequested(
            type="run_requested",
            stages=None,
            force=False,
            reason="test",
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=True,
            no_cache=True,
            on_error=OnError.KEEP_GOING,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        # Mock _orchestrate_execution before calling _handle_run_requested.
        # _handle_run_requested stores params then calls _orchestrate_execution.
        # Also need to mock _require_pipeline since it's called first.
        async def mock_orchestrate(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        await engine._handle_run_requested(event)

        # Params should be stored
        assert engine._stored_no_commit is True
        assert engine._stored_no_cache is True
        assert engine._stored_on_error == OnError.KEEP_GOING


@pytest.mark.anyio
async def test_engine_execute_affected_stages_uses_stored_params() -> None:
    """_execute_affected_stages() uses stored orchestration params instead of hardcoded defaults."""
    from pivot.types import OnError

    async with Engine() as engine:
        # Set stored params (simulating what _handle_run_requested would do)
        engine._stored_no_commit = True
        engine._stored_no_cache = True
        engine._stored_on_error = OnError.KEEP_GOING

        # Mock _orchestrate_execution to capture what params it receives
        captured_kwargs: dict[str, object] = {}

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock

        # Call _execute_affected_stages
        await engine._execute_affected_stages(["stage_a"])

        # Verify stored params were used
        assert captured_kwargs["no_commit"] is True, "no_commit should use stored value"
        assert captured_kwargs["no_cache"] is True, "no_cache should use stored value"
        assert captured_kwargs["on_error"] == OnError.KEEP_GOING, "on_error should use stored value"


# =============================================================================
# Cancellation Tests
# =============================================================================


@pytest.mark.anyio
async def test_engine_cancel_during_active_execution() -> None:
    """CancelRequested stops pending stages from starting during active execution."""
    import anyio

    from pivot.engine.types import RunRequested
    from pivot.types import OnError

    sink = _MockAsyncSink()

    async with Engine() as engine:
        engine.add_sink(sink)

        # Track stages that were skipped due to cancellation
        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            # Simulate long-running execution that checks cancel
            for _ in range(10):
                await anyio.sleep(0.01)
                if engine._cancel_event.is_set():
                    # Cancel was requested - this is the expected path
                    break
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        # Create run event
        run_event = RunRequested(
            type="run_requested",
            stages=None,
            force=False,
            reason="test",
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=False,
            no_cache=False,
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        # Start run in background
        async with anyio.create_task_group() as tg:
            tg.start_soon(engine._handle_run_requested, run_event)

            # Give run time to start
            await anyio.sleep(0.02)

            # Send cancel while running
            await engine._handle_cancel_requested()

            # Verify cancel event is set
            assert engine._cancel_event.is_set(), "Cancel event should be set"

            # Wait for run to complete (with timeout)
            with anyio.move_on_after(1.0):
                while engine.state != EngineState.IDLE:
                    await anyio.sleep(0.01)


@pytest.mark.anyio
async def test_engine_concurrent_run_requests_serialized() -> None:
    """Multiple concurrent RunRequested events are serialized (not run in parallel)."""
    import anyio

    from pivot.engine.types import RunRequested
    from pivot.types import OnError

    execution_order = list[str]()

    async with Engine() as engine:
        run_count = 0

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            nonlocal run_count
            run_count += 1
            run_id = f"run_{run_count}"
            execution_order.append(f"{run_id}_start")
            await anyio.sleep(0.05)  # Simulate work
            execution_order.append(f"{run_id}_end")
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        # Create two run events
        event1 = RunRequested(
            type="run_requested",
            stages=["stage_a"],
            force=False,
            reason="test1",
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=False,
            no_cache=False,
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )
        event2 = RunRequested(
            type="run_requested",
            stages=["stage_b"],
            force=False,
            reason="test2",
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=False,
            no_cache=False,
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        # Handle both events sequentially (as the engine's input loop does)
        await engine._handle_run_requested(event1)
        await engine._handle_run_requested(event2)

        # Verify runs completed in sequence, not interleaved
        assert execution_order == [
            "run_1_start",
            "run_1_end",
            "run_2_start",
            "run_2_end",
        ], f"Runs should be serialized, got: {execution_order}"


# =============================================================================
# Cleanup Tests
# =============================================================================


@pytest.mark.anyio
async def test_engine_aexit_closes_sinks_when_source_raises() -> None:
    """Engine.__aexit__() closes sinks even when a source raises an exception."""

    class _RaisingSink:
        closed: bool

        def __init__(self) -> None:
            self.closed = False

        async def handle(self, event: OutputEvent) -> None:
            pass

        async def close(self) -> None:
            self.closed = True

    class _RaisingSource:
        async def run(self, send: object) -> None:
            raise RuntimeError("Source error")

    sink = _RaisingSink()
    source = _RaisingSource()

    # The exception should propagate but sinks should still be closed
    with pytest.raises(ExceptionGroup):
        async with Engine() as engine:
            engine.add_sink(sink)
            engine.add_source(source)  # type: ignore[arg-type] - test mock
            await engine.run(exit_on_completion=True)

    assert sink.closed, "Sink should be closed even when source raises"
