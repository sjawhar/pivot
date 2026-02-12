"""Tests for the Engine class."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict, cast
from unittest.mock import MagicMock

import anyio
import networkx as nx
import pytest

from pivot import loaders, outputs, parameters, project
from pivot.engine import engine as engine_mod
from pivot.engine import graph as engine_graph
from pivot.engine.engine import Engine
from pivot.engine.sinks import ResultCollectorSink
from pivot.engine.sources import OneShotSource
from pivot.engine.types import (
    CodeOrConfigChanged,
    DataArtifactChanged,
    EngineDiagnostic,
    EngineState,
    InputEvent,
    NodeType,
    OutputEvent,
    RunRequested,
    StageExecutionState,
)
from pivot.engine.worker_pool import WorkerPool
from pivot.storage import cache, lock
from pivot.storage import state as state_mod
from pivot.types import LockData, OnError
from tests.helpers import register_test_stage

if TYPE_CHECKING:
    from pivot.executor import core as executor_core
    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo


class _CoordinatorSkipOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_coordinator_skip_stage(
    _input_path: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _CoordinatorSkipOutputs:
    return {"output": pathlib.Path("output.txt")}


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


# Shared mutable state for deferred-event test helpers.
# Reset via the _reset_deferred_test_state fixture below.
_deferred_call_log: list[str] = []
_deferred_infinite_count: int = 0
_deferred_multi_count: int = 0


@pytest.fixture(autouse=False)
def _reset_deferred_test_state() -> None:
    """Reset shared state used by deferred-event test helpers."""
    global _deferred_infinite_count, _deferred_multi_count
    _deferred_call_log.clear()
    _deferred_infinite_count = 0
    _deferred_multi_count = 0


async def _helper_multi_level_defer(engine: Engine, event: InputEvent) -> None:
    """Module-level handler that defers for 3 levels then stops."""
    global _deferred_multi_count
    _deferred_multi_count += 1
    if _deferred_multi_count < 4:
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(
                type="data_artifact_changed",
                paths=[f"level{_deferred_multi_count}.csv"],
            ),
        )


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
        engine._scheduler.set_state("stage_a", StageExecutionState.RUNNING)

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
        engine._scheduler.set_state("stage_a", StageExecutionState.RUNNING)

        # Input artifact should not be filtered (no producer)
        assert engine._should_filter_path(input_path) is False


@pytest.mark.anyio
async def test_engine_get_affected_stages_for_paths_returns_empty_without_graph() -> None:
    """_get_affected_stages_for_paths() returns empty list when graph is None."""
    async with Engine() as engine:
        assert engine._graph is None
        assert engine._get_affected_stages_for_paths([pathlib.Path("any/path.csv")]) == []


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
    """Engine stores all orchestration params from RunRequested for watch re-runs."""

    async with Engine() as engine:
        # Initial state - defaults
        assert engine._stored_no_commit is False
        assert engine._stored_on_error == OnError.FAIL
        assert engine._stored_parallel is True
        assert engine._stored_max_workers is None

        # Create event with non-default params
        event = RunRequested(
            type="run_requested",
            stages=None,
            force=False,
            reason="test",
            single_stage=False,
            parallel=False,
            max_workers=4,
            no_commit=True,
            on_error=OnError.KEEP_GOING,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        # Mock _orchestrate_execution before calling _handle_run_requested.
        # _handle_run_requested stores params then spawns _run_execution_task.
        # Also need to mock _require_pipeline since it's called first.
        async def mock_orchestrate(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        # Provide a task group so _handle_run_requested can spawn the execution task
        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._handle_run_requested(event)

        # Params should be stored
        assert engine._stored_no_commit is True
        assert engine._stored_on_error == OnError.KEEP_GOING
        assert engine._stored_parallel is False
        assert engine._stored_max_workers == 4


@pytest.mark.anyio
async def test_engine_execute_affected_stages_uses_stored_params() -> None:
    """_execute_affected_stages() uses stored orchestration params instead of hardcoded defaults."""

    async with Engine() as engine:
        # Set stored params (simulating what _handle_run_requested would do)
        engine._stored_no_commit = True
        engine._stored_on_error = OnError.KEEP_GOING
        engine._stored_parallel = True
        engine._stored_max_workers = 4

        # Mock _orchestrate_execution to capture what params it receives
        captured_kwargs: dict[str, object] = {}

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        # Provide task group for non-blocking run handling
        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._execute_affected_stages(["stage_a"])

        # Verify stored params were used
        assert captured_kwargs["no_commit"] is True, "no_commit should use stored value"
        assert captured_kwargs["on_error"] == OnError.KEEP_GOING, "on_error should use stored value"
        assert captured_kwargs["parallel"] is True, "parallel should use stored value"
        assert captured_kwargs["max_workers"] == 4, "max_workers should use stored value"


@pytest.mark.anyio
async def test_engine_execute_affected_stages_propagates_non_parallel_mode() -> None:
    """_execute_affected_stages() correctly propagates parallel=False to orchestration."""

    async with Engine() as engine:
        engine._stored_parallel = False
        engine._stored_max_workers = None
        engine._stored_no_commit = False
        engine._stored_on_error = OnError.FAIL

        captured_kwargs: dict[str, object] = {}

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        # Provide task group for non-blocking run handling
        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._execute_affected_stages(["stage_a"])

        assert captured_kwargs["parallel"] is False, "parallel=False should be propagated"
        assert captured_kwargs["max_workers"] is None, "max_workers=None should be propagated"


# =============================================================================
# Cancellation Tests
# =============================================================================


@pytest.mark.anyio
async def test_engine_cancel_during_active_execution() -> None:
    """CancelRequested stops pending stages from starting during active execution."""

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
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        # Provide a task group for spawning execution tasks
        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._handle_run_requested(run_event)

            # Give run time to start (execution task runs in background)
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
async def test_engine_cancel_requested_noop_when_idle() -> None:
    async with Engine() as engine:
        engine._run_state = engine_mod._RunState.IDLE
        engine._cancel_event = anyio.Event()

        await engine._handle_cancel_requested()

        assert engine._run_state == engine_mod._RunState.IDLE
        assert not engine._cancel_event.is_set(), "Cancel should not be set while idle"


@pytest.mark.anyio
async def test_engine_cancel_requested_transitions_to_cancelling() -> None:
    async with Engine() as engine:
        engine._run_state = engine_mod._RunState.RUNNING
        engine._cancel_event = anyio.Event()
        pool = WorkerPool()
        engine._worker_pool = pool

        await engine._handle_cancel_requested()

        assert engine._run_state == engine_mod._RunState.CANCELLING
        assert engine._cancel_event.is_set(), "Cancel should be set for running engine"
        assert pool._accepting is False


@pytest.mark.anyio
async def test_engine_cancel_requested_noop_when_cancelling() -> None:
    async with Engine() as engine:
        engine._run_state = engine_mod._RunState.CANCELLING
        engine._cancel_event = anyio.Event()

        await engine._handle_cancel_requested()

        assert engine._run_state == engine_mod._RunState.CANCELLING
        assert not engine._cancel_event.is_set(), "Cancel should not be re-set"


@pytest.mark.anyio
async def test_engine_concurrent_run_requests_coalesced() -> None:
    """A second RunRequested while running cancels current and restarts with latest event.

    With non-blocking run handling, a second request while running sets
    _restart_pending and cancels the current run. When the current run
    completes, _run_execution_task replays the pending restart.
    """
    execution_order = list[str]()

    async with Engine() as engine:
        run_count = 0

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            nonlocal run_count
            run_count += 1
            run_id = f"run_{run_count}"
            execution_order.append(f"{run_id}_start")
            # Simulate work, check cancel
            for _ in range(10):
                await anyio.sleep(0.01)
                if engine._cancel_event.is_set():
                    execution_order.append(f"{run_id}_cancelled")
                    return {}
            execution_order.append(f"{run_id}_end")
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        event1 = RunRequested(
            type="run_requested",
            stages=["stage_a"],
            force=False,
            reason="test1",
            single_stage=False,
            parallel=True,
            max_workers=None,
            no_commit=False,
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
            on_error=OnError.FAIL,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )

        # Handle both events as the input loop would
        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg

            # First run starts immediately
            await engine._handle_run_requested(event1)
            # Give it time to start
            await anyio.sleep(0.02)

            # Second request while first is running → cancels first, queues restart
            await engine._handle_run_requested(event2)

        # First run was cancelled, second run completed
        assert "run_1_start" in execution_order, f"First run should have started: {execution_order}"
        assert "run_2_start" in execution_order, (
            f"Second run should have started: {execution_order}"
        )
        assert "run_2_end" in execution_order, (
            f"Second run should have completed: {execution_order}"
        )


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


# =============================================================================
# Worker Restart on Code/Config Change Tests
# =============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("parallel", "max_workers", "expected_restart_calls"),
    [
        pytest.param(True, 4, [(3, 4)], id="parallel-restarts-workers"),
        pytest.param(False, None, [], id="non-parallel-skips-restart"),
    ],
)
async def test_engine_handle_code_or_config_changed_restart_behavior(
    monkeypatch: pytest.MonkeyPatch,
    parallel: bool,
    max_workers: int | None,
    expected_restart_calls: list[tuple[int, int | None]],
) -> None:
    """_handle_code_or_config_changed restarts workers only in parallel mode."""

    restart_calls = list[tuple[int, int | None]]()

    def mock_restart_workers(stage_count: int, max_workers: int | None = None) -> int:
        restart_calls.append((stage_count, max_workers))
        return stage_count

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        engine._stored_parallel = parallel
        engine._stored_max_workers = max_workers

        mock_pipeline = MagicMock()
        mock_pipeline.list_stages.return_value = ["a", "b", "c"]
        mock_pipeline.snapshot.return_value = {}
        mock_pipeline._registry = None
        mock_pipeline.get.return_value = MagicMock()
        mock_pipeline.resolve_external_dependencies.return_value = None
        mock_pipeline.invalidate_dag_cache.return_value = None
        engine._pipeline = mock_pipeline

        engine._reload_registry = lambda: ({}, None)  # type: ignore[assignment] - test mock

        async def mock_emit_reload(*_args: object) -> None:
            pass

        engine._emit_reload_event = mock_emit_reload  # pyright: ignore[reportAttributeAccessIssue]

        async def mock_execute_affected(stages: list[str]) -> None:
            pass

        engine._execute_affected_stages = mock_execute_affected  # type: ignore[assignment] - test mock

        def mock_build_graph(*_a: object) -> MagicMock:
            return MagicMock()

        def mock_get_watch_paths(*_a: object) -> set[str]:
            return set[str]()

        monkeypatch.setattr("pivot.engine.engine.engine_graph.build_graph", mock_build_graph)
        monkeypatch.setattr(
            "pivot.engine.engine.engine_graph.get_watch_paths", mock_get_watch_paths
        )

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert restart_calls == expected_restart_calls


@pytest.mark.anyio
async def test_engine_handle_code_or_config_changed_no_restart_on_failed_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_code_or_config_changed does NOT restart workers when reload fails."""

    restart_calls = list[object]()

    def mock_restart_workers(*args: object, **kwargs: object) -> int:
        restart_calls.append(args)
        return 1

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        engine._stored_parallel = True

        # Mock reload to FAIL
        engine._reload_registry = lambda: None  # type: ignore[assignment] - test mock
        engine._invalidate_caches = lambda: None  # type: ignore[assignment] - test mock

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert len(restart_calls) == 0, "Should not restart workers on failed reload"


@pytest.mark.anyio
async def test_engine_handle_code_or_config_changed_continues_on_restart_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_code_or_config_changed continues if restart_workers raises."""

    def mock_restart_workers(stage_count: int, max_workers: int | None = None) -> int:
        raise RuntimeError("loky pool broken")

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    executed = list[list[str]]()

    async with Engine() as engine:
        engine._stored_parallel = True
        engine._stored_max_workers = 2

        mock_pipeline = MagicMock()
        mock_pipeline.list_stages.return_value = ["a", "b"]
        mock_pipeline.snapshot.return_value = {}
        mock_pipeline._registry = None
        mock_pipeline.resolve_external_dependencies.return_value = None
        mock_pipeline.invalidate_dag_cache.return_value = None
        engine._pipeline = mock_pipeline

        engine._reload_registry = lambda: ({}, None)  # type: ignore[assignment] - test mock

        async def mock_emit_reload(*_args: object) -> None:
            pass

        engine._emit_reload_event = mock_emit_reload  # pyright: ignore[reportAttributeAccessIssue]

        async def mock_execute_affected(stages: list[str]) -> None:
            executed.append(stages)

        engine._execute_affected_stages = mock_execute_affected  # type: ignore[assignment] - test mock

        def mock_build_graph(*_a: object) -> MagicMock:
            return MagicMock()

        def mock_get_watch_paths(*_a: object) -> set[str]:
            return set[str]()

        monkeypatch.setattr("pivot.engine.engine.engine_graph.build_graph", mock_build_graph)
        monkeypatch.setattr(
            "pivot.engine.engine.engine_graph.get_watch_paths", mock_get_watch_paths
        )

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert len(executed) == 1, "Execution should proceed despite restart failure"
        assert executed[0] == ["a", "b"]


@pytest.mark.anyio
async def test_engine_handle_data_artifact_changed_does_not_restart_workers(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_data_artifact_changed does NOT restart worker pool (data-only change)."""
    restart_calls = list[object]()

    def mock_restart_workers(*args: object, **kwargs: object) -> int:
        restart_calls.append(args)
        return 1

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        # Build minimal graph: input.csv -> stage_a
        input_path = tmp_path / "input.csv"

        g: nx.DiGraph[str] = nx.DiGraph()
        input_node = engine_graph.artifact_node(input_path)
        stage_node_name = engine_graph.stage_node("stage_a")
        g.add_node(input_node, type=NodeType.ARTIFACT)
        g.add_node(stage_node_name, type=NodeType.STAGE)
        g.add_edge(input_node, stage_node_name)
        engine._graph = g

        # Mock _execute_affected_stages to avoid actual execution
        executed_stages = list[list[str]]()

        async def mock_execute(stages: list[str]) -> None:
            executed_stages.append(stages)

        engine._execute_affected_stages = mock_execute  # type: ignore[assignment] - test mock

        event = DataArtifactChanged(type="data_artifact_changed", paths=[str(input_path)])
        await engine._handle_data_artifact_changed(event)

        assert len(restart_calls) == 0, "Data changes should NOT restart workers"


async def _helper_tracking_handle_with_redeferral(engine: Engine, event: InputEvent) -> None:
    """Module-level handler that defers one additional event on first call."""
    _deferred_call_log.append(event["type"])
    if len(_deferred_call_log) == 1:
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["nested.csv"]),
        )


@pytest.mark.anyio
@pytest.mark.usefixtures("_reset_deferred_test_state")
async def test_deferred_events_during_processing_are_not_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If handling a deferred event defers another for the same stage, both are processed."""
    async with Engine() as engine:
        monkeypatch.setattr(
            engine,
            "_handle_input_event",
            lambda event: _helper_tracking_handle_with_redeferral(engine, event),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType] - monkeypatch lambda
        )

        # Seed one deferred event
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["initial.csv"]),
        )

        await engine._process_deferred_events("stage_a")

        assert len(_deferred_call_log) == 2, (
            f"Expected 2 events processed, got {len(_deferred_call_log)}"
        )
        assert "stage_a" not in engine._deferred_events, "No leftover deferred events"


async def _helper_infinite_defer_handle(engine: Engine, event: InputEvent) -> None:
    """Module-level handler that always defers another event — simulates infinite loop."""
    global _deferred_infinite_count
    _deferred_infinite_count += 1
    engine._defer_event_for_stage(
        "stage_a",
        DataArtifactChanged(type="data_artifact_changed", paths=["loop.csv"]),
    )


@pytest.mark.anyio
@pytest.mark.usefixtures("_reset_deferred_test_state")
async def test_deferred_events_max_iterations_guard_emits_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infinite deferral loop is caught by guard; diagnostic event emitted, events dropped."""
    emitted_events: list[OutputEvent] = []

    async def _capture_emit(event: OutputEvent) -> None:
        emitted_events.append(event)

    async with Engine() as engine:
        monkeypatch.setattr(engine, "emit", _capture_emit)

        monkeypatch.setattr(
            engine,
            "_handle_input_event",
            lambda event: _helper_infinite_defer_handle(engine, event),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType] - monkeypatch lambda
        )

        # Seed one event
        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["seed.csv"]),
        )

        # Should not hang — guard trips
        await engine._process_deferred_events("stage_a")

        # Guard stopped iteration at max
        assert _deferred_infinite_count == engine._DEFERRED_MAX_ITERATIONS, (
            f"Expected {engine._DEFERRED_MAX_ITERATIONS} iterations, got {_deferred_infinite_count}"
        )

        # Diagnostic event emitted
        diagnostics = [e for e in emitted_events if e["type"] == "engine_diagnostic"]
        assert len(diagnostics) == 1, f"Expected 1 diagnostic event, got {len(diagnostics)}"
        diag: EngineDiagnostic = diagnostics[0]
        assert "stage_a" in diag["message"], "Diagnostic should name the stage"
        assert "stage_a" not in engine._deferred_events, "Remaining events should be dropped"


@pytest.mark.anyio
@pytest.mark.usefixtures("_reset_deferred_test_state")
async def test_deferred_events_multi_level_redeferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify while-loop processes multiple levels of re-deferral (not just 1)."""
    async with Engine() as engine:
        monkeypatch.setattr(
            engine,
            "_handle_input_event",
            lambda event: _helper_multi_level_defer(engine, event),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType] - monkeypatch lambda
        )

        engine._defer_event_for_stage(
            "stage_a",
            DataArtifactChanged(type="data_artifact_changed", paths=["seed.csv"]),
        )

        await engine._process_deferred_events("stage_a")

        assert _deferred_multi_count == 4, (  # type: ignore[comparison-overlap] - mutated by async handler
            f"Expected 4 events processed (seed + 3 levels), got {_deferred_multi_count}"
        )
        assert "stage_a" not in engine._deferred_events, "No leftover deferred events"


# =============================================================================
# Coordinator-Side Skip Detection Tests
# =============================================================================


@pytest.mark.anyio
async def test_coordinator_skip_avoids_worker_dispatch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """Stages that can skip via generation check are completed inline,
    never dispatched to the worker pool."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    register_test_stage(
        _helper_coordinator_skip_stage,
        name="skip_stage",
        pipeline=test_pipeline,
    )
    stage_info = test_pipeline.get("skip_stage")

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    _ = input_path.write_text("input")
    _ = output_path.write_text("output")

    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    (cache_dir / "files").mkdir(parents=True, exist_ok=True)
    lock.get_stages_dir(state_dir).mkdir(parents=True, exist_ok=True)

    fp = test_pipeline.ensure_fingerprint("skip_stage")
    current_params = parameters.get_effective_params(
        stage_info["params"], "skip_stage", parameters.ParamsOverrides()
    )

    with state_mod.StateDB(state_dir / "state.db") as state_db:
        dep_hash, _ = cache.hash_file(input_path, state_db)
        out_hash, _ = cache.hash_file(output_path, state_db)

        lock_data = LockData(
            code_manifest=fp,
            params=current_params,
            dep_hashes={str(input_path): {"hash": dep_hash}},
            output_hashes={str(output_path): {"hash": out_hash}},
        )

        stage_lock = lock.StageLock("skip_stage", lock.get_stages_dir(state_dir))
        stage_lock.write(lock_data)

        dep_generation = state_db.increment_generation(input_path)
        normalized_dep = str(project.normalize_path(input_path))
        state_db.record_dep_generations("skip_stage", {normalized_dep: dep_generation})

    async with Engine(pipeline=test_pipeline) as engine:
        await engine._initialize_orchestration(["skip_stage"], max_workers=1)
        engine._scheduler.acquire_mutexes("skip_stage")
        await engine._set_stage_state("skip_stage", StageExecutionState.PREPARING)

        results: dict[str, executor_core.ExecutionSummary] = {}
        engine._worker_pool = MagicMock()

        skipped = await engine._try_skip_in_coordinator(
            stage_name="skip_stage",
            stage_info=stage_info,
            overrides=parameters.ParamsOverrides(),
            checkout_modes=cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
            force=False,
            cache_dir=cache_dir,
            state_dir=state_dir,
            project_root=tmp_path,
            results=results,
            run_id="run-test",
        )

        assert skipped is True, "Coordinator should skip inline when generations match"
        assert results["skip_stage"]["status"] == "skipped"
        engine._worker_pool.submit.assert_not_called()


@pytest.mark.anyio
async def test_all_cached_pipeline_skips_without_workers(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """When all stages are cached, _start_ready_stages completes them inline
    without calling WorkerPool.submit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    register_test_stage(
        _helper_coordinator_skip_stage,
        name="cached_stage",
        pipeline=test_pipeline,
    )
    stage_info = test_pipeline.get("cached_stage")

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    _ = input_path.write_text("input")
    _ = output_path.write_text("output")

    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    (cache_dir / "files").mkdir(parents=True, exist_ok=True)
    lock.get_stages_dir(state_dir).mkdir(parents=True, exist_ok=True)

    fp = test_pipeline.ensure_fingerprint("cached_stage")
    current_params = parameters.get_effective_params(
        stage_info["params"], "cached_stage", parameters.ParamsOverrides()
    )

    with state_mod.StateDB(state_dir / "state.db") as state_db:
        dep_hash, _ = cache.hash_file(input_path, state_db)
        out_hash, _ = cache.hash_file(output_path, state_db)

        lock_data = LockData(
            code_manifest=fp,
            params=current_params,
            dep_hashes={str(input_path): {"hash": dep_hash}},
            output_hashes={str(output_path): {"hash": out_hash}},
        )

        stage_lock = lock.StageLock("cached_stage", lock.get_stages_dir(state_dir))
        stage_lock.write(lock_data)

        dep_generation = state_db.increment_generation(input_path)
        normalized_dep = str(project.normalize_path(input_path))
        state_db.record_dep_generations("cached_stage", {normalized_dep: dep_generation})

    async with Engine(pipeline=test_pipeline) as engine:
        await engine._initialize_orchestration(["cached_stage"], max_workers=1)

        results: dict[str, executor_core.ExecutionSummary] = {}
        await engine._start_ready_stages(
            cache_dir=cache_dir,
            overrides=parameters.ParamsOverrides(),
            checkout_modes=cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
            force=False,
            no_commit=False,
            stage_start_times={},
            run_id="run-test",
            project_root=tmp_path,
            state_dir=state_dir,
            results=results,
        )

        assert engine._worker_pool is None, "Pool should not be created when all stages skip"
        assert "cached_stage" in results, "Skipped stage should be in results"
        assert results["cached_stage"]["status"] == "skipped"


@pytest.mark.anyio
async def test_try_skip_returns_false_when_force(
    tmp_path: pathlib.Path,
) -> None:
    """_try_skip_in_coordinator returns False immediately when force=True."""
    async with Engine() as engine:
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_info = cast("RegistryStageInfo", MagicMock())
        skipped = await engine._try_skip_in_coordinator(
            stage_name="any_stage",
            stage_info=stage_info,
            overrides={},
            checkout_modes=[],
            force=True,
            cache_dir=tmp_path,
            state_dir=tmp_path,
            project_root=tmp_path,
            results=results,
            run_id="test",
        )
        assert skipped is False, "Should not skip when force=True"
        assert results == {}, "No results should be recorded"


@pytest.mark.anyio
async def test_try_skip_returns_false_when_no_lock(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_try_skip_in_coordinator returns False when no lock file exists."""
    monkeypatch.setattr(
        "pivot.registry.get_stage_state_dir",
        lambda _info, _default: tmp_path,  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    )

    async with Engine() as engine:
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_info = cast("RegistryStageInfo", MagicMock())
        skipped = await engine._try_skip_in_coordinator(
            stage_name="no_lock_stage",
            stage_info=stage_info,
            overrides={},
            checkout_modes=[],
            force=False,
            cache_dir=tmp_path,
            state_dir=tmp_path,
            project_root=tmp_path,
            results=results,
            run_id="test",
        )
        assert skipped is False, "Should not skip when no lock file"


@pytest.mark.anyio
async def test_coordinator_tier2_skip_when_generation_unavailable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """When Tier 1 (generation) has no data, Tier 2 (lock file comparison) still
    allows coordinator-side skip — verifying the full fallback path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    register_test_stage(
        _helper_coordinator_skip_stage,
        name="tier2_stage",
        pipeline=test_pipeline,
    )
    stage_info = test_pipeline.get("tier2_stage")

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    _ = input_path.write_text("input")
    _ = output_path.write_text("output")

    state_dir = tmp_path / ".pivot"
    cache_dir = state_dir / "cache"
    (cache_dir / "files").mkdir(parents=True, exist_ok=True)
    lock.get_stages_dir(state_dir).mkdir(parents=True, exist_ok=True)

    fp = test_pipeline.ensure_fingerprint("tier2_stage")
    current_params = parameters.get_effective_params(
        stage_info["params"], "tier2_stage", parameters.ParamsOverrides()
    )

    # Write lock data and file hashes — but do NOT record dep generations.
    # This makes Tier 1 (can_skip_via_generation) return False,
    # forcing the code through Tier 2 (skip.check_stage).
    with state_mod.StateDB(state_dir / "state.db") as state_db:
        dep_hash, _ = cache.hash_file(input_path, state_db)
        out_hash, _ = cache.hash_file(output_path, state_db)

        # Use normalized paths (absolute) as hash_dependencies does
        normalized_dep = str(project.normalize_path(input_path))
        normalized_out = str(project.normalize_path(output_path))

        lock_data = LockData(
            code_manifest=fp,
            params=current_params,
            dep_hashes={normalized_dep: {"hash": dep_hash}},
            output_hashes={normalized_out: {"hash": out_hash}},
        )

        stage_lock = lock.StageLock("tier2_stage", lock.get_stages_dir(state_dir))
        stage_lock.write(lock_data)

        # Deliberately NOT calling state_db.record_dep_generations —
        # this is what forces the Tier 2 path.

    async with Engine(pipeline=test_pipeline) as engine:
        await engine._initialize_orchestration(["tier2_stage"], max_workers=1)
        engine._scheduler.acquire_mutexes("tier2_stage")
        await engine._set_stage_state("tier2_stage", StageExecutionState.PREPARING)

        results: dict[str, executor_core.ExecutionSummary] = {}
        engine._worker_pool = MagicMock()

        skipped = await engine._try_skip_in_coordinator(
            stage_name="tier2_stage",
            stage_info=stage_info,
            overrides=parameters.ParamsOverrides(),
            checkout_modes=cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
            force=False,
            cache_dir=cache_dir,
            state_dir=state_dir,
            project_root=tmp_path,
            results=results,
            run_id="run-tier2-test",
        )

        assert skipped is True, "Coordinator should skip via Tier 2 when generations unavailable"
        assert results["tier2_stage"]["status"] == "skipped"
        assert results["tier2_stage"]["reason"] == "unchanged", (
            "Tier 2 skip reason should be 'unchanged', not 'unchanged (generation)'"
        )
        engine._worker_pool.submit.assert_not_called()
