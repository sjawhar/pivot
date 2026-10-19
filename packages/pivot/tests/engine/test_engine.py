"""Tests for the Engine class."""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import anyio
import networkx as nx
import pytest

from pivot import exceptions, loaders, parameters, project, types
from pivot.compose import Pipeline, stage
from pivot.engine import engine as engine_mod
from pivot.engine import graph as engine_graph
from pivot.engine.engine import Engine
from pivot.engine.sinks import ResultCollectorSink
from pivot.engine.sources import OneShotSource
from pivot.engine.types import (
    CancelRequested,
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
from pivot.executor import core as executor_core
from pivot.storage import cache, lock
from pivot.storage import state as state_mod
from pivot.storage import store as store_mod
from pivot.types import FileHash, LockData, OnError, StageStatus, identity_key

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo


@stage
def _helper_watch_scale(raw: dict[str, int]) -> dict[str, int]:
    return {key: value * 2 for key, value in raw.items()}


@stage
def _helper_watch_total(scaled: dict[str, int]) -> dict[str, int]:
    return {"total": sum(scaled.values())}


@stage
def _helper_watch_merge(
    first: dict[str, int],
    second: dict[str, int],
) -> dict[str, int]:
    return {**first, **second}


@stage
def _helper_cancel_first() -> str:
    return "first"


@stage
def _helper_cancel_second(value: str) -> str:
    return f"{value} second"


@stage
def _helper_cancel_third(value: str) -> str:
    return f"{value} third"


@stage
def _helper_cancel_other() -> str:
    return "other"


def _helper_watch_pipeline(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Pipeline:
    (tmp_path / ".pivot").mkdir()
    (tmp_path / ".git").mkdir()
    raw_path = tmp_path / "data" / "raw" / "raw.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text('{"a": 1, "b": 2}')
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    with Pipeline("watch", root=tmp_path) as pipeline:
        raw = pipeline.input(
            "raw.json",
            path="data/raw/raw.json",
            t=cast("type[dict[str, int]]", dict),
        )
        scaled = _helper_watch_scale(raw)
        _helper_watch_total(scaled)

    return pipeline


def _helper_prepare_coordinator_skip(
    tmp_path: pathlib.Path,
    pipeline: Pipeline,
    *,
    record_generations: bool,
) -> tuple[str, RegistryStageInfo]:
    """Create a valid stage lock and optional generation record for the scale stage."""
    from pivot.storage import store as store_mod

    stage_name = "watch/_helper_watch_scale"
    stage_info = pipeline.get_stage(stage_name)
    dep = stage_info["deps"]["raw"]
    out = stage_info["outs"][0]
    store = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )
    output_path = store.resolve_display_path(out)
    output_path.parent.mkdir(parents=True)
    output_path.write_text("a: 2\nb: 4\n")
    state_dir = stage_info["state_dir"] or tmp_path / ".pivot"
    lock.get_stages_dir(state_dir).mkdir(parents=True, exist_ok=True)
    params = parameters.get_effective_params(
        stage_info["params"],
        stage_name,
        parameters.ParamsOverrides(),
    )

    with state_mod.StateDB(state_dir) as state_db:
        dep_hash, _ = cache.hash_file(tmp_path / "data" / "raw" / "raw.json", state_db)
        output_hash, _ = cache.hash_file(output_path, state_db)
        lock.StageLock(stage_name, lock.get_stages_dir(state_dir)).write(
            LockData(
                code_manifest=pipeline.ensure_fingerprint(stage_name),
                params=params,
                dep_hashes={dep.identity: FileHash(hash=dep_hash)},
                output_hashes={out.identity: FileHash(hash=output_hash)},
            )
        )
        if record_generations:
            dep_key = identity_key(dep.identity)
            generation = state_db.increment_generation(dep_key)
            state_db.record_dep_generations(stage_name, {dep_key: generation})

    return stage_name, stage_info


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch defers an event for an output its producer is still writing."""
    from pivot.storage import store as store_mod

    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    scale_stage = "watch/_helper_watch_scale"
    total_stage = "watch/_helper_watch_total"
    scale_info = pipeline.get_stage(scale_stage)
    output_path = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    ).resolve_display_path(scale_info["outs"][0])

    async with Engine(pipeline=pipeline) as engine:
        engine._graph = engine_graph.build_graph(
            {
                scale_stage: scale_info,
                total_stage: pipeline.get_stage(total_stage),
            }
        )
        engine._scheduler.set_state(scale_stage, StageExecutionState.RUNNING)
        executed = list[list[str]]()

        async def _capture_execution(stages: list[str]) -> None:
            executed.append(stages)

        engine._execute_affected_stages = _capture_execution
        await engine._handle_data_artifact_changed(
            DataArtifactChanged(type="data_artifact_changed", paths=[str(output_path)])
        )

        assert list(engine._deferred_events[scale_stage]) == [
            DataArtifactChanged(type="data_artifact_changed", paths=[str(output_path)])
        ]
        assert executed == []


@pytest.mark.anyio
async def test_engine_handle_data_artifact_changed_runs_consumers_and_downstream(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watched raw input change executes its consumer and downstream stages."""
    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    scale_stage = "watch/_helper_watch_scale"
    total_stage = "watch/_helper_watch_total"
    executed = list[list[str]]()

    async def _capture_execution(stages: list[str]) -> None:
        executed.append(stages)

    async with Engine(pipeline=pipeline) as engine:
        engine._execute_affected_stages = _capture_execution
        await engine._handle_data_artifact_changed(
            DataArtifactChanged(
                type="data_artifact_changed",
                paths=[str(tmp_path / "data" / "raw" / "raw.json")],
            )
        )

    assert executed == [[scale_stage, total_stage]]


@pytest.mark.anyio
async def test_engine_ignores_its_completed_output_write(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed event for an engine-written output must not restart consumers."""
    from pivot.engine.sources import FilesystemSource
    from pivot.storage import store as store_mod

    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    scale_stage = "watch/_helper_watch_scale"
    total_stage = "watch/_helper_watch_total"
    scale_info = pipeline.get_stage(scale_stage)
    output_path = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    ).resolve_display_path(scale_info["outs"][0])
    output_path.parent.mkdir(parents=True)
    output_path.write_text("a: 2\nb: 4\n")
    executed = list[list[str]]()

    async def _capture_execution(stages: list[str]) -> None:
        executed.append(stages)

    async with Engine(pipeline=pipeline) as engine:
        engine.add_source(FilesystemSource(watch_paths=[tmp_path]))
        engine._graph = engine_graph.build_graph(
            {
                scale_stage: scale_info,
                total_stage: pipeline.get_stage(total_stage),
            }
        )
        await engine._initialize_orchestration([scale_stage, total_stage], max_workers=1)
        await engine._set_stage_state(scale_stage, StageExecutionState.RUNNING)
        await engine._handle_stage_completion(
            scale_stage,
            {
                "status": StageStatus.RAN,
                "reason": "ran",
                "input_hash": "input",
                "output_lines": [],
            },
            time.perf_counter(),
            tmp_path / ".pivot",
        )
        engine._execute_affected_stages = _capture_execution
        await engine._handle_data_artifact_changed(
            DataArtifactChanged(type="data_artifact_changed", paths=[str(output_path)])
        )

    assert executed == []


@pytest.mark.anyio
async def test_engine_distinguishes_own_and_external_directory_output_writes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory-output child writes are suppressed only when pivot wrote them."""
    from pivot.engine.sources import FilesystemSource
    from pivot.storage import store as store_mod

    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    scale_stage = "watch/_helper_watch_scale"
    total_stage = "watch/_helper_watch_total"
    scale_info = pipeline.get_stage(scale_stage)
    total_info = pipeline.get_stage(total_stage)
    directory_ref = types.ArtifactRef(
        identity=scale_info["outs"][0].identity,
        format=loaders.Text(),
        python_type=str,
        tag=types.ArtifactTag.DIRECTORY,
    )
    scale_info["outs"] = [directory_ref]
    total_info["deps"]["scaled"] = directory_ref
    output_dir = store_mod.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    ).resolve_display_path(directory_ref)
    child_path = output_dir / "results" / "sub" / "child.txt"
    child_path.parent.mkdir(parents=True)
    child_path.write_text("pivot write")
    executed = list[list[str]]()

    async def _capture_execution(stages: list[str]) -> None:
        executed.append(stages)

    async with Engine(pipeline=pipeline) as engine:
        engine.add_source(FilesystemSource(watch_paths=[tmp_path]))
        engine._graph = engine_graph.build_graph(
            {
                scale_stage: scale_info,
                total_stage: total_info,
            }
        )
        await engine._initialize_orchestration([scale_stage, total_stage], max_workers=1)
        await engine._set_stage_state(scale_stage, StageExecutionState.RUNNING)
        await engine._handle_stage_completion(
            scale_stage,
            {
                "status": StageStatus.RAN,
                "reason": "ran",
                "input_hash": "input",
                "output_lines": [],
            },
            time.perf_counter(),
            tmp_path / ".pivot",
        )
        engine._execute_affected_stages = _capture_execution
        await engine._handle_data_artifact_changed(
            DataArtifactChanged(type="data_artifact_changed", paths=[str(child_path)])
        )
        assert executed == []

        new_external_path = output_dir / "results" / "sub" / "new.txt"
        new_external_path.write_text("external nested write")
        await engine._handle_data_artifact_changed(
            DataArtifactChanged(type="data_artifact_changed", paths=[str(new_external_path)])
        )

    assert executed == [[total_stage]]


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple changed inputs affecting one stage are returned once."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / ".git").mkdir()
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "first.json").write_text('{"first": 1}')
    (raw_dir / "second.json").write_text('{"second": 2}')
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    with Pipeline("watch", root=tmp_path) as pipeline:
        first = pipeline.input(
            "first.json",
            path="data/raw/first.json",
            t=cast("type[dict[str, int]]", dict),
        )
        second = pipeline.input(
            "second.json",
            path="data/raw/second.json",
            t=cast("type[dict[str, int]]", dict),
        )
        _helper_watch_merge(first, second)

    async with Engine(pipeline=pipeline) as engine:
        affected = engine._get_affected_stages_for_paths(
            [
                raw_dir / "first.json",
                raw_dir / "second.json",
            ]
        )

    assert affected == ["watch/_helper_watch_merge"]


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
            resource_capacity={"gpu": 2},
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
        assert engine._stored_resource_capacity == {"gpu": 2}


@pytest.mark.anyio
async def test_engine_execute_affected_stages_uses_stored_params() -> None:
    """_execute_affected_stages() uses stored orchestration params instead of hardcoded defaults."""

    async with Engine() as engine:
        # Set stored params (simulating what _handle_run_requested would do)
        engine._stored_no_commit = True
        engine._stored_on_error = OnError.KEEP_GOING
        engine._stored_parallel = True
        engine._stored_max_workers = 4
        engine._stored_resource_capacity = {"gpu": 2}

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
        assert captured_kwargs["resource_capacity"] == {"gpu": 2}


@pytest.mark.anyio
async def test_engine_execute_affected_stages_propagates_non_parallel_mode() -> None:
    """_execute_affected_stages() correctly propagates parallel=False to orchestration."""

    async with Engine() as engine:
        engine._stored_parallel = False
        engine._stored_max_workers = None
        engine._stored_no_commit = False
        engine._stored_on_error = OnError.FAIL
        engine._stored_resource_capacity = {}

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
            resource_capacity={},
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
async def test_engine_cancel_during_coordinator_skip_releases_resources_without_submit(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation before first submission finalizes every remaining stage once."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    with Pipeline("cancel", root=tmp_path) as pipeline:
        first = _helper_cancel_first()
        _helper_cancel_second(first)
        _helper_cancel_other()

    stage_names = {
        "cancel/_helper_cancel_first",
        "cancel/_helper_cancel_second",
        "cancel/_helper_cancel_other",
    }
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir()
    events = list[OutputEvent]()
    pool = MagicMock(spec=WorkerPool)
    pool.submit.side_effect = AssertionError("cancelled stage was submitted")
    results = dict[str, executor_core.ExecutionSummary]()

    async with Engine(pipeline=pipeline) as engine:
        engine._run_state = engine_mod._RunState.RUNNING

        async def _cancel_during_skip(*_args: object, **_kwargs: object) -> bool:
            await engine._handle_cancel_requested()
            return False

        monkeypatch.setattr(engine, "_try_skip_in_coordinator", _cancel_during_skip)
        monkeypatch.setattr(engine, "_ensure_worker_pool", lambda: pool)
        output_recv = engine._output_recv
        output_send = engine._output_send
        assert output_recv is not None
        assert output_send is not None

        async def _capture_events() -> None:
            async for event in output_recv:
                events.append(event)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_capture_events)
            results = await engine._orchestrate_execution(
                stages=None,
                force=False,
                single_stage=False,
                parallel=True,
                max_workers=1,
                resource_capacity={},
                no_commit=True,
                on_error=OnError.FAIL,
                cache_dir=cache_dir,
            )
            await output_send.aclose()

        assert set(results) == stage_names
        assert all(result["status"] == StageStatus.CANCELLED for result in results.values())
        assert all(
            engine._scheduler.get_state(stage_name) == StageExecutionState.COMPLETED
            for stage_name in stage_names
        )
        terminal_stages = [
            event["stage"]
            for event in events
            if event["type"] == "stage_completed" and event["status"] == StageStatus.CANCELLED
        ]
        assert sorted(terminal_stages) == sorted(stage_names)
        assert len(terminal_stages) == len(set(terminal_stages))
        pool.submit.assert_not_called()
        engine._scheduler.set_state("cancel/_helper_cancel_first", StageExecutionState.READY)
        assert engine._scheduler.can_start("cancel/_helper_cancel_first")


@pytest.mark.anyio
async def test_engine_cancel_after_worker_completion_finalizes_second_dispatch_wave(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation while the second stage prepares finalizes all later stages."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    with Pipeline("cancel_late", root=tmp_path) as pipeline:
        first = _helper_cancel_first()
        second = _helper_cancel_second(first)
        _helper_cancel_third(second)
        _helper_cancel_other()

    first_stage = "cancel_late/_helper_cancel_first"
    cancelled_stages = {
        "cancel_late/_helper_cancel_second",
        "cancel_late/_helper_cancel_third",
        "cancel_late/_helper_cancel_other",
    }
    all_stages = {first_stage, *cancelled_stages}
    cache_dir = tmp_path / ".pivot" / "cache"
    cache_dir.mkdir()
    events = list[OutputEvent]()
    results = dict[str, executor_core.ExecutionSummary]()

    async with Engine(pipeline=pipeline) as engine:
        engine._run_state = engine_mod._RunState.RUNNING
        output_recv = engine._output_recv
        output_send = engine._output_send
        assert output_recv is not None
        assert output_send is not None

        async def _cancel_on_second_preparing() -> None:
            async for event in output_recv:
                events.append(event)
                if (
                    event["type"] == "stage_state_changed"
                    and event["stage"] == "cancel_late/_helper_cancel_second"
                    and event["state"] == StageExecutionState.PREPARING
                ):
                    await engine._handle_cancel_requested()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_cancel_on_second_preparing)
            results = await engine._orchestrate_execution(
                stages=None,
                force=False,
                single_stage=False,
                parallel=True,
                max_workers=1,
                resource_capacity={},
                no_commit=True,
                on_error=OnError.FAIL,
                cache_dir=cache_dir,
            )
            await output_send.aclose()

        assert set(results) == all_stages
        assert results[first_stage]["status"] == StageStatus.RAN
        assert all(
            results[stage_name]["status"] == StageStatus.CANCELLED
            for stage_name in cancelled_stages
        )
        assert all(
            engine._scheduler.get_state(stage_name) == StageExecutionState.COMPLETED
            for stage_name in all_stages
        )
        terminal_stages = [event["stage"] for event in events if event["type"] == "stage_completed"]
        assert sorted(terminal_stages) == sorted(all_stages)
        assert len(terminal_stages) == len(set(terminal_stages))
        engine._scheduler.set_state(first_stage, StageExecutionState.READY)
        assert engine._scheduler.can_start(first_stage)


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
            resource_capacity={},
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
            resource_capacity={},
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


def _helper_run_request(stages: list[str] | None, *, force: bool = False) -> RunRequested:
    return RunRequested(
        type="run_requested",
        stages=stages,
        force=force,
        reason="test",
        single_stage=False,
        parallel=True,
        max_workers=None,
        resource_capacity={},
        no_commit=False,
        on_error=OnError.FAIL,
        cache_dir=None,
        allow_uncached_incremental=False,
        checkout_missing=False,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("first", "second", "replayed"),
    [
        pytest.param(
            (["stage_a"], False), (["stage_b"], False), (["stage_a", "stage_b"], False), id="union"
        ),
        pytest.param((None, False), (["stage_b"], False), (None, False), id="all-stays-all"),
        pytest.param(
            (["stage_a"], True),
            (["stage_b"], False),
            (["stage_a", "stage_b"], True),
            id="forced-stays-forced",
        ),
    ],
)
async def test_engine_restart_covers_the_cancelled_run_and_the_new_request(
    first: tuple[list[str] | None, bool],
    second: tuple[list[str] | None, bool],
    replayed: tuple[list[str] | None, bool],
) -> None:
    """A request that cancels a run replays both: a restart never narrows what was going to run."""
    calls = list[tuple[list[str] | None, bool]]()

    async with Engine() as engine:

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            calls.append(
                (cast("list[str] | None", kwargs["stages"]), cast("bool", kwargs["force"]))
            )
            if len(calls) == 1:
                with anyio.fail_after(5):
                    await engine._cancel_event.wait()
            return {}

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._handle_run_requested(_helper_run_request(first[0], force=first[1]))
            await anyio.wait_all_tasks_blocked()
            await engine._handle_run_requested(_helper_run_request(second[0], force=second[1]))

    assert calls == [first, replayed]


@pytest.mark.anyio
async def test_engine_restart_after_deferred_reload_skips_removed_stages() -> None:
    """A restart replayed after a deferred reload drops the stages that reload removed."""
    stage_list = ["stage_a", "stage_gone"]
    calls = list[list[str] | None]()

    async with Engine() as engine:
        pipeline = MagicMock()
        pipeline.list_stages.side_effect = lambda: list(stage_list)
        engine._pipeline = pipeline

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            stages = cast("list[str] | None", kwargs["stages"])
            unknown = [s for s in stages or [] if s not in stage_list]
            if unknown:
                raise exceptions.StageNotFoundError(unknown, available_stages=stage_list)
            calls.append(stages)
            if len(calls) == 1:
                with anyio.fail_after(5):
                    await engine._cancel_event.wait()
            return {}

        async def mock_code_changed(_event: CodeOrConfigChanged) -> None:
            stage_list.remove("stage_gone")
            await engine._execute_affected_stages(list(stage_list))

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._handle_code_or_config_changed = mock_code_changed  # pyright: ignore[reportAttributeAccessIssue] - test mock

        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._handle_run_requested(_helper_run_request(["stage_a", "stage_gone"]))
            await anyio.wait_all_tasks_blocked()
            engine._reload_pending = {"pipeline.py"}
            await engine._handle_run_requested(_helper_run_request(["stage_gone"]))

    assert calls[0] == ["stage_a", "stage_gone"]
    assert calls[-1] == ["stage_a"], "the replayed restart runs what still exists"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("steps", "replayed"),
    [
        pytest.param(["stage_b", "stage_c"], ["stage_a", "stage_b", "stage_c"], id="edits-only"),
        pytest.param(["cancel", "stage_c"], ["stage_c"], id="cancel-then-edit"),
        pytest.param(["stage_c", "cancel"], ["stage_c"], id="edit-then-cancel"),
    ],
)
async def test_engine_run_request_while_cancelling_waits_for_the_run_to_end(
    steps: list[str],
    replayed: list[str],
) -> None:
    """A request made while a run is cancelling starts no second run alongside it.

    Requests that cancel a run replay with its stages once it ends. An explicit cancel, made
    before or after those requests, stops the run's own stages from replaying; the requests
    still replay.
    """
    calls = list[list[str] | None]()
    in_flight = 0
    most_in_flight = 0
    release = anyio.Event()

    async with Engine() as engine:

        async def mock_orchestrate(**kwargs: object) -> dict[str, object]:
            nonlocal in_flight, most_in_flight
            in_flight += 1
            most_in_flight = max(most_in_flight, in_flight)
            calls.append(cast("list[str] | None", kwargs["stages"]))
            try:
                if len(calls) == 1:
                    # A stage still running keeps the cancelled run in CANCELLING
                    with anyio.fail_after(5):
                        await engine._cancel_event.wait()
                        await release.wait()
                return {}
            finally:
                in_flight -= 1

        engine._orchestrate_execution = mock_orchestrate  # pyright: ignore[reportAttributeAccessIssue] - test mock
        engine._require_pipeline = lambda: None  # pyright: ignore[reportAttributeAccessIssue] - test mock

        async with anyio.create_task_group() as tg:
            engine._run_task_group = tg
            await engine._handle_input_event(_helper_run_request(["stage_a"]))
            await anyio.wait_all_tasks_blocked()
            for step in steps:
                if step == "cancel":
                    await engine._handle_input_event(CancelRequested(type="cancel_requested"))
                else:
                    await engine._handle_input_event(_helper_run_request([step]))
                assert engine._run_state == engine_mod._RunState.CANCELLING
            await anyio.wait_all_tasks_blocked()
            release.set()

    assert most_in_flight == 1, f"a second run started while the first was cancelling: {calls}"
    assert calls == [["stage_a"], replayed]


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


async def _helper_reload_ok() -> dict[str, RegistryStageInfo]:
    """Stand-in for Engine._reload_registry: a reload that succeeded with no old stages."""
    return {}


async def _helper_reload_failed() -> str:
    """Stand-in for Engine._reload_registry: a reload that found the pipeline invalid."""
    return "Failed to load pipeline.py"


def _helper_bipartite_chain(names: list[str]) -> nx.DiGraph[str]:
    """Bipartite graph in which each stage consumes the output of the stage before it."""
    g: nx.DiGraph[str] = nx.DiGraph()
    for previous, name in zip([None, *names], names, strict=False):
        stage_node = engine_graph.stage_node(name)
        output = engine_graph.artifact_node(types.ArtifactIdentity(name, None))
        g.add_node(stage_node, type=NodeType.STAGE)
        g.add_node(output, type=NodeType.ARTIFACT)
        g.add_edge(stage_node, output)
        if previous is not None:
            g.add_edge(
                engine_graph.artifact_node(types.ArtifactIdentity(previous, None)), stage_node
            )
    return g


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("parallel", "max_workers", "expected_restart_calls"),
    [
        pytest.param(True, 4, [(1, 4)], id="parallel-restarts-workers"),
        pytest.param(False, None, [], id="non-parallel-skips-restart"),
    ],
)
async def test_engine_handle_code_or_config_changed_restart_behavior(
    monkeypatch: pytest.MonkeyPatch,
    parallel: bool,
    max_workers: int | None,
    expected_restart_calls: list[tuple[int, int | None]],
) -> None:
    """A reload restarts workers only in parallel mode, one for each stage of the reloaded
    pipeline that can run at once: a chain of three needs one."""
    restart_calls = list[tuple[int, int | None]]()

    def mock_restart_workers(width: int, max_workers: int | None = None) -> int:
        restart_calls.append((width, max_workers))
        return width

    monkeypatch.setattr("pivot.executor.core.restart_workers", mock_restart_workers)

    async with Engine() as engine:
        engine._stored_parallel = parallel
        engine._stored_max_workers = max_workers

        mock_pipeline = MagicMock()
        mock_pipeline.list_stages.return_value = ["a", "b", "c"]
        mock_pipeline.snapshot.return_value = {}
        mock_pipeline._registry = None
        mock_pipeline.get.return_value = MagicMock()
        mock_pipeline.invalidate_dag_cache.return_value = None
        engine._pipeline = mock_pipeline
        engine._reload_registry = _helper_reload_ok  # type: ignore[assignment] - test mock

        async def mock_emit_reload(*_args: object) -> None:
            pass

        async def mock_execute_affected(_stages: list[str]) -> None:
            pass

        engine._emit_reload_event = mock_emit_reload  # pyright: ignore[reportAttributeAccessIssue] - test seam
        engine._execute_affected_stages = mock_execute_affected  # pyright: ignore[reportAttributeAccessIssue] - test seam

        def mock_build_graph(*_args: object) -> nx.DiGraph[str]:
            return _helper_bipartite_chain(["a", "b", "c"])

        monkeypatch.setattr("pivot.engine.engine.engine_graph.build_graph", mock_build_graph)
        await engine._handle_code_or_config_changed(
            CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        )

    assert restart_calls == expected_restart_calls


@pytest.mark.anyio
async def test_engine_code_reload_restores_project_watch_root(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload keeps project Python files inside the filesystem watch scope."""
    from pivot.engine.sources import FilesystemSource

    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    source = FilesystemSource(watch_paths=[])

    async with Engine(pipeline=pipeline) as engine:
        engine.add_source(source)
        engine._stored_parallel = False
        engine._reload_registry = _helper_reload_ok  # type: ignore[assignment] - test seam

        async def _ignore_reload(*_args: object) -> None:
            pass

        async def _ignore_execution(_stages: list[str]) -> None:
            pass

        engine._emit_reload_event = _ignore_reload  # pyright: ignore[reportAttributeAccessIssue] - test seam
        engine._execute_affected_stages = _ignore_execution  # pyright: ignore[reportAttributeAccessIssue] - test seam
        await engine._handle_code_or_config_changed(
            CodeOrConfigChanged(
                type="code_or_config_changed",
                paths=[str(tmp_path / "pipeline.py")],
            )
        )

    assert source.watch_paths == [tmp_path]


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
        engine._reload_registry = _helper_reload_failed  # type: ignore[assignment] - test mock
        engine._invalidate_caches = lambda: None  # type: ignore[assignment] - test mock

        event = CodeOrConfigChanged(type="code_or_config_changed", paths=["pipeline.py"])
        await engine._handle_code_or_config_changed(event)

        assert len(restart_calls) == 0, "Should not restart workers on failed reload"


@pytest.mark.anyio
async def test_engine_handle_code_or_config_changed_continues_on_restart_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_handle_code_or_config_changed continues if restart_workers raises."""

    def mock_restart_workers(width: int, max_workers: int | None = None) -> int:
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

        engine._reload_registry = _helper_reload_ok  # type: ignore[assignment] - test mock

        async def mock_emit_reload(*_args: object) -> None:
            pass

        engine._emit_reload_event = mock_emit_reload  # pyright: ignore[reportAttributeAccessIssue]

        async def mock_execute_affected(stages: list[str]) -> None:
            executed.append(stages)

        engine._execute_affected_stages = mock_execute_affected  # type: ignore[assignment] - test mock

        def mock_build_graph(*_a: object) -> MagicMock:
            return MagicMock()

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


def _helper_store_spec(cache_dir: pathlib.Path, project_root: pathlib.Path) -> store_mod.StoreSpec:
    return store_mod.StoreSpec(
        kind="workspace",
        cache_dir=str(cache_dir / "files"),
        project_root=str(project_root),
        pipeline_name="watch",
        input_bindings={"raw.json": "data/raw/raw.json"},
        checkout_modes=cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
    )


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
            force=True,
            store_spec=_helper_store_spec(tmp_path, tmp_path),
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
    test_pipeline: Pipeline,
) -> None:
    """_try_skip_in_coordinator returns False when no lock file exists."""
    monkeypatch.setattr(
        "pivot.registry.get_stage_state_dir",
        lambda _info, _default: tmp_path,  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    )

    async with Engine(pipeline=test_pipeline) as engine:
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_info = cast("RegistryStageInfo", MagicMock())
        skipped = await engine._try_skip_in_coordinator(
            stage_name="no_lock_stage",
            stage_info=stage_info,
            overrides={},
            force=False,
            store_spec=_helper_store_spec(tmp_path, tmp_path),
            state_dir=tmp_path,
            project_root=tmp_path,
            results=results,
            run_id="test",
        )
        assert skipped is False, "Should not skip when no lock file"


@pytest.mark.anyio
async def test_coordinator_skip_avoids_worker_dispatch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation-matched stages complete without submitting worker work."""
    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    stage_name, stage_info = _helper_prepare_coordinator_skip(
        tmp_path,
        pipeline,
        record_generations=True,
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)
    results: dict[str, executor_core.ExecutionSummary] = {}

    async with Engine(pipeline=pipeline) as engine:
        engine._graph = engine_graph.build_graph({stage_name: stage_info})
        await engine._initialize_orchestration([stage_name], max_workers=1)
        engine._scheduler.acquire_resources(stage_name)
        await engine._set_stage_state(stage_name, StageExecutionState.PREPARING)
        pool = MagicMock()
        engine._worker_pool = pool

        skipped = await engine._try_skip_in_coordinator(
            stage_name=stage_name,
            stage_info=stage_info,
            overrides=parameters.ParamsOverrides(),
            force=False,
            store_spec=_helper_store_spec(cache_dir, tmp_path),
            state_dir=tmp_path / ".pivot",
            project_root=tmp_path,
            results=results,
            run_id="run-test",
        )

    assert skipped is True
    assert results[stage_name]["status"] == StageStatus.CACHED
    pool.submit.assert_not_called()


@pytest.mark.anyio
async def test_all_cached_pipeline_skips_without_workers(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generation-matched pipeline completes without creating a worker pool."""
    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    stage_name, stage_info = _helper_prepare_coordinator_skip(
        tmp_path,
        pipeline,
        record_generations=True,
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)
    results: dict[str, executor_core.ExecutionSummary] = {}

    async with Engine(pipeline=pipeline) as engine:
        engine._graph = engine_graph.build_graph({stage_name: stage_info})
        await engine._initialize_orchestration([stage_name], max_workers=1)
        await engine._start_ready_stages(
            cache_dir=cache_dir,
            overrides=parameters.ParamsOverrides(),
            checkout_modes=cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
            force=False,
            no_commit=False,
            stage_start_times={},
            run_id="run-test",
            project_root=tmp_path,
            state_dir=tmp_path / ".pivot",
            results=results,
        )

    assert engine._worker_pool is None
    assert results[stage_name]["status"] == StageStatus.CACHED


@pytest.mark.anyio
async def test_coordinator_tier2_skip_when_generation_unavailable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock comparison skips unchanged outputs when generations are unavailable."""
    pipeline = _helper_watch_pipeline(tmp_path, monkeypatch)
    stage_name, stage_info = _helper_prepare_coordinator_skip(
        tmp_path,
        pipeline,
        record_generations=False,
    )
    cache_dir = tmp_path / ".pivot" / "cache"
    (cache_dir / "files").mkdir(parents=True)
    results: dict[str, executor_core.ExecutionSummary] = {}

    async with Engine(pipeline=pipeline) as engine:
        engine._graph = engine_graph.build_graph({stage_name: stage_info})
        await engine._initialize_orchestration([stage_name], max_workers=1)
        engine._scheduler.acquire_resources(stage_name)
        await engine._set_stage_state(stage_name, StageExecutionState.PREPARING)
        pool = MagicMock()
        engine._worker_pool = pool

        skipped = await engine._try_skip_in_coordinator(
            stage_name=stage_name,
            stage_info=stage_info,
            overrides=parameters.ParamsOverrides(),
            force=False,
            store_spec=_helper_store_spec(cache_dir, tmp_path),
            state_dir=tmp_path / ".pivot",
            project_root=tmp_path,
            results=results,
            run_id="run-tier2-test",
        )

    assert skipped is True
    assert results[stage_name]["status"] == StageStatus.CACHED
    assert results[stage_name]["reason"] == "unchanged"
    pool.submit.assert_not_called()
