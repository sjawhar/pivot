"""Tests for engine type definitions."""

from __future__ import annotations

from pivot.engine import types
from pivot.types import OnError, StageStatus


def test_stage_execution_state_ordering() -> None:
    """Stage states have logical ordering for comparison."""
    assert types.StageExecutionState.PENDING < types.StageExecutionState.BLOCKED
    assert types.StageExecutionState.BLOCKED < types.StageExecutionState.READY
    assert types.StageExecutionState.READY < types.StageExecutionState.PREPARING
    assert types.StageExecutionState.PREPARING < types.StageExecutionState.RUNNING
    assert types.StageExecutionState.RUNNING < types.StageExecutionState.COMPLETED


def test_stage_execution_state_comparison() -> None:
    """Can use >= comparisons for state checks."""
    state = types.StageExecutionState.RUNNING
    assert state >= types.StageExecutionState.PREPARING  # Execution has begun
    assert state < types.StageExecutionState.COMPLETED  # Not done yet


def test_node_type_enum() -> None:
    """NodeType distinguishes artifacts from stages."""
    assert types.NodeType.ARTIFACT.value == "artifact"
    assert types.NodeType.STAGE.value == "stage"


def test_engine_state_enum() -> None:
    """EngineState has idle and active states."""
    assert types.EngineState.IDLE.value == "idle"
    assert types.EngineState.ACTIVE.value == "active"


def test_data_artifact_changed_event() -> None:
    """DataArtifactChanged event has required fields."""
    event: types.DataArtifactChanged = {
        "type": "data_artifact_changed",
        "paths": ["/path/to/data.csv", "/path/to/other.csv"],
    }
    assert event["type"] == "data_artifact_changed"
    assert len(event["paths"]) == 2


def test_code_or_config_changed_event() -> None:
    """CodeOrConfigChanged event has required fields."""
    event: types.CodeOrConfigChanged = {
        "type": "code_or_config_changed",
        "paths": ["/path/to/stages.py"],
    }
    assert event["type"] == "code_or_config_changed"


def test_run_requested_event() -> None:
    """RunRequested event has required fields."""
    event: types.RunRequested = {
        "type": "run_requested",
        "stages": ["train", "evaluate"],
        "force": False,
        "reason": "cli",
        "single_stage": False,
        "parallel": True,
        "max_workers": None,
        "no_commit": False,
        "on_error": OnError.FAIL,
        "cache_dir": None,
        "allow_uncached_incremental": False,
        "checkout_missing": False,
    }
    assert event["type"] == "run_requested"
    assert event["stages"] == ["train", "evaluate"]
    assert event["force"] is False

    # stages can be None (all stages)
    event_all: types.RunRequested = {
        "type": "run_requested",
        "stages": None,
        "force": True,
        "reason": "agent:run-123",
        "single_stage": False,
        "parallel": True,
        "max_workers": None,
        "no_commit": False,
        "on_error": OnError.FAIL,
        "cache_dir": None,
        "allow_uncached_incremental": False,
        "checkout_missing": False,
    }
    assert event_all["stages"] is None


def test_cancel_requested_event() -> None:
    """CancelRequested event has required fields."""
    event: types.CancelRequested = {"type": "cancel_requested"}
    assert event["type"] == "cancel_requested"


def test_input_event_union() -> None:
    """InputEvent is a union of all input event types."""
    # This test verifies the type alias exists and accepts all event types
    events: list[types.InputEvent] = [
        {"type": "data_artifact_changed", "paths": []},
        {"type": "code_or_config_changed", "paths": []},
        {
            "type": "run_requested",
            "stages": None,
            "force": False,
            "reason": "test",
            "single_stage": False,
            "parallel": True,
            "max_workers": None,
            "no_commit": False,
            "on_error": OnError.FAIL,
            "cache_dir": None,
            "allow_uncached_incremental": False,
            "checkout_missing": False,
        },
        {"type": "cancel_requested"},
    ]
    assert len(events) == 4


def test_engine_state_changed_event() -> None:
    """EngineStateChanged event has required fields."""
    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    assert event["type"] == "engine_state_changed"
    assert event["state"] == types.EngineState.ACTIVE


def test_pipeline_reloaded_event() -> None:
    """PipelineReloaded event has required fields."""
    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages": ["new_stage", "changed_stage"],
        "stages_added": ["new_stage"],
        "stages_removed": ["old_stage"],
        "stages_modified": ["changed_stage"],
        "error": None,
    }
    assert event["type"] == "pipeline_reloaded"
    assert event["stages_added"] == ["new_stage"]
    assert event["error"] is None

    # With error
    event_err: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages": [],
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": "SyntaxError in stages.py",
    }
    assert event_err["error"] == "SyntaxError in stages.py"


def test_stage_started_event() -> None:
    """StageStarted event has required fields."""
    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 3,
        "total": 5,
    }
    assert event["type"] == "stage_started"
    assert event["stage"] == "train"
    assert event["index"] == 3
    assert event["total"] == 5


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
        "input_hash": "abc123",
    }
    assert event["type"] == "stage_completed"
    assert event["status"] == StageStatus.RAN
    assert event["index"] == 3
    assert event["total"] == 5
    assert event["input_hash"] == "abc123"

    # Skipped stage
    event_skip: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "evaluate",
        "status": StageStatus.SKIPPED,
        "reason": "unchanged",
        "duration_ms": 0.0,
        "index": 4,
        "total": 5,
        "input_hash": None,
    }
    assert event_skip["status"] == StageStatus.SKIPPED


def test_log_line_event() -> None:
    """LogLine event has required fields."""
    event: types.LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Epoch 1/10 loss=0.5",
        "is_stderr": False,
    }
    assert event["type"] == "log_line"
    assert event["is_stderr"] is False

    event_err: types.LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Warning: deprecated API",
        "is_stderr": True,
    }
    assert event_err["is_stderr"] is True


def test_stage_state_changed_event() -> None:
    """StageStateChanged event has required fields."""
    event: types.StageStateChanged = {
        "type": "stage_state_changed",
        "stage": "train",
        "state": types.StageExecutionState.RUNNING,
        "previous_state": types.StageExecutionState.PREPARING,
    }
    assert event["type"] == "stage_state_changed"
    assert event["stage"] == "train"
    assert event["state"] == types.StageExecutionState.RUNNING
    assert event["previous_state"] == types.StageExecutionState.PREPARING


def test_output_event_union() -> None:
    """OutputEvent is a union of all output event types."""
    events: list[types.OutputEvent] = [
        {"type": "engine_state_changed", "state": types.EngineState.IDLE},
        {
            "type": "pipeline_reloaded",
            "stages": [],
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
            "input_hash": None,
        },
        {
            "type": "stage_state_changed",
            "stage": "x",
            "state": types.StageExecutionState.RUNNING,
            "previous_state": types.StageExecutionState.PREPARING,
        },
        {"type": "log_line", "stage": "x", "line": "", "is_stderr": False},
    ]
    assert len(events) == 6


async def test_async_event_source_protocol_defined() -> None:
    """EventSource protocol is importable and has run method signature."""
    from pivot.engine.types import EventSource

    # Verify protocol has required method
    assert hasattr(EventSource, "run")


async def test_async_event_sink_protocol_defined() -> None:
    """EventSink protocol is importable and has handle/close signatures."""
    from pivot.engine.types import EventSink

    assert hasattr(EventSink, "handle")
    assert hasattr(EventSink, "close")
