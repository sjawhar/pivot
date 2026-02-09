"""Tests for engine type definitions."""

from __future__ import annotations

from pivot.engine import types
from pivot.types import ChangeType, OnError, StageExplanation, StageStatus


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
        "seq": 0,
        "state": types.EngineState.ACTIVE,
    }
    assert event["type"] == "engine_state_changed"
    assert event["state"] == types.EngineState.ACTIVE


def test_pipeline_reloaded_event() -> None:
    """PipelineReloaded event has required fields."""
    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "seq": 0,
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
        "seq": 1,
        "stages": [],
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": "SyntaxError in stages.py",
    }
    assert event_err["error"] == "SyntaxError in stages.py"


def test_stage_started_event() -> None:
    """StageStarted event has required fields."""
    event: types.StageStarted = types.StageStarted(
        type="stage_started",
        seq=0,
        stage="train",
        index=3,
        total=5,
    )
    assert event["type"] == "stage_started"
    assert event["stage"] == "train"
    assert event["index"] == 3
    assert event["total"] == 5


def test_stage_started_without_explanation_is_valid() -> None:
    """StageStarted without explanation field is backward-compatible."""
    event: types.StageStarted = types.StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=1,
    )
    assert "explanation" not in event, "explanation should not be present when omitted"


def test_stage_started_with_explanation_none() -> None:
    """StageStarted accepts explanation=None."""
    event: types.StageStarted = types.StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=1,
        explanation=None,
    )
    assert "explanation" in event, "explanation should be present when set to None"
    assert event["explanation"] is None


def test_stage_started_with_explanation_data() -> None:
    """StageStarted carries a full StageExplanation when provided."""
    explanation = StageExplanation(
        stage_name="train",
        will_run=True,
        is_forced=False,
        reason="Code changed",
        code_changes=[
            {
                "key": "train",
                "old_hash": "aaa",
                "new_hash": "bbb",
                "change_type": ChangeType.MODIFIED,
            }
        ],
        param_changes=[],
        dep_changes=[],
        upstream_stale=[],
    )
    event: types.StageStarted = types.StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=1,
        explanation=explanation,
    )
    assert "explanation" in event, "explanation should be present when provided"
    explanation_value = event["explanation"]
    assert explanation_value is not None
    assert explanation_value["will_run"] is True
    assert explanation_value["reason"] == "Code changed"
    assert len(explanation_value["code_changes"]) == 1


def test_stage_completed_event() -> None:
    """StageCompleted event has required fields."""
    event: types.StageCompleted = types.StageCompleted(
        type="stage_completed",
        seq=0,
        stage="train",
        status=StageStatus.RAN,
        reason="inputs changed",
        duration_ms=1234.5,
        index=3,
        total=5,
        input_hash="abc123",
    )
    assert event["type"] == "stage_completed"
    assert event["status"] == StageStatus.RAN
    assert event["index"] == 3
    assert event["total"] == 5
    assert event["input_hash"] == "abc123"

    # Skipped stage
    event_skip: types.StageCompleted = types.StageCompleted(
        type="stage_completed",
        seq=1,
        stage="evaluate",
        status=StageStatus.SKIPPED,
        reason="unchanged",
        duration_ms=0.0,
        index=4,
        total=5,
        input_hash=None,
    )
    assert event_skip["status"] == StageStatus.SKIPPED


def test_stage_completed_without_output_summary_is_valid() -> None:
    """StageCompleted without output_summary field is backward-compatible."""
    event: types.StageCompleted = types.StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.RAN,
        reason="inputs changed",
        duration_ms=100.0,
        index=0,
        total=1,
        input_hash="abc123",
    )
    assert "output_summary" not in event, "output_summary should not be present when omitted"


def test_stage_completed_with_output_summary_none() -> None:
    """StageCompleted accepts output_summary=None (e.g., skipped stages)."""
    event: types.StageCompleted = types.StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.SKIPPED,
        reason="unchanged",
        duration_ms=0.0,
        index=0,
        total=1,
        input_hash=None,
        output_summary=None,
    )
    assert "output_summary" in event, "output_summary should be present when set to None"
    assert event["output_summary"] is None


def test_stage_completed_with_populated_output_summary() -> None:
    """StageCompleted carries output change summaries when provided."""
    summary = [
        types.OutputChangeSummary(
            path="output.csv",
            change_type=None,
            output_type="out",
            old_hash=None,
            new_hash="def456",
        ),
        types.OutputChangeSummary(
            path="metrics.json",
            change_type=None,
            output_type="metric",
            old_hash=None,
            new_hash="ghi789",
        ),
    ]
    event: types.StageCompleted = types.StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.RAN,
        reason="inputs changed",
        duration_ms=1500.0,
        index=0,
        total=1,
        input_hash="abc123",
        output_summary=summary,
    )
    assert "output_summary" in event, "output_summary should be present when provided"
    output_summary = event["output_summary"]
    assert output_summary is not None
    assert len(output_summary) == 2
    assert output_summary[0]["path"] == "output.csv"
    assert output_summary[0]["output_type"] == "out"
    assert output_summary[0]["new_hash"] == "def456"
    assert output_summary[1]["output_type"] == "metric"


def test_output_change_summary_typeddict() -> None:
    """OutputChangeSummary has all required fields."""
    summary = types.OutputChangeSummary(
        path="data/output.csv",
        change_type="modified",
        output_type="out",
        old_hash="aaa111",
        new_hash="bbb222",
    )
    assert summary["path"] == "data/output.csv"
    assert summary["change_type"] == "modified"
    assert summary["output_type"] == "out"
    assert summary["old_hash"] == "aaa111"
    assert summary["new_hash"] == "bbb222"

    # With None values (no old hash, unknown change type)
    summary_new = types.OutputChangeSummary(
        path="new_output.csv",
        change_type=None,
        output_type="plot",
        old_hash=None,
        new_hash="ccc333",
    )
    assert summary_new["change_type"] is None
    assert summary_new["old_hash"] is None


def test_log_line_event() -> None:
    """LogLine event has required fields."""
    event: types.LogLine = {
        "type": "log_line",
        "seq": 0,
        "stage": "train",
        "line": "Epoch 1/10 loss=0.5",
        "is_stderr": False,
    }
    assert event["type"] == "log_line"
    assert event["is_stderr"] is False

    event_err: types.LogLine = {
        "type": "log_line",
        "seq": 1,
        "stage": "train",
        "line": "Warning: deprecated API",
        "is_stderr": True,
    }
    assert event_err["is_stderr"] is True


def test_stage_state_changed_event() -> None:
    """StageStateChanged event has required fields."""
    event: types.StageStateChanged = {
        "type": "stage_state_changed",
        "seq": 0,
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
        {"type": "engine_state_changed", "seq": 0, "state": types.EngineState.IDLE},
        {
            "type": "pipeline_reloaded",
            "seq": 1,
            "stages": [],
            "stages_added": [],
            "stages_removed": [],
            "stages_modified": [],
            "error": None,
        },
        types.StageStarted(type="stage_started", seq=2, stage="x", index=1, total=1),
        types.StageCompleted(
            type="stage_completed",
            seq=3,
            stage="x",
            status=StageStatus.RAN,
            reason="",
            duration_ms=0,
            index=1,
            total=1,
            input_hash=None,
        ),
        {
            "type": "stage_state_changed",
            "seq": 4,
            "stage": "x",
            "state": types.StageExecutionState.RUNNING,
            "previous_state": types.StageExecutionState.PREPARING,
        },
        {"type": "log_line", "seq": 5, "stage": "x", "line": "", "is_stderr": False},
        {
            "type": "sink_state_changed",
            "seq": 6,
            "sink_id": "ConsoleSink",
            "state": types.SinkState.ENABLED,
            "reason": "manual",
            "failure_count": 0,
            "backoff_s": None,
        },
    ]
    assert len(events) == 7


def test_output_events_define_seq_field() -> None:
    assert "seq" in types.EngineStateChanged.__annotations__
    assert "seq" in types.PipelineReloaded.__annotations__
    assert "seq" in types.StageStarted.__annotations__
    assert "seq" in types.StageCompleted.__annotations__
    assert "seq" in types.StageStateChanged.__annotations__
    assert "seq" in types.LogLine.__annotations__


def test_output_events_define_run_id_field() -> None:
    assert "run_id" in types.EngineStateChanged.__annotations__
    assert "run_id" in types.PipelineReloaded.__annotations__
    assert "run_id" in types.StageStarted.__annotations__
    assert "run_id" in types.StageCompleted.__annotations__
    assert "run_id" in types.StageStateChanged.__annotations__
    assert "run_id" in types.LogLine.__annotations__
    assert "run_id" in types.SinkStateChanged.__annotations__


def test_sink_state_enum() -> None:
    assert types.SinkState.ENABLED.value == "enabled"
    assert types.SinkState.DISABLED.value == "disabled"


def test_sink_state_changed_event() -> None:
    event: types.SinkStateChanged = {
        "type": "sink_state_changed",
        "seq": 1,
        "sink_id": "ConsoleSink",
        "state": types.SinkState.DISABLED,
        "reason": "exception",
        "failure_count": 5,
        "backoff_s": 1.0,
    }
    assert event["state"] == types.SinkState.DISABLED


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
