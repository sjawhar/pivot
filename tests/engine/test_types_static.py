"""Static type tests for engine types (validated by type checker, not pytest)."""

from pivot.engine.types import PipelineReloaded, StageCompleted  # noqa: TC001
from pivot.types import CompletionType, StageStatus


def test_stage_completed_status_is_completion_type() -> None:
    """StageCompleted.status should only accept CompletionType values."""
    event: StageCompleted = {
        "type": "stage_completed",
        "stage": "test",
        "status": StageStatus.RAN,  # Valid
        "reason": "success",
        "duration_ms": 100.0,
        "index": 1,
        "total": 1,
    }
    # This assignment validates the type
    _status: CompletionType = event["status"]
    assert _status == StageStatus.RAN


def test_pipeline_reloaded_has_stages_field() -> None:
    """PipelineReloaded should have a stages field with sorted stage list."""
    event: PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages": ["stage_a", "stage_b"],  # Topologically sorted
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": None,
    }
    assert event["stages"] == ["stage_a", "stage_b"]
