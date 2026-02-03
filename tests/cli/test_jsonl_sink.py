from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot.cli._run_common import JsonlSink
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    LogLine,
    PipelineReloaded,
    StageCompleted,
    StageStarted,
)
from pivot.types import CompletionType, StageStatus

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# JsonlSink Event Handling Tests
# =============================================================================


@pytest.fixture
def collected_events() -> list[dict[str, object]]:
    """Fixture to collect events passed to callback."""
    return list[dict[str, object]]()


@pytest.fixture
def callback(collected_events: list[dict[str, object]]) -> Callable[[dict[str, object]], None]:
    """Fixture for callback that collects events."""

    def _callback(event: dict[str, object]) -> None:
        collected_events.append(event)

    return _callback


@pytest.mark.anyio
async def test_jsonl_sink_converts_stage_started_to_stage_start(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink converts engine 'stage_started' to JSONL 'stage_start' type."""
    sink = JsonlSink(callback)

    event = StageStarted(type="stage_started", stage="my_stage", index=0, total=3)
    await sink.handle(event)

    assert len(collected_events) == 1
    # Engine uses 'stage_started', but JSONL schema uses 'stage_start'
    assert collected_events[0]["type"] == "stage_start"
    assert collected_events[0]["stage"] == "my_stage"
    assert collected_events[0]["index"] == 0
    assert collected_events[0]["total"] == 3


@pytest.mark.anyio
async def test_jsonl_sink_converts_stage_completed_to_stage_complete(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink converts engine 'stage_completed' to JSONL 'stage_complete' type."""
    sink = JsonlSink(callback)

    event = StageCompleted(
        type="stage_completed",
        stage="my_stage",
        status=StageStatus.RAN,
        reason="executed",
        duration_ms=1234.5,
        index=1,
        total=3,
    )
    await sink.handle(event)

    assert len(collected_events) == 1
    # Engine uses 'stage_completed', but JSONL schema uses 'stage_complete'
    assert collected_events[0]["type"] == "stage_complete"
    assert collected_events[0]["status"] == "ran"  # Enum converted to string


@pytest.mark.anyio
async def test_jsonl_sink_passes_pipeline_reloaded_event(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink passes pipeline_reloaded events to callback."""
    sink = JsonlSink(callback)

    event = PipelineReloaded(
        type="pipeline_reloaded",
        stages=["stage_a", "stage_b", "stage_c"],
        stages_added=["stage_c"],
        stages_removed=[],
        stages_modified=["stage_a"],
        error=None,
    )
    await sink.handle(event)

    assert len(collected_events) == 1
    assert collected_events[0]["type"] == "pipeline_reloaded"
    assert collected_events[0]["stages"] == ["stage_a", "stage_b", "stage_c"]
    assert collected_events[0]["stages_added"] == ["stage_c"]
    assert collected_events[0]["stages_removed"] == []
    assert collected_events[0]["stages_modified"] == ["stage_a"]
    assert collected_events[0]["error"] is None


@pytest.mark.anyio
async def test_jsonl_sink_passes_pipeline_reloaded_with_error(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink passes pipeline_reloaded events with errors to callback."""
    sink = JsonlSink(callback)

    event = PipelineReloaded(
        type="pipeline_reloaded",
        stages=[],
        stages_added=[],
        stages_removed=[],
        stages_modified=[],
        error="SyntaxError in pipeline.py",
    )
    await sink.handle(event)

    assert len(collected_events) == 1
    assert collected_events[0]["type"] == "pipeline_reloaded"
    assert collected_events[0]["error"] == "SyntaxError in pipeline.py"


@pytest.mark.anyio
async def test_jsonl_sink_converts_engine_state_changed_enum_to_string(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink converts state enum to string for engine_state_changed."""
    sink = JsonlSink(callback)

    event = EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE)
    await sink.handle(event)

    assert len(collected_events) == 1
    assert collected_events[0]["type"] == "engine_state_changed"
    assert collected_events[0]["state"] == "active"  # Enum converted to string


@pytest.mark.anyio
async def test_jsonl_sink_handles_engine_state_idle(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink handles engine_state_changed with IDLE state."""
    sink = JsonlSink(callback)

    event = EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE)
    await sink.handle(event)

    assert len(collected_events) == 1
    assert collected_events[0]["state"] == "idle"


@pytest.mark.anyio
async def test_jsonl_sink_ignores_log_line_events(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
) -> None:
    """JsonlSink ignores log_line events."""
    sink = JsonlSink(callback)

    event = LogLine(type="log_line", stage="my_stage", line="some output", is_stderr=False)
    await sink.handle(event)

    assert len(collected_events) == 0


@pytest.mark.anyio
async def test_jsonl_sink_close_is_noop(
    callback: Callable[[dict[str, object]], None],
) -> None:
    """JsonlSink.close() completes without error."""
    sink = JsonlSink(callback)
    await sink.close()  # Should not raise


# =============================================================================
# JsonlSink StageStatus Coverage Tests
# =============================================================================


@pytest.mark.parametrize(
    ("status", "reason", "duration"),
    [
        pytest.param(StageStatus.SKIPPED, "unchanged", 0.0, id="skipped"),
        pytest.param(StageStatus.FAILED, "execution error", 1234.5, id="failed"),
        pytest.param(StageStatus.RAN, "executed", 100.0, id="ran"),
        pytest.param(StageStatus.RAN, "executed", 0.0, id="instant_completion"),
        pytest.param(StageStatus.RAN, "executed", 999999.9, id="long_duration"),
    ],
)
@pytest.mark.anyio
async def test_jsonl_sink_handles_all_stage_statuses(
    callback: Callable[[dict[str, object]], None],
    collected_events: list[dict[str, object]],
    status: CompletionType,
    reason: str,
    duration: float,
) -> None:
    """JsonlSink correctly converts all StageStatus values to strings."""
    sink = JsonlSink(callback)

    event = StageCompleted(
        type="stage_completed",
        stage="test_stage",
        status=status,
        reason=reason,
        duration_ms=duration,
        index=0,
        total=1,
    )
    await sink.handle(event)

    assert len(collected_events) == 1
    assert collected_events[0]["type"] == "stage_complete"
    assert collected_events[0]["status"] == status.value
    assert collected_events[0]["reason"] == reason
    assert collected_events[0]["duration_ms"] == duration
