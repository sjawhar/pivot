import time
from typing import cast

from pivot import registry
from pivot.executor import core as executor_core
from pivot.types import (
    RunJsonEvent,
    StageCompleteEvent,
    StageResult,
    StageStartEvent,
    StageStatus,
)

# =============================================================================
# StageLifecycle Unit Tests
# =============================================================================


def _make_stage_state(name: str, index: int = 1) -> executor_core.StageState:
    """Create a minimal StageState for testing."""
    return executor_core.StageState(
        name=name,
        index=index,
        info=registry.RegistryStageInfo(
            name=name,
            func=lambda: None,
            deps={},
            deps_paths=[],
            outs=[],
            outs_paths=[],
            fingerprint={"main": "test_fingerprint"},
            params=None,
            variant=None,
            mutex=[],
            signature=None,
            dep_specs={},
            out_specs={},
            params_arg_name=None,
        ),
        upstream=[],
        upstream_unfinished=set(),
        downstream=[],
        mutex=[],
    )


def test_lifecycle_mark_started_updates_state() -> None:
    """mark_started sets status and start_time on the state."""
    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1")

    lifecycle.mark_started(state)

    assert state.status == StageStatus.IN_PROGRESS
    assert state.start_time is not None
    assert state.start_time > 0


def test_lifecycle_mark_completed_updates_state() -> None:
    """mark_completed sets result, status, and end_time on the state."""
    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1")
    state.start_time = time.perf_counter() - 1.0  # Started 1s ago

    result = StageResult(status=StageStatus.RAN, reason="completed", output_lines=[])
    lifecycle.mark_completed(state, result)

    assert state.status == StageStatus.RAN
    assert state.result == result
    assert state.end_time is not None


def test_lifecycle_mark_failed_updates_state() -> None:
    """mark_failed creates a FAILED result and updates state."""
    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1")
    state.start_time = time.perf_counter() - 1.0

    lifecycle.mark_failed(state, "some error")

    assert state.status == StageStatus.FAILED
    assert state.result is not None
    assert state.result["status"] == StageStatus.FAILED
    assert state.result["reason"] == "some error"
    assert state.end_time is not None


def test_lifecycle_mark_skipped_upstream_updates_state() -> None:
    """mark_skipped_upstream marks stage as SKIPPED with upstream failure reason."""
    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage2")

    lifecycle.mark_skipped_upstream(state, "stage1")

    assert state.status == StageStatus.SKIPPED
    assert state.result is not None
    assert state.result["status"] == StageStatus.SKIPPED
    assert "upstream 'stage1' failed" in state.result["reason"]
    # end_time should NOT be set for skipped stages (they never started)
    assert state.end_time is None


def test_lifecycle_mark_started_calls_progress_callback() -> None:
    """mark_started calls progress_callback when provided."""
    events = list[RunJsonEvent]()

    def callback(event: RunJsonEvent) -> None:
        events.append(event)

    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        progress_callback=callback,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1", index=1)

    lifecycle.mark_started(state)

    assert len(events) == 1
    event = cast("StageStartEvent", events[0])
    assert event["type"] == "stage_start"
    assert event["stage"] == "stage1"
    assert event["index"] == 1
    assert event["total"] == 3


def test_lifecycle_mark_completed_calls_progress_callback() -> None:
    """mark_completed calls progress_callback when provided."""
    events = list[RunJsonEvent]()

    def callback(event: RunJsonEvent) -> None:
        events.append(event)

    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        progress_callback=callback,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1", index=1)
    state.start_time = time.perf_counter() - 1.0

    result = StageResult(status=StageStatus.RAN, reason="done", output_lines=[])
    lifecycle.mark_completed(state, result)

    assert len(events) == 1
    event = cast("StageCompleteEvent", events[0])
    assert event["type"] == "stage_complete"
    assert event["stage"] == "stage1"
    assert event["status"] == StageStatus.RAN
    assert event["reason"] == "done"
    assert event["duration_ms"] > 0


def test_lifecycle_mark_skipped_upstream_calls_progress_callback() -> None:
    """mark_skipped_upstream calls progress_callback when provided."""
    events = list[RunJsonEvent]()

    def callback(event: RunJsonEvent) -> None:
        events.append(event)

    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        progress_callback=callback,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage2", index=2)

    lifecycle.mark_skipped_upstream(state, "stage1")

    assert len(events) == 1
    event = cast("StageCompleteEvent", events[0])
    assert event["type"] == "stage_complete"
    assert event["stage"] == "stage2"
    assert event["status"] == StageStatus.SKIPPED
    assert "upstream 'stage1' failed" in event["reason"]


def test_lifecycle_calls_progress_callback_for_full_flow() -> None:
    """mark_started and mark_completed call progress_callback for full stage lifecycle."""
    events = list[RunJsonEvent]()

    def callback(event: RunJsonEvent) -> None:
        events.append(event)

    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        progress_callback=callback,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1", index=1)

    lifecycle.mark_started(state)

    assert len(events) == 1
    assert events[0]["type"] == "stage_start"
    assert events[0]["stage"] == "stage1"

    result = StageResult(status=StageStatus.RAN, reason="done", output_lines=[])
    lifecycle.mark_completed(state, result)

    assert len(events) == 2
    assert events[1]["type"] == "stage_complete"
    assert events[1]["stage"] == "stage1"
    complete_event = cast("StageCompleteEvent", events[1])
    assert complete_event["status"] == StageStatus.RAN


# =============================================================================
# Integration Tests: _handle_stage_failure with lifecycle
# =============================================================================


def test_handle_stage_failure_marks_downstream_with_progress_callback() -> None:
    """_handle_stage_failure marks downstream stages as SKIPPED and calls progress_callback."""
    events = list[RunJsonEvent]()

    def callback(event: RunJsonEvent) -> None:
        events.append(event)

    lifecycle = executor_core.StageLifecycle(
        total_stages=3,
        progress_callback=callback,
        run_id="test_run_123",
    )

    # Create a simple A -> B -> C pipeline where A fails
    stage_a = _make_stage_state("stage_a", index=1)
    stage_b = _make_stage_state("stage_b", index=2)
    stage_c = _make_stage_state("stage_c", index=3)

    # Set up the downstream relationships
    stage_a.downstream = ["stage_b"]
    stage_b.downstream = ["stage_c"]

    stage_states = {
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_c": stage_c,
    }

    # Mark A as failed and handle downstream
    executor_core._handle_stage_failure("stage_a", stage_states, lifecycle)

    # B and C should be marked as SKIPPED
    assert stage_b.status == StageStatus.SKIPPED
    assert stage_c.status == StageStatus.SKIPPED
    assert stage_b.result is not None
    assert "upstream 'stage_a' failed" in stage_b.result["reason"]
    assert stage_c.result is not None
    assert "upstream 'stage_a' failed" in stage_c.result["reason"]

    # Check that progress_callback was called for skipped stages
    assert len(events) == 2, "Should have sent 2 progress events (for B and C)"

    skipped_stages = {cast("StageCompleteEvent", e)["stage"] for e in events}
    assert skipped_stages == {"stage_b", "stage_c"}

    for event in events:
        complete_event = cast("StageCompleteEvent", event)
        assert complete_event["status"] == StageStatus.SKIPPED


def test_handle_stage_failure_without_lifecycle_still_works() -> None:
    """_handle_stage_failure works without lifecycle (no notifications)."""
    stage_a = _make_stage_state("stage_a", index=1)
    stage_b = _make_stage_state("stage_b", index=2)
    stage_a.downstream = ["stage_b"]

    stage_states = {"stage_a": stage_a, "stage_b": stage_b}

    # Call without lifecycle
    executor_core._handle_stage_failure("stage_a", stage_states, lifecycle=None)

    # B should still be marked as SKIPPED
    assert stage_b.status == StageStatus.SKIPPED
    assert stage_b.result is not None
    assert "upstream 'stage_a' failed" in stage_b.result["reason"]
