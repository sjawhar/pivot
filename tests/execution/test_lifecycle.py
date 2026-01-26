import queue
import time
from typing import cast

from pivot import registry
from pivot.executor import core as executor_core
from pivot.types import (
    RunJsonEvent,
    StageCompleteEvent,
    StageResult,
    StageStatus,
    TuiMessageType,
    TuiQueue,
    TuiStatusMessage,
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
        tui_queue=None,
        con=None,
        progress_callback=None,
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
        tui_queue=None,
        con=None,
        progress_callback=None,
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
        tui_queue=None,
        con=None,
        progress_callback=None,
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
        tui_queue=None,
        con=None,
        progress_callback=None,
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


def test_lifecycle_mark_started_sends_tui_message() -> None:
    """mark_started sends TUI status message when queue is provided."""
    # TUI queue uses stdlib queue.Queue (inter-thread, not cross-process)
    tui_queue: TuiQueue = queue.Queue()

    lifecycle = executor_core.StageLifecycle(
        tui_queue=tui_queue,
        con=None,
        progress_callback=None,
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1", index=1)

    lifecycle.mark_started(state)

    # Check TUI message was sent - cast to TuiStatusMessage for type safety
    msg = cast("TuiStatusMessage", tui_queue.get(timeout=1.0))
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "stage1"
    assert msg["status"] == StageStatus.IN_PROGRESS
    assert msg["run_id"] == "test_run_123"


def test_lifecycle_mark_skipped_upstream_sends_tui_message() -> None:
    """mark_skipped_upstream sends TUI status message (critical for history bug fix)."""
    tui_queue: TuiQueue = queue.Queue()

    lifecycle = executor_core.StageLifecycle(
        tui_queue=tui_queue,
        con=None,
        progress_callback=None,
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("downstream_stage", index=2)

    lifecycle.mark_skipped_upstream(state, "upstream_stage")

    # Check TUI message was sent - cast to TuiStatusMessage for type safety
    msg = cast("TuiStatusMessage", tui_queue.get(timeout=1.0))
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "downstream_stage"
    assert msg["status"] == StageStatus.SKIPPED
    assert "upstream 'upstream_stage' failed" in msg["reason"]
    assert msg["run_id"] == "test_run_123"


def test_lifecycle_mark_completed_sends_tui_message() -> None:
    """mark_completed sends TUI status message."""
    tui_queue: TuiQueue = queue.Queue()

    lifecycle = executor_core.StageLifecycle(
        tui_queue=tui_queue,
        con=None,
        progress_callback=None,
        total_stages=3,
        run_id="test_run_123",
    )
    state = _make_stage_state("stage1", index=1)
    state.start_time = time.perf_counter() - 1.0

    result = StageResult(status=StageStatus.RAN, reason="success", output_lines=[])
    lifecycle.mark_completed(state, result)

    msg = cast("TuiStatusMessage", tui_queue.get(timeout=1.0))
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "stage1"
    assert msg["status"] == StageStatus.RAN
    assert msg["reason"] == "success"
    assert msg["run_id"] == "test_run_123"
    assert msg["elapsed"] is not None


def test_lifecycle_calls_progress_callback() -> None:
    """mark_started and mark_completed call progress_callback when provided."""
    events = list[RunJsonEvent]()

    def callback(event: RunJsonEvent) -> None:
        events.append(event)

    lifecycle = executor_core.StageLifecycle(
        tui_queue=None,
        con=None,
        progress_callback=callback,
        total_stages=3,
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
    # Cast to StageCompleteEvent to access status field (type narrowing on discriminant not automatic)
    complete_event = cast("StageCompleteEvent", events[1])
    assert complete_event["status"] == StageStatus.RAN


# =============================================================================
# Integration Tests: _handle_stage_failure with lifecycle
# =============================================================================


def test_handle_stage_failure_marks_downstream_with_notifications() -> None:
    """_handle_stage_failure marks downstream stages as SKIPPED with TUI notifications."""
    tui_queue: TuiQueue = queue.Queue()

    lifecycle = executor_core.StageLifecycle(
        tui_queue=tui_queue,
        con=None,
        progress_callback=None,
        total_stages=3,
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

    # Check that TUI messages were sent for skipped stages
    messages = list[TuiStatusMessage]()
    while not tui_queue.empty():
        msg = tui_queue.get_nowait()
        messages.append(cast("TuiStatusMessage", msg))

    assert len(messages) == 2, "Should have sent 2 TUI messages (for B and C)"

    skipped_stages = {m["stage"] for m in messages}
    assert skipped_stages == {"stage_b", "stage_c"}

    for msg in messages:
        assert msg["status"] == StageStatus.SKIPPED
        assert msg["run_id"] == "test_run_123"


def test_handle_stage_failure_without_lifecycle_still_works() -> None:
    """_handle_stage_failure works without lifecycle (no TUI notifications)."""
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


def test_lifecycle_mark_skipped_upstream_console_shows_correct_index() -> None:
    """mark_skipped_upstream shows the stage's correct index in console output, not 0."""
    import io

    from pivot.tui import console

    # Capture console output
    output = io.StringIO()
    con = console.Console(stream=output, color=False)

    lifecycle = executor_core.StageLifecycle(
        tui_queue=None,
        con=con,
        progress_callback=None,
        total_stages=10,
        run_id="test_run",
    )
    state = _make_stage_state("downstream_stage", index=5)

    lifecycle.mark_skipped_upstream(state, "upstream_stage")

    console_output = output.getvalue()
    # The bug: console shows [0/10] instead of [5/10]
    # Fixed: console should show the stage's actual index
    assert "[5/10]" in console_output, f"Expected [5/10] in output, got: {console_output}"
    assert "[0/10]" not in console_output, f"Bug: [0/10] found in output: {console_output}"
