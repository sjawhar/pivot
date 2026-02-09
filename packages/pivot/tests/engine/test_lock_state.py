"""Tests for WAITING_ON_LOCK execution state and worker→engine communication."""

from __future__ import annotations

from io import StringIO

import rich.console

from pivot.engine.sinks import ConsoleSink
from pivot.engine.types import StageExecutionState, StageStateChanged
from pivot.types import DisplayCategory, LogMessage, OutputMessage, OutputMessageKind, StateChange


def test_waiting_on_lock_state_ordering() -> None:
    """WAITING_ON_LOCK sits between PREPARING and RUNNING in IntEnum ordering."""
    assert StageExecutionState.PREPARING < StageExecutionState.WAITING_ON_LOCK, (
        "WAITING_ON_LOCK should come after PREPARING"
    )
    assert StageExecutionState.WAITING_ON_LOCK < StageExecutionState.RUNNING, (
        "WAITING_ON_LOCK should come before RUNNING"
    )
    # Verify full chain is intact
    assert (
        StageExecutionState.PENDING
        < StageExecutionState.BLOCKED
        < StageExecutionState.READY
        < StageExecutionState.PREPARING
        < StageExecutionState.WAITING_ON_LOCK
        < StageExecutionState.RUNNING
        < StageExecutionState.COMPLETED
    ), "Full state ordering must be maintained"


def test_state_message_output_message_type() -> None:
    """State change messages can be created as OutputMessage TypedDicts."""
    log_msg: OutputMessage = LogMessage(
        kind=OutputMessageKind.LOG, stage="my_stage", line="some output", is_stderr=False
    )
    assert log_msg is not None
    assert log_msg["stage"] == "my_stage"

    state_msg: OutputMessage = StateChange(
        kind=OutputMessageKind.STATE, stage="my_stage", state="waiting_on_lock"
    )
    assert state_msg is not None
    assert state_msg["kind"] == OutputMessageKind.STATE
    assert state_msg["stage"] == "my_stage"
    assert state_msg["state"] == "waiting_on_lock"

    none_msg: OutputMessage = None
    assert none_msg is None


async def test_console_sink_displays_waiting_on_lock() -> None:
    """ConsoleSink prints waiting status for WAITING_ON_LOCK state change."""
    output = StringIO()
    console = rich.console.Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageStateChanged(
        type="stage_state_changed",
        stage="train",
        state=StageExecutionState.WAITING_ON_LOCK,
        previous_state=StageExecutionState.PREPARING,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result, "Stage name should appear in output"
    assert "waiting for artifact lock" in result, "Lock waiting message should appear"


async def test_console_sink_ignores_other_state_changes() -> None:
    """ConsoleSink does not print for non-WAITING_ON_LOCK state changes."""
    output = StringIO()
    console = rich.console.Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageStateChanged(
        type="stage_state_changed",
        stage="train",
        state=StageExecutionState.RUNNING,
        previous_state=StageExecutionState.PREPARING,
    )
    await sink.handle(event)

    assert output.getvalue() == "", "Should not print for RUNNING state change"


def test_display_category_has_waiting_on_lock() -> None:
    """DisplayCategory enum includes WAITING_ON_LOCK value."""
    assert DisplayCategory.WAITING_ON_LOCK == "waiting_on_lock", (
        "DisplayCategory.WAITING_ON_LOCK should have correct value"
    )
