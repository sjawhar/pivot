"""Tests for lock-waiting state messaging and display."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import rich.console

from pivot.engine import sinks, types

if TYPE_CHECKING:
    from pivot.types import OutputMessage, StateMessage

# =============================================================================
# StageExecutionState.WAITING_ON_LOCK ordering
# =============================================================================


def test_waiting_on_lock_exists() -> None:
    """WAITING_ON_LOCK is a valid StageExecutionState member."""
    assert hasattr(types.StageExecutionState, "WAITING_ON_LOCK")


def test_waiting_on_lock_ordering() -> None:
    """WAITING_ON_LOCK sits between PREPARING and RUNNING."""
    assert types.StageExecutionState.PREPARING < types.StageExecutionState.WAITING_ON_LOCK
    assert types.StageExecutionState.WAITING_ON_LOCK < types.StageExecutionState.RUNNING


def test_waiting_on_lock_in_execution_range() -> None:
    """WAITING_ON_LOCK counts as 'execution has begun' (>= PREPARING)."""
    state = types.StageExecutionState.WAITING_ON_LOCK
    assert state >= types.StageExecutionState.PREPARING, "Should be in execution range"
    assert state < types.StageExecutionState.COMPLETED, "Should not be terminal"


# =============================================================================
# StateMessage / OutputMessage type
# =============================================================================


def test_state_message_structure() -> None:
    """StateMessage is a 3-tuple with __state__ sentinel."""
    msg: StateMessage = ("__state__", "train", "WAITING_ON_LOCK")
    assert msg[0] == "__state__"
    assert msg[1] == "train"
    assert msg[2] == "WAITING_ON_LOCK"


def test_state_message_is_valid_output_message() -> None:
    """StateMessage is a valid OutputMessage variant."""
    msg: OutputMessage = ("__state__", "train", "WAITING_ON_LOCK")
    assert msg is not None
    assert msg[0] == "__state__"


def test_output_message_log_line_still_works() -> None:
    """Regular log line OutputMessage is unchanged."""
    msg: OutputMessage = ("train", "Epoch 1/10", False)
    assert msg is not None
    assert msg[0] == "train"
    assert msg[1] == "Epoch 1/10"
    assert msg[2] is False


def test_output_message_none_still_works() -> None:
    """None sentinel OutputMessage is unchanged."""
    msg: OutputMessage = None
    assert msg is None


def test_state_message_distinguishable_from_log_line() -> None:
    """__state__ sentinel distinguishes state messages from log lines."""
    state_msg: OutputMessage = ("__state__", "train", "WAITING_ON_LOCK")
    log_msg: OutputMessage = ("train", "output line", True)

    assert state_msg is not None
    assert state_msg[0] == "__state__"

    assert log_msg is not None
    assert log_msg[0] != "__state__"


# =============================================================================
# ConsoleSink handling of WAITING_ON_LOCK
# =============================================================================


@pytest.mark.anyio
async def test_console_sink_displays_waiting_on_lock() -> None:
    """ConsoleSink prints waiting message for WAITING_ON_LOCK state change."""
    console = rich.console.Console(file=None, force_terminal=False, record=True)
    sink = sinks.ConsoleSink(console=console, show_output=False)

    event: types.StageStateChanged = {
        "type": "stage_state_changed",
        "stage": "train",
        "state": types.StageExecutionState.WAITING_ON_LOCK,
        "previous_state": types.StageExecutionState.PREPARING,
    }
    await sink.handle(event)

    output = console.export_text()
    assert "train" in output, "Should contain stage name"
    assert "waiting on artifact lock" in output, "Should contain lock waiting message"


@pytest.mark.anyio
async def test_console_sink_ignores_other_state_changes() -> None:
    """ConsoleSink does not print for non-WAITING_ON_LOCK state changes."""
    console = rich.console.Console(file=None, force_terminal=False, record=True)
    sink = sinks.ConsoleSink(console=console, show_output=False)

    event: types.StageStateChanged = {
        "type": "stage_state_changed",
        "stage": "train",
        "state": types.StageExecutionState.RUNNING,
        "previous_state": types.StageExecutionState.WAITING_ON_LOCK,
    }
    await sink.handle(event)

    output = console.export_text()
    assert "waiting on artifact lock" not in output, "Should not print for RUNNING state"
