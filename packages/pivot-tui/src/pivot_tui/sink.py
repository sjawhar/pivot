"""TUI event sink for Pivot engine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread

from pivot.types import (
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiStatusMessage,
)

if TYPE_CHECKING:
    from pivot.engine.types import (
        LogLine,
        OutputEvent,
        StageCompleted,
        StageStarted,
    )
    from pivot_tui.run import MessagePoster

__all__ = ["TuiSink"]


def _make_started_message(event: StageStarted, run_id: str) -> TuiStatusMessage:
    """Build TuiStatusMessage for stage_started event."""
    return TuiStatusMessage(
        type=TuiMessageType.STATUS,
        stage=event["stage"],
        index=event["index"],
        total=event["total"],
        status=StageStatus.IN_PROGRESS,
        reason="",
        elapsed=None,
        run_id=run_id,
    )


def _make_completed_message(event: StageCompleted, run_id: str) -> TuiStatusMessage:
    """Build TuiStatusMessage for stage_completed event."""
    return TuiStatusMessage(
        type=TuiMessageType.STATUS,
        stage=event["stage"],
        index=event["index"],
        total=event["total"],
        status=event["status"],
        reason=event["reason"],
        elapsed=event["duration_ms"] / 1000.0,
        run_id=run_id,
    )


def _make_log_message(event: LogLine) -> TuiLogMessage:
    """Build TuiLogMessage for log_line event."""
    return TuiLogMessage(
        type=TuiMessageType.LOG,
        stage=event["stage"],
        line=event["line"],
        is_stderr=event["is_stderr"],
        timestamp=time.time(),
    )


class TuiSink:
    """Async sink that forwards events directly to TUI via post_message.

    Uses Textual's thread-safe post_message() to send events directly
    to the TUI app without an intermediate queue.

    Note: Textual's internal message queue is unbounded. For TUI display
    this is acceptable as message rates are bounded by stage execution speed.
    """

    _app: MessagePoster
    _run_id: str

    def __init__(self, *, app: MessagePoster, run_id: str) -> None:
        self._app = app
        self._run_id = run_id

    async def handle(self, event: OutputEvent) -> None:
        """Convert event to TUI message and post directly to app.

        post_message is thread-safe per Textual docs, so this can be
        called from any thread/async context.
        """
        from pivot_tui.run import TuiUpdate

        match event["type"]:
            case "stage_started":
                self._app.post_message(TuiUpdate(_make_started_message(event, self._run_id)))
            case "stage_completed":
                self._app.post_message(TuiUpdate(_make_completed_message(event, self._run_id)))
            case "log_line":
                self._app.post_message(TuiUpdate(_make_log_message(event)))
            case _:
                pass  # Ignore engine_state_changed, pipeline_reloaded, etc.

    async def close(self) -> None:
        """Signal TUI that sink is closing.

        Uses a brief timeout to avoid blocking the event loop if TUI is unresponsive.
        Textual's post_message is thread-safe but we run it via to_thread.run_sync
        with a timeout to ensure the engine can shut down cleanly.
        """
        from pivot_tui.run import TuiShutdown

        with anyio.move_on_after(1.0):  # 1 second timeout
            await anyio.to_thread.run_sync(lambda: self._app.post_message(TuiShutdown()))
