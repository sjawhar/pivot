from __future__ import annotations

import time
from typing import TYPE_CHECKING

import anyio
import anyio.to_thread
import rich.console
import rich.markup

from pivot.engine.types import StageCompleted
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
        StageStarted,
    )
    from pivot.tui.run import MessagePoster

__all__ = [
    "ConsoleSink",
    "ResultCollectorSink",
    "TuiSink",
]


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


class ConsoleSink:
    """Async sink that prints stage events to console."""

    _console: rich.console.Console
    _show_output: bool

    def __init__(self, *, console: rich.console.Console, show_output: bool = False) -> None:
        self._console = console
        self._show_output = show_output

    async def handle(self, event: OutputEvent) -> None:
        """Handle output event by printing to console."""
        match event["type"]:
            case "stage_started":
                self._console.print(f"Running {event['stage']}...")
            case "stage_completed":
                stage = event["stage"]
                duration = event["duration_ms"] / 1000
                match event["status"]:
                    case StageStatus.SKIPPED:
                        self._console.print(f"  {stage}: skipped")
                    case StageStatus.RAN:
                        self._console.print(f"  {stage}: done ({duration:.1f}s)")
                    case StageStatus.FAILED:
                        self._console.print(f"  {stage}: [red]FAILED[/red]")
                        if event["reason"]:
                            # Indent each line of the error for readability
                            # Escape to prevent Rich markup injection from error messages
                            for line in event["reason"].rstrip().split("\n"):
                                self._console.print(f"    [dim]{rich.markup.escape(line)}[/dim]")
            case "log_line" if self._show_output:
                stage = event["stage"]
                # Escape line content to prevent Rich markup injection from stage output
                line = rich.markup.escape(event["line"])
                if event["is_stderr"]:
                    self._console.print(f"[red]\\[{stage}][/red] [red]{line}[/red]")
                else:
                    self._console.print(f"\\[{stage}] {line}")
            case _:
                pass  # Ignore other events

    async def close(self) -> None:
        """No cleanup needed."""


class ResultCollectorSink:
    """Async sink that collects stage results for programmatic access."""

    _results: dict[str, StageCompleted]
    _lock: anyio.Lock

    def __init__(self) -> None:
        self._results = dict[str, StageCompleted]()
        self._lock = anyio.Lock()

    async def handle(self, event: OutputEvent) -> None:
        """Collect stage_completed events."""
        if event["type"] != "stage_completed":
            return

        async with self._lock:
            self._results[event["stage"]] = event

    async def get_results(self) -> dict[str, StageCompleted]:
        """Get collected results. Call after run() completes."""
        async with self._lock:
            return dict(self._results)

    async def close(self) -> None:
        """No cleanup needed."""


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
        from pivot.tui.run import TuiUpdate

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
        from pivot.tui.run import TuiShutdown

        with anyio.move_on_after(1.0):  # 1 second timeout
            await anyio.to_thread.run_sync(lambda: self._app.post_message(TuiShutdown()))
