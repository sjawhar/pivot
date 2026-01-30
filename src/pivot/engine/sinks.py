"""Event sink implementations for the engine."""

from __future__ import annotations

import contextlib
import queue
import time
from typing import TYPE_CHECKING

from pivot.engine.types import EngineState
from pivot.types import (
    StageDisplayStatus,
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.engine.types import (
        EngineStateChanged,
        LogLine,
        OutputEvent,
        PipelineReloaded,
        StageCompleted,
        StageStarted,
    )
    from pivot.tui.console import Console
    from pivot.types import TuiQueue

__all__ = ["ConsoleSink", "JsonlSink", "TuiSink", "WatchSink"]


class ConsoleSink:
    """Event sink that outputs to console with colors and progress tracking."""

    _console: Console

    def __init__(self, console: Console) -> None:
        """Initialize with a Console instance."""
        self._console = console

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by writing to console."""
        match event["type"]:
            case "stage_started":
                self._handle_stage_started(event)
            case "stage_completed":
                self._handle_stage_completed(event)
            case _:
                pass  # Ignore other event types

    def _handle_stage_started(self, event: StageStarted) -> None:
        """Handle stage started event."""
        self._console.stage_start(
            name=event["stage"],
            index=event["index"],
            total=event["total"],
            status=StageDisplayStatus.RUNNING,
        )

    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        self._console.stage_result(
            name=event["stage"],
            index=event["index"],
            total=event["total"],
            status=event["status"],
            reason=event["reason"],
            duration=event["duration_ms"] / 1000.0,
        )

    def close(self) -> None:
        """Close the underlying console."""
        self._console.close()


class TuiSink:
    """Event sink that forwards events to TUI via queue."""

    _queue: TuiQueue
    _run_id: str

    def __init__(self, tui_queue: TuiQueue, run_id: str) -> None:
        """Initialize with a TUI message queue."""
        self._queue = tui_queue
        self._run_id = run_id

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by sending to TUI queue."""
        match event["type"]:
            case "stage_started":
                self._handle_stage_started(event)
            case "stage_completed":
                self._handle_stage_completed(event)
            case "log_line":
                self._handle_log_line(event)
            case _:
                pass

    def _handle_stage_started(self, event: StageStarted) -> None:
        """Handle stage started event."""
        msg = TuiStatusMessage(
            type=TuiMessageType.STATUS,
            stage=event["stage"],
            index=event["index"],
            total=event["total"],
            status=StageStatus.READY,
            reason="",
            elapsed=None,
            run_id=self._run_id,
        )
        self._queue.put(msg)

    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        msg = TuiStatusMessage(
            type=TuiMessageType.STATUS,
            stage=event["stage"],
            index=event["index"],
            total=event["total"],
            status=event["status"],
            reason=event["reason"],
            elapsed=event["duration_ms"] / 1000.0,
            run_id=self._run_id,
        )
        self._queue.put(msg)

    def _handle_log_line(self, event: LogLine) -> None:
        """Handle log line event."""
        msg = TuiLogMessage(
            type=TuiMessageType.LOG,
            stage=event["stage"],
            line=event["line"],
            is_stderr=event["is_stderr"],
            timestamp=time.time(),
        )
        self._queue.put(msg)

    def close(self) -> None:
        """Signal TUI termination by sending None."""
        self._queue.put(None)


class JsonlSink:
    """Event sink that emits JSONL events via callback.

    Translates engine events to the existing RunJsonEvent format for
    backwards compatibility with --json output.
    """

    _callback: Callable[[dict[str, object]], None]

    def __init__(self, callback: Callable[[dict[str, object]], None]) -> None:
        """Initialize with a callback that receives event dicts."""
        self._callback = callback

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by converting and calling callback."""
        match event["type"]:
            case "stage_started":
                self._handle_stage_started(event)
            case "stage_completed":
                self._handle_stage_completed(event)
            case _:
                pass

    def _handle_stage_started(self, event: StageStarted) -> None:
        """Handle stage started event."""
        json_event: dict[str, object] = {
            "type": "stage_start",
            "stage": event["stage"],
            "index": event["index"],
            "total": event["total"],
        }
        self._callback(json_event)

    def _handle_stage_completed(self, event: StageCompleted) -> None:
        """Handle stage completed event."""
        json_event: dict[str, object] = {
            "type": "stage_complete",
            "stage": event["stage"],
            "status": event["status"].value,  # Convert enum to string
            "reason": event["reason"],
            "duration_ms": event["duration_ms"],
            "index": event["index"],
            "total": event["total"],
        }
        self._callback(json_event)

    def close(self) -> None:
        """No cleanup needed for callback-based sink."""
        pass


class WatchSink:
    """Event sink for watch mode that sends status updates to TUI.

    Translates engine state changes and pipeline reloads into TuiWatchMessage
    and TuiReloadMessage for the TUI to display watch mode status.
    """

    _queue: TuiQueue

    def __init__(self, tui_queue: TuiQueue) -> None:
        """Initialize with a TUI message queue."""
        self._queue = tui_queue

    def handle(self, event: OutputEvent) -> None:
        """Handle an output event by sending watch status to TUI queue."""
        match event["type"]:
            case "engine_state_changed":
                self._handle_engine_state_changed(event)
            case "pipeline_reloaded":
                self._handle_pipeline_reloaded(event)
            case _:
                pass

    def _handle_engine_state_changed(self, event: EngineStateChanged) -> None:
        """Handle engine state change event."""
        match event["state"]:
            case EngineState.ACTIVE:
                msg = TuiWatchMessage(
                    type=TuiMessageType.WATCH,
                    status=WatchStatus.DETECTING,
                    message="Running stages...",
                )
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(msg)
            case EngineState.IDLE:
                msg = TuiWatchMessage(
                    type=TuiMessageType.WATCH,
                    status=WatchStatus.WAITING,
                    message="Watching for changes...",
                )
                with contextlib.suppress(queue.Full):
                    self._queue.put_nowait(msg)
            case _:
                pass

    def _handle_pipeline_reloaded(self, event: PipelineReloaded) -> None:
        """Handle pipeline reload event."""
        if event["error"]:
            # Send error message
            msg = TuiWatchMessage(
                type=TuiMessageType.WATCH,
                status=WatchStatus.ERROR,
                message=f"Pipeline invalid: {event['error']}",
            )
            with contextlib.suppress(queue.Full):
                self._queue.put(msg, timeout=1.0)
        else:
            # Send restarting status message
            restart_msg = TuiWatchMessage(
                type=TuiMessageType.WATCH,
                status=WatchStatus.RESTARTING,
                message="Reloading code...",
            )
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(restart_msg)

            # Import registry here to avoid circular import
            from pivot import registry

            # Send reload notification with stage list from registry
            new_stages = list(registry.REGISTRY.list_stages())
            reload_msg = TuiReloadMessage(
                type=TuiMessageType.RELOAD,
                stages=new_stages,
            )
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(reload_msg)

    def close(self) -> None:
        """No cleanup needed for watch sink."""
        pass
