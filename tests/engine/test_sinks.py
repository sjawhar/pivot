"""Tests for event sinks."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

from pivot.engine import sinks, types
from pivot.types import StageStatus, TuiMessageType, TuiQueue


def test_console_sink_handles_stage_started() -> None:
    """ConsoleSink prints stage start message."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    console_mock.stage_start.assert_called_once()
    call_args = console_mock.stage_start.call_args
    assert call_args.kwargs["name"] == "train"
    assert call_args.kwargs["index"] == 2
    assert call_args.kwargs["total"] == 5


def test_console_sink_handles_stage_completed() -> None:
    """ConsoleSink prints stage result message."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    console_mock.stage_result.assert_called_once()
    call_args = console_mock.stage_result.call_args
    assert call_args.kwargs["name"] == "train"
    assert call_args.kwargs["index"] == 2
    assert call_args.kwargs["total"] == 5
    assert call_args.kwargs["status"] == StageStatus.RAN
    assert call_args.kwargs["reason"] == "inputs changed"
    assert call_args.kwargs["duration"] == 1.2345


def test_console_sink_ignores_other_events() -> None:
    """ConsoleSink ignores events it doesn't handle."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    sink.handle(event)

    # Should not call any console methods
    console_mock.stage_start.assert_not_called()
    console_mock.stage_result.assert_not_called()


def test_console_sink_close_closes_console() -> None:
    """ConsoleSink.close() calls console.close()."""
    console_mock = MagicMock()
    sink = sinks.ConsoleSink(console=console_mock)

    sink.close()

    console_mock.close.assert_called_once()


# =============================================================================
# TuiSink Tests
# =============================================================================


def test_tui_sink_handles_stage_started() -> None:
    """TuiSink sends status message for stage started."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "train"
    assert msg["index"] == 2
    assert msg["total"] == 5


def test_tui_sink_handles_stage_completed() -> None:
    """TuiSink sends status message for stage completed."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "train"
    assert msg["status"] == StageStatus.RAN


def test_tui_sink_handles_log_line() -> None:
    """TuiSink sends log message for log line."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    event: types.LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Epoch 1/10",
        "is_stderr": False,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.LOG
    assert msg["stage"] == "train"
    assert msg["line"] == "Epoch 1/10"
    assert msg["is_stderr"] is False


def test_tui_sink_close_sends_none() -> None:
    """TuiSink.close() sends None to signal termination."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.TuiSink(tui_queue=tui_queue, run_id="test-run")

    sink.close()

    msg = tui_queue.get_nowait()
    assert msg is None


# =============================================================================
# JsonlSink Tests
# =============================================================================


def test_jsonl_sink_handles_stage_started() -> None:
    """JsonlSink calls callback with StageStartEvent dict."""
    events_received: list[dict[str, object]] = []

    def callback(event: dict[str, object]) -> None:
        events_received.append(event)

    sink = sinks.JsonlSink(callback=callback)

    event: types.StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    assert len(events_received) == 1
    assert events_received[0]["type"] == "stage_start"
    assert events_received[0]["stage"] == "train"
    assert events_received[0]["index"] == 2
    assert events_received[0]["total"] == 5


def test_jsonl_sink_handles_stage_completed() -> None:
    """JsonlSink calls callback with StageCompleteEvent dict."""
    events_received: list[dict[str, object]] = []

    def callback(event: dict[str, object]) -> None:
        events_received.append(event)

    sink = sinks.JsonlSink(callback=callback)

    event: types.StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "status": StageStatus.RAN,
        "reason": "inputs changed",
        "duration_ms": 1234.5,
        "index": 2,
        "total": 5,
    }
    sink.handle(event)

    assert len(events_received) == 1
    assert events_received[0]["type"] == "stage_complete"
    assert events_received[0]["stage"] == "train"
    assert events_received[0]["status"] == "ran"


def test_jsonl_sink_close_is_noop() -> None:
    """JsonlSink.close() does nothing."""
    sink = sinks.JsonlSink(callback=lambda _: None)
    sink.close()  # Should not raise
