"""Tests for event sinks."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

from pivot.engine import sinks, types
from pivot.types import (
    StageStatus,
    TuiMessage,
    TuiMessageType,
    TuiQueue,
    TuiWatchMessage,
    WatchStatus,
)


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


# =============================================================================
# WatchSink Tests
# =============================================================================


def test_watch_sink_handles_engine_state_active() -> None:
    """WatchSink sends detecting message when engine becomes active."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.WATCH
    assert msg["status"].value == "detecting"
    assert "Running" in msg["message"]


def test_watch_sink_handles_engine_state_idle() -> None:
    """WatchSink sends waiting message when engine becomes idle."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.IDLE,
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.WATCH
    assert msg["status"].value == "waiting"
    assert "Watching" in msg["message"]


def test_watch_sink_handles_pipeline_reloaded_error() -> None:
    """WatchSink sends error message when pipeline reload fails."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": "Syntax error in stages.py",
    }
    sink.handle(event)

    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.WATCH
    assert msg["status"].value == "error"
    assert "Syntax error" in msg["message"]


def test_watch_sink_handles_pipeline_reloaded_success() -> None:
    """WatchSink sends restart and reload messages when pipeline reload succeeds."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": ["new_stage"],
        "stages_removed": [],
        "stages_modified": [],
        "error": None,
    }
    sink.handle(event)

    # First message should be restarting status
    msg1 = tui_queue.get_nowait()
    assert msg1 is not None
    assert msg1["type"] == TuiMessageType.WATCH
    assert msg1["status"].value == "restarting"

    # Second message should be reload notification
    msg2 = tui_queue.get_nowait()
    assert msg2 is not None
    assert msg2["type"] == TuiMessageType.RELOAD


def test_watch_sink_close_is_noop() -> None:
    """WatchSink.close() does nothing."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.WatchSink(tui_queue=tui_queue)
    sink.close()  # Should not raise


def test_watch_sink_delivers_both_messages_on_reload() -> None:
    """WatchSink delivers both restart and reload messages on successful reload."""
    tui_queue: TuiQueue = queue.Queue()
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": ["new_stage"],
        "stages_removed": ["old_stage"],
        "stages_modified": ["changed_stage"],
        "error": None,
    }
    sink.handle(event)

    # First message: restarting status (can be dropped if queue full)
    msg1 = tui_queue.get_nowait()
    assert msg1 is not None
    assert msg1["type"] == TuiMessageType.WATCH
    assert msg1["status"] == WatchStatus.RESTARTING

    # Second message: reload with stage data (uses blocking put to ensure delivery)
    msg2 = tui_queue.get_nowait()
    assert msg2 is not None
    assert msg2["type"] == TuiMessageType.RELOAD
    assert msg2["stages_added"] == ["new_stage"]
    assert msg2["stages_removed"] == ["old_stage"]
    assert msg2["stages_modified"] == ["changed_stage"]


def test_watch_sink_suppresses_queue_full_for_restart_message() -> None:
    """WatchSink silently drops restart message if queue is full."""
    # Create a queue with maxsize=1, pre-fill it
    tui_queue: TuiQueue = queue.Queue(maxsize=1)
    tui_queue.put(
        {"type": TuiMessageType.WATCH, "status": WatchStatus.WAITING, "message": "filler"}
    )
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": [],
        "stages_removed": [],
        "stages_modified": [],
        "error": None,
    }

    # Drain queue in background so reload_msg can be delivered after timeout
    import threading

    def drain_after_delay() -> None:
        import time

        time.sleep(0.1)
        tui_queue.get()

    drainer = threading.Thread(target=drain_after_delay)
    drainer.start()

    # This should not raise - restart_msg dropped, reload_msg waits then succeeds
    sink.handle(event)
    drainer.join()

    # Should have the reload message (restart was dropped due to full queue)
    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.RELOAD


def test_watch_sink_suppresses_queue_full_for_reload_message_after_timeout() -> None:
    """WatchSink silently drops reload message if queue stays full after timeout."""
    tui_queue: TuiQueue = queue.Queue(maxsize=1)
    # Fill queue and never drain it
    filler_msg = TuiWatchMessage(
        type=TuiMessageType.WATCH, status=WatchStatus.WAITING, message="filler"
    )
    tui_queue.put(filler_msg)
    sink = sinks.WatchSink(tui_queue=tui_queue)

    event: types.PipelineReloaded = {
        "type": "pipeline_reloaded",
        "stages_added": ["critical_stage"],
        "stages_removed": [],
        "stages_modified": [],
        "error": None,
    }

    # Mock put to immediately raise queue.Full (simulating timeout expiration)
    original_put = tui_queue.put

    def mock_put(msg: TuiMessage, block: bool = True, timeout: float | None = None) -> None:
        if timeout is not None:
            # Blocking put with timeout - simulate timeout expiration
            raise queue.Full
        # Non-blocking put (from put_nowait) - use original which will suppress queue.Full
        original_put(msg, block=block, timeout=timeout)

    with patch.object(tui_queue, "put", side_effect=mock_put):
        # Should not raise - reload_msg is dropped after timeout
        sink.handle(event)

    # Queue should still only have the original filler message
    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == TuiMessageType.WATCH
    # Type narrowing: msg is TuiWatchMessage since type == WATCH
    assert msg["status"] == WatchStatus.WAITING, "Original message should still be there"

    # Queue should be empty - reload message was dropped
    assert tui_queue.empty(), "Reload message should have been dropped"
