"""Tests for TuiSink direct post_message behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot.engine.sinks import TuiSink
from pivot.engine.types import (  # noqa: TC001 - used at runtime for TypedDict
    LogLine,
    StageCompleted,
    StageStarted,
)
from pivot.tui.run import TuiShutdown, TuiUpdate
from pivot.types import StageStatus, is_tui_log_message, is_tui_status_message

if TYPE_CHECKING:
    import textual.message


class MockApp:
    """Mock app that implements MessagePoster protocol."""

    messages: list[textual.message.Message]

    def __init__(self) -> None:
        self.messages = []

    def post_message(self, message: textual.message.Message) -> bool:
        self.messages.append(message)
        return True


@pytest.mark.anyio
async def test_tui_sink_posts_stage_started() -> None:
    """TuiSink posts TuiUpdate for stage_started events."""
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run")

    event: StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 0,
        "total": 3,
    }
    await sink.handle(event)

    assert len(app.messages) == 1
    msg = app.messages[0]
    assert isinstance(msg, TuiUpdate)
    assert is_tui_status_message(msg.msg)
    assert msg.msg["stage"] == "train"


@pytest.mark.anyio
async def test_tui_sink_posts_stage_completed() -> None:
    """TuiSink posts TuiUpdate for stage_completed events."""
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run")

    event: StageCompleted = {
        "type": "stage_completed",
        "stage": "train",
        "index": 0,
        "total": 3,
        "status": StageStatus.RAN,
        "reason": "success",
        "duration_ms": 1500,
    }
    await sink.handle(event)

    assert len(app.messages) == 1
    msg = app.messages[0]
    assert isinstance(msg, TuiUpdate)
    assert is_tui_status_message(msg.msg)
    assert msg.msg["stage"] == "train"
    assert msg.msg["status"] == StageStatus.RAN


@pytest.mark.anyio
async def test_tui_sink_posts_log_line() -> None:
    """TuiSink posts TuiUpdate for log_line events."""
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run")

    event: LogLine = {
        "type": "log_line",
        "stage": "train",
        "line": "Processing data...",
        "is_stderr": False,
    }
    await sink.handle(event)

    assert len(app.messages) == 1
    msg = app.messages[0]
    assert isinstance(msg, TuiUpdate)
    assert is_tui_log_message(msg.msg)
    assert msg.msg["line"] == "Processing data..."


@pytest.mark.anyio
async def test_tui_sink_ignores_unknown_events() -> None:
    """TuiSink ignores events it doesn't handle."""
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run")

    event = {"type": "engine_state_changed", "state": "running"}
    await sink.handle(event)  # pyright: ignore[reportArgumentType] - testing unknown event type

    assert len(app.messages) == 0


@pytest.mark.anyio
async def test_tui_sink_close_posts_shutdown() -> None:
    """TuiSink.close() posts TuiShutdown message."""
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run")

    await sink.close()

    assert len(app.messages) == 1
    assert isinstance(app.messages[0], TuiShutdown)


@pytest.mark.anyio
async def test_tui_sink_handles_multiple_events() -> None:
    """TuiSink handles multiple events in sequence."""
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run")

    events = [
        {"type": "stage_started", "stage": "a", "index": 0, "total": 2},
        {"type": "log_line", "stage": "a", "line": "hello", "is_stderr": False},
        {
            "type": "stage_completed",
            "stage": "a",
            "index": 0,
            "total": 2,
            "status": StageStatus.RAN,
            "reason": "",
            "duration_ms": 100,
        },
        {"type": "stage_started", "stage": "b", "index": 1, "total": 2},
    ]

    for event in events:
        await sink.handle(event)  # pyright: ignore[reportArgumentType] - dict literals

    assert len(app.messages) == 4


@pytest.mark.anyio
async def test_tui_sink_handles_post_message_failure() -> None:
    """TuiSink handles post_message returning False gracefully.

    Per Textual docs, post_message() can return False if the app's message
    queue is closed or shutting down. This should not raise an exception.
    """

    class FailingApp:
        """Mock app that rejects all messages."""

        messages: list[textual.message.Message]

        def __init__(self) -> None:
            self.messages = []

        def post_message(self, message: textual.message.Message) -> bool:
            self.messages.append(message)
            return False  # Simulate app shutdown or queue closed

    app = FailingApp()
    sink = TuiSink(app=app, run_id="test")

    # Should not raise even if post_message returns False
    event: StageStarted = {
        "type": "stage_started",
        "stage": "train",
        "index": 0,
        "total": 1,
    }
    await sink.handle(event)

    # Message was attempted but rejected
    assert len(app.messages) == 1, "Message should have been posted (even if rejected)"


@pytest.mark.anyio
async def test_tui_sink_posts_multiple_event_types_in_sequence() -> None:
    """TuiSink handles a realistic sequence of events without errors.

    This test verifies the sink handles a complete stage lifecycle:
    started -> log -> completed.
    """
    app = MockApp()
    sink = TuiSink(app=app, run_id="test-run-123")

    # Simulate complete stage lifecycle
    events = [
        {"type": "stage_started", "stage": "process", "index": 0, "total": 1},
        {"type": "log_line", "stage": "process", "line": "Processing data...", "is_stderr": False},
        {"type": "log_line", "stage": "process", "line": "Complete!", "is_stderr": False},
        {
            "type": "stage_completed",
            "stage": "process",
            "index": 0,
            "total": 1,
            "status": StageStatus.RAN,
            "reason": "code changed",
            "duration_ms": 2500.0,
        },
    ]

    for event in events:
        await sink.handle(event)  # pyright: ignore[reportArgumentType]

    assert len(app.messages) == 4, "Should have posted all events"

    # Verify message types
    from pivot.tui.run import TuiUpdate

    assert all(isinstance(msg, TuiUpdate) for msg in app.messages), (
        "All messages should be TuiUpdate instances"
    )
