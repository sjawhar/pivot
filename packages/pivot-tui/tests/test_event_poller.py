"""Tests for EventPoller — polls events_since and posts TUI messages."""

from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnusedCallResult=false
# pyright: reportAny=false
# pyright: reportOptionalSubscript=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportUnusedParameter=false
# pyright: reportUnknownLambdaType=false
import threading
import time
from typing import TYPE_CHECKING, cast

import anyio
import pytest

from pivot.types import (
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)
from pivot_tui.event_poller import EventPoller
from pivot_tui.rpc_client_impl import RpcPivotClient
from pivot_tui.testing import FakeRpcServer

if TYPE_CHECKING:
    from pathlib import Path

    from pivot_tui.client import PivotRpc


async def _wait_for_socket(socket_path: Path, timeout: float = 1.0) -> None:
    with anyio.move_on_after(timeout):
        while not socket_path.exists():
            await anyio.sleep(0.01)
    assert socket_path.exists(), "Socket should exist"


@pytest.mark.anyio
async def test_poller_converts_stage_started(tmp_path: Path) -> None:
    """Poller converts stage_started events to TuiStatusMessage."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 2,
            "run_id": "run-1",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1
    msg = messages[0]
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "train"
    assert msg["status"] == StageStatus.IN_PROGRESS
    assert msg["index"] == 0
    assert msg["total"] == 2
    assert msg["run_id"] == "run-1"


@pytest.mark.anyio
async def test_poller_converts_stage_completed(tmp_path: Path) -> None:
    """Poller converts stage_completed events to TuiStatusMessage."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_completed",
            "stage": "train",
            "index": 0,
            "total": 1,
            "status": "ran",
            "reason": "",
            "duration_ms": 1500,
            "run_id": "run-2",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1
    msg = cast("TuiStatusMessage", messages[0])
    assert msg["type"] == TuiMessageType.STATUS
    assert msg["stage"] == "train"
    assert msg["status"] == StageStatus.RAN
    assert msg["elapsed"] == 1.5
    assert msg["run_id"] == "run-2"


@pytest.mark.anyio
async def test_poller_converts_log_line(tmp_path: Path) -> None:
    """Poller converts log_line events to TuiLogMessage."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "log_line",
            "stage": "train",
            "line": "Epoch 1/10",
            "is_stderr": False,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()
    before = time.time()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1
    msg = cast("TuiLogMessage", messages[0])
    assert msg["type"] == TuiMessageType.LOG
    assert msg["stage"] == "train"
    assert msg["line"] == "Epoch 1/10"
    assert msg["is_stderr"] is False
    assert msg["timestamp"] >= before


@pytest.mark.anyio
async def test_poller_converts_engine_state_idle_to_watch(tmp_path: Path) -> None:
    """Poller converts engine_state_changed to idle → TuiWatchMessage."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "engine_state_changed",
            "new_state": "idle",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1
    msg = cast("TuiWatchMessage", messages[0])
    assert msg["type"] == TuiMessageType.WATCH
    assert msg["status"] == WatchStatus.WAITING


@pytest.mark.anyio
async def test_poller_ignores_engine_state_active(tmp_path: Path) -> None:
    """Poller ignores engine_state_changed with non-idle state."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "engine_state_changed",
            "new_state": "active",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # Only the polling continues, no messages from "active" state
    assert all(m.get("type") != TuiMessageType.WATCH for m in messages)  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_poller_ignores_unknown_event_types(tmp_path: Path) -> None:
    """Poller ignores events with unknown type."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event({"type": "unknown_event", "data": "foo"})

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) == 0


@pytest.mark.anyio
async def test_poller_handles_multiple_events(tmp_path: Path) -> None:
    """Poller processes a batch of multiple events in order."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-3",
        }
    )
    server.inject_event(
        {
            "type": "log_line",
            "stage": "train",
            "line": "processing...",
            "is_stderr": False,
        }
    )
    server.inject_event(
        {
            "type": "stage_completed",
            "stage": "train",
            "index": 0,
            "total": 1,
            "status": "ran",
            "reason": "",
            "duration_ms": 500,
            "run_id": "run-3",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 3
    assert messages[0]["type"] == TuiMessageType.STATUS
    assert messages[1]["type"] == TuiMessageType.LOG
    assert messages[2]["type"] == TuiMessageType.STATUS


@pytest.mark.anyio
async def test_poller_tracks_version_no_duplicates(tmp_path: Path) -> None:
    """Poller tracks version and doesn't re-deliver old events."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-4",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        # Let it poll a few cycles — same event should only appear once
        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.5)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # The event should be posted exactly once despite multiple polls
    status_messages = [m for m in messages if m["type"] == TuiMessageType.STATUS]
    assert len(status_messages) == 1


@pytest.mark.anyio
async def test_poller_advances_version_per_event_no_duplicates_on_error(tmp_path: Path) -> None:
    """Poller advances version after each event, even if one fails mid-batch.

    Regression test for Fix A: If event 2/4 throws, events 1 and 3-4 are posted.
    Version must advance past event 2 (even though it failed) so next poll doesn't
    re-fetch and re-post events 1-2.
    """
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    # Event 1: good
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "first",
            "index": 0,
            "total": 3,
            "run_id": "run-version-test",
        }
    )
    # Event 2: bad (missing required "stage" key for stage_completed)
    server.inject_event({"type": "stage_completed"})
    # Event 3: good
    server.inject_event(
        {
            "type": "log_line",
            "stage": "first",
            "line": "after error",
            "is_stderr": False,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        # First poll: events 1 and 3 posted, event 2 skipped
        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # Should have exactly 2 messages (events 1 and 3), not 3 or more
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
    assert messages[0]["type"] == TuiMessageType.STATUS
    assert messages[0]["stage"] == "first"
    assert messages[1]["type"] == TuiMessageType.LOG
    assert messages[1]["line"] == "after error"  # type: ignore[typeddict-item]


@pytest.mark.anyio
async def test_poller_retries_on_connection_error(tmp_path: Path) -> None:
    """Poller retries on error and picks up events after recovery."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    # Inject error for first call, then clear it
    server.inject_error("events_since", -32000, "temporary failure")

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            # Let it fail and retry once
            await anyio.sleep(0.3)
            # Clear error and inject an event
            server.clear_error("events_since")
            server.inject_event(
                {
                    "type": "stage_started",
                    "stage": "train",
                    "index": 0,
                    "total": 1,
                    "run_id": "run-5",
                }
            )
            # Wait for recovery poll
            await anyio.sleep(1.5)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # Should have recovered and posted the event
    assert len(messages) >= 1
    assert messages[0]["type"] == TuiMessageType.STATUS


@pytest.mark.anyio
async def test_poller_stop_terminates_loop(tmp_path: Path) -> None:
    """Calling stop() causes run() to exit."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()
        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.05)
            poller.stop()
            # Task group should exit cleanly — run() returns

        await client.disconnect()
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_poller_stop_interrupts_blocked_poll() -> None:
    """stop() interrupts a blocked events_since call promptly."""

    class _SlowClient:
        async def events_since(self, version: int) -> None:
            await anyio.sleep(30)

    poller = EventPoller(cast("PivotRpc", cast("object", _SlowClient())), lambda _msg: None)
    stop_times = list[float]()

    def _stopper() -> None:
        time.sleep(0.05)
        stop_times.append(time.perf_counter())
        poller.stop()

    thread = threading.Thread(target=_stopper, daemon=True)
    thread.start()

    async with anyio.create_task_group() as tg:
        tg.start_soon(poller.run)

    thread.join(timeout=1.0)
    assert stop_times, "stop() should have been invoked"
    elapsed_since_stop = time.perf_counter() - stop_times[0]
    assert elapsed_since_stop < 0.2, f"Poller should stop quickly, took {elapsed_since_stop:.3f}s"


# =============================================================================
# pipeline_reloaded → TuiReloadMessage
# =============================================================================


@pytest.mark.anyio
async def test_poller_converts_pipeline_reloaded(tmp_path: Path) -> None:
    """Poller converts pipeline_reloaded events to TuiReloadMessage."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "pipeline_reloaded",
            "stages": ["train", "eval"],
            "stages_added": ["eval"],
            "stages_removed": [],
            "stages_modified": ["train"],
            "error": None,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received pipeline_reloaded message"
    msg = cast("TuiReloadMessage", messages[0])
    assert msg["type"] == TuiMessageType.RELOAD, "Message type should be RELOAD"
    assert msg["stages"] == ["train", "eval"], "stages should match"
    assert msg["stages_added"] == ["eval"], "stages_added should match"
    assert msg["stages_removed"] == [], "stages_removed should match"
    assert msg["stages_modified"] == ["train"], "stages_modified should match"


@pytest.mark.anyio
async def test_poller_converts_pipeline_reloaded_with_error(tmp_path: Path) -> None:
    """Poller handles pipeline_reloaded events with error field set."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "pipeline_reloaded",
            "stages": ["train"],
            "stages_added": [],
            "stages_removed": [],
            "stages_modified": [],
            "error": "SyntaxError in pipeline.py",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # Should still produce a reload message even when error is set
    assert len(messages) >= 1, "Should produce reload message even with error"
    msg = cast("TuiReloadMessage", messages[0])
    assert msg["type"] == TuiMessageType.RELOAD, "Message type should be RELOAD"
    assert msg["stages"] == ["train"], "stages should match"


# =============================================================================
# Oneshot mode: stop on EndOfStream / fatal errors
# =============================================================================


@pytest.mark.anyio
async def test_poller_oneshot_stops_on_end_of_stream(tmp_path: Path) -> None:
    """In oneshot mode, poller stops when the connection closes (EndOfStream)."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-eos",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append, oneshot=True)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            # Disconnect client to trigger EndOfStream on next poll
            await client.disconnect()

        # run() should have returned after ClosedResourceError — task group exits cleanly
        assert len(messages) >= 1, "Should have received at least one event before shutdown"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_poller_oneshot_accepts_parameter(tmp_path: Path) -> None:
    """EventPoller accepts oneshot parameter in constructor."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        # Should not raise
        poller = EventPoller(client, lambda _msg: None, oneshot=True)
        assert poller._oneshot is True, "oneshot should be stored"

        poller2 = EventPoller(client, lambda _msg: None)
        assert poller2._oneshot is False, "default should be False"

        await client.disconnect()
    finally:
        await server.stop()


# =============================================================================
# Malformed events → return None (no crash)
# =============================================================================


@pytest.mark.anyio
async def test_poller_handles_malformed_stage_started(tmp_path: Path) -> None:
    """Malformed stage_started event (wrong type for stage) returns None, no crash."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    # Missing required fields
    server.inject_event({"type": "stage_started", "stage": 123, "index": "bad", "total": 1})

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # Should have no messages since the event was malformed
    assert len(messages) == 0, "Malformed event should be silently dropped"


@pytest.mark.anyio
async def test_poller_handles_malformed_log_line(tmp_path: Path) -> None:
    """Malformed log_line event (wrong type for line) returns None, no crash."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event({"type": "log_line", "stage": "train", "line": 42, "is_stderr": "nope"})

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) == 0, "Malformed log_line event should be silently dropped"


# =============================================================================
# Explanation and output_summary passthrough
# =============================================================================


@pytest.mark.anyio
async def test_poller_started_carries_explanation(tmp_path: Path) -> None:
    """stage_started event with explanation → TuiStatusMessage carries explanation."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    explanation = {
        "stage_name": "train",
        "will_run": True,
        "is_forced": False,
        "reason": "Code changed",
        "code_changes": [
            {"key": "func:train", "old_hash": "aaa", "new_hash": "bbb", "change_type": "modified"}
        ],
        "param_changes": [],
        "dep_changes": [],
        "upstream_stale": [],
    }
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-expl",
            "explanation": explanation,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert msg["stage"] == "train"
    assert "explanation" in msg, "TuiStatusMessage should carry explanation"
    assert msg["explanation"]["stage_name"] == "train"
    assert msg["explanation"]["reason"] == "Code changed"


@pytest.mark.anyio
async def test_poller_started_without_explanation(tmp_path: Path) -> None:
    """stage_started event without explanation → TuiStatusMessage has no explanation key."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-no-expl",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert "explanation" not in msg, "No explanation key when event lacks it"


@pytest.mark.anyio
async def test_poller_completed_carries_output_summary(tmp_path: Path) -> None:
    """stage_completed event with output_summary → TuiStatusMessage carries output_summary."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    output_summary = [
        {
            "path": "output.csv",
            "change_type": "modified",
            "output_type": "out",
            "old_hash": "abc123",
            "new_hash": "def456",
        },
        {
            "path": "metrics.json",
            "change_type": "added",
            "output_type": "metric",
            "old_hash": None,
            "new_hash": "ghi789",
        },
    ]
    server.inject_event(
        {
            "type": "stage_completed",
            "stage": "train",
            "index": 0,
            "total": 1,
            "status": "ran",
            "reason": "",
            "duration_ms": 1200,
            "run_id": "run-out",
            "output_summary": output_summary,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert msg["stage"] == "train"
    assert "output_summary" in msg, "TuiStatusMessage should carry output_summary"
    summary = msg["output_summary"]
    assert isinstance(summary, list), "output_summary should be a list"
    assert len(summary) == 2, "Should have two output entries"
    assert summary[0]["path"] == "output.csv"  # type: ignore[index]
    assert summary[1]["path"] == "metrics.json"  # type: ignore[index]


@pytest.mark.anyio
async def test_poller_completed_without_output_summary(tmp_path: Path) -> None:
    """stage_completed event without output_summary → TuiStatusMessage has no output_summary key."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_completed",
            "stage": "train",
            "index": 0,
            "total": 1,
            "status": "ran",
            "reason": "",
            "duration_ms": 500,
            "run_id": "run-no-out",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert "output_summary" not in msg, "No output_summary key when event lacks it"


# =============================================================================
# Fix (a): Malformed event mid-batch → remaining events still processed
# =============================================================================


@pytest.mark.anyio
async def test_process_batch_skips_malformed_event_continues_remaining(tmp_path: Path) -> None:
    """A malformed event mid-batch doesn't prevent remaining events from being processed."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    # Good event, then bad event (missing required "stage" key for stage_started),
    # then another good event
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "first",
            "index": 0,
            "total": 3,
            "run_id": "run-batch",
        }
    )
    # This event will cause a KeyError in _make_completed_message (missing "stage")
    server.inject_event({"type": "stage_completed"})
    server.inject_event(
        {
            "type": "log_line",
            "stage": "first",
            "line": "still here",
            "is_stderr": False,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # First and third events should have been processed; second was malformed
    assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"
    assert messages[0]["type"] == TuiMessageType.STATUS, "First event should be STATUS"
    assert messages[0]["stage"] == "first"
    assert messages[1]["type"] == TuiMessageType.LOG, "Third event should be LOG"
    assert messages[1]["line"] == "still here"  # type: ignore[typeddict-item]


# =============================================================================
# Fix (b): RpcProtocolError in watch mode → poller stops (fatal)
# =============================================================================


@pytest.mark.anyio
async def test_poller_stops_on_rpc_protocol_error_in_watch_mode() -> None:
    """RpcProtocolError is fatal — poller stops instead of retrying forever."""
    from pivot_tui.rpc_client_impl import RpcProtocolError as _RpcProtocolError

    call_count = 0

    class _FailingClient:
        async def events_since(self, version: int) -> None:
            nonlocal call_count
            call_count += 1
            raise _RpcProtocolError("dict", "str")

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()
    poller = EventPoller(
        cast("PivotRpc", cast("object", _FailingClient())),
        messages.append,
        oneshot=False,  # watch mode
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(poller.run)
        # Give enough time for multiple retries if it were retrying
        await anyio.sleep(0.5)

    # Should have stopped after first call, not retried
    assert call_count == 1, f"Expected exactly 1 call (fatal), got {call_count}"
    assert poller._stop_event.is_set(), "Poller should have stopped"


@pytest.mark.anyio
async def test_poller_stops_on_json_decode_error_in_watch_mode() -> None:
    """json.JSONDecodeError is fatal — poller stops instead of retrying."""
    import json as _json

    call_count = 0

    class _FailingClient:
        async def events_since(self, version: int) -> None:
            nonlocal call_count
            call_count += 1
            raise _json.JSONDecodeError("bad json", "", 0)

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()
    poller = EventPoller(
        cast("PivotRpc", cast("object", _FailingClient())),
        messages.append,
        oneshot=False,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(poller.run)
        await anyio.sleep(0.5)

    assert call_count == 1, f"Expected exactly 1 call (fatal), got {call_count}"
    assert poller._stop_event.is_set(), "Poller should have stopped"


# =============================================================================
# Fix (c): Invalid explanation/output_summary payloads → graceful skip
# =============================================================================


@pytest.mark.anyio
async def test_invalid_explanation_payload_skipped(tmp_path: Path) -> None:
    """Non-dict explanation payload is skipped — no 'explanation' key on message."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-bad-expl",
            "explanation": "not a dict",  # invalid: should be dict with stage_name
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert msg["stage"] == "train", "Message should still be created"
    assert "explanation" not in msg, "Invalid explanation should be skipped"


@pytest.mark.anyio
async def test_explanation_dict_missing_stage_name_skipped(tmp_path: Path) -> None:
    """Dict explanation without 'stage_name' key is skipped."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-bad-expl2",
            "explanation": {"will_run": True},  # missing stage_name
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert "explanation" not in msg, "Explanation missing stage_name should be skipped"


@pytest.mark.anyio
async def test_explanation_missing_multiple_required_keys_skipped(tmp_path: Path) -> None:
    """Explanation missing multiple required keys (e.g., will_run, reason) is skipped.

    Regression test for Fix B: Only checking 'stage_name' was insufficient.
    All 7 required keys must be present: stage_name, will_run, is_forced, reason,
    code_changes, param_changes, dep_changes, upstream_stale.
    """
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-incomplete-expl",
            "explanation": {
                "stage_name": "train",
                # Missing: will_run, is_forced, reason, code_changes, param_changes, dep_changes, upstream_stale
            },
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert msg["stage"] == "train", "Message should still be created"
    assert "explanation" not in msg, "Incomplete explanation should be skipped"


@pytest.mark.anyio
async def test_invalid_output_summary_payload_skipped(tmp_path: Path) -> None:
    """Non-list output_summary payload is skipped — no 'output_summary' key on message."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_completed",
            "stage": "train",
            "index": 0,
            "total": 1,
            "status": "ran",
            "reason": "",
            "duration_ms": 500,
            "run_id": "run-bad-out",
            "output_summary": "not a list",  # invalid: should be list
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    assert len(messages) >= 1, "Should have received at least one message"
    msg = cast("TuiStatusMessage", messages[0])
    assert msg["stage"] == "train", "Message should still be created"
    assert "output_summary" not in msg, "Invalid output_summary should be skipped"


# =============================================================================
# Fix (Issue A): Unknown stage status → skip event with warning
# =============================================================================


@pytest.mark.anyio
async def test_unknown_stage_status_skipped(tmp_path: Path) -> None:
    """Unknown stage status value is skipped with warning — remaining events processed."""
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    # Good event, then bad status, then another good event
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "first",
            "index": 0,
            "total": 2,
            "run_id": "run-status-test",
        }
    )
    server.inject_event(
        {
            "type": "stage_completed",
            "stage": "first",
            "index": 0,
            "total": 2,
            "status": "unknown_garbage",  # Invalid status
            "reason": "",
            "duration_ms": 100,
            "run_id": "run-status-test",
        }
    )
    server.inject_event(
        {
            "type": "log_line",
            "stage": "first",
            "line": "still processing",
            "is_stderr": False,
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            poller.stop()

        await client.disconnect()
    finally:
        await server.stop()

    # First and third events should be processed; second was skipped
    assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"
    assert messages[0]["type"] == TuiMessageType.STATUS, "First event should be STATUS"
    assert messages[0]["stage"] == "first"
    assert messages[0]["status"] == StageStatus.IN_PROGRESS
    assert messages[1]["type"] == TuiMessageType.LOG, "Third event should be LOG"
    assert messages[1]["line"] == "still processing"  # type: ignore[typeddict-item]


# =============================================================================
# Fix (Issue B): Oneshot disconnect stops cleanly (logs at WARNING)
# =============================================================================


@pytest.mark.anyio
async def test_oneshot_disconnect_stops_cleanly(tmp_path: Path) -> None:
    """In oneshot mode, connection close stops poller cleanly (verified by existing test).

    The code change from DEBUG to WARNING log level improves diagnostic visibility.
    The existing test_poller_oneshot_stops_on_end_of_stream already verifies the
    behavior is correct (poller stops, no hang). This test verifies the same with
    explicit disconnect timing.
    """
    socket_path = tmp_path / "poller.sock"
    server = FakeRpcServer()
    server.inject_event(
        {
            "type": "stage_started",
            "stage": "train",
            "index": 0,
            "total": 1,
            "run_id": "run-log-test",
        }
    )

    messages = list[TuiStatusMessage | TuiLogMessage | TuiWatchMessage | TuiReloadMessage]()

    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        client = RpcPivotClient()
        await client.connect(socket_path)

        poller = EventPoller(client, messages.append, oneshot=True)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poller.run)
            await anyio.sleep(0.3)
            await client.disconnect()

        assert len(messages) >= 1, "Should have received at least one event before shutdown"
        assert poller._stop_event.is_set(), "Poller should have stopped after disconnect"
    finally:
        await server.stop()
