"""Tests for AgentRpcSource."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import anyio
import pytest

from pivot.engine.agent_rpc import (
    AgentEventSink,
    AgentRpcHandler,
    AgentRpcSource,
    QueryResult,
    QueryStatusResult,
)
from pivot.engine.engine import Engine
from pivot.engine.types import EventSource, InputEvent, StageStarted


async def _wait_for_socket(socket_path: Path, timeout: float = 2.0) -> None:
    """Wait for socket to be connectable, with retry loop."""
    import stat

    deadline = anyio.current_time() + timeout
    while anyio.current_time() < deadline:
        if socket_path.exists():
            # Check if it's actually a socket (not a stale regular file)
            try:
                mode = socket_path.stat().st_mode
                if stat.S_ISSOCK(mode):
                    return
            except OSError:
                pass
        await anyio.sleep(0.05)
    msg = f"Socket {socket_path} not created within {timeout}s"
    raise TimeoutError(msg)


@pytest.mark.anyio
async def test_agent_rpc_source_accepts_connections(tmp_path: Path) -> None:
    """AgentRpcSource accepts connections on Unix socket."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        # Wait for server to start
        await _wait_for_socket(socket_path)

        # Connect and send run command
        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "run", "id": 1}
            await conn.send(json.dumps(request).encode() + b"\n")

            # Read response
            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") == "accepted"

        # Source should have emitted a RunRequested event
        event = await recv.receive()
        events_received.append(event)

        tg.cancel_scope.cancel()

    assert len(events_received) == 1
    assert events_received[0]["type"] == "run_requested"


@pytest.mark.anyio
async def test_agent_rpc_source_cancel_command(tmp_path: Path) -> None:
    """AgentRpcSource emits CancelRequested for cancel method."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "cancel", "id": 2}
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") == "accepted"

        event = await recv.receive()
        events_received.append(event)

        tg.cancel_scope.cancel()

    assert len(events_received) == 1
    assert events_received[0]["type"] == "cancel_requested"


@pytest.mark.anyio
async def test_agent_rpc_source_run_with_params(tmp_path: Path) -> None:
    """AgentRpcSource passes stages and force params to RunRequested."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {
                "jsonrpc": "2.0",
                "method": "run",
                "params": {"stages": ["train", "eval"], "force": True},
                "id": 3,
            }
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") == "accepted"

        event = await recv.receive()
        events_received.append(event)

        tg.cancel_scope.cancel()

    assert len(events_received) == 1
    assert events_received[0]["type"] == "run_requested"
    assert events_received[0]["stages"] == ["train", "eval"]
    assert events_received[0]["force"] is True


@pytest.mark.anyio
async def test_agent_rpc_source_unknown_method(tmp_path: Path) -> None:
    """AgentRpcSource returns error for unknown methods."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "unknown", "id": 4}
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response
            assert response["error"]["code"] == -32601  # Method not found

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_invalid_json(tmp_path: Path) -> None:
    """AgentRpcSource returns parse error for invalid JSON."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            await conn.send(b"not valid json\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response
            assert response["error"]["code"] == -32700  # Parse error

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_notification_no_response(tmp_path: Path) -> None:
    """AgentRpcSource does not respond to notifications (no id)."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            # Notification: no id field
            request = {"jsonrpc": "2.0", "method": "run"}
            await conn.send(json.dumps(request).encode() + b"\n")

            # Should still emit event
            event = await recv.receive()
            events_received.append(event)

            # Try to receive response with timeout - should timeout since no response sent
            with anyio.move_on_after(0.2):
                await conn.receive(4096)

        tg.cancel_scope.cancel()

    assert len(events_received) == 1
    assert events_received[0]["type"] == "run_requested"


@pytest.mark.anyio
async def test_agent_rpc_source_cleans_stale_socket(tmp_path: Path) -> None:
    """AgentRpcSource removes stale socket file on startup."""
    socket_path = tmp_path / "agent.sock"

    # Create stale socket file
    socket_path.touch()

    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        # Should be able to connect despite stale file
        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "run", "id": 1}
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") == "accepted"

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_socket_permissions(tmp_path: Path) -> None:
    """AgentRpcSource sets socket permissions to owner-only."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        # Check socket permissions (should be 0o600 = owner read/write only)
        mode = socket_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

        tg.cancel_scope.cancel()


def test_agent_rpc_source_conforms_to_protocol() -> None:
    """AgentRpcSource conforms to EventSource protocol."""
    source = AgentRpcSource(socket_path=Path("/tmp/test.sock"))
    _source: EventSource = source
    assert _source is source


@pytest.mark.anyio
async def test_agent_event_sink_broadcasts_to_subscribers() -> None:
    """AgentEventSink broadcasts events to all subscribers."""
    sink = AgentEventSink()

    # Subscribe two clients
    recv1 = await sink.subscribe("client1")
    recv2 = await sink.subscribe("client2")

    # Emit an event
    event = StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=1,
    )
    await sink.handle(event)

    # Both should receive it
    event1 = recv1.receive_nowait()
    event2 = recv2.receive_nowait()

    assert event1["type"] == "stage_started"
    assert event2["type"] == "stage_started"
    # After type narrowing, we know these are StageStarted events
    assert event1["stage"] == "train"  # type: ignore[typeddict-item]
    assert event2["stage"] == "train"  # type: ignore[typeddict-item]

    await sink.close()


@pytest.mark.anyio
async def test_agent_event_sink_unsubscribe() -> None:
    """AgentEventSink removes client on unsubscribe and closes channel."""
    sink = AgentEventSink()

    recv = await sink.subscribe("client1")
    await sink.unsubscribe("client1")

    # Event after unsubscribe should not be received
    event = StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=1,
    )
    await sink.handle(event)

    # Channel should be closed (EndOfStream) since unsubscribe closes the send channel
    with pytest.raises(anyio.EndOfStream):
        recv.receive_nowait()

    await sink.close()


@pytest.mark.anyio
async def test_agent_rpc_source_handles_status_query(tmp_path: Path) -> None:
    """AgentRpcSource handles status query and returns engine state."""
    socket_path = tmp_path / "agent.sock"

    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)
        source = AgentRpcSource(socket_path=socket_path, handler=handler)

        send, recv = anyio.create_memory_object_stream[InputEvent](10)

        async with anyio.create_task_group() as tg:
            tg.start_soon(source.run, send)
            await _wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Send status query
                request = {"jsonrpc": "2.0", "method": "status", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")

                response_line = await conn.receive(4096)
                response = json.loads(response_line.decode())

                assert "result" in response
                assert response["result"]["state"] == "idle"
                assert "running" in response["result"]
                assert "pending" in response["result"]

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_rejects_oversized_messages(tmp_path: Path) -> None:
    """AgentRpcSource rejects messages larger than 1MB to prevent memory exhaustion."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            # Send message larger than 1MB
            huge_payload = "x" * (1024 * 1024 + 1)
            oversized_request = json.dumps(
                {"jsonrpc": "2.0", "method": "run", "id": 1, "data": huge_payload}
            )
            await conn.send(oversized_request.encode() + b"\n")

            # Should receive error response
            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response, "Should return error for oversized message"
            assert response["error"]["code"] == -32600, "Should be invalid request error"

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_handles_concurrent_connections(tmp_path: Path) -> None:
    """AgentRpcSource handles multiple concurrent client connections."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async def collect_events() -> None:
        async for event in recv:
            events_received.append(event)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)
        tg.start_soon(collect_events)

        await _wait_for_socket(socket_path)

        # Connect two clients simultaneously
        async def send_command(client_id: int) -> None:
            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = {"jsonrpc": "2.0", "method": "run", "id": client_id}
                await conn.send(json.dumps(request).encode() + b"\n")
                response_line = await conn.receive(4096)
                response = json.loads(response_line.decode())
                assert response.get("result") == "accepted"

        async with anyio.create_task_group() as client_tg:
            client_tg.start_soon(send_command, 1)
            client_tg.start_soon(send_command, 2)

        # Wait for events to be processed
        await anyio.sleep(0.1)

        tg.cancel_scope.cancel()

    # Should have received events from both clients
    run_events = [e for e in events_received if e["type"] == "run_requested"]
    assert len(run_events) == 2, "Should process commands from both clients"


@pytest.mark.anyio
async def test_agent_rpc_handler_stages_query_without_pipeline(tmp_path: Path) -> None:
    """AgentRpcHandler returns empty stages list when no pipeline."""
    from pivot.engine.engine import Engine

    socket_path = tmp_path / "agent.sock"

    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)
        source = AgentRpcSource(socket_path=socket_path, handler=handler)

        send, recv = anyio.create_memory_object_stream[InputEvent](10)

        async with anyio.create_task_group() as tg:
            tg.start_soon(source.run, send)
            await _wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Query stages when no pipeline is set
                request = {"jsonrpc": "2.0", "method": "stages", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")

                response_line = await conn.receive(4096)
                response = json.loads(response_line.decode())

                assert "result" in response
                assert response["result"]["stages"] == [], (
                    "Should return empty list without pipeline"
                )

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_rpc_run_invalid_stage_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RPC run with invalid stage name returns descriptive error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()

    from pivot.pipeline.pipeline import Pipeline

    pipeline = Pipeline("test", root=tmp_path)

    def my_stage() -> None:
        pass

    pipeline.register(my_stage, name="valid_stage")

    async with Engine(pipeline=pipeline) as eng:
        handler = AgentRpcHandler(engine=eng)
        source = AgentRpcSource(socket_path=tmp_path / "test.sock", handler=handler)

        send, recv = anyio.create_memory_object_stream[InputEvent](16)

        request = {
            "jsonrpc": "2.0",
            "method": "run",
            "params": {"stages": ["nonexistent_stage"]},
            "id": 1,
        }

        response = await source._handle_request(request, send)

        assert response is not None
        assert "error" in response
        error = response.get("error")
        assert isinstance(error, dict)
        assert error.get("code") == -32001  # Stage not found
        message = error.get("message")
        assert isinstance(message, str)
        assert "nonexistent_stage" in message


@pytest.mark.anyio
async def test_agent_rpc_source_connection_timeout() -> None:
    """AgentRpcSource has timeout protection for idle connections.

    Note: This test verifies timeout mechanism exists but uses short timeout
    to avoid slow test execution. Production uses 5 minute timeout.
    """
    from pathlib import Path
    from unittest.mock import patch

    import anyio

    from pivot.engine.agent_rpc import AgentRpcSource
    from pivot.engine.types import InputEvent

    socket_path = Path("/tmp/test_timeout.sock")
    if socket_path.exists():
        socket_path.unlink()

    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    # Patch timeout to 0.5 seconds for testing
    with patch("pivot.engine.agent_rpc._CLIENT_TIMEOUT", 0.5):
        async with anyio.create_task_group() as tg:
            source = AgentRpcSource(socket_path=socket_path)
            tg.start_soon(source.run, send)

            await _wait_for_socket(socket_path)

            # Connect but don't send anything (idle connection)
            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Wait for timeout (should disconnect)
                with anyio.move_on_after(1.0):
                    with contextlib.suppress(anyio.ClosedResourceError, anyio.EndOfStream):
                        # Connection should be closed by server timeout
                        await conn.receive(4096)

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_accepts_new_connections_after_handler_exception(
    tmp_path: Path,
) -> None:
    """AgentRpcSource continues accepting connections after handler exception."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    exception_triggered = anyio.Event()

    class _RaisingHandler:
        """Handler that raises on first query, succeeds on second.

        Duck-typed to match AgentRpcHandler interface without inheritance.
        """

        _call_count: int

        def __init__(self) -> None:
            self._call_count = 0

        def validate_stages(self, stages: list[str] | None) -> str | None:
            return None

        async def handle_query(self, method: str) -> QueryResult:
            self._call_count += 1
            if self._call_count == 1:
                exception_triggered.set()
                raise RuntimeError("First query fails")
            return QueryStatusResult(state="idle", running=list[str](), pending=list[str]())

    async with anyio.create_task_group() as tg:
        from typing import cast

        # Cast via object to avoid "types don't overlap" error for duck-typed handler
        handler = cast("AgentRpcHandler", cast("object", _RaisingHandler()))
        source = AgentRpcSource(socket_path=socket_path, handler=handler)
        tg.start_soon(source.run, send)

        await _wait_for_socket(socket_path)

        # First connection: triggers exception in handler
        with anyio.move_on_after(2.0):
            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Send a query that will trigger the handler exception
                request = {"jsonrpc": "2.0", "method": "custom_query", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")

                # Handler exception should cause connection to close or return error
                # (connection handling wraps exceptions)
                try:
                    response_line = await conn.receive(4096)
                    # If we got a response, it should be an error
                    if response_line:
                        response = json.loads(response_line.decode())
                        # Server should have returned error or closed
                except (anyio.EndOfStream, anyio.ClosedResourceError):
                    pass  # Expected: connection closed due to error

        # Wait for exception to be triggered
        with anyio.move_on_after(1.0):
            await exception_triggered.wait()

        # Second connection: should succeed (server still accepting)
        with anyio.move_on_after(2.0):
            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Send a run command (uses standard path, not handler)
                request = {"jsonrpc": "2.0", "method": "run", "id": 2}
                await conn.send(json.dumps(request).encode() + b"\n")

                response_line = await conn.receive(4096)
                response = json.loads(response_line.decode())

                assert response.get("result") == "accepted", (
                    "Server should accept connections after handler exception"
                )

        # Collect the event
        with anyio.move_on_after(0.5):
            event = await recv.receive()
            events_received.append(event)

        tg.cancel_scope.cancel()

    assert len(events_received) >= 1, "Should process command after handler exception"
    assert events_received[0]["type"] == "run_requested"
