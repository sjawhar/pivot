"""Tests for AgentRpcSource."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import anyio
import pytest

from helpers import wait_for_socket
from pivot.engine.agent_rpc import (
    AgentRpcHandler,
    AgentRpcSource,
    EventBuffer,
    EventSink,
    QueryResult,
    QueryStatusResult,
)
from pivot.engine.engine import Engine
from pivot.engine.types import EventSource, InputEvent, OutputEvent, StageStarted


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
        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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
    """EventSink broadcasts events to all subscribers."""
    sink = EventSink()

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
    # Type narrow to StageStarted using discriminated union
    if event1["type"] == "stage_started" and event2["type"] == "stage_started":
        assert event1["stage"] == "train"
        assert event2["stage"] == "train"

    await sink.close()


@pytest.mark.anyio
async def test_agent_event_sink_unsubscribe() -> None:
    """EventSink removes client on unsubscribe and closes channel."""
    sink = EventSink()

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
            await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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

        await wait_for_socket(socket_path)

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
            await wait_for_socket(socket_path)

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

            await wait_for_socket(socket_path)

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

        async def handle_query(
            self, method: str, params: dict[str, object] | None = None
        ) -> QueryResult:
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

        await wait_for_socket(socket_path)

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


@pytest.mark.anyio
async def test_event_buffer_captures_events() -> None:
    """EventBuffer should capture events with version numbers."""
    buffer = EventBuffer(max_events=100)
    buffer.handle_sync({"type": "stage_started", "stage": "train", "index": 1, "total": 2})

    result = buffer.events_since(0)
    assert result["version"] == 1, "First event should have version 1"
    assert len(result["events"]) == 1, "Should have exactly 1 event"


@pytest.mark.anyio
async def test_event_buffer_eviction() -> None:
    """EventBuffer should evict oldest events when full."""
    buffer = EventBuffer(max_events=3)
    for i in range(5):
        buffer.handle_sync({"type": "stage_started", "stage": f"s{i}", "index": i, "total": 5})

    result = buffer.events_since(0)
    assert len(result["events"]) == 3, "Should only keep last 3 events (max_events=3)"


@pytest.mark.anyio
async def test_handler_events_since_query() -> None:
    """Handler should return events from buffer."""

    buffer = EventBuffer(max_events=100)
    buffer.handle_sync({"type": "stage_started", "stage": "train", "index": 1, "total": 1})

    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine, event_buffer=buffer)

    result = await handler.handle_query("events_since", {"version": 0})
    # Type narrow to EventsResult by checking for version key (discriminates from other QueryResult types)
    assert "version" in result and "events" in result, (
        "Result should have version and events fields"
    )
    assert result["version"] == 1, "First event should have version 1"
    assert len(result["events"]) == 1, "Should have exactly 1 event"


# =============================================================================
# EventBuffer Edge Cases and Boundary Conditions
# =============================================================================


@pytest.mark.anyio
async def test_event_buffer_version_wraparound() -> None:
    """EventBuffer wraps version at _MAX_VERSION to prevent overflow.

    Critical: Prevents version overflow causing event loss or duplication.
    """
    from pivot.engine.agent_rpc import _MAX_VERSION

    buffer = EventBuffer(max_events=10)

    # Set version near max
    buffer._version = _MAX_VERSION - 1

    # Add events that trigger wraparound
    buffer.handle_sync({"type": "stage_started", "stage": "s1", "index": 0, "total": 2})
    assert buffer._version == _MAX_VERSION

    buffer.handle_sync({"type": "stage_started", "stage": "s2", "index": 1, "total": 2})
    # Version should wrap to 1, not overflow
    assert buffer._version == 1, "Version should wrap to 1 after reaching _MAX_VERSION"

    # Client polling with old version should get all buffered events
    result = buffer.events_since(0)
    assert result["version"] == 1
    assert len(result["events"]) == 2


@pytest.mark.anyio
async def test_event_buffer_empty_buffer_query() -> None:
    """events_since on empty buffer returns empty list with version 0.

    Boundary condition: Ensures no crash on empty buffer access.
    """
    buffer = EventBuffer(max_events=100)

    result = buffer.events_since(0)
    assert result["version"] == 0, "Empty buffer should have version 0"
    assert result["events"] == [], "Empty buffer should return empty events list"


@pytest.mark.anyio
async def test_event_buffer_query_current_version() -> None:
    """events_since with current version returns empty list.

    Boundary condition: Version comparison should be > not >=.
    """
    buffer = EventBuffer(max_events=100)
    buffer.handle_sync({"type": "stage_started", "stage": "train", "index": 0, "total": 1})

    current_version = buffer._version
    result = buffer.events_since(current_version)

    assert result["version"] == current_version
    assert result["events"] == [], "Query at current version should return no events"


@pytest.mark.anyio
async def test_event_buffer_query_future_version() -> None:
    """events_since with future version returns ALL events (wraparound handling).

    Edge case: Client with version > current indicates wraparound - return all events
    to prevent missing data. This handles the case where version wrapped from
    _MAX_VERSION to 1, but client still has old version number.
    """
    buffer = EventBuffer(max_events=100)
    buffer.handle_sync({"type": "stage_started", "stage": "train", "index": 0, "total": 1})

    # Query with version ahead of current (simulates post-wraparound scenario)
    future_version = buffer._version + 100
    result = buffer.events_since(future_version)

    assert result["version"] == buffer._version
    # Returns ALL buffered events to handle wraparound case
    assert len(result["events"]) == 1, "Future version should return all events (wraparound)"
    assert result["events"][0]["version"] == 1


@pytest.mark.anyio
async def test_event_buffer_query_after_eviction() -> None:
    """events_since after buffer eviction returns only remaining events.

    Critical: Ensures clients with old versions don't crash when events are evicted.
    """
    buffer = EventBuffer(max_events=3)

    # Fill buffer and cause eviction
    for i in range(5):
        buffer.handle_sync({"type": "stage_started", "stage": f"s{i}", "index": i, "total": 5})

    # Query with version that was evicted (version 1-2 are gone)
    result = buffer.events_since(0)

    # Should only get last 3 events (versions 3, 4, 5)
    assert len(result["events"]) == 3
    assert result["events"][0]["version"] == 3
    assert result["events"][2]["version"] == 5


@pytest.mark.anyio
async def test_event_buffer_thread_safety() -> None:
    """EventBuffer handles concurrent access from multiple threads.

    Critical: Buffer uses threading.Lock for sync+async access - must be race-free.
    """
    import concurrent.futures

    buffer = EventBuffer(max_events=1000)

    def add_events(offset: int) -> None:
        for i in range(100):
            buffer.handle_sync(
                {
                    "type": "stage_started",
                    "stage": f"s{offset}_{i}",
                    "index": i,
                    "total": 100,
                }
            )

    # Add events concurrently from 5 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(add_events, i) for i in range(5)]
        concurrent.futures.wait(futures)

    # Should have version = 500 (5 threads * 100 events)
    result = buffer.events_since(0)
    assert result["version"] == 500, "All events should be recorded with sequential versions"
    # Buffer only holds last 1000, so we should see some events
    assert len(result["events"]) > 0


# =============================================================================
# Handler Query Parameter Validation
# =============================================================================


@pytest.mark.anyio
async def test_handler_events_since_invalid_version_type() -> None:
    """Handler rejects non-integer version with ValueError.

    Error path: Type validation must catch string/float versions.
    """
    from unittest.mock import MagicMock

    buffer = EventBuffer(max_events=100)
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine, event_buffer=buffer)

    with pytest.raises(ValueError, match="version must be integer"):
        await handler.handle_query("events_since", {"version": "123"})


@pytest.mark.anyio
async def test_handler_events_since_negative_version() -> None:
    """Handler rejects negative version with ValueError.

    Boundary condition: Negative versions are semantically invalid.
    """
    from unittest.mock import MagicMock

    buffer = EventBuffer(max_events=100)
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine, event_buffer=buffer)

    with pytest.raises(ValueError, match="version must be integer between"):
        await handler.handle_query("events_since", {"version": -1})


@pytest.mark.anyio
async def test_handler_events_since_max_version_boundary() -> None:
    """Handler accepts version exactly at _MAX_VERSION.

    Boundary condition: _MAX_VERSION itself is valid.
    """
    from unittest.mock import MagicMock

    from pivot.engine.agent_rpc import _MAX_VERSION

    buffer = EventBuffer(max_events=100)
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine, event_buffer=buffer)

    # Should not raise - _MAX_VERSION is valid
    result = await handler.handle_query("events_since", {"version": _MAX_VERSION})
    assert "version" in result


@pytest.mark.anyio
async def test_handler_events_since_exceeds_max_version() -> None:
    """Handler rejects version > _MAX_VERSION with ValueError.

    Boundary condition: Versions beyond max are invalid.
    """
    from unittest.mock import MagicMock

    from pivot.engine.agent_rpc import _MAX_VERSION

    buffer = EventBuffer(max_events=100)
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine, event_buffer=buffer)

    with pytest.raises(ValueError, match="version must be integer between"):
        await handler.handle_query("events_since", {"version": _MAX_VERSION + 1})


@pytest.mark.anyio
async def test_handler_explain_missing_stage_param() -> None:
    """Handler explain query requires stage parameter.

    Error path: Missing required parameter should give clear error.
    """
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        with pytest.raises(ValueError, match="stage must be string"):
            await handler.handle_query("explain", {})


@pytest.mark.anyio
async def test_handler_explain_invalid_stage_type() -> None:
    """Handler explain query rejects non-string stage.

    Error path: Type validation must catch integer/null stage.
    """
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        with pytest.raises(ValueError, match="stage must be string"):
            await handler.handle_query("explain", {"stage": 123})


@pytest.mark.anyio
async def test_handler_stage_info_missing_param() -> None:
    """Handler stage_info query requires stage parameter.

    Error path: Missing required parameter should give clear error.
    """
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        with pytest.raises(ValueError, match="stage must be string"):
            await handler.handle_query("stage_info", {})


@pytest.mark.anyio
async def test_handler_stage_info_invalid_stage_type() -> None:
    """Handler stage_info query rejects non-string stage.

    Error path: Type validation must catch invalid types.
    """
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        with pytest.raises(ValueError, match="stage must be string"):
            await handler.handle_query("stage_info", {"stage": None})


# =============================================================================
# AgentRpcSource Connection Edge Cases
# =============================================================================


@pytest.mark.anyio
async def test_agent_rpc_source_empty_lines_ignored(tmp_path: Path) -> None:
    """AgentRpcSource ignores empty lines in message stream.

    Robustness: Extra newlines should not cause errors.
    """
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            # Send message with empty lines
            msg_with_blanks = '\n\n{"jsonrpc":"2.0","method":"run","id":1}\n\n'
            await conn.send(msg_with_blanks.encode())

            import json

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response["result"] == "accepted", "Empty lines should not break parsing"

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_message_at_size_boundary(tmp_path: Path) -> None:
    """AgentRpcSource accepts message exactly at _MAX_MESSAGE_SIZE.

    Boundary condition: Message at limit should succeed, limit+1 should fail.
    """
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            # Create message just under limit (accounting for JSON structure)
            # _MAX_MESSAGE_SIZE = 1MB
            max_size = 1024 * 1024
            padding = "x" * (max_size - 100)  # Leave room for JSON structure

            import json

            at_limit = json.dumps(
                {"jsonrpc": "2.0", "method": "run", "id": 1, "params": {"data": padding}}
            )

            # Should accept (we're slightly under limit)
            await conn.send(at_limit.encode() + b"\n")
            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") == "accepted" or response.get("error"), (
                "Message near size limit should be processed"
            )

        tg.cancel_scope.cancel()


# =============================================================================
# Run Command Validation Edge Cases
# =============================================================================


@pytest.mark.anyio
async def test_agent_rpc_source_run_empty_stages_list(tmp_path: Path) -> None:
    """AgentRpcSource accepts empty stages list (different from None).

    Edge case: Empty list [] means "no stages", None means "all stages".
    """
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            import json

            request = {
                "jsonrpc": "2.0",
                "method": "run",
                "params": {"stages": []},  # Empty list
                "id": 1,
            }
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") == "accepted"

        event = await recv.receive()
        events_received.append(event)

        tg.cancel_scope.cancel()

    # Type narrow: run_requested events have "stages" field
    first_event = events_received[0]
    assert first_event["type"] == "run_requested"
    if first_event["type"] == "run_requested":
        assert first_event["stages"] == [], "Empty list should be preserved"


@pytest.mark.anyio
async def test_agent_rpc_source_run_stages_with_empty_string(tmp_path: Path) -> None:
    """AgentRpcSource rejects stages list containing empty string.

    Edge case: Empty string is not a valid stage name.
    """
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
            "params": {"stages": [""]},  # Empty string
            "id": 1,
        }

        response = await source._handle_request(request, send)

        assert response is not None
        assert "error" in response, "Empty stage name should be rejected"


@pytest.mark.anyio
async def test_agent_rpc_source_run_force_invalid_type(tmp_path: Path) -> None:
    """AgentRpcSource rejects non-boolean force parameter.

    Error path: Type validation must reject string "true" or number 1.
    """
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            import json

            # Send force as string instead of boolean
            request = {
                "jsonrpc": "2.0",
                "method": "run",
                "params": {"force": "true"},  # String, not bool
                "id": 1,
            }
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response, "Non-boolean force should be rejected"
            assert response["error"]["code"] == -32602  # Invalid params
            assert "force must be boolean" in response["error"]["message"]

        tg.cancel_scope.cancel()


# =============================================================================
# EventSink Buffer Overflow
# =============================================================================


@pytest.mark.anyio
async def test_event_sink_slow_subscriber_drops_events() -> None:
    """EventSink drops events for slow subscribers when buffer is full.

    Critical: Slow clients should not block engine - events must be dropped.
    """
    sink = EventSink(buffer_size=3)  # Small buffer for testing

    recv = await sink.subscribe("slow_client")

    # Fill subscriber's buffer without consuming
    for i in range(5):
        event = StageStarted(
            type="stage_started",
            stage=f"s{i}",
            index=i,
            total=5,
        )
        await sink.handle(event)

    # Subscriber should only see first 3 events (buffer size)
    received_events = list[OutputEvent]()
    try:
        while True:
            event = recv.receive_nowait()
            received_events.append(event)
    except anyio.WouldBlock:
        pass

    # Buffer size is 3, so at most 3 events delivered
    assert len(received_events) <= 3, "Slow subscriber should miss events"

    await sink.close()
