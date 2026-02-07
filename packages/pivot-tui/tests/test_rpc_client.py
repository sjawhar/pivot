# tests/tui/test_rpc_client.py
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anyio
import pytest

from pivot_tui import rpc_client

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.anyio
async def test_send_run_command_single_stage(tmp_path: Path) -> None:
    """send_run_command sends correct JSON-RPC for single stage."""
    socket_path = tmp_path / "agent.sock"
    received_requests = list[dict[str, Any]]()
    result: bool | None = None

    async def mock_server() -> None:
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                data = await conn.receive(4096)
                request = json.loads(data.decode().strip())
                received_requests.append(request)
                response = {"jsonrpc": "2.0", "result": "accepted", "id": request["id"]}
                await conn.send(json.dumps(response).encode() + b"\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(mock_server)
        await anyio.sleep(0.05)  # Let server start

        result = await rpc_client.send_run_command(socket_path, stages=["my_stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is True
    assert len(received_requests) == 1
    assert received_requests[0]["method"] == "run"
    assert received_requests[0]["params"]["stages"] == ["my_stage"]
    assert received_requests[0]["params"]["force"] is True


@pytest.mark.anyio
async def test_send_run_command_all_stages(tmp_path: Path) -> None:
    """send_run_command sends stages=None for all stages."""
    socket_path = tmp_path / "agent.sock"
    received_requests = list[dict[str, Any]]()
    result: bool | None = None

    async def mock_server() -> None:
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                data = await conn.receive(4096)
                request = json.loads(data.decode().strip())
                received_requests.append(request)
                response = {"jsonrpc": "2.0", "result": "accepted", "id": request["id"]}
                await conn.send(json.dumps(response).encode() + b"\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(mock_server)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=None, force=True)
        tg.cancel_scope.cancel()

    assert result is True
    assert received_requests[0]["params"]["stages"] is None


@pytest.mark.anyio
async def test_send_run_command_socket_not_found(tmp_path: Path) -> None:
    """send_run_command returns False when socket doesn't exist."""
    socket_path = tmp_path / "nonexistent.sock"

    result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)

    assert result is False


@pytest.mark.anyio
async def test_send_run_command_connection_refused(tmp_path: Path) -> None:
    """send_run_command returns False on connection error."""
    socket_path = tmp_path / "agent.sock"
    # Create file but not a socket
    socket_path.touch()

    result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)

    assert result is False


@pytest.mark.anyio
async def test_send_run_command_server_closes_without_response(tmp_path: Path) -> None:
    """send_run_command returns False when server closes connection without response."""
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def mock_server_close_immediately() -> None:
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            # Read request but close without sending response
            await conn.receive(4096)
            await conn.aclose()

    async with anyio.create_task_group() as tg:
        tg.start_soon(mock_server_close_immediately)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is False


@pytest.mark.anyio
async def test_send_run_command_chunked_response(tmp_path: Path) -> None:
    """send_run_command handles responses sent in multiple chunks."""
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def mock_server_chunked() -> None:
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                await conn.receive(4096)
                # Send response in multiple small chunks
                response = '{"jsonrpc": "2.0", "result": "accepted", "id": 1}\n'
                for char in response:
                    await conn.send(char.encode())
                    await anyio.sleep(0.001)

    async with anyio.create_task_group() as tg:
        tg.start_soon(mock_server_chunked)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is True


@pytest.mark.anyio
async def test_send_run_command_timeout(tmp_path: Path) -> None:
    """send_run_command returns False when server times out.

    This tests the TimeoutError exception path (rpc_client.py:56) which
    prevents the TUI from hanging when the server is unresponsive.
    """
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def slow_server() -> None:
        """Server that never responds, causing timeout."""
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                # Receive request but never send response
                await conn.receive(4096)
                # Sleep longer than RPC_TIMEOUT (5.0 seconds)
                await anyio.sleep(10.0)

    async with anyio.create_task_group() as tg:
        tg.start_soon(slow_server)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is False, "Should return False on timeout"


@pytest.mark.anyio
async def test_send_run_command_malformed_json_response(tmp_path: Path) -> None:
    """send_run_command returns False when server sends invalid JSON.

    This tests the json.JSONDecodeError exception path (rpc_client.py:59)
    which prevents the TUI from crashing on malformed server responses.
    """
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def bad_json_server() -> None:
        """Server that sends invalid JSON."""
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                await conn.receive(4096)
                # Send malformed JSON (not valid JSON-RPC)
                await conn.send(b"not json at all\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(bad_json_server)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is False, "Should return False on malformed JSON"


@pytest.mark.anyio
async def test_send_run_command_non_accepted_result(tmp_path: Path) -> None:
    """send_run_command returns False when server result is not 'accepted'.

    This tests the result comparison (rpc_client.py:55) to ensure non-accepted
    results (like 'rejected' or 'error') are properly handled.
    """
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def rejecting_server() -> None:
        """Server that rejects the request."""
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                data = await conn.receive(4096)
                request = json.loads(data.decode().strip())
                # Send valid JSON-RPC but with result != "accepted"
                response = {"jsonrpc": "2.0", "result": "rejected", "id": request["id"]}
                await conn.send(json.dumps(response).encode() + b"\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(rejecting_server)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is False, "Should return False when result is not 'accepted'"


@pytest.mark.anyio
async def test_send_run_command_error_response(tmp_path: Path) -> None:
    """send_run_command returns False when server sends JSON-RPC error.

    This tests error response handling to ensure JSON-RPC error responses
    don't crash the client.
    """
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def error_server() -> None:
        """Server that sends JSON-RPC error response."""
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                data = await conn.receive(4096)
                request = json.loads(data.decode().strip())
                # Send JSON-RPC error response
                response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": "Unknown stage"},
                    "id": request["id"],
                }
                await conn.send(json.dumps(response).encode() + b"\n")

    async with anyio.create_task_group() as tg:
        tg.start_soon(error_server)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is False, "Should return False on JSON-RPC error response"


@pytest.mark.anyio
async def test_send_run_command_partial_response(tmp_path: Path) -> None:
    """send_run_command returns False when server sends incomplete response.

    This tests handling of truncated responses that might occur with
    network issues or server crashes.
    """
    socket_path = tmp_path / "agent.sock"
    result: bool | None = None

    async def partial_response_server() -> None:
        """Server that sends incomplete JSON."""
        listener = await anyio.create_unix_listener(str(socket_path))
        async with listener:
            conn = await listener.accept()
            async with conn:
                await conn.receive(4096)
                # Send incomplete JSON (missing closing brace)
                await conn.send(b'{"jsonrpc": "2.0", "result": "accept\n')

    async with anyio.create_task_group() as tg:
        tg.start_soon(partial_response_server)
        await anyio.sleep(0.05)

        result = await rpc_client.send_run_command(socket_path, stages=["stage"], force=True)
        tg.cancel_scope.cancel()

    assert result is False, "Should return False on incomplete response"
