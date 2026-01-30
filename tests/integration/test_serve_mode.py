"""Integration tests for serve mode (headless daemon via Engine).

Tests E2E flow: socket creation, status queries, event broadcast.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from typing import TYPE_CHECKING

import anyio
import pytest

from helpers import register_test_stage, wait_for_socket
from pivot import config
from pivot.engine.agent_rpc import AgentRpcHandler, AgentRpcSource, EventSink
from pivot.engine.engine import Engine
from pivot.engine.sources import OneShotSource

if TYPE_CHECKING:
    import pathlib

    from pivot.engine.types import StageStarted
    from pivot.pipeline.pipeline import Pipeline


def _helper_noop(params: None) -> dict[str, str]:
    """No-op stage for testing."""
    return {"result": "ok"}


# =============================================================================
# Socket Integration Tests
# =============================================================================


@pytest.fixture
def socket_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a path for the Unix socket."""
    return tmp_path / "agent.sock"


@pytest.mark.anyio
async def test_socket_creation_and_permissions(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource creates socket with correct permissions."""

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    register_test_stage(_helper_noop, name="test_stage")

    rpc_source = AgentRpcSource(socket_path=socket_path)

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)

            # Wait for socket to be created
            await wait_for_socket(socket_path)

            # Check permissions (owner-only)
            mode = socket_path.stat().st_mode
            assert mode & 0o777 == 0o600, f"Expected 0o600, got {oct(mode & 0o777)}"

            tg.cancel_scope.cancel()


# =============================================================================
# EventSink Integration Tests
# =============================================================================


@pytest.mark.anyio
async def test_event_sink_broadcasts_to_multiple_subscribers() -> None:
    """EventSink broadcasts events to all subscribers."""
    sink = EventSink()

    recv1 = await sink.subscribe("client1")
    recv2 = await sink.subscribe("client2")

    # Emit an event
    event: StageStarted = {
        "type": "stage_started",
        "stage": "test",
        "index": 0,
        "total": 1,
    }
    await sink.handle(event)

    # Both clients should receive it
    event1 = recv1.receive_nowait()
    event2 = recv2.receive_nowait()

    # Type narrow using discriminated union on "type" field
    assert event1["type"] == "stage_started" and event1["stage"] == "test"
    assert event2["type"] == "stage_started" and event2["stage"] == "test"

    await sink.close()


@pytest.mark.anyio
async def test_unsubscribe_stops_event_delivery() -> None:
    """Unsubscribed clients stop receiving events."""
    sink = EventSink()

    recv = await sink.subscribe("client")

    # Unsubscribe
    await sink.unsubscribe("client")

    # Emit an event
    event: StageStarted = {
        "type": "stage_started",
        "stage": "test",
        "index": 0,
        "total": 1,
    }
    await sink.handle(event)

    # Channel should be closed - raises EndOfStream on closed channels
    with pytest.raises(anyio.EndOfStream):
        recv.receive_nowait()


# =============================================================================
# AgentRpcHandler Integration Tests
# =============================================================================


@pytest.mark.anyio
async def test_agent_rpc_handler_status_query(test_pipeline: Pipeline) -> None:
    """AgentRpcHandler.handle_query returns status correctly."""
    async with Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine)

        result = await handler.handle_query("status")
        # Type narrow by checking for "state" key (discriminates QueryStatusResult from other types)
        assert "state" in result and result["state"] == "idle"
        assert "running" in result and result["running"] == []
        assert "pending" in result and result["pending"] == []


@pytest.mark.anyio
async def test_agent_rpc_handler_stages_query(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcHandler.handle_query returns stages correctly."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    register_test_stage(_helper_noop, name="stage_a")
    register_test_stage(_helper_noop, name="stage_b")

    async with Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine)

        result = await handler.handle_query("stages")

        assert "stages" in result
        assert set(result["stages"]) >= {"stage_a", "stage_b"}


@pytest.mark.anyio
async def test_agent_rpc_handler_unknown_method() -> None:
    """AgentRpcHandler.handle_query raises ValueError for unknown methods."""
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        with pytest.raises(ValueError, match="Unknown query method"):
            await handler.handle_query("unknown_method")


@pytest.mark.anyio
async def test_agent_rpc_handler_stages_query_no_pipeline() -> None:
    """AgentRpcHandler.handle_query returns empty list when no pipeline."""
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        result = await handler.handle_query("stages")

        assert result == {"stages": []}


# =============================================================================
# Socket Buffer Handling Tests
# =============================================================================


@pytest.mark.anyio
async def test_socket_handles_chunked_messages(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource correctly reassembles messages split across chunks."""

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Need to create engine first to create handler, then create source with handler
    async with Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine)
        rpc_source = AgentRpcSource(socket_path=socket_path, handler=handler)
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)

            # Wait for socket to be created
            await wait_for_socket(socket_path)

            # Connect and send a request in multiple chunks
            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Split the JSON-RPC request into chunks
                request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
                chunk1 = request[:10].encode()  # '{"jsonrpc"'
                chunk2 = request[10:25].encode()  # ':"2.0","method"'
                chunk3 = request[25:].encode()  # ':"status","id":1}\n'

                await conn.send(chunk1)
                await anyio.sleep(0.05)  # Brief delay between chunks
                await conn.send(chunk2)
                await anyio.sleep(0.05)
                await conn.send(chunk3)

                # Read response
                response_data = await conn.receive(1024)
                response = json.loads(response_data.decode().strip())

                assert response["id"] == 1
                assert "result" in response
                assert response["result"]["state"] == "idle"

            tg.cancel_scope.cancel()


# =============================================================================
# Sink Error Handling Tests
# =============================================================================


class _FailingSink:
    """Test sink that raises an exception on handle()."""

    async def handle(self, event: object) -> None:
        raise RuntimeError("Sink failure for testing")

    async def close(self) -> None:
        pass


class _CollectingSink:
    """Test sink that collects events for verification."""

    events: list[object]

    def __init__(self) -> None:
        self.events = list[object]()

    async def handle(self, event: object) -> None:
        self.events.append(event)

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_sink_error_does_not_crash_engine(test_pipeline: Pipeline) -> None:
    """Errors in sink.handle() are logged but don't crash the engine."""

    collecting_sink = _CollectingSink()

    async with Engine(pipeline=test_pipeline) as engine:
        # Add a failing sink and a collecting sink
        engine.add_sink(_FailingSink())
        engine.add_sink(collecting_sink)

        # Add source to trigger a run
        source = OneShotSource(
            stages=None,
            force=True,
            reason="test",
        )
        engine.add_source(source)

        # Run should complete despite the failing sink
        await engine.run(exit_on_completion=True)

    # The collecting sink should have received events despite the failing sink
    assert len(collecting_sink.events) > 0
    event_types = [e.get("type") for e in collecting_sink.events if isinstance(e, dict)]
    assert "engine_state_changed" in event_types


# =============================================================================
# CLI Serve Mode E2E Tests
# =============================================================================


def test_serve_mode_cli_responds_to_status_query(
    tmp_path: pathlib.Path,
) -> None:
    """E2E test: pivot repro --watch --serve creates working RPC endpoint.

    This test verifies the complete CLI integration - that _run_serve_mode
    correctly wires up AgentRpcHandler and EventSink. Without this test,
    we caught bugs where components worked individually but weren't connected.
    """

    # Create minimal project structure with a valid pipeline
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    pipeline_code = """\
import pathlib
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)

def noop_stage() -> None:
    pass

pipeline.register(noop_stage, name="noop")
"""
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    socket_path = tmp_path / ".pivot" / "agent.sock"

    # Start serve mode as subprocess
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    proc = subprocess.Popen(
        ["uv", "run", "pivot", "repro", "--watch", "--serve"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for socket to be created (poll up to 10s)
        for _ in range(100):
            if socket_path.exists():
                break
            time.sleep(0.1)
        else:
            stdout, stderr = proc.communicate(timeout=1)
            pytest.fail(
                "Socket not created within 10s.\n"
                + f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        # Connect and send status query
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(str(socket_path))
            request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
            sock.sendall(request.encode())
            response_data = sock.recv(1024)
            response = json.loads(response_data.decode().strip())

        # Should get valid status response, NOT "Method not found" error
        assert "error" not in response, f"Expected status result, got error: {response}"
        assert "result" in response
        assert response["result"]["state"] in ("idle", "active")

        # Also test stages query
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(str(socket_path))
            request = '{"jsonrpc":"2.0","method":"stages","id":2}\n'
            sock.sendall(request.encode())
            response_data = sock.recv(1024)
            response = json.loads(response_data.decode().strip())

        assert "error" not in response, f"Expected stages result, got error: {response}"
        assert "result" in response
        assert "stages" in response["result"]

    finally:
        # Clean up subprocess
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# =============================================================================
# AgentRpcSource Handler Tests (Component-level)
# =============================================================================


@pytest.mark.anyio
async def test_rpc_source_without_handler_returns_method_not_found(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource without handler returns 'Method not found' for queries.

    Copilot Review Comment 1: This test demonstrates the BUG - when AgentRpcSource
    is created without a handler (as _run_serve_mode currently does), status
    queries fail with 'Method not found' instead of returning engine state.

    The fix: _run_serve_mode must create AgentRpcHandler and pass it to
    AgentRpcSource(socket_path=socket_path, handler=handler).
    """

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create source WITHOUT handler (mimics current _run_serve_mode bug)
    rpc_source = AgentRpcSource(socket_path=socket_path, handler=None)

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)
            await wait_for_socket(socket_path)

            # Connect and send status query
            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
                await conn.send(request.encode())
                response_data = await conn.receive(1024)
                response = json.loads(response_data.decode().strip())

                # BUG: Without handler, queries return "Method not found"
                assert "error" in response, "Expected error without handler"
                assert response["error"]["code"] == -32601  # Method not found

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_rpc_source_with_handler_returns_status(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource WITH handler returns valid status for queries.

    This is the correct behavior - when handler is provided, status queries work.
    """

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    async with Engine(pipeline=test_pipeline) as engine:
        # Create source WITH handler (correct implementation)
        handler = AgentRpcHandler(engine=engine)
        rpc_source = AgentRpcSource(socket_path=socket_path, handler=handler)
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)
            await wait_for_socket(socket_path)

            # Connect and send status query
            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
                await conn.send(request.encode())
                response_data = await conn.receive(1024)
                response = json.loads(response_data.decode().strip())

                # SUCCESS: With handler, queries return valid results
                assert "error" not in response, f"Got error: {response}"
                assert "result" in response
                assert response["result"]["state"] in ("idle", "active")

            tg.cancel_scope.cancel()


# =============================================================================
# EventBuffer Integration Tests
# =============================================================================


@pytest.mark.anyio
async def test_event_buffer_integration_with_engine(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """EventBuffer correctly captures events from engine execution.

    Integration: Verifies buffer is wired as sink and receives events.
    """
    from pivot import config
    from pivot.engine.agent_rpc import EventBuffer
    from pivot.engine.sources import OneShotSource

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    register_test_stage(_helper_noop, name="test_stage")

    buffer = EventBuffer(max_events=100)

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_sink(buffer)
        source = OneShotSource(stages=None, force=True, reason="test")
        engine.add_source(source)

        await engine.run(exit_on_completion=True)

    # Buffer should have captured events
    result = buffer.events_since(0)
    assert result["version"] > 0, "Buffer should have recorded events"
    assert len(result["events"]) > 0, "Buffer should contain events from execution"

    # Check for expected event types
    event_types = [e["event"]["type"] for e in result["events"]]
    assert "engine_state_changed" in event_types


@pytest.mark.anyio
async def test_event_buffer_polling_during_execution(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """EventBuffer allows polling events_since during execution.

    Integration: Simulates client polling while engine is running.
    """
    import time

    from pivot import config
    from pivot.engine.agent_rpc import EventBuffer
    from pivot.engine.sources import OneShotSource

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    def _slow_stage(params: None) -> dict[str, str]:
        """Stage that takes time to execute."""
        time.sleep(0.2)
        return {"result": "done"}

    register_test_stage(_slow_stage, name="slow_stage")

    buffer = EventBuffer(max_events=100)
    versions_seen = list[int]()

    async def poll_buffer() -> None:
        """Simulate client polling every 50ms."""
        last_version = 0
        for _ in range(10):
            await anyio.sleep(0.05)
            result = buffer.events_since(last_version)
            if result["version"] > last_version:
                versions_seen.append(result["version"])
                last_version = result["version"]

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_sink(buffer)
        source = OneShotSource(stages=None, force=True, reason="test")
        engine.add_source(source)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poll_buffer)

            async def run_engine() -> None:
                await engine.run(exit_on_completion=True)

            tg.start_soon(run_engine)

    # Should have seen multiple version updates during execution
    assert len(versions_seen) > 0, "Polling should capture events during execution"


# =============================================================================
# Error Handling Edge Cases
# =============================================================================
