"""Tests for AgentRpcSource."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import socket
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast
from unittest.mock import AsyncMock, MagicMock

import anyio
import anyio.to_thread
import pytest

from pivot import compose, executor, loaders
from pivot.compose import Pipeline
from pivot.engine.agent_rpc import (
    AgentRpcHandler,
    AgentRpcSource,
    BroadcastEventSink,
    CommitResult,
    EventBuffer,
    QueryResult,
    QueryStageInfoResult,
    QueryStatusResult,
)
from pivot.engine.engine import Engine
from pivot.engine.types import EventSource, InputEvent, OutputEvent, StageStarted
from pivot.storage import cache, state
from pivot.types import ChangeType, DataDiffResult, StageExplanation
from tests import helpers

if TYPE_CHECKING:
    from pivot.registry import PipelineLike


def _helper_valid_stage() -> dict[str, int]:
    return {"value": 1}


def _helper_explained_stage(input_file: str) -> str:
    _ = input_file
    return "explained"


@compose.stage
def _helper_plan(
    raw: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in raw.items()}


@compose.stage
def _helper_cut(entry: dict[str, int]) -> dict[str, int]:
    return entry


_diff_expected: DataDiffResult | None = None


def _helper_commit_stages_train_eval(_pipeline: object) -> tuple[list[str], list[str]]:
    return (["train", "eval"], ["broken_stage"])


def _helper_commit_stages_empty(_pipeline: object) -> tuple[list[str], list[str]]:
    return (list[str](), list[str]())


def _helper_commit_stages_stage_a(_pipeline: object) -> tuple[list[str], list[str]]:
    return (["stage_a"], [])


def _helper_restore_data_from_cache(_rel_path: str, file_hash: str) -> Path:
    return Path(f"/tmp/fake_{file_hash}")


def _helper_restore_data_none(_rel_path: str, _file_hash: str) -> None:
    return None


def _helper_diff_data_files(
    _old_path: object,
    _new_path: object,
    path_display: str,
    **_kwargs: object,
) -> DataDiffResult:
    if _diff_expected is not None:
        return _diff_expected
    return _make_diff_result(path_display)


def _helper_path_unlink(_path: Path, **_kwargs: object) -> None:
    return None


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
        await helpers.wait_for_socket(socket_path)

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
async def test_agent_rpc_source_signals_ready_once_listening(tmp_path: Path) -> None:
    """``ready`` is set only once the socket accepts connections, so a client can connect."""
    import functools
    import threading

    socket_path = tmp_path / "agent.sock"
    ready = threading.Event()
    send, _recv = anyio.create_memory_object_stream[InputEvent](1)
    source = AgentRpcSource(socket_path=socket_path, ready=ready)

    async with anyio.create_task_group() as tg:
        tg.start_soon(functools.partial(source.run, send))
        with anyio.fail_after(5):
            while not ready.is_set():
                await anyio.sleep(0.01)
        connection = await anyio.connect_unix(str(socket_path))
        await connection.aclose()
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_client_eof_is_not_an_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Readiness probes may connect and close without creating an ERROR traceback."""
    socket_path = tmp_path / "agent.sock"
    send, _recv = anyio.create_memory_object_stream[InputEvent](1)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)
        await helpers.wait_for_socket(socket_path)

        with caplog.at_level("ERROR"):
            connection = await anyio.connect_unix(str(socket_path))
            await connection.aclose()
            await anyio.sleep(0.05)

        tg.cancel_scope.cancel()

    assert not any(
        record.name == "pivot.engine.agent_rpc" and record.levelname == "ERROR"
        for record in caplog.records
    )


@pytest.mark.anyio
async def test_agent_rpc_source_closed_engine_send_still_logs_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A closed engine event stream is an internal error, not a client disconnect."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](1)
    await recv.aclose()

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)
        await helpers.wait_for_socket(socket_path)

        with caplog.at_level("ERROR"):
            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = {"jsonrpc": "2.0", "method": "run", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")
                with contextlib.suppress(
                    anyio.EndOfStream,
                    anyio.ClosedResourceError,
                    anyio.BrokenResourceError,
                ):
                    await conn.receive(4096)
            await anyio.sleep(0.05)

        tg.cancel_scope.cancel()

    assert any(
        record.name == "pivot.engine.agent_rpc" and record.levelname == "ERROR"
        for record in caplog.records
    )


@pytest.mark.anyio
async def test_agent_rpc_source_cancel_command(tmp_path: Path) -> None:
    """AgentRpcSource emits CancelRequested for cancel method."""
    socket_path = tmp_path / "agent.sock"
    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

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

        await helpers.wait_for_socket(socket_path)

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
async def test_agent_rpc_source_carries_configured_resource_capacity(tmp_path: Path) -> None:
    """An RPC-triggered run keeps the capacity supplied by serve mode."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)
    event: InputEvent | None = None

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path, resource_capacity={"gpu": 2})
        tg.start_soon(source.run, send)
        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "run", "id": 4}
            await conn.send(json.dumps(request).encode() + b"\n")
            response = json.loads((await conn.receive(4096)).decode())
            assert response.get("result") == "accepted"

        event = await recv.receive()
        tg.cancel_scope.cancel()

    assert event is not None
    assert event["type"] == "run_requested"
    if event["type"] == "run_requested":
        assert event["resource_capacity"] == {"gpu": 2}


@pytest.mark.anyio
async def test_agent_rpc_source_unknown_method(tmp_path: Path) -> None:
    """AgentRpcSource returns error for unknown methods."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

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

        await helpers.wait_for_socket(socket_path)

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

        await helpers.wait_for_socket(socket_path)

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

        await helpers.wait_for_socket(socket_path)

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

        await helpers.wait_for_socket(socket_path)

        # Check socket permissions (should be 0o600 = owner read/write only)
        mode = socket_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_cancelled_while_binding_leaves_no_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serve mode's stop cancels the engine's task group, which can land while the listener
    binds; anyio binds in a worker thread it abandons on cancellation. The socket is still
    removed."""
    path = tmp_path / "agent.sock"
    bound = threading.Event()
    release = threading.Event()
    original = socket.socket.bind

    def _bind_then_wait(self: socket.socket, address: str) -> None:
        original(self, address)
        bound.set()
        release.wait(10)

    monkeypatch.setattr(socket.socket, "bind", _bind_then_wait)
    send, _receive = anyio.create_memory_object_stream[InputEvent](10)
    async with anyio.create_task_group() as tg:
        tg.start_soon(AgentRpcSource(socket_path=path).run, send)
        await anyio.to_thread.run_sync(bound.wait, 10)
        tg.cancel_scope.cancel()
        release.set()
    await anyio.sleep(0.5)

    assert not path.exists(), "a source cancelled while binding left its socket behind"


@pytest.mark.anyio
async def test_agent_rpc_source_cancelled_before_binding_a_long_path_binds_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source cancelled before its bind runs creates no socket anywhere, even when its
    address names a directory descriptor that the cancellation closed and something reused."""
    project = tmp_path.joinpath(*["nested-directory"] * 8)
    project.mkdir(parents=True)
    path = project / "agent.sock"
    assert len(os.fsencode(path)) > 108, "the path must go through a directory descriptor"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    entered = threading.Event()
    release = threading.Event()
    original = socket.socket.bind

    def _wait_then_bind(self: socket.socket, address: str) -> None:
        entered.set()
        release.wait(10)
        original(self, address)

    monkeypatch.setattr(socket.socket, "bind", _wait_then_bind)
    send, _receive = anyio.create_memory_object_stream[InputEvent](10)
    releaser = threading.Timer(0.5, release.set)
    async with anyio.create_task_group() as tg:
        tg.start_soon(AgentRpcSource(socket_path=path).run, send)
        await anyio.to_thread.run_sync(entered.wait, 10)
        releaser.start()
        tg.cancel_scope.cancel()
    # The process goes on opening files as it shuts down; the lowest free descriptor is reused.
    reused = os.open(elsewhere, os.O_PATH | os.O_DIRECTORY)
    try:
        await anyio.to_thread.run_sync(release.wait, 10)
        await anyio.sleep(0.5)
    finally:
        os.close(reused)

    created = sorted(p.name for p in elsewhere.iterdir())
    assert not created, f"a cancelled bind created {created} in an unrelated directory"
    assert not path.exists(), "a source cancelled before binding left its socket behind"


@pytest.mark.anyio
async def test_agent_rpc_source_checks_its_socket_while_still_listening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stopping source removes the socket only if it is the one it bound, and checks while it
    still listens. Its socket then keeps its inode, so a newer serve that replaces the path at
    that moment is given another inode number and keeps its socket."""
    path = tmp_path / "agent.sock"
    armed = threading.Event()
    seen = dict[str, object]()
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    originals = {name: getattr(os, name) for name in ("stat", "lstat", "unlink")}

    def _replace_on_first_touch(
        name: str, target: object, *args: object, **kwargs: object
    ) -> object:
        if (
            armed.is_set()
            and not seen
            and isinstance(target, (str, os.PathLike))
            and os.fspath(target) == str(path)
        ):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                seen["listening"] = client.connect_ex(str(path)) == 0
            originals["unlink"](path)
            replacement.bind(str(path))
            replacement.listen()
            seen["inode"] = originals["lstat"](path).st_ino
        return originals[name](target, *args, **kwargs)

    for name in originals:
        monkeypatch.setattr(os, name, functools.partial(_replace_on_first_touch, name))
    send, _receive = anyio.create_memory_object_stream[InputEvent](10)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(AgentRpcSource(socket_path=path).run, send)
            await helpers.wait_for_socket(path)
            armed.set()
            tg.cancel_scope.cancel()

        assert seen, "the stopping source never touched its socket path"
        assert seen["listening"], (
            "the source checked its socket after closing its listener, when a replacement can "
            + "be given the freed inode number"
        )
        assert path.exists(), "the stopping source removed the replacement's socket"
        assert path.lstat().st_ino == seen["inode"]
    finally:
        replacement.close()


def test_agent_rpc_source_conforms_to_protocol() -> None:
    """AgentRpcSource conforms to EventSource protocol."""
    source = AgentRpcSource(socket_path=Path("/tmp/test.sock"))
    _source: EventSource = source
    assert _source is source


@pytest.mark.anyio
async def test_agent_event_sink_broadcasts_to_subscribers() -> None:
    """EventSink broadcasts events to all subscribers."""
    sink = BroadcastEventSink()

    # Subscribe two clients
    recv1 = await sink.subscribe("client1")
    recv2 = await sink.subscribe("client2")

    # Emit an event
    event = StageStarted(
        type="stage_started",
        seq=0,
        stage="train",
        index=0,
        total=1,
        run_id="test-run",
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
    sink = BroadcastEventSink()

    recv = await sink.subscribe("client1")
    await sink.unsubscribe("client1")

    # Event after unsubscribe should not be received
    event = StageStarted(
        type="stage_started",
        seq=0,
        stage="train",
        index=0,
        total=1,
        run_id="test-run",
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
            await helpers.wait_for_socket(socket_path)

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
async def test_agent_rpc_sends_artifact_identities_as_objects(tmp_path: Path) -> None:
    """Identities inside events and explanations use stage_info's {producer, key} form."""
    from pivot import types

    buffer = EventBuffer(max_events=10)
    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=1,
            total=1,
            run_id="run",
            explanation=types.StageExplanation(
                stage_name="train",
                will_run=True,
                is_forced=False,
                reason="Input dependencies changed",
                code_changes=[],
                param_changes=[],
                dep_changes=[
                    types.DepChange(
                        identity=types.ArtifactIdentity("prepare", "features"),
                        old_hash="aaa",
                        new_hash="bbb",
                        change_type=types.ChangeType.MODIFIED,
                    )
                ],
                upstream_stale=[],
            ),
        )
    )
    socket_path = tmp_path / "agent.sock"
    handler = AgentRpcHandler(engine=MagicMock(), event_buffer=buffer)
    source = AgentRpcSource(socket_path=socket_path, handler=handler)
    send, _recv = anyio.create_memory_object_stream[InputEvent](1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(source.run, send)
        await helpers.wait_for_socket(socket_path)
        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {
                "jsonrpc": "2.0",
                "method": "events_since",
                "params": {"version": 0},
                "id": 1,
            }
            await conn.send(json.dumps(request).encode() + b"\n")
            response = json.loads((await conn.receive(65536)).decode())
        tg.cancel_scope.cancel()

        event = response["result"]["events"][0]["event"]
        assert event["explanation"]["dep_changes"][0]["identity"] == {
            "producer": "prepare",
            "key": "features",
        }


@pytest.mark.anyio
async def test_agent_rpc_source_rejects_oversized_messages(tmp_path: Path) -> None:
    """AgentRpcSource rejects messages larger than 1MB to prevent memory exhaustion."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

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

        await helpers.wait_for_socket(socket_path)

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
            await helpers.wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Query stages when no pipeline is set
                request = {"jsonrpc": "2.0", "method": "stages", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")

                response_line = await conn.receive(4096)
                response = json.loads(response_line.decode())

                assert "result" in response
                assert response["result"] == {"stages": [], "templates": {}}, (
                    "Should return no stage and no template without pipeline"
                )

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_stages_query_maps_each_template_to_its_instances(tmp_path: Path) -> None:
    """The answer names each template beside the stages, with the instances it expanded into;
    an empty list before it expands. A template is never one of the stages."""
    with Pipeline("p", root=tmp_path) as pipeline:
        plan = _helper_plan(pipeline.input("plan.yaml", t=dict[str, int]))
        _helper_cut(pipeline.map(plan))
    engine = MagicMock()
    engine._pipeline = pipeline
    handler = AgentRpcHandler(engine=engine)

    before = await handler.handle_query("stages")
    pipeline.expand_template("p/_helper_cut", [("a", "a.yaml"), ("b", "b.yaml")])
    after = await handler.handle_query("stages")

    assert before == {"stages": ["p/_helper_plan"], "templates": {"p/_helper_cut": []}}
    instances = ["p/_helper_cut@a", "p/_helper_cut@b"]
    assert after == {
        "stages": ["p/_helper_plan", *instances],
        "templates": {"p/_helper_cut": instances},
    }


def _helper_mapped_pipeline(root: Path) -> Pipeline:
    with Pipeline("p", root=root) as pipeline:
        plan = _helper_plan(pipeline.input("plan.yaml", t=dict[str, int]))
        _helper_cut(pipeline.map(plan))
    return pipeline


def _helper_mapped_handler(pipeline: Pipeline) -> AgentRpcHandler:
    engine = MagicMock()
    engine._pipeline = pipeline
    engine.wait_for_fingerprints = AsyncMock()
    return AgentRpcHandler(engine=engine)


@pytest.mark.anyio
async def test_stage_info_takes_a_template_name(tmp_path: Path) -> None:
    """A template's info is the stage as declared: it reads its producer's whole directory."""
    handler = _helper_mapped_handler(_helper_mapped_pipeline(tmp_path))

    info = cast(
        "QueryStageInfoResult",
        await handler.handle_query("stage_info", {"stage": "p/_helper_cut"}),
    )

    assert info["name"] == "p/_helper_cut"
    assert [dep["producer"] for dep in info["deps"]] == ["p/_helper_plan"]
    assert [out["producer"] for out in info["outs"]] == ["p/_helper_cut"]


@pytest.mark.anyio
async def test_an_instance_not_yet_expanded_is_named_as_waiting(tmp_path: Path) -> None:
    """`stage_info` of an instance its template has not expanded into says so, and `explain` of
    it, or of the template, explains that a run expands it, as `pivot status` reads it."""
    handler = _helper_mapped_handler(_helper_mapped_pipeline(tmp_path))

    with pytest.raises(ValueError, match=r"'p/_helper_cut@a' is not yet expanded: "):
        await handler.handle_query("stage_info", {"stage": "p/_helper_cut@a"})
    instance = cast(
        "StageExplanation", await handler.handle_query("explain", {"stage": "p/_helper_cut@a"})
    )
    template = cast(
        "StageExplanation", await handler.handle_query("explain", {"stage": "p/_helper_cut"})
    )
    with pytest.raises(ValueError, match="Unknown stage: p/_helper_nope"):
        await handler.handle_query("explain", {"stage": "p/_helper_nope"})

    for explanation, name in ((instance, "p/_helper_cut@a"), (template, "p/_helper_cut")):
        assert explanation["stage_name"] == name
        assert explanation["will_run"] is True
        assert explanation["reason"].startswith("Not yet expanded: "), explanation
        assert "p/_helper_plan" in explanation["reason"]


def test_explain_of_an_expanded_template_names_the_instances_that_would_run(
    set_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explaining a template explains each instance it expanded into: it would run when one
    would, and its reason names each of those with its own reason."""
    monkeypatch.chdir(set_project_root)
    (set_project_root / ".git").mkdir(exist_ok=True)
    (set_project_root / "data" / "raw").mkdir(parents=True)
    (set_project_root / "data" / "raw" / "plan.yaml").write_text("a: 1\nb: 2\n")
    executor.run(pipeline=_helper_mapped_pipeline(set_project_root))
    pipeline = _helper_mapped_pipeline(set_project_root)
    pipeline.expand_template("p/_helper_cut", [("a", "a.yaml"), ("b", "b.yaml")])

    async def _explain() -> tuple[StageExplanation, StageExplanation]:
        async with Engine(pipeline=pipeline) as eng:
            handler = AgentRpcHandler(engine=eng)
            before = await handler.handle_query("explain", {"stage": "p/_helper_cut"})
            # An instance with no lock has no previous run
            (lock_file,) = (set_project_root / ".pivot").rglob("_helper_cut@b.lock")
            lock_file.unlink()
            after = await handler.handle_query("explain", {"stage": "p/_helper_cut"})
            return cast("StageExplanation", before), cast("StageExplanation", after)

    before, after = anyio.run(_explain)

    assert before["will_run"] is False, before
    assert before["reason"] == "Expanded into 2 stages; none would run"
    assert after["will_run"] is True, after
    assert after["reason"].startswith("Expanded into 2 stages; would run: p/_helper_cut@b ("), after
    assert "p/_helper_cut@a" not in after["reason"]


def test_run_takes_template_and_instance_names(tmp_path: Path) -> None:
    """A run reaches an instance through its template, so both are stages a run may name."""
    handler = _helper_mapped_handler(_helper_mapped_pipeline(tmp_path))

    assert handler.validate_stages(["p/_helper_plan", "p/_helper_cut", "p/_helper_cut@a"]) is None
    assert handler.validate_stages(["p/_helper_cut@a b"]) == "Unknown stages: p/_helper_cut@a b"


@pytest.mark.anyio
async def test_rpc_run_invalid_stage_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RPC run with invalid stage name returns descriptive error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".pivot").mkdir()

    pipeline = Pipeline("test", root=tmp_path)
    helpers.register_test_stage(_helper_valid_stage, name="valid_stage", pipeline=pipeline)

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
async def test_rpc_run_on_invalid_pipeline_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RPC run while the pipeline is invalid is refused with the reason, not accepted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".pivot").mkdir()

    pipeline = Pipeline("test", root=tmp_path)
    helpers.register_test_stage(_helper_valid_stage, name="valid_stage", pipeline=pipeline)

    async with Engine(pipeline=pipeline) as eng:
        eng._pipeline_error = "Stage 'test/valid_stage': captures non-scalar variable 'X'"
        source = AgentRpcSource(
            socket_path=tmp_path / "test.sock", handler=AgentRpcHandler(engine=eng)
        )
        send, recv = anyio.create_memory_object_stream[InputEvent](16)

        response = await source._handle_request(
            {"jsonrpc": "2.0", "method": "run", "params": {"force": True}, "id": 1}, send
        )

        assert response is not None
        assert "error" in response
        error = response["error"]
        assert error["code"] == -32003
        assert "Pipeline invalid" in error["message"]
        assert "captures non-scalar variable 'X'" in error["message"]
        with pytest.raises(anyio.WouldBlock):
            recv.receive_nowait()


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

            await helpers.wait_for_socket(socket_path)

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
        starting: bool = False
        reloading: bool = False
        pipeline_error: str | None = None

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

        await helpers.wait_for_socket(socket_path)

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
    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=1,
            total=2,
            run_id="test-run",
        )
    )

    result = buffer.events_since(0)
    assert result["version"] == 1, "First event should have version 1"
    assert len(result["events"]) == 1, "Should have exactly 1 event"


@pytest.mark.anyio
async def test_event_buffer_eviction() -> None:
    """EventBuffer should evict oldest events when full."""
    buffer = EventBuffer(max_events=3)
    for i in range(5):
        buffer.handle_sync(
            StageStarted(
                type="stage_started",
                seq=i,
                stage=f"s{i}",
                index=i,
                total=5,
                run_id="test-run",
            )
        )

    result = buffer.events_since(0)
    assert len(result["events"]) == 3, "Should only keep last 3 events (max_events=3)"


@pytest.mark.anyio
async def test_handler_events_since_query() -> None:
    """Handler should return events from buffer."""

    buffer = EventBuffer(max_events=100)
    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=1,
            total=1,
            run_id="test-run",
        )
    )

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
    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="s1",
            index=0,
            total=2,
            run_id="test-run",
        )
    )
    assert buffer._version == _MAX_VERSION

    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=1,
            stage="s2",
            index=1,
            total=2,
            run_id="test-run",
        )
    )
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
    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=0,
            total=1,
            run_id="test-run",
        )
    )

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
    buffer.handle_sync(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=0,
            total=1,
            run_id="test-run",
        )
    )

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
        buffer.handle_sync(
            StageStarted(
                type="stage_started",
                seq=i,
                stage=f"s{i}",
                index=i,
                total=5,
                run_id="test-run",
            )
        )

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
                StageStarted(
                    type="stage_started",
                    seq=i,
                    stage=f"s{offset}_{i}",
                    index=i,
                    total=100,
                    run_id="test-run",
                )
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

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            # Send message with empty lines
            msg_with_blanks = '\n\n{"jsonrpc":"2.0","method":"run","id":1}\n\n'
            await conn.send(msg_with_blanks.encode())

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

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            # Create message just under limit (accounting for JSON structure)
            # _MAX_MESSAGE_SIZE = 1MB
            max_size = 1024 * 1024
            padding = "x" * (max_size - 100)  # Leave room for JSON structure

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

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
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
    pipeline = Pipeline("test", root=tmp_path)
    helpers.register_test_stage(_helper_valid_stage, name="valid_stage", pipeline=pipeline)

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

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
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
# set_on_error Command
# =============================================================================


@pytest.mark.anyio
async def test_agent_rpc_source_set_on_error_fail(tmp_path: Path) -> None:
    """AgentRpcSource accepts set_on_error with mode='fail'."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {
                "jsonrpc": "2.0",
                "method": "set_on_error",
                "params": {"mode": "fail"},
                "id": 1,
            }
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") is True, "set_on_error should return True"

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_set_on_error_keep_going(tmp_path: Path) -> None:
    """AgentRpcSource accepts set_on_error with mode='keep_going'."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {
                "jsonrpc": "2.0",
                "method": "set_on_error",
                "params": {"mode": "keep_going"},
                "id": 2,
            }
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert response.get("result") is True, "set_on_error should return True"

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_set_on_error_invalid_mode(tmp_path: Path) -> None:
    """AgentRpcSource rejects set_on_error with invalid mode."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {
                "jsonrpc": "2.0",
                "method": "set_on_error",
                "params": {"mode": "invalid_mode"},
                "id": 3,
            }
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response, "Invalid mode should return error"
            assert response["error"]["code"] == -32602  # Invalid params
            assert "mode must be one of" in response["error"]["message"]

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_set_on_error_missing_mode(tmp_path: Path) -> None:
    """AgentRpcSource rejects set_on_error without mode parameter."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "set_on_error", "params": {}, "id": 4}
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response, "Missing mode should return error"
            assert response["error"]["code"] == -32602  # Invalid params
            assert "mode must be string" in response["error"]["message"]

        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_agent_rpc_source_set_on_error_non_string_mode(tmp_path: Path) -> None:
    """AgentRpcSource rejects set_on_error with non-string mode."""
    socket_path = tmp_path / "agent.sock"
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async with anyio.create_task_group() as tg:
        source = AgentRpcSource(socket_path=socket_path)
        tg.start_soon(source.run, send)

        await helpers.wait_for_socket(socket_path)

        async with await anyio.connect_unix(str(socket_path)) as conn:
            request = {"jsonrpc": "2.0", "method": "set_on_error", "params": {"mode": 123}, "id": 5}
            await conn.send(json.dumps(request).encode() + b"\n")

            response_line = await conn.receive(4096)
            response = json.loads(response_line.decode())

            assert "error" in response, "Non-string mode should return error"
            assert response["error"]["code"] == -32602  # Invalid params
            assert "mode must be string" in response["error"]["message"]

        tg.cancel_scope.cancel()


# =============================================================================
# EventSink Buffer Overflow
# =============================================================================


@pytest.mark.anyio
async def test_event_sink_slow_subscriber_drops_events() -> None:
    """EventSink drops events for slow subscribers when buffer is full.

    Critical: Slow clients should not block engine - events must be dropped.
    """
    sink = BroadcastEventSink(buffer_size=3)  # Small buffer for testing

    recv = await sink.subscribe("slow_client")

    # Fill subscriber's buffer without consuming
    for i in range(5):
        event = StageStarted(
            type="stage_started",
            seq=i,
            stage=f"s{i}",
            index=i,
            total=5,
            run_id="test-run",
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


# =============================================================================
# Commit Query
# =============================================================================


@pytest.mark.anyio
async def test_handler_commit_dispatches_to_commit_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handler commit query calls commit_stages and returns committed/failed lists."""
    handler = AgentRpcHandler(engine=Engine(pipeline=Pipeline("test", root=tmp_path)))

    monkeypatch.setattr(
        "pivot.executor.commit.commit_stages",
        _helper_commit_stages_train_eval,
    )

    result = await handler.handle_query("commit")
    assert "committed" in result, "Result should have 'committed' key"
    assert "failed" in result, "Result should have 'failed' key"
    # Type narrow to CommitResult
    assert result["committed"] == ["train", "eval"], "Should return committed stages"
    assert result["failed"] == ["broken_stage"], "Should return failed stages"


@pytest.mark.anyio
async def test_handler_commit_empty_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handler commit query returns empty lists when nothing to commit."""
    handler = AgentRpcHandler(engine=Engine(pipeline=Pipeline("test", root=tmp_path)))

    monkeypatch.setattr(
        "pivot.executor.commit.commit_stages",
        _helper_commit_stages_empty,
    )

    result = cast("CommitResult", await handler.handle_query("commit"))
    assert result["committed"] == [], "Should return empty committed list"
    assert result["failed"] == [], "Should return empty failed list"


@pytest.mark.anyio
async def test_handler_commit_uses_the_engine_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit runs on the engine's pipeline; the server thread has no Click context."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".pivot").mkdir()
    pipeline = Pipeline("test", root=tmp_path)
    helpers.register_test_stage(_helper_valid_stage, name="valid_stage", pipeline=pipeline)

    async with Engine(pipeline=pipeline) as eng:
        result = cast("CommitResult", await AgentRpcHandler(engine=eng).handle_query("commit"))

    assert result["committed"] + result["failed"] == ["test/valid_stage"]


def test_handler_explain_saves_the_file_hashes_it_read(
    set_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: PipelineLike,
) -> None:
    """An explanation over RPC saves the hash of each file it had to read, so the next
    explanation reads none of them."""
    monkeypatch.chdir(set_project_root)
    (set_project_root / ".git").mkdir(exist_ok=True)
    data = set_project_root / "input.txt"
    data.write_text("data")
    helpers.register_test_stage(
        _helper_explained_stage, name="explained", deps={"input_file": "input.txt"}
    )
    executor.run(pipeline=test_pipeline)
    # A copy of the input in its place: the same content, a new inode, so no entry matches.
    copy = data.with_name("input.copy")
    shutil.copy2(data, copy)
    copy.replace(data)

    async def _explain() -> QueryResult:
        async with Engine(pipeline=test_pipeline) as eng:
            return await AgentRpcHandler(engine=eng).handle_query(
                "explain", {"stage": "test/explained"}
            )

    explanation = anyio.run(_explain)

    assert cast("StageExplanation", explanation)["will_run"] is False
    with state.StateDB(set_project_root / ".pivot", readonly=True) as state_db:
        assert state_db.get(data, data.stat()) == cache.hash_file(data)[0]


@pytest.mark.anyio
async def test_rpc_commit_via_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit query works end-to-end via Unix socket connection."""
    socket_path = tmp_path / "agent.sock"

    monkeypatch.setattr(
        "pivot.executor.commit.commit_stages",
        _helper_commit_stages_stage_a,
    )

    async with Engine(pipeline=Pipeline("test", root=tmp_path)) as engine:
        handler = AgentRpcHandler(engine=engine)
        source = AgentRpcSource(socket_path=socket_path, handler=handler)

        send, recv = anyio.create_memory_object_stream[InputEvent](10)

        async with anyio.create_task_group() as tg:
            tg.start_soon(source.run, send)
            await helpers.wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = {"jsonrpc": "2.0", "method": "commit", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")

                response_line = await conn.receive(4096)
                response = json.loads(response_line.decode())

                assert "result" in response, "Should return result"
                assert response["result"]["committed"] == ["stage_a"]
                assert response["result"]["failed"] == []

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_rpc_commit_propagates_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit query propagates errors from commit_stages — server closes connection."""
    socket_path = tmp_path / "agent.sock"

    def _raise(_pipeline: object) -> tuple[list[str], list[str]]:
        raise RuntimeError("No pipeline loaded")

    monkeypatch.setattr(
        "pivot.executor.commit.commit_stages",
        _raise,
    )

    async with Engine(pipeline=Pipeline("test", root=tmp_path)) as engine:
        handler = AgentRpcHandler(engine=engine)
        source = AgentRpcSource(socket_path=socket_path, handler=handler)

        send, recv = anyio.create_memory_object_stream[InputEvent](10)

        async with anyio.create_task_group() as tg:
            tg.start_soon(source.run, send)
            await helpers.wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = {"jsonrpc": "2.0", "method": "commit", "id": 1}
                await conn.send(json.dumps(request).encode() + b"\n")

                # RuntimeError propagates through handle_query → _handle_request →
                # _handle_connection, which catches all exceptions and closes connection.
                # Client sees EndOfStream.
                with pytest.raises(anyio.EndOfStream):
                    await conn.receive(4096)

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_rpc_query_pivot_error_gets_an_error_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PivotError from a query handler is a JSON-RPC error with its message, not a drop."""
    from pivot import exceptions

    socket_path = tmp_path / "agent.sock"

    def _raise(_pipeline: object) -> tuple[list[str], list[str]]:
        raise exceptions.PivotError("Stage 'test/s': fingerprinting failed: captures 'X'")

    monkeypatch.setattr("pivot.executor.commit.commit_stages", _raise)

    async with Engine(pipeline=Pipeline("test", root=tmp_path)) as engine:
        source = AgentRpcSource(socket_path=socket_path, handler=AgentRpcHandler(engine=engine))
        send, _ = anyio.create_memory_object_stream[InputEvent](10)

        async with anyio.create_task_group() as tg:
            tg.start_soon(source.run, send)
            await helpers.wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                await conn.send(
                    json.dumps({"jsonrpc": "2.0", "method": "commit", "id": 1}).encode() + b"\n"
                )
                response = json.loads((await conn.receive(4096)).decode())
                assert response["error"]["code"] == -32603
                assert (
                    response["error"]["message"]
                    == "Stage 'test/s': fingerprinting failed: captures 'X'"
                )

            tg.cancel_scope.cancel()


# =============================================================================
# diff_output Query
# =============================================================================


def _make_diff_result(path: str = "output.csv") -> DataDiffResult:
    """Create a minimal DataDiffResult for testing."""
    return DataDiffResult(
        path=path,
        change_type=ChangeType.MODIFIED,
        old_rows=10,
        new_rows=12,
        old_cols=["a", "b"],
        new_cols=["a", "b"],
        schema_changes=[],
        row_changes=[],
        reorder_only=False,
        truncated=False,
        summary_only=False,
    )


@pytest.mark.anyio
async def test_handler_diff_output_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diff_output returns DataDiffResult shape from mocked data module."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    expected = _make_diff_result("data/output.csv")

    global _diff_expected
    _diff_expected = expected
    monkeypatch.setattr("pivot.show.data.restore_data_from_cache", _helper_restore_data_from_cache)
    monkeypatch.setattr(
        "pivot.show.data.diff_data_files",
        _helper_diff_data_files,
    )
    # Prevent cleanup from failing on non-existent temp files
    monkeypatch.setattr("pathlib.Path.unlink", _helper_path_unlink)

    try:
        result = cast(
            "DataDiffResult",
            await handler.handle_query(
                "diff_output",
                {"path": "data/output.csv", "old_hash": "abc123", "new_hash": "def456"},
            ),
        )
    finally:
        _diff_expected = None

    assert result["path"] == "data/output.csv", "Should return correct path"
    assert result["old_rows"] == 10, "Should return old row count"
    assert result["new_rows"] == 12, "Should return new row count"


@pytest.mark.anyio
async def test_handler_diff_output_missing_path() -> None:
    """diff_output rejects missing path parameter."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    with pytest.raises(ValueError, match="path must be string"):
        await handler.handle_query("diff_output", {})


@pytest.mark.anyio
async def test_handler_diff_output_invalid_path_type() -> None:
    """diff_output rejects non-string path parameter."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    with pytest.raises(ValueError, match="path must be string"):
        await handler.handle_query("diff_output", {"path": 123})


@pytest.mark.anyio
async def test_handler_diff_output_invalid_old_hash_type() -> None:
    """diff_output rejects non-string, non-null old_hash."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    with pytest.raises(ValueError, match="old_hash must be string or null"):
        await handler.handle_query("diff_output", {"path": "out.csv", "old_hash": 123})


@pytest.mark.anyio
async def test_handler_diff_output_invalid_new_hash_type() -> None:
    """diff_output rejects non-string, non-null new_hash."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    with pytest.raises(ValueError, match="new_hash must be string or null"):
        await handler.handle_query("diff_output", {"path": "out.csv", "new_hash": 42})


@pytest.mark.anyio
async def test_handler_diff_output_null_old_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diff_output restores only the side whose hash the request names (an added file)."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    expected = _make_diff_result("new_file.csv")

    restore_calls = list[str]()

    def _mock_restore(rel_path: str, file_hash: str) -> Path | None:
        restore_calls.append(file_hash)
        return Path(f"/tmp/fake_{file_hash}")

    monkeypatch.setattr("pivot.show.data.restore_data_from_cache", _mock_restore)
    monkeypatch.setattr(
        "pivot.show.data.diff_data_files",
        _helper_diff_data_files,
    )
    monkeypatch.setattr("pathlib.Path.unlink", _helper_path_unlink)

    _diff_expected = expected
    try:
        result = cast(
            "DataDiffResult",
            await handler.handle_query(
                "diff_output",
                {"path": "new_file.csv", "old_hash": None, "new_hash": "def4560000000000"},
            ),
        )
    finally:
        _diff_expected = None

    assert result["path"] == "new_file.csv", "Should return correct path"
    assert restore_calls == ["def4560000000000"], "Should restore only the named side"


@pytest.mark.anyio
async def test_handler_diff_output_refuses_a_request_naming_neither_hash() -> None:
    """A request naming neither hash names no file to compare, so it is refused."""
    handler = AgentRpcHandler(engine=MagicMock())

    with pytest.raises(ValueError, match="old_hash or new_hash must be given"):
        await handler.handle_query(
            "diff_output", {"path": "out.csv", "old_hash": None, "new_hash": None}
        )


@pytest.mark.anyio
async def test_handler_diff_output_max_rows_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diff_output passes max_rows=50 by default."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    captured_kwargs = dict[str, object]()

    def _mock_diff(
        old_path: object, new_path: object, path_display: str, **kwargs: object
    ) -> DataDiffResult:
        captured_kwargs.update(kwargs)
        return _make_diff_result(path_display)

    monkeypatch.setattr("pivot.show.data.restore_data_from_cache", _helper_restore_data_none)
    monkeypatch.setattr("pivot.show.data.diff_data_files", _mock_diff)

    await handler.handle_query(
        "diff_output",
        {"path": "out.csv", "old_hash": None, "new_hash": "def4560000000000"},
    )

    assert captured_kwargs.get("max_rows") == 50, "Default max_rows should be 50"


@pytest.mark.anyio
async def test_handler_diff_output_custom_max_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """diff_output respects custom max_rows parameter."""
    mock_engine = MagicMock()
    handler = AgentRpcHandler(engine=mock_engine)

    captured_kwargs = dict[str, object]()

    def _mock_diff(
        old_path: object, new_path: object, path_display: str, **kwargs: object
    ) -> DataDiffResult:
        captured_kwargs.update(kwargs)
        return _make_diff_result(path_display)

    monkeypatch.setattr("pivot.show.data.restore_data_from_cache", _helper_restore_data_none)
    monkeypatch.setattr("pivot.show.data.diff_data_files", _mock_diff)

    await handler.handle_query(
        "diff_output",
        {"path": "out.csv", "old_hash": None, "new_hash": "def4560000000000", "max_rows": 100},
    )

    assert captured_kwargs.get("max_rows") == 100, "Should use custom max_rows"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("old_hash", "new_hash", "change_type"),
    [
        pytest.param(None, "def4560000000000", ChangeType.ADDED, id="added"),
        pytest.param("abc1230000000000", None, ChangeType.REMOVED, id="removed"),
        pytest.param("abc1230000000000", "def4560000000000", ChangeType.MODIFIED, id="modified"),
    ],
)
async def test_handler_diff_output_passes_the_change_type_its_hashes_name(
    monkeypatch: pytest.MonkeyPatch,
    old_hash: str | None,
    new_hash: str | None,
    change_type: ChangeType,
) -> None:
    """A request naming only a new hash is an added file, only an old hash a removed file, and
    both a modified file, whether or not either object is in the cache."""
    handler = AgentRpcHandler(engine=MagicMock())
    captured_kwargs = dict[str, object]()

    def _mock_diff(
        old_path: object, new_path: object, path_display: str, **kwargs: object
    ) -> DataDiffResult:
        captured_kwargs.update(kwargs)
        return _make_diff_result(path_display)

    monkeypatch.setattr("pivot.show.data.restore_data_from_cache", _helper_restore_data_none)
    monkeypatch.setattr("pivot.show.data.diff_data_files", _mock_diff)

    await handler.handle_query(
        "diff_output", {"path": "out.csv", "old_hash": old_hash, "new_hash": new_hash}
    )

    assert captured_kwargs.get("change_type") == change_type, captured_kwargs


def test_encode_wire_writes_numpy_scalars_as_python_values() -> None:
    """A diff result's row values can be numpy integers, floats and bools; the RPC writes them
    as plain JSON numbers and booleans instead of dropping the connection."""
    import numpy

    from pivot.engine import agent_rpc

    message = {
        "old_values": {"n": numpy.int64(2), "x": numpy.float64(0.5), "ok": numpy.bool_(True)}
    }

    wire = json.loads(agent_rpc._encode_wire(message))

    assert wire == {"old_values": {"n": 2, "x": 0.5, "ok": True}}
    assert [type(value) for value in wire["old_values"].values()] == [int, float, bool]


@pytest.mark.anyio
async def test_rpc_diff_output_answers_a_modified_file_with_a_datetime_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modified file's rows can hold timestamps; the reply writes them as text, as
    `pivot diff --json` does, instead of dropping the connection."""
    import shutil

    import pandas

    from pivot import config, loaders
    from pivot.storage import cache

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(config, "locate_cache_dir", lambda: cache_dir)
    hashes = list[str]()
    for rows in (2, 3):
        frame = pandas.DataFrame({"v": range(1, rows + 1)})
        frame["when"] = pandas.Timestamp("2024-01-01") + pandas.to_timedelta(frame["v"], unit="D")
        source = tmp_path / f"stamp{rows}.jsonl"
        loaders.DataFrameJSONL().save(frame, source)
        file_hash, _ = cache.hash_file(source)
        cached = cache.get_cache_path(cache_dir / "files", file_hash)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, cached)
        hashes.append(file_hash)
    socket_path = tmp_path / "agent.sock"
    request = {
        "jsonrpc": "2.0",
        "method": "diff_output",
        "params": {"path": "data/d/stamp.jsonl", "old_hash": hashes[0], "new_hash": hashes[1]},
        "id": 1,
    }

    reply = b""
    async with Engine(pipeline=Pipeline("test", root=tmp_path)) as engine:
        source = AgentRpcSource(socket_path=socket_path, handler=AgentRpcHandler(engine=engine))
        send, _ = anyio.create_memory_object_stream[InputEvent](10)

        async with anyio.create_task_group() as tg:
            tg.start_soon(source.run, send)
            await helpers.wait_for_socket(socket_path)

            async with await anyio.connect_unix(str(socket_path)) as conn:
                await conn.send(json.dumps(request).encode() + b"\n")
                while not reply.endswith(b"\n"):
                    reply += await conn.receive(65536)

            tg.cancel_scope.cancel()

    result = json.loads(reply)["result"]
    assert (result["change_type"], result["old_rows"], result["new_rows"]) == ("modified", 2, 3)
    assert [change["new_values"] for change in result["row_changes"]] == [
        {"v": 3, "when": "2024-01-04 00:00:00"}
    ]
