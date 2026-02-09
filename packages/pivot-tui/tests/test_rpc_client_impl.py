from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportAny=false
# pyright: reportUnusedCallResult=false
# pyright: reportUntypedFunctionDecorator=false
import contextlib
import json
from typing import TYPE_CHECKING

import anyio
import pytest

from pivot_tui.rpc_client_impl import RpcError, RpcPivotClient, RpcProtocolError
from pivot_tui.testing import FakeRpcServer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from anyio.abc import SocketListener


async def _wait_for_socket(socket_path: Path, timeout: float = 1.0) -> None:
    with anyio.move_on_after(timeout):
        while not socket_path.exists():
            await anyio.sleep(0.01)
    assert socket_path.exists()


@contextlib.asynccontextmanager
async def _connected(
    socket_path: Path,
) -> AsyncGenerator[tuple[FakeRpcServer, RpcPivotClient]]:
    server = FakeRpcServer()
    await server.start(socket_path)
    await _wait_for_socket(socket_path)
    client = RpcPivotClient()
    await client.connect(socket_path)
    try:
        yield server, client
    finally:
        await client.disconnect()
        await server.stop()


# --- run ---


@pytest.mark.anyio
async def test_run_all_stages(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (_server, client):
        result = await client.run()
        assert result is True


@pytest.mark.anyio
async def test_run_specific_stages(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        result = await client.run(["train", "eval"], force=True)
        assert result is True
        assert len(server._run_requests) == 1
        assert server._run_requests[0]["stages"] == ["train", "eval"]
        assert server._run_requests[0]["force"] is True


# --- cancel ---


@pytest.mark.anyio
async def test_cancel(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (_server, client):
        result = await client.cancel()
        assert result is True


# --- commit ---


@pytest.mark.anyio
async def test_commit(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_commit_result(["train", "eval"], failed=["broken"])
        result = await client.commit()
        assert result["committed"] == ["train", "eval"]
        assert result["failed"] == ["broken"]


# --- status ---


@pytest.mark.anyio
async def test_status_idle(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_status("idle")
        result = await client.status()
        assert result["state"] == "idle"
        assert result["running"] == []
        assert result["pending"] == []


@pytest.mark.anyio
async def test_status_active(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_status("active", running=["train"], pending=["eval"])
        result = await client.status()
        assert result["state"] == "active"
        assert result["running"] == ["train"]
        assert result["pending"] == ["eval"]


# --- stages ---


@pytest.mark.anyio
async def test_stages(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_stages(["train", "eval", "report"])
        result = await client.stages()
        assert result == ["train", "eval", "report"]


@pytest.mark.anyio
async def test_stages_empty(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (_server, client):
        result = await client.stages()
        assert result == []


# --- stage_info ---


@pytest.mark.anyio
async def test_stage_info(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_stage_info("train", deps=["input.csv"], outs=["model.pkl"])
        result = await client.stage_info("train")
        assert result["name"] == "train"
        assert result["deps"] == ["input.csv"]
        assert result["outs"] == ["model.pkl"]


# --- explain ---


@pytest.mark.anyio
async def test_explain(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_explanation(
            "train",
            {
                "stage_name": "train",
                "will_run": True,
                "is_forced": False,
                "reason": "Code changed",
                "code_changes": [{"file": "train.py", "change": "modified"}],
                "param_changes": [],
                "dep_changes": [],
                "upstream_stale": ["prepare"],
            },
        )
        result = await client.explain("train")
        assert result["stage_name"] == "train"
        assert result["will_run"] is True
        assert result["is_forced"] is False
        assert result["reason"] == "Code changed"
        assert len(result["code_changes"]) == 1
        assert result["upstream_stale"] == ["prepare"]


# --- events_since ---


@pytest.mark.anyio
async def test_events_since(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.inject_event({"type": "stage_started", "stage": "train"})
        server.inject_event({"type": "stage_completed", "stage": "train"})
        result = await client.events_since(0)
        assert result["version"] == 2
        assert len(result["events"]) == 2
        assert result["events"][0]["event"]["type"] == "stage_started"
        assert result["events"][1]["event"]["type"] == "stage_completed"


@pytest.mark.anyio
async def test_events_since_partial(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.inject_event({"type": "stage_started", "stage": "train"})
        server.inject_event({"type": "stage_completed", "stage": "train"})
        result = await client.events_since(1)
        assert result["version"] == 2
        assert len(result["events"]) == 1
        assert result["events"][0]["version"] == 2


# --- set_on_error ---


@pytest.mark.anyio
async def test_set_on_error(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        result = await client.set_on_error("keep_going")
        assert result is True
        assert server._on_error_mode == "keep_going"


# --- diff_output ---


@pytest.mark.anyio
async def test_diff_output(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.set_diff_result("output.csv", {"added": 5, "removed": 2, "rows": []})
        result = await client.diff_output("output.csv", "abc123", "def456", max_rows=100)
        assert result["added"] == 5
        assert result["removed"] == 2


# --- error handling ---


@pytest.mark.anyio
async def test_rpc_error_raised_on_server_error(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (server, client):
        server.inject_error("status", -32099, "Something broke")
        with pytest.raises(RpcError) as exc_info:
            await client.status()
        assert exc_info.value.code == -32099
        assert exc_info.value.message == "Something broke"


@pytest.mark.anyio
async def test_rpc_error_on_unknown_stage(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (_server, client):
        with pytest.raises(RpcError) as exc_info:
            await client.stage_info("nonexistent")
        assert exc_info.value.code == -32602


@pytest.mark.anyio
async def test_rpc_error_on_unknown_explain(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (_server, client):
        with pytest.raises(RpcError) as exc_info:
            await client.explain("nonexistent")
        assert exc_info.value.code == -32602


@pytest.mark.anyio
async def test_rpc_error_on_missing_diff(tmp_path: Path) -> None:
    async with _connected(tmp_path / "t.sock") as (_server, client):
        with pytest.raises(RpcError) as exc_info:
            await client.diff_output("missing.csv", None, None)
        assert exc_info.value.code == -32602


@pytest.mark.anyio
async def test_not_connected_raises_runtime_error() -> None:
    client = RpcPivotClient()
    with pytest.raises(RuntimeError, match="Not connected"):
        await client.run()


@pytest.mark.anyio
async def test_disconnect_when_not_connected() -> None:
    client = RpcPivotClient()
    await client.disconnect()


# --- buffer: two JSON responses in one TCP segment ---


@pytest.mark.anyio
async def test_two_responses_in_one_tcp_segment(tmp_path: Path) -> None:
    socket_path = tmp_path / "buf.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, buf = buf.split(b"\n", 1)
            req1 = json.loads(line.decode())
            req1_id = req1["id"]
            next_id = req1_id + 1
            resp1 = json.dumps({"jsonrpc": "2.0", "result": "first", "id": req1_id})
            resp2 = json.dumps({"jsonrpc": "2.0", "result": "second", "id": next_id})
            await conn.send((resp1 + "\n" + resp2 + "\n").encode())
            # Still need to read request 2 to keep connection alive
            while b"\n" not in buf:
                buf += await conn.receive(4096)

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            r1 = await client._call("a")
            r2 = await client._call("b")
            assert r1 == "first"
            assert r2 == "second"
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_call_lock_serializes_requests(tmp_path: Path) -> None:
    socket_path = tmp_path / "lock.sock"
    second_request_early = anyio.Event()

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, buf = buf.split(b"\n", 1)
            req1 = json.loads(line.decode())
            req1_id = req1["id"]
            with anyio.move_on_after(0.2):
                while b"\n" not in buf:
                    buf += await conn.receive(4096)
            if b"\n" in buf:
                second_request_early.set()
            resp1 = json.dumps({"jsonrpc": "2.0", "result": "first", "id": req1_id})
            await conn.send((resp1 + "\n").encode())
            with anyio.fail_after(1.0):
                while b"\n" not in buf:
                    buf += await conn.receive(4096)
            line2, _ = buf.split(b"\n", 1)
            req2 = json.loads(line2.decode())
            resp2 = json.dumps({"jsonrpc": "2.0", "result": "second", "id": req2["id"]})
            await conn.send((resp2 + "\n").encode())

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        results: list[str] = []

        async def _call(method: str) -> None:
            result = await client._call(method)
            assert isinstance(result, str)
            results.append(result)

        try:
            async with anyio.create_task_group() as call_tg:
                call_tg.start_soon(_call, "a")
                call_tg.start_soon(_call, "b")
            assert not second_request_early.is_set(), "Second request sent before first response"
            assert sorted(results) == ["first", "second"]
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


# --- response ID correlation ---


@pytest.mark.anyio
async def test_mismatched_response_id_skipped(tmp_path: Path) -> None:
    """Client skips responses with wrong ID and returns the matching one."""
    socket_path = tmp_path / "id.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, _ = buf.split(b"\n", 1)
            req = json.loads(line.decode())
            req_id = req["id"]
            # Send a response with WRONG id first, then the correct one
            wrong = json.dumps({"jsonrpc": "2.0", "result": "wrong", "id": req_id + 999})
            correct = json.dumps({"jsonrpc": "2.0", "result": "correct", "id": req_id})
            await conn.send((wrong + "\n" + correct + "\n").encode())

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            result = await client._call("test")
            assert result == "correct", f"Expected 'correct', got {result!r}"
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


# --- malformed response raises RpcError, not AssertionError ---


@pytest.mark.anyio
async def test_malformed_status_state_raises_rpc_protocol_error(tmp_path: Path) -> None:
    """status() with invalid state type raises RpcError, not AssertionError."""
    socket_path = tmp_path / "mal.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, _ = buf.split(b"\n", 1)
            req = json.loads(line.decode())
            # Send result with invalid state (int instead of str)
            resp = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "result": {"state": 42, "running": [], "pending": []},
                    "id": req["id"],
                }
            )
            await conn.send((resp + "\n").encode())

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError):
                await client.status()
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_malformed_stage_info_name_raises_rpc_protocol_error(tmp_path: Path) -> None:
    """stage_info() with non-string name raises RpcProtocolError, not AssertionError."""
    socket_path = tmp_path / "mal2.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, _ = buf.split(b"\n", 1)
            req = json.loads(line.decode())
            resp = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "result": {"name": 123, "deps": [], "outs": []},
                    "id": req["id"],
                }
            )
            await conn.send((resp + "\n").encode())

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError):
                await client.stage_info("train")
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_malformed_events_raises_rpc_protocol_error(tmp_path: Path) -> None:
    """events_since() with non-list events raises RpcProtocolError, not AssertionError."""
    socket_path = tmp_path / "mal3.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, _ = buf.split(b"\n", 1)
            req = json.loads(line.decode())
            resp = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "result": {"events": "not-a-list", "version": 1},
                    "id": req["id"],
                }
            )
            await conn.send((resp + "\n").encode())

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError):
                await client.events_since(0)
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["code_changes", "param_changes", "dep_changes"])
async def test_explain_non_list_changes_raises_rpc_protocol_error(
    tmp_path: Path,
    field: str,
) -> None:
    async with _connected(tmp_path / f"{field}.sock") as (server, client):
        explanation: dict[str, object] = {
            "stage_name": "train",
            "will_run": True,
            "is_forced": False,
            "reason": "Code changed",
            "code_changes": [{"file": "train.py", "change": "modified"}],
            "param_changes": [],
            "dep_changes": [],
            "upstream_stale": [],
        }
        explanation[field] = "not-a-list"
        server.set_explanation("train", explanation)
        with pytest.raises(RpcProtocolError, match="expected list"):
            await client.explain("train")


@pytest.mark.anyio
async def test_as_dict_non_dict_raises_rpc_protocol_error() -> None:
    """_as_dict with non-dict raises RpcProtocolError, not AssertionError."""
    from pivot_tui.rpc_client_impl import _as_dict

    with pytest.raises(RpcProtocolError, match="expected dict"):
        _as_dict("not-a-dict")


@pytest.mark.anyio
async def test_as_str_list_non_list_raises_rpc_protocol_error() -> None:
    """_as_str_list with non-list raises RpcProtocolError, not AssertionError."""
    from pivot_tui.rpc_client_impl import _as_str_list

    with pytest.raises(RpcProtocolError, match="expected list"):
        _as_str_list(42)


def test_as_str_list_non_str_items_raise_rpc_protocol_error() -> None:
    from pivot_tui.rpc_client_impl import _as_str_list

    with pytest.raises(RpcProtocolError, match=r"expected list\[str\]"):
        _as_str_list(["ok", 1])


# --- corrupted JSON and buffer overflow ---


@pytest.mark.anyio
async def test_corrupted_json_raises_rpc_protocol_error(tmp_path: Path) -> None:
    """Corrupted JSON in response raises RpcProtocolError, not json.JSONDecodeError."""
    socket_path = tmp_path / "corrupt.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            _ = buf.split(b"\n", 1)
            await conn.send(b"not valid json\n")

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError, match="valid UTF-8 JSON"):
                await client._call("test")
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_invalid_utf8_json_raises_rpc_protocol_error(tmp_path: Path) -> None:
    socket_path = tmp_path / "utf8.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            await conn.send(b"\xff\xfe\n")

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError, match="valid UTF-8 JSON"):
                await client._call("test")
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_non_object_json_raises_rpc_protocol_error(tmp_path: Path) -> None:
    socket_path = tmp_path / "obj.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            await conn.send(b"[]\n")

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError, match="expected JSON object"):
                await client._call("test")
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_buffer_exceeds_1mb_raises_rpc_protocol_error(tmp_path: Path) -> None:
    """Buffer exceeding 1MB raises RpcProtocolError on pre-receive check."""
    socket_path = tmp_path / "overflow.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            _ = buf.split(b"\n", 1)
            # Send data in chunks: first chunk fills buffer to exactly 1MB,
            # second chunk triggers pre-receive check (buffer >= 1MB)
            chunk1 = b"x" * 1_048_576
            chunk2 = b"y" * 100
            await conn.send(chunk1)
            await anyio.sleep(0.01)  # Small delay to ensure separate receives
            await conn.send(chunk2)

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            with pytest.raises(RpcProtocolError, match="message under 1MB"):
                await client._call("test")
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_near_limit_message_with_early_newline_succeeds(tmp_path: Path) -> None:
    """Valid message near 1MB limit with early newline should NOT raise."""
    socket_path = tmp_path / "near_limit.sock"

    async def _raw_server(listener: SocketListener) -> None:
        conn = await listener.accept()
        async with conn:
            buf = b""
            while b"\n" not in buf:
                buf += await conn.receive(4096)
            line, _ = buf.split(b"\n", 1)
            req = json.loads(line.decode())
            # Send a valid response: large JSON object with newline early in chunk
            # Total size under 1MB, newline appears before buffer fills
            large_value = "x" * 500_000
            resp = json.dumps({"jsonrpc": "2.0", "result": {"data": large_value}, "id": req["id"]})
            await conn.send((resp + "\n").encode())

    listener = await anyio.create_unix_listener(socket_path)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_raw_server, listener)
        client = RpcPivotClient()
        await client.connect(socket_path)
        try:
            result = await client._call("test")
            assert isinstance(result, dict)
            assert "data" in result
            assert len(result["data"]) == 500_000
        finally:
            await client.disconnect()
            tg.cancel_scope.cancel()
