from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnusedCallResult=false
# pyright: reportInvalidCast=false
# pyright: reportDuplicateImport=false
import json
from typing import TYPE_CHECKING, Protocol, cast

import anyio
import pytest

from pivot_tui.testing import FakeRpcServer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _MoveOnAfter(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class _SocketStream(Protocol):
    async def receive(self, max_bytes: int) -> bytes: ...

    async def send(self, item: bytes) -> None: ...

    async def __aenter__(self) -> _SocketStream: ...

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class _Anyio(Protocol):
    def move_on_after(self, delay: float) -> _MoveOnAfter: ...

    async def sleep(self, delay: float) -> None: ...

    async def connect_unix(self, path: str) -> _SocketStream: ...


class _PytestMark(Protocol):
    def anyio(self, func: Callable[..., object]) -> Callable[..., object]: ...


class _Pytest(Protocol):
    mark: _PytestMark


class _FakeRpcServer(Protocol):
    async def start(self, socket_path: Path) -> None: ...

    async def stop(self) -> None: ...

    def set_stages(self, stages: list[str]) -> None: ...

    def set_status(
        self, state: str, running: list[str] | None = None, pending: list[str] | None = None
    ) -> None: ...

    def inject_event(self, event: dict[str, object]) -> None: ...

    def set_explanation(self, stage: str, explanation: dict[str, object]) -> None: ...

    def set_stage_info(self, stage: str, deps: list[str], outs: list[str]) -> None: ...

    def set_commit_result(self, committed: list[str], failed: list[str] | None = None) -> None: ...

    def inject_error(self, method: str, code: int, message: str) -> None: ...


anyio = cast("_Anyio", anyio)  # type: ignore[reportGeneralTypeIssues] - module cast
pytest = cast("_Pytest", pytest)  # type: ignore[reportGeneralTypeIssues] - module cast
FakeRpcServer = cast("type[_FakeRpcServer]", FakeRpcServer)


async def _wait_for_socket(socket_path: Path, timeout: float = 1.0) -> None:
    with anyio.move_on_after(timeout):
        while not socket_path.exists():
            await anyio.sleep(0.01)
    assert socket_path.exists(), "Socket should exist"


async def _send_request(socket_path: Path, request: dict[str, object]) -> dict[str, object]:
    async with await anyio.connect_unix(str(socket_path)) as conn:
        await conn.send(json.dumps(request).encode() + b"\n")
        response_line = await conn.receive(4096)
    return cast("dict[str, object]", json.loads(response_line.decode()))


@pytest.mark.anyio
async def test_fake_rpc_server_run_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "run", "id": 1}
        )
        assert response.get("result") == "accepted", "Run should return accepted"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_cancel_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "cancel", "id": 2}
        )
        assert response.get("result") == "accepted", "Cancel should return accepted"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_status_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.set_status("active", running=["train"], pending=["eval"])
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "status", "id": 3}
        )
        assert response.get("result") == {
            "state": "active",
            "running": ["train"],
            "pending": ["eval"],
        }, "Status should reflect configured state"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_stages_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.set_stages(["train", "eval"])
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "stages", "id": 4}
        )
        assert response.get("result") == {"stages": ["train", "eval"]}, (
            "Stages should return configured list"
        )
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_stage_info_known_stage(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.set_stage_info("train", deps=["input.csv"], outs=["output.csv"])
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path,
            {"jsonrpc": "2.0", "method": "stage_info", "params": {"stage": "train"}, "id": 5},
        )
        assert response.get("result") == {
            "name": "train",
            "deps": ["input.csv"],
            "outs": ["output.csv"],
        }, "stage_info should return configured info"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_stage_info_unknown_stage(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path,
            {
                "jsonrpc": "2.0",
                "method": "stage_info",
                "params": {"stage": "missing"},
                "id": 6,
            },
        )
        error = response.get("error")
        assert isinstance(error, dict), "Error response should be a dict"
        assert error.get("code") == -32602, "Unknown stage should return invalid params"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_explain_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.set_explanation(
        "train",
        {
            "stage_name": "train",
            "will_run": True,
            "is_forced": False,
            "reason": "No previous run",
            "code_changes": [],
            "param_changes": [],
            "dep_changes": [],
            "upstream_stale": [],
        },
    )
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path,
            {"jsonrpc": "2.0", "method": "explain", "params": {"stage": "train"}, "id": 7},
        )
        result = response.get("result")
        assert isinstance(result, dict), "Explain should return a result dict"
        assert result.get("reason") == "No previous run", "Explain should return configured data"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_events_since_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.inject_event({"type": "stage_started", "stage": "train", "index": 0, "total": 1})
    server.inject_event({"type": "stage_completed", "stage": "train", "index": 0, "total": 1})
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path,
            {"jsonrpc": "2.0", "method": "events_since", "params": {"version": 0}, "id": 8},
        )
        result = response.get("result")
        assert isinstance(result, dict), "events_since should return a result dict"
        assert result.get("version") == 2, "events_since should return latest version"
        events = result.get("events")
        assert isinstance(events, list), "events_since should return events list"
        assert len(events) == 2, "events_since should return injected events"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_commit_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.set_commit_result(["train"], failed=["eval"])
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "commit", "id": 9}
        )
        assert response.get("result") == {"committed": ["train"], "failed": ["eval"]}, (
            "Commit should return configured result"
        )
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_set_on_error_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path,
            {
                "jsonrpc": "2.0",
                "method": "set_on_error",
                "params": {"mode": "keep_going"},
                "id": 10,
            },
        )
        assert response.get("result") is True, "set_on_error should return True"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_unknown_method(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "unknown", "id": 11}
        )
        error = response.get("error")
        assert isinstance(error, dict), "Unknown method should return error dict"
        assert error.get("code") == -32601, "Unknown method should return method not found"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_injected_error(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    server.inject_error("status", -32099, "boom")
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        response: dict[str, object] = await _send_request(
            socket_path, {"jsonrpc": "2.0", "method": "status", "id": 12}
        )
        error = response.get("error")
        assert isinstance(error, dict), "Injected error should return error dict"
        assert error.get("code") == -32099, "Injected error code should be returned"
        assert error.get("message") == "boom", "Injected error message should be returned"
    finally:
        await server.stop()


@pytest.mark.anyio
async def test_fake_rpc_server_parse_error(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake.sock"
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        await _wait_for_socket(socket_path)
        async with await anyio.connect_unix(str(socket_path)) as conn:
            await conn.send(b"not valid json\n")
            response_line = await conn.receive(4096)
        response = cast("dict[str, object]", json.loads(response_line.decode()))
        error = response.get("error")
        assert isinstance(error, dict), "Parse error should return error dict"
        assert error.get("code") == -32700, "Invalid JSON should return parse error"
    finally:
        await server.stop()
