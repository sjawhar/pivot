"""RPC client implementing PivotClient protocol over Unix socket."""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingImports=false
# pyright: reportImplicitRelativeImport=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAny=false
# pyright: reportUnnecessaryCast=false

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal, cast

import anyio

from pivot_tui.client import (
    CommitResult,
    EngineStatus,
    EventBatch,
    StageInfoResult,
    VersionedEvent,
)

if TYPE_CHECKING:
    from pathlib import Path

    from anyio.abc import SocketStream

    from pivot.types import StageExplanation

__all__ = ["RpcError", "RpcProtocolError", "RpcPivotClient"]

_logger = logging.getLogger(__name__)

_RPC_TIMEOUT = 30.0
_MAX_BUFFER = 1_048_576  # 1MB, matching server's max message size


class RpcError(Exception):
    """Raised when the server returns a JSON-RPC error response."""

    code: int
    message: str

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"RPC error {code}: {message}")


class RpcProtocolError(RpcError):
    """Raised when the server response has unexpected types or structure."""

    def __init__(self, expected: str, got: str) -> None:
        super().__init__(-1, f"Protocol error: expected {expected}, got {got}")


class RpcPivotClient:
    """PivotClient implementation using JSON-RPC 2.0 over Unix socket.

    Maintains a persistent connection. Each method sends a request and
    waits for the corresponding response.
    """

    _conn: SocketStream | None
    _request_id: int
    _buffer: bytes
    _call_lock: anyio.Lock | None

    def __init__(self) -> None:
        self._conn = None
        self._request_id = 0
        self._buffer = b""
        self._call_lock = None

    async def connect(self, socket_path: Path) -> None:
        """Connect to the engine's Unix socket."""
        if self._conn is not None:
            await self.disconnect()
        self._buffer = b""
        self._call_lock = anyio.Lock()
        self._conn = await anyio.connect_unix(str(socket_path))

    async def disconnect(self) -> None:
        """Close the connection."""
        if self._conn is not None:
            await self._conn.aclose()
            self._conn = None
        self._buffer = b""
        self._call_lock = None

    async def run(self, stages: list[str] | None = None, *, force: bool = False) -> bool:
        result = await self._call("run", {"stages": stages, "force": force})
        return result == "accepted"

    async def cancel(self) -> bool:
        result = await self._call("cancel")
        return result == "accepted"

    async def commit(self) -> CommitResult:
        r = _as_dict(await self._call("commit"))
        return CommitResult(
            committed=_as_str_list(r["committed"]),
            failed=_as_str_list(r["failed"]),
        )

    async def status(self) -> EngineStatus:
        r = _as_dict(await self._call("status"))
        state = r["state"]
        if state not in ("idle", "active"):
            raise RpcProtocolError("'idle' or 'active'", repr(state))
        return EngineStatus(
            state=state,  # type: ignore[typeddict-item] - narrowed by check above
            running=_as_str_list(r["running"]),
            pending=_as_str_list(r["pending"]),
        )

    async def stages(self) -> list[str]:
        r = _as_dict(await self._call("stages"))
        return _as_str_list(r["stages"])

    async def stage_info(self, stage: str) -> StageInfoResult:
        r = _as_dict(await self._call("stage_info", {"stage": stage}))
        name = r["name"]
        if not isinstance(name, str):
            raise RpcProtocolError("str", type(name).__name__)
        return StageInfoResult(
            name=name,
            deps=_as_str_list(r["deps"]),
            outs=_as_str_list(r["outs"]),
        )

    async def explain(self, stage: str) -> StageExplanation:
        from pivot.types import CodeChange, DepChange, ParamChange, StageExplanation

        r = _as_dict(await self._call("explain", {"stage": stage}))
        stage_name = r["stage_name"]
        will_run = r["will_run"]
        is_forced = r["is_forced"]
        reason = r["reason"]
        code_changes = r["code_changes"]
        param_changes = r["param_changes"]
        dep_changes = r["dep_changes"]
        if not isinstance(stage_name, str):
            raise RpcProtocolError("str", type(stage_name).__name__)
        if not isinstance(will_run, bool):
            raise RpcProtocolError("bool", type(will_run).__name__)
        if not isinstance(is_forced, bool):
            raise RpcProtocolError("bool", type(is_forced).__name__)
        if not isinstance(reason, str):
            raise RpcProtocolError("str", type(reason).__name__)
        if not isinstance(code_changes, list):
            raise RpcProtocolError("list", type(code_changes).__name__)
        if not isinstance(param_changes, list):
            raise RpcProtocolError("list", type(param_changes).__name__)
        if not isinstance(dep_changes, list):
            raise RpcProtocolError("list", type(dep_changes).__name__)
        return StageExplanation(
            stage_name=stage_name,
            will_run=will_run,
            is_forced=is_forced,
            reason=reason,
            code_changes=cast("list[CodeChange]", code_changes),
            param_changes=cast("list[ParamChange]", param_changes),
            dep_changes=cast("list[DepChange]", dep_changes),
            upstream_stale=_as_str_list(r["upstream_stale"]),
        )

    async def events_since(self, version: int) -> EventBatch:
        r = _as_dict(await self._call("events_since", {"version": version}))
        raw_events = r["events"]
        if not isinstance(raw_events, list):
            raise RpcProtocolError("list", type(raw_events).__name__)
        events = list[VersionedEvent]()
        for raw in raw_events:
            e = _as_dict(raw)
            v = e["version"]
            if not isinstance(v, int):
                raise RpcProtocolError("int", type(v).__name__)
            event_data = e["event"]
            if not isinstance(event_data, dict):
                raise RpcProtocolError("dict", type(event_data).__name__)
            events.append(VersionedEvent(version=v, event=dict(event_data)))
        ver = r["version"]
        if not isinstance(ver, int):
            raise RpcProtocolError("int", type(ver).__name__)
        return EventBatch(version=ver, events=events)

    async def set_on_error(self, mode: Literal["fail", "keep_going"]) -> bool:
        result = await self._call("set_on_error", {"mode": mode})
        return bool(result)

    async def diff_output(
        self,
        path: str,
        old_hash: str | None,
        new_hash: str | None,
        *,
        max_rows: int = 50,
    ) -> dict[str, object]:
        result = await self._call(
            "diff_output",
            {"path": path, "old_hash": old_hash, "new_hash": new_hash, "max_rows": max_rows},
        )
        return _as_dict(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call(
        self,
        method: str,
        params: dict[str, object] | None = None,
    ) -> object:
        """Send a JSON-RPC request and return the result, or raise RpcError."""
        if self._conn is None:
            raise RuntimeError("Not connected — call connect() first")
        if self._call_lock is None:
            raise RuntimeError("Not connected — call connect() first")

        async with self._call_lock:
            self._request_id += 1
            request_id = self._request_id
            request: dict[str, object] = {
                "jsonrpc": "2.0",
                "method": method,
                "id": request_id,
            }
            if params is not None:
                request["params"] = params

            payload = json.dumps(request).encode() + b"\n"

            with anyio.fail_after(_RPC_TIMEOUT):
                await self._conn.send(payload)
                # Read responses until we find one matching our request ID
                while True:
                    response = await self._read_response()
                    resp_id = response.get("id")
                    if resp_id == request_id:
                        break
                    _logger.warning(
                        "Unexpected response ID %s (expected %s), skipping", resp_id, request_id
                    )

        if "error" in response:
            err = _as_dict(response["error"])
            code = err.get("code")
            message = err.get("message")
            if not isinstance(code, int):
                raise RpcProtocolError("int", type(code).__name__)
            if not isinstance(message, str):
                raise RpcProtocolError("str", type(message).__name__)
            raise RpcError(code, message)

        return response.get("result")

    async def _read_response(self) -> dict[str, object]:
        """Read a single newline-delimited JSON response from the connection."""
        if self._conn is None:
            raise RuntimeError("Not connected")
        while b"\n" not in self._buffer:
            if len(self._buffer) >= _MAX_BUFFER:
                raise RpcProtocolError("message under 1MB", f"{len(self._buffer)} bytes in buffer")
            max_to_receive = _MAX_BUFFER - len(self._buffer)
            chunk: bytes = await self._conn.receive(max_to_receive)
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            parsed = json.loads(line.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RpcProtocolError("valid UTF-8 JSON", repr(line[:100])) from exc
        if not isinstance(parsed, dict):
            raise RpcProtocolError("JSON object", type(parsed).__name__)
        return cast("dict[str, object]", parsed)


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RpcProtocolError("dict", type(value).__name__)
    return cast("dict[str, object]", value)


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise RpcProtocolError("list", type(value).__name__)
    items = list[str]()
    for item in value:
        if not isinstance(item, str):
            raise RpcProtocolError("list[str]", type(item).__name__)
        items.append(item)
    return items
