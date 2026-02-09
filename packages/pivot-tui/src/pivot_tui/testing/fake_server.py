from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportAny=false
# pyright: reportUnusedCallResult=false
import contextlib
import json
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import anyio  # type: ignore[reportMissingImports] - runtime dependency

if TYPE_CHECKING:
    from anyio.abc import SocketListener, SocketStream, TaskGroup

_MAX_MESSAGE_SIZE = 1024 * 1024
_MAX_VERSION = 2**63 - 1

type JsonRpcId = str | int | None


class JsonRpcErrorDetail(TypedDict):
    code: int
    message: str


class JsonRpcSuccessResponse(TypedDict):
    jsonrpc: Literal["2.0"]
    result: object
    id: JsonRpcId


class JsonRpcErrorResponse(TypedDict):
    jsonrpc: Literal["2.0"]
    error: JsonRpcErrorDetail
    id: JsonRpcId


type JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse


class EventsResult(TypedDict):
    version: int
    events: list[VersionedEvent]


class VersionedEvent(TypedDict):
    version: int
    event: dict[str, object]


class FakeRpcServer:
    """Fake JSON-RPC server for TUI testing.

    Configurable test double that mimics the engine's RPC interface.
    """

    _socket_path: Path | None
    _listener: SocketListener | None
    _task_group: TaskGroup | None
    _committed: list[str]
    _failed: list[str]
    _stages: list[str]
    _status_state: Literal["idle", "active"]
    _running: list[str]
    _pending: list[str]
    _events: list[VersionedEvent]
    _event_version: int
    _explanations: dict[str, dict[str, object]]
    _stage_infos: dict[str, dict[str, object]]
    _error_for_method: dict[str, tuple[int, str]]
    _on_error_mode: str
    _run_requests: list[dict[str, object]]
    _diff_results: dict[str, dict[str, object]]

    def __init__(self) -> None:
        self._stages = list[str]()
        self._status_state = "idle"
        self._running = list[str]()
        self._pending = list[str]()
        self._events = list[VersionedEvent]()
        self._event_version = 0
        self._explanations = dict[str, dict[str, object]]()
        self._stage_infos = dict[str, dict[str, object]]()
        self._error_for_method = dict[str, tuple[int, str]]()
        self._committed = list[str]()
        self._failed = list[str]()
        self._on_error_mode = "fail"
        self._run_requests = list[dict[str, object]]()
        self._diff_results = dict[str, dict[str, object]]()
        self._socket_path = None
        self._listener = None
        self._task_group = None

    def set_stages(self, stages: list[str]) -> None:
        self._stages = list(stages)

    def set_status(
        self,
        state: Literal["idle", "active"],
        running: list[str] | None = None,
        pending: list[str] | None = None,
    ) -> None:
        self._status_state = state
        self._running = list(running) if running is not None else list[str]()
        self._pending = list(pending) if pending is not None else list[str]()

    def inject_event(self, event: dict[str, object]) -> None:
        self._event_version += 1
        if self._event_version > _MAX_VERSION:
            self._event_version = 1
        self._events.append(VersionedEvent(version=self._event_version, event=dict(event)))

    def set_explanation(self, stage: str, explanation: dict[str, object]) -> None:
        self._explanations[stage] = dict(explanation)

    def set_stage_info(self, stage: str, deps: list[str], outs: list[str]) -> None:
        self._stage_infos[stage] = {"name": stage, "deps": list(deps), "outs": list(outs)}

    def set_commit_result(self, committed: list[str], failed: list[str] | None = None) -> None:
        self._committed = list(committed)
        self._failed = list(failed) if failed is not None else list[str]()

    def set_diff_result(self, path: str, result: dict[str, object]) -> None:
        self._diff_results[path] = dict(result)

    def inject_error(self, method: str, code: int, message: str) -> None:
        self._error_for_method[method] = (code, message)

    def clear_error(self, method: str) -> None:
        _ = self._error_for_method.pop(method, None)

    async def start(self, socket_path: Path) -> None:
        if self._task_group is not None:
            raise RuntimeError("Server already started")
        self._socket_path = socket_path
        with contextlib.suppress(OSError):
            if socket_path.exists():
                socket_path.unlink()
        self._listener = await anyio.create_unix_listener(socket_path)
        self._task_group = anyio.create_task_group()
        task_group = self._task_group
        assert task_group is not None
        await task_group.__aenter__()
        listener = self._listener
        assert listener is not None
        task_group.start_soon(self._serve, listener)

    async def stop(self) -> None:
        if self._task_group is None:
            return
        task_group = self._task_group
        task_group.cancel_scope.cancel()
        await task_group.__aexit__(None, None, None)
        self._task_group = None
        self._listener = None
        if self._socket_path is not None:
            with contextlib.suppress(OSError):
                if self._socket_path.exists():
                    self._socket_path.unlink()
        self._socket_path = None

    async def __aenter__(self) -> FakeRpcServer:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.stop()

    async def _serve(self, listener: SocketListener) -> None:
        async with listener, anyio.create_task_group() as tg:
            while True:
                conn = await listener.accept()
                tg.start_soon(self._handle_connection, conn)

    async def _handle_connection(self, conn: SocketStream) -> None:
        async with conn:
            try:
                await self._process_requests(conn)
            except Exception:
                return

    async def _process_requests(self, conn: SocketStream) -> None:
        buffer: bytes = b""
        while True:
            if len(buffer) >= _MAX_MESSAGE_SIZE:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Request too large"},
                    "id": None,
                }
                await conn.send(json.dumps(error_response).encode() + b"\n")
                return

            max_to_receive = _MAX_MESSAGE_SIZE - len(buffer)
            chunk = await conn.receive(max_to_receive)
            if not chunk:
                break
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    request: object = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "Parse error"},
                        "id": None,
                    }
                    await conn.send(json.dumps(error_response).encode() + b"\n")
                    continue

                response = await self._handle_request(request)
                if response is not None:
                    await conn.send(json.dumps(response).encode() + b"\n")

    async def _handle_request(self, request: object) -> JsonRpcResponse | None:
        validated = self._validate_request(request)
        if validated is None:
            return self._json_rpc_error(-32600, "Invalid Request", None, force=True)

        method, params, request_id = validated

        injected = self._error_for_method.get(method)
        if injected is not None:
            code, message = injected
            return self._json_rpc_error(code, message, request_id)

        match method:
            case "run":
                stages = params.get("stages")
                force = params.get("force", False)
                self._run_requests.append({"stages": stages, "force": force})
                return self._json_rpc_response("accepted", request_id)
            case "cancel":
                return self._json_rpc_response("accepted", request_id)
            case "commit":
                return self._json_rpc_response(
                    {"committed": list(self._committed), "failed": list(self._failed)},
                    request_id,
                )
            case "status":
                return self._json_rpc_response(
                    {
                        "state": self._status_state,
                        "running": list(self._running),
                        "pending": list(self._pending),
                    },
                    request_id,
                )
            case "stages":
                return self._json_rpc_response({"stages": list(self._stages)}, request_id)
            case "stage_info":
                stage = params.get("stage")
                if not isinstance(stage, str):
                    return self._json_rpc_error(
                        -32602, "Invalid params: stage must be string", request_id
                    )
                info = self._stage_infos.get(stage)
                if info is None:
                    return self._json_rpc_error(
                        -32602, f"Invalid params: Unknown stage: {stage}", request_id
                    )
                return self._json_rpc_response(dict(info), request_id)
            case "explain":
                stage = params.get("stage")
                if not isinstance(stage, str):
                    return self._json_rpc_error(
                        -32602, "Invalid params: stage must be string", request_id
                    )
                explanation = self._explanations.get(stage)
                if explanation is None:
                    return self._json_rpc_error(
                        -32602, f"Invalid params: Unknown stage: {stage}", request_id
                    )
                return self._json_rpc_response(dict(explanation), request_id)
            case "events_since":
                version = params.get("version", 0)
                if isinstance(version, bool) or not isinstance(version, int):
                    return self._json_rpc_error(
                        -32602,
                        f"Invalid params: version must be integer between 0 and {_MAX_VERSION}",
                        request_id,
                    )
                if version < 0 or version > _MAX_VERSION:
                    return self._json_rpc_error(
                        -32602,
                        f"Invalid params: version must be integer between 0 and {_MAX_VERSION}",
                        request_id,
                    )
                result = self._events_since(version)
                return self._json_rpc_response(result, request_id)
            case "set_on_error":
                mode = params.get("mode")
                if not isinstance(mode, str):
                    return self._json_rpc_error(
                        -32602, "Invalid params: mode must be string", request_id
                    )
                if mode not in ("fail", "keep_going"):
                    return self._json_rpc_error(
                        -32602, f"Invalid on_error mode: {mode}", request_id
                    )
                self._on_error_mode = mode
                return self._json_rpc_response(True, request_id)
            case "diff_output":
                path = params.get("path")
                if not isinstance(path, str):
                    return self._json_rpc_error(
                        -32602, "Invalid params: path must be string", request_id
                    )
                result = self._diff_results.get(path)
                if result is None:
                    return self._json_rpc_error(
                        -32602, f"Invalid params: No diff result for: {path}", request_id
                    )
                return self._json_rpc_response(dict(result), request_id)
            case _:
                return self._json_rpc_error(-32601, "Method not found", request_id)

    def _events_since(self, since_version: int) -> EventsResult:
        if since_version > self._event_version:
            events = list(self._events)
        else:
            events = [event for event in self._events if event["version"] > since_version]
        return {"version": self._event_version, "events": events}

    @staticmethod
    def _validate_request(
        raw: object,
    ) -> tuple[str, dict[str, object], JsonRpcId] | None:
        if not isinstance(raw, dict):
            return None
        raw_map = cast("dict[str, object]", raw)
        if raw_map.get("jsonrpc") != "2.0":
            return None
        method = raw_map.get("method")
        if not isinstance(method, str):
            return None
        params = raw_map.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None
        request_id = raw_map.get("id")
        if request_id is None or isinstance(request_id, str | int):
            return method, dict(params), request_id
        return None

    @staticmethod
    def _json_rpc_response(result: object, request_id: JsonRpcId) -> JsonRpcSuccessResponse | None:
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "result": result, "id": request_id}

    @staticmethod
    def _json_rpc_error(
        code: int, message: str, request_id: JsonRpcId, *, force: bool = False
    ) -> JsonRpcErrorResponse | None:
        if request_id is None and not force:
            return None
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id,
        }
