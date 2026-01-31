from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, ValidationError, field_validator

from pivot.engine.types import CancelRequested, EngineState, OutputEvent, RunRequested
from pivot.types import OnError

if TYPE_CHECKING:
    from pathlib import Path

    from anyio.abc import SocketStream

    from pivot.engine.engine import Engine
    from pivot.engine.types import InputEvent

__all__ = ["AgentEventSink", "AgentRpcHandler", "AgentRpcSource"]

_logger = logging.getLogger(__name__)

# JSON-RPC 2.0: id can be string, number, or null (for notifications)
type JsonRpcId = str | int | None


class JsonRpcErrorDetail(TypedDict):
    """JSON-RPC error object structure."""

    code: int
    message: str


class JsonRpcSuccessResponse(TypedDict):
    """JSON-RPC success response structure."""

    jsonrpc: Literal["2.0"]
    result: object
    id: JsonRpcId


class JsonRpcErrorResponse(TypedDict):
    """JSON-RPC error response structure."""

    jsonrpc: Literal["2.0"]
    error: JsonRpcErrorDetail
    id: JsonRpcId


type JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request validation model."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, object] = dict[str, object]()
    id: str | int | None = None

    @field_validator("jsonrpc", mode="before")
    @classmethod
    def validate_jsonrpc(cls, v: object) -> Literal["2.0"]:
        """Validate jsonrpc version is exactly '2.0'."""
        if v != "2.0":
            raise ValueError("Only JSON-RPC 2.0 is supported (jsonrpc must be '2.0')")
        return "2.0"

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v: object) -> str | int | None:
        """Validate id is string, int, or None."""
        if v is None or isinstance(v, str | int):
            return v
        raise ValueError("id must be string, integer, or null")


class QueryStatusResult(TypedDict):
    """Result type for the 'status' query."""

    state: Literal["idle", "active"]
    running: list[str]
    pending: list[str]


class QueryStagesResult(TypedDict):
    """Result type for the 'stages' query."""

    stages: list[str]


type QueryResult = QueryStatusResult | QueryStagesResult


def _validate_json_rpc_request(
    raw: object,
) -> tuple[str, dict[str, object], JsonRpcId] | None:
    """Validate and extract fields from a JSON-RPC request.

    Returns (method, params, id) tuple if valid, None if invalid.
    """
    if not isinstance(raw, dict):
        return None

    try:
        request = JsonRpcRequest.model_validate(raw)
        return request.method, request.params, request.id
    except ValidationError:
        return None


# Limits for security
_MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
_CLIENT_TIMEOUT = 300  # 5 minutes

# Custom error codes
_ERR_STAGE_NOT_FOUND = -32001


def _json_rpc_response(result: object, request_id: JsonRpcId) -> JsonRpcSuccessResponse | None:
    """Build a JSON-RPC response, or None for notifications (no request_id)."""
    if request_id is None:
        return None
    return JsonRpcSuccessResponse(jsonrpc="2.0", result=result, id=request_id)


def _json_rpc_error(code: int, message: str, request_id: JsonRpcId) -> JsonRpcErrorResponse | None:
    """Build a JSON-RPC error response, or None for notifications."""
    if request_id is None:
        return None
    return JsonRpcErrorResponse(
        jsonrpc="2.0", error=JsonRpcErrorDetail(code=code, message=message), id=request_id
    )


def _validate_stages_param(raw: object) -> list[str] | None | str:
    """Validate 'stages' parameter for run command.

    Returns:
        list[str]: Valid stages list
        None: stages was None (meaning all stages)
        str: Error message if validation failed
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        return "stages must be list of strings"
    for item in cast("list[object]", raw):
        if not isinstance(item, str):
            return "stages must be list of strings"
    return cast("list[str]", raw)


class AgentRpcHandler:
    """Handles JSON-RPC queries that need engine/inspector access."""

    _engine: Engine

    def __init__(self, *, engine: Engine) -> None:
        self._engine = engine

    def validate_stages(self, stages: list[str] | None) -> str | None:
        """Validate stage names exist. Returns error message or None if valid."""
        if stages is None:
            return None
        pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
        if pipeline is None:
            return "No pipeline loaded"
        available = set(pipeline.list_stages())
        unknown = [s for s in stages if s not in available]
        if unknown:
            return f"Unknown stages: {', '.join(unknown)}"
        return None

    async def handle_query(self, method: str) -> QueryResult:
        """Handle a query request and return the result."""
        match method:
            case "status":
                return QueryStatusResult(
                    state="idle" if self._engine.state == EngineState.IDLE else "active",
                    running=list[str](),
                    pending=list[str](),
                )
            case "stages":
                # Access _pipeline directly - this handler is tightly coupled to engine internals
                pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
                if pipeline is None:
                    return QueryStagesResult(stages=list[str]())
                return QueryStagesResult(stages=pipeline.list_stages())
            case _:
                raise ValueError(f"Unknown query method: {method}")


class AgentRpcSource:
    """Async source that receives commands from agents via Unix socket.

    Implements JSON-RPC 2.0 over Unix socket. Commands (run, cancel) become
    input events. Queries are handled directly and return responses.
    """

    _socket_path: Path
    _handler: AgentRpcHandler | None

    def __init__(self, *, socket_path: Path, handler: AgentRpcHandler | None = None) -> None:
        self._socket_path = socket_path
        self._handler = handler

    async def run(self, send: MemoryObjectSendStream[InputEvent]) -> None:
        """Listen for agent connections and process requests."""
        # Clean up stale socket (suppress errors if file changed/removed)
        with contextlib.suppress(OSError):
            if self._socket_path.exists():
                self._socket_path.unlink()

        listener = await anyio.create_unix_listener(self._socket_path)
        self._socket_path.chmod(0o600)  # Owner-only access

        try:
            async with listener, anyio.create_task_group() as tg:
                while True:
                    conn = await listener.accept()
                    tg.start_soon(self._handle_connection, conn, send)
        finally:
            # Clean up socket file on exit (suppress errors to avoid masking original exception)
            with contextlib.suppress(OSError):
                if self._socket_path.exists():
                    self._socket_path.unlink()

    async def _handle_connection(
        self,
        conn: SocketStream,
        send: MemoryObjectSendStream[InputEvent],
    ) -> None:
        """Handle a single agent connection."""
        async with conn:
            try:
                with anyio.move_on_after(_CLIENT_TIMEOUT) as cancel_scope:
                    await self._process_requests(conn, send)

                if cancel_scope.cancelled_caught:
                    # Client timed out - send error before closing
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": "Connection timeout"},
                        "id": None,
                    }
                    with anyio.move_on_after(1.0):  # Brief timeout for error send
                        await conn.send(json.dumps(error_response).encode() + b"\n")
            except Exception:
                _logger.exception("Error handling agent connection")

    async def _process_requests(
        self,
        conn: SocketStream,
        send: MemoryObjectSendStream[InputEvent],
    ) -> None:
        """Process JSON-RPC requests from a connection."""
        buffer = b""

        while True:
            # Check buffer size BEFORE receiving more data to prevent unbounded growth
            if len(buffer) >= _MAX_MESSAGE_SIZE:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Request too large"},
                    "id": None,
                }
                await conn.send(json.dumps(error_response).encode() + b"\n")
                return

            # Limit receive size to prevent single-chunk overflow
            max_to_receive = _MAX_MESSAGE_SIZE - len(buffer)
            chunk = await conn.receive(max_to_receive)
            if not chunk:
                break

            buffer += chunk

            # Process complete lines
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue

                try:
                    request = json.loads(line.decode())
                    response = await self._handle_request(request, send)

                    if response is not None:
                        await conn.send(json.dumps(response).encode() + b"\n")

                except (json.JSONDecodeError, UnicodeDecodeError):
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "Parse error"},
                        "id": None,
                    }
                    await conn.send(json.dumps(error_response).encode() + b"\n")

    async def _handle_request(
        self,
        request: object,
        send: MemoryObjectSendStream[InputEvent],
    ) -> JsonRpcResponse | None:
        """Handle a single JSON-RPC request.

        Returns response dict, or None for notifications.
        """
        # Validate and extract request fields
        validated = _validate_json_rpc_request(request)
        if validated is None:
            return _json_rpc_error(-32600, "Invalid Request", None)

        method, params, request_id = validated

        # Commands become input events
        if method == "run":
            # Validate stages param
            stages = _validate_stages_param(params.get("stages"))
            if isinstance(stages, str):
                return _json_rpc_error(-32602, f"Invalid params: {stages}", request_id)

            # Validate force param
            force = params.get("force", False)
            if not isinstance(force, bool):
                return _json_rpc_error(-32602, "Invalid params: force must be boolean", request_id)

            # Validate stage names exist (if handler available)
            if self._handler is not None and stages is not None:
                validation_error = self._handler.validate_stages(stages)
                if validation_error:
                    return _json_rpc_error(_ERR_STAGE_NOT_FOUND, validation_error, request_id)

            event = RunRequested(
                type="run_requested",
                stages=stages,
                force=force,
                reason="agent",
                single_stage=False,
                parallel=True,
                max_workers=None,
                no_commit=False,
                no_cache=False,
                on_error=OnError.FAIL,
                cache_dir=None,
                allow_uncached_incremental=False,
                checkout_missing=False,
            )
            await send.send(event)
            return _json_rpc_response("accepted", request_id)

        if method == "cancel":
            await send.send(CancelRequested(type="cancel_requested"))
            return _json_rpc_response("accepted", request_id)

        # Queries handled by handler (if available)
        if self._handler is not None:
            try:
                result = await self._handler.handle_query(method)
                return _json_rpc_response(result, request_id)
            except ValueError:
                return _json_rpc_error(-32601, "Method not found", request_id)

        # Unknown method, no handler
        return _json_rpc_error(-32601, "Method not found", request_id)


class AgentEventSink:
    """Async sink that broadcasts events to connected agents.

    Thread safety: All subscriber dict operations are protected by a lock
    to prevent race conditions between handle(), subscribe(), and unsubscribe().
    """

    _buffer_size: int
    _subscribers: dict[str, MemoryObjectSendStream[OutputEvent]]
    _lock: anyio.Lock

    def __init__(self, buffer_size: int = 64) -> None:
        self._buffer_size = buffer_size
        self._subscribers = dict[str, MemoryObjectSendStream[OutputEvent]]()
        self._lock = anyio.Lock()

    async def subscribe(self, client_id: str) -> MemoryObjectReceiveStream[OutputEvent]:
        """Subscribe a client to receive events. Returns receive channel."""
        send, recv = anyio.create_memory_object_stream[OutputEvent](self._buffer_size)
        async with self._lock:
            self._subscribers[client_id] = send
        return recv

    async def unsubscribe(self, client_id: str) -> None:
        """Unsubscribe a client and close its send channel."""
        async with self._lock:
            send = self._subscribers.pop(client_id, None)
        if send is not None:
            await send.aclose()

    async def handle(self, event: OutputEvent) -> None:
        """Broadcast event to all subscribers."""
        to_remove = list[str]()
        async with self._lock:
            for client_id, send in list(self._subscribers.items()):
                try:
                    send.send_nowait(event)
                except anyio.WouldBlock:
                    # Client too slow, drop event
                    _logger.debug("Dropping event for slow client %s", client_id)
                except anyio.ClosedResourceError:
                    # Client disconnected, mark for removal
                    _logger.debug("Client %s disconnected, removing subscriber", client_id)
                    to_remove.append(client_id)
            # Clean up disconnected subscribers
            for client_id in to_remove:
                self._subscribers.pop(client_id, None)

    async def close(self) -> None:
        """Close all subscriber channels.

        Errors closing individual channels are logged but do not prevent
        other channels from being closed.
        """
        errors = list[tuple[str, Exception]]()
        async with self._lock:
            for client_id, send in list(self._subscribers.items()):
                try:
                    await send.aclose()
                except Exception as e:
                    errors.append((client_id, e))
            self._subscribers.clear()
        for client_id, error in errors:
            _logger.warning("Error closing channel for client %s: %s", client_id, error)
