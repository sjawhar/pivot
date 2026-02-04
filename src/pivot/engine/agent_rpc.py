from __future__ import annotations

import contextlib
import enum
import json
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, ValidationError, field_validator

from pivot import explain as explain_mod
from pivot import parameters
from pivot.config import io as config_io
from pivot.engine.types import CancelRequested, EngineState, OutputEvent, RunRequested
from pivot.types import OnError, StageExplanation

if TYPE_CHECKING:
    from pathlib import Path

    from anyio.abc import SocketStream

    from pivot.engine.engine import Engine
    from pivot.engine.types import InputEvent
    from pivot.registry import RegistryStageInfo

__all__ = [
    "AgentRpcHandler",
    "AgentRpcSource",
    "BroadcastEventSink",
    "EventBuffer",
    "EventsResult",
    "VersionedEvent",
]

_logger = logging.getLogger(__name__)

_MAX_VERSION = 2**63 - 1


def _json_default(obj: object) -> object:
    """Custom JSON encoder for enums and other non-serializable types."""
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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


class QueryStageInfoResult(TypedDict):
    """Result type for stage_info query."""

    name: str
    deps: list[str]
    outs: list[str]


type QueryResult = (
    QueryStatusResult | QueryStagesResult | EventsResult | StageExplanation | QueryStageInfoResult
)


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
    # Cast to list[object] for type-safe iteration, then validate each item
    for item in cast("list[object]", raw):
        if not isinstance(item, str):
            return "stages must be list of strings"
    # After validation, we know all items are strings
    return cast("list[str]", raw)


class AgentRpcHandler:
    """Handles JSON-RPC queries that need engine/inspector access."""

    _engine: Engine
    _event_buffer: EventBuffer | None

    def __init__(
        self,
        *,
        engine: Engine,
        event_buffer: EventBuffer | None = None,
    ) -> None:
        self._engine = engine
        self._event_buffer = event_buffer

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

    def _get_stage_info(self, params: dict[str, object]) -> tuple[str, RegistryStageInfo]:
        """Extract and validate stage param, return (stage_name, reg_info)."""
        stage = params.get("stage")
        if not isinstance(stage, str):
            raise ValueError("stage must be string")
        pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
        if pipeline is None:
            raise ValueError("No pipeline loaded")
        try:
            reg_info = pipeline.get(stage)
        except KeyError:
            raise ValueError(f"Unknown stage: {stage}") from None
        return stage, reg_info

    async def handle_query(
        self, method: str, params: dict[str, object] | None = None
    ) -> QueryResult:
        """Handle a query request and return the result."""
        if params is None:
            params = {}

        match method:
            case "events_since":
                if self._event_buffer is None:
                    raise ValueError("Event buffer not configured")
                version = params.get("version", 0)
                # Note: isinstance(True, int) is True in Python, so check bool explicitly
                if isinstance(version, bool) or not isinstance(version, int):
                    raise ValueError(f"version must be integer between 0 and {_MAX_VERSION}")
                if version < 0 or version > _MAX_VERSION:
                    raise ValueError(f"version must be integer between 0 and {_MAX_VERSION}")
                return self._event_buffer.events_since(version)
            case "status":
                return QueryStatusResult(
                    state="idle" if self._engine.state == EngineState.IDLE else "active",
                    running=list[str](),
                    pending=list[str](),
                )
            case "stages":
                pipeline = self._engine._pipeline  # pyright: ignore[reportPrivateUsage]
                if pipeline is None:
                    return QueryStagesResult(stages=list[str]())
                return QueryStagesResult(stages=pipeline.list_stages())
            case "explain":
                stage, reg_info = self._get_stage_info(params)

                def _get_explanation() -> StageExplanation:
                    try:
                        overrides = parameters.load_params_yaml()
                    except Exception as e:
                        raise ValueError(f"Failed to load params.yaml: {e}") from e
                    return explain_mod.get_stage_explanation(
                        stage_name=stage,
                        fingerprint=reg_info["fingerprint"],
                        deps=reg_info["deps_paths"],
                        outs_paths=reg_info["outs_paths"],
                        params_instance=reg_info["params"],
                        overrides=overrides,
                        state_dir=config_io.get_state_dir(),
                    )

                return await anyio.to_thread.run_sync(_get_explanation)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType] - anyio stub issue
            case "stage_info":
                stage, reg_info = self._get_stage_info(params)
                return QueryStageInfoResult(
                    name=stage,
                    deps=reg_info["deps_paths"],
                    outs=reg_info["outs_paths"],
                )
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

        try:
            self._socket_path.chmod(0o600)  # Owner-only access
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
                        await conn.send(
                            json.dumps(response, default=_json_default).encode() + b"\n"
                        )

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
                result = await self._handler.handle_query(method, params)
                return _json_rpc_response(result, request_id)
            except ValueError as e:
                # Distinguish "unknown method" from "invalid params" by checking message
                msg = str(e)
                if "Unknown query method" in msg:
                    return _json_rpc_error(-32601, "Method not found", request_id)
                # Invalid params or other handler errors
                return _json_rpc_error(-32602, f"Invalid params: {msg}", request_id)

        # Unknown method, no handler
        return _json_rpc_error(-32601, "Method not found", request_id)


class BroadcastEventSink:
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


class VersionedEvent(TypedDict):
    """Event with version number."""

    version: int
    event: OutputEvent


class EventsResult(TypedDict):
    """Result of events_since query."""

    version: int
    events: list[VersionedEvent]


class EventBuffer:
    """Ring buffer for event polling via events_since.

    Thread-safe: uses threading.Lock since buffer is accessed from both
    sync (events_since) and async (handle) contexts.
    """

    _max_events: int
    _events: deque[tuple[int, OutputEvent]]
    _version: int
    _lock: threading.Lock

    def __init__(self, max_events: int = 1000) -> None:
        self._max_events = max_events
        self._events = deque[tuple[int, OutputEvent]](maxlen=max_events)
        self._version = 0
        self._lock = threading.Lock()

    def handle_sync(self, event: OutputEvent) -> None:
        """Store event with version number (sync, thread-safe).

        Version wraps at _MAX_VERSION to prevent overflow. Clients polling
        with a version from before wrap will receive all buffered events.
        """
        with self._lock:
            self._version += 1
            # Wrap version to prevent overflow - clients will get all events on wrap
            if self._version > _MAX_VERSION:
                self._version = 1
            self._events.append((self._version, event))

    async def handle(self, event: OutputEvent) -> None:
        """Async wrapper for sink interface compatibility."""
        self.handle_sync(event)

    def events_since(self, since_version: int) -> EventsResult:
        """Return events after the given version.

        If since_version > current version (e.g., after version wrap), returns
        all buffered events to ensure clients don't miss events. Clients should
        dedupe by version since wraparound may return previously-seen events.
        """
        with self._lock:
            # Handle version wrap: if client's version is higher than current,
            # they're from before a wrap - return all events
            if since_version > self._version:
                result = [VersionedEvent(version=v, event=e) for v, e in self._events]
            else:
                result = [
                    VersionedEvent(version=v, event=e) for v, e in self._events if v > since_version
                ]
            return EventsResult(version=self._version, events=result)

    async def close(self) -> None:
        """No cleanup needed for EventBuffer."""
