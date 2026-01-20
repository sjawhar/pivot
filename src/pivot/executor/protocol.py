"""JSON-RPC protocol for worker-coordinator communication.

Workers connect to the coordinator via Unix socket and use JSON-RPC 2.0
to register, request tasks, report results, and send heartbeats.

Methods:
- register: Register as a worker
- request_task: Request the next available task
- report_result: Report task completion/failure
- heartbeat: Keep-alive signal
- get_state: Get current execution state snapshot
- get_events_since: Get events since a version for catch-up
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, final

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.executor.state import ExecutionState
    from pivot.types import ExecutionSnapshot, StageResult, StateEvent, WorkerStageTask

# Type aliases for callbacks
ResultCallback = Callable[[str, str, "StageResult"], None]
DisconnectCallback = Callable[[str], None]

logger = logging.getLogger(__name__)

# Protocol limits
_MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
_CLIENT_TIMEOUT = 300.0  # 5 minutes idle timeout

# JSON-RPC 2.0 error codes
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

# Custom error codes
_WORKER_NOT_REGISTERED = -32001
_NO_TASKS_AVAILABLE = -32002
_SHUTTING_DOWN = -32003


def _make_error(code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create JSON-RPC error object."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return error


def _make_response(
    req_id: int | str | None, result: Any = None, error: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create JSON-RPC response object."""
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    return response


@final
class WorkerProtocolHandler:
    """Handles JSON-RPC protocol for a single worker connection.

    The handler is created per-connection and delegates actual work
    to the Coordinator instance.
    """

    def __init__(
        self,
        state: ExecutionState,
        task_queue: asyncio.Queue[WorkerStageTask],
        result_callback: ResultCallback,
        disconnect_callback: DisconnectCallback,
        shutdown_event: asyncio.Event,
    ) -> None:
        self._state = state
        self._task_queue = task_queue
        self._result_callback = result_callback
        self._disconnect_callback = disconnect_callback
        self._shutdown_event = shutdown_event
        self._worker_id: str | None = None

    @property
    def worker_id(self) -> str | None:
        """Return the registered worker ID, if any."""
        return self._worker_id

    async def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch a JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        is_notification = "id" not in request

        # Validate params type
        if not isinstance(params, dict):
            if is_notification:
                return None
            return _make_response(
                req_id, error=_make_error(_INVALID_PARAMS, "Params must be an object")
            )

        # Narrow type after isinstance check
        typed_params = cast("dict[str, Any]", params)

        try:
            if method == "register":
                result = self._handle_register(typed_params)
            elif method == "request_task":
                result = await self._handle_request_task()
            elif method == "report_result":
                result = self._handle_report_result(typed_params)
            elif method == "heartbeat":
                result = self._handle_heartbeat()
            elif method == "get_state":
                result = self._handle_get_state()
            elif method == "get_events_since":
                result = self._handle_get_events_since(typed_params)
            else:
                if is_notification:
                    return None
                return _make_response(
                    req_id, error=_make_error(_METHOD_NOT_FOUND, f"Method not found: {method}")
                )

            if is_notification:
                return None
            return _make_response(req_id, result=result)

        except _WorkerNotRegisteredError:
            if is_notification:
                return None
            return _make_response(
                req_id, error=_make_error(_WORKER_NOT_REGISTERED, "Worker not registered")
            )
        except Exception:
            logger.exception("Error handling RPC request")
            if is_notification:
                return None
            return _make_response(req_id, error=_make_error(_INTERNAL_ERROR, "Internal error"))

    def _handle_register(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle register() method."""
        worker_id = params.get("worker_id")
        if not worker_id or not isinstance(worker_id, str):
            raise ValueError("worker_id required")

        self._worker_id = worker_id
        self._state.register_worker(worker_id)
        logger.info(f"Worker registered: {worker_id}")

        return {"status": "registered", "worker_id": worker_id}

    async def _handle_request_task(self) -> dict[str, Any]:
        """Handle request_task() method."""
        if self._worker_id is None:
            raise _WorkerNotRegisteredError

        # Check for shutdown
        if self._shutdown_event.is_set():
            return {"task": None, "shutdown": True}

        # Try to get a task with timeout
        try:
            task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
            return {"task": task, "shutdown": False}
        except TimeoutError:
            # No task available right now
            return {"task": None, "shutdown": False}

    def _handle_report_result(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle report_result() method."""
        if self._worker_id is None:
            raise _WorkerNotRegisteredError

        stage_name = params.get("stage_name")
        result = params.get("result")

        if not stage_name or result is None:
            raise ValueError("stage_name and result required")

        # Call back to coordinator to process result
        self._result_callback(self._worker_id, stage_name, cast("StageResult", result))

        return {"status": "acknowledged"}

    def _handle_heartbeat(self) -> dict[str, Any]:
        """Handle heartbeat() method."""
        if self._worker_id is None:
            raise _WorkerNotRegisteredError

        self._state.worker_heartbeat(self._worker_id)
        return {"status": "ok"}

    def _handle_get_state(self) -> ExecutionSnapshot:
        """Handle get_state() method."""
        return self._state.get_snapshot()

    def _handle_get_events_since(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle get_events_since() method."""
        version = params.get("version", 0)
        if not isinstance(version, int):
            raise ValueError("version must be an integer")

        events: list[StateEvent] = self._state.get_events_since(version)
        return {"events": events, "current_version": self._state.version}

    def on_disconnect(self) -> None:
        """Called when the client disconnects."""
        if self._worker_id is not None:
            # Notify coordinator to handle orphaned stages before unregistering
            self._disconnect_callback(self._worker_id)
            self._state.unregister_worker(self._worker_id)
            logger.info(f"Worker disconnected: {self._worker_id}")


@final
class CoordinatorServer:
    """JSON-RPC server for coordinator-worker communication.

    Listens on a Unix socket and accepts worker connections.
    Each worker gets a WorkerProtocolHandler instance.
    """

    def __init__(
        self,
        state: ExecutionState,
        task_queue: asyncio.Queue[WorkerStageTask],
        result_callback: ResultCallback,
        disconnect_callback: DisconnectCallback,
        socket_path: Path,
    ) -> None:
        self._state = state
        self._task_queue = task_queue
        self._result_callback = result_callback
        self._disconnect_callback = disconnect_callback
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None
        self._shutdown_event = asyncio.Event()
        self._handlers: list[WorkerProtocolHandler] = []

    @property
    def socket_path(self) -> Path:
        """Return the socket path."""
        return self._socket_path

    @property
    def connected_workers(self) -> int:
        """Return number of connected workers."""
        return len([h for h in self._handlers if h.worker_id is not None])

    async def start(self) -> asyncio.Server:
        """Start the server, binding to the Unix socket."""
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for stale socket
        if self._socket_path.exists():
            if self._is_socket_alive():
                msg = f"Another coordinator is already running at {self._socket_path}"
                raise RuntimeError(msg)
            # Stale socket - remove it
            logger.info(f"Removing stale socket: {self._socket_path}")
            self._socket_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
            limit=_MAX_MESSAGE_SIZE,
        )

        # Set socket permissions to owner-only
        os.chmod(self._socket_path, 0o600)

        logger.info(f"Coordinator server listening on {self._socket_path}")
        return self._server

    async def stop(self) -> None:
        """Stop the server and clean up."""
        self._shutdown_event.set()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self._socket_path.unlink(missing_ok=True)
        logger.info("Coordinator server stopped")

    def _is_socket_alive(self) -> bool:
        """Check if another server is listening on the socket."""
        test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            test_sock.settimeout(1.0)
            test_sock.connect(str(self._socket_path))
            return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False
        finally:
            test_sock.close()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a connected client."""
        handler = WorkerProtocolHandler(
            self._state,
            self._task_queue,
            self._result_callback,
            self._disconnect_callback,
            self._shutdown_event,
        )
        self._handlers.append(handler)
        peer = writer.get_extra_info("peername") or "unknown"
        logger.debug(f"Client connected: {peer}")

        try:
            while not self._shutdown_event.is_set():
                # Read with timeout
                try:
                    data = await asyncio.wait_for(reader.readline(), timeout=_CLIENT_TIMEOUT)
                except TimeoutError:
                    logger.debug(f"Client {peer} timed out")
                    break

                if not data:
                    break

                # Check message size
                if len(data) > _MAX_MESSAGE_SIZE:
                    response = _make_response(
                        None, error=_make_error(_INVALID_REQUEST, "Request too large")
                    )
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
                    break

                # Parse JSON
                try:
                    request = json.loads(data)
                except json.JSONDecodeError as e:
                    response = _make_response(
                        None, error=_make_error(_PARSE_ERROR, f"Parse error: {e}")
                    )
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
                    continue

                # Validate request structure
                if not isinstance(request, dict):
                    response = _make_response(
                        None, error=_make_error(_INVALID_REQUEST, "Request must be an object")
                    )
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
                    continue

                # Dispatch and respond
                response = await handler.dispatch(cast("dict[str, Any]", request))
                if response is not None:
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()

        except (ConnectionResetError, BrokenPipeError):
            logger.debug(f"Client {peer} disconnected")
        except Exception:
            logger.exception(f"Error handling client {peer}")
        finally:
            handler.on_disconnect()
            self._handlers.remove(handler)
            writer.close()
            await writer.wait_closed()


class _WorkerNotRegisteredError(Exception):
    """Raised when a worker tries to perform an action before registering."""


# =============================================================================
# Client-side helpers for workers
# =============================================================================


@final
class WorkerClient:
    """Client for workers to communicate with the coordinator."""

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id = 0

    async def connect(self) -> None:
        """Connect to the coordinator."""
        self._reader, self._writer = await asyncio.open_unix_connection(str(self._socket_path))
        logger.debug(f"Connected to coordinator at {self._socket_path}")

    async def disconnect(self) -> None:
        """Disconnect from the coordinator."""
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Make a JSON-RPC call and return the result."""
        if self._reader is None or self._writer is None:
            raise RuntimeError("Not connected")

        self._request_id += 1
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        self._writer.write(json.dumps(request).encode() + b"\n")
        await self._writer.drain()

        data = await self._reader.readline()
        if not data:
            raise RuntimeError("Connection closed")

        response = json.loads(data)
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"RPC error {error['code']}: {error['message']}")

        return response.get("result")

    async def register(self, worker_id: str, pid: int) -> dict[str, Any]:
        """Register as a worker."""
        return await self._call("register", {"worker_id": worker_id, "pid": pid})

    async def request_task(self) -> dict[str, Any]:
        """Request the next available task."""
        return await self._call("request_task")

    async def report_result(self, stage_name: str, result: StageResult) -> dict[str, Any]:
        """Report task completion."""
        return await self._call("report_result", {"stage_name": stage_name, "result": result})

    async def heartbeat(self) -> dict[str, Any]:
        """Send a heartbeat."""
        return await self._call("heartbeat")

    async def get_state(self) -> ExecutionSnapshot:
        """Get the current execution state snapshot."""
        return await self._call("get_state")

    async def get_events_since(self, version: int) -> dict[str, Any]:
        """Get events since a version."""
        return await self._call("get_events_since", {"version": version})
