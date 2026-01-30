from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import socket
import uuid
from typing import TYPE_CHECKING, Any, cast

from pivot import registry
from pivot.types import (
    AgentCancelResult,
    AgentRunParams,
    AgentRunStartResult,
    AgentStageInfo,
    AgentStagesResult,
    AgentStatusResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.engine.engine import Engine

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
_EXECUTION_IN_PROGRESS = -32001
_STAGE_NOT_FOUND = -32002


def _make_error(code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create JSON-RPC error object."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return error


def _make_response(
    req_id: int | str | None, result: object = None, error: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create JSON-RPC response object."""
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    return response


class AgentServer:
    """JSON-RPC server for agent control of pipeline execution.

    This is a stateless RPC facade - all execution state is managed by Engine.
    """

    _engine: Engine
    _socket_path: Path
    _connected_clients: int

    def __init__(self, engine: Engine, socket_path: Path) -> None:
        self._engine = engine
        self._socket_path = socket_path
        self._server: asyncio.Server | None = None
        self._connected_clients = 0

    @property
    def socket_path(self) -> Path:
        """Return the socket path."""
        return self._socket_path

    @property
    def connected(self) -> bool:
        """Return True if any clients are connected."""
        return self._connected_clients > 0

    @property
    def connected_count(self) -> int:
        """Return the number of connected clients."""
        return self._connected_clients

    async def start(self) -> asyncio.Server:
        """Start the server, binding to the Unix socket."""
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for stale socket
        if self._socket_path.exists():
            if self._is_socket_alive():
                msg = f"Another server is already running at {self._socket_path}"
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

        logger.info(f"Agent server listening on {self._socket_path}")
        return self._server

    async def stop(self) -> None:
        """Stop the server and clean up."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self._socket_path.unlink(missing_ok=True)
        logger.info("Agent server stopped")

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
        self._connected_clients += 1
        peer = writer.get_extra_info("peername") or "unknown"
        logger.debug(f"Client connected: {peer}")

        try:
            while True:
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
                response = await self._dispatch(cast("dict[str, Any]", request))
                if response is not None:
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()

        except (ConnectionResetError, BrokenPipeError):
            logger.debug(f"Client {peer} disconnected")
        except Exception:
            logger.exception(f"Error handling client {peer}")
        finally:
            self._connected_clients -= 1
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
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

        try:
            if method == "run":
                result = await self._handle_run(cast("AgentRunParams", cast("object", params)))
            elif method == "status":
                result = await self._handle_status(cast("dict[str, Any]", params))
            elif method == "stages":
                result = await self._handle_stages()
            elif method == "cancel":
                result = await self._handle_cancel()
            else:
                if is_notification:
                    return None
                return _make_response(
                    req_id, error=_make_error(_METHOD_NOT_FOUND, f"Method not found: {method}")
                )

            if is_notification:
                return None
            return _make_response(req_id, result=result)

        except _ExecutionInProgressError:
            if is_notification:
                return None
            return _make_response(
                req_id, error=_make_error(_EXECUTION_IN_PROGRESS, "Execution already in progress")
            )
        except _StageNotFoundError as e:
            if is_notification:
                return None
            data = {"suggestions": e.suggestions} if e.suggestions else None
            return _make_response(
                req_id, error=_make_error(_STAGE_NOT_FOUND, f"Stage not found: {e.stage}", data)
            )
        except Exception:
            logger.exception("Error handling RPC request")
            if is_notification:
                return None
            return _make_response(req_id, error=_make_error(_INTERNAL_ERROR, "Internal error"))

    async def _handle_run(self, params: AgentRunParams) -> AgentRunStartResult:
        """Handle run() RPC method."""
        stages = params["stages"] if "stages" in params else None
        if stages:
            all_stages = set(registry.REGISTRY.list_stages())
            for stage in stages:
                if stage not in all_stages:
                    suggestions = difflib.get_close_matches(stage, all_stages, n=3, cutoff=0.6)
                    raise _StageNotFoundError(stage, suggestions)

        run_id = str(uuid.uuid4())[:12]
        force = params["force"] if "force" in params else False

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._engine.try_start_run,
            run_id,
            stages,
            force,
        )

        if "reason" in result:
            raise _ExecutionInProgressError

        return result

    async def _handle_status(self, params: dict[str, Any]) -> AgentStatusResult:
        """Handle status() RPC method.

        Uses run_in_executor to avoid blocking the event loop on lock acquisition.
        """
        run_id = params["run_id"] if "run_id" in params else None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._engine.get_execution_status, run_id)

    async def _handle_stages(self) -> AgentStagesResult:
        """Handle stages() RPC method."""
        stages = list[AgentStageInfo]()
        for name in registry.REGISTRY.list_stages():
            info = registry.REGISTRY.get(name)
            stages.append(
                AgentStageInfo(
                    name=name,
                    deps=info["deps_paths"],
                    # Registry always stores single-file outputs (multi-file are expanded)
                    outs=[str(out.path) for out in info["outs"]],
                )
            )
        return AgentStagesResult(stages=stages)

    async def _handle_cancel(self) -> AgentCancelResult:
        """Handle cancel() RPC method.

        Uses run_in_executor to avoid blocking the event loop on lock acquisition.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._engine.request_cancel)


class _ExecutionInProgressError(Exception):
    """Raised when execution is already in progress."""


class _StageNotFoundError(Exception):
    """Raised when a stage is not found."""

    stage: str
    suggestions: list[str]

    def __init__(self, stage: str, suggestions: list[str]) -> None:
        self.stage = stage
        self.suggestions = suggestions
        super().__init__(f"Stage not found: {stage}")
