"""Tests for agent_server module."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict
from unittest.mock import MagicMock

import pytest

from helpers import register_test_stage
from pivot import loaders, outputs
from pivot.tui import agent_server
from pivot.types import (
    AgentCancelResult,
    AgentRunRejection,
    AgentRunStartResult,
    AgentState,
    AgentStatusResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _StagesOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions
# =============================================================================


def _helper_stages_noop(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _StagesOutputs:
    _ = input_file
    return {"output": pathlib.Path("output.csv")}


def _helper_run_noop() -> None:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock WatchEngine for testing."""
    engine = MagicMock(spec=["try_start_agent_run", "get_agent_status", "request_agent_cancel"])
    engine.get_agent_status.return_value = AgentStatusResult(state=AgentState.IDLE)
    engine.request_agent_cancel.return_value = AgentCancelResult(cancelled=False)
    return engine


@pytest.fixture
def socket_path(tmp_path: Path) -> Path:
    """Create a temporary socket path."""
    return tmp_path / "test.sock"


# =============================================================================
# Helper Functions
# =============================================================================


def test_make_error_creates_proper_error_object() -> None:
    """Test _make_error creates proper error object."""
    error = agent_server._make_error(-32600, "Test error")
    assert error["code"] == -32600
    assert error["message"] == "Test error"
    assert "data" not in error


def test_make_error_with_data_field() -> None:
    """Test _make_error with data field."""
    error = agent_server._make_error(-32600, "Test error", {"detail": "info"})
    assert error["code"] == -32600
    assert error["message"] == "Test error"
    assert error["data"] == {"detail": "info"}


def test_make_response_with_result() -> None:
    """Test _make_response with result."""
    response = agent_server._make_response(1, result={"status": "ok"})
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"] == {"status": "ok"}
    assert "error" not in response


def test_make_response_with_error() -> None:
    """Test _make_response with error."""
    error = {"code": -32600, "message": "Invalid request"}
    response = agent_server._make_response(1, error=error)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["error"] == error
    assert "result" not in response


# =============================================================================
# AgentServer Lifecycle
# =============================================================================


@pytest.mark.asyncio
async def test_server_start_stop(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test server starts and stops cleanly."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    # Start server
    await server.start()
    assert socket_path.exists()

    # Stop server
    await server.stop()
    assert not socket_path.exists()


@pytest.mark.asyncio
async def test_server_socket_permissions(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test socket has restrictive permissions."""
    server = agent_server.AgentServer(mock_engine, socket_path)
    await server.start()

    # Check permissions (0o600 = owner read/write only)
    mode = socket_path.stat().st_mode & 0o777
    assert mode == 0o600

    await server.stop()


@pytest.mark.asyncio
async def test_connected_property(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test connected property."""
    server = agent_server.AgentServer(mock_engine, socket_path)
    await server.start()

    assert not server.connected
    assert server.connected_count == 0

    await server.stop()


# =============================================================================
# JSON-RPC Dispatch
# =============================================================================


@pytest.mark.asyncio
async def test_dispatch_method_not_found(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test dispatch returns error for unknown method."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "unknown"}
    response = await server._dispatch(request)

    assert response is not None
    assert response["error"]["code"] == agent_server._METHOD_NOT_FOUND
    assert "unknown" in response["error"]["message"]


@pytest.mark.asyncio
async def test_dispatch_invalid_params(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test dispatch returns error for non-object params."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "status", "params": ["array"]}
    response = await server._dispatch(request)

    assert response is not None
    assert response["error"]["code"] == agent_server._INVALID_PARAMS


@pytest.mark.asyncio
async def test_dispatch_notification_no_response(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test notifications (no id) don't return a response."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "method": "status"}  # No id = notification
    response = await server._dispatch(request)

    assert response is None


@pytest.mark.asyncio
async def test_dispatch_status(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test status method dispatch."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "status"}
    response = await server._dispatch(request)

    assert response is not None
    assert response["result"]["state"] == "idle"
    mock_engine.get_agent_status.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_cancel(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test cancel method dispatch."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "cancel"}
    response = await server._dispatch(request)

    assert response is not None
    assert response["result"]["cancelled"] is False
    mock_engine.request_agent_cancel.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_stages_returns_registered_stages(
    mock_engine: MagicMock, socket_path: Path
) -> None:
    """Test stages method returns all registered stages."""

    # Register a test stage (clean_registry autouse fixture handles cleanup)
    register_test_stage(
        _helper_stages_noop,
        name="test_stage",
    )

    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "stages"}
    response = await server._dispatch(request)

    assert response is not None
    assert "result" in response
    stages = response["result"]["stages"]
    assert any(s["name"] == "test_stage" for s in stages)
    test_stage = next(s for s in stages if s["name"] == "test_stage")
    # Registry normalizes paths to absolute
    assert len(test_stage["deps"]) == 1
    assert test_stage["deps"][0].endswith("input.csv")
    assert len(test_stage["outs"]) == 1
    assert test_stage["outs"][0].endswith("output.csv")


@pytest.mark.asyncio
async def test_dispatch_run_queues_execution(mock_engine: MagicMock, socket_path: Path) -> None:
    """Test run() queues execution request to engine via try_start_agent_run."""

    # Register a test stage (clean_registry autouse fixture handles cleanup)
    register_test_stage(_helper_run_noop, name="run_test")

    # Mock try_start_agent_run to return success
    mock_engine.try_start_agent_run.return_value = AgentRunStartResult(
        run_id="test123",
        status="started",
        stages_queued=["run_test"],
    )

    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"stages": ["run_test"]}}
    response = await server._dispatch(request)

    assert response is not None
    assert "result" in response
    assert response["result"]["status"] == "started"
    assert response["result"]["run_id"] == "test123"
    assert response["result"]["stages_queued"] == ["run_test"]
    mock_engine.try_start_agent_run.assert_called_once()
    # Verify the call args - run_id is generated, stages passed, force=False
    call_args = mock_engine.try_start_agent_run.call_args
    assert call_args[0][1] == ["run_test"]  # stages
    assert call_args[0][2] is False  # force


@pytest.mark.asyncio
async def test_dispatch_run_with_invalid_stage_returns_error(
    mock_engine: MagicMock, socket_path: Path
) -> None:
    """Test run() with non-existent stage returns STAGE_NOT_FOUND error."""
    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"stages": ["nonexistent"]}}
    response = await server._dispatch(request)

    assert response is not None
    assert "error" in response
    assert response["error"]["code"] == agent_server._STAGE_NOT_FOUND
    assert "nonexistent" in response["error"]["message"]


@pytest.mark.asyncio
async def test_dispatch_run_while_running_returns_error(
    mock_engine: MagicMock, socket_path: Path
) -> None:
    """Test run() while execution in progress returns error."""
    # Mock try_start_agent_run to return rejection (engine is already running)
    mock_engine.try_start_agent_run.return_value = AgentRunRejection(
        reason="not_ready",
        current_state="running",
    )

    server = agent_server.AgentServer(mock_engine, socket_path)

    request = {"jsonrpc": "2.0", "id": 1, "method": "run"}
    response = await server._dispatch(request)

    assert response is not None
    assert "error" in response
    assert response["error"]["code"] == agent_server._EXECUTION_IN_PROGRESS


# =============================================================================
# Exception Classes
# =============================================================================


def test_stage_not_found_with_suggestions() -> None:
    """Test exception stores stage and suggestions."""
    error = agent_server._StageNotFoundError("trnasform", ["transform", "translate"])
    assert error.stage == "trnasform"
    assert error.suggestions == ["transform", "translate"]
    assert "trnasform" in str(error)
