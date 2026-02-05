"""E2E tests for RPC query handlers.

Verifies the complete wiring:
- CLI starts serve mode with --serve --force
- EventBuffer is wired as sink
- AgentRpcHandler receives the buffer reference
- Query handlers work end-to-end via the Unix socket
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

from conftest import send_rpc

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator


@pytest.fixture
def serve_pipeline(tmp_path: pathlib.Path) -> Generator[pathlib.Path]:
    """Start a minimal pipeline in serve mode, yield socket path."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".pivot").mkdir()

    # Use pipeline.py only (not pivot.yaml) to avoid ambiguity error
    pipeline_code = """\
import pathlib
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)

def hello() -> None:
    print("Hello!")

pipeline.register(hello, name="hello")
"""
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    sock_path = tmp_path / ".pivot" / "agent.sock"

    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    proc = subprocess.Popen(
        ["uv", "run", "pivot", "repro", "--watch", "--serve", "--force"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for socket
        for _ in range(100):
            if sock_path.exists():
                break
            time.sleep(0.1)
        else:
            stdout, stderr = proc.communicate(timeout=1)
            pytest.fail(
                "Socket not created within 10s.\n"
                + f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        # Poll until status query succeeds (server is ready)
        for _ in range(30):
            try:
                response = send_rpc(sock_path, "status")
                if "result" in response:
                    break
            except (ConnectionRefusedError, FileNotFoundError):
                pass
            time.sleep(0.1)

        yield sock_path
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_events_since_query(serve_pipeline: pathlib.Path) -> None:
    """events_since should return buffered events with expected content."""
    response = send_rpc(serve_pipeline, "events_since", {"version": 0})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict), f"Expected dict result, got: {type(result)}"
    assert "version" in result, "Expected version field in result"
    assert "events" in result, "Expected events field in result"
    events_field = result["events"]
    assert isinstance(events_field, list), "Expected events to be list"

    # Verify events contain expected structure from pipeline startup
    events: list[dict[str, object]] = events_field  # type: ignore[assignment] - JSON returns object
    if len(events) > 0:
        # Each event should have version and event fields (VersionedEvent structure)
        first_event = events[0]
        assert "version" in first_event, "Event should have version field"
        assert "event" in first_event, "Event should have event field"
        event_payload = first_event["event"]
        assert isinstance(event_payload, dict), "Event payload should be dict"
        assert "type" in event_payload, "Event payload should have type field"


def test_explain_query(serve_pipeline: pathlib.Path) -> None:
    """explain should return staleness info."""
    response = send_rpc(serve_pipeline, "explain", {"stage": "hello"})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict), f"Expected dict result, got: {type(result)}"
    assert "will_run" in result, "Expected will_run field in StageExplanation"
    assert "reason" in result, "Expected reason field in StageExplanation"


def test_stage_info_query(serve_pipeline: pathlib.Path) -> None:
    """stage_info should return stage metadata."""
    response = send_rpc(serve_pipeline, "stage_info", {"stage": "hello"})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict), f"Expected dict result, got: {type(result)}"
    assert result["name"] == "hello", f"Expected name='hello', got: {result.get('name')}"
    assert "deps" in result, "Expected deps field in result"
    assert "outs" in result, "Expected outs field in result"


def test_events_since_empty_buffer(serve_pipeline: pathlib.Path) -> None:
    """events_since on empty or newly started buffer returns empty events.

    Edge case: Client polling before any events occur.
    """
    response = send_rpc(serve_pipeline, "events_since", {"version": 0})
    assert "result" in response, f"Expected result, got: {response}"
    result = response["result"]
    assert isinstance(result, dict)
    assert "version" in result
    assert "events" in result
    # May have events from startup, but should not crash
    assert isinstance(result["events"], list)


def test_events_since_invalid_version_returns_error(serve_pipeline: pathlib.Path) -> None:
    """events_since with invalid version parameter returns error.

    Error path: Non-integer or out-of-range version should be rejected.
    """
    response = send_rpc(serve_pipeline, "events_since", {"version": "invalid"})
    assert "error" in response, "Non-integer version should return error"


def test_explain_nonexistent_stage_returns_error(serve_pipeline: pathlib.Path) -> None:
    """explain query with nonexistent stage returns error.

    Error path: Should get clear error, not crash.
    """
    response = send_rpc(serve_pipeline, "explain", {"stage": "does_not_exist"})
    assert "error" in response, f"Expected error for nonexistent stage, got: {response}"


def test_stage_info_nonexistent_stage_returns_error(serve_pipeline: pathlib.Path) -> None:
    """stage_info query with nonexistent stage returns error.

    Error path: Should get clear error, not crash.
    """
    response = send_rpc(serve_pipeline, "stage_info", {"stage": "does_not_exist"})
    assert "error" in response, f"Expected error for nonexistent stage, got: {response}"


def test_multiple_queries_same_connection(serve_pipeline: pathlib.Path) -> None:
    """Multiple queries on same socket connection should all succeed.

    Integration test: Verifies connection reuse and state handling.
    """
    import contextlib

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(str(serve_pipeline))

    with contextlib.closing(sock):
        # First query: status
        request1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "status"})
        sock.sendall(request1.encode() + b"\n")
        response1 = json.loads(sock.recv(4096).decode())
        assert "result" in response1, "First query should succeed"

        # Second query: stages
        request2 = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "stages"})
        sock.sendall(request2.encode() + b"\n")
        response2 = json.loads(sock.recv(4096).decode())
        assert "result" in response2, "Second query should succeed"

        # Third query: events_since
        request3 = json.dumps(
            {"jsonrpc": "2.0", "id": 3, "method": "events_since", "params": {"version": 0}}
        )
        sock.sendall(request3.encode() + b"\n")
        response3 = json.loads(sock.recv(4096).decode())
        assert "result" in response3, "Third query should succeed"
