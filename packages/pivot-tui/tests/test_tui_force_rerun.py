"""Integration test for TUI force re-run functionality.

Verifies the complete flow: RPC socket creation, sending a force re-run command,
and command acceptance. Uses a simple noop stage (like other integration tests)
to avoid watch path issues with non-existent output files.
"""

from __future__ import annotations

import os
import signal
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
    """Start a minimal pipeline in serve mode, yield socket path.

    Uses a simple noop stage (no outputs) to avoid watchfiles issues with
    non-existent artifact paths. The force re-run functionality is tested
    by verifying the RPC command is accepted and events are generated.
    """
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".pivot").mkdir()

    # Use pipeline.py with a simple stage (no outputs)
    pipeline_code = """\
import pathlib
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)

def my_stage() -> None:
    print("my_stage executed")

pipeline.register(my_stage, name="my_stage")

def other_stage() -> None:
    print("other_stage executed")

pipeline.register(other_stage, name="other_stage")
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


def test_force_rerun_single_stage_accepted(serve_pipeline: pathlib.Path) -> None:
    """Force re-run command for single stage is accepted by the engine.

    Verifies that the RPC "run" method with force=True and a specific stage
    returns "accepted", indicating the command was received and queued.
    """
    response = send_rpc(serve_pipeline, "run", {"stages": ["my_stage"], "force": True})
    assert "result" in response, f"Expected result, got: {response}"
    assert response["result"] == "accepted", f"Expected 'accepted', got: {response['result']}"


def test_force_rerun_all_stages_accepted(serve_pipeline: pathlib.Path) -> None:
    """Force re-run command for all stages (stages=None) is accepted.

    Verifies that sending stages=None with force=True is accepted,
    which triggers all stages to re-run.
    """
    response = send_rpc(serve_pipeline, "run", {"stages": None, "force": True})
    assert "result" in response, f"Expected result, got: {response}"
    assert response["result"] == "accepted", f"Expected 'accepted', got: {response['result']}"


def test_force_rerun_multiple_stages_accepted(serve_pipeline: pathlib.Path) -> None:
    """Force re-run command for multiple specific stages is accepted.

    Verifies that multiple stage names can be passed in the stages list.
    """
    response = send_rpc(
        serve_pipeline, "run", {"stages": ["my_stage", "other_stage"], "force": True}
    )
    assert "result" in response, f"Expected result, got: {response}"
    assert response["result"] == "accepted"


def test_force_rerun_without_force_flag(serve_pipeline: pathlib.Path) -> None:
    """Run command without force flag is also accepted (for completeness).

    Verifies that the run command works with force=False as well.
    """
    response = send_rpc(serve_pipeline, "run", {"stages": ["my_stage"], "force": False})
    assert "result" in response, f"Expected result, got: {response}"
    assert response["result"] == "accepted"


def test_force_rerun_nonexistent_stage_returns_error(serve_pipeline: pathlib.Path) -> None:
    """Force re-run command for nonexistent stage returns error.

    Verifies that the server validates stage names and returns an appropriate
    error for stages that don't exist.
    """
    response = send_rpc(serve_pipeline, "run", {"stages": ["nonexistent_stage"], "force": True})
    assert "error" in response, f"Expected error for nonexistent stage, got: {response}"


def test_force_rerun_triggers_events(serve_pipeline: pathlib.Path) -> None:
    """Force re-run should generate events that can be observed via events_since.

    Verifies the end-to-end flow: send force re-run, then verify events are
    generated (indicating the engine processed the command).
    """
    # Get initial event version
    initial_response = send_rpc(serve_pipeline, "events_since", {"version": 0})
    assert "result" in initial_response
    initial_result = initial_response["result"]
    assert isinstance(initial_result, dict), f"Expected dict result, got: {type(initial_result)}"
    assert "version" in initial_result, "Expected version field"
    initial_version = initial_result["version"]

    # Send force re-run command
    run_response = send_rpc(serve_pipeline, "run", {"stages": ["my_stage"], "force": True})
    assert run_response.get("result") == "accepted"

    # Wait a bit for the engine to process
    time.sleep(0.5)

    # Check for new events
    events_response = send_rpc(serve_pipeline, "events_since", {"version": initial_version})
    assert "result" in events_response, f"Expected events result, got: {events_response}"
    events_result = events_response["result"]
    assert isinstance(events_result, dict), f"Expected dict result, got: {type(events_result)}"
    assert "version" in events_result, "Expected version field"

    # The version should have advanced (indicating new events were generated)
    new_version = events_result["version"]
    assert new_version > initial_version, "Events should have been generated"
