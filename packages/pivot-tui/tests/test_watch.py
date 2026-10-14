"""Tests for TUI watch mode functionality."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

from conftest import send_rpc, wait_for_engine_ready

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator


@pytest.fixture
def tui_watch_pipeline(tmp_path: pathlib.Path) -> Generator[pathlib.Path]:
    """Start a minimal pipeline in TUI watch mode (without --serve), yield socket path."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".pivot").mkdir()

    # Use pipeline.py only (not pivot.yaml) to avoid ambiguity error
    pipeline_code = """\
import pathlib
from pivot.compose import Pipeline as ComposePipeline, stage

_cp = ComposePipeline("test", root=pathlib.Path(__file__).parent)

@stage
def hello() -> None:
    print("Hello!")

with _cp:
    hello()

pipeline = _cp
"""
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    sock_path = tmp_path / ".pivot" / "agent.sock"
    stdout_path = tmp_path / "tui.stdout"
    stderr_path = tmp_path / "tui.stderr"

    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    # TERM needed for TUI mode
    env["TERM"] = "xterm-256color"

    # Start TUI watch mode WITHOUT --serve - the socket should still be created.
    # Output goes to files, not pipes: nothing reads a pipe until teardown, so the
    # TUI's screen output and the server's per-connection logs would fill it and block.
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            ["uv", "run", "pivot", "repro", "--watch", "--tui", "--force"],
            cwd=tmp_path,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            stdin=subprocess.PIPE,  # TUI needs stdin
        )

    try:
        # Wait for socket to be created
        for _ in range(100):
            if sock_path.exists():
                break
            time.sleep(0.1)
        else:
            pytest.fail(
                "Socket not created within 10s.\n"
                + f"stdout: {stdout_path.read_text(errors='replace')}\n"
                + f"stderr: {stderr_path.read_text(errors='replace')}"
            )

        wait_for_engine_ready(sock_path)

        yield sock_path
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_tui_watch_mode_creates_rpc_socket(tui_watch_pipeline: pathlib.Path) -> None:
    """TUI watch mode should create agent.sock even without --serve.

    This enables the TUI to send force re-run commands via the RPC socket.
    """
    sock_path = tui_watch_pipeline

    # Verify the socket exists and is accessible
    assert sock_path.exists(), "Socket should exist"

    # Verify we can send RPC commands
    response = send_rpc(sock_path, "status")
    assert "result" in response, f"Expected result in response: {response}"


def test_tui_watch_mode_accepts_run_command(tui_watch_pipeline: pathlib.Path) -> None:
    """TUI watch mode RPC socket should accept run commands.

    This verifies the socket is fully functional for force re-run use cases.
    """
    sock_path = tui_watch_pipeline

    # Send a force run command for a specific stage
    response = send_rpc(sock_path, "run", {"stages": ["test/hello"], "force": True})
    assert response.get("result") == "accepted", f"Expected 'accepted': {response}"
