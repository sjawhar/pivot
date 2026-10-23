"""Tests for TUI watch mode functionality."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from conftest import send_rpc, wait_for_engine_ready

if TYPE_CHECKING:
    from collections.abc import Generator

_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


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
            [_PIVOT_BIN, "repro", "--watch", "--tui", "--force"],
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


def test_tui_watch_mode_runs_in_a_project_whose_socket_path_exceeds_the_unix_socket_limit(
    tmp_path: pathlib.Path,
) -> None:
    """The TUI reaches its engine through ``.pivot/agent.sock`` in a project at any depth.

    A Unix socket address holds at most 104 bytes on macOS and 108 on Linux; this project's
    socket path is longer than both. The run's statuses reach the TUI only through its RPC
    client, and the TUI writes each message it receives to ``--tui-log``.
    """
    project = tmp_path / ("deep-project-" * 10)
    (project / ".pivot").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "pipeline.py").write_text("""\
import pathlib
from pivot.compose import Pipeline, stage

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)

@stage
def hello() -> None:
    print("Hello!")

with pipeline:
    hello()
""")
    assert len(os.fsencode(project / ".pivot" / "agent.sock")) > 108
    tui_log = tmp_path / "tui.jsonl"
    stdout_path = tmp_path / "tui.stdout"
    stderr_path = tmp_path / "tui.stderr"
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    env["TERM"] = "xterm-256color"

    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            [_PIVOT_BIN, "repro", "--watch", "--tui", "--force", "--tui-log", str(tui_log)],
            cwd=project,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            stdin=subprocess.PIPE,
        )
    try:
        statuses = list[str]()
        deadline = time.monotonic() + 30
        while "ran" not in statuses and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
            if tui_log.exists():
                messages = [json.loads(line) for line in tui_log.read_text().splitlines()]
                statuses = [m["status"] for m in messages if m["type"] == "status"]

        assert "ran" in statuses, (
            f"the TUI received no completed run (statuses {statuses}, exit code {proc.poll()}).\n"
            + f"stderr: {stderr_path.read_text(errors='replace')[-2000:]}"
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
