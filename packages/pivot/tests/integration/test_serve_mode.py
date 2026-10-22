"""Integration tests for serve mode (headless daemon via Engine).

Tests E2E flow: socket creation, status queries, event broadcast.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING

import anyio
import pytest

from pivot import config
from pivot.engine.agent_rpc import AgentRpcHandler, AgentRpcSource, BroadcastEventSink
from pivot.engine.engine import Engine
from pivot.engine.sources import OneShotSource
from pivot.engine.types import StageStarted
from tests import helpers
from tests.conftest import send_rpc

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

    from pivot.registry import PipelineLike as Pipeline

_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


def _helper_noop() -> dict[str, str]:
    """No-op stage for testing."""
    return {"result": "ok"}


# =============================================================================
# Socket Integration Tests
# =============================================================================


@pytest.fixture
def socket_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a path for the Unix socket."""
    return tmp_path / "agent.sock"


@pytest.mark.anyio
async def test_socket_creation_and_permissions(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource creates socket with correct permissions."""

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    helpers.register_test_stage(_helper_noop, name="test_stage")

    rpc_source = AgentRpcSource(socket_path=socket_path)

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)

            # Wait for socket to be created
            await helpers.wait_for_socket(socket_path)

            # Check permissions (owner-only)
            mode = socket_path.stat().st_mode
            assert mode & 0o777 == 0o600, f"Expected 0o600, got {oct(mode & 0o777)}"

            tg.cancel_scope.cancel()


# =============================================================================
# EventSink Integration Tests
# =============================================================================


@pytest.mark.anyio
async def test_event_sink_broadcasts_to_multiple_subscribers() -> None:
    """EventSink broadcasts events to all subscribers."""
    sink = BroadcastEventSink()

    recv1 = await sink.subscribe("client1")
    recv2 = await sink.subscribe("client2")

    # Emit an event
    event: StageStarted = StageStarted(
        type="stage_started",
        seq=0,
        stage="test",
        index=0,
        total=1,
    )
    await sink.handle(event)

    # Both clients should receive it
    event1 = recv1.receive_nowait()
    event2 = recv2.receive_nowait()

    # Type narrow using discriminated union on "type" field
    assert event1["type"] == "stage_started" and event1["stage"] == "test"
    assert event2["type"] == "stage_started" and event2["stage"] == "test"

    await sink.close()


@pytest.mark.anyio
async def test_unsubscribe_stops_event_delivery() -> None:
    """Unsubscribed clients stop receiving events."""
    sink = BroadcastEventSink()

    recv = await sink.subscribe("client")

    # Unsubscribe
    await sink.unsubscribe("client")

    # Emit an event
    event: StageStarted = StageStarted(
        type="stage_started",
        seq=0,
        stage="test",
        index=0,
        total=1,
    )
    await sink.handle(event)

    # Channel should be closed - raises EndOfStream on closed channels
    with pytest.raises(anyio.EndOfStream):
        recv.receive_nowait()


# =============================================================================
# AgentRpcHandler Integration Tests
# =============================================================================


@pytest.mark.anyio
async def test_agent_rpc_handler_status_query(test_pipeline: Pipeline) -> None:
    """AgentRpcHandler.handle_query returns status correctly."""
    async with Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine)

        result = await handler.handle_query("status")
        # Type narrow by checking for "state" key (discriminates QueryStatusResult from other types)
        assert "state" in result and result["state"] == "idle"
        assert "running" in result and result["running"] == []
        assert "pending" in result and result["pending"] == []


@pytest.mark.anyio
async def test_agent_rpc_handler_stages_query(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcHandler.handle_query returns stages correctly."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    helpers.register_test_stage(_helper_noop, name="stage_a")
    helpers.register_test_stage(_helper_noop, name="stage_b")

    async with Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine)

        result = await handler.handle_query("stages")

        assert "stages" in result
        assert set(result["stages"]) >= {"test/stage_a", "test/stage_b"}


@pytest.mark.anyio
async def test_agent_rpc_handler_unknown_method() -> None:
    """AgentRpcHandler.handle_query raises ValueError for unknown methods."""
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        with pytest.raises(ValueError, match="Unknown query method"):
            await handler.handle_query("unknown_method")


@pytest.mark.anyio
async def test_agent_rpc_handler_stages_query_no_pipeline() -> None:
    """AgentRpcHandler.handle_query returns empty list when no pipeline."""
    async with Engine() as engine:
        handler = AgentRpcHandler(engine=engine)

        result = await handler.handle_query("stages")

        assert result == {"stages": []}


# =============================================================================
# Socket Buffer Handling Tests
# =============================================================================


@pytest.mark.anyio
async def test_socket_handles_chunked_messages(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource correctly reassembles messages split across chunks."""

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # Need to create engine first to create handler, then create source with handler
    async with Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine)
        rpc_source = AgentRpcSource(socket_path=socket_path, handler=handler)
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)

            # Wait for socket to be created
            await helpers.wait_for_socket(socket_path)

            # Connect and send a request in multiple chunks
            async with await anyio.connect_unix(str(socket_path)) as conn:
                # Split the JSON-RPC request into chunks
                request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
                chunk1 = request[:10].encode()  # '{"jsonrpc"'
                chunk2 = request[10:25].encode()  # ':"2.0","method"'
                chunk3 = request[25:].encode()  # ':"status","id":1}\n'

                await conn.send(chunk1)
                await anyio.sleep(0.05)  # Brief delay between chunks
                await conn.send(chunk2)
                await anyio.sleep(0.05)
                await conn.send(chunk3)

                # Read response
                response_data = await conn.receive(1024)
                response = json.loads(response_data.decode().strip())

                assert response["id"] == 1
                assert "result" in response
                assert response["result"]["state"] == "idle"

            tg.cancel_scope.cancel()


# =============================================================================
# Sink Error Handling Tests
# =============================================================================


class _FailingSink:
    """Test sink that raises an exception on handle()."""

    async def handle(self, event: object) -> None:
        raise RuntimeError("Sink failure for testing")

    async def close(self) -> None:
        pass


class _CollectingSink:
    """Test sink that collects events for verification."""

    events: list[object]

    def __init__(self) -> None:
        self.events = list[object]()

    async def handle(self, event: object) -> None:
        self.events.append(event)

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_sink_error_does_not_crash_engine(test_pipeline: Pipeline) -> None:
    """Errors in sink.handle() are logged but don't crash the engine."""

    collecting_sink = _CollectingSink()

    async with Engine(pipeline=test_pipeline) as engine:
        # Add a failing sink and a collecting sink
        engine.add_sink(_FailingSink())
        engine.add_sink(collecting_sink)

        # Add source to trigger a run
        source = OneShotSource(
            stages=None,
            force=True,
            reason="test",
        )
        engine.add_source(source)

        # Run should complete despite the failing sink
        await engine.run(exit_on_completion=True)

    # The collecting sink should have received events despite the failing sink
    assert len(collecting_sink.events) > 0
    event_types = [e.get("type") for e in collecting_sink.events if isinstance(e, dict)]
    assert "engine_state_changed" in event_types


# =============================================================================
# CLI Serve Mode E2E Tests
# =============================================================================

_NOOP_PIPELINE = """\
import pathlib

from pivot.compose import Pipeline, stage


pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


@stage
def noop() -> None:
    pass


with pipeline:
    noop()
"""

# Runs serve in-process through pivot's embedding entry points; its docstring gives the modes.
_INPROC_SERVE = pathlib.Path(__file__).with_name("inproc_serve.py")


def _helper_project(project: pathlib.Path) -> None:
    """Create a project at ``project`` whose pipeline has one stage that does nothing."""
    (project / ".pivot").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "pipeline.py").write_text(_NOOP_PIPELINE)


def _helper_wait_for_status(
    proc: subprocess.Popen[bytes], socket_path: pathlib.Path, deadline: float
) -> dict[str, object] | None:
    """Return serve's answer to ``status``, or None once ``proc`` exits or ``deadline`` passes.

    Binding creates the socket file before the listener accepts connections, so this retries
    the RPC rather than treating the socket's existence as readiness.
    """
    response: dict[str, object] | None = None
    while response is None and proc.poll() is None and time.monotonic() < deadline:
        try:
            response = send_rpc(socket_path, "status")
        except (ConnectionRefusedError, FileNotFoundError):
            time.sleep(0.1)
    return response


def test_serve_mode_cli_responds_to_status_query(
    tmp_path: pathlib.Path,
) -> None:
    """E2E test: pivot repro --watch --serve creates working RPC endpoint.

    This test verifies the complete CLI integration - that _run_serve_mode
    correctly wires up AgentRpcHandler and EventSink. Without this test,
    we caught bugs where components worked individually but weren't connected.
    """

    # Create minimal project structure with a valid pipeline
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    pipeline_code = """\
import pathlib

from pivot.compose import Pipeline, stage


pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


@stage
def noop_stage() -> None:
    pass


with pipeline:
    noop_stage()
"""
    (tmp_path / "pipeline.py").write_text(pipeline_code)

    socket_path = tmp_path / ".pivot" / "agent.sock"

    # Start serve mode as subprocess
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    proc = subprocess.Popen(
        [_PIVOT_BIN, "repro", "--watch", "--serve"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        response = _helper_wait_for_status(proc, socket_path, time.monotonic() + 10)
        if response is None:
            stdout, stderr = proc.communicate(timeout=1)
            pytest.fail(
                "Serve mode did not answer status within 10s.\n"
                + f"exit: {proc.poll()}\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )
        # Should get valid status response, NOT "Method not found" error
        assert "error" not in response, f"Expected status result, got error: {response}"
        assert "result" in response
        assert isinstance(response["result"], dict)
        assert response["result"]["state"] in ("idle", "active")

        # Also test stages query
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(str(socket_path))
            request = '{"jsonrpc":"2.0","method":"stages","id":2}\n'
            sock.sendall(request.encode())
            response_data = sock.recv(1024)
            stages_response = json.loads(response_data.decode().strip())

        assert isinstance(stages_response, dict)
        assert "error" not in stages_response, (
            f"Expected stages result, got error: {stages_response}"
        )
        assert "result" in stages_response
        assert isinstance(stages_response["result"], dict)
        assert "stages" in stages_response["result"]

    finally:
        # Clean up subprocess
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_serve_mode_serves_a_project_whose_socket_path_exceeds_the_unix_socket_limit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project at any depth can serve: a Unix socket address holds at most 104 bytes on
    macOS and 108 on Linux, and this project's ``.pivot/agent.sock`` is longer than both.

    An agent in the project connects to ``.pivot/agent.sock`` by its relative path.
    """
    project = tmp_path / ("deep-project-" * 10)
    _helper_project(project)
    socket_path = project / ".pivot" / "agent.sock"
    assert len(os.fsencode(socket_path)) > 108, "the socket path must exceed every platform limit"
    stdout_path = tmp_path / "serve.stdout"
    stderr_path = tmp_path / "serve.stderr"
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            [_PIVOT_BIN, "repro", "--watch", "--serve"],
            cwd=project,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )
    monkeypatch.chdir(project)
    try:
        response = _helper_wait_for_status(
            proc, pathlib.Path(".pivot", "agent.sock"), time.monotonic() + 30
        )
        assert response is not None, (
            f"serve mode did not answer on .pivot/agent.sock (exit code {proc.poll()}).\n"
            + f"stdout: {stdout_path.read_text(errors='replace')}\n"
            + f"stderr: {stderr_path.read_text(errors='replace')}"
        )
        assert "result" in response, f"Expected status result, got: {response}"
        assert socket_path.is_socket(), "the socket lives in the project's .pivot directory"
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _helper_deliver(proc: subprocess.Popen[bytes], to_group: bool, signum: int) -> None:
    """Send ``signum`` to the process, or to its whole process group as a service manager does."""
    if to_group:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signum)
    else:
        proc.send_signal(signum)


@pytest.mark.parametrize(
    ("stops", "inherited_sigint"),
    [
        pytest.param([(False, signal.SIGTERM, 0.0)], signal.default_int_handler, id="sigterm"),
        pytest.param([(False, signal.SIGINT, 0.0)], signal.default_int_handler, id="sigint"),
        pytest.param(
            [(False, signal.SIGINT, 0.0), (False, signal.SIGINT, 0.05)],
            signal.default_int_handler,
            id="sigint-twice",
        ),
        *(
            pytest.param(
                [(False, signal.SIGTERM, 0.0), (False, signal.SIGTERM, delay)],
                signal.default_int_handler,
                id=f"sigterm-twice-{int(delay * 1000)}ms-apart",
            )
            for delay in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3)
        ),
        pytest.param(
            [(False, signal.SIGTERM, 0.0), (False, signal.SIGTERM, 0.3)],
            signal.SIG_IGN,
            id="sigterm-twice-300ms-apart-sigint-ignored",
        ),
        pytest.param(
            [(False, signal.SIGTERM, 0.0), (True, signal.SIGTERM, 0.1)],
            signal.default_int_handler,
            id="sigterm-then-group-as-timeout-does",
        ),
        pytest.param(
            [(True, signal.SIGTERM, 0.0)],
            signal.default_int_handler,
            id="group-sigterm-as-systemctl-stop-does",
        ),
        pytest.param(
            [(False, signal.SIGTERM, 0.0), (False, signal.SIGINT, 0.1)],
            signal.default_int_handler,
            id="sigterm-then-sigint",
        ),
        pytest.param([(False, signal.SIGTERM, 0.0)], signal.SIG_IGN, id="sigterm-sigint-ignored"),
    ],
)
def test_serve_mode_stopped_by_signals_exits_cleanly_and_leaves_nothing_behind(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stops: list[tuple[bool, int, float]],
    inherited_sigint: Callable[[int, types.FrameType | None], object] | signal.Handlers,
) -> None:
    """A service manager stops a server with SIGTERM, often more than once (GNU timeout signals
    the process and then its group) and often to the whole process group (systemd). Serve mode
    shuts down as Ctrl-C does whatever arrives: exit 0, no task failing during shutdown, and
    neither its socket nor multiprocessing's manager directory left behind."""
    project = tmp_path / "project"
    _helper_project(project)
    socket_path = project / ".pivot" / "agent.sock"
    # Short, because multiprocessing binds its manager's socket under TMPDIR.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    stderr_path = tmp_path / "serve.stderr"
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    try:
        # The child inherits this process's SIGINT disposition: ignored stays ignored, handled
        # resets to the default.
        previous_sigint = signal.signal(signal.SIGINT, inherited_sigint)
        try:
            with stderr_path.open("wb") as stderr_file:
                proc = subprocess.Popen(
                    [_PIVOT_BIN, "repro", "--watch", "--serve"],
                    cwd=project,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    start_new_session=True,
                )
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        monkeypatch.chdir(project)
        try:
            deadline = time.monotonic() + 30
            response = _helper_wait_for_status(proc, pathlib.Path(".pivot", "agent.sock"), deadline)
            assert response is not None, f"serve mode did not answer (exit {proc.poll()})"
            while not list(private_tmp.glob("pymp-*")) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert list(private_tmp.glob("pymp-*")), "the worker pool's manager never started"
            for to_group, signum, delay in stops:
                time.sleep(delay)
                _helper_deliver(proc, to_group, signum)
            exit_code = proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

        stderr = stderr_path.read_text(errors="replace")
        assert exit_code == 0, f"serve mode exited {exit_code} on {stops}:\n{stderr}"
        assert "unhandled exception during asyncio.run() shutdown" not in stderr, (
            f"a task failed while serve mode shut down on {stops}:\n{stderr}"
        )
        assert "KeyboardInterrupt" not in stderr, (
            f"a stop signal escaped as an exception:\n{stderr}"
        )
        assert not socket_path.exists(), "serve mode left .pivot/agent.sock behind"
        # The manager removes its directory as it exits, just after pivot does.
        deadline = time.monotonic() + 5
        while list(private_tmp.iterdir()) and time.monotonic() < deadline:
            time.sleep(0.1)
        left = sorted(str(p.relative_to(private_tmp)) for p in private_tmp.rglob("*"))
        assert not left, f"serve mode left {left} in TMPDIR"
    finally:
        shutil.rmtree(private_tmp)


@pytest.mark.parametrize("signame", ["TERM", "INT"])
def test_inprocess_serve_restores_stop_signals_for_children(
    tmp_path: pathlib.Path, signame: str
) -> None:
    """An application that invokes pivot's CLI in-process can still stop a child it starts
    afterward. The standalone pivot command exits immediately, but CLI embedders do not."""
    project = tmp_path / "project"
    _helper_project(project)
    result = subprocess.run(
        [sys.executable, str(_INPROC_SERVE), str(project), "click", signame],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert "serve returned" in result.stdout, result.stdout
    assert "after: SIGINT=default_int_handler SIGTERM=0" in result.stdout, result.stdout
    assert "SIGTERM stopped it" in result.stdout, result.stdout


@pytest.mark.parametrize("signame", ["TERM", "INT"])
def test_public_main_restores_stop_signals_for_children(
    tmp_path: pathlib.Path, signame: str
) -> None:
    """The public `pivot.cli.main()` function is also an embedding API, not only the console
    script's entry point. It restores its caller's stop handlers after serve returns."""
    project = tmp_path / "project"
    _helper_project(project)
    result = subprocess.run(
        [sys.executable, str(_INPROC_SERVE), str(project), "main", signame],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert "after: SIGINT=default_int_handler SIGTERM=0" in result.stdout, result.stdout
    assert "SIGTERM stopped it" in result.stdout, result.stdout


# A library that installs a SIGTERM handler before Python starts, as a C application that
# embeds Python can, and then drops itself from the environment children inherit.
_HOST_HANDLER_C = r"""
#include <signal.h>
#include <stdlib.h>
#include <string.h>

static void host_handler(int sig) { (void)sig; }

__attribute__((constructor)) static void install_host_handler(void) {
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = host_handler;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    unsetenv("LD_PRELOAD");
}
"""


def _helper_host_handler_library(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build the host-handler library. Without a C compiler, skip locally and fail in CI."""
    compiler = shutil.which("cc")
    if compiler is None:
        message = "the host-handler library needs a C compiler, which is not installed"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    source = tmp_path / "host_handler.c"
    source.write_text(_HOST_HANDLER_C)
    library = tmp_path / "host_handler.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    return library


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses LD_PRELOAD and /proc")
@pytest.mark.parametrize("entry", ["main", "click"])
def test_inprocess_serve_leaves_a_stop_handler_installed_outside_python(
    tmp_path: pathlib.Path, entry: str
) -> None:
    """A host application can install SIGTERM's handler in C before Python starts. Python's
    getsignal() then reports None and cannot put that handler back, so serve leaves SIGTERM
    alone: it stops on Ctrl-C and returns as on main, and the host's handler and SIGINT's are
    in place afterwards."""
    library = _helper_host_handler_library(tmp_path)
    project = tmp_path / "project"
    _helper_project(project)
    result = subprocess.run(
        [sys.executable, str(_INPROC_SERVE), str(project), entry, "INT"],
        capture_output=True,
        text=True,
        timeout=40,
        env={**os.environ, "LD_PRELOAD": str(library)},
    )
    assert result.returncode == 0, result.stderr
    assert "kernel before: SIGINT=caught SIGTERM=caught" in result.stdout, (
        f"the host's SIGTERM handler is not installed:\n{result.stdout}"
    )
    assert "before: SIGINT=default_int_handler SIGTERM=None" in result.stdout, result.stdout
    outcome = "serve raised SystemExit(0)" if entry == "main" else "serve returned"
    assert outcome in result.stdout, f"{result.stdout}\n{result.stderr[-2000:]}"
    assert "after: SIGINT=default_int_handler SIGTERM=None" in result.stdout, result.stdout
    assert "kernel after: SIGINT=caught SIGTERM=caught" in result.stdout, result.stdout
    assert "SIGTERM stopped it" in result.stdout, result.stdout


@pytest.mark.parametrize("signame", ["TERM", "INT"])
def test_inprocess_serve_keeps_the_callers_signal_wakeup_fd(
    tmp_path: pathlib.Path, signame: str
) -> None:
    """An application that feeds signals to its own event loop sets a signal wakeup fd. Serve
    in-process uses its own while it runs, and the caller's is set again once serve returns."""
    project = tmp_path / "project"
    _helper_project(project)
    result = subprocess.run(
        [sys.executable, str(_INPROC_SERVE), str(project), "main", signame, "wakeup"],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    fds = dict(
        line.split(": ", 1) for line in result.stdout.splitlines() if line.startswith("wakeup fd")
    )
    assert "wakeup fd before" in fds, result.stdout
    assert fds.get("wakeup fd after") == fds["wakeup fd before"], result.stdout


# Serves as the console script does and resends SIGTERM at each disposition change after a stop.
_STOP_WINDOW = pathlib.Path(__file__).with_name("serve_stop_window.py")


def test_a_sigterm_during_serves_shutdown_changes_nothing(tmp_path: pathlib.Path) -> None:
    """After the first stop, a later SIGTERM is a no-op: SIGTERM's disposition goes from serve's
    handler straight to SIG_IGN and never passes through SIG_DFL, which would let that SIGTERM
    kill the process on whichever thread the kernel picks."""
    project = tmp_path / "project"
    _helper_project(project)
    # Short, because multiprocessing binds its manager's socket under TMPDIR.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    try:
        result = subprocess.run(
            [sys.executable, str(_STOP_WINDOW), str(project)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        resends = [line for line in result.stderr.splitlines() if line.startswith("probe:")]
        assert result.returncode == 0, f"serve exited {result.returncode} after {resends}"
        assert resends, f"the probe never sent SIGTERM again:\n{result.stderr[-2000:]}"
        assert not (project / ".pivot" / "agent.sock").exists(), "serve left its socket"
    finally:
        shutil.rmtree(private_tmp)


@pytest.mark.parametrize("entry", ["main", "click"])
def test_serve_from_a_worker_thread_answers_and_leaves_signal_handlers_alone(
    tmp_path: pathlib.Path, entry: str
) -> None:
    """Python delivers signals only to the main thread, so an embedder that runs serve on
    another thread gets it without stop-signal handling: it answers, and the process's SIGINT
    and SIGTERM handlers stay the embedder's. Stopping it is the embedder's job."""
    project = tmp_path / "project"
    _helper_project(project)
    # Short, because multiprocessing binds its manager's socket under TMPDIR. The probe exits
    # while serve still runs, so the manager's directory stays there until this removes it.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    try:
        result = subprocess.run(
            [sys.executable, str(_INPROC_SERVE), str(project), entry, "thread"],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        report = json.loads(lines[-1])
        assert report["answer"] and '"result"' in report["answer"], (
            f"serve on a worker thread did not answer status: {report}\n{result.stderr[-2000:]}"
        )
        assert lines[0].removeprefix("before: ") == lines[1].removeprefix("serving: "), (
            f"serve on a worker thread changed the process's signal handlers: {lines[:2]}"
        )
    finally:
        shutil.rmtree(private_tmp)


def test_group_sigterm_during_a_stage_is_not_logged_as_a_failure(tmp_path: pathlib.Path) -> None:
    """`systemctl stop` sends SIGTERM to the whole cgroup. A running worker receives that
    signal directly, but a deliberate service stop must not emit a stage-failure traceback."""
    project = tmp_path / "project"
    (project / ".pivot").mkdir(parents=True)
    (project / ".git").mkdir()
    marker = project / "started"
    (project / "pipeline.py").write_text(f"""\
import pathlib
import time

from pivot.compose import Pipeline, stage


pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)


@stage
def slow() -> None:
    pathlib.Path({str(marker)!r}).write_text("started")
    time.sleep(30)


with pipeline:
    slow()
""")
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    stderr_path = tmp_path / "serve.stderr"
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    try:
        with stderr_path.open("wb") as stderr_file:
            proc = subprocess.Popen(
                [_PIVOT_BIN, "repro", "--watch", "--serve", "--force"],
                cwd=project,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 30
            while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert marker.exists(), f"the stage did not start (exit {proc.poll()})"
            os.killpg(proc.pid, signal.SIGTERM)
            assert proc.wait(timeout=20) == 0
        finally:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        stderr = stderr_path.read_text(errors="replace")
        assert "TerminatedWorkerError" not in stderr, stderr
        assert "Stage test/slow failed with exception" not in stderr, stderr
    finally:
        shutil.rmtree(private_tmp)


@pytest.mark.parametrize("long_path", [False, True], ids=["short-path", "long-path"])
def test_stopping_first_live_serve_keeps_replacement_socket(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, long_path: bool
) -> None:
    """A newer serve takes the project's socket path as current startup behavior permits. When
    the older serve stops, it must leave the newer socket in place."""
    project = tmp_path / (("deep-project-" * 10) if long_path else "project")
    _helper_project(project)
    socket_path = project / ".pivot" / "agent.sock"
    if long_path:
        assert len(os.fsencode(socket_path)) > 108
    tmpdirs = [pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp")) for _ in range(2)]
    procs = list[subprocess.Popen[bytes]]()
    first_inode: int | None = None
    try:
        for index, private_tmp in enumerate(tmpdirs):
            stderr = (tmp_path / f"serve-{index}.stderr").open("wb")
            procs.append(
                subprocess.Popen(
                    [_PIVOT_BIN, "repro", "--watch", "--serve"],
                    cwd=project,
                    env={
                        **os.environ,
                        "TMPDIR": str(private_tmp),
                        "PIVOT_CACHE_DIR": str(tmp_path / f"cache-{index}"),
                    },
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
            )
            stderr.close()
            monkeypatch.chdir(project)
            deadline = time.monotonic() + 30
            response: dict[str, object] | None = None
            while response is None and procs[-1].poll() is None and time.monotonic() < deadline:
                if index and socket_path.exists() and socket_path.lstat().st_ino == first_inode:
                    time.sleep(0.1)
                    continue
                try:
                    response = send_rpc(pathlib.Path(".pivot", "agent.sock"), "status")
                except (ConnectionRefusedError, FileNotFoundError):
                    time.sleep(0.1)
            assert response is not None, f"serve {index} did not answer (exit {procs[-1].poll()})"
            if index == 0:
                first_inode = socket_path.lstat().st_ino

        replacement_inode = socket_path.lstat().st_ino
        procs[0].send_signal(signal.SIGTERM)
        assert procs[0].wait(timeout=15) == 0
        assert procs[1].poll() is None, "the replacement serve exited when the first stopped"
        assert socket_path.exists(), "the first serve removed the replacement socket"
        assert socket_path.lstat().st_ino == replacement_inode
        assert send_rpc(pathlib.Path(".pivot", "agent.sock"), "status")["result"]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for private_tmp in tmpdirs:
            shutil.rmtree(private_tmp)


# =============================================================================
# AgentRpcSource Handler Tests (Component-level)
# =============================================================================


@pytest.mark.anyio
async def test_rpc_source_without_handler_returns_method_not_found(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource without handler returns 'Method not found' for queries.

    Copilot Review Comment 1: This test demonstrates the BUG - when AgentRpcSource
    is created without a handler (as _run_serve_mode currently does), status
    queries fail with 'Method not found' instead of returning engine state.

    The fix: _run_serve_mode must create AgentRpcHandler and pass it to
    AgentRpcSource(socket_path=socket_path, handler=handler).
    """

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # Create source WITHOUT handler (mimics current _run_serve_mode bug)
    rpc_source = AgentRpcSource(socket_path=socket_path, handler=None)

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)
            await helpers.wait_for_socket(socket_path)

            # Connect and send status query
            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
                await conn.send(request.encode())
                response_data = await conn.receive(1024)
                response = json.loads(response_data.decode().strip())

                # BUG: Without handler, queries return "Method not found"
                assert "error" in response, "Expected error without handler"
                assert response["error"]["code"] == -32601  # Method not found

            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_rpc_source_with_handler_returns_status(
    socket_path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """AgentRpcSource WITH handler returns valid status for queries.

    This is the correct behavior - when handler is provided, status queries work.
    """

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    async with Engine(pipeline=test_pipeline) as engine:
        # Create source WITH handler (correct implementation)
        handler = AgentRpcHandler(engine=engine)
        rpc_source = AgentRpcSource(socket_path=socket_path, handler=handler)
        engine.add_source(rpc_source)

        async with anyio.create_task_group() as tg:

            async def run_engine() -> None:
                await engine.run(exit_on_completion=False)

            tg.start_soon(run_engine)
            await helpers.wait_for_socket(socket_path)

            # Connect and send status query
            async with await anyio.connect_unix(str(socket_path)) as conn:
                request = '{"jsonrpc":"2.0","method":"status","id":1}\n'
                await conn.send(request.encode())
                response_data = await conn.receive(1024)
                response = json.loads(response_data.decode().strip())

                # SUCCESS: With handler, queries return valid results
                assert "error" not in response, f"Got error: {response}"
                assert "result" in response
                assert response["result"]["state"] in ("idle", "active")

            tg.cancel_scope.cancel()


# =============================================================================
# EventBuffer Integration Tests
# =============================================================================


@pytest.mark.anyio
async def test_event_buffer_integration_with_engine(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """EventBuffer correctly captures events from engine execution.

    Integration: Verifies buffer is wired as sink and receives events.
    """
    from pivot import config
    from pivot.engine.agent_rpc import EventBuffer
    from pivot.engine.sources import OneShotSource

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    helpers.register_test_stage(_helper_noop, name="test_stage")

    buffer = EventBuffer(max_events=100)

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_sink(buffer)
        source = OneShotSource(stages=None, force=True, reason="test")
        engine.add_source(source)

        await engine.run(exit_on_completion=True)

    # Buffer should have captured events
    result = buffer.events_since(0)
    assert result["version"] > 0, "Buffer should have recorded events"
    assert len(result["events"]) > 0, "Buffer should contain events from execution"

    # Check for expected event types
    event_types = [e["event"]["type"] for e in result["events"]]
    assert "engine_state_changed" in event_types


@pytest.mark.anyio
async def test_event_buffer_polling_during_execution(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    test_pipeline: Pipeline,
) -> None:
    """EventBuffer allows polling events_since during execution.

    Integration: Simulates client polling while engine is running.
    """
    import time

    from pivot import config
    from pivot.engine.agent_rpc import EventBuffer
    from pivot.engine.sources import OneShotSource

    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(config, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pivot").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)

    def _slow_stage() -> dict[str, str]:
        """Stage that takes time to execute."""
        time.sleep(0.2)
        return {"result": "done"}

    helpers.register_test_stage(_slow_stage, name="slow_stage")

    buffer = EventBuffer(max_events=100)
    versions_seen = list[int]()

    async def poll_buffer() -> None:
        """Simulate client polling every 50ms."""
        last_version = 0
        for _ in range(10):
            await anyio.sleep(0.05)
            result = buffer.events_since(last_version)
            if result["version"] > last_version:
                versions_seen.append(result["version"])
                last_version = result["version"]

    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_sink(buffer)
        source = OneShotSource(stages=None, force=True, reason="test")
        engine.add_source(source)

        async with anyio.create_task_group() as tg:
            tg.start_soon(poll_buffer)

            async def run_engine() -> None:
                await engine.run(exit_on_completion=True)

            tg.start_soon(run_engine)

    # Should have seen multiple version updates during execution
    assert len(versions_seen) > 0, "Polling should capture events during execution"


# =============================================================================
# Error Handling Edge Cases
# =============================================================================
