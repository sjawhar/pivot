"""End-to-end tests for serve mode through the installed pivot command and CLI entry points."""

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
import sysconfig
import tempfile
import time
from typing import TYPE_CHECKING, NoReturn

import pytest

from pivot import cli
from tests.conftest import send_rpc
from tests.integration import serve_test_support

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")

# =============================================================================
# CLI Serve Mode E2E Tests
# =============================================================================
# Runs serve in-process through pivot's embedding entry points; its docstring gives the modes.
_INPROC_SERVE = pathlib.Path(__file__).with_name("inproc_serve.py")


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
    serve_test_support.create_noop_project(project)
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
    serve_test_support.create_noop_project(project)
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


# Libraries that install a C SIGTERM handler: from an LD_PRELOAD constructor before Python
# starts (as a C application that embeds Python can), or when Python calls them after start (as
# a C extension can, unseen by signal.getsignal()). The handler reports each SIGTERM on stderr.
_HOST_HANDLER_C = r"""
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void host_handler(int sig) {
    static const char message[] = "host-handler: SIGTERM\n";
    (void)sig;
    (void)!write(2, message, sizeof message - 1);
}

int install_sigterm_handler(void) {
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = host_handler;
    sigemptyset(&action.sa_mask);
    return sigaction(SIGTERM, &action, NULL);
}

__attribute__((constructor)) static void install_when_preloaded(void) {
    if (getenv("LD_PRELOAD") != NULL) {
        install_sigterm_handler();
        unsetenv("LD_PRELOAD");
    }
}
"""

# A C program that installs its own SIGTERM handler, then embeds Python and runs a script:
# `host <python> <script> <args...>`.
_EMBEDDING_HOST_C = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

static void host_handler(int sig) {
    static const char message[] = "host-handler: SIGTERM\n";
    (void)sig;
    (void)!write(2, message, sizeof message - 1);
}

int main(int argc, char **argv) {
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = host_handler;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    PyConfig_SetBytesString(&config, &config.program_name, argv[1]);
    PyConfig_SetBytesArgv(&config, argc - 1, argv + 1);
    PyStatus status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
    return Py_RunMain();
}
"""


def _helper_build_prerequisite_missing(message: str) -> NoReturn:
    """Skip locally and fail in CI, where every C build these tests need must work."""
    if os.environ.get("CI"):
        pytest.fail(message)
    pytest.skip(message)


def _helper_c_compiler() -> str:
    compiler = shutil.which("cc")
    if compiler is None:
        _helper_build_prerequisite_missing("these tests need a C compiler, which is not installed")
    return compiler


@pytest.fixture(scope="module")
def host_handler_library(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build the host-handler library once for the signal-ownership tests."""
    compiler = _helper_c_compiler()
    source = tmp_path_factory.mktemp("host-handler") / "host_handler.c"
    source.write_text(_HOST_HANDLER_C)
    library = source.with_suffix(".so")
    subprocess.run([compiler, "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    return library


def _helper_embedding_host(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build the embedding host against the Python running the tests."""
    compiler = _helper_c_compiler()
    variables = sysconfig.get_config_vars()
    include = pathlib.Path(variables["INCLUDEPY"])
    libdir = pathlib.Path(variables["LIBDIR"])
    if not variables["Py_ENABLE_SHARED"] or not (include / "Python.h").is_file():
        _helper_build_prerequisite_missing(
            f"the embedding host needs Python's headers and shared library (in {include} and {libdir})"
        )
    source = tmp_path / "host.c"
    source.write_text(_EMBEDDING_HOST_C)
    host = tmp_path / "host"
    command = [compiler, str(source), "-o", str(host), f"-I{include}", f"-L{libdir}"]
    command += [f"-Wl,-rpath,{libdir}", f"-lpython{variables['LDVERSION']}"]
    subprocess.run([*command, *str(variables["LIBS"]).split(), "-lm"], check=True)
    return host


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses LD_PRELOAD and /proc")
@pytest.mark.parametrize("handler", ["default", "python", "c-before", "c-after"])
@pytest.mark.parametrize("entry", ["main", "click"])
def test_inprocess_serve_leaves_the_callers_signal_state_untouched(
    tmp_path: pathlib.Path, entry: str, handler: str, host_handler_library: pathlib.Path
) -> None:
    """Only the `pivot` command owns its process. An application that runs serve in-process
    keeps its own signal state, as on main: its SIGTERM and Ctrl-C handlers, as Python and the
    kernel report them, its signal mask and its wakeup fd are the same before, during and after
    serve. Its own SIGTERM handler decides what a SIGTERM during serve does."""
    library = host_handler_library
    project = tmp_path / "project"
    serve_test_support.create_noop_project(project)
    env = {**os.environ, "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    if handler == "c-before":
        env["LD_PRELOAD"] = str(library)
    result = subprocess.run(
        [sys.executable, str(_INPROC_SERVE), str(project), entry, handler, str(library)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.splitlines()[-1])
    before, serving, after = report["before"], report["serving"], report["after"]
    expect_caught = "no" if handler == "default" else "yes"
    assert before["kernel"]["SIGTERM:SigCgt"] == expect_caught, report
    assert before["wakeup_fd"] != -1, report
    # While serve runs, asyncio's Runner holds Ctrl-C as on main; SIGTERM stays the caller's.
    assert serving["kernel"] == before["kernel"], f"serve changed the kernel's state: {report}"
    assert serving["handlers"]["SIGTERM"] == before["handlers"]["SIGTERM"], (
        f"serve took SIGTERM: {report}"
    )
    assert after == before, f"serve left the caller's signal state changed: {report}"
    outcome = "raised SystemExit(0)" if entry == "main" else "returned"
    assert report["outcome"] == outcome, f"{report}\n{result.stderr[-2000:]}"
    assert not report["socket_left"], report
    # The caller's wakeup fd received Ctrl-C, and a SIGTERM its Python handler caught.
    assert signal.SIGINT in report["wakeup_bytes"], report
    assert (signal.SIGTERM in report["wakeup_bytes"]) == (handler == "python"), report
    if handler != "default":
        assert report["answers_after_sigterm"], f"a SIGTERM to the caller stopped serve: {report}"
    assert report["python_handler_ran"] == ([signal.SIGTERM] if handler == "python" else [])
    assert report["child_after_sigterm"] == -signal.SIGTERM, report


def test_cli_module_run_as_a_script_owns_stop_signals(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run as a script, pivot's CLI module owns its process as the `pivot` command does: a
    SIGTERM stops serve, which removes its socket and exits 0."""
    project = tmp_path / "project"
    serve_test_support.create_noop_project(project)
    script = pathlib.Path(cli.__file__ or "")
    assert script.name == "__init__.py", script
    # Short, because multiprocessing binds its manager's socket under TMPDIR.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script), "repro", "--watch", "--serve"],
            cwd=project,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        monkeypatch.chdir(project)
        try:
            response = _helper_wait_for_status(
                proc, pathlib.Path(".pivot", "agent.sock"), time.monotonic() + 30
            )
            assert response is not None, f"serve did not answer (exit {proc.poll()})"
            proc.send_signal(signal.SIGTERM)
            _, stderr = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert proc.returncode == 0, f"exit {proc.returncode}: {stderr.decode()[-2000:]}"
        assert not (project / ".pivot" / "agent.sock").exists(), "serve left its socket"
    finally:
        shutil.rmtree(private_tmp)


# A sitecustomize module, which Python imports at start from PYTHONPATH before it runs the script:
# a launcher can install a Python SIGTERM handler this way. As the LD_PRELOAD library does, it
# drops its marker from the environment, so the Python processes pivot starts do not install it.
_LAUNCHER_SITECUSTOMIZE = """\
import os
import signal


def _launcher_handler(_signum, _frame):
    os.write(2, b"host-handler: SIGTERM\\n")


if os.environ.pop("PIVOT_TEST_LAUNCHER_HANDLER", None):
    signal.signal(signal.SIGTERM, _launcher_handler)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses LD_PRELOAD and /proc")
@pytest.mark.parametrize("launcher", ["embedding-host", "ld-preload", "sitecustomize"])
def test_pivot_command_leaves_a_sigterm_handler_its_launcher_installed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    launcher: str,
    host_handler_library: pathlib.Path,
) -> None:
    """What launched the `pivot` command can give SIGTERM a handler before the script runs: in C
    before Python starts (a C program that embeds Python, an LD_PRELOAD library), or in Python
    (a sitecustomize module). That launcher owns SIGTERM, so serve leaves it alone, as on main:
    the launcher's handler runs on SIGTERM, serve keeps answering, and Ctrl-C stops serve, which
    exits 0 and removes its socket."""
    project = tmp_path / "project"
    serve_test_support.create_noop_project(project)
    pivot_script = pathlib.Path(_PIVOT_BIN)
    env = {**os.environ, "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    argv = [str(pivot_script)]
    if launcher == "embedding-host":
        argv = [str(_helper_embedding_host(tmp_path)), sys.executable, str(pivot_script)]
    elif launcher == "ld-preload":
        env["LD_PRELOAD"] = str(host_handler_library)
    else:
        site_dir = tmp_path / "site"
        site_dir.mkdir()
        (site_dir / "sitecustomize.py").write_text(_LAUNCHER_SITECUSTOMIZE)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(site_dir), env.get("PYTHONPATH")]))
        env["PIVOT_TEST_LAUNCHER_HANDLER"] = "1"
    # Short, because multiprocessing binds its manager's socket under TMPDIR.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    env["TMPDIR"] = str(private_tmp)
    stderr_path = tmp_path / "serve.stderr"
    socket_path = pathlib.Path(".pivot", "agent.sock")
    try:
        with stderr_path.open("wb") as stderr_file:
            proc = subprocess.Popen(
                [*argv, "repro", "--watch", "--serve"],
                cwd=project,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                start_new_session=True,
            )
        monkeypatch.chdir(project)
        try:
            response = _helper_wait_for_status(proc, socket_path, time.monotonic() + 30)
            assert response is not None, (
                f"serve did not answer (exit {proc.poll()}):\n{stderr_path.read_text()[-2000:]}"
            )
            proc.send_signal(signal.SIGTERM)
            time.sleep(2)
            assert proc.poll() is None, (
                f"SIGTERM stopped serve (exit {proc.returncode}):\n{stderr_path.read_text()[-2000:]}"
            )
            assert "host-handler: SIGTERM" in stderr_path.read_text(), (
                f"the launcher's SIGTERM handler did not run:\n{stderr_path.read_text()[-2000:]}"
            )
            assert send_rpc(socket_path, "status") is not None, "serve stopped answering"
            proc.send_signal(signal.SIGINT)
            exit_code = proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        assert exit_code == 0, f"exit {exit_code} after Ctrl-C:\n{stderr_path.read_text()[-2000:]}"
        assert not (project / socket_path).exists(), "serve left its socket"
    finally:
        shutil.rmtree(private_tmp)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/<pid>/status")
def test_pivot_command_leaves_sigint_ignored_when_its_launcher_ignores_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launcher can start the `pivot` command with Ctrl-C ignored (a background job of a
    non-interactive shell, `trap '' INT`). That launcher owns SIGINT, so serve leaves it ignored:
    a SIGINT changes nothing and serve keeps answering. SIGTERM, still at its default, stops
    serve, which exits 0 and removes its socket."""
    project = tmp_path / "project"
    serve_test_support.create_noop_project(project)
    # Short, because multiprocessing binds its manager's socket under TMPDIR.
    private_tmp = pathlib.Path(tempfile.mkdtemp(prefix="t", dir="/tmp"))
    env = {**os.environ, "TMPDIR": str(private_tmp), "PIVOT_CACHE_DIR": str(tmp_path / "cache")}
    stderr_path = tmp_path / "serve.stderr"
    socket_path = pathlib.Path(".pivot", "agent.sock")
    try:
        # The launcher ignores SIGINT; the child inherits SIG_IGN across exec.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with stderr_path.open("wb") as stderr_file:
                proc = subprocess.Popen(
                    [_PIVOT_BIN, "repro", "--watch", "--serve"],
                    cwd=project,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    start_new_session=True,
                )
        finally:
            signal.signal(signal.SIGINT, previous)
        monkeypatch.chdir(project)
        try:
            response = _helper_wait_for_status(proc, socket_path, time.monotonic() + 30)
            assert response is not None, (
                f"serve did not answer (exit {proc.poll()}):\n{stderr_path.read_text()[-2000:]}"
            )
            serving = serve_test_support.process_signal_disposition(proc.pid, signal.SIGINT)
            proc.send_signal(signal.SIGINT)
            time.sleep(2)
            assert proc.poll() is None, (
                f"the launcher ignored SIGINT, but SIGINT stopped serve (exit {proc.returncode}); "
                + f"SIGINT while serving was {serving}:\n{stderr_path.read_text()[-2000:]}"
            )
            assert serving == "ignored", f"serve replaced the launcher's SIG_IGN: SIGINT {serving}"
            assert send_rpc(socket_path, "status") is not None, "serve stopped answering"
            proc.send_signal(signal.SIGTERM)
            exit_code = proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        assert exit_code == 0, f"exit {exit_code} after SIGTERM:\n{stderr_path.read_text()[-2000:]}"
        assert not (project / socket_path).exists(), "serve left its socket"
    finally:
        shutil.rmtree(private_tmp)


# Serves as the console script does and resends SIGTERM at each disposition change after a stop.
_STOP_WINDOW = pathlib.Path(__file__).with_name("serve_stop_window.py")


def test_a_sigterm_during_serves_shutdown_changes_nothing(tmp_path: pathlib.Path) -> None:
    """After the first stop, a later SIGTERM is a no-op: SIGTERM's disposition goes from serve's
    handler straight to SIG_IGN and never passes through SIG_DFL, which would let that SIGTERM
    kill the process on whichever thread the kernel picks."""
    project = tmp_path / "project"
    serve_test_support.create_noop_project(project)
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
    serve_test_support.create_noop_project(project)
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
        assert "Stage test/slow failed" not in stderr, stderr
    finally:
        shutil.rmtree(private_tmp)


@pytest.mark.parametrize("long_path", [False, True], ids=["short-path", "long-path"])
def test_stopping_first_live_serve_keeps_replacement_socket(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, long_path: bool
) -> None:
    """A newer serve takes the project's socket path as current startup behavior permits. When
    the older serve stops, it must leave the newer socket in place."""
    project = tmp_path / (("deep-project-" * 10) if long_path else "project")
    serve_test_support.create_noop_project(project)
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
                # The replacement serve removes the first serve's socket before it binds its own,
                # so the path can vanish between two reads. Read it once; a missing path means
                # the replacement has not bound yet.
                with contextlib.suppress(FileNotFoundError):
                    if index and socket_path.lstat().st_ino == first_inode:
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
