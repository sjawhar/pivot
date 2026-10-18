"""A worker whose Pivot process dies or cancels unwinds the running stage before it exits.

Each test drives the installed `pivot` binary in a fresh project, as a user does, with a stage
that starts a child process and stops it in a ``finally`` block, the way a stage that starts a
detached container removes it.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

from pivot.executor import worker

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads the /proc process table"
)

_PIPELINE = """\
import os
import pathlib
import subprocess
import time

from pivot import Pipeline, stage


@stage
def hold() -> str:
    markers = pathlib.Path({markers!r})
    child = subprocess.Popen(["sleep", "300"])
    (markers / "worker.pid").write_text(str(os.getpid()))
    (markers / "child.pid").write_text(str(child.pid))
    (markers / "started").write_text("")
    try:
        time.sleep(300)
        return "done"
    finally:
        child.kill()
        child.wait()
        time.sleep(1)  # cleanup takes a while, as `docker rm --force` does
        (markers / "finally").write_text("")
        if {hang_in_finally!r}:
            time.sleep(600)


with Pipeline("t") as pipeline:
    hold()
"""


def _helper_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process, not a zombie waiting to be reaped."""
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _helper_wait_until(predicate: object, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return True
        time.sleep(0.1)
    return callable(predicate) and bool(predicate())


def _helper_start_stage(
    tmp_path: pathlib.Path, *, hang_in_finally: bool
) -> tuple[subprocess.Popen[bytes], pathlib.Path]:
    """Start `pivot repro` in its own session on a project whose one stage is running, and
    return the process and the directory the stage writes its markers to."""
    root = tmp_path / "project"
    markers = tmp_path / "markers"
    root.mkdir()
    markers.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run([_PIVOT, "init"], cwd=root, env=env, check=True, capture_output=True)
    (root / "pipeline.py").write_text(
        _PIPELINE.format(markers=str(markers), hang_in_finally=hang_in_finally)
    )
    proc = subprocess.Popen(
        [_PIVOT, "repro"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not _helper_wait_until(lambda: (markers / "started").exists(), 120):
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.fail("the stage never started")
    return proc, markers


@contextlib.contextmanager
def _helper_running_stage(tmp_path: pathlib.Path, *, hang_in_finally: bool = False):  # noqa: ANN202 - test helper
    proc, markers = _helper_start_stage(tmp_path, hang_in_finally=hang_in_finally)
    pids = [int((markers / name).read_text()) for name in ("worker.pid", "child.pid")]
    try:
        yield proc, markers
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        proc.wait(timeout=10)


def _helper_pids(markers: pathlib.Path) -> tuple[int, int]:
    return int((markers / "worker.pid").read_text()), int((markers / "child.pid").read_text())


def test_killed_pivot_lets_the_running_stage_clean_up(tmp_path: pathlib.Path) -> None:
    """SIGKILL of `pivot repro` alone: the worker unwinds its running stage, so the stage's
    ``finally`` block runs and stops the child it started, and then the worker exits."""
    with _helper_running_stage(tmp_path) as (proc, markers):
        worker_pid, child_pid = _helper_pids(markers)

        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)

        assert _helper_wait_until(lambda: (markers / "finally").exists(), 15), (
            "the stage's finally block never ran"
        )
        assert _helper_wait_until(lambda: not _helper_alive(worker_pid), 10), "worker survived"
        assert not _helper_alive(child_pid), "the stage's child survived"


def test_killed_pivot_cuts_off_a_finally_block_that_hangs(tmp_path: pathlib.Path) -> None:
    """A ``finally`` block that never returns does not keep the worker alive: it is cut off
    once ``worker.TERMINATION_GRACE_S`` has passed."""
    with _helper_running_stage(tmp_path, hang_in_finally=True) as (proc, markers):
        worker_pid, _child_pid = _helper_pids(markers)

        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)

        assert _helper_wait_until(lambda: (markers / "finally").exists(), 15)
        assert _helper_alive(worker_pid), "the worker is still in the stage's finally block"
        assert _helper_wait_until(
            lambda: not _helper_alive(worker_pid), worker.TERMINATION_GRACE_S + 5
        ), "the hung finally block kept the worker alive past the grace"


def test_interrupted_pivot_lets_the_running_stage_clean_up(tmp_path: pathlib.Path) -> None:
    """Ctrl-C (SIGINT to `pivot repro`, which cancels its workers) also unwinds the running
    stage before its worker is stopped, instead of killing the worker mid-stage."""
    with _helper_running_stage(tmp_path) as (proc, markers):
        worker_pid, child_pid = _helper_pids(markers)

        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=worker.TERMINATION_GRACE_S + 15)

        assert (markers / "finally").exists(), "the stage's finally block never ran"
        assert _helper_wait_until(lambda: not _helper_alive(worker_pid), 10), "worker survived"
        assert not _helper_alive(child_pid), "the stage's child survived"


def test_ctrl_c_at_the_terminal_lets_the_running_stage_clean_up(tmp_path: pathlib.Path) -> None:
    """Ctrl-C at a terminal sends SIGINT to pivot and its workers at once; the stage's
    ``finally`` block still runs to its end while pivot cancels the worker."""
    with _helper_running_stage(tmp_path) as (proc, markers):
        worker_pid, child_pid = _helper_pids(markers)

        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=worker.TERMINATION_GRACE_S + 15)

        assert (markers / "finally").exists(), "the stage's finally block never ran"
        assert _helper_wait_until(lambda: not _helper_alive(worker_pid), 10), "worker survived"
        assert not _helper_alive(child_pid), "the stage's child survived"
