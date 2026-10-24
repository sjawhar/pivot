"""A worker whose Pivot process dies or is interrupted unwinds its running stage, then exits.

Each test drives the installed `pivot` binary in a fresh project, as a user does. The project
has a stage `hold` that starts a child process (in its own session, so only the stage or Pivot
can stop it) and waits, stopping the child the way the test's ``cleanup`` says, as a stage that
starts a detached container removes it. Most tests add a second stage, `quick`, that returns at
once and leaves its worker idle, so pivot runs two workers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from pivot.executor import worker
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads the /proc process table"
)

_PIPELINE = """\
import ctypes
import os
import pathlib
import signal
import subprocess
import time

from pivot import Pipeline, stage

MARKERS = {markers!r}
CLEANUP = {cleanup!r}
WITH_QUICK = {with_quick!r}


def _mark(name, text=""):
    (pathlib.Path(MARKERS) / name).write_text(text)


def _stop_child(child):
    child.kill()
    child.wait()
    time.sleep(2)  # cleanup takes a while, as `docker rm --force` does


def _block_in_c():
    # Relocking a default mutex blocks in C forever; no signal handler runs meanwhile.
    libc = ctypes.CDLL(None)
    mutex = ctypes.create_string_buffer(64)
    libc.pthread_mutex_init(mutex, None)
    libc.pthread_mutex_lock(mutex)
    libc.pthread_mutex_lock(mutex)


@stage
def hold() -> str:
    if CLEANUP == "own-sigterm-handler":
        signal.signal(signal.SIGTERM, lambda *_: None)
    child = subprocess.Popen(["sleep", "300"], start_new_session=True)
    _mark("worker.pid", str(os.getpid()))
    _mark("child.pid", str(child.pid))
    _mark("started")
    if CLEANUP in ("none", "own-sigterm-handler"):
        time.sleep(300)
        return "done"
    if CLEANUP == "blocked-in-c":
        _block_in_c()
        return "done"
    if CLEANUP == "except-keyboard-interrupt":
        try:
            time.sleep(300)
        except KeyboardInterrupt:
            _stop_child(child)
            _mark("cleaned")
            raise
        return "done"
    if CLEANUP == "swallow":
        try:
            time.sleep(300)
        except BaseException:
            _stop_child(child)
        return "fallback"
    try:
        time.sleep(5 if CLEANUP == "short" else 300)
        return "done"
    finally:
        if CLEANUP == "hang":
            _mark("cleanup-started")
            time.sleep(600)
        _stop_child(child)
        _mark("cleaned")


@stage
def quick() -> str:
    _mark("quick.pid", str(os.getpid()))
    return "quick"


with Pipeline("t") as pipeline:
    hold()
    if WITH_QUICK:
        quick()
"""


def _helper_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process, not a zombie waiting to be reaped."""
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _helper_wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


@dataclasses.dataclass
class _Run:
    """A `pivot repro` in its own session, with its stages started."""

    proc: subprocess.Popen[bytes]
    root: pathlib.Path
    markers: pathlib.Path
    log: pathlib.Path
    pid_names: tuple[str, ...]

    def pid(self, name: str) -> int:
        return int((self.markers / f"{name}.pid").read_text())

    def stop(self, how: str) -> None:
        """SIGKILL pivot alone, SIGINT pivot alone (its cancel), or SIGINT pivot's whole
        process group (Ctrl-C at a terminal); then wait for pivot to exit. An interrupted
        pivot must exit cleanly, reporting no error."""
        if how == "kill":
            self.proc.send_signal(signal.SIGKILL)
        elif how == "interrupt":
            self.proc.send_signal(signal.SIGINT)
        else:
            assert how == "ctrl-c"
            os.killpg(self.proc.pid, signal.SIGINT)
        self.proc.wait(timeout=worker.TERMINATION_GRACE_S + 20)
        if how != "kill":
            output = self.log.read_text()
            assert self.proc.returncode == 0, output
            assert "Error:" not in output, output

    def assert_workers_and_child_gone(self) -> None:
        for name in self.pid_names:
            pid = self.pid(name)
            assert _helper_wait_until(
                lambda pid=pid: not _helper_alive(pid), worker.TERMINATION_GRACE_S + 5
            ), f"{name} process {pid} survived"


@contextlib.contextmanager
def _helper_running(
    tmp_path: pathlib.Path,
    cleanup: str,
    *,
    watch: bool = False,
    workers: int = 2,
    ignore_sigint: bool = False,
) -> Iterator[_Run]:
    """Start `pivot repro` on the project, with ``workers`` workers (one runs `hold` alone;
    two add `quick`), and SIGINT inherited as ignored when ``ignore_sigint``, as a background
    job of a non-interactive shell inherits it."""
    root = tmp_path / "project"
    markers = tmp_path / "markers"
    root.mkdir()
    markers.mkdir()
    with_quick = workers > 1
    env = helpers.cli_env(tmp_path / "home")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for args in (["init"], ["config", "set", "core.max_workers", str(workers)]):
        subprocess.run([helpers.PIVOT, *args], cwd=root, env=env, check=True, capture_output=True)
    (root / "pipeline.py").write_text(
        _PIPELINE.format(markers=str(markers), cleanup=cleanup, with_quick=with_quick)
    )
    command = [helpers.PIVOT, "repro", *(["--watch"] if watch else [])]
    if ignore_sigint:
        command = ["sh", "-c", 'trap "" INT; exec "$0" "$@"', *command]
    log = tmp_path / "pivot.log"
    with log.open("wb") as output:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_names = ("worker", "child", *(("quick",) if with_quick else ()))
    run = _Run(proc, root, markers, log, pid_names)
    try:
        started = _helper_wait_until(
            lambda: proc.poll() is not None
            or (
                (markers / "started").exists()
                and (not with_quick or (markers / "quick.pid").exists())
            ),
            120,
        )
        assert started and proc.poll() is None, f"the stages never started:\n{log.read_text()}"
        time.sleep(0.5)  # `quick`'s worker reports its result and goes idle
        yield run
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        for name in pid_names:
            with contextlib.suppress(OSError, ValueError):
                os.kill(run.pid(name), signal.SIGKILL)
        proc.wait(timeout=10)


@pytest.mark.parametrize("how", ["kill", "interrupt", "ctrl-c"])
@pytest.mark.parametrize("cleanup", ["finally", "except-keyboard-interrupt"])
def test_stopped_pivot_lets_the_running_stage_clean_up(
    tmp_path: pathlib.Path, cleanup: str, how: str
) -> None:
    """However pivot stops, the running stage's ``finally`` block or ``except
    KeyboardInterrupt`` cleanup runs to its end, though the other worker is idle and exits at
    once; then both workers exit, and the stage's child is gone."""
    with _helper_running(tmp_path, cleanup) as run:
        run.stop(how)

        assert _helper_wait_until(lambda: (run.markers / "cleaned").exists(), 15), (
            "the stage's cleanup never finished"
        )
        run.assert_workers_and_child_gone()


def test_ctrl_c_in_watch_mode_lets_the_running_stage_clean_up(tmp_path: pathlib.Path) -> None:
    """`pivot repro --watch` stops its startup run the same way on Ctrl-C at a terminal."""
    with _helper_running(tmp_path, "finally", watch=True) as run:
        run.stop("ctrl-c")

        assert _helper_wait_until(lambda: (run.markers / "cleaned").exists(), 15), (
            "the stage's cleanup never finished"
        )
        run.assert_workers_and_child_gone()


@pytest.mark.parametrize("how", ["kill", "interrupt"])
def test_cleanup_that_hangs_is_cut_off_and_its_child_killed(
    tmp_path: pathlib.Path, how: str
) -> None:
    """A cleanup that never returns, before it stops the stage's child, does not keep the
    worker alive past ``worker.TERMINATION_GRACE_S``, and the child dies with the worker."""
    with _helper_running(tmp_path, "hang") as run:
        run.stop(how)

        assert _helper_wait_until(lambda: (run.markers / "cleanup-started").exists(), 15)
        run.assert_workers_and_child_gone()


@pytest.mark.parametrize("how", ["interrupt", "ctrl-c"])
@pytest.mark.parametrize("cleanup", ["own-sigterm-handler", "blocked-in-c", "hang"])
def test_interrupted_pivot_exits_when_its_only_worker_outlives_the_grace(
    tmp_path: pathlib.Path, cleanup: str, how: str
) -> None:
    """With one worker, whose stage swallows SIGTERM, blocks in C where no signal handler
    runs, or hangs in its cleanup, an interrupted pivot kills the worker and its child after
    ``worker.TERMINATION_GRACE_S`` and exits cleanly."""
    with _helper_running(tmp_path, cleanup, workers=1) as run:
        run.stop(how)

        run.assert_workers_and_child_gone()


@pytest.mark.parametrize("how", ["kill", "interrupt"])
def test_child_of_a_stage_without_cleanup_dies_with_its_worker(
    tmp_path: pathlib.Path, how: str
) -> None:
    """A child a stage starts and never stops is killed when its worker exits, not left to
    init."""
    with _helper_running(tmp_path, "none") as run:
        run.stop(how)

        run.assert_workers_and_child_gone()


@pytest.mark.parametrize("cleanup", ["own-sigterm-handler", "blocked-in-c"])
def test_worker_that_ignores_the_stop_still_exits_when_pivot_dies(
    tmp_path: pathlib.Path, cleanup: str
) -> None:
    """When pivot dies, a worker whose stage installed its own SIGTERM handler, or is blocked
    in C, still exits within ``worker.TERMINATION_GRACE_S``, killing the stage's child."""
    with _helper_running(tmp_path, cleanup) as run:
        run.stop("kill")

        run.assert_workers_and_child_gone()


def test_stage_that_swallows_the_stop_commits_nothing(tmp_path: pathlib.Path) -> None:
    """A stage that catches the stop and returns a value has that value discarded: it is not
    written as the stage's output, and no lock is written for it."""
    with _helper_running(tmp_path, "swallow") as run:
        run.stop("kill")

        run.assert_workers_and_child_gone()
        assert not (run.root / "data" / "t" / "hold.txt").exists()
        assert not (run.root / ".pivot" / "stages" / "t" / "hold.lock").exists()


def test_sigint_inherited_as_ignored_does_not_interrupt_the_stage(tmp_path: pathlib.Path) -> None:
    """Pivot started with SIGINT ignored, as a background job of a non-interactive shell is,
    ignores a SIGINT to its process group, and so do its workers: the stage runs to the end."""
    with _helper_running(tmp_path, "short", ignore_sigint=True) as run:
        os.killpg(run.proc.pid, signal.SIGINT)
        run.proc.wait(timeout=60)

        output = run.log.read_text()
        assert run.proc.returncode == 0, output
        assert (run.root / "data" / "t" / "hold.txt").read_text() == "done", output
