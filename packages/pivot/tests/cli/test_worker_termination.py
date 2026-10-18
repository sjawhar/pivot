"""A worker whose Pivot process dies or is interrupted unwinds its running stage, then exits.

Each test drives the installed `pivot` binary in a fresh project, as a user does. The project
has two stages and two workers: `hold` starts a child process (in its own session, so only the
stage or Pivot can stop it) and waits; `quick` returns at once, leaving its worker idle. `hold`
stops its child the way the test's ``cleanup`` says, as a stage that starts a detached container
removes it.
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads the /proc process table"
)

_PIPELINE = """\
import os
import pathlib
import signal
import subprocess
import time

from pivot import Pipeline, stage

MARKERS = {markers!r}
CLEANUP = {cleanup!r}


def _mark(name, text=""):
    (pathlib.Path(MARKERS) / name).write_text(text)


def _stop_child(child):
    child.kill()
    child.wait()
    time.sleep(2)  # cleanup takes a while, as `docker rm --force` does


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
        time.sleep(300)
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
    """A `pivot repro` in its own session, with both stages started."""

    proc: subprocess.Popen[bytes]
    root: pathlib.Path
    markers: pathlib.Path

    def pid(self, name: str) -> int:
        return int((self.markers / f"{name}.pid").read_text())

    def stop(self, how: str) -> None:
        """SIGKILL pivot alone, SIGINT pivot alone (its cancel), or SIGINT pivot's whole
        process group (Ctrl-C at a terminal); then wait for pivot to exit."""
        if how == "kill":
            self.proc.send_signal(signal.SIGKILL)
        elif how == "interrupt":
            self.proc.send_signal(signal.SIGINT)
        else:
            assert how == "ctrl-c"
            os.killpg(self.proc.pid, signal.SIGINT)
        self.proc.wait(timeout=worker.TERMINATION_GRACE_S + 20)

    def assert_workers_and_child_gone(self) -> None:
        for name in ("worker", "quick", "child"):
            pid = self.pid(name)
            assert _helper_wait_until(
                lambda pid=pid: not _helper_alive(pid), worker.TERMINATION_GRACE_S + 5
            ), f"{name} process {pid} survived"


@contextlib.contextmanager
def _helper_running(tmp_path: pathlib.Path, cleanup: str, *, watch: bool = False) -> Iterator[_Run]:
    root = tmp_path / "project"
    markers = tmp_path / "markers"
    root.mkdir()
    markers.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for args in (["init"], ["config", "set", "core.max_workers", "2"]):
        subprocess.run([_PIVOT, *args], cwd=root, env=env, check=True, capture_output=True)
    (root / "pipeline.py").write_text(_PIPELINE.format(markers=str(markers), cleanup=cleanup))
    log = tmp_path / "pivot.log"
    with log.open("wb") as output:
        proc = subprocess.Popen(
            [_PIVOT, "repro", *(["--watch"] if watch else [])],
            cwd=root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    run = _Run(proc, root, markers)
    try:
        started = _helper_wait_until(
            lambda: proc.poll() is not None
            or ((markers / "started").exists() and (markers / "quick.pid").exists()),
            120,
        )
        assert started and proc.poll() is None, f"the stages never started:\n{log.read_text()}"
        time.sleep(0.5)  # `quick`'s worker reports its result and goes idle
        yield run
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        for name in ("worker", "quick", "child"):
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


@pytest.mark.parametrize("how", ["kill", "interrupt"])
def test_child_of_a_stage_without_cleanup_dies_with_its_worker(
    tmp_path: pathlib.Path, how: str
) -> None:
    """A child a stage starts and never stops is killed when its worker exits, not left to
    init."""
    with _helper_running(tmp_path, "none") as run:
        run.stop(how)

        run.assert_workers_and_child_gone()


def test_stage_that_replaces_the_sigterm_handler_is_stopped_after_the_grace(
    tmp_path: pathlib.Path,
) -> None:
    """When pivot dies, a worker whose stage installed its own SIGTERM handler still exits
    within ``worker.TERMINATION_GRACE_S``, killing the stage's child."""
    with _helper_running(tmp_path, "own-sigterm-handler") as run:
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
