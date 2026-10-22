"""A signal burst during runner cleanup must neither orphan the command nor hang run.py.

The measured command gets SIGINT, a grace period, a kill, a session sweep, and reaping in that
order. Further SIGINTs and SIGTERMs cannot interrupt that cleanup.

The lock-window probe models a raising handler immediately after a non-blocking Popen lock
acquire. Cleanup must reap directly rather than depend on that possibly stranded lock.
"""

from __future__ import annotations

import contextlib
import pathlib
import signal
import subprocess
import sys
import time

import pytest

import run
import run_test_helpers

_BENCHMARKS = pathlib.Path(run.__file__).parent
_RUN_PY = pathlib.Path(run.__file__)
# A stand-in that exits on SIGINT, leaving `sleep 600` in its session for the sweep, as pivot
# leaves loky's resource tracker. Once run.measure() is waiting for it, it interrupts its parent,
# the probe, with `first`.
_INTERRUPTS_ITS_PARENT = """#!/bin/sh
trap 'exit 130' INT
sleep 600 &
ps -o sid= -p $$ > {sid_file}
sleep 0.5
kill -{first} $PPID
while :; do wait; done
"""
# Calls run.measure() with a custom Popen lock. Once the first signal arrives, the next
# non-blocking acquire sends SIGTERM before it returns, modeling a raising handler in that window.
# Direct reaping must not depend on the lock.
_PROBE = """
import os, signal, subprocess, sys, threading
sys.path.insert(0, sys.argv[1])
import run

locks = []


class Lock:
    def __init__(self):
        self._lock = threading.Lock()
        self.armed = False

    def acquire(self, blocking=True, timeout=-1):
        got = self._lock.acquire(blocking, timeout)
        if got and not blocking and self.armed:
            self.armed = False
            os.kill(os.getpid(), signal.SIGTERM)
        return got

    def release(self):
        self._lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


init = subprocess.Popen.__init__


def tracked_init(self, *args, **kwargs):
    init(self, *args, **kwargs)
    self._waitpid_lock = Lock()
    locks.append(self._waitpid_lock)


subprocess.Popen.__init__ = tracked_init


def arm_once(handler):
    def first(signum, frame):
        for lock in locks:
            lock.armed = True
        signal.signal(signum, handler)
        handler(signum, frame)
    return first


on_term = lambda signum, _frame: sys.exit(128 + signum)
signal.signal(signal.SIGTERM, arm_once(on_term))
signal.signal(signal.SIGINT, arm_once(signal.default_int_handler))
run.measure([sys.argv[2], "repro"], sys.argv[3])
"""


def _helper_read_when_written(path: pathlib.Path) -> str:
    deadline = time.monotonic() + 20
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path.name} was never written"
        time.sleep(0.01)
    return path.read_text()


def _helper_alive_after_settling(sid: int) -> list[int]:
    """Members of session `sid` still alive (not zombies) after up to 5 s."""
    deadline = time.monotonic() + 5
    alive = run.session_members(sid)
    while alive and time.monotonic() < deadline:
        time.sleep(0.05)
        alive = run.session_members(sid)
    return alive


def _helper_semaphores(sid: int) -> list[pathlib.Path]:
    return list(pathlib.Path("/dev/shm").glob(f"sem.loky-{sid}-*"))


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_a_burst_of_signals_during_cleanup_still_stops_the_command(
    tmp_path: pathlib.Path, signum: signal.Signals
) -> None:
    sid_file, sigint_file = tmp_path / "sid", tmp_path / "got-sigint"
    pivot = run_test_helpers.make_stand_in(tmp_path, run_test_helpers.DEAF_STAND_IN)
    argv = [sys.executable, str(_RUN_PY), *run_test_helpers.run_argv(tmp_path, pivot)]
    harness = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sid = 0
    try:
        sid = int(_helper_read_when_written(sid_file))
        harness.send_signal(signal.SIGINT)
        # The stand-in has its SIGINT, so run.py is waiting out the grace period.
        _helper_read_when_written(sigint_file)

        # Further signals stay blocked until the measured session has been cleaned up.
        deadline = time.monotonic() + 2
        while harness.poll() is None and time.monotonic() < deadline:
            with contextlib.suppress(ProcessLookupError):
                harness.send_signal(signum)
            time.sleep(0.001)
        harness.wait(timeout=30)

        left = _helper_alive_after_settling(sid)
        assert not left, f"a burst of signals during cleanup orphaned the measured session: {left}"
        assert not _helper_semaphores(sid), "the burst leaked loky semaphore files"
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait()
        if sid:
            run_test_helpers.kill_session(sid)


@pytest.mark.parametrize(
    "first",
    [
        # SIGTERM arms the injected lock before cleanup. A Popen-based cleanup would take it
        # again through its non-blocking poll.
        pytest.param(signal.SIGTERM, id="sigterm-then-sigterm"),
        # SIGINT arms the lock while Popen's wait unwinds. A Popen-based cleanup would take it
        # again after its interrupt handling.
        pytest.param(signal.SIGINT, id="ctrl-c-then-sigterm"),
    ],
)
def test_a_second_signal_in_popens_lock_window_does_not_hang_run_py(
    tmp_path: pathlib.Path, first: signal.Signals
) -> None:
    sid_file = tmp_path / "sid"
    pivot = tmp_path / "stand-in-pivot"
    pivot.write_text(_INTERRUPTS_ITS_PARENT.format(sid_file=sid_file, first=first.name[3:]))
    pivot.chmod(0o755)
    probe = subprocess.Popen(
        [sys.executable, "-c", _PROBE, str(_BENCHMARKS), str(pivot), str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sid = 0
    try:
        sid = int(_helper_read_when_written(sid_file))
        try:
            exit_code = probe.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail("run.measure() never returned after the second signal: run.py hangs")
        assert exit_code in (-first, 128 + first, -signal.SIGTERM, 128 + signal.SIGTERM), exit_code
        left = _helper_alive_after_settling(sid)
        assert not left, f"the second signal orphaned the measured session: {left}"
    finally:
        if probe.poll() is None:
            probe.kill()
            probe.wait()
        if sid:
            run_test_helpers.kill_session(sid)
