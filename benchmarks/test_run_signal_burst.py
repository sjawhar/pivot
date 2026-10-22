"""A further signal during run.py's interrupt cleanup must neither orphan the command nor hang run.py.

When run.py is interrupted, `_stop` sends the measured command SIGINT, waits out a grace period,
and then, in a `finally`, kills the command and SIGKILLs whatever is left in its session. Two ways
a further Ctrl-C or SIGTERM breaks that:

- It arrives while the `finally` runs. It raises out of the kill or the sweep, and what is left
  of the command's session keeps running at PPID 1. A burst of signals from a supervisor or a
  script does this every time; so does a second Ctrl-C about 0.26 s after the first.
- Its handler runs just after CPython's `Popen.poll()` or `Popen.wait(timeout)` has taken the
  Popen's internal lock with a non-blocking acquire, before the `try` that releases it. The
  lock stays held, so the `finally`'s `proc.wait()` blocks forever and run.py never exits.
  `pkill -TERM -f run.py` on a `uv run` run (pkill's SIGTERM and the one uv forwards) hung it
  this way on a real pivot.
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

import run

_BENCHMARKS = pathlib.Path(run.__file__).parent
_RUN_PY = pathlib.Path(run.__file__)
# A stand-in for pivot whose `repro` ignores SIGINT, as a pivot whose shutdown outlasts the grace
# period would, so run.py's cleanup has to kill it. It writes its session id, and a marker once
# SIGINT has reached it.
_DEAF = """#!/bin/sh
case "$1" in
  init) mkdir -p .pivot ;;
  repro) trap 'echo int > {sigint_file}' INT
         sleep 600 & ps -o sid= -p $$ > {sid_file}; while :; do wait; done ;;
esac
"""
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
# Calls run.measure() in this process, as run.py's main loop does, with run.py's SIGTERM handler.
# Popen's internal lock is wrapped so that, once the first signal has arrived, the next
# non-blocking acquire that succeeds is followed at once by a real SIGTERM to this process: the
# moment at which a signal handler that raises leaves the lock held.
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


def _helper_kill_session(sid: int) -> None:
    for pid in run.session_members(sid) if sid else []:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_a_burst_of_signals_during_cleanup_still_stops_the_command(
    tmp_path: pathlib.Path, signum: signal.Signals
) -> None:
    sid_file, sigint_file = tmp_path / "sid", tmp_path / "got-sigint"
    pivot = tmp_path / "stand-in-pivot"
    pivot.write_text(_DEAF.format(sid_file=sid_file, sigint_file=sigint_file))
    pivot.chmod(0o755)
    argv = [sys.executable, str(_RUN_PY), "--project", "chain-10", "--variant", f"a={pivot}"]
    argv += ["--reps", "1", "--work", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out")]
    harness = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sid = 0
    try:
        sid = int(_helper_read_when_written(sid_file))
        harness.send_signal(signal.SIGINT)
        # The stand-in has its SIGINT, so run.py is waiting out the grace period.
        _helper_read_when_written(sigint_file)

        # One signal a millisecond until run.py exits: the first ends the grace period, and the
        # rest arrive while run.py kills the command and sweeps its session.
        deadline = time.monotonic() + 2
        while harness.poll() is None and time.monotonic() < deadline:
            with contextlib.suppress(ProcessLookupError):
                harness.send_signal(signum)
            time.sleep(0.001)
        harness.wait(timeout=30)

        left = _helper_alive_after_settling(sid)
        assert not left, f"a burst of signals during cleanup orphaned the measured session: {left}"
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait()
        _helper_kill_session(sid)


@pytest.mark.parametrize(
    "first",
    [
        # The second signal lands in _stop itself, in the Popen.poll() that Popen.send_signal()
        # makes before it sends pivot SIGINT.
        pytest.param(signal.SIGTERM, id="sigterm-then-sigterm"),
        # The second signal lands in the 0.25 s that CPython's Popen.wait() waits for the child
        # after a KeyboardInterrupt, before run.py's own cleanup starts.
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
        _helper_kill_session(sid)
