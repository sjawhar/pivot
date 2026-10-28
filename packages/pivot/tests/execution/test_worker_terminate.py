"""``worker.terminate`` and ``worker._tgkill``, called directly.

`packages/pivot/tests/cli/test_worker_termination.py` drives the installed `pivot` binary and
shows what these do for a user; these tests call them with no pivot process in the way.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import signal
import subprocess
import sys
import threading

import pytest

from pivot.executor import worker

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="sends Linux thread-directed signals"
)


@pytest.mark.parametrize("path", ["tgkill", "os-kill"])
def test_terminate_sends_sigterm(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """``worker.terminate`` sends SIGTERM through the tgkill system call, or through
    ``os.kill`` where it knows no tgkill number for the ABI."""
    if path == "os-kill":
        monkeypatch.setattr(worker, "_tgkill", lambda: None)
    proc = subprocess.Popen(["sleep", "60"])
    try:
        worker.terminate(proc.pid)
        assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("path", ["tgkill", "os-kill"])
def test_terminate_raises_process_lookup_error_for_a_process_that_is_gone(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """A worker that has exited and been reaped raises ProcessLookupError, which
    ``WorkerPool._stop_workers`` ignores. The send goes to a pid this process still owns as a
    reaped child, so no signal can reach a process that reused the number."""
    if path == "os-kill":
        monkeypatch.setattr(worker, "_tgkill", lambda: None)
    proc = subprocess.Popen(["true"])
    proc.wait()

    with pytest.raises(ProcessLookupError):
        worker.terminate(proc.pid)


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EPERM])
def test_terminate_falls_back_to_os_kill_when_tgkill_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: int
) -> None:
    """A tgkill that fails for anything but a process that is gone (a sandbox that refuses the
    system call, ENOSYS; a policy that refuses the signal, EPERM) still stops the worker:
    ``terminate`` sends SIGTERM to the process and warns, naming the worker and the error."""

    def _failing(_tgid: int, _tid: int, _sig: int) -> int:
        ctypes.set_errno(error)
        return -1

    monkeypatch.setattr(worker, "_tgkill", lambda: _failing)
    proc = subprocess.Popen(["sleep", "60"])
    try:
        with caplog.at_level(logging.WARNING, logger=worker.logger.name):
            worker.terminate(proc.pid)
        assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        proc.kill()
        proc.wait()

    warnings = [r for r in caplog.records if "tgkill could not send SIGTERM" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert warnings[0].levelno == logging.WARNING
    assert str(proc.pid) in warnings[0].getMessage()
    assert os.strerror(error) in warnings[0].getMessage()


@pytest.mark.parametrize("abi", ["darwin", "x86_64-linux-gnux32", "mips64el-linux-gnuabi64", None])
def test_terminate_without_a_tgkill_number_uses_os_kill_and_logs_it_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, abi: str | None
) -> None:
    """An ABI not in ``_TGKILL_SYSCALLS`` (macOS's triplet, an x32 process on an x86-64
    machine, a Linux ABI not in the table, or an interpreter with no triplet) gets no tgkill
    number: ``terminate`` sends SIGTERM with ``os.kill``, and ``_tgkill`` says so once, at debug
    level, naming the triplet."""
    if abi is None:
        monkeypatch.delattr(sys.implementation, "_multiarch", raising=False)
    else:
        monkeypatch.setattr(sys.implementation, "_multiarch", abi, raising=False)
    worker._tgkill.cache_clear()
    try:
        with caplog.at_level(logging.DEBUG, logger=worker.logger.name):
            assert worker._tgkill() is None
            proc = subprocess.Popen(["sleep", "60"])
            try:
                worker.terminate(proc.pid)
                worker.terminate(proc.pid)
                assert proc.wait(timeout=10) == -signal.SIGTERM
            finally:
                proc.kill()
                proc.wait()
        logged = [r for r in caplog.records if "No tgkill system call number" in r.getMessage()]
        assert len(logged) == 1, [r.getMessage() for r in caplog.records]
        assert logged[0].levelno == logging.DEBUG
        assert str(abi) in logged[0].getMessage()
    finally:
        monkeypatch.undo()
        worker._tgkill.cache_clear()


def test_tgkill_system_call_number_fits_this_architecture() -> None:
    """The number ``worker._tgkill`` passes to ``syscall()`` for this process's ABI is tgkill's:
    signal 0 (a check that sends nothing) to this process's own main thread succeeds, and to a
    thread of another process fails with ESRCH. A wrong number runs another system call."""
    tgkill = worker._tgkill()
    abi = getattr(sys.implementation, "_multiarch", None)
    assert tgkill is not None, f"no tgkill system call number for {abi}"
    assert tgkill(os.getpid(), os.getpid(), 0) == 0, os.strerror(ctypes.get_errno())
    other = subprocess.Popen(["sleep", "60"])
    try:
        assert tgkill(os.getpid(), other.pid, 0) == -1
        assert ctypes.get_errno() == errno.ESRCH
    finally:
        other.kill()
        other.wait()


def test_tgkill_takes_the_thread_group_then_the_thread() -> None:
    """``_tgkill`` passes ``tgid`` then ``tid``: signal 0 to a thread of this process other than
    its main thread succeeds only in that order. ``terminate`` passes the process id as both, so
    it cannot tell the order."""
    tgkill = worker._tgkill()
    assert tgkill is not None
    ready, done = threading.Event(), threading.Event()
    native = list[int]()

    def _park() -> None:
        native.append(threading.get_native_id())
        ready.set()
        done.wait()

    thread = threading.Thread(target=_park)
    thread.start()
    try:
        assert ready.wait(10)
        assert tgkill(os.getpid(), native[0], 0) == 0, os.strerror(ctypes.get_errno())
    finally:
        done.set()
        thread.join()
