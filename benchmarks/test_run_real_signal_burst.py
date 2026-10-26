"""A second stop signal lets a real pivot finish its own SIGINT cleanup."""

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

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_RUN_PY = pathlib.Path(run.__file__)


def _helper_repro_pid(harness: subprocess.Popen[bytes]) -> int:
    deadline = time.monotonic() + 60
    children = pathlib.Path(f"/proc/{harness.pid}/task/{harness.pid}/children")
    while time.monotonic() < deadline:
        for raw_pid in children.read_text().split():
            args = pathlib.Path(f"/proc/{raw_pid}/cmdline").read_bytes().split(b"\0")
            if b"repro" in args:
                return int(raw_pid)
        time.sleep(0.05)
    raise AssertionError("run.py never started pivot repro")


def _helper_wait_for_members(sid: int) -> list[int]:
    deadline = time.monotonic() + 60
    members = run.session_members(sid)
    while len(members) < 4 and time.monotonic() < deadline:
        time.sleep(0.05)
        members = run.session_members(sid)
    return members


def _helper_session(pid: int) -> int:
    return int(pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[3])


def _helper_wait_for_stop(harness: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 10
    status = pathlib.Path(f"/proc/{harness.pid}/stat")
    while time.monotonic() < deadline:
        if status.read_text().rsplit(")", 1)[1].split()[0] == "T":
            return
        time.sleep(0.01)
    raise AssertionError("run.py did not stop")


def _helper_semaphores(sid: int) -> set[pathlib.Path]:
    return set(pathlib.Path("/dev/shm").glob(f"sem.loky-{sid}-*"))


def _helper_wait_for_cleanup(
    sid: int, before: set[pathlib.Path]
) -> tuple[list[int], set[pathlib.Path]]:
    deadline = time.monotonic() + 10
    left = run.session_members(sid)
    semaphores = _helper_semaphores(sid) - before
    while (left or semaphores) and time.monotonic() < deadline:
        time.sleep(0.05)
        left = run.session_members(sid)
        semaphores = _helper_semaphores(sid) - before
    return left, semaphores


@pytest.mark.parametrize(
    ("first", "second", "delay_s"),
    [
        pytest.param(
            first,
            second,
            delay_ms / 1_000,
            id=f"{first.name[3:].lower()}-{second.name[3:].lower()}-{delay_ms}ms",
        )
        for first, second in (
            (signal.SIGINT, signal.SIGINT),
            (signal.SIGINT, signal.SIGTERM),
            (signal.SIGTERM, signal.SIGINT),
        )
        for delay_ms in (2, 5, 20)
    ],
)
def test_real_pivot_second_signal_keeps_graceful_cleanup(
    tmp_path: pathlib.Path, first: signal.Signals, second: signal.Signals, delay_s: float
) -> None:
    argv = [sys.executable, str(_RUN_PY), "--project", "chain-100", "--variant", f"a={_PIVOT}"]
    argv += ["--reps", "1", "--work", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out")]
    before = set(pathlib.Path("/dev/shm").glob("sem.loky-*"))
    harness = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sid = 0
    try:
        sid = _helper_session(_helper_repro_pid(harness))
        assert len(_helper_wait_for_members(sid)) >= 4, "pivot did not start its worker session"
        harness.send_signal(first)
        time.sleep(delay_s)
        with contextlib.suppress(ProcessLookupError):
            harness.send_signal(second)

        exit_code = harness.wait(timeout=30)
        left, semaphores = _helper_wait_for_cleanup(sid, before)
        assert exit_code in (-first, 128 + first, -second, 128 + second), exit_code
        assert not left, f"measured session still has live processes: {left}"
        assert not semaphores, f"pivot leaked semaphore files: {sorted(semaphores)}"
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait()
        run_test_helpers.kill_session(sid)
        for path in _helper_semaphores(sid) - before:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


def test_real_pivot_pending_stop_signals_keep_graceful_cleanup(tmp_path: pathlib.Path) -> None:
    argv = [sys.executable, str(_RUN_PY), "--project", "chain-100", "--variant", f"a={_PIVOT}"]
    argv += ["--reps", "1", "--work", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out")]
    before = set(pathlib.Path("/dev/shm").glob("sem.loky-*"))
    harness = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sid = 0
    try:
        sid = _helper_session(_helper_repro_pid(harness))
        assert len(_helper_wait_for_members(sid)) >= 4, "pivot did not start its worker session"
        harness.send_signal(signal.SIGSTOP)
        _helper_wait_for_stop(harness)
        harness.send_signal(signal.SIGINT)
        harness.send_signal(signal.SIGTERM)
        harness.send_signal(signal.SIGCONT)

        exit_code = harness.wait(timeout=30)
        left, semaphores = _helper_wait_for_cleanup(sid, before)
        assert exit_code in (
            -signal.SIGINT,
            128 + signal.SIGINT,
            -signal.SIGTERM,
            128 + signal.SIGTERM,
        ), exit_code
        assert not left, f"measured session still has live processes: {left}"
        assert not semaphores, f"pivot leaked semaphore files: {sorted(semaphores)}"
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait()
        run_test_helpers.kill_session(sid)
        for path in _helper_semaphores(sid) - before:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
