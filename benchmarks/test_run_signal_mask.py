"""An interrupted in-process measurement restores the caller's stop-signal mask."""

from __future__ import annotations

import signal
import subprocess
import typing
from typing import TYPE_CHECKING

import pytest

import run

if TYPE_CHECKING:
    import pathlib


def _helper_mask() -> frozenset[int]:
    return frozenset(signal.pthread_sigmask(signal.SIG_BLOCK, set[signal.Signals]()))


def test_measure_restores_the_mask_after_wait_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _helper_mask()
    wait = typing.cast(
        "typing.Callable[[subprocess.Popen[bytes], float | None], int]", subprocess.Popen.wait
    )
    calls = 0

    def interrupted_wait(self: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        nonlocal calls
        if calls == 0:
            calls += 1
            raise RuntimeError("injected wait exception")
        return wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted_wait)
    try:
        with pytest.raises(RuntimeError, match="injected wait exception"):
            run.measure(["sh", "-c", "sleep 60"], tmp_path)
        after = _helper_mask()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)

    assert after == before


def test_measure_restores_the_mask_after_a_real_sigint(tmp_path: pathlib.Path) -> None:
    before = _helper_mask()
    handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            run.measure(["sh", "-c", "sleep 0.1; kill -INT $PPID; sleep 60"], tmp_path)
        after = _helper_mask()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)
        signal.signal(signal.SIGINT, handler)

    assert after == before
