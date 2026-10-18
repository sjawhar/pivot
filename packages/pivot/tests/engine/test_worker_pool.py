# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import errno
import os
import signal
import time
from typing import TYPE_CHECKING, cast

import pytest

from pivot.engine import worker_pool
from pivot.engine.worker_pool import WorkerPool
from pivot.executor import core as executor_core

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess


def _helper_identity(value: int) -> int:
    return value


def _helper_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _helper_pid() -> int:
    return os.getpid()


def test_restart_workers_replaces_live_workers() -> None:
    """A code reload must replace workers rather than returning loky's live singleton."""
    executor_core.restart_workers(1, 1)
    first = executor_core.create_executor(1).submit(_helper_pid).result(timeout=60)

    executor_core.restart_workers(1, 1)
    second = executor_core.create_executor(1).submit(_helper_pid).result(timeout=60)

    assert second != first, "restart_workers should replace every live worker"
    executor_core._cleanup_worker_pool()


@pytest.mark.slow
def test_warm_workers_survive_lokys_default_idle_timeout() -> None:
    """Interactive runs stay warm even when a user pauses longer than loky's 10 s default."""
    executor_core.restart_workers(1, 1)
    executor = executor_core.create_executor(1)
    first = executor.submit(_helper_pid).result(timeout=60)

    time.sleep(11)

    second = executor.submit(_helper_pid).result(timeout=60)
    assert second == first, "the idle worker should remain ready until Engine shutdown"
    executor_core._cleanup_worker_pool()


def test_worker_pool_stop_accepting_rejects_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    future = pool.submit(_helper_identity, 1)
    # The first result includes spawning a fresh worker process (several seconds under load).
    assert future.result(timeout=60) == 1

    pool.stop_accepting()
    with pytest.raises(RuntimeError, match="accepting"):
        pool.submit(_helper_identity, 2)

    pool.shutdown()


def test_worker_pool_hard_cancel_blocks_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    pool.submit(_helper_sleep, 5)
    pool.hard_cancel()

    with pytest.raises(RuntimeError, match="accepting"):
        pool.submit(_helper_identity, 3)


def test_worker_pool_submit_requires_start() -> None:
    pool = WorkerPool()

    with pytest.raises(RuntimeError, match="not started"):
        pool.submit(_helper_identity, 1)


def test_worker_pool_output_queue_requires_start() -> None:
    pool = WorkerPool()

    with pytest.raises(RuntimeError, match="not started"):
        pool.output_queue()


def test_file_stamp_propagates_descriptor_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running out of descriptors must not read as an unreadable (changed) user-code file."""

    def _no_descriptors(*_args: object, **_kwargs: object) -> os.stat_result:
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(os, "stat", _no_descriptors)
    with pytest.raises(OSError, match="Too many open files"):
        worker_pool._file_stamp(__file__)


def test_failed_manager_replacement_does_not_poison_later_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose new manager fails to start leaves the pool able to start the next run."""
    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        pool.end_run()
        manager = pool._manager
        assert manager is not None
        process = cast("BaseProcess", manager._process)  # pyright: ignore[reportAttributeAccessIssue] - not in typeshed
        assert process.pid is not None
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=10)

        real_create = executor_core.create_manager
        calls = list[int]()

        def _fail_once() -> object:
            calls.append(1)
            if len(calls) == 1:
                raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
            return real_create()

        monkeypatch.setattr(executor_core, "create_manager", _fail_once)
        with pytest.raises(OSError, match="Resource temporarily unavailable"):
            pool.start(max_workers=1)

        pool.start(max_workers=1)
        pool.output_queue().put_nowait(None)
        assert pool.output_queue().get(timeout=10) is None
    finally:
        pool.shutdown()
