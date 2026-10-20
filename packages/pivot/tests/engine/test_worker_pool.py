# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import concurrent.futures
import errno
import os
import pathlib
import signal
import threading
import time
from typing import TYPE_CHECKING, cast

import loky.reusable_executor
import pytest

from pivot.engine import worker_pool
from pivot.engine.worker_pool import WorkerPool
from pivot.executor import core as executor_core

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager  # noqa: F401
    from multiprocessing.process import BaseProcess  # noqa: F401


def _helper_identity(value: int) -> int:
    return value


def _helper_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _helper_pid() -> int:
    return os.getpid()


def _helper_wait_for_release(ready_dir: str, release: str) -> int:
    ready_path = pathlib.Path(ready_dir) / str(os.getpid())
    ready_path.write_text("")
    release_path = pathlib.Path(release)
    deadline = time.monotonic() + 60
    while not release_path.exists():
        assert time.monotonic() < deadline, "test did not release waiting workers"
        time.sleep(0.01)
    return os.getpid()


def _helper_process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_unknown_loky_reusable_executor_layout_names_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(loky.reusable_executor, "_executor", raising=False)
    monkeypatch.delattr(loky.reusable_executor, "_executor_storage", raising=False)

    with pytest.raises(RuntimeError, match=r"Unsupported loky \d"):
        _ = executor_core._reusable_worker_target(1, kill_workers=False)


def test_short_parents_schedule_no_timer_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    started = list[object]()

    class _Timer:
        daemon: bool

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            started.append(self)
            self.daemon = False

        def start(self) -> None:
            pass

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(threading, "Timer", _Timer)
    pool = WorkerPool(_idle_timeout_s=10)
    for stage in range(100):
        name = str(stage)
        pool.keep_consumers_warm(name, 1)
        pool.release_consumers(name)

    assert started == []


def test_keepalive_completion_and_cancellation_leave_no_pending_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Executor:
        _processes: dict[int, object] = {1: object()}
        _pending_work_items: dict[int, object] = {}

    class _Manager:
        def Barrier(self, _parties: int) -> object:  # noqa: N802
            return object()

    submitted = list[concurrent.futures.Future[None]]()

    def submit_ping(*_args: object) -> concurrent.futures.Future[None]:
        future = concurrent.futures.Future[None]()
        submitted.append(future)
        return future

    monkeypatch.setattr(executor_core, "submit_to_live_workers", submit_ping)
    pool = WorkerPool(
        _executor=cast("loky.ProcessPoolExecutor", cast("object", _Executor())),
        _manager=cast("SyncManager", cast("object", _Manager())),
        _idle_timeout_s=10,
    )

    pool.keep_consumers_warm("completed", 1)
    pool._keepalives["completed"] = (0, 1)
    pool.run_due_keepalives()
    submitted[-1].set_result(None)
    pool.release_consumers("completed")
    assert pool._pings == {}

    pool.keep_consumers_warm("cancelled", 1)
    pool._keepalives["cancelled"] = (0, 1)
    pool.run_due_keepalives()
    pool._stop_keepalives()
    assert submitted[-1].cancelled()
    assert pool._pings == {}


@pytest.mark.slow
def test_idle_workers_exit_while_another_stage_runs(tmp_path: pathlib.Path) -> None:
    idle_timeout_s = executor_core.one_shot_worker_idle_timeout()
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    release = tmp_path / "release"
    pool = WorkerPool()
    try:
        pool.start(max_workers=4, idle_timeout_s=idle_timeout_s)
        slow = pool.submit(_helper_sleep, idle_timeout_s + 5)
        waiting = [
            pool.submit(_helper_wait_for_release, str(ready_dir), str(release)) for _ in range(3)
        ]
        deadline = time.monotonic() + 60
        while len(list(ready_dir.iterdir())) < len(waiting):
            assert time.monotonic() < deadline, "workers did not start"
            time.sleep(0.01)
        release.write_text("")
        idle_pids = {future.result(timeout=60) for future in waiting}

        deadline = time.monotonic() + idle_timeout_s + 5
        while any(_helper_process_is_alive(pid) for pid in idle_pids):
            assert not slow.done(), "the slow stage ended before idle workers exited"
            assert time.monotonic() < deadline, "idle workers stayed alive"
            time.sleep(0.01)
    finally:
        pool.hard_cancel()


def test_restart_workers_replaces_live_workers() -> None:
    """A code reload must replace workers rather than returning loky's live singleton."""
    executor_core.restart_workers(1, idle_timeout_s=None)
    first = executor_core.create_executor(1).submit(_helper_pid).result(timeout=60)

    executor_core.restart_workers(1, idle_timeout_s=None)
    second = executor_core.create_executor(1).submit(_helper_pid).result(timeout=60)

    assert second != first, "restart_workers should replace every live worker"
    executor_core._cleanup_worker_pool()


@pytest.mark.slow
def test_warm_workers_survive_lokys_default_idle_timeout() -> None:
    """Long-lived workers stay warm after a pause longer than loky's default timeout."""
    executor_core.restart_workers(1, idle_timeout_s=None)
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
