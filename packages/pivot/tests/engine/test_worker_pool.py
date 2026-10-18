# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

from pivot.engine.worker_pool import WorkerPool
from pivot.executor import core as executor_core

if TYPE_CHECKING:
    import pathlib


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


def test_source_stamped_within_clock_granularity_of_dispatch_counts_as_changed(
    tmp_path: pathlib.Path,
) -> None:
    """A coarse file clock can stamp an edit slightly before an earlier coordinator clock read."""
    source = tmp_path / "helper.py"
    source.write_text("x = 1\n")
    changed_ns = source.stat().st_ctime_ns

    recent = WorkerPool()
    recent._run_started_ns = changed_ns + 5_000_000
    recent.record_user_code_files([str(source)], changed_ns + 5_000_000)
    assert recent._code_changed, "an edit 5 ms before dispatch may postdate the worker's import"

    old = WorkerPool()
    old._run_started_ns = changed_ns + 1_000_000_000
    old.record_user_code_files([str(source)], changed_ns + 1_000_000_000)
    assert not old._code_changed, "a file unchanged for a second before dispatch is current"
