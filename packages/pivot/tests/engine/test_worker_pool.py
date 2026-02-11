# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import time

import pytest

from pivot.engine.worker_pool import WorkerPool


def _helper_identity(value: int) -> int:
    return value


def _helper_sleep(seconds: float) -> None:
    time.sleep(seconds)


def test_worker_pool_stop_accepting_rejects_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    future = pool.submit(_helper_identity, 1)
    assert future.result(timeout=5) == 1

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
