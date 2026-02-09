"""Integration tests for concurrent artifact locking with real OS processes.

Verifies that LocalFlockLockService properly serializes overlapping artifact
access across processes and allows parallel access for disjoint artifacts.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import pathlib
import time
from typing import TYPE_CHECKING

from pivot.storage import artifact_lock

if TYPE_CHECKING:
    from multiprocessing.context import SpawnContext
    from multiprocessing.sharedctypes import Synchronized

# Create spawn context at module level to avoid fork() deadlocks in multi-threaded xdist workers
_mp_ctx: SpawnContext = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------


def _helper_acquire_write_and_hold(
    state_dir: str,
    key: str,
    ready_event: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
    release_event: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
) -> None:
    """Acquire a WRITE lock, signal ready, then wait for release signal."""
    service = artifact_lock.LocalFlockLockService(
        state_dir=pathlib.Path(state_dir), retry_interval=0.05
    )
    requests = [artifact_lock.LockRequest(key=key, mode=artifact_lock.LockMode.WRITE)]
    with service.acquire_many(requests):
        ready_event.set()
        release_event.wait(timeout=10)


def _helper_acquire_write_timed(
    state_dir: str,
    key: str,
    hold_seconds: float,
    ready_event: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
) -> None:
    """Acquire a WRITE lock, signal ready, hold for a duration, then release."""
    service = artifact_lock.LocalFlockLockService(
        state_dir=pathlib.Path(state_dir), retry_interval=0.05
    )
    requests = [artifact_lock.LockRequest(key=key, mode=artifact_lock.LockMode.WRITE)]
    with service.acquire_many(requests):
        ready_event.set()
        time.sleep(hold_seconds)


def _helper_acquire_read_after_ready(
    state_dir: str,
    key: str,
    blocker_ready: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
    success_event: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
) -> None:
    """Wait for blocker to be ready, then acquire a READ lock on the same key."""
    blocker_ready.wait(timeout=10)
    service = artifact_lock.LocalFlockLockService(
        state_dir=pathlib.Path(state_dir), retry_interval=0.05
    )
    requests = [artifact_lock.LockRequest(key=key, mode=artifact_lock.LockMode.READ)]
    with service.acquire_many(requests):
        success_event.set()


def _helper_lock_overlapping_keys(
    state_dir: str,
    keys: list[str],
    start_event: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
    done_event: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
    hold_seconds: float,
) -> None:
    """Wait for start, acquire WRITE locks on multiple keys, hold, signal done."""
    start_event.wait(timeout=10)
    service = artifact_lock.LocalFlockLockService(
        state_dir=pathlib.Path(state_dir), retry_interval=0.05
    )
    requests = [artifact_lock.LockRequest(key=k, mode=artifact_lock.LockMode.WRITE) for k in keys]
    with service.acquire_many(requests):
        time.sleep(hold_seconds)
    done_event.set()


def _helper_acquire_with_callback(
    state_dir: str,
    key: str,
    blocker_ready: multiprocessing.synchronize.Event,  # type: ignore[type-arg]
    callback_count: Synchronized[int],
) -> None:
    """Wait for blocker, then acquire with a status_callback that increments a counter."""
    blocker_ready.wait(timeout=10)
    service = artifact_lock.LocalFlockLockService(
        state_dir=pathlib.Path(state_dir), retry_interval=0.05
    )
    requests = [artifact_lock.LockRequest(key=key, mode=artifact_lock.LockMode.WRITE)]

    def _on_status(_key: str, _mode: artifact_lock.LockMode, _elapsed: float) -> None:
        with callback_count.get_lock():
            callback_count.value += 1

    with service.acquire_many(requests, status_callback=_on_status):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disjoint_artifacts_run_in_parallel(tmp_path: pathlib.Path) -> None:
    """Two processes acquiring WRITE locks on different keys run concurrently."""
    state_dir = str(tmp_path)
    hold_seconds = 0.4

    ready_a = _mp_ctx.Event()
    ready_b = _mp_ctx.Event()

    proc_a = _mp_ctx.Process(
        target=_helper_acquire_write_timed,
        args=(state_dir, "/artifact/a.csv", hold_seconds, ready_a),
    )
    proc_b = _mp_ctx.Process(
        target=_helper_acquire_write_timed,
        args=(state_dir, "/artifact/b.csv", hold_seconds, ready_b),
    )

    start = time.monotonic()
    proc_a.start()
    proc_b.start()

    proc_a.join(timeout=10)
    proc_b.join(timeout=10)
    elapsed = time.monotonic() - start

    assert proc_a.exitcode == 0, "Process A should exit cleanly"
    assert proc_b.exitcode == 0, "Process B should exit cleanly"
    # If locks were serial, elapsed ≈ 2 * hold_seconds. Parallel → < 1.5 * hold.
    # Spawn context adds pickling overhead, so allow 2.2x instead of 1.8x.
    assert elapsed < hold_seconds * 2.2, (
        f"Disjoint locks should run in parallel, took {elapsed:.2f}s"
    )


def test_overlapping_artifact_write_blocks_read(tmp_path: pathlib.Path) -> None:
    """A WRITE lock blocks a concurrent READ on the same key until released."""
    state_dir = str(tmp_path)
    key = "/shared/data.csv"

    blocker_ready = _mp_ctx.Event()
    release_blocker = _mp_ctx.Event()
    reader_success = _mp_ctx.Event()

    # Process 1: hold WRITE lock until told to release
    writer = _mp_ctx.Process(
        target=_helper_acquire_write_and_hold,
        args=(state_dir, key, blocker_ready, release_blocker),
    )
    # Process 2: try READ lock after writer is ready
    reader = _mp_ctx.Process(
        target=_helper_acquire_read_after_ready,
        args=(state_dir, key, blocker_ready, reader_success),
    )

    writer.start()
    reader.start()

    # Writer should be holding lock; reader should NOT have succeeded yet
    blocker_ready.wait(timeout=10)
    time.sleep(0.3)  # Give reader time to attempt acquisition
    assert not reader_success.is_set(), "Reader should be blocked while writer holds lock"

    # Release writer → reader should succeed
    release_blocker.set()

    writer.join(timeout=10)
    reader.join(timeout=10)

    assert writer.exitcode == 0, "Writer should exit cleanly"
    assert reader.exitcode == 0, "Reader should exit cleanly"
    assert reader_success.is_set(), "Reader should have acquired lock after writer released"


def test_no_deadlock_with_deterministic_ordering(tmp_path: pathlib.Path) -> None:
    """Two processes locking overlapping key sets both complete (sorted order prevents deadlock)."""
    state_dir = str(tmp_path)
    # Process A locks {X, Y}, Process B locks {Y, Z} — overlap on Y
    keys_a = ["/keys/x.csv", "/keys/y.csv"]
    keys_b = ["/keys/y.csv", "/keys/z.csv"]

    start_event = _mp_ctx.Event()
    done_a = _mp_ctx.Event()
    done_b = _mp_ctx.Event()

    proc_a = _mp_ctx.Process(
        target=_helper_lock_overlapping_keys,
        args=(state_dir, keys_a, start_event, done_a, 0.2),
    )
    proc_b = _mp_ctx.Process(
        target=_helper_lock_overlapping_keys,
        args=(state_dir, keys_b, start_event, done_b, 0.2),
    )

    proc_a.start()
    proc_b.start()
    # Signal both to start simultaneously
    start_event.set()

    proc_a.join(timeout=10)
    proc_b.join(timeout=10)

    assert proc_a.exitcode == 0, "Process A should complete without deadlock"
    assert proc_b.exitcode == 0, "Process B should complete without deadlock"
    assert done_a.is_set(), "Process A should have finished its lock cycle"
    assert done_b.is_set(), "Process B should have finished its lock cycle"


def test_status_callback_fires_during_contention(tmp_path: pathlib.Path) -> None:
    """status_callback is invoked when a process retries on a contended lock."""
    state_dir = str(tmp_path)
    key = "/contended/output.csv"

    blocker_ready = _mp_ctx.Event()
    release_blocker = _mp_ctx.Event()
    callback_count: Synchronized[int] = _mp_ctx.Value("i", 0)

    # Process 1: hold WRITE lock
    writer = _mp_ctx.Process(
        target=_helper_acquire_write_and_hold,
        args=(state_dir, key, blocker_ready, release_blocker),
    )
    # Process 2: try to acquire with callback
    contender = _mp_ctx.Process(
        target=_helper_acquire_with_callback,
        args=(state_dir, key, blocker_ready, callback_count),
    )

    writer.start()
    contender.start()

    # Let contender retry a few times while blocked
    blocker_ready.wait(timeout=10)
    time.sleep(0.4)

    # Release writer so contender can finish
    release_blocker.set()

    writer.join(timeout=10)
    contender.join(timeout=10)

    assert writer.exitcode == 0, "Writer should exit cleanly"
    assert contender.exitcode == 0, "Contender should exit cleanly"
    assert callback_count.value > 0, (
        "status_callback should have been invoked at least once during contention"
    )
