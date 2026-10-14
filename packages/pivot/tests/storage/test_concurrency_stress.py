"""Multi-process stress tests for state shared through one .pivot directory.

Processes start together on a barrier (no sleeps as synchronisation) and the assertions are on
final state only, so the tests are deterministic however the OS interleaves the processes.
"""

from __future__ import annotations

import fcntl
import multiprocessing
import os
from typing import TYPE_CHECKING

import pytest

from pivot.storage import cache, state
from pivot.types import is_dir_hash

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable
    from multiprocessing.synchronize import Barrier

_PROCESSES = 6
_INCREMENTS = 40
_FILES_PER_PROCESS = 25


def _hammer_state_dir(state_dir: pathlib.Path, worker: int, barrier: Barrier) -> None:
    """Increment one shared counter and record per-worker file hashes, interleaved with reads."""
    data_dir = state_dir.parent / f"data_{worker}"
    data_dir.mkdir()
    barrier.wait(timeout=60)
    with state.StateDB(state_dir) as db:
        for i in range(_INCREMENTS):
            db.increment_generation("shared/stage")
            if i < _FILES_PER_PROCESS:
                path = data_dir / f"f{i}.txt"
                path.write_text(f"{worker}-{i}")
                db.save(path, path.stat(), f"{worker:08x}{i:08x}")
            with state.StateDB(state_dir, readonly=True) as reader:
                assert reader.get_generation("shared/stage") is not None


def _run_processes(
    target: Callable[..., None], args_for: list[tuple[object, ...]]
) -> list[int | None]:
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=target, args=args) for args in args_for]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    return [process.exitcode for process in processes]


def test_concurrent_writers_lose_no_generation_increments_or_hash_entries(
    tmp_path: pathlib.Path,
) -> None:
    state_dir = tmp_path / ".pivot"
    state.StateDB(state_dir).close()
    barrier = multiprocessing.get_context("spawn").Barrier(_PROCESSES)

    exit_codes = _run_processes(
        _hammer_state_dir, [(state_dir, worker, barrier) for worker in range(_PROCESSES)]
    )

    assert exit_codes == [0] * _PROCESSES, f"a writer process failed: {exit_codes}"
    with state.StateDB(state_dir, readonly=True) as db:
        assert db.get_generation("shared/stage") == _PROCESSES * _INCREMENTS, (
            "every increment from every process must be counted exactly once"
        )
        for worker in range(_PROCESSES):
            for i in range(_FILES_PER_PROCESS):
                path = tmp_path / f"data_{worker}" / f"f{i}.txt"
                assert db.get(path, path.stat()) == f"{worker:08x}{i:08x}"


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 8 (storage-fixes Slice 2 PR): "
    + "directory restore lock is unlinked while a waiter holds it",
)
def test_directory_restore_keeps_the_lock_file_a_waiter_holds(tmp_path: pathlib.Path) -> None:
    """Restores of one directory exclude each other through one lock file.

    A second restorer waiting on that lock holds an fd to the lock file. If the first restorer
    unlinks the file when it finishes, a third restorer creates a new file and locks it while
    the waiter locks the old one, and both restore the same directory at once.
    """
    source = tmp_path / "out"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_text("a")
    (source / "nested" / "deep.txt").write_text("deep")
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    output_hash = cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(output_hash)
    cache.remove_output(source)
    lock_path = cache._get_lock_path(source, state_dir)

    waiter = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        assert cache.restore_from_cache(
            source,
            output_hash,
            cache_dir,
            checkout_modes=[cache.CheckoutMode.COPY],
            state_dir=state_dir,
        )
        assert (source / "nested" / "deep.txt").read_text() == "deep"

        assert lock_path.exists(), "the restore deleted the lock file the waiter holds"
        assert lock_path.stat().st_ino == os.fstat(waiter).st_ino
        fcntl.flock(waiter, fcntl.LOCK_EX | fcntl.LOCK_NB)
        newcomer = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(newcomer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(newcomer)
    finally:
        os.close(waiter)
