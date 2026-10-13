"""Multi-process stress tests for state shared through one .pivot directory.

Processes start together on a barrier (no sleeps as synchronisation) and the assertions are on
final state only, so the tests are deterministic however the OS interleaves the processes.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING

import pytest

from pivot.storage import cache, state
from pivot.types import is_dir_hash

if TYPE_CHECKING:
    import pathlib
    from multiprocessing.synchronize import Barrier

    from pivot.types import DirHash

_PROCESSES = 6
_INCREMENTS = 40
_FILES_PER_PROCESS = 25
_RESTORES_PER_PROCESS = 30


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


def _run_processes(target: object, args_for: list[tuple[object, ...]]) -> list[int | None]:
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=target, args=args) for args in args_for]  # pyright: ignore[reportArgumentType]
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


def _restore_directory(
    target: pathlib.Path,
    output_hash: DirHash,
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    barrier: Barrier,
) -> None:
    barrier.wait(timeout=60)
    for _ in range(_RESTORES_PER_PROCESS):
        assert cache.restore_from_cache(
            target,
            output_hash,
            cache_dir,
            checkout_modes=[cache.CheckoutMode.COPY],
            state_dir=state_dir,
        )


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 8 (#461): restore lock file is unlinked after release "
    + "(cache.py:738-739), so two restores hold 'the' lock on different inodes and the "
    + "directory swap fails with Errno 39",
)
def test_concurrent_directory_restores_leave_one_complete_tree_and_no_temps(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "out"
    (source / "nested").mkdir(parents=True)
    expected = {f"f{i}.txt": f"content {i}" for i in range(20)}
    expected["nested/deep.txt"] = "deep"
    for rel, content in expected.items():
        (source / rel).write_text(content)
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    state_dir.mkdir()
    output_hash = cache.save_to_cache(source, cache_dir, checkout_mode=cache.CheckoutMode.COPY)
    assert is_dir_hash(output_hash)
    cache.remove_output(source)
    barrier = multiprocessing.get_context("spawn").Barrier(_PROCESSES)

    exit_codes = _run_processes(
        _restore_directory,
        [(source, output_hash, cache_dir, state_dir, barrier) for _ in range(_PROCESSES)],
    )

    assert exit_codes == [0] * _PROCESSES, f"a restore process failed: {exit_codes}"
    restored = {
        str(path.relative_to(source)): path.read_text()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert restored == expected, "the restored tree must be complete and contain nothing else"
    leftovers = [
        name
        for name in os.listdir(tmp_path)
        if name.startswith((cache._RESTORE_TEMP_PREFIX, cache._BACKUP_TEMP_PREFIX))
    ]
    assert leftovers == [], f"restore temp/backup dirs left behind: {leftovers}"
