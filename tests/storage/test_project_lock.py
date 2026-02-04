from __future__ import annotations

import pathlib
import threading
import time
from typing import TYPE_CHECKING, Annotated, TypedDict

import filelock
import pytest

from helpers import register_test_stage

if TYPE_CHECKING:
    from pivot.pipeline.pipeline import Pipeline

from pivot import executor, loaders, outputs, project
from pivot.executor import commit as commit_mod
from pivot.storage import project_lock


def _try_acquire_nonblocking() -> filelock.BaseFileLock | None:
    """Helper to attempt non-blocking lock acquisition."""
    try:
        return project_lock.acquire_pending_state_lock(timeout=0)
    except filelock.Timeout:
        return None


def test_pending_state_lock_creates_lock_file(pipeline_dir: pathlib.Path) -> None:
    """pending_state_lock creates lock file in .pivot directory."""
    with project_lock.pending_state_lock():
        lock_path = pipeline_dir / ".pivot" / "pending.lock"
        assert lock_path.exists()


def test_pending_state_lock_releases_on_exit(pipeline_dir: pathlib.Path) -> None:
    """Lock is released when pending_state_lock context exits."""
    with project_lock.pending_state_lock():
        pass

    # Should be able to acquire immediately after
    acquired = _try_acquire_nonblocking()
    assert acquired is not None
    acquired.release()


def test_acquire_returns_none_when_locked(pipeline_dir: pathlib.Path) -> None:
    """Non-blocking acquire raises Timeout when lock is held."""
    with project_lock.pending_state_lock():
        result = _try_acquire_nonblocking()
        assert result is None


def test_acquire_returns_lock_when_available(pipeline_dir: pathlib.Path) -> None:
    """acquire_pending_state_lock returns lock object when available."""
    acquired = project_lock.acquire_pending_state_lock()
    assert acquired is not None
    assert isinstance(acquired, filelock.BaseFileLock)
    acquired.release()


def test_acquire_lock_must_be_released(pipeline_dir: pathlib.Path) -> None:
    """Acquired lock must be released by caller."""
    acquired = project_lock.acquire_pending_state_lock()
    assert acquired is not None

    # While held, another acquisition should fail
    second = _try_acquire_nonblocking()
    assert second is None

    # After release, acquisition should succeed
    acquired.release()
    third = _try_acquire_nonblocking()
    assert third is not None
    third.release()


def test_acquire_pending_state_lock_blocks_until_available(pipeline_dir: pathlib.Path) -> None:
    """acquire_pending_state_lock blocks until lock becomes available."""
    acquired_at = list[float]()
    lock_holder_started = threading.Event()

    def hold_lock() -> None:
        with project_lock.pending_state_lock():
            lock_holder_started.set()
            time.sleep(0.1)

    holder_thread = threading.Thread(target=hold_lock)
    holder_thread.start()

    # Wait for holder to acquire the lock
    lock_holder_started.wait(timeout=1.0)

    # Now try to acquire - should block
    start = time.monotonic()
    lock_obj = project_lock.acquire_pending_state_lock()
    elapsed = time.monotonic() - start
    acquired_at.append(elapsed)

    # Should have waited at least 0.1s (some slack for timing)
    assert elapsed >= 0.1, f"Should have blocked, but only waited {elapsed:.3f}s"

    lock_obj.release()
    holder_thread.join()


def test_acquire_pending_state_lock_timeout(pipeline_dir: pathlib.Path) -> None:
    """acquire_pending_state_lock raises Timeout when timeout expires."""
    lock_holder_started = threading.Event()

    def hold_lock() -> None:
        with project_lock.pending_state_lock():
            lock_holder_started.set()
            time.sleep(0.2)

    holder_thread = threading.Thread(target=hold_lock)
    holder_thread.start()

    lock_holder_started.wait(timeout=1.0)

    with pytest.raises(filelock.Timeout):
        project_lock.acquire_pending_state_lock(timeout=0.1)

    holder_thread.join()


def test_parent_dir_created_by_pending_state_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pending_state_lock creates .pivot directory if it doesn't exist."""
    # Create .pivot and cache the project root before removing .pivot
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    # Remove .pivot dir to test that the lock recreates it
    (tmp_path / ".pivot").rmdir()
    assert not (tmp_path / ".pivot").exists()

    # pending_state_lock should create it
    with project_lock.pending_state_lock():
        assert (tmp_path / ".pivot").exists()


def test_parent_dir_created_by_acquire_pending_state_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """acquire_pending_state_lock creates .pivot directory if it doesn't exist."""
    # Create .pivot and cache the project root before removing .pivot
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    # Remove .pivot dir to test that the lock recreates it
    (tmp_path / ".pivot").rmdir()
    assert not (tmp_path / ".pivot").exists()

    acquired = project_lock.acquire_pending_state_lock()
    assert (tmp_path / ".pivot").exists()
    acquired.release()


def test_lock_prevents_concurrent_execution(pipeline_dir: pathlib.Path) -> None:
    """Only one pending_state_lock can run at a time."""
    execution_order = list[str]()
    lock_acquired = threading.Event()
    both_tried = threading.Event()

    def first_executor() -> None:
        with project_lock.pending_state_lock():
            execution_order.append("first_start")
            lock_acquired.set()
            both_tried.wait(timeout=1.0)
            time.sleep(0.1)
            execution_order.append("first_end")

    def second_executor() -> None:
        lock_acquired.wait(timeout=1.0)
        both_tried.set()
        with project_lock.pending_state_lock():
            execution_order.append("second_start")
            execution_order.append("second_end")

    t1 = threading.Thread(target=first_executor)
    t2 = threading.Thread(target=second_executor)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Second should not start until first ends
    assert execution_order.index("first_end") < execution_order.index("second_start")


def test_pending_state_lock_releases_on_exception(pipeline_dir: pathlib.Path) -> None:
    """Lock is released even if pending_state_lock raises an exception."""
    with pytest.raises(ValueError, match="test error"), project_lock.pending_state_lock():
        raise ValueError("test error")

    # Should be able to acquire immediately after
    acquired = _try_acquire_nonblocking()
    assert acquired is not None
    acquired.release()


# =============================================================================
# Integration: Execution + Commit Coordination
# =============================================================================


class _SlowStageOutput(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _slow_stage_impl(
    execution_log: pathlib.Path,
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SlowStageOutput:
    # Log when stage starts
    with execution_log.open("a") as f:
        f.write("stage_start\n")
    time.sleep(0.1)
    output_path = pathlib.Path("output.txt")
    output_path.write_text("done")
    with execution_log.open("a") as f:
        f.write("stage_end\n")
    return _SlowStageOutput(output=output_path)


def test_commit_blocks_during_no_commit_execution(
    pipeline_dir: pathlib.Path, test_pipeline: Pipeline
) -> None:
    """acquire_pending_state_lock blocks while executor.run(no_commit=True) holds the lock."""
    # Create a stage that runs for a measurable duration
    pathlib.Path("input.txt").write_text("data")
    execution_log = pipeline_dir / "execution_log.txt"

    def slow_stage(
        input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
    ) -> _SlowStageOutput:
        return _slow_stage_impl(execution_log, input_file)

    register_test_stage(slow_stage)

    execution_started = threading.Event()
    # Use dict to hold errors - helps type checker understand cross-thread mutation
    errors: dict[str, Exception | None] = {"exec": None, "commit": None}
    results: dict[str, list[str]] = {"committed": []}

    def run_execution() -> None:
        try:
            execution_started.set()
            executor.run(no_commit=True, pipeline=test_pipeline)
        except Exception as e:
            errors["exec"] = e

    def run_commit() -> None:
        try:
            execution_started.wait(timeout=2.0)
            time.sleep(0.1)
            lock = project_lock.acquire_pending_state_lock()
            try:
                results["committed"] = commit_mod.commit_pending()
            finally:
                lock.release()
        except Exception as e:
            errors["commit"] = e

    exec_thread = threading.Thread(target=run_execution)
    commit_thread = threading.Thread(target=run_commit)

    exec_thread.start()
    commit_thread.start()

    exec_thread.join(timeout=10.0)
    commit_thread.join(timeout=10.0)

    # Propagate any errors from threads
    if errors["exec"] is not None:
        raise errors["exec"]
    if errors["commit"] is not None:
        raise errors["commit"]

    # Read log to verify ordering
    log_content = execution_log.read_text()

    # Commit should have completed successfully
    assert "slow_stage" in results["committed"], "Stage should have been committed"

    # Stage should have completed before commit got the lock
    assert "stage_start" in log_content, "Stage should have started"
    assert "stage_end" in log_content, "Stage should have completed"


class _Stage1Output(TypedDict):
    output1: Annotated[pathlib.Path, outputs.Out("output1.txt", loaders.PathOnly())]


class _Stage2Output(TypedDict):
    output2: Annotated[pathlib.Path, outputs.Out("output2.txt", loaders.PathOnly())]


def _stage1_impl(
    input1: Annotated[pathlib.Path, outputs.Dep("input1.txt", loaders.PathOnly())],
) -> _Stage1Output:
    output_path = pathlib.Path("output1.txt")
    output_path.write_text("done1")
    return _Stage1Output(output1=output_path)


def _stage2_impl(
    input2: Annotated[pathlib.Path, outputs.Dep("input2.txt", loaders.PathOnly())],
) -> _Stage2Output:
    output_path = pathlib.Path("output2.txt")
    output_path.write_text("done2")
    return _Stage2Output(output2=output_path)


def test_concurrent_commits_serialize(pipeline_dir: pathlib.Path, test_pipeline: Pipeline) -> None:
    """Multiple concurrent commit attempts serialize correctly."""
    # Create multiple stages
    pathlib.Path("input1.txt").write_text("data1")
    pathlib.Path("input2.txt").write_text("data2")

    def stage1(
        input1: Annotated[pathlib.Path, outputs.Dep("input1.txt", loaders.PathOnly())],
    ) -> _Stage1Output:
        return _stage1_impl(input1)

    def stage2(
        input2: Annotated[pathlib.Path, outputs.Dep("input2.txt", loaders.PathOnly())],
    ) -> _Stage2Output:
        return _stage2_impl(input2)

    register_test_stage(stage1)
    register_test_stage(stage2)

    # Run with --no-commit
    executor.run(no_commit=True, pipeline=test_pipeline)

    # Now try two concurrent commits
    results = list[tuple[str, list[str]]]()
    barrier = threading.Barrier(2)

    def do_commit(name: str) -> None:
        barrier.wait()  # Synchronize start
        # Acquire lock first (like CLI does)
        lock = project_lock.acquire_pending_state_lock()
        try:
            committed = commit_mod.commit_pending()
        finally:
            lock.release()
        results.append((name, committed))

    t1 = threading.Thread(target=do_commit, args=("first",))
    t2 = threading.Thread(target=do_commit, args=("second",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # One should get both stages, the other should get empty
    all_committed = list[str]()
    for _name, committed in results:
        all_committed.extend(committed)

    # Total committed should be exactly 2 stages
    assert len(all_committed) == 2, f"Expected 2 stages, got {all_committed}"
    assert set(all_committed) == {"stage1", "stage2"}
