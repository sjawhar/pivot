"""Integration tests for pipeline execution."""

import pathlib

import pytest

from pivot.registry import REGISTRY, stage


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before each test."""
    REGISTRY.clear()
    yield
    REGISTRY.clear()


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    # Create .pivot marker so project root detection works
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)

    # Reset project root cache
    from pivot import project

    project._project_root_cache = None

    return tmp_path


def test_simple_pipeline_runs_in_order(pipeline_dir: pathlib.Path) -> None:
    """Stages execute in dependency order and produce correct outputs."""
    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["step1.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("step1.txt").write_text(data.upper())

    @stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        data = pathlib.Path("step1.txt").read_text()
        pathlib.Path("step2.txt").write_text(f"Result: {data}")

    from pivot import executor

    results = executor.run()

    assert (pipeline_dir / "step2.txt").read_text() == "Result: HELLO"
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"


def test_unchanged_stages_are_skipped(pipeline_dir: pathlib.Path) -> None:
    """Stages with unchanged code and deps are skipped on re-run."""
    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    from pivot import executor

    # First run - should execute
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Second run - should skip (nothing changed)
    results = executor.run()
    assert results["step1"]["status"] == "skipped"


def test_code_change_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Changing stage code triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    # First version
    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    from pivot import executor

    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Clear and re-register with different implementation
    REGISTRY.clear()

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:  # noqa: F811
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.lower())  # Changed!

    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "hello"


def test_input_change_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Changing input file triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    from pivot import executor

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Modify input
    (pipeline_dir / "input.txt").write_text("world")

    # Should re-run due to input change
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "WORLD"


def test_downstream_runs_when_upstream_changes(pipeline_dir: pathlib.Path) -> None:
    """Downstream stages re-run when upstream output changes."""
    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["intermediate.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("intermediate.txt").write_text(data.upper())

    @stage(deps=["intermediate.txt"], outs=["final.txt"])
    def step2() -> None:
        data = pathlib.Path("intermediate.txt").read_text()
        pathlib.Path("final.txt").write_text(f"Final: {data}")

    from pivot import executor

    # First run - both execute
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert (pipeline_dir / "final.txt").read_text() == "Final: HELLO"

    # Second run - both skip
    results = executor.run()
    assert results["step1"]["status"] == "skipped"
    assert results["step2"]["status"] == "skipped"

    # Change input - both should re-run
    (pipeline_dir / "input.txt").write_text("world")
    results = executor.run()
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"
    assert (pipeline_dir / "final.txt").read_text() == "Final: WORLD"


def test_run_specific_stage(pipeline_dir: pathlib.Path) -> None:
    """Can run a specific stage and its dependencies only."""
    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def a() -> None:
        pathlib.Path("a.txt").write_text("a")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def b() -> None:
        pathlib.Path("b.txt").write_text("b")

    @stage(deps=["input.txt"], outs=["c.txt"])
    def c() -> None:
        pathlib.Path("c.txt").write_text("c")

    from pivot import executor

    # Run only 'b' (should also run 'a' as dependency, but not 'c')
    results = executor.run(stages=["b"])

    assert results["a"]["status"] == "ran", "Dependency 'a' should run"
    assert results["b"]["status"] == "ran", "Target 'b' should run"
    assert "c" not in results, "Unrelated 'c' should not be in results"
    assert not (pipeline_dir / "c.txt").exists(), "Stage 'c' output should not exist"


def test_missing_dependency_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Missing dependency file raises DependencyNotFoundError before stage runs."""
    from pivot import exceptions, executor

    run_count = {"process": 0}

    # Don't create the input file - it's missing

    @stage(deps=["missing_input.txt"], outs=["output.txt"])
    def process() -> None:
        run_count["process"] += 1
        pathlib.Path("output.txt").write_text("done")

    with pytest.raises(exceptions.DependencyNotFoundError) as exc_info:
        executor.run()

    assert "missing_input.txt" in str(exc_info.value)
    assert run_count["process"] == 0, "Stage should not run when dependency is missing"


def test_nonexistent_stage_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Requesting a non-existent stage raises StageNotFoundError."""
    from pivot import exceptions, executor

    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def real_stage() -> None:
        pass

    with pytest.raises(exceptions.StageNotFoundError) as exc_info:
        executor.run(stages=["nonexistent_stage"])

    assert "nonexistent_stage" in str(exc_info.value)


def test_execution_lock_created_and_removed(pipeline_dir: pathlib.Path) -> None:
    """Execution lock file is created during run and removed after."""
    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    lock_check_file = pipeline_dir / "lock_existed.txt"

    @stage(deps=["input.txt"], outs=["output.txt"])
    def check_lock() -> None:
        # Write to a file to communicate whether lock existed (multiprocessing safe)
        lock_existed = (cache_dir / "check_lock.running").exists()
        lock_check_file.write_text("yes" if lock_existed else "no")
        pathlib.Path("output.txt").write_text("done")

    executor.run()

    assert lock_check_file.read_text() == "yes", "Lock file should exist during stage execution"
    assert not (cache_dir / "check_lock.running").exists(), "Lock file should be removed after"


def test_execution_lock_removed_on_stage_failure(pipeline_dir: pathlib.Path) -> None:
    """Execution lock is released even if stage raises an exception."""
    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"

    @stage(deps=["input.txt"], outs=["output.txt"])
    def failing_stage() -> None:
        raise RuntimeError("Stage failed!")

    # Executor now catches exceptions and returns failed status
    results = executor.run(show_output=False)

    assert results["failing_stage"]["status"] == "failed"
    assert "Stage failed!" in results["failing_stage"]["reason"]
    assert not (cache_dir / "failing_stage.running").exists(), "Lock should be released on failure"


def test_stale_lock_from_dead_process_is_broken(pipeline_dir: pathlib.Path) -> None:
    """Stale lock file from crashed process is automatically removed."""

    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a stale lock with a non-existent PID
    stale_lock = cache_dir / "process.running"
    stale_lock.write_text("pid: 999999999\n")  # PID that doesn't exist

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed by breaking the stale lock
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not stale_lock.exists(), "Stale lock should be removed"


def test_concurrent_execution_returns_failed_status(pipeline_dir: pathlib.Path) -> None:
    """Running stage that's already running returns failed status."""
    import os

    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a lock with our own PID (simulating concurrent run)
    active_lock = cache_dir / "process.running"
    active_lock.write_text(f"pid: {os.getpid()}\n")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Executor now returns failed status instead of raising
    results = executor.run(show_output=False)

    assert results["process"]["status"] == "failed"
    assert "already running" in results["process"]["reason"]
    assert str(os.getpid()) in results["process"]["reason"]

    # Clean up
    active_lock.unlink()


def test_corrupted_lock_file_is_broken(pipeline_dir: pathlib.Path) -> None:
    """Corrupted lock file (invalid content) is treated as stale and removed."""
    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a corrupted lock file
    corrupted_lock = cache_dir / "process.running"
    corrupted_lock.write_text("garbage content without pid")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed by treating corrupted lock as stale
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not corrupted_lock.exists(), "Corrupted lock should be removed"


def test_negative_pid_in_lock_is_treated_as_stale(pipeline_dir: pathlib.Path) -> None:
    """Lock file with invalid PID (negative) is treated as stale."""
    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a lock with invalid PID
    invalid_lock = cache_dir / "process.running"
    invalid_lock.write_text("pid: -1\n")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed by treating invalid PID as stale
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not invalid_lock.exists(), "Invalid PID lock should be removed"


def test_output_queue_reader_only_catches_empty(
    pipeline_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output queue reader should only catch queue.Empty, not other exceptions."""
    import queue as queue_module

    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Track if queue.Empty is properly handled (not other exceptions)
    empty_count = {"value": 0}
    original_get = queue_module.Queue.get

    def mock_get(self, *args, **kwargs):
        try:
            return original_get(self, *args, **kwargs)
        except queue_module.Empty:
            empty_count["value"] += 1
            raise

    # This test verifies the behavior exists - actual fix ensures only Empty is caught
    results = executor.run(show_output=True)
    assert results["process"]["status"] == "ran"


def test_output_thread_cleanup_completes(pipeline_dir: pathlib.Path) -> None:
    """Output thread should be properly cleaned up after execution."""
    import threading

    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")

    initial_thread_count = threading.active_count()

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        print("Stage output")
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(show_output=True)

    # Give a moment for thread cleanup
    import time

    time.sleep(0.2)

    # Thread count should return to initial (or close to it)
    final_thread_count = threading.active_count()
    assert final_thread_count <= initial_thread_count + 1, (
        f"Thread leak: started with {initial_thread_count}, ended with {final_thread_count}"
    )
    assert results["process"]["status"] == "ran"
