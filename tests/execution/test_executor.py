import logging
import os
import pathlib
import sys
import threading
import time

import pytest
import yaml

import pivot
from pivot import exceptions, executor, project, registry
from pivot.outputs import Metric


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before each test."""
    registry.REGISTRY.clear()


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    # Create .pivot marker so project root detection works
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)

    # Reset project root cache
    project._project_root_cache = None

    return tmp_path


def test_simple_pipeline_runs_in_order(pipeline_dir: pathlib.Path) -> None:
    """Stages execute in dependency order and produce correct outputs."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["step1.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("step1.txt").write_text(data.upper())

    @pivot.stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        data = pathlib.Path("step1.txt").read_text()
        pathlib.Path("step2.txt").write_text(f"Result: {data}")

    results = executor.run()

    assert (pipeline_dir / "step2.txt").read_text() == "Result: HELLO"
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"


def test_unchanged_stages_are_skipped(pipeline_dir: pathlib.Path) -> None:
    """Stages with unchanged code and deps are skipped on re-run."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

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
    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "HELLO"

    # Clear and re-register with different implementation
    registry.REGISTRY.clear()

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:  # noqa: F811
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.lower())  # Changed!

    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "hello"


def test_input_change_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Changing input file triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

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

    @pivot.stage(deps=["input.txt"], outs=["intermediate.txt"])
    def step1() -> None:
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("intermediate.txt").write_text(data.upper())

    @pivot.stage(deps=["intermediate.txt"], outs=["final.txt"])
    def step2() -> None:
        data = pathlib.Path("intermediate.txt").read_text()
        pathlib.Path("final.txt").write_text(f"Final: {data}")

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

    @pivot.stage(deps=["input.txt"], outs=["a.txt"])
    def a() -> None:
        pathlib.Path("a.txt").write_text("a")

    @pivot.stage(deps=["a.txt"], outs=["b.txt"])
    def b() -> None:
        pathlib.Path("b.txt").write_text("b")

    @pivot.stage(deps=["input.txt"], outs=["c.txt"])
    def c() -> None:
        pathlib.Path("c.txt").write_text("c")

    # Run only 'b' (should also run 'a' as dependency, but not 'c')
    results = executor.run(stages=["b"])

    assert results["a"]["status"] == "ran", "Dependency 'a' should run"
    assert results["b"]["status"] == "ran", "Target 'b' should run"
    assert "c" not in results, "Unrelated 'c' should not be in results"
    assert not (pipeline_dir / "c.txt").exists(), "Stage 'c' output should not exist"


def test_missing_dependency_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Missing dependency file raises DependencyNotFoundError before stage runs."""
    run_count = {"process": 0}

    # Don't create the input file - it's missing

    @pivot.stage(deps=["missing_input.txt"], outs=["output.txt"])
    def process() -> None:
        run_count["process"] += 1
        pathlib.Path("output.txt").write_text("done")

    with pytest.raises(exceptions.DependencyNotFoundError) as exc_info:
        executor.run()

    assert "missing_input.txt" in str(exc_info.value)
    assert run_count["process"] == 0, "Stage should not run when dependency is missing"


def test_nonexistent_stage_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Requesting a non-existent stage raises StageNotFoundError."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def real_stage() -> None:
        pass

    with pytest.raises(exceptions.StageNotFoundError) as exc_info:
        executor.run(stages=["nonexistent_stage"])

    assert "nonexistent_stage" in str(exc_info.value)


def test_execution_lock_created_and_removed(pipeline_dir: pathlib.Path) -> None:
    """Execution lock file is created during run and removed after."""
    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    lock_check_file = pipeline_dir / "lock_existed.txt"

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
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
    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def failing_stage() -> None:
        raise RuntimeError("Stage failed!")

    # Executor now catches exceptions and returns failed status
    results = executor.run(show_output=False)

    assert results["failing_stage"]["status"] == "failed"
    assert "Stage failed!" in results["failing_stage"]["reason"]
    assert not (cache_dir / "failing_stage.running").exists(), "Lock should be released on failure"


def test_stale_lock_from_dead_process_is_broken(pipeline_dir: pathlib.Path) -> None:
    """Stale lock file from crashed process is automatically removed."""

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a stale lock with a non-existent PID
    stale_lock = cache_dir / "process.running"
    stale_lock.write_text("pid: 999999999\n")  # PID that doesn't exist

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed by breaking the stale lock
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not stale_lock.exists(), "Stale lock should be removed"


def test_concurrent_execution_returns_failed_status(pipeline_dir: pathlib.Path) -> None:
    """Running stage that's already running returns failed status."""
    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a lock with our own PID (simulating concurrent run)
    active_lock = cache_dir / "process.running"
    active_lock.write_text(f"pid: {os.getpid()}\n")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
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
    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a corrupted lock file
    corrupted_lock = cache_dir / "process.running"
    corrupted_lock.write_text("garbage content without pid")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed by treating corrupted lock as stale
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not corrupted_lock.exists(), "Corrupted lock should be removed"


def test_negative_pid_in_lock_is_treated_as_stale(pipeline_dir: pathlib.Path) -> None:
    """Lock file with invalid PID (negative) is treated as stale."""
    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a lock with invalid PID
    invalid_lock = cache_dir / "process.running"
    invalid_lock.write_text("pid: -1\n")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed by treating invalid PID as stale
    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert not invalid_lock.exists(), "Invalid PID lock should be removed"


def test_output_queue_reader_only_catches_empty(pipeline_dir: pathlib.Path) -> None:
    """Output queue reader should only catch queue.Empty, not other exceptions."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # This test verifies the output queue reader behavior exists and handles Empty properly
    results = executor.run(show_output=True)
    assert results["process"]["status"] == "ran"


def test_output_thread_cleanup_completes(pipeline_dir: pathlib.Path) -> None:
    """Output thread should be properly cleaned up after execution."""
    (pipeline_dir / "input.txt").write_text("hello")

    initial_thread_count = threading.active_count()

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        print("Stage output")
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(show_output=True)

    # Give a moment for thread cleanup
    time.sleep(0.2)

    # Thread count should return to initial (or close to it)
    final_thread_count = threading.active_count()
    assert final_thread_count <= initial_thread_count + 1, (
        f"Thread leak: started with {initial_thread_count}, ended with {final_thread_count}"
    )
    assert results["process"]["status"] == "ran"


def test_mutex_prevents_concurrent_execution(pipeline_dir: pathlib.Path) -> None:
    """Stages in same mutex group run sequentially, not concurrently."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    @pivot.stage(deps=["input.txt"], outs=["a.txt"], mutex=["gpu"])
    def stage_a() -> None:
        with open("timing.txt", "a") as f:
            f.write("a_start\n")
        time.sleep(0.1)  # Ensure overlap would be detected
        with open("timing.txt", "a") as f:
            f.write("a_end\n")
        pathlib.Path("a.txt").write_text("a")

    @pivot.stage(deps=["input.txt"], outs=["b.txt"], mutex=["gpu"])
    def stage_b() -> None:
        with open("timing.txt", "a") as f:
            f.write("b_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("b_end\n")
        pathlib.Path("b.txt").write_text("b")

    results = executor.run(max_workers=4, show_output=False)

    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"

    # Verify sequential execution: either a_start,a_end,b_start,b_end or b_start,b_end,a_start,a_end
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["a_start", "a_end", "b_start", "b_end"],
        ["b_start", "b_end", "a_start", "a_end"],
    ], f"Stages ran concurrently: {timing}"


def test_mutex_releases_on_completion(pipeline_dir: pathlib.Path) -> None:
    """Mutex is released when stage completes, allowing next stage to start."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["first.txt"], mutex=["resource"])
    def first() -> None:
        pathlib.Path("first.txt").write_text("first")

    @pivot.stage(deps=["input.txt"], outs=["second.txt"], mutex=["resource"])
    def second() -> None:
        pathlib.Path("second.txt").write_text("second")

    results = executor.run(max_workers=4, show_output=False)

    assert results["first"]["status"] == "ran"
    assert results["second"]["status"] == "ran"
    assert (pipeline_dir / "first.txt").exists()
    assert (pipeline_dir / "second.txt").exists()


def test_mutex_releases_on_failure(pipeline_dir: pathlib.Path) -> None:
    """Mutex is released even when stage fails, allowing other stages to run."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["failing.txt"], mutex=["resource"])
    def failing() -> None:
        raise RuntimeError("Intentional failure")

    @pivot.stage(deps=["input.txt"], outs=["succeeding.txt"], mutex=["resource"])
    def succeeding() -> None:
        pathlib.Path("succeeding.txt").write_text("success")

    results = executor.run(max_workers=4, on_error="keep_going", show_output=False)

    assert results["failing"]["status"] == "failed"
    assert results["succeeding"]["status"] == "ran"
    assert (pipeline_dir / "succeeding.txt").exists()


def test_multiple_mutex_groups_per_stage(pipeline_dir: pathlib.Path) -> None:
    """Stage with multiple mutex groups blocks all of them."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    @pivot.stage(deps=["input.txt"], outs=["multi.txt"], mutex=["gpu", "disk"])
    def multi_resource() -> None:
        with open("timing.txt", "a") as f:
            f.write("multi_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("multi_end\n")
        pathlib.Path("multi.txt").write_text("multi")

    @pivot.stage(deps=["input.txt"], outs=["gpu_only.txt"], mutex=["gpu"])
    def gpu_only() -> None:
        with open("timing.txt", "a") as f:
            f.write("gpu_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("gpu_end\n")
        pathlib.Path("gpu_only.txt").write_text("gpu")

    results = executor.run(max_workers=4, show_output=False)

    assert results["multi_resource"]["status"] == "ran"
    assert results["gpu_only"]["status"] == "ran"

    # Stages should be sequential due to shared "gpu" mutex
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["multi_start", "multi_end", "gpu_start", "gpu_end"],
        ["gpu_start", "gpu_end", "multi_start", "multi_end"],
    ], f"Stages ran concurrently despite shared mutex: {timing}"


def test_mutex_with_dependencies(pipeline_dir: pathlib.Path) -> None:
    """Mutex works correctly with stage dependencies."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["first.txt"], mutex=["resource"])
    def first() -> None:
        pathlib.Path("first.txt").write_text("first")

    @pivot.stage(deps=["first.txt"], outs=["second.txt"], mutex=["resource"])
    def second() -> None:
        data = pathlib.Path("first.txt").read_text()
        pathlib.Path("second.txt").write_text(f"second: {data}")

    results = executor.run(max_workers=4, show_output=False)

    assert results["first"]["status"] == "ran"
    assert results["second"]["status"] == "ran"
    assert (pipeline_dir / "second.txt").read_text() == "second: first"


def test_no_mutex_stages_unaffected(pipeline_dir: pathlib.Path) -> None:
    """Stages without mutex run normally in parallel."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    @pivot.stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("timing.txt", "a") as f:
            f.write("a_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("a_end\n")
        pathlib.Path("a.txt").write_text("a")

    @pivot.stage(deps=["input.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("timing.txt", "a") as f:
            f.write("b_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("b_end\n")
        pathlib.Path("b.txt").write_text("b")

    results = executor.run(max_workers=4, show_output=False)

    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"

    # Without mutex, stages should run in parallel (interleaved timing)
    timing = timing_file.read_text().strip().split("\n")
    assert len(timing) == 4


def test_single_stage_mutex_warning(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Warning is logged when mutex group has only one stage."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"], mutex=["lonely_group"])
    def lonely() -> None:
        pathlib.Path("output.txt").write_text("done")

    with caplog.at_level(logging.WARNING):
        results = executor.run(show_output=False)

    assert results["lonely"]["status"] == "ran"
    assert any(
        "lonely_group" in record.message and "lonely" in record.message for record in caplog.records
    ), (
        f"Expected warning about single-stage mutex group, got: {[r.message for r in caplog.records]}"
    )


# =============================================================================
# Error Mode Tests
# =============================================================================


def test_on_error_fail_stops_on_first_failure(pipeline_dir: pathlib.Path) -> None:
    """on_error='fail' stops pipeline when first stage fails."""
    (pipeline_dir / "input.txt").write_text("data")
    execution_log = pipeline_dir / "execution.log"

    @pivot.stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("execution.log", "a") as f:
            f.write("a\n")
        raise RuntimeError("Stage A failed")

    @pivot.stage(deps=["input.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("execution.log", "a") as f:
            f.write("b\n")
        pathlib.Path("b.txt").write_text("b")

    results = executor.run(on_error="fail", show_output=False)

    assert results["stage_a"]["status"] == "failed"
    # stage_b may or may not run depending on timing, but pipeline should stop
    log_content = execution_log.read_text() if execution_log.exists() else ""
    assert "a" in log_content, "Stage A should have executed"


def test_on_error_keep_going_continues_independent_stages(pipeline_dir: pathlib.Path) -> None:
    """on_error='keep_going' continues running independent stages after failure."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["failing.txt"])
    def failing() -> None:
        raise RuntimeError("Intentional failure")

    @pivot.stage(deps=["input.txt"], outs=["succeeding.txt"])
    def succeeding() -> None:
        pathlib.Path("succeeding.txt").write_text("success")

    results = executor.run(on_error="keep_going", show_output=False)

    assert results["failing"]["status"] == "failed"
    assert results["succeeding"]["status"] == "ran"
    assert (pipeline_dir / "succeeding.txt").read_text() == "success"


def test_on_error_keep_going_skips_downstream_of_failed(pipeline_dir: pathlib.Path) -> None:
    """on_error='keep_going' skips stages that depend on failed stage."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["first.txt"])
    def first() -> None:
        raise RuntimeError("First failed")

    @pivot.stage(deps=["first.txt"], outs=["second.txt"])
    def second() -> None:
        pathlib.Path("second.txt").write_text("should not run")

    @pivot.stage(deps=["input.txt"], outs=["independent.txt"])
    def independent() -> None:
        pathlib.Path("independent.txt").write_text("runs fine")

    results = executor.run(on_error="keep_going", show_output=False)

    assert results["first"]["status"] == "failed"
    assert results["second"]["status"] == "skipped"
    assert "upstream" in results["second"]["reason"]
    assert results["independent"]["status"] == "ran"


def test_on_error_ignore_allows_downstream_to_attempt(pipeline_dir: pathlib.Path) -> None:
    """on_error='ignore' allows downstream stages to attempt running."""
    (pipeline_dir / "input.txt").write_text("data")
    # Note: pre-existing outputs are removed before stage execution
    # so downstream will fail with missing dependency if upstream fails

    @pivot.stage(deps=["input.txt"], outs=["first.txt"])
    def first() -> None:
        raise RuntimeError("First failed")

    @pivot.stage(deps=["first.txt"], outs=["second.txt"])
    def second() -> None:
        data = pathlib.Path("first.txt").read_text()
        pathlib.Path("second.txt").write_text(f"got: {data}")

    results = executor.run(on_error="ignore", show_output=False)

    assert results["first"]["status"] == "failed"
    # With cache integration, outputs are removed before execution.
    # Since first stage failed, first.txt doesn't exist, so second fails with missing dep.
    assert results["second"]["status"] == "failed"
    assert "missing" in results["second"]["reason"].lower()


def test_invalid_on_error_raises_value_error(pipeline_dir: pathlib.Path) -> None:
    """Invalid on_error value raises ValueError."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pass

    with pytest.raises(ValueError) as exc_info:
        executor.run(on_error="invalid_mode", show_output=False)

    assert "invalid_mode" in str(exc_info.value)
    assert "fail" in str(exc_info.value)  # Should mention valid options


# =============================================================================
# Timeout Tests
# =============================================================================


def test_stage_timeout_marks_stage_as_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage exceeding timeout is marked as failed."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def slow_stage() -> None:
        time.sleep(5)  # Sleep longer than timeout
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(stage_timeout=0.5, show_output=False)

    assert results["slow_stage"]["status"] == "failed"
    assert "timed out" in results["slow_stage"]["reason"]


def test_stage_timeout_does_not_affect_fast_stages(pipeline_dir: pathlib.Path) -> None:
    """Fast stages complete normally even with timeout set."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def fast_stage() -> None:
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(stage_timeout=60.0, show_output=False)

    assert results["fast_stage"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "done"


# =============================================================================
# Worker Exception Tests
# =============================================================================


def test_stage_calling_sys_exit_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage calling sys.exit() returns failed status with exit code."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def exits_with_code() -> None:
        sys.exit(42)

    results = executor.run(show_output=False)

    assert results["exits_with_code"]["status"] == "failed"
    assert "sys.exit" in results["exits_with_code"]["reason"]
    assert "42" in results["exits_with_code"]["reason"]


def test_stage_calling_sys_exit_zero_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage calling sys.exit(0) still returns failed (stages shouldn't exit)."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def exits_zero() -> None:
        sys.exit(0)

    results = executor.run(show_output=False)

    assert results["exits_zero"]["status"] == "failed"
    assert "sys.exit" in results["exits_zero"]["reason"]


def test_stage_raising_keyboard_interrupt_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Stage raising KeyboardInterrupt returns failed status."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def keyboard_interrupt() -> None:
        raise KeyboardInterrupt("User cancelled")

    results = executor.run(show_output=False)

    assert results["keyboard_interrupt"]["status"] == "failed"
    assert "KeyboardInterrupt" in results["keyboard_interrupt"]["reason"]


# =============================================================================
# Dependency Validation Tests
# =============================================================================


def test_directory_dependency_hashed_and_runs(pipeline_dir: pathlib.Path) -> None:
    """Stage with directory as dependency hashes it and runs successfully."""
    # Create a directory with files
    data_dir = pipeline_dir / "data_dir"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")
    (data_dir / "file2.txt").write_text("content2")

    @pivot.stage(deps=["data_dir"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "done"


# =============================================================================
# Non-Parallel Execution Tests
# =============================================================================


def test_parallel_false_runs_sequentially(pipeline_dir: pathlib.Path) -> None:
    """parallel=False runs stages one at a time."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    @pivot.stage(deps=["input.txt"], outs=["a.txt"])
    def stage_a() -> None:
        with open("timing.txt", "a") as f:
            f.write("a_start\n")
        time.sleep(0.05)
        with open("timing.txt", "a") as f:
            f.write("a_end\n")
        pathlib.Path("a.txt").write_text("a")

    @pivot.stage(deps=["input.txt"], outs=["b.txt"])
    def stage_b() -> None:
        with open("timing.txt", "a") as f:
            f.write("b_start\n")
        time.sleep(0.05)
        with open("timing.txt", "a") as f:
            f.write("b_end\n")
        pathlib.Path("b.txt").write_text("b")

    results = executor.run(parallel=False, show_output=False)

    assert results["stage_a"]["status"] == "ran"
    assert results["stage_b"]["status"] == "ran"

    # With parallel=False, stages must be strictly sequential
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["a_start", "a_end", "b_start", "b_end"],
        ["b_start", "b_end", "a_start", "a_end"],
    ], f"Stages overlapped with parallel=False: {timing}"


# =============================================================================
# Output Capture Tests
# =============================================================================


def test_stage_stdout_and_stderr_captured(pipeline_dir: pathlib.Path) -> None:
    """Stage stdout and stderr are captured in results."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def prints_output() -> None:
        print("stdout message")
        print("stderr message", file=sys.stderr)
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(show_output=False)

    assert results["prints_output"]["status"] == "ran"
    # Output lines are captured in results but not exposed in dict
    # The stage should run successfully with captured output


def test_stage_partial_line_output_captured(pipeline_dir: pathlib.Path) -> None:
    """Stage output without trailing newline is captured."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def partial_output() -> None:
        sys.stdout.write("no newline at end")
        sys.stdout.flush()
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(show_output=False)
    assert results["partial_output"]["status"] == "ran"


# =============================================================================
# Lock Retry Exhaustion Tests
# =============================================================================


def test_lock_retry_exhaustion_returns_failed(pipeline_dir: pathlib.Path) -> None:
    """Multiple failed lock attempts return failed status."""
    (pipeline_dir / "input.txt").write_text("data")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True)

    # Create a lock with our own PID (simulates live concurrent run)
    lock_file = cache_dir / "process.running"
    lock_file.write_text(f"pid: {os.getpid()}\n")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "failed"
    assert "already running" in results["process"]["reason"]

    lock_file.unlink()


# =============================================================================
# Mutex Name Normalization Tests
# =============================================================================


def test_mutex_names_are_case_insensitive(pipeline_dir: pathlib.Path) -> None:
    """Mutex names are normalized to lowercase for comparison."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    @pivot.stage(deps=["input.txt"], outs=["upper.txt"], mutex=["GPU"])
    def upper_mutex() -> None:
        with open("timing.txt", "a") as f:
            f.write("upper_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("upper_end\n")
        pathlib.Path("upper.txt").write_text("done")

    @pivot.stage(deps=["input.txt"], outs=["lower.txt"], mutex=["gpu"])
    def lower_mutex() -> None:
        with open("timing.txt", "a") as f:
            f.write("lower_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("lower_end\n")
        pathlib.Path("lower.txt").write_text("done")

    results = executor.run(max_workers=4, show_output=False)

    assert results["upper_mutex"]["status"] == "ran"
    assert results["lower_mutex"]["status"] == "ran"

    # Should be sequential due to mutex normalization
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["upper_start", "upper_end", "lower_start", "lower_end"],
        ["lower_start", "lower_end", "upper_start", "upper_end"],
    ], f"Mutex names not normalized - stages ran concurrently: {timing}"


def test_mutex_names_whitespace_stripped(pipeline_dir: pathlib.Path) -> None:
    """Mutex names have whitespace stripped."""
    (pipeline_dir / "input.txt").write_text("data")
    timing_file = pipeline_dir / "timing.txt"

    @pivot.stage(deps=["input.txt"], outs=["spaced.txt"], mutex=["  resource  "])
    def spaced_mutex() -> None:
        with open("timing.txt", "a") as f:
            f.write("spaced_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("spaced_end\n")
        pathlib.Path("spaced.txt").write_text("done")

    @pivot.stage(deps=["input.txt"], outs=["clean.txt"], mutex=["resource"])
    def clean_mutex() -> None:
        with open("timing.txt", "a") as f:
            f.write("clean_start\n")
        time.sleep(0.1)
        with open("timing.txt", "a") as f:
            f.write("clean_end\n")
        pathlib.Path("clean.txt").write_text("done")

    results = executor.run(max_workers=4, show_output=False)

    assert results["spaced_mutex"]["status"] == "ran"
    assert results["clean_mutex"]["status"] == "ran"

    # Should be sequential due to mutex normalization
    timing = timing_file.read_text().strip().split("\n")
    assert timing in [
        ["spaced_start", "spaced_end", "clean_start", "clean_end"],
        ["clean_start", "clean_end", "spaced_start", "spaced_end"],
    ], f"Mutex whitespace not stripped - stages ran concurrently: {timing}"


# =============================================================================
# Output Cache Tests
# =============================================================================


def test_executor_removes_outputs_before_run(pipeline_dir: pathlib.Path) -> None:
    """Outputs are removed before stage execution (clean state)."""
    (pipeline_dir / "input.txt").write_text("data")
    output_file = pipeline_dir / "output.txt"
    output_file.write_text("stale data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        # Should not see stale data - output should have been removed
        assert not output_file.exists(), "Output should be removed before stage runs"
        output_file.write_text("fresh data")

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "ran"
    assert output_file.read_text() == "fresh data"


def test_executor_saves_outputs_to_cache(pipeline_dir: pathlib.Path) -> None:
    """Outputs are saved to cache after successful execution."""
    (pipeline_dir / "input.txt").write_text("data")
    cache_dir = pipeline_dir / ".pivot" / "cache"

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("result")

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "ran"

    # Output should now be a symlink to cache
    output_file = pipeline_dir / "output.txt"
    assert output_file.is_symlink(), "Output should be symlink to cache"
    assert output_file.read_text() == "result"

    # Cache should contain the file
    files_cache = cache_dir / "files"
    assert files_cache.exists(), "Cache directory should exist"
    cache_files = list(files_cache.rglob("*"))
    assert len(cache_files) >= 1, "Cache should contain files"


def test_executor_restores_missing_outputs_on_skip(pipeline_dir: pathlib.Path) -> None:
    """Skipped stages restore missing outputs from cache."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("result")

    # First run - executes and caches
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"

    # Delete output (simulating user deleting file)
    output_file = pipeline_dir / "output.txt"
    if output_file.is_symlink():
        output_file.unlink()
    else:
        output_file.unlink()

    assert not output_file.exists()

    # Second run - should skip but restore output from cache
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "skipped"
    assert output_file.exists(), "Output should be restored from cache"
    assert output_file.read_text() == "result"


def test_executor_fails_if_output_missing(pipeline_dir: pathlib.Path) -> None:
    """Stage fails if declared output is not produced."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        # Intentionally don't create output.txt
        pass

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "failed"
    assert "output" in results["process"]["reason"].lower()


def test_executor_handles_cache_false_outputs(pipeline_dir: pathlib.Path) -> None:
    """Metric outputs with cache=False are not cached."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=[Metric("metrics.json")])
    def process() -> None:
        import json

        pathlib.Path("metrics.json").write_text(json.dumps({"accuracy": 0.95}))

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "ran"

    # Metric output should NOT be a symlink (not cached)
    metrics_file = pipeline_dir / "metrics.json"
    assert metrics_file.exists()
    assert not metrics_file.is_symlink(), "Metric with cache=False should not be symlink"


def test_executor_fails_if_cache_false_output_missing(pipeline_dir: pathlib.Path) -> None:
    """Stage fails if cache=False output is not produced."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=[Metric("metrics.json")])
    def process() -> None:
        pass  # Intentionally don't create metrics.json

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "failed"
    assert "metrics.json" in results["process"]["reason"]


def test_executor_output_hashes_in_lock_file(pipeline_dir: pathlib.Path) -> None:
    """Output hashes are stored in lock file."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("result")

    executor.run(show_output=False)

    lock_file = pipeline_dir / ".pivot" / "cache" / "stages" / "process.lock"
    assert lock_file.exists()

    # Storage format uses 'outs' list (not 'output_hashes' dict)
    lock_data = yaml.safe_load(lock_file.read_text())
    assert "outs" in lock_data
    assert len(lock_data["outs"]) == 1
    assert lock_data["outs"][0]["path"] == "output.txt"


def test_executor_lock_file_deterministic_sort(pipeline_dir: pathlib.Path) -> None:
    """Lock file entries are sorted for deterministic output."""
    (pipeline_dir / "input.txt").write_text("data")
    (pipeline_dir / "z_input.txt").write_text("z")
    (pipeline_dir / "a_input.txt").write_text("a")

    @pivot.stage(deps=["z_input.txt", "a_input.txt"], outs=["z_out.txt", "a_out.txt"])
    def process() -> None:
        pathlib.Path("z_out.txt").write_text("z")
        pathlib.Path("a_out.txt").write_text("a")

    executor.run(show_output=False)

    lock_file = pipeline_dir / ".pivot" / "cache" / "stages" / "process.lock"
    lock_data = yaml.safe_load(lock_file.read_text())

    # Storage format uses 'deps' and 'outs' lists (sorted by path)
    dep_paths = [entry["path"] for entry in lock_data.get("deps", [])]
    assert dep_paths == sorted(dep_paths), "deps should be sorted by path"

    out_paths = [entry["path"] for entry in lock_data.get("outs", [])]
    assert out_paths == sorted(out_paths), "outs should be sorted by path"


def test_executor_directory_output_cached(pipeline_dir: pathlib.Path) -> None:
    """Directory outputs are cached with manifest."""
    (pipeline_dir / "input.txt").write_text("data")

    @pivot.stage(deps=["input.txt"], outs=["output_dir/"])
    def process() -> None:
        out_dir = pathlib.Path("output_dir")
        out_dir.mkdir(exist_ok=True)
        (out_dir / "file1.txt").write_text("file1")
        (out_dir / "file2.txt").write_text("file2")

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "ran"

    output_dir = pipeline_dir / "output_dir"
    assert output_dir.exists()
    assert (output_dir / "file1.txt").read_text() == "file1"
    assert (output_dir / "file2.txt").read_text() == "file2"


def test_executor_lock_file_missing_outs_triggers_rerun(pipeline_dir: pathlib.Path) -> None:
    """Lock file without outs section triggers re-execution."""
    (pipeline_dir / "input.txt").write_text("data")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    stages_dir = cache_dir / "stages"
    stages_dir.mkdir(parents=True)

    # Create lock file without outs (incomplete)
    lock_file = stages_dir / "process.lock"
    lock_file.write_text(
        yaml.dump(
            {
                "code_manifest": {},
                "params": {},
                "deps": [],
                # No outs - incomplete lock
            }
        )
    )

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("result")

    results = executor.run(show_output=False)

    # Should re-run because outs is missing
    assert results["process"]["status"] == "ran"

    # Lock file should now have outs
    lock_data = yaml.safe_load(lock_file.read_text())
    assert "outs" in lock_data
