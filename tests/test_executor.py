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
    """Stages execute in dependency order."""
    execution_log = list[str]()

    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["step1.txt"])
    def step1() -> None:
        execution_log.append("step1")
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("step1.txt").write_text(data.upper())

    @stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        execution_log.append("step2")
        data = pathlib.Path("step1.txt").read_text()
        pathlib.Path("step2.txt").write_text(f"Result: {data}")

    from pivot import executor

    results = executor.run()

    assert execution_log == ["step1", "step2"], "Stages should run in dependency order"
    assert (pipeline_dir / "step2.txt").read_text() == "Result: HELLO"
    assert results["step1"]["status"] == "ran"
    assert results["step2"]["status"] == "ran"


def test_unchanged_stages_are_skipped(pipeline_dir: pathlib.Path) -> None:
    """Stages with unchanged code and deps are skipped on re-run."""
    run_count = {"step1": 0}

    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def step1() -> None:
        run_count["step1"] += 1
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    from pivot import executor

    # First run - should execute
    results = executor.run()
    assert run_count["step1"] == 1
    assert results["step1"]["status"] == "ran"

    # Second run - should skip (nothing changed)
    results = executor.run()
    assert run_count["step1"] == 1, "Stage should not run again"
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
    run_count = {"process": 0}

    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        run_count["process"] += 1
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    from pivot import executor

    # First run
    results = executor.run()
    assert run_count["process"] == 1
    assert results["process"]["status"] == "ran"

    # Modify input
    (pipeline_dir / "input.txt").write_text("world")

    # Should re-run due to input change
    results = executor.run()
    assert run_count["process"] == 2
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "WORLD"


def test_downstream_runs_when_upstream_changes(pipeline_dir: pathlib.Path) -> None:
    """Downstream stages re-run when upstream output changes."""
    run_count = {"step1": 0, "step2": 0}

    (pipeline_dir / "input.txt").write_text("hello")

    @stage(deps=["input.txt"], outs=["intermediate.txt"])
    def step1() -> None:
        run_count["step1"] += 1
        data = pathlib.Path("input.txt").read_text()
        pathlib.Path("intermediate.txt").write_text(data.upper())

    @stage(deps=["intermediate.txt"], outs=["final.txt"])
    def step2() -> None:
        run_count["step2"] += 1
        data = pathlib.Path("intermediate.txt").read_text()
        pathlib.Path("final.txt").write_text(f"Final: {data}")

    from pivot import executor

    # First run - both execute
    executor.run()
    assert run_count == {"step1": 1, "step2": 1}

    # Second run - both skip
    executor.run()
    assert run_count == {"step1": 1, "step2": 1}

    # Change input - both should re-run
    (pipeline_dir / "input.txt").write_text("world")
    executor.run()
    assert run_count == {"step1": 2, "step2": 2}
    assert (pipeline_dir / "final.txt").read_text() == "Final: WORLD"


def test_run_specific_stage(pipeline_dir: pathlib.Path) -> None:
    """Can run a specific stage and its dependencies only."""
    run_count = {"a": 0, "b": 0, "c": 0}

    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["a.txt"])
    def a() -> None:
        run_count["a"] += 1
        pathlib.Path("a.txt").write_text("a")

    @stage(deps=["a.txt"], outs=["b.txt"])
    def b() -> None:
        run_count["b"] += 1
        pathlib.Path("b.txt").write_text("b")

    @stage(deps=["input.txt"], outs=["c.txt"])
    def c() -> None:
        run_count["c"] += 1
        pathlib.Path("c.txt").write_text("c")

    from pivot import executor

    # Run only 'b' (should also run 'a' as dependency, but not 'c')
    executor.run(stages=["b"])

    assert run_count["a"] == 1, "Dependency 'a' should run"
    assert run_count["b"] == 1, "Target 'b' should run"
    assert run_count["c"] == 0, "Unrelated 'c' should not run"


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
    lock_file_existed_during_run = False
    cache_dir = pipeline_dir / ".pivot" / "cache"

    @stage(deps=["input.txt"], outs=["output.txt"])
    def check_lock() -> None:
        nonlocal lock_file_existed_during_run
        lock_file_existed_during_run = (cache_dir / "check_lock.running").exists()
        pathlib.Path("output.txt").write_text("done")

    executor.run()

    assert lock_file_existed_during_run, "Lock file should exist during stage execution"
    assert not (cache_dir / "check_lock.running").exists(), "Lock file should be removed after"


def test_execution_lock_removed_on_stage_failure(pipeline_dir: pathlib.Path) -> None:
    """Execution lock is released even if stage raises an exception."""
    from pivot import executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"

    @stage(deps=["input.txt"], outs=["output.txt"])
    def failing_stage() -> None:
        raise RuntimeError("Stage failed!")

    with pytest.raises(RuntimeError, match="Stage failed!"):
        executor.run()

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


def test_concurrent_execution_raises_error(pipeline_dir: pathlib.Path) -> None:
    """Running stage that's already running raises StageAlreadyRunningError."""
    import os

    from pivot import exceptions, executor

    (pipeline_dir / "input.txt").write_text("hello")
    cache_dir = pipeline_dir / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a lock with our own PID (simulating concurrent run)
    active_lock = cache_dir / "process.running"
    active_lock.write_text(f"pid: {os.getpid()}\n")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    with pytest.raises(exceptions.StageAlreadyRunningError) as exc_info:
        executor.run()

    assert "already running" in str(exc_info.value)
    assert str(os.getpid()) in str(exc_info.value)

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
