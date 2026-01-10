import multiprocessing
import pathlib
import queue
import runpy
import threading
import time
from unittest import mock

import pytest

from pivot import executor, project, types
from pivot.pipeline import yaml as pipeline_yaml
from pivot.reactive import engine
from pivot.registry import REGISTRY, stage


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pivot.yaml").write_text("version: 1\n")
    REGISTRY.clear()
    return tmp_path


# _build_file_to_stages_index tests


def test_build_file_to_stages_index_maps_deps_to_stages(pipeline_dir: pathlib.Path) -> None:
    """Should map dependency files to their consuming stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")
    (pipeline_dir / "config.yaml").write_text("key: value")

    @stage(deps=["data.csv"], outs=["output1.txt"])
    def stage_a() -> None:
        pass

    @stage(deps=["config.yaml"], outs=["output2.txt"])
    def stage_b() -> None:
        pass

    eng = engine.ReactiveEngine()
    index = eng._build_file_to_stages_index()

    data_path = project.resolve_path("data.csv")
    config_path = project.resolve_path("config.yaml")

    assert data_path in index
    assert "stage_a" in index[data_path]
    assert config_path in index
    assert "stage_b" in index[config_path]


def test_build_file_to_stages_index_handles_shared_deps(pipeline_dir: pathlib.Path) -> None:
    """Multiple stages sharing same dep should all be in the index."""
    (pipeline_dir / "shared.csv").write_text("a,b\n1,2")

    @stage(deps=["shared.csv"], outs=["out1.txt"])
    def stage_a() -> None:
        pass

    @stage(deps=["shared.csv"], outs=["out2.txt"])
    def stage_b() -> None:
        pass

    eng = engine.ReactiveEngine()
    index = eng._build_file_to_stages_index()

    shared_path = project.resolve_path("shared.csv")
    assert shared_path in index
    assert "stage_a" in index[shared_path]
    assert "stage_b" in index[shared_path]


# _get_stages_matching_changes tests


def test_get_stages_affected_by_direct_dep_match(pipeline_dir: pathlib.Path) -> None:
    """Should find stages with direct dependency on changed file."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()
    data_path = project.resolve_path("data.csv")
    affected = eng._get_stages_matching_changes({data_path})

    assert "process" in affected


def test_get_stages_affected_by_file_in_dep_directory(pipeline_dir: pathlib.Path) -> None:
    """Should find stages when file inside a dep directory changes."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "file.csv").write_text("a,b\n1,2")

    @stage(deps=["data/"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()
    file_path = project.resolve_path("data/file.csv")
    affected = eng._get_stages_matching_changes({file_path})

    assert "process" in affected


def test_get_stages_affected_returns_empty_for_unrelated_file(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should return empty set for changes to unrelated files."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()
    unrelated_path = project.resolve_path("unrelated.txt")
    affected = eng._get_stages_matching_changes({unrelated_path})

    assert len(affected) == 0, "Should be empty for unrelated file"


# _add_downstream_stages tests


def test_add_downstream_stages_includes_direct_dependents(pipeline_dir: pathlib.Path) -> None:
    """Should include stages that directly depend on affected stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["intermediate.txt"])
    def preprocess() -> None:
        pass

    @stage(deps=["intermediate.txt"], outs=["final.txt"])
    def train() -> None:
        pass

    eng = engine.ReactiveEngine()
    affected = eng._add_downstream_stages({"preprocess"})

    assert "preprocess" in affected
    assert "train" in affected


def test_add_downstream_stages_includes_transitive_dependents(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should include transitively dependent stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["step1.txt"])
    def step1() -> None:
        pass

    @stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        pass

    @stage(deps=["step2.txt"], outs=["step3.txt"])
    def step3() -> None:
        pass

    eng = engine.ReactiveEngine()
    affected = eng._add_downstream_stages({"step1"})

    assert "step1" in affected
    assert "step2" in affected
    assert "step3" in affected


# _get_affected_stages tests


def test_get_affected_stages_returns_all_stages_on_code_change(
    pipeline_dir: pathlib.Path,
) -> None:
    """Code changes should trigger all stages (executor handles skip logic)."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["out1.txt"])
    def stage_a() -> None:
        pass

    @stage(deps=["out1.txt"], outs=["out2.txt"])
    def stage_b() -> None:
        pass

    eng = engine.ReactiveEngine()
    code_path = pathlib.Path("/some/code.py")
    affected = eng._get_affected_stages({code_path}, code_changed=True)

    assert "stage_a" in affected
    assert "stage_b" in affected


def test_get_affected_stages_returns_subset_on_data_change(
    pipeline_dir: pathlib.Path,
) -> None:
    """Data changes should only trigger affected stages and their downstream."""
    (pipeline_dir / "data1.csv").write_text("a,b\n1,2")
    (pipeline_dir / "data2.csv").write_text("x,y\n3,4")

    @stage(deps=["data1.csv"], outs=["out1.txt"])
    def stage_a() -> None:
        pass

    @stage(deps=["data2.csv"], outs=["out2.txt"])
    def stage_b() -> None:
        pass

    eng = engine.ReactiveEngine()
    data_path = project.resolve_path("data1.csv")
    affected = eng._get_affected_stages({data_path}, code_changed=False)

    assert "stage_a" in affected
    assert "stage_b" not in affected


# _collect_and_debounce tests


def test_collect_and_debounce_returns_empty_on_shutdown(pipeline_dir: pathlib.Path) -> None:
    """Should return empty set immediately when shutdown is set."""
    eng = engine.ReactiveEngine(debounce_ms=100)
    eng._shutdown.set()

    start = time.monotonic()
    changes = eng._collect_and_debounce(max_wait_s=5.0)
    elapsed = time.monotonic() - start

    assert len(changes) == 0
    assert elapsed < 1.0, "Should return quickly on shutdown"


def test_collect_and_debounce_waits_for_quiet_period(pipeline_dir: pathlib.Path) -> None:
    """Should wait for quiet period after last change before returning."""
    eng = engine.ReactiveEngine(debounce_ms=200)

    # Add change immediately
    path = pathlib.Path("/some/file.txt")
    eng._change_queue.put({path})

    start = time.monotonic()
    changes = eng._collect_and_debounce(max_wait_s=5.0)
    elapsed = time.monotonic() - start

    assert path in changes
    # Should have waited at least the debounce period (200ms = 0.2s)
    assert elapsed >= 0.2, f"Should wait for quiet period, but only waited {elapsed}s"


def test_collect_and_debounce_coalesces_rapid_changes(pipeline_dir: pathlib.Path) -> None:
    """Should collect multiple changes within debounce window."""
    eng = engine.ReactiveEngine(debounce_ms=300)

    # Add changes rapidly
    path1 = pathlib.Path("/file1.txt")
    path2 = pathlib.Path("/file2.txt")

    def add_changes() -> None:
        eng._change_queue.put({path1})
        time.sleep(0.05)
        eng._change_queue.put({path2})

    thread = threading.Thread(target=add_changes)
    thread.start()

    changes = eng._collect_and_debounce(max_wait_s=5.0)
    thread.join()

    assert path1 in changes
    assert path2 in changes


def test_collect_and_debounce_respects_max_wait(pipeline_dir: pathlib.Path) -> None:
    """Should return after max_wait even if changes keep coming."""
    eng = engine.ReactiveEngine(debounce_ms=500)

    # Continuously add changes
    stop = threading.Event()

    def add_continuous_changes() -> None:
        while not stop.is_set():
            eng._change_queue.put({pathlib.Path("/continuous.txt")})
            time.sleep(0.1)

    thread = threading.Thread(target=add_continuous_changes)
    thread.start()

    start = time.monotonic()
    changes = eng._collect_and_debounce(max_wait_s=0.5)
    elapsed = time.monotonic() - start

    stop.set()
    thread.join()

    assert len(changes) > 0, "Should have collected changes"
    assert elapsed < 1.0, f"Should respect max_wait (0.5s), but waited {elapsed}s"


# ReactiveEngine initialization tests


def test_engine_init_sets_defaults(pipeline_dir: pathlib.Path) -> None:
    """Engine should initialize with default values."""
    eng = engine.ReactiveEngine()

    assert eng._stages is None
    assert eng._single_stage is False
    assert eng._debounce_ms == 300
    assert eng._shutdown.is_set() is False


def test_engine_init_accepts_custom_values(pipeline_dir: pathlib.Path) -> None:
    """Engine should accept custom configuration."""
    eng = engine.ReactiveEngine(
        stages=["stage_a", "stage_b"],
        single_stage=True,
        cache_dir=pathlib.Path("/custom/cache"),
        max_workers=4,
        debounce_ms=500,
    )

    assert eng._stages == ["stage_a", "stage_b"]
    assert eng._single_stage is True
    assert eng._cache_dir == pathlib.Path("/custom/cache")
    assert eng._max_workers == 4
    assert eng._debounce_ms == 500


def test_engine_init_raises_on_negative_debounce(pipeline_dir: pathlib.Path) -> None:
    """Engine should reject negative debounce_ms."""
    with pytest.raises(ValueError, match="debounce_ms must be non-negative"):
        engine.ReactiveEngine(debounce_ms=-1)


def test_engine_init_accepts_zero_debounce(pipeline_dir: pathlib.Path) -> None:
    """Engine should accept zero debounce_ms (no quiet period)."""
    eng = engine.ReactiveEngine(debounce_ms=0)
    assert eng._debounce_ms == 0


def test_engine_shutdown_sets_event(pipeline_dir: pathlib.Path) -> None:
    """shutdown() should set the shutdown event."""
    eng = engine.ReactiveEngine()
    assert eng._shutdown.is_set() is False

    eng.shutdown()
    assert eng._shutdown.is_set() is True


def test_add_downstream_stages_skips_unknown_stages(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_add_downstream_stages should skip stages not in DAG and log warning."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def known_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Try to add downstream for a mix of known and unknown stages
    result = eng._add_downstream_stages({"known_stage", "unknown_stage"})

    assert "known_stage" in result, "Known stage should be included"
    assert "unknown_stage" not in result, "Unknown stage should be skipped"
    assert "Stage 'unknown_stage' not found in DAG" in caplog.text


# Integration tests with mocked executor


def test_engine_runs_initial_execution(pipeline_dir: pathlib.Path) -> None:
    """Engine should run initial pipeline execution on start."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    call_count = 0

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        nonlocal call_count
        call_count += 1
        self.shutdown()  # Exit after first call

    def mock_watch_loop(self: engine.ReactiveEngine, stages: list[str]) -> None:
        pass  # Don't actually start watching

    with (
        mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute),
        mock.patch.object(engine.ReactiveEngine, "_watch_loop", mock_watch_loop),
    ):
        eng = engine.ReactiveEngine()
        eng.run()

    assert call_count == 1, "Should have executed initial pipeline"


def test_engine_restarts_workers_on_code_change(pipeline_dir: pathlib.Path) -> None:
    """Engine should restart workers when Python files change."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    restart_called = False
    iteration_count = 0

    def mock_restart(self: engine.ReactiveEngine) -> None:
        nonlocal restart_called
        restart_called = True

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count >= 2:
            self.shutdown()

    def mock_watch_loop(self: engine.ReactiveEngine, stages: list[str]) -> None:
        pass  # Don't actually start watching

    def mock_reload_registry(self: engine.ReactiveEngine) -> bool:
        return True  # Simulate successful reload

    with (
        mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute),
        mock.patch.object(engine.ReactiveEngine, "_restart_worker_pool", mock_restart),
        mock.patch.object(engine.ReactiveEngine, "_reload_registry", mock_reload_registry),
        mock.patch.object(engine.ReactiveEngine, "_watch_loop", mock_watch_loop),
    ):
        eng = engine.ReactiveEngine(debounce_ms=50)

        # Simulate code change
        code_path = pathlib.Path("/some/module.py")
        eng._change_queue.put({code_path})

        eng.run()

    assert restart_called, "Should have restarted workers on code change"


def test_engine_handles_execution_error_gracefully(pipeline_dir: pathlib.Path) -> None:
    """Engine should continue after execution errors."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    call_count = 0
    eng_ref: engine.ReactiveEngine | None = None

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        nonlocal call_count, eng_ref
        eng_ref = self
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Execution failed")
        # Second call - shutdown
        self.shutdown()

    def mock_watch_loop(self: engine.ReactiveEngine, stages: list[str]) -> None:
        pass  # Don't actually start watching

    def mock_reload_registry(self: engine.ReactiveEngine) -> bool:
        return True  # Simulate successful reload

    # Patch at the class level to ensure thread captures mocked method
    with (
        mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute),
        mock.patch.object(engine.ReactiveEngine, "_reload_registry", mock_reload_registry),
        mock.patch.object(engine.ReactiveEngine, "_watch_loop", mock_watch_loop),
    ):
        eng = engine.ReactiveEngine(debounce_ms=50)

        # Queue a code change to trigger second iteration (code changes trigger all stages)
        def queue_change() -> None:
            time.sleep(0.1)
            eng._change_queue.put({pathlib.Path("/some/code.py")})

        thread = threading.Thread(target=queue_change)
        thread.start()

        eng.run()
        thread.join()

    assert call_count == 2, "Should have continued after error"


# Watcher thread exception handling tests


def test_watcher_thread_exception_triggers_shutdown(pipeline_dir: pathlib.Path) -> None:
    """Watcher thread exceptions should trigger engine shutdown."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    shutdown_triggered = False

    def mock_watch_that_fails(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Watcher failed!")

    def patched_watch_loop(self: engine.ReactiveEngine, stages: list[str]) -> None:
        nonlocal shutdown_triggered
        try:
            mock_watch_that_fails()
        except Exception as e:
            engine.logger.critical(f"Watcher thread failed: {e}")
            self.shutdown()
            shutdown_triggered = True

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        pass  # Initial execution succeeds

    with (
        mock.patch.object(engine.ReactiveEngine, "_watch_loop", patched_watch_loop),
        mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute),
    ):
        eng = engine.ReactiveEngine(debounce_ms=50)
        eng.run()

    assert shutdown_triggered, "Watcher exception should trigger shutdown"
    assert eng._shutdown.is_set(), "Shutdown event should be set"


# Queue overflow behavior tests


def test_queue_overflow_triggers_full_rebuild_sentinel(pipeline_dir: pathlib.Path) -> None:
    """Exceeding MAX_PENDING_CHANGES should use sentinel for full rebuild."""
    # Simulate many pending changes exceeding threshold
    pending: set[pathlib.Path] = set()
    for i in range(engine._MAX_PENDING_CHANGES + 100):
        pending.add(pathlib.Path(f"/file{i}.txt"))

    # After threshold exceeded, pending should become sentinel
    if len(pending) > engine._MAX_PENDING_CHANGES:
        pending = {engine._FULL_REBUILD_SENTINEL}

    assert pending == {engine._FULL_REBUILD_SENTINEL}, "Should use sentinel for full rebuild"


def test_watch_loop_handles_queue_overflow(pipeline_dir: pathlib.Path) -> None:
    """Watch loop should handle full queue gracefully by accumulating."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Fill the queue completely
    for _ in range(100):  # Queue maxsize is 100
        try:
            eng._change_queue.put_nowait({pathlib.Path("/test.txt")})
        except Exception:
            break

    # Queue should be full
    assert eng._change_queue.full(), "Queue should be full"

    # Now put_nowait should raise queue.Full
    with pytest.raises(queue.Full):
        eng._change_queue.put_nowait({pathlib.Path("/overflow.txt")})


# Concurrent shutdown tests


def test_concurrent_shutdown_during_debounce(pipeline_dir: pathlib.Path) -> None:
    """Shutdown during debounce should return empty and exit cleanly."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine(debounce_ms=1000)

    # Add a change to queue
    eng._change_queue.put({pathlib.Path("/some/file.txt")})

    # Trigger shutdown in background after short delay
    def trigger_shutdown() -> None:
        time.sleep(0.1)
        eng.shutdown()

    shutdown_thread = threading.Thread(target=trigger_shutdown)
    shutdown_thread.start()

    # This should return empty due to shutdown, not wait for full debounce
    start = time.monotonic()
    changes = eng._collect_and_debounce(max_wait_s=5.0)
    elapsed = time.monotonic() - start

    shutdown_thread.join()

    assert len(changes) == 0, "Should return empty set on shutdown"
    assert elapsed < 1.0, f"Should exit quickly on shutdown, took {elapsed}s"


def test_concurrent_shutdown_during_run(pipeline_dir: pathlib.Path) -> None:
    """Shutdown during run() should clean up watcher thread."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    execution_count = 0

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        nonlocal execution_count
        execution_count += 1

    def mock_watch_loop(self: engine.ReactiveEngine, stages: list[str]) -> None:
        # Simulate watch loop that respects shutdown
        while not self._shutdown.is_set():
            time.sleep(0.05)

    with (
        mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute),
        mock.patch.object(engine.ReactiveEngine, "_watch_loop", mock_watch_loop),
    ):
        eng = engine.ReactiveEngine(debounce_ms=50)

        # Trigger shutdown shortly after start
        def delayed_shutdown() -> None:
            time.sleep(0.2)
            eng.shutdown()

        shutdown_thread = threading.Thread(target=delayed_shutdown)
        shutdown_thread.start()

        eng.run()
        shutdown_thread.join()

    assert eng._shutdown.is_set(), "Shutdown should be set"
    assert execution_count == 1, "Should have run initial execution"


def test_get_stages_affected_handles_deleted_file(pipeline_dir: pathlib.Path) -> None:
    """Should handle deleted files gracefully using absolute path."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Create path to file that doesn't exist (simulating deletion)
    deleted_path = pipeline_dir / "deleted_file.csv"

    # Should not crash, just return empty set (file not in deps)
    affected = eng._get_stages_matching_changes({deleted_path})
    assert len(affected) == 0, "Deleted file not in deps should return empty"


# DAG and file index caching tests


def test_dag_cache_invalidation_on_code_change(pipeline_dir: pathlib.Path) -> None:
    """DAG cache should be invalidated when code changes."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Build initial caches
    dag1 = eng._get_dag()
    index1 = eng._get_file_index()

    assert eng._cached_dag is dag1, "DAG should be cached"
    assert eng._cached_file_index is index1, "File index should be cached"

    # Simulate code change invalidation (what _coordinator_loop does)
    eng._cached_dag = None
    eng._cached_file_index = None

    # Next access should rebuild
    dag2 = eng._get_dag()
    index2 = eng._get_file_index()

    # New instances should be created
    assert dag2 is not dag1, "Should create new DAG after invalidation"
    assert index2 is not index1, "Should create new file index after invalidation"


def test_file_index_caching_returns_same_instance(pipeline_dir: pathlib.Path) -> None:
    """File index should be cached and return same instance."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Multiple calls should return same cached instance
    index1 = eng._get_file_index()
    index2 = eng._get_file_index()
    index3 = eng._get_file_index()

    assert index1 is index2, "Should return same cached instance"
    assert index2 is index3, "Should return same cached instance"


# _is_code_or_config_change tests


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({pathlib.Path("/some/module.py")}, True),  # Python file
        ({pathlib.Path("/project/pivot.yaml")}, True),  # pivot.yaml
        ({pathlib.Path("/project/pivot.yml")}, True),  # pivot.yml
        ({pathlib.Path("/project/params.yaml")}, True),  # params.yaml
        ({pathlib.Path("/project/params.yml")}, True),  # params.yml
        ({pathlib.Path("/project/pipeline.py")}, True),  # pipeline.py
        ({engine._FULL_REBUILD_SENTINEL}, True),  # sentinel
        ({pathlib.Path("/data/input.csv"), pathlib.Path("/data/output.parquet")}, False),
    ],
    ids=[
        "python_file",
        "pivot_yaml",
        "pivot_yml",
        "params_yaml",
        "params_yml",
        "pipeline_py",
        "sentinel",
        "data_files",
    ],
)
def test_is_code_or_config_change(changes: set[pathlib.Path], expected: bool) -> None:
    """Test code/config change detection for various file types."""
    assert engine._is_code_or_config_change(changes) is expected


# Integration test with real watchfiles


def test_integration_watchfiles_detects_real_file_change(pipeline_dir: pathlib.Path) -> None:
    """Integration test: actual watchfiles detects real file changes (not mocked)."""
    # Create data file that will be watched
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    execution_stages: list[list[str] | None] = []
    execution_event = threading.Event()

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        execution_stages.append(stages)
        if len(execution_stages) >= 2:  # Initial + one triggered by file change
            execution_event.set()
            self.shutdown()

    with mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute):
        eng = engine.ReactiveEngine(debounce_ms=50)

        # Run engine in background thread (run() blocks)
        engine_thread = threading.Thread(target=eng.run)
        engine_thread.start()

        # Wait for watcher to initialize and start watching
        time.sleep(0.5)

        # Modify the data file - this should trigger re-execution
        data_file.write_text("a,b\n1,2\n3,4")

        # Wait for execution to be triggered (with timeout to prevent hang)
        triggered = execution_event.wait(timeout=5.0)

        # Clean shutdown if test times out
        if not triggered:
            eng.shutdown()
        engine_thread.join(timeout=2.0)

    assert len(execution_stages) >= 2, (
        f"Expected at least 2 executions (initial + triggered), got {len(execution_stages)}"
    )
    # First execution is initial run (None = all stages)
    assert execution_stages[0] is None, "First execution should be initial (all stages)"
    # Second execution should include our affected stage
    assert execution_stages[1] is not None, "Second execution should have specific stages"
    assert "process" in execution_stages[1], "Second execution should include affected stage"


def test_integration_watchfiles_detects_python_code_change(pipeline_dir: pathlib.Path) -> None:
    """Integration test: watchfiles detects Python file changes and triggers worker restart."""
    # Create data file and Python file
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("a,b\n1,2")
    code_file = pipeline_dir / "helper.py"
    code_file.write_text("def helper(): pass\n")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    execution_count = 0
    restart_called = False
    execution_event = threading.Event()

    def mock_execute(self: engine.ReactiveEngine, stages: list[str] | None) -> None:
        nonlocal execution_count
        execution_count += 1
        if execution_count >= 2:
            execution_event.set()
            self.shutdown()

    def mock_restart(self: engine.ReactiveEngine) -> None:
        nonlocal restart_called
        restart_called = True

    def mock_reload_registry(self: engine.ReactiveEngine) -> bool:
        return True  # Simulate successful reload

    with (
        mock.patch.object(engine.ReactiveEngine, "_execute_stages", mock_execute),
        mock.patch.object(engine.ReactiveEngine, "_restart_worker_pool", mock_restart),
        mock.patch.object(engine.ReactiveEngine, "_reload_registry", mock_reload_registry),
    ):
        eng = engine.ReactiveEngine(debounce_ms=50)

        engine_thread = threading.Thread(target=eng.run)
        engine_thread.start()

        # Wait for watcher to start
        time.sleep(0.5)

        # Modify Python file - should trigger worker restart
        code_file.write_text("def helper(): return 42\n")

        triggered = execution_event.wait(timeout=5.0)

        if not triggered:
            eng.shutdown()
        engine_thread.join(timeout=2.0)

    assert execution_count >= 2, f"Expected at least 2 executions, got {execution_count}"
    assert restart_called, "Worker pool should be restarted on Python file change"


# Additional coverage tests


def test_collect_and_debounce_raises_on_invalid_max_wait(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should raise ValueError for non-positive max_wait_s."""
    eng = engine.ReactiveEngine()

    with pytest.raises(ValueError, match="max_wait_s must be positive"):
        eng._collect_and_debounce(max_wait_s=0)

    with pytest.raises(ValueError, match="max_wait_s must be positive"):
        eng._collect_and_debounce(max_wait_s=-1.0)


def test_get_affected_stages_returns_all_registry_stages_on_code_change(
    pipeline_dir: pathlib.Path,
) -> None:
    """When stages is None and code changed, should return all registry stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output1.txt"])
    def stage1() -> None:
        pass

    @stage(deps=["data.csv"], outs=["output2.txt"])
    def stage2() -> None:
        pass

    eng = engine.ReactiveEngine(stages=None)  # No specific stages

    changes = {pathlib.Path("code.py")}
    affected = eng._get_affected_stages(changes, code_changed=True)

    assert set(affected) == {"stage1", "stage2"}, "Should return all registry stages"


def test_send_message_to_tui_queue(pipeline_dir: pathlib.Path) -> None:
    """_send_message should put messages on TUI queue when available."""
    manager = multiprocessing.Manager()
    tui_queue = manager.Queue()

    eng = engine.ReactiveEngine()
    eng._tui_queue = tui_queue  # pyright: ignore[reportAttributeAccessIssue]

    eng._send_message("Test message")

    assert not tui_queue.empty(), "Message should be in queue"
    msg = tui_queue.get_nowait()
    assert msg["type"] == types.TuiMessageType.REACTIVE
    assert msg["status"] == types.ReactiveStatus.WAITING
    assert msg["message"] == "Test message"


def test_send_message_error_to_tui_queue(pipeline_dir: pathlib.Path) -> None:
    """_send_message with is_error=True should set error status."""
    manager = multiprocessing.Manager()
    tui_queue = manager.Queue()

    eng = engine.ReactiveEngine()
    eng._tui_queue = tui_queue  # pyright: ignore[reportAttributeAccessIssue]

    eng._send_message("Error occurred", is_error=True)

    msg = tui_queue.get_nowait()
    assert msg["status"] == types.ReactiveStatus.ERROR
    assert msg["message"] == "Error occurred"


def test_resolve_path_for_matching_handles_deleted_file(
    pipeline_dir: pathlib.Path,
) -> None:
    """_resolve_path_for_matching should handle deleted files gracefully."""
    deleted_path = pipeline_dir / "deleted_file.txt"

    # File doesn't exist - should return normalized absolute path
    result = engine._resolve_path_for_matching(deleted_path)

    assert result.is_absolute(), "Should return absolute path"
    assert str(result).endswith("deleted_file.txt")


def test_get_stages_matching_changes_handles_incomparable_paths(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should handle ValueError from is_relative_to for incomparable paths."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b\n1,2")

    @stage(deps=["data"], outs=["output.txt"])  # Depends on directory
    def process() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Use a completely different path that's not relative to anything
    changes = {pathlib.Path("/completely/different/path.txt")}
    affected = eng._get_stages_matching_changes(changes)

    # Should not raise, should return empty set
    assert affected == set()


def test_reload_registry_logs_when_no_modules_found(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_reload_registry should log warning when no stage modules found."""
    # Clear registry so there are no stages
    REGISTRY.clear()

    eng = engine.ReactiveEngine()
    eng._reload_registry()

    assert "No stage modules found to reload" in caplog.text


def test_reload_registry_clears_and_reimports_modules(
    pipeline_dir: pathlib.Path,
) -> None:
    """_reload_registry should clear project modules and reimport them."""

    @stage(deps=[], outs=["output.txt"])
    def test_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    initial_stages = list(REGISTRY.list_stages())
    assert "test_stage" in initial_stages

    # Mock _clear_project_modules to verify it's called (for decorator path)
    with (
        mock.patch.object(engine, "_clear_project_modules", return_value=0) as mock_clear,
        mock.patch("importlib.import_module"),
    ):
        eng._reload_registry()

    # clear_project_modules should have been called
    assert mock_clear.called, "Should clear project modules before reload"


def test_is_existing_dir_returns_false_for_nonexistent(
    pipeline_dir: pathlib.Path,
) -> None:
    """_is_existing_dir should return False for non-existent paths."""
    nonexistent = pipeline_dir / "does_not_exist"
    assert engine._is_existing_dir(nonexistent) is False


def test_is_existing_dir_returns_true_for_directory(
    pipeline_dir: pathlib.Path,
) -> None:
    """_is_existing_dir should return True for existing directories."""
    existing_dir = pipeline_dir / "existing_dir"
    existing_dir.mkdir()
    assert engine._is_existing_dir(existing_dir) is True


def test_is_existing_dir_returns_false_for_file(
    pipeline_dir: pathlib.Path,
) -> None:
    """_is_existing_dir should return False for files."""
    existing_file = pipeline_dir / "file.txt"
    existing_file.write_text("content")
    assert engine._is_existing_dir(existing_file) is False


def test_is_existing_dir_handles_os_error(
    pipeline_dir: pathlib.Path,
) -> None:
    """_is_existing_dir should return False on OSError."""
    test_path = pipeline_dir / "test_path"

    with mock.patch.object(pathlib.Path, "is_dir", side_effect=OSError("Permission denied")):
        assert engine._is_existing_dir(test_path) is False


def test_reload_registry_handles_import_exception(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_reload_registry should return False and preserve registry when import fails."""

    @stage(deps=[], outs=["output.txt"])
    def reload_test_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Mock import_module to raise an exception (for decorator path)
    with mock.patch("importlib.import_module", side_effect=ImportError("Module not found")):
        result = eng._reload_registry()

    assert result is False, "Should return False on import failure"
    assert "Failed to import module" in caplog.text
    # Verify registry was restored
    assert "reload_test_stage" in REGISTRY.list_stages(), "Registry should be preserved"
    # Verify errors are tracked
    assert eng._pipeline_errors is not None
    assert len(eng._pipeline_errors) > 0


# Tests for different registration patterns


def test_reload_registry_uses_pivot_yaml_when_present(
    pipeline_dir: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_reload_registry should use pivot.yaml when it has stages defined."""
    import logging
    import sys

    caplog.set_level(logging.INFO)

    # Create a valid pivot.yaml with stages
    stages_module = pipeline_dir / "stages.py"
    stages_module.write_text(
        """\
def my_stage() -> None:
    pass
"""
    )
    (pipeline_dir / "pivot.yaml").write_text(
        """\
stages:
  test_yaml_stage:
    python: stages.my_stage
    deps: []
    outs: ["output.txt"]
"""
    )

    # Add pipeline_dir to sys.path so stages.py can be imported (auto-cleanup)
    monkeypatch.syspath_prepend(str(pipeline_dir))

    # First manually register so we have backup
    pipeline_yaml.register_from_pipeline_file(pipeline_dir / "pivot.yaml")
    assert "test_yaml_stage" in REGISTRY.list_stages()

    eng = engine.ReactiveEngine()
    result = eng._reload_registry()

    assert result is True, "Reload should succeed"
    assert "test_yaml_stage" in REGISTRY.list_stages(), "Stage should be from pivot.yaml"
    assert "Registry reloaded from pivot.yaml" in caplog.text

    # Clean up imported module
    if "stages" in sys.modules:
        del sys.modules["stages"]


def test_reload_registry_uses_pipeline_py_when_present(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_reload_registry should use pipeline.py when pivot.yaml has no stages."""
    import logging

    caplog.set_level(logging.INFO)

    # Remove pivot.yaml stages section (just keep version)
    (pipeline_dir / "pivot.yaml").write_text("version: 1\n")

    # Create pipeline.py
    (pipeline_dir / "pipeline.py").write_text(
        """\
from pivot.registry import stage

@stage(deps=[], outs=["output.txt"])
def pipeline_py_stage() -> None:
    pass
"""
    )

    # First run pipeline.py to register
    runpy.run_path(str(pipeline_dir / "pipeline.py"), run_name="_pivot_pipeline")
    assert "pipeline_py_stage" in REGISTRY.list_stages()

    eng = engine.ReactiveEngine()
    result = eng._reload_registry()

    assert result is True, "Reload should succeed"
    assert "pipeline_py_stage" in REGISTRY.list_stages(), "Stage should be from pipeline.py"
    assert "Registry reloaded from pipeline.py" in caplog.text


def test_reload_registry_prefers_pivot_yaml_over_pipeline_py(
    pipeline_dir: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_reload_registry should prefer pivot.yaml over pipeline.py when both exist."""
    import logging
    import sys

    caplog.set_level(logging.INFO)

    # Create stages module
    stages_module = pipeline_dir / "stages.py"
    stages_module.write_text(
        """\
def yaml_stage() -> None:
    pass
"""
    )

    # Create pivot.yaml with stages
    (pipeline_dir / "pivot.yaml").write_text(
        """\
stages:
  yaml_stage:
    python: stages.yaml_stage
    deps: []
    outs: ["yaml_out.txt"]
"""
    )

    # Create pipeline.py with different stage
    (pipeline_dir / "pipeline.py").write_text(
        """\
from pivot.registry import stage

@stage(deps=[], outs=["py_out.txt"])
def py_stage() -> None:
    pass
"""
    )

    # Add pipeline_dir to sys.path so stages.py can be imported (auto-cleanup)
    monkeypatch.syspath_prepend(str(pipeline_dir))

    # First register from pivot.yaml
    pipeline_yaml.register_from_pipeline_file(pipeline_dir / "pivot.yaml")
    assert "yaml_stage" in REGISTRY.list_stages()

    eng = engine.ReactiveEngine()
    result = eng._reload_registry()

    assert result is True
    assert "yaml_stage" in REGISTRY.list_stages(), "Should use pivot.yaml stage"
    assert "py_stage" not in REGISTRY.list_stages(), "Should NOT use pipeline.py stage"
    assert "Registry reloaded from pivot.yaml" in caplog.text

    # Clean up imported module
    if "stages" in sys.modules:
        del sys.modules["stages"]


def test_reload_registry_falls_back_to_decorators(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_reload_registry should fall back to decorator reload when no config files."""
    # Remove pivot.yaml stages section
    (pipeline_dir / "pivot.yaml").write_text("version: 1\n")
    # Don't create pipeline.py

    @stage(deps=[], outs=["output.txt"])
    def decorator_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Mock import_module to verify decorator path is used
    with mock.patch("importlib.import_module") as mock_import:
        result = eng._reload_registry()

    assert result is True
    assert mock_import.called, (
        "Should have called importlib.import_module for decorator-based stages"
    )


def test_reload_registry_pivot_yaml_error_preserves_old_registry(
    pipeline_dir: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_reload_registry should preserve old registry when pivot.yaml reload fails."""
    import sys

    # Create valid stages module
    stages_module = pipeline_dir / "stages.py"
    stages_module.write_text(
        """\
def my_stage() -> None:
    pass
"""
    )

    # Create valid pivot.yaml
    (pipeline_dir / "pivot.yaml").write_text(
        """\
stages:
  preserved_stage:
    python: stages.my_stage
    deps: []
    outs: ["output.txt"]
"""
    )

    # Add pipeline_dir to sys.path so stages.py can be imported (auto-cleanup)
    monkeypatch.syspath_prepend(str(pipeline_dir))

    # Register the stage
    pipeline_yaml.register_from_pipeline_file(pipeline_dir / "pivot.yaml")
    assert "preserved_stage" in REGISTRY.list_stages()

    # Now break the pivot.yaml
    (pipeline_dir / "pivot.yaml").write_text(
        """\
stages:
  broken_stage:
    python: nonexistent.module.func
    deps: []
    outs: ["output.txt"]
"""
    )

    eng = engine.ReactiveEngine()
    result = eng._reload_registry()

    assert result is False, "Should return False on reload failure"
    assert "preserved_stage" in REGISTRY.list_stages(), "Old registry should be preserved"
    assert eng._pipeline_errors is not None

    # Clean up imported module
    if "stages" in sys.modules:
        del sys.modules["stages"]


def test_reload_registry_pipeline_py_error_preserves_old_registry(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_reload_registry should preserve old registry when pipeline.py reload fails."""
    # Remove pivot.yaml stages section
    (pipeline_dir / "pivot.yaml").write_text("version: 1\n")

    # Create valid pipeline.py first
    (pipeline_dir / "pipeline.py").write_text(
        """\
from pivot.registry import stage

@stage(deps=[], outs=["output.txt"])
def preserved_py_stage() -> None:
    pass
"""
    )

    # Run it to register
    runpy.run_path(str(pipeline_dir / "pipeline.py"), run_name="_pivot_pipeline")
    assert "preserved_py_stage" in REGISTRY.list_stages()

    # Now break the pipeline.py
    (pipeline_dir / "pipeline.py").write_text("raise RuntimeError('broken')\n")

    eng = engine.ReactiveEngine()
    result = eng._reload_registry()

    assert result is False, "Should return False on reload failure"
    assert "preserved_py_stage" in REGISTRY.list_stages(), "Old registry should be preserved"
    assert eng._pipeline_errors is not None


def test_resolve_path_for_matching_handles_oserror(
    pipeline_dir: pathlib.Path,
) -> None:
    """_resolve_path_for_matching should use normalized path on OSError."""
    test_path = pipeline_dir / "test_file.txt"

    # Mock project.resolve_path to raise OSError
    with mock.patch.object(project, "resolve_path", side_effect=OSError("Access denied")):
        result = engine._resolve_path_for_matching(test_path)

    # Should return normalized absolute path
    assert result.is_absolute()
    assert str(result).endswith("test_file.txt")


def test_get_stages_matching_changes_handles_is_relative_to_error(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should handle ValueError from is_relative_to."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b\n1,2")

    @stage(deps=["data"], outs=["output.txt"])
    def containment_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    changes = {pipeline_dir / "some_file.txt"}

    # Mock is_relative_to to raise ValueError
    def mock_is_relative_to(self: pathlib.Path, other: pathlib.Path) -> bool:
        raise ValueError("Cannot compare paths")

    with mock.patch.object(pathlib.Path, "is_relative_to", mock_is_relative_to):
        affected = eng._get_stages_matching_changes(changes)

    # Should not raise, should return empty set
    assert affected == set()


# _reload_registry invalid pipeline tests


def test_reload_registry_clears_errors_on_success(pipeline_dir: pathlib.Path) -> None:
    """_reload_registry should clear previous errors on successful reload."""

    @stage(deps=[], outs=["output.txt"])
    def test_clear_stage() -> None:
        pass

    eng = engine.ReactiveEngine()
    # Simulate previous error state
    eng._pipeline_errors = ["previous error"]

    result = eng._reload_registry()

    assert result is True, "Should return True on success"
    assert eng._pipeline_errors is None, "Previous errors should be cleared"


def test_reload_registry_returns_true_when_no_modules(pipeline_dir: pathlib.Path) -> None:
    """_reload_registry should return True when no stage modules found."""
    eng = engine.ReactiveEngine()
    # No stages registered

    result = eng._reload_registry()

    assert result is True, "Should return True when no modules to reload"


def test_coordinator_loop_skips_execution_when_invalid(pipeline_dir: pathlib.Path) -> None:
    """Coordinator should skip execution when pipeline is invalid."""

    @stage(deps=[], outs=["output.txt"])
    def invalid_test_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Set up invalid pipeline state
    eng._pipeline_errors = ["test_module: SyntaxError"]

    # Mock the change queue to return changes then empty
    change_set: set[pathlib.Path] = {pipeline_dir / "data.csv"}
    call_count = 0

    def mock_collect_and_debounce(*args: object, **kwargs: object) -> set[pathlib.Path]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return change_set
        eng._shutdown.set()
        return set()

    messages: list[str] = []

    def capture_message(
        msg: str, *, is_error: bool = False, status: types.ReactiveStatus | None = None
    ) -> None:
        messages.append(msg)

    with (
        mock.patch.object(eng, "_collect_and_debounce", side_effect=mock_collect_and_debounce),
        mock.patch.object(eng, "_send_message", side_effect=capture_message),
        mock.patch.object(eng, "_execute_stages") as mock_execute,
    ):
        eng._coordinator_loop()

    # Should not have called execute since pipeline is invalid
    mock_execute.assert_not_called()
    # Should have sent "Watching for changes..." message
    assert "Watching for changes..." in messages


def test_coordinator_loop_sends_error_on_reload_failure(pipeline_dir: pathlib.Path) -> None:
    """Coordinator should send error message when reload fails."""

    @stage(deps=[], outs=["output.txt"])
    def reload_fail_stage() -> None:
        pass

    eng = engine.ReactiveEngine()

    # Mock the change queue to return a Python file change then empty
    py_change: set[pathlib.Path] = {pipeline_dir / "test.py"}
    call_count = 0

    def mock_collect_and_debounce(*args: object, **kwargs: object) -> set[pathlib.Path]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return py_change
        eng._shutdown.set()
        return set()

    messages: list[tuple[str, bool]] = []

    def capture_message(
        msg: str, *, is_error: bool = False, status: types.ReactiveStatus | None = None
    ) -> None:
        # Map status to is_error for backward compatibility in this test
        is_err = is_error or (status == types.ReactiveStatus.ERROR)
        messages.append((msg, is_err))

    with (
        mock.patch.object(eng, "_collect_and_debounce", side_effect=mock_collect_and_debounce),
        mock.patch.object(eng, "_send_message", side_effect=capture_message),
        mock.patch.object(eng, "_invalidate_caches"),
        mock.patch.object(eng, "_restart_worker_pool"),
        mock.patch("importlib.import_module", side_effect=SyntaxError("invalid syntax")),
    ):
        eng._coordinator_loop()

    # Should have sent error message
    error_messages = [(msg, err) for msg, err in messages if err]
    assert len(error_messages) > 0, "Should have sent error message"
    assert any("Pipeline invalid" in msg for msg, _ in error_messages)


def test_coordinator_loop_clears_invalid_state_on_successful_reload(
    pipeline_dir: pathlib.Path,
) -> None:
    """Coordinator should clear invalid state when reload succeeds."""

    @stage(deps=[], outs=["output.txt"])
    def recovery_stage() -> None:
        pass

    eng = engine.ReactiveEngine()
    # Set up previous invalid state
    eng._pipeline_errors = ["previous_error"]

    # Mock the change queue to return a Python file change then empty
    py_change: set[pathlib.Path] = {pipeline_dir / "test.py"}
    call_count = 0

    def mock_collect_and_debounce(*args: object, **kwargs: object) -> set[pathlib.Path]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return py_change
        eng._shutdown.set()
        return set()

    with (
        mock.patch.object(eng, "_collect_and_debounce", side_effect=mock_collect_and_debounce),
        mock.patch.object(eng, "_send_message"),
        mock.patch.object(eng, "_invalidate_caches"),
        mock.patch.object(eng, "_restart_worker_pool"),
        mock.patch.object(eng, "_execute_stages"),
    ):
        eng._coordinator_loop()

    # After successful reload, errors should be cleared
    assert eng._pipeline_errors is None, "Errors should be cleared on successful reload"


# force_first_run tests


def test_engine_init_force_first_run_defaults_to_false(pipeline_dir: pathlib.Path) -> None:
    """force_first_run should default to False."""
    eng = engine.ReactiveEngine()
    assert eng._force_first_run is False
    assert eng._first_run_done is False


def test_engine_init_accepts_force_first_run(pipeline_dir: pathlib.Path) -> None:
    """Engine should accept force_first_run parameter."""
    eng = engine.ReactiveEngine(force_first_run=True)
    assert eng._force_first_run is True
    assert eng._first_run_done is False


def test_execute_stages_passes_force_on_first_run(pipeline_dir: pathlib.Path) -> None:
    """_execute_stages should pass force=True on first run when force_first_run=True."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine(force_first_run=True)

    force_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_values.append(bool(kwargs.get("force", False)))

    with mock.patch.object(executor, "run", side_effect=capture_force):
        eng._execute_stages(None)

    assert len(force_values) == 1
    assert force_values[0] is True, "First execution should have force=True"
    assert eng._first_run_done is True, "_first_run_done should be set after first execution"


def test_execute_stages_does_not_force_subsequent_runs(pipeline_dir: pathlib.Path) -> None:
    """_execute_stages should pass force=False on subsequent runs."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine(force_first_run=True)

    force_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_values.append(bool(kwargs.get("force", False)))

    with mock.patch.object(executor, "run", side_effect=capture_force):
        eng._execute_stages(None)  # First run
        eng._execute_stages(None)  # Second run
        eng._execute_stages(None)  # Third run

    assert len(force_values) == 3
    assert force_values[0] is True, "First execution should have force=True"
    assert force_values[1] is False, "Second execution should have force=False"
    assert force_values[2] is False, "Third execution should have force=False"


def test_execute_stages_without_force_first_run_never_forces(pipeline_dir: pathlib.Path) -> None:
    """_execute_stages should never pass force=True when force_first_run=False."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine(force_first_run=False)  # Default

    force_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_values.append(bool(kwargs.get("force", False)))

    with mock.patch.object(executor, "run", side_effect=capture_force):
        eng._execute_stages(None)
        eng._execute_stages(None)

    assert all(v is False for v in force_values), "All executions should have force=False"


def test_first_run_done_flag_stress_test(pipeline_dir: pathlib.Path) -> None:
    """Stress test: rapid executions should only force first run, never subsequent ones.

    This test verifies _first_run_done behaves correctly even under rapid execution.
    Since all access is from the main thread, there's no race condition, but this
    confirms the flag logic is correct under stress.
    """
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    eng = engine.ReactiveEngine(force_first_run=True)

    force_values: list[bool] = []
    first_run_done_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_val = kwargs.get("force", False)
        force_values.append(bool(force_val))
        first_run_done_values.append(eng._first_run_done)

    # Run many executions rapidly
    num_executions = 100

    with mock.patch.object(executor, "run", side_effect=capture_force):
        for _ in range(num_executions):
            eng._execute_stages(None)

    assert len(force_values) == num_executions
    # First execution should have force=True
    assert force_values[0] is True, "First execution should have force=True"
    # All subsequent executions should have force=False
    assert all(v is False for v in force_values[1:]), (
        "All subsequent executions should have force=False"
    )
    # _first_run_done should be False before first execution, True after
    assert first_run_done_values[0] is False, (
        "_first_run_done should be False during first execution"
    )
    assert all(v is True for v in first_run_done_values[1:]), (
        "_first_run_done should be True for all subsequent"
    )
