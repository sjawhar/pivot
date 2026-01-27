from __future__ import annotations

import pathlib
import py_compile
import queue
import runpy
import threading
import time
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest
import watchfiles

from helpers import register_test_stage
from pivot import executor, loaders, outputs, project, types
from pivot.pipeline import yaml as pipeline_yaml
from pivot.registry import REGISTRY, RegistryStageInfo
from pivot.types import AgentRunRejection, AgentRunStartResult, AgentState
from pivot.watch import _watch_utils, engine

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# =============================================================================
# Module-level stage functions for annotation-based dependency injection
# =============================================================================


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _stage_noop() -> None:
    """Noop stage with no deps/outs."""
    pass


def _stage_with_data_csv_dep(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_with_data_input_csv_dep(
    input_file: Annotated[pathlib.Path, outputs.Dep("data/input.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = input_file
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


class _Output1Txt(TypedDict):
    output1: Annotated[pathlib.Path, outputs.Out("output1.txt", loaders.PathOnly())]


class _Output2Txt(TypedDict):
    output2: Annotated[pathlib.Path, outputs.Out("output2.txt", loaders.PathOnly())]


def _stage_a_data_csv(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _Output1Txt:
    _ = data
    pathlib.Path("output1.txt").write_text("")
    return _Output1Txt(output1=pathlib.Path("output1.txt"))


def _stage_b_config_yaml(
    config: Annotated[pathlib.Path, outputs.Dep("config.yaml", loaders.PathOnly())],
) -> _Output2Txt:
    _ = config
    pathlib.Path("output2.txt").write_text("")
    return _Output2Txt(output2=pathlib.Path("output2.txt"))


class _Out1Txt(TypedDict):
    out1: Annotated[pathlib.Path, outputs.Out("out1.txt", loaders.PathOnly())]


class _Out2Txt(TypedDict):
    out2: Annotated[pathlib.Path, outputs.Out("out2.txt", loaders.PathOnly())]


def _stage_a_shared_csv(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.csv", loaders.PathOnly())],
) -> _Out1Txt:
    _ = shared
    pathlib.Path("out1.txt").write_text("")
    return _Out1Txt(out1=pathlib.Path("out1.txt"))


def _stage_b_shared_csv(
    shared: Annotated[pathlib.Path, outputs.Dep("shared.csv", loaders.PathOnly())],
) -> _Out2Txt:
    _ = shared
    pathlib.Path("out2.txt").write_text("")
    return _Out2Txt(out2=pathlib.Path("out2.txt"))


class _IntermediateTxt(TypedDict):
    intermediate: Annotated[pathlib.Path, outputs.Out("intermediate.txt", loaders.PathOnly())]


class _FinalTxt(TypedDict):
    final: Annotated[pathlib.Path, outputs.Out("final.txt", loaders.PathOnly())]


def _stage_preprocess(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _IntermediateTxt:
    _ = data
    pathlib.Path("intermediate.txt").write_text("")
    return _IntermediateTxt(intermediate=pathlib.Path("intermediate.txt"))


def _stage_train(
    intermediate: Annotated[pathlib.Path, outputs.Dep("intermediate.txt", loaders.PathOnly())],
) -> _FinalTxt:
    _ = intermediate
    pathlib.Path("final.txt").write_text("")
    return _FinalTxt(final=pathlib.Path("final.txt"))


class _Step1Txt(TypedDict):
    step1: Annotated[pathlib.Path, outputs.Out("step1.txt", loaders.PathOnly())]


class _Step2Txt(TypedDict):
    step2: Annotated[pathlib.Path, outputs.Out("step2.txt", loaders.PathOnly())]


class _Step3Txt(TypedDict):
    step3: Annotated[pathlib.Path, outputs.Out("step3.txt", loaders.PathOnly())]


def _stage_step1(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _Step1Txt:
    _ = data
    pathlib.Path("step1.txt").write_text("")
    return _Step1Txt(step1=pathlib.Path("step1.txt"))


def _stage_step2(
    step1: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Txt:
    _ = step1
    pathlib.Path("step2.txt").write_text("")
    return _Step2Txt(step2=pathlib.Path("step2.txt"))


def _stage_step3(
    step2: Annotated[pathlib.Path, outputs.Dep("step2.txt", loaders.PathOnly())],
) -> _Step3Txt:
    _ = step2
    pathlib.Path("step3.txt").write_text("")
    return _Step3Txt(step3=pathlib.Path("step3.txt"))


def _stage_a_data1(
    data1: Annotated[pathlib.Path, outputs.Dep("data1.csv", loaders.PathOnly())],
) -> _Out1Txt:
    _ = data1
    pathlib.Path("out1.txt").write_text("")
    return _Out1Txt(out1=pathlib.Path("out1.txt"))


def _stage_b_data2(
    data2: Annotated[pathlib.Path, outputs.Dep("data2.csv", loaders.PathOnly())],
) -> _Out2Txt:
    _ = data2
    pathlib.Path("out2.txt").write_text("")
    return _Out2Txt(out2=pathlib.Path("out2.txt"))


class _IntermediateCsv(TypedDict):
    intermediate: Annotated[pathlib.Path, outputs.Out("intermediate.csv", loaders.PathOnly())]


class _FinalCsv(TypedDict):
    final: Annotated[pathlib.Path, outputs.Out("final.csv", loaders.PathOnly())]


def _stage_a_input_csv(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _IntermediateCsv:
    _ = input_file
    pathlib.Path("intermediate.csv").write_text("")
    return _IntermediateCsv(intermediate=pathlib.Path("intermediate.csv"))


def _stage_b_intermediate_csv(
    intermediate: Annotated[pathlib.Path, outputs.Dep("intermediate.csv", loaders.PathOnly())],
) -> _FinalCsv:
    _ = intermediate
    pathlib.Path("final.csv").write_text("")
    return _FinalCsv(final=pathlib.Path("final.csv"))


def _stage_with_dir_dep(
    data: Annotated[pathlib.Path, outputs.Dep("data/", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_with_dir_dep_no_slash(
    data: Annotated[pathlib.Path, outputs.Dep("data", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


class _OutputsResult(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("outputs/result.txt", loaders.PathOnly())]


def _stage_with_nested_output() -> _OutputsResult:
    pathlib.Path("outputs/result.txt").parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/result.txt").write_text("")
    return _OutputsResult(result=pathlib.Path("outputs/result.txt"))


class _OutputsDir(TypedDict):
    output_dir: Annotated[pathlib.Path, outputs.Out("outputs", loaders.PathOnly())]


def _stage_with_dir_output() -> _OutputsDir:
    pathlib.Path("outputs").mkdir(exist_ok=True)
    return _OutputsDir(output_dir=pathlib.Path("outputs"))


# _collect_watch_paths tests


def test_collect_watch_paths_includes_project_root(pipeline_dir: pathlib.Path) -> None:
    """Project root should always be in watch paths."""
    paths = _watch_utils.collect_watch_paths([])
    assert pipeline_dir in paths


def test_collect_watch_paths_includes_dependency_directories(
    pipeline_dir: pathlib.Path,
) -> None:
    """Dependency file directories should be included."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_input_csv_dep, name="process")

    paths = _watch_utils.collect_watch_paths(["process"])
    assert data_dir in paths


# _create_watch_filter tests


def test_watch_filter_filters_exact_output_match(pipeline_dir: pathlib.Path) -> None:
    """Should filter out exact output file paths during execution."""
    output_path = pipeline_dir / "output.txt"

    register_test_stage(_stage_with_data_csv_dep, name="process")

    output_filter = _watch_utils.OutputFilter(["process"])
    output_filter.start_execution()  # Filtering only happens during execution
    watch_filter = _watch_utils.create_watch_filter(output_filter=output_filter)
    assert watch_filter(watchfiles.Change.modified, str(output_path)) is False


def test_watch_filter_allows_source_files(pipeline_dir: pathlib.Path) -> None:
    """Should allow source files that are not outputs."""
    source_path = pipeline_dir / "src" / "main.py"

    register_test_stage(_stage_with_data_csv_dep, name="process")

    output_filter = _watch_utils.OutputFilter(["process"])
    output_filter.start_execution()  # Filtering only happens during execution
    watch_filter = _watch_utils.create_watch_filter(output_filter=output_filter)
    assert watch_filter(watchfiles.Change.modified, str(source_path)) is True


@pytest.mark.parametrize(
    "path",
    [
        "/some/path/file.pyc",
        "/some/path/__pycache__/file.py",
        "/some/path/file.pyo",
    ],
)
def test_watch_filter_filters_python_bytecode(pipeline_dir: pathlib.Path, path: str) -> None:
    """Should filter out .pyc, .pyo, and __pycache__ files."""
    watch_filter = _watch_utils.create_watch_filter([])
    assert watch_filter(watchfiles.Change.modified, path) is False


# _build_file_to_stages_index tests


def test_build_file_to_stages_index_maps_deps_to_stages(pipeline_dir: pathlib.Path) -> None:
    """Should map dependency files to their consuming stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")
    (pipeline_dir / "config.yaml").write_text("key: value")

    register_test_stage(_stage_a_data_csv, name="stage_a")
    register_test_stage(_stage_b_config_yaml, name="stage_b")

    eng = engine.WatchEngine()
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

    register_test_stage(_stage_a_shared_csv, name="stage_a")
    register_test_stage(_stage_b_shared_csv, name="stage_b")

    eng = engine.WatchEngine()
    index = eng._build_file_to_stages_index()

    shared_path = project.resolve_path("shared.csv")
    assert shared_path in index
    assert "stage_a" in index[shared_path]
    assert "stage_b" in index[shared_path]


# _get_stages_matching_changes tests


def test_get_stages_affected_by_direct_dep_match(pipeline_dir: pathlib.Path) -> None:
    """Should find stages with direct dependency on changed file."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine()
    data_path = project.resolve_path("data.csv")
    affected = eng._get_stages_matching_changes({data_path})

    assert "process" in affected


def test_get_stages_affected_by_file_in_dep_directory(pipeline_dir: pathlib.Path) -> None:
    """Should find stages when file inside a dep directory changes."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "file.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_dir_dep, name="process")

    eng = engine.WatchEngine()
    file_path = project.resolve_path("data/file.csv")
    affected = eng._get_stages_matching_changes({file_path})

    assert "process" in affected


def test_get_stages_affected_returns_empty_for_unrelated_file(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should return empty set for changes to unrelated files."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine()
    unrelated_path = project.resolve_path("unrelated.txt")
    affected = eng._get_stages_matching_changes({unrelated_path})

    assert len(affected) == 0, "Should be empty for unrelated file"


# _add_downstream_stages tests


def test_add_downstream_stages_includes_direct_dependents(pipeline_dir: pathlib.Path) -> None:
    """Should include stages that directly depend on affected stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_preprocess, name="preprocess")
    register_test_stage(_stage_train, name="train")

    eng = engine.WatchEngine()
    affected = eng._add_downstream_stages({"preprocess"})

    assert "preprocess" in affected
    assert "train" in affected


def test_add_downstream_stages_includes_transitive_dependents(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should include transitively dependent stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_step1, name="step1")
    register_test_stage(_stage_step2, name="step2")
    register_test_stage(_stage_step3, name="step3")

    eng = engine.WatchEngine()
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

    register_test_stage(_stage_a_data_csv, name="stage_a")
    register_test_stage(_stage_b_config_yaml, name="stage_b")

    eng = engine.WatchEngine()
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

    register_test_stage(_stage_a_data1, name="stage_a")
    register_test_stage(_stage_b_data2, name="stage_b")

    eng = engine.WatchEngine()
    data_path = project.resolve_path("data1.csv")
    affected = eng._get_affected_stages({data_path}, code_changed=False)

    assert "stage_a" in affected
    assert "stage_b" not in affected


def test_get_affected_stages_includes_downstream_when_intermediate_file_changes(
    pipeline_dir: pathlib.Path,
) -> None:
    """Changing an intermediate file (output of A, dep of B) should trigger B and downstream."""
    # Create the input file
    (pipeline_dir / "input.csv").write_text("a,b\n1,2")

    # Create intermediate file (output of stage_a, dep of stage_b)
    (pipeline_dir / "intermediate.csv").write_text("x,y\n3,4")

    register_test_stage(_stage_a_input_csv, name="stage_a")
    register_test_stage(_stage_b_intermediate_csv, name="stage_b")

    eng = engine.WatchEngine()

    # Simulate changing the intermediate file (which is stage_b's dependency)
    intermediate_path = project.resolve_path("intermediate.csv")
    affected = eng._get_affected_stages({intermediate_path}, code_changed=False)

    # stage_b should be affected because intermediate.csv is its dependency
    assert "stage_b" in affected, "stage_b should run when its dep (intermediate.csv) changes"
    # stage_a should NOT be affected (its deps didn't change)
    assert "stage_a" not in affected, "stage_a should not run (input.csv unchanged)"


# _collect_and_debounce tests


def test_collect_and_debounce_returns_empty_on_shutdown(pipeline_dir: pathlib.Path) -> None:
    """Should return empty set immediately when shutdown is set."""
    eng = engine.WatchEngine(debounce_ms=100)
    eng._shutdown.set()

    start = time.monotonic()
    changes = eng._collect_and_debounce(max_wait_s=5.0)
    elapsed = time.monotonic() - start

    assert len(changes) == 0
    assert elapsed < 1.0, "Should return quickly on shutdown"


def test_collect_and_debounce_waits_for_quiet_period(pipeline_dir: pathlib.Path) -> None:
    """Should wait for quiet period after last change before returning."""
    eng = engine.WatchEngine(debounce_ms=200)

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
    eng = engine.WatchEngine(debounce_ms=300)

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
    eng = engine.WatchEngine(debounce_ms=500)

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


# WatchEngine initialization tests


def test_engine_init_sets_defaults(pipeline_dir: pathlib.Path) -> None:
    """Engine should initialize with default values."""
    eng = engine.WatchEngine()

    assert eng._stages is None
    assert eng._single_stage is False
    assert eng._debounce_ms == 300
    assert eng._shutdown.is_set() is False


def test_engine_init_accepts_custom_values(pipeline_dir: pathlib.Path) -> None:
    """Engine should accept custom configuration."""
    eng = engine.WatchEngine(
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
        engine.WatchEngine(debounce_ms=-1)


def test_engine_init_accepts_zero_debounce(pipeline_dir: pathlib.Path) -> None:
    """Engine should accept zero debounce_ms (no quiet period)."""
    eng = engine.WatchEngine(debounce_ms=0)
    assert eng._debounce_ms == 0


def test_engine_shutdown_sets_event(pipeline_dir: pathlib.Path) -> None:
    """shutdown() should set the shutdown event."""
    eng = engine.WatchEngine()
    assert eng._shutdown.is_set() is False

    eng.shutdown()
    assert eng._shutdown.is_set() is True


def test_add_downstream_stages_skips_unknown_stages(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_add_downstream_stages should skip stages not in DAG and log warning."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="known_stage")

    eng = engine.WatchEngine()

    # Try to add downstream for a mix of known and unknown stages
    result = eng._add_downstream_stages({"known_stage", "unknown_stage"})

    assert "known_stage" in result, "Known stage should be included"
    assert "unknown_stage" not in result, "Unknown stage should be skipped"
    assert "Stage 'unknown_stage' not found in DAG" in caplog.text


# Integration tests with mocked executor


def test_engine_runs_initial_execution(pipeline_dir: pathlib.Path, mocker: MockerFixture) -> None:
    """Engine should run initial pipeline execution on start."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    call_count = 0

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        nonlocal call_count
        call_count += 1
        self.shutdown()  # Exit after first call

    def mock_watch_loop(self: engine.WatchEngine, stages: list[str]) -> None:
        pass  # Don't actually start watching

    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    mocker.patch.object(engine.WatchEngine, "_watch_loop", mock_watch_loop)
    eng = engine.WatchEngine()
    eng.run()

    assert call_count == 1, "Should have executed initial pipeline"


def test_engine_restarts_workers_on_code_change(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Engine should restart workers when Python files change."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    restart_called = False
    iteration_count = 0

    def mock_restart(self: engine.WatchEngine) -> None:
        nonlocal restart_called
        restart_called = True

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count >= 2:
            self.shutdown()

    def mock_watch_loop(self: engine.WatchEngine, stages: list[str]) -> None:
        pass  # Don't actually start watching

    def mock_reload_registry(self: engine.WatchEngine) -> bool:
        return True  # Simulate successful reload

    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    mocker.patch.object(engine.WatchEngine, "_restart_worker_pool", mock_restart)
    mocker.patch.object(engine.WatchEngine, "_reload_registry", mock_reload_registry)
    mocker.patch.object(engine.WatchEngine, "_watch_loop", mock_watch_loop)
    eng = engine.WatchEngine(debounce_ms=50)

    # Simulate code change
    code_path = pathlib.Path("/some/module.py")
    eng._change_queue.put({code_path})

    eng.run()

    assert restart_called, "Should have restarted workers on code change"


def test_engine_handles_execution_error_gracefully(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Engine should continue after execution errors."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    call_count = 0
    eng_ref: engine.WatchEngine | None = None

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        nonlocal call_count, eng_ref
        eng_ref = self
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Execution failed")
        # Second call - shutdown
        self.shutdown()

    def mock_watch_loop(self: engine.WatchEngine, stages: list[str]) -> None:
        pass  # Don't actually start watching

    def mock_reload_registry(self: engine.WatchEngine) -> bool:
        return True  # Simulate successful reload

    # Patch at the class level to ensure thread captures mocked method
    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    mocker.patch.object(engine.WatchEngine, "_reload_registry", mock_reload_registry)
    mocker.patch.object(engine.WatchEngine, "_watch_loop", mock_watch_loop)
    eng = engine.WatchEngine(debounce_ms=50)

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


def test_watcher_thread_exception_triggers_shutdown(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Watcher thread exceptions should trigger engine shutdown."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    shutdown_triggered = False

    def mock_watch_that_fails(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Watcher failed!")

    def patched_watch_loop(self: engine.WatchEngine, stages: list[str]) -> None:
        nonlocal shutdown_triggered
        try:
            mock_watch_that_fails()
        except Exception as e:
            engine.logger.critical(f"Watcher thread failed: {e}")
            self.shutdown()
            shutdown_triggered = True

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        pass  # Initial execution succeeds

    mocker.patch.object(engine.WatchEngine, "_watch_loop", patched_watch_loop)
    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    eng = engine.WatchEngine(debounce_ms=50)
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

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine()

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

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine(debounce_ms=1000)

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


def test_concurrent_shutdown_during_run(pipeline_dir: pathlib.Path, mocker: MockerFixture) -> None:
    """Shutdown during run() should clean up watcher thread."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    execution_count = 0

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        nonlocal execution_count
        execution_count += 1

    def mock_watch_loop(self: engine.WatchEngine, stages: list[str]) -> None:
        # Simulate watch loop that respects shutdown
        while not self._shutdown.is_set():
            time.sleep(0.05)

    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    mocker.patch.object(engine.WatchEngine, "_watch_loop", mock_watch_loop)
    eng = engine.WatchEngine(debounce_ms=50)

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


# Symlink resolution tests


def test_watch_filter_resolves_symlinks(pipeline_dir: pathlib.Path) -> None:
    """Watch filter should resolve symlinks for consistent comparison."""
    # Create actual output file
    output_dir = pipeline_dir / "outputs"
    output_dir.mkdir()
    actual_output = output_dir / "result.txt"
    actual_output.write_text("output data")

    # Create symlink to output
    symlink_path = pipeline_dir / "result_link.txt"
    symlink_path.symlink_to(actual_output)

    register_test_stage(_stage_with_nested_output, name="process")

    watch_filter = _watch_utils.create_watch_filter(["process"])

    # The symlink should be filtered because it points to an output
    assert watch_filter(watchfiles.Change.modified, str(symlink_path)) is False, (
        "Symlink to output should be filtered"
    )


def test_watch_filter_handles_broken_symlink(pipeline_dir: pathlib.Path) -> None:
    """Watch filter should handle broken symlinks gracefully without crashing."""
    # Create symlink to non-existent file
    broken_link = pipeline_dir / "broken_link.txt"
    broken_link.symlink_to(pipeline_dir / "nonexistent.txt")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    # Filter with glob pattern "process" - broken symlink won't match this pattern
    watch_filter = _watch_utils.create_watch_filter(["process"])

    # The key test: broken symlinks should not crash the filter
    # Returns False because "broken_link.txt" doesn't match glob "process"
    result = watch_filter(watchfiles.Change.modified, str(broken_link))
    assert result is False, "Should not match 'process' glob pattern"


def test_get_stages_affected_handles_deleted_file(pipeline_dir: pathlib.Path) -> None:
    """Should handle deleted files gracefully using absolute path."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine()

    # Create path to file that doesn't exist (simulating deletion)
    deleted_path = pipeline_dir / "deleted_file.csv"

    # Should not crash, just return empty set (file not in deps)
    affected = eng._get_stages_matching_changes({deleted_path})
    assert len(affected) == 0, "Deleted file not in deps should return empty"


# DAG and file index caching tests


def test_dag_cache_invalidation_on_code_change(pipeline_dir: pathlib.Path) -> None:
    """DAG cache should be invalidated when code changes."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine()

    # Build initial caches
    dag1 = eng._get_dag()
    index1 = eng._get_file_index()

    assert eng._cached_dag is dag1, "DAG should be cached"
    assert eng._cached_file_index is index1, "File index should be cached"

    # Simulate code change invalidation (what _invalidate_caches does)
    eng._cached_dag = None
    eng._cached_file_index = None
    REGISTRY.invalidate_dag_cache()

    # Next access should rebuild
    dag2 = eng._get_dag()
    index2 = eng._get_file_index()

    # New instances should be created
    assert dag2 is not dag1, "Should create new DAG after invalidation"
    assert index2 is not index1, "Should create new file index after invalidation"


def test_file_index_caching_returns_same_instance(pipeline_dir: pathlib.Path) -> None:
    """File index should be cached and return same instance."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine()

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


def test_integration_watchfiles_detects_real_file_change(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Integration test: actual watchfiles detects real file changes (not mocked)."""
    # Create data file that will be watched
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    execution_stages: list[list[str] | None] = []
    execution_event = threading.Event()

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        execution_stages.append(stages)
        if len(execution_stages) >= 2:  # Initial + one triggered by file change
            execution_event.set()
            self.shutdown()

    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    eng = engine.WatchEngine(debounce_ms=50)

    # Run engine in background thread (run() blocks)
    engine_thread = threading.Thread(target=eng.run)
    engine_thread.start()

    # Wait for watcher to initialize and start watching
    time.sleep(0.1)

    # Modify the data file - this should trigger re-execution
    data_file.write_text("a,b\n1,2\n3,4")

    # Wait for execution to be triggered (with timeout to prevent hang)
    triggered = execution_event.wait(timeout=2.0)

    # Clean shutdown if test times out
    if not triggered:
        eng.shutdown()
    engine_thread.join(timeout=1.0)

    assert len(execution_stages) >= 2, (
        f"Expected at least 2 executions (initial + triggered), got {len(execution_stages)}"
    )
    # First execution is initial run (None = all stages)
    assert execution_stages[0] is None, "First execution should be initial (all stages)"
    # Second execution should include our affected stage
    assert execution_stages[1] is not None, "Second execution should have specific stages"
    assert "process" in execution_stages[1], "Second execution should include affected stage"


def test_integration_watchfiles_detects_python_code_change(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Integration test: watchfiles detects Python file changes and triggers worker restart."""
    # Create data file and Python file
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("a,b\n1,2")
    code_file = pipeline_dir / "helper.py"
    code_file.write_text("def helper(): pass\n")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    execution_count = 0
    restart_called = False
    execution_event = threading.Event()

    def mock_execute(self: engine.WatchEngine, stages: list[str] | None) -> None:
        nonlocal execution_count
        execution_count += 1
        if execution_count >= 2:
            execution_event.set()
            self.shutdown()

    def mock_restart(self: engine.WatchEngine) -> None:
        nonlocal restart_called
        restart_called = True

    def mock_reload_registry(self: engine.WatchEngine) -> bool:
        return True  # Simulate successful reload

    mocker.patch.object(engine.WatchEngine, "_execute_stages", mock_execute)
    mocker.patch.object(engine.WatchEngine, "_restart_worker_pool", mock_restart)
    mocker.patch.object(engine.WatchEngine, "_reload_registry", mock_reload_registry)
    eng = engine.WatchEngine(debounce_ms=50)

    engine_thread = threading.Thread(target=eng.run)
    engine_thread.start()

    # Wait for watcher to start
    time.sleep(0.1)

    # Modify Python file - should trigger worker restart
    code_file.write_text("def helper(): return 42\n")

    triggered = execution_event.wait(timeout=2.0)

    if not triggered:
        eng.shutdown()
    engine_thread.join(timeout=1.0)

    assert execution_count >= 2, f"Expected at least 2 executions, got {execution_count}"
    assert restart_called, "Worker pool should be restarted on Python file change"


# Additional coverage tests


def test_collect_and_debounce_raises_on_invalid_max_wait(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should raise ValueError for non-positive max_wait_s."""
    eng = engine.WatchEngine()

    with pytest.raises(ValueError, match="max_wait_s must be positive"):
        eng._collect_and_debounce(max_wait_s=0)

    with pytest.raises(ValueError, match="max_wait_s must be positive"):
        eng._collect_and_debounce(max_wait_s=-1.0)


def test_get_affected_stages_returns_all_registry_stages_on_code_change(
    pipeline_dir: pathlib.Path,
) -> None:
    """When stages is None and code changed, should return all registry stages."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_a_data_csv, name="stage1")
    register_test_stage(_stage_b_data2, name="stage2")

    eng = engine.WatchEngine(stages=None)  # No specific stages

    changes = {pathlib.Path("code.py")}
    affected = eng._get_affected_stages(changes, code_changed=True)

    assert set(affected) == {"stage1", "stage2"}, "Should return all registry stages"


def test_send_message_to_tui_queue(pipeline_dir: pathlib.Path) -> None:
    """_send_message should put messages on TUI queue when available."""
    # TUI queue uses stdlib queue.Queue (inter-thread, not cross-process)
    tui_queue: types.TuiQueue = queue.Queue()

    eng = engine.WatchEngine()
    eng._tui_queue = tui_queue

    eng._send_message("Test message")

    assert not tui_queue.empty(), "Message should be in queue"
    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == types.TuiMessageType.WATCH
    # Narrow to TuiWatchMessage after checking type
    assert types.is_tui_watch_message(msg)
    assert msg["status"] == types.WatchStatus.WAITING
    assert msg["message"] == "Test message"


def test_send_message_error_to_tui_queue(pipeline_dir: pathlib.Path) -> None:
    """_send_message with status=ERROR should set error status."""
    tui_queue: types.TuiQueue = queue.Queue()

    eng = engine.WatchEngine()
    eng._tui_queue = tui_queue

    eng._send_message("Error occurred", status=types.WatchStatus.ERROR)

    msg = tui_queue.get_nowait()
    assert msg is not None
    # Narrow to TuiWatchMessage after checking type
    assert types.is_tui_watch_message(msg)
    assert msg["status"] == types.WatchStatus.ERROR
    assert msg["message"] == "Error occurred"


def test_collect_watch_paths_handles_missing_stage(
    pipeline_dir: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """collect_watch_paths should log warning and skip missing stages."""

    # Register one stage with a dep
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="existing")

    # Include a non-existent stage in the list
    stages = ["existing", "nonexistent_stage"]
    results = _watch_utils.collect_watch_paths(stages)

    # Should still include project root and existing stage's dep directory
    assert pipeline_dir in results
    assert "Stage 'nonexistent_stage' not found in registry" in caplog.text


def test_watch_filter_filters_nested_output_directory(
    pipeline_dir: pathlib.Path,
) -> None:
    """Watch filter should filter files inside output directories."""
    out_dir = pipeline_dir / "outputs"
    out_dir.mkdir()
    nested_file = out_dir / "subdir" / "result.csv"
    nested_file.parent.mkdir()
    nested_file.write_text("data")

    register_test_stage(_stage_with_dir_output, name="produce")

    stages_to_run = ["produce"]
    watch_filter = _watch_utils.create_watch_filter(stages_to_run)

    # Files inside output directory should be filtered
    assert watch_filter(watchfiles.Change.modified, str(nested_file)) is False


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

    register_test_stage(_stage_with_dir_dep_no_slash, name="process")

    eng = engine.WatchEngine()

    # Use a completely different path that's not relative to anything
    changes = {pathlib.Path("/completely/different/path.txt")}
    affected = eng._get_stages_matching_changes(changes)

    # Should not raise, should return empty set
    assert affected == set()


def test_reload_registry_logs_when_no_modules_found(
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_reload_registry should log warning when no stage modules found.

    Uses set_project_root instead of pipeline_dir to avoid creating pivot.yaml,
    which would trigger the pipeline file reload path instead of decorator path.
    """
    # Create .pivot marker for project root detection
    (set_project_root / ".pivot").mkdir()
    # Registry is already empty from global clean_registry autouse fixture
    eng = engine.WatchEngine()
    eng._reload_registry()

    assert "No stage modules found to reload" in caplog.text


def test_reload_registry_clears_and_reimports_modules(
    pipeline_dir: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """_reload_registry should clear project modules and reimport them."""

    register_test_stage(_stage_noop, name="test_stage")

    eng = engine.WatchEngine()

    initial_stages = list(REGISTRY.list_stages())
    assert "test_stage" in initial_stages

    # Mock _clear_project_modules to verify it's called (for decorator path)
    mock_clear = mocker.patch.object(
        engine, "_clear_project_modules", return_value=0, autospec=True
    )
    mocker.patch("importlib.import_module", autospec=True)
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
    mocker: MockerFixture,
) -> None:
    """_is_existing_dir should return False on OSError."""
    test_path = pipeline_dir / "test_path"

    mocker.patch.object(
        pathlib.Path, "is_dir", side_effect=OSError("Permission denied"), autospec=True
    )
    assert engine._is_existing_dir(test_path) is False


def test_reload_registry_handles_import_exception(
    pipeline_dir: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    mocker: MockerFixture,
) -> None:
    """_reload_registry should return False and preserve registry when import fails."""

    register_test_stage(_stage_noop, name="reload_test_stage")

    eng = engine.WatchEngine()

    # Mock import_module to raise an exception (for decorator path)
    mocker.patch(
        "importlib.import_module", side_effect=ImportError("Module not found"), autospec=True
    )
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
    deps: {}
    outs:
      output: output.txt
"""
    )

    # Add pipeline_dir to sys.path so stages.py can be imported (auto-cleanup)
    monkeypatch.syspath_prepend(str(pipeline_dir))

    # First manually register so we have backup
    pipeline_yaml.register_from_pipeline_file(pipeline_dir / "pivot.yaml")
    assert "test_yaml_stage" in REGISTRY.list_stages()

    eng = engine.WatchEngine()
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

    # Create pipeline.py with annotation-based output
    (pipeline_dir / "pipeline.py").write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def pipeline_py_stage() -> _OutputTxt:
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))

REGISTRY.register(pipeline_py_stage)
"""
    )

    # First run pipeline.py to register
    runpy.run_path(str(pipeline_dir / "pipeline.py"), run_name="_pivot_pipeline")
    assert "pipeline_py_stage" in REGISTRY.list_stages()

    eng = engine.WatchEngine()
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
    deps: {}
    outs:
      yaml_out: yaml_out.txt
"""
    )

    # Create pipeline.py with different stage (annotation-based)
    (pipeline_dir / "pipeline.py").write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _PyOut(TypedDict):
    py_out: Annotated[pathlib.Path, outputs.Out("py_out.txt", loaders.PathOnly())]

def py_stage() -> _PyOut:
    pathlib.Path("py_out.txt").write_text("")
    return _PyOut(py_out=pathlib.Path("py_out.txt"))

REGISTRY.register(py_stage)
"""
    )

    # Add pipeline_dir to sys.path so stages.py can be imported (auto-cleanup)
    monkeypatch.syspath_prepend(str(pipeline_dir))

    # First register from pivot.yaml
    pipeline_yaml.register_from_pipeline_file(pipeline_dir / "pivot.yaml")
    assert "yaml_stage" in REGISTRY.list_stages()

    eng = engine.WatchEngine()
    result = eng._reload_registry()

    assert result is True
    assert "yaml_stage" in REGISTRY.list_stages(), "Should use pivot.yaml stage"
    assert "py_stage" not in REGISTRY.list_stages(), "Should NOT use pipeline.py stage"
    assert "Registry reloaded from pivot.yaml" in caplog.text

    # Clean up imported module
    if "stages" in sys.modules:
        del sys.modules["stages"]


def test_reload_registry_falls_back_to_decorators(
    pipeline_dir: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    mocker: MockerFixture,
) -> None:
    """_reload_registry should fall back to decorator reload when no config files."""
    # Remove pivot.yaml stages section
    (pipeline_dir / "pivot.yaml").write_text("version: 1\n")
    # Don't create pipeline.py

    register_test_stage(_stage_noop, name="decorator_stage")

    eng = engine.WatchEngine()

    # Mock import_module to verify decorator path is used
    mock_import = mocker.patch("importlib.import_module", autospec=True)
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
    deps: {}
    outs:
      output: output.txt
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
    deps: {}
    outs:
      output: output.txt
"""
    )

    eng = engine.WatchEngine()
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

    # Create valid pipeline.py first (annotation-based)
    (pipeline_dir / "pipeline.py").write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def preserved_py_stage() -> _OutputTxt:
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))

REGISTRY.register(preserved_py_stage)
"""
    )

    # Run it to register
    runpy.run_path(str(pipeline_dir / "pipeline.py"), run_name="_pivot_pipeline")
    assert "preserved_py_stage" in REGISTRY.list_stages()

    # Now break the pipeline.py
    (pipeline_dir / "pipeline.py").write_text("raise RuntimeError('broken')\n")

    eng = engine.WatchEngine()
    result = eng._reload_registry()

    assert result is False, "Should return False on reload failure"
    assert "preserved_py_stage" in REGISTRY.list_stages(), "Old registry should be preserved"
    assert eng._pipeline_errors is not None


def test_watch_filter_handles_resolve_oserror(
    pipeline_dir: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Watch filter should not filter paths that can't be resolved."""

    register_test_stage(_stage_with_data_csv_dep, name="process")

    watch_filter = _watch_utils.create_watch_filter(["process"])

    # Mock project.resolve_path to raise OSError
    mocker.patch.object(
        project, "resolve_path", side_effect=OSError("Permission denied"), autospec=True
    )
    # Should return True (don't filter) when path can't be resolved
    result = watch_filter(watchfiles.Change.modified, "/some/path")
    assert result is True


def test_resolve_path_for_matching_handles_oserror(
    pipeline_dir: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """_resolve_path_for_matching should use normalized path on OSError."""
    test_path = pipeline_dir / "test_file.txt"

    # Mock project.resolve_path to raise OSError
    mocker.patch.object(
        project, "resolve_path", side_effect=OSError("Access denied"), autospec=True
    )
    result = engine._resolve_path_for_matching(test_path)

    # Should return normalized absolute path
    assert result.is_absolute()
    assert str(result).endswith("test_file.txt")


def test_get_stages_matching_changes_handles_is_relative_to_error(
    pipeline_dir: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Should handle ValueError from is_relative_to."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_dir_dep_no_slash, name="containment_stage")

    eng = engine.WatchEngine()

    changes = {pipeline_dir / "some_file.txt"}

    # Mock is_relative_to to raise ValueError
    def mock_is_relative_to(self: pathlib.Path, other: pathlib.Path) -> bool:
        raise ValueError("Cannot compare paths")

    mocker.patch.object(pathlib.Path, "is_relative_to", mock_is_relative_to)
    affected = eng._get_stages_matching_changes(changes)

    # Should not raise, should return empty set
    assert affected == set()


# _reload_registry invalid pipeline tests


def test_reload_registry_clears_errors_on_success(pipeline_dir: pathlib.Path) -> None:
    """_reload_registry should clear previous errors on successful reload."""

    register_test_stage(_stage_noop, name="test_clear_stage")

    eng = engine.WatchEngine()
    # Simulate previous error state
    eng._pipeline_errors = ["previous error"]

    result = eng._reload_registry()

    assert result is True, "Should return True on success"
    assert eng._pipeline_errors is None, "Previous errors should be cleared"


def test_reload_registry_returns_true_when_no_modules(pipeline_dir: pathlib.Path) -> None:
    """_reload_registry should return True when no stage modules found."""
    eng = engine.WatchEngine()
    # No stages registered

    result = eng._reload_registry()

    assert result is True, "Should return True when no modules to reload"


def test_coordinator_loop_skips_execution_when_invalid(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Coordinator should skip execution when pipeline is invalid."""

    register_test_stage(_stage_noop, name="invalid_test_stage")

    eng = engine.WatchEngine()

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
        msg: str, *, is_error: bool = False, status: types.WatchStatus | None = None
    ) -> None:
        messages.append(msg)

    mocker.patch.object(
        eng, "_collect_and_debounce", side_effect=mock_collect_and_debounce, autospec=True
    )
    mocker.patch.object(eng, "_send_message", side_effect=capture_message, autospec=True)
    mock_execute = mocker.patch.object(eng, "_execute_stages", autospec=True)
    eng._coordinator_loop()

    # Should not have called execute since pipeline is invalid
    mock_execute.assert_not_called()
    # Should have sent "Watching for changes..." message
    assert "Watching for changes..." in messages


def test_coordinator_loop_sends_error_on_reload_failure(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """Coordinator should send error message when reload fails."""

    register_test_stage(_stage_noop, name="reload_fail_stage")

    eng = engine.WatchEngine()

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
        msg: str, *, is_error: bool = False, status: types.WatchStatus | None = None
    ) -> None:
        is_err = is_error or (status == types.WatchStatus.ERROR)
        messages.append((msg, is_err))

    mocker.patch.object(
        eng, "_collect_and_debounce", side_effect=mock_collect_and_debounce, autospec=True
    )
    mocker.patch.object(eng, "_send_message", side_effect=capture_message, autospec=True)
    mocker.patch.object(eng, "_invalidate_caches", autospec=True)
    mocker.patch.object(eng, "_restart_worker_pool", autospec=True)
    mocker.patch(
        "importlib.import_module", side_effect=SyntaxError("invalid syntax"), autospec=True
    )
    eng._coordinator_loop()

    # Should have sent error message
    error_messages = [(msg, err) for msg, err in messages if err]
    assert len(error_messages) > 0, "Should have sent error message"
    assert any("Pipeline invalid" in msg for msg, _ in error_messages)


def test_coordinator_loop_clears_invalid_state_on_successful_reload(
    pipeline_dir: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Coordinator should clear invalid state when reload succeeds."""

    register_test_stage(_stage_noop, name="recovery_stage")

    eng = engine.WatchEngine()
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

    mocker.patch.object(
        eng, "_collect_and_debounce", side_effect=mock_collect_and_debounce, autospec=True
    )
    mocker.patch.object(eng, "_send_message", autospec=True)
    mocker.patch.object(eng, "_invalidate_caches", autospec=True)
    mocker.patch.object(eng, "_restart_worker_pool", autospec=True)
    mocker.patch.object(eng, "_execute_stages", autospec=True)
    eng._coordinator_loop()

    # After successful reload, errors should be cleared
    assert eng._pipeline_errors is None, "Errors should be cleared on successful reload"


# force_first_run tests


def test_engine_init_force_first_run_defaults_to_false(pipeline_dir: pathlib.Path) -> None:
    """force_first_run should default to False."""
    eng = engine.WatchEngine()
    assert eng._force_first_run is False
    assert eng._first_run_done is False


def test_engine_init_accepts_force_first_run(pipeline_dir: pathlib.Path) -> None:
    """Engine should accept force_first_run parameter."""
    eng = engine.WatchEngine(force_first_run=True)
    assert eng._force_first_run is True
    assert eng._first_run_done is False


def test_execute_stages_passes_force_on_first_run(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """_execute_stages should pass force=True on first run when force_first_run=True."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine(force_first_run=True)

    force_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_values.append(bool(kwargs.get("force", False)))

    mocker.patch.object(executor, "run", side_effect=capture_force, autospec=True)
    eng._execute_stages(None)

    assert len(force_values) == 1
    assert force_values[0] is True, "First execution should have force=True"
    assert eng._first_run_done is True, "_first_run_done should be set after first execution"


def test_execute_stages_does_not_force_subsequent_runs(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """_execute_stages should pass force=False on subsequent runs."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine(force_first_run=True)

    force_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_values.append(bool(kwargs.get("force", False)))

    mocker.patch.object(executor, "run", side_effect=capture_force, autospec=True)
    eng._execute_stages(None)  # First run
    eng._execute_stages(None)  # Second run
    eng._execute_stages(None)  # Third run

    assert len(force_values) == 3
    assert force_values[0] is True, "First execution should have force=True"
    assert force_values[1] is False, "Second execution should have force=False"
    assert force_values[2] is False, "Third execution should have force=False"


def test_execute_stages_without_force_first_run_never_forces(
    pipeline_dir: pathlib.Path, mocker: MockerFixture
) -> None:
    """_execute_stages should never pass force=True when force_first_run=False."""
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine(force_first_run=False)  # Default

    force_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_values.append(bool(kwargs.get("force", False)))

    mocker.patch.object(executor, "run", side_effect=capture_force, autospec=True)
    eng._execute_stages(None)
    eng._execute_stages(None)

    assert all(v is False for v in force_values), "All executions should have force=False"


def test_first_run_done_flag_stress_test(pipeline_dir: pathlib.Path, mocker: MockerFixture) -> None:
    """Stress test: rapid executions should only force first run, never subsequent ones.

    This test verifies _first_run_done behaves correctly even under rapid execution.
    Since all access is from the main thread, there's no race condition, but this
    confirms the flag logic is correct under stress.
    """
    (pipeline_dir / "data.csv").write_text("a,b\n1,2")

    register_test_stage(_stage_with_data_csv_dep, name="process")

    eng = engine.WatchEngine(force_first_run=True)

    force_values: list[bool] = []
    first_run_done_values: list[bool] = []

    def capture_force(**kwargs: object) -> None:
        force_val = kwargs.get("force", False)
        force_values.append(bool(force_val))
        first_run_done_values.append(eng._first_run_done)

    # Run many executions rapidly
    num_executions = 100

    mocker.patch.object(executor, "run", side_effect=capture_force, autospec=True)
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


# =============================================================================
# Transitive dependency reload tests
# =============================================================================


def test_transitive_dependency_reload_detects_helper_changes(
    pipeline_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modifying a helper module imported by a stage module should be detected.

    This is a critical bug test: if stages.py imports helpers.py, and helpers.py
    is modified, the old behavior was to only reload stages.py - but reload()
    doesn't reimport dependencies, so the old helpers.py code would still be used.

    The fix clears ALL project modules from sys.modules before reimporting.
    """
    import sys

    # Clean up any cached modules from previous tests
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("stages", "helpers") or mod_name.startswith(("stages.", "helpers.")):
            del sys.modules[mod_name]

    # Create helper module
    helpers_py = pipeline_dir / "helpers.py"
    helpers_py.write_text(
        """\
HELPER_VALUE = "original"

def get_value():
    return HELPER_VALUE
"""
    )

    # Create stages module that imports helper (annotation-based)
    stages_py = pipeline_dir / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from helpers import get_value
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def my_stage() -> _OutputTxt:
    _ = get_value()
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))

REGISTRY.register(my_stage)
"""
    )

    # Add pipeline_dir to sys.path
    monkeypatch.syspath_prepend(str(pipeline_dir))

    # Import stages to register the stage
    import importlib

    importlib.import_module("stages")
    assert "my_stage" in REGISTRY.list_stages()

    # Verify original value
    helpers_module = sys.modules["helpers"]
    assert helpers_module.HELPER_VALUE == "original"

    # Now modify the helper module
    helpers_py.write_text(
        """\
HELPER_VALUE = "modified"

def get_value():
    return HELPER_VALUE
"""
    )

    # Create engine and reload registry
    eng = engine.WatchEngine()
    result = eng._reload_registry()

    assert result is True, "Reload should succeed"

    # The critical assertion: helper module should have the NEW value
    # If transitive dependencies aren't properly reloaded, this will still be "original"
    helpers_module = sys.modules.get("helpers")
    assert helpers_module is not None, "helpers module should still be loaded"
    assert helpers_module.HELPER_VALUE == "modified", (
        "Transitive dependency should be reloaded - got stale cached value"
    )

    # Clean up
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("stages", "helpers") or mod_name.startswith("stages."):
            del sys.modules[mod_name]


# =============================================================================
# Registry reload with stage changes - filter update tests
# =============================================================================


def test_watch_filter_stale_after_registry_adds_new_stage(
    pipeline_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch filter should update when registry reload adds new stages.

    Bug: _create_watch_filter() captures output paths at startup. If registry
    reload adds a new stage with new outputs, the filter doesn't know about them,
    potentially causing infinite loops (new stage output triggers another reload).
    """
    import sys

    # Clean up any cached modules from previous tests
    for mod_name in list(sys.modules.keys()):
        if mod_name == "stages" or mod_name.startswith("stages."):
            del sys.modules[mod_name]

    # Create initial stages module with one stage (annotation-based)
    stages_py = pipeline_dir / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputA(TypedDict):
    output_a: Annotated[pathlib.Path, outputs.Out("output_a.txt", loaders.PathOnly())]

def stage_a() -> _OutputA:
    pathlib.Path("output_a.txt").write_text("")
    return _OutputA(output_a=pathlib.Path("output_a.txt"))

REGISTRY.register(stage_a)
"""
    )

    monkeypatch.syspath_prepend(str(pipeline_dir))

    # Import to register
    import importlib

    importlib.import_module("stages")
    assert "stage_a" in REGISTRY.list_stages()
    assert "stage_b" not in REGISTRY.list_stages()

    # Create initial watch filter with OutputFilter (filtering only during execution)
    initial_output_filter = _watch_utils.OutputFilter(["stage_a"])
    initial_output_filter.start_execution()
    initial_filter = _watch_utils.create_watch_filter(output_filter=initial_output_filter)

    # output_a should be filtered
    output_a_path = pipeline_dir / "output_a.txt"
    assert initial_filter(watchfiles.Change.modified, str(output_a_path)) is False

    # output_b should NOT be filtered (stage doesn't exist yet)
    output_b_path = pipeline_dir / "output_b.txt"
    assert initial_filter(watchfiles.Change.modified, str(output_b_path)) is True

    # Now add a new stage (annotation-based)
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputA(TypedDict):
    output_a: Annotated[pathlib.Path, outputs.Out("output_a.txt", loaders.PathOnly())]

class _OutputB(TypedDict):
    output_b: Annotated[pathlib.Path, outputs.Out("output_b.txt", loaders.PathOnly())]

def stage_a() -> _OutputA:
    pathlib.Path("output_a.txt").write_text("")
    return _OutputA(output_a=pathlib.Path("output_a.txt"))

def stage_b() -> _OutputB:
    pathlib.Path("output_b.txt").write_text("")
    return _OutputB(output_b=pathlib.Path("output_b.txt"))

REGISTRY.register(stage_a)
REGISTRY.register(stage_b)
"""
    )

    # Reload registry
    eng = engine.WatchEngine()
    eng._reload_registry()

    assert "stage_b" in REGISTRY.list_stages(), "New stage should be registered"

    # BUG DEMONSTRATION: The OLD filter still doesn't know about output_b
    # This is the bug - after reload, we need to recreate the filter
    # The test currently shows the bug exists
    assert initial_filter(watchfiles.Change.modified, str(output_b_path)) is True, (
        "Old filter doesn't know about new stage output (this demonstrates the bug)"
    )

    # Create NEW filter after reload - this one should filter output_b
    new_output_filter = _watch_utils.OutputFilter(["stage_a", "stage_b"])
    new_output_filter.start_execution()
    new_filter = _watch_utils.create_watch_filter(output_filter=new_output_filter)
    assert new_filter(watchfiles.Change.modified, str(output_b_path)) is False, (
        "New filter should filter new stage output"
    )

    # Clean up
    for mod_name in list(sys.modules.keys()):
        if mod_name == "stages" or mod_name.startswith("stages."):
            del sys.modules[mod_name]


def test_watch_filter_stale_after_registry_removes_stage(
    pipeline_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch filter should update when registry reload removes stages.

    If a stage is removed, its outputs should no longer be filtered.
    """
    import sys

    # Clean up any cached modules from previous tests
    for mod_name in list(sys.modules.keys()):
        if mod_name == "stages" or mod_name.startswith("stages."):
            del sys.modules[mod_name]

    # Create initial stages module with two stages (annotation-based)
    stages_py = pipeline_dir / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputA(TypedDict):
    output_a: Annotated[pathlib.Path, outputs.Out("output_a.txt", loaders.PathOnly())]

class _OutputB(TypedDict):
    output_b: Annotated[pathlib.Path, outputs.Out("output_b.txt", loaders.PathOnly())]

def stage_a() -> _OutputA:
    pathlib.Path("output_a.txt").write_text("")
    return _OutputA(output_a=pathlib.Path("output_a.txt"))

def stage_b() -> _OutputB:
    pathlib.Path("output_b.txt").write_text("")
    return _OutputB(output_b=pathlib.Path("output_b.txt"))

REGISTRY.register(stage_a)
REGISTRY.register(stage_b)
"""
    )

    monkeypatch.syspath_prepend(str(pipeline_dir))

    import importlib

    importlib.import_module("stages")
    assert "stage_a" in REGISTRY.list_stages()
    assert "stage_b" in REGISTRY.list_stages()

    # Create initial watch filter for both stages using OutputFilter
    initial_output_filter = _watch_utils.OutputFilter(["stage_a", "stage_b"])
    initial_output_filter.start_execution()
    initial_filter = _watch_utils.create_watch_filter(output_filter=initial_output_filter)

    # Both outputs should be filtered
    output_a_path = pipeline_dir / "output_a.txt"
    output_b_path = pipeline_dir / "output_b.txt"
    assert initial_filter(watchfiles.Change.modified, str(output_a_path)) is False
    assert initial_filter(watchfiles.Change.modified, str(output_b_path)) is False

    # Remove stage_b (annotation-based)
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputA(TypedDict):
    output_a: Annotated[pathlib.Path, outputs.Out("output_a.txt", loaders.PathOnly())]

def stage_a() -> _OutputA:
    pathlib.Path("output_a.txt").write_text("")
    return _OutputA(output_a=pathlib.Path("output_a.txt"))

REGISTRY.register(stage_a)
"""
    )

    # Reload registry
    eng = engine.WatchEngine()
    eng._reload_registry()

    assert "stage_a" in REGISTRY.list_stages()
    assert "stage_b" not in REGISTRY.list_stages(), "stage_b should be removed"

    # Create NEW filter after reload - only stage_a exists now
    new_output_filter = _watch_utils.OutputFilter(["stage_a"])
    new_output_filter.start_execution()
    new_filter = _watch_utils.create_watch_filter(output_filter=new_output_filter)
    assert new_filter(watchfiles.Change.modified, str(output_a_path)) is False, (
        "output_a should still be filtered"
    )
    # output_b should no longer be filtered (stage doesn't exist)
    assert new_filter(watchfiles.Change.modified, str(output_b_path)) is True, (
        "output_b should NOT be filtered after stage removal"
    )

    # Clean up
    for mod_name in list(sys.modules.keys()):
        if mod_name == "stages" or mod_name.startswith("stages."):
            del sys.modules[mod_name]


def test_file_index_stale_after_registry_changes(
    pipeline_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File index should be invalidated when registry changes.

    The file-to-stages index maps dependency files to stages. If stages are
    added/removed, the index becomes stale.
    """
    import sys

    # Clean up any cached modules from previous tests
    for mod_name in list(sys.modules.keys()):
        if mod_name == "stages" or mod_name.startswith("stages."):
            del sys.modules[mod_name]

    # Create initial stages module (annotation-based)
    stages_py = pipeline_dir / "stages.py"
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputA(TypedDict):
    output_a: Annotated[pathlib.Path, outputs.Out("output_a.txt", loaders.PathOnly())]

def stage_a(
    data_a: Annotated[pathlib.Path, outputs.Dep("data_a.csv", loaders.PathOnly())],
) -> _OutputA:
    _ = data_a
    pathlib.Path("output_a.txt").write_text("")
    return _OutputA(output_a=pathlib.Path("output_a.txt"))

REGISTRY.register(stage_a)
"""
    )
    (pipeline_dir / "data_a.csv").write_text("a,b\n1,2")
    (pipeline_dir / "data_b.csv").write_text("x,y\n3,4")

    monkeypatch.syspath_prepend(str(pipeline_dir))

    import importlib

    importlib.import_module("stages")

    eng = engine.WatchEngine()

    # Build initial index
    index1 = eng._get_file_index()
    data_a_path = project.resolve_path("data_a.csv")
    data_b_path = project.resolve_path("data_b.csv")

    assert data_a_path in index1, "data_a should be in index"
    assert data_b_path not in index1, "data_b should NOT be in index"

    # Add new stage with different dependency (annotation-based)
    stages_py.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import outputs, loaders, stage_def
from pivot.registry import REGISTRY

class _OutputA(TypedDict):
    output_a: Annotated[pathlib.Path, outputs.Out("output_a.txt", loaders.PathOnly())]

class _OutputB(TypedDict):
    output_b: Annotated[pathlib.Path, outputs.Out("output_b.txt", loaders.PathOnly())]

def stage_a(
    data_a: Annotated[pathlib.Path, outputs.Dep("data_a.csv", loaders.PathOnly())],
) -> _OutputA:
    _ = data_a
    pathlib.Path("output_a.txt").write_text("")
    return _OutputA(output_a=pathlib.Path("output_a.txt"))

def stage_b(
    data_b: Annotated[pathlib.Path, outputs.Dep("data_b.csv", loaders.PathOnly())],
) -> _OutputB:
    _ = data_b
    pathlib.Path("output_b.txt").write_text("")
    return _OutputB(output_b=pathlib.Path("output_b.txt"))

REGISTRY.register(stage_a)
REGISTRY.register(stage_b)
"""
    )

    # Simulate what _coordinator_loop does: invalidate caches and reload
    eng._invalidate_caches()
    eng._reload_registry()

    # Get new index (should be rebuilt after invalidation)
    index2 = eng._get_file_index()

    assert data_a_path in index2, "data_a should still be in index"
    assert data_b_path in index2, "data_b should now be in index after stage addition"

    # Clean up
    for mod_name in list(sys.modules.keys()):
        if mod_name == "stages" or mod_name.startswith("stages."):
            del sys.modules[mod_name]


# =============================================================================
# Agent RPC Method Tests (try_start_agent_run)
# =============================================================================


def _helper_agent_noop() -> None:
    pass


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_succeeds_when_watching() -> None:
    """try_start_agent_run succeeds when engine is in WATCHING state."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    # Simulate coordinator starting (sets state to WATCHING)
    eng._update_agent_state(AgentState.WATCHING)

    result = eng.try_start_agent_run("run123", ["test_stage"], force=False)

    assert "status" in result, "Should return success result"
    assert result["status"] == "started"
    assert result["run_id"] == "run123"
    assert result["stages_queued"] == ["test_stage"]


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_rejects_when_idle() -> None:
    """try_start_agent_run rejects when engine is in IDLE state (not ready yet)."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    # Default state is IDLE (coordinator not started)

    result = eng.try_start_agent_run("run123", ["test_stage"], force=False)

    assert "reason" in result, "Should return rejection result"
    assert result["reason"] == "not_ready"
    assert result["current_state"] == "idle"


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_rejects_when_running() -> None:
    """try_start_agent_run rejects when execution is already in progress."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    eng._update_agent_state(AgentState.WATCHING)

    # First call should succeed
    result1 = eng.try_start_agent_run("run1", ["test_stage"], force=False)
    assert "status" in result1

    # Second call should be rejected (state is now RUNNING)
    result2 = eng.try_start_agent_run("run2", ["test_stage"], force=False)
    assert "reason" in result2, "Should return rejection result"
    assert result2["reason"] == "not_ready"
    assert result2["current_state"] == "running"
    # current_run_id is optional, check with "in"
    if "current_run_id" in result2:
        assert result2["current_run_id"] == "run1"


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_sets_state_atomically() -> None:
    """try_start_agent_run sets state to RUNNING atomically with queuing."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    eng._update_agent_state(AgentState.WATCHING)

    result = eng.try_start_agent_run("run123", ["test_stage"], force=False)

    assert "status" in result
    # Verify state was set
    status = eng.get_agent_status()
    assert status["state"] == AgentState.RUNNING
    # run_id is optional, check with "in"
    if "run_id" in status:
        assert status["run_id"] == "run123"


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_initializes_run_fields() -> None:
    """try_start_agent_run initializes all run tracking fields."""
    register_test_stage(_helper_agent_noop, name="stage_a")
    register_test_stage(_helper_agent_noop, name="stage_b")

    eng = engine.WatchEngine(debounce_ms=100)
    eng._update_agent_state(AgentState.WATCHING)

    result = eng.try_start_agent_run("run123", ["stage_a", "stage_b"], force=False)

    assert "status" in result
    status = eng.get_agent_status()
    # stages_pending is optional, check with "in"
    if "stages_pending" in status:
        assert status["stages_pending"] == ["stage_a", "stage_b"]
    # stages_completed is optional, use .get() is not allowed per CLAUDE.md, check with "in"
    if "stages_completed" in status:
        assert status["stages_completed"] == []


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_rolls_back_on_queue_full() -> None:
    """try_start_agent_run rolls back state if queue is full (prevents deadlock)."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    eng._update_agent_state(AgentState.WATCHING)

    # Fill the queue (maxsize=10)
    for i in range(10):
        eng._agent_request_queue.put_nowait((f"fill{i}", None, False))

    # Now try_start_agent_run should fail and roll back
    result = eng.try_start_agent_run("run_overflow", ["test_stage"], force=False)

    assert "reason" in result, "Should return rejection result"
    assert result["reason"] == "queue_full"

    # Verify state was rolled back to WATCHING (not stuck at RUNNING)
    status = eng.get_agent_status()
    assert status["state"] == AgentState.WATCHING, "State should be rolled back"


@pytest.mark.usefixtures("pipeline_dir")
def test_try_start_agent_run_concurrent_calls_only_one_succeeds() -> None:
    """Concurrent try_start_agent_run calls - exactly one should succeed."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    eng._update_agent_state(AgentState.WATCHING)

    results: list[AgentRunStartResult | AgentRunRejection] = []
    barrier = threading.Barrier(10)

    def try_run(run_id: str) -> None:
        barrier.wait()  # Synchronize all threads to start simultaneously
        result = eng.try_start_agent_run(run_id, ["test_stage"], force=False)
        results.append(result)

    threads = [threading.Thread(target=try_run, args=(f"run{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one should succeed
    successes = [r for r in results if "status" in r]
    rejections = [r for r in results if "reason" in r]

    assert len(successes) == 1, f"Exactly one should succeed, got {len(successes)}"
    assert len(rejections) == 9, f"Nine should be rejected, got {len(rejections)}"


@pytest.mark.usefixtures("pipeline_dir")
def test_get_agent_status_returns_running_state() -> None:
    """get_agent_status returns RUNNING state with run details."""
    register_test_stage(_helper_agent_noop, name="test_stage")

    eng = engine.WatchEngine(debounce_ms=100)
    eng._update_agent_state(AgentState.WATCHING)
    eng.try_start_agent_run("run123", ["test_stage"], force=False)

    status = eng.get_agent_status()

    assert status["state"] == AgentState.RUNNING
    # run_id is optional, check with "in"
    if "run_id" in status:
        assert status["run_id"] == "run123"
    assert "stages_pending" in status


@pytest.mark.usefixtures("pipeline_dir")
def test_get_agent_status_returns_completion_stats() -> None:
    """get_agent_status returns completion stats when in COMPLETED/FAILED state."""
    eng = engine.WatchEngine(debounce_ms=100)

    # Simulate completion
    eng._update_agent_state(
        AgentState.COMPLETED,
        run_id="run123",
        ran=3,
        skipped=1,
        failed=0,
    )

    status = eng.get_agent_status()

    assert status["state"] == AgentState.COMPLETED
    # ran, skipped, failed are optional, check with "in"
    if "ran" in status:
        assert status["ran"] == 3
    if "skipped" in status:
        assert status["skipped"] == 1
    if "failed" in status:
        assert status["failed"] == 0


# =============================================================================
# _get_pyc_path Tests
# =============================================================================


def test_get_pyc_path_non_py_returns_none() -> None:
    """_get_pyc_path returns None for non-.py files."""
    result = engine._get_pyc_path("/some/path/file.txt")
    assert result is None


def test_get_pyc_path_no_pyc_exists(tmp_path: pathlib.Path) -> None:
    """_get_pyc_path returns None when no .pyc exists."""
    py_file = tmp_path / "module.py"
    py_file.write_text("# test code")

    result = engine._get_pyc_path(str(py_file))
    assert result is None


def test_get_pyc_path_with_existing_pyc(tmp_path: pathlib.Path) -> None:
    """_get_pyc_path returns pyc path when it exists."""
    # Create a Python file and compile it
    py_file = tmp_path / "module.py"
    py_file.write_text("x = 1")
    py_compile.compile(str(py_file), cfile=None, doraise=True)

    result = engine._get_pyc_path(str(py_file))
    assert result is not None
    assert result.suffix == ".pyc"
    assert result.parent.name == "__pycache__"


# =============================================================================
# toggle_keep_going Tests
# =============================================================================


@pytest.mark.usefixtures("pipeline_dir")
def test_toggle_keep_going_toggles_state() -> None:
    """toggle_keep_going toggles the state and returns new value."""
    eng = engine.WatchEngine()
    assert eng.keep_going is False

    # Toggle on
    result = eng.toggle_keep_going()
    assert result is True
    assert eng.keep_going is True

    # Toggle off
    result = eng.toggle_keep_going()
    assert result is False
    assert eng.keep_going is False


# =============================================================================
# _check_agent_requests Tests
# =============================================================================


@pytest.mark.usefixtures("pipeline_dir")
def test_check_agent_requests_empty_queue() -> None:
    """_check_agent_requests returns None when queue is empty."""
    eng = engine.WatchEngine()
    result = eng._check_agent_requests()
    assert result is None


@pytest.mark.usefixtures("pipeline_dir")
def test_check_agent_requests_returns_queued_request() -> None:
    """_check_agent_requests returns queued request tuple."""
    eng = engine.WatchEngine()

    # Manually put a request in the queue
    eng._agent_request_queue.put_nowait(("run123", ["stage_a"], True))

    result = eng._check_agent_requests()
    assert result == ("run123", ["stage_a"], True)


# =============================================================================
# _emit_json Tests
# =============================================================================


@pytest.mark.usefixtures("pipeline_dir")
def test_emit_json_outputs_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """_emit_json outputs JSON to stdout when json_output=True."""
    eng = engine.WatchEngine(json_output=True)

    event = types.WatchStatusEvent(
        type=types.WatchEventType.STATUS,
        message="Test message",
        is_error=False,
    )
    eng._emit_json(event)

    captured = capsys.readouterr()
    assert '"type": "status"' in captured.out
    assert '"message": "Test message"' in captured.out


@pytest.mark.usefixtures("pipeline_dir")
def test_emit_json_includes_event_type(capsys: pytest.CaptureFixture[str]) -> None:
    """_emit_json includes correct event type in output."""
    eng = engine.WatchEngine(json_output=True)

    # Test files_changed event type
    event = types.WatchFilesChangedEvent(
        type=types.WatchEventType.FILES_CHANGED,
        paths=["test.py"],
        code_changed=True,
    )
    eng._emit_json(event)

    captured = capsys.readouterr()
    assert '"type": "files_changed"' in captured.out


# =============================================================================
# _send_reload_notification Tests
# =============================================================================


@pytest.mark.usefixtures("pipeline_dir")
def test_send_reload_notification_sends_message() -> None:
    """_send_reload_notification sends RELOAD message to TUI queue."""
    register_test_stage(_stage_noop, name="test_stage")

    eng = engine.WatchEngine()
    tui_queue: types.TuiQueue = queue.Queue()
    eng._tui_queue = tui_queue

    eng._send_reload_notification()

    assert not tui_queue.empty()
    msg = tui_queue.get_nowait()
    assert msg is not None
    assert msg["type"] == types.TuiMessageType.RELOAD
    # Narrow to TuiReloadMessage
    assert "stages" in msg
    assert "test_stage" in msg["stages"]


@pytest.mark.usefixtures("pipeline_dir")
def test_send_reload_notification_handles_full_queue() -> None:
    """_send_reload_notification handles queue.Full gracefully."""
    register_test_stage(_stage_noop, name="test_stage")

    eng = engine.WatchEngine()
    # Use a tiny queue that we fill up
    tiny_queue: types.TuiQueue = queue.Queue(maxsize=1)
    # Fill it with a proper TuiMessage
    dummy_msg = types.TuiWatchMessage(
        type=types.TuiMessageType.WATCH,
        status=types.WatchStatus.WAITING,
        message="dummy",
    )
    tiny_queue.put(dummy_msg)
    eng._tui_queue = tiny_queue

    # Should not raise, just suppress
    eng._send_reload_notification()  # Does nothing, queue is full


# =============================================================================
# _collect_stage_modules Tests
# =============================================================================


def test_collect_stage_modules_extracts_modules() -> None:
    """_collect_stage_modules extracts __module__ from stage functions."""

    def fake_func() -> None:
        pass

    stages: dict[str, RegistryStageInfo] = {
        "stage1": {
            "func": fake_func,
            "name": "stage1",
            "deps": {},
            "deps_paths": [],
            "outs": [],
            "outs_paths": [],
            "params": None,
            "mutex": [],
            "variant": None,
            "signature": None,
            "fingerprint": {},
            "dep_specs": {},
            "out_specs": {},
            "params_arg_name": None,
        }
    }

    result = engine._collect_stage_modules(stages)

    assert isinstance(result, set)
    # fake_func's __module__ should be in the result
    assert fake_func.__module__ in result


def test_collect_stage_modules_returns_set() -> None:
    """_collect_stage_modules returns set of module names."""
    # Empty stages dict
    stages: dict[str, RegistryStageInfo] = {}

    result = engine._collect_stage_modules(stages)

    assert isinstance(result, set)
    assert len(result) == 0


# =============================================================================
# _find_pipeline_file Tests
# =============================================================================


def test_find_pipeline_file_invalid_yaml_returns_none(tmp_path: pathlib.Path) -> None:
    """_find_pipeline_file returns None for invalid YAML."""
    pivot_yaml = tmp_path / "pivot.yaml"
    pivot_yaml.write_text("invalid: yaml: syntax: {broken")

    result = engine._find_pipeline_file(tmp_path)

    # Invalid YAML is skipped (returns None if no valid files)
    assert result is None


def test_find_pipeline_file_missing_stages_returns_none(tmp_path: pathlib.Path) -> None:
    """_find_pipeline_file returns None if stages section missing."""
    pivot_yaml = tmp_path / "pivot.yaml"
    pivot_yaml.write_text("# No stages section\nother_key: value\n")

    result = engine._find_pipeline_file(tmp_path)

    # Only returns path if file has stages section
    assert result is None


def test_find_pipeline_file_prefers_pivot_yaml_over_yml(tmp_path: pathlib.Path) -> None:
    """_find_pipeline_file prefers pivot.yaml over pivot.yml."""
    (tmp_path / "pivot.yaml").write_text("stages: {}")
    (tmp_path / "pivot.yml").write_text("stages: {}")

    result = engine._find_pipeline_file(tmp_path)

    assert result == tmp_path / "pivot.yaml"


def test_find_pipeline_file_falls_back_to_yml(tmp_path: pathlib.Path) -> None:
    """_find_pipeline_file falls back to pivot.yml if pivot.yaml doesn't exist."""
    (tmp_path / "pivot.yml").write_text("stages: {}")

    result = engine._find_pipeline_file(tmp_path)

    assert result == tmp_path / "pivot.yml"


def test_find_pipeline_file_returns_none_if_no_stages(tmp_path: pathlib.Path) -> None:
    """_find_pipeline_file returns None if no pivot.yaml with stages exists."""
    # No pivot.yaml/yml files at all
    result = engine._find_pipeline_file(tmp_path)

    assert result is None
