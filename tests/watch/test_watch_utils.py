"""Tests for _watch_utils module."""

import pathlib
from typing import Annotated, TypedDict

import pytest
from watchfiles import Change

from helpers import register_test_stage
from pivot import ignore, loaders, outputs, project
from pivot.watch import _watch_utils

from .watch_types import CreateFilterForStages


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _stage_with_file_dep(
    input_file: Annotated[pathlib.Path, outputs.Dep("data/input.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = input_file
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_with_dir_dep(
    dep_dir: Annotated[pathlib.Path, outputs.Dep("data_dir", loaders.PathOnly())],
) -> _OutputTxt:
    _ = dep_dir
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_with_nonexistent_dep(
    file: Annotated[pathlib.Path, outputs.Dep("nonexistent/file.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = file
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


class _Output1(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output1.txt", loaders.PathOnly())]


class _Output2(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output2.txt", loaders.PathOnly())]


class _IntermediateOut(TypedDict):
    intermediate: Annotated[pathlib.Path, outputs.Out("intermediate.csv", loaders.PathOnly())]


class _FinalOut(TypedDict):
    final: Annotated[pathlib.Path, outputs.Out("final.csv", loaders.PathOnly())]


class _OutputDir(TypedDict):
    output_dir: Annotated[pathlib.Path, outputs.Out("output_dir/", loaders.PathOnly())]


def _stage_output_txt() -> _OutputTxt:
    """Stage that just produces output.txt."""
    pathlib.Path("output.txt").write_text("")
    return _OutputTxt(output=pathlib.Path("output.txt"))


def _stage_output1() -> _Output1:
    """Stage that produces output1.txt."""
    pathlib.Path("output1.txt").write_text("")
    return _Output1(output=pathlib.Path("output1.txt"))


def _stage_output2() -> _Output2:
    """Stage that produces output2.txt."""
    pathlib.Path("output2.txt").write_text("")
    return _Output2(output=pathlib.Path("output2.txt"))


def _stage_intermediate(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _IntermediateOut:
    """Stage that consumes input.csv and produces intermediate.csv."""
    _ = input_file
    pathlib.Path("intermediate.csv").write_text("")
    return _IntermediateOut(intermediate=pathlib.Path("intermediate.csv"))


def _stage_final(
    intermediate: Annotated[pathlib.Path, outputs.Dep("intermediate.csv", loaders.PathOnly())],
) -> _FinalOut:
    """Stage that consumes intermediate.csv and produces final.csv."""
    _ = intermediate
    pathlib.Path("final.csv").write_text("")
    return _FinalOut(final=pathlib.Path("final.csv"))


def _stage_output_dir() -> _OutputDir:
    """Stage that produces an output directory."""
    pathlib.Path("output_dir").mkdir(exist_ok=True)
    return _OutputDir(output_dir=pathlib.Path("output_dir"))


def _noop() -> None:
    """Module-level no-op function for stage registration in tests."""


# =============================================================================
# collect_watch_paths tests
# =============================================================================


def test_collect_watch_paths_includes_project_root(tmp_path: pathlib.Path) -> None:
    """Should always include project root in watch paths."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(project, "_project_root_cache", tmp_path)

        paths = _watch_utils.collect_watch_paths([])

        assert tmp_path in paths


def test_collect_watch_paths_includes_dep_directories(
    tmp_path: pathlib.Path, set_project_root: pathlib.Path
) -> None:
    """Should include directories containing dependencies."""
    dep_file = set_project_root / "data" / "input.csv"
    dep_file.parent.mkdir(parents=True, exist_ok=True)
    dep_file.write_text("x,y\n1,2\n")

    register_test_stage(_stage_with_file_dep, name="my_stage")

    paths = _watch_utils.collect_watch_paths(["my_stage"])

    assert set_project_root in paths
    assert dep_file.parent in paths


def test_collect_watch_paths_includes_directory_deps_directly(
    set_project_root: pathlib.Path,
) -> None:
    """Should include directory dependencies directly (not their parent)."""
    dep_dir = set_project_root / "data_dir"
    dep_dir.mkdir()
    (dep_dir / "file.csv").write_text("x,y\n1,2\n")

    register_test_stage(_stage_with_dir_dep, name="my_stage")

    paths = _watch_utils.collect_watch_paths(["my_stage"])

    assert set_project_root in paths
    assert dep_dir in paths, "Directory dependency should be added directly"
    assert dep_dir.parent not in paths or dep_dir.parent == set_project_root


def test_collect_watch_paths_skips_unknown_stages(
    tmp_path: pathlib.Path, set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Should skip unknown stages with warning."""
    paths = _watch_utils.collect_watch_paths(["nonexistent_stage"])

    assert set_project_root in paths
    assert "not found" in caplog.text


def test_collect_watch_paths_handles_nonexistent_deps(
    tmp_path: pathlib.Path, set_project_root: pathlib.Path
) -> None:
    """Should skip dependencies that don't exist."""
    register_test_stage(_stage_with_nonexistent_dep, name="my_stage")

    paths = _watch_utils.collect_watch_paths(["my_stage"])

    # Should only have project root (nonexistent dep is skipped)
    assert set_project_root in paths
    assert len(paths) == 1


# =============================================================================
# get_output_paths_for_stages tests
# =============================================================================


def test_get_output_paths_for_stages_returns_outputs(
    set_project_root: pathlib.Path,
) -> None:
    """Should return output paths for specified stages."""
    register_test_stage(_stage_output_txt, name="my_stage")

    result = _watch_utils.get_output_paths_for_stages(["my_stage"])

    assert str(set_project_root / "output.txt") in result


def test_get_output_paths_for_stages_skips_unknown(
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Should skip unknown stages with warning."""
    result = _watch_utils.get_output_paths_for_stages(["nonexistent_stage"])

    assert result == set()
    assert "not found" in caplog.text


def test_get_output_paths_for_stages_multiple_stages(
    set_project_root: pathlib.Path,
) -> None:
    """Should collect outputs from multiple stages."""
    register_test_stage(_stage_output1, name="stage1")
    register_test_stage(_stage_output2, name="stage2")

    result = _watch_utils.get_output_paths_for_stages(["stage1", "stage2"])

    assert str(set_project_root / "output1.txt") in result
    assert str(set_project_root / "output2.txt") in result


# =============================================================================
# create_watch_filter tests
# =============================================================================


def test_create_watch_filter_excludes_outputs_during_execution(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """Should filter out output files of watched stages during execution."""
    output_file = set_project_root / "output.txt"
    output_file.write_text("data")

    register_test_stage(_stage_output_txt, name="my_stage")

    output_filter = _watch_utils.OutputFilter(["my_stage"])
    watch_filter = create_filter_for_stages(["my_stage"], output_filter=output_filter)

    # Before execution - outputs should pass through
    assert watch_filter(Change.modified, str(output_file)) is True

    # During execution - outputs should be filtered
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(output_file)) is False
    output_filter.end_execution()


def test_create_watch_filter_allows_non_outputs(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """Should allow files that are not stage outputs."""
    other_file = set_project_root / "other.txt"
    other_file.write_text("data")

    register_test_stage(_stage_output_txt, name="my_stage")

    watch_filter = create_filter_for_stages(["my_stage"])

    assert watch_filter(Change.modified, str(other_file)) is True


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("module.pyc", id="pyc"),
        pytest.param("module.pyo", id="pyo"),
        pytest.param("__pycache__/module.pyc", id="pycache_dir"),
    ],
)
def test_create_watch_filter_excludes_python_bytecode(
    set_project_root: pathlib.Path, path: str
) -> None:
    """Should filter out .pyc files and __pycache__ directories."""
    watch_filter = _watch_utils.create_watch_filter()

    assert watch_filter(Change.modified, path) is False


def test_create_watch_filter_excludes_files_in_output_directories_during_execution(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """Should filter out files inside output directories during execution."""
    output_dir = set_project_root / "output_dir"
    output_dir.mkdir()
    file_in_output = output_dir / "file.txt"
    file_in_output.write_text("data")

    register_test_stage(_stage_output_dir, name="my_stage")

    output_filter = _watch_utils.OutputFilter(["my_stage"])
    watch_filter = create_filter_for_stages(["my_stage"], output_filter=output_filter)

    # During execution - files in output directories should be filtered
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(file_in_output)) is False
    output_filter.end_execution()


def test_create_watch_filter_allows_unresolvable_paths(
    set_project_root: pathlib.Path,
) -> None:
    """Should allow paths that can't be resolved."""
    watch_filter = _watch_utils.create_watch_filter()

    # Path that doesn't exist and can't be resolved
    assert watch_filter(Change.added, "/nonexistent/path/file.txt") is True


def test_create_watch_filter_with_glob_patterns(
    set_project_root: pathlib.Path,
) -> None:
    """Should apply glob patterns when specified."""
    py_file = set_project_root / "script.py"
    txt_file = set_project_root / "data.txt"
    py_file.write_text("# code")
    txt_file.write_text("data")

    watch_filter = _watch_utils.create_watch_filter(watch_globs=["*.py"])

    assert watch_filter(Change.modified, str(py_file)) is True
    assert watch_filter(Change.modified, str(txt_file)) is False


def test_create_watch_filter_glob_matches_full_path_pattern(
    set_project_root: pathlib.Path,
) -> None:
    """Should match globs against full path when pattern includes wildcards."""
    nested_file = set_project_root / "src" / "module.py"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text("# code")

    # Use pattern that matches full path (fnmatch doesn't support **)
    watch_filter = _watch_utils.create_watch_filter(watch_globs=["*/src/*.py"])

    assert watch_filter(Change.modified, str(nested_file)) is True


def test_create_watch_filter_glob_with_no_match(
    set_project_root: pathlib.Path,
) -> None:
    """Should reject files that don't match any glob pattern."""
    csv_file = set_project_root / "data.csv"
    csv_file.write_text("x,y\n1,2")

    watch_filter = _watch_utils.create_watch_filter(watch_globs=["*.py"])

    assert watch_filter(Change.modified, str(csv_file)) is False


def test_create_watch_filter_filters_all_outputs_during_execution(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """During execution, all outputs are filtered to prevent infinite loops.

    After execution, external modifications to intermediate files ARE detected
    (this was a bug fix - previously they were filtered even when not executing).
    """
    input_file = set_project_root / "input.csv"
    intermediate_file = set_project_root / "intermediate.csv"
    final_file = set_project_root / "final.csv"

    input_file.write_text("a,b\n1,2")
    intermediate_file.write_text("x,y\n3,4")

    # stage_a produces intermediate.csv
    register_test_stage(_stage_intermediate, name="stage_a")
    # stage_b consumes intermediate.csv
    register_test_stage(_stage_final, name="stage_b")

    output_filter = _watch_utils.OutputFilter(["stage_a", "stage_b"])
    watch_filter = create_filter_for_stages(["stage_a", "stage_b"], output_filter=output_filter)

    # During execution - outputs are filtered to prevent infinite loops
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(intermediate_file)) is False, (
        "Intermediate outputs are filtered during execution"
    )
    final_file.write_text("result")
    assert watch_filter(Change.modified, str(final_file)) is False, (
        "Terminal outputs are filtered during execution"
    )
    output_filter.end_execution()

    # After execution - external modifications are detected (bug fix)
    assert watch_filter(Change.modified, str(intermediate_file)) is True, (
        "External modifications to intermediate files are detected after execution"
    )


# =============================================================================
# IgnoreFilter integration tests
# =============================================================================


def test_create_watch_filter_uses_ignore_filter(
    set_project_root: pathlib.Path,
) -> None:
    """Watch filter should respect .pivotignore patterns."""
    # Create .pivotignore with custom pattern
    pivotignore = set_project_root / ".pivotignore"
    pivotignore.write_text("*.log\ntemp/\n")

    # Create test files
    log_file = set_project_root / "app.log"
    log_file.write_text("log data")
    py_file = set_project_root / "app.py"
    py_file.write_text("# code")

    ignore_filter = ignore.IgnoreFilter(project_root=set_project_root)
    watch_filter = _watch_utils.create_watch_filter(ignore_filter=ignore_filter)

    assert watch_filter(Change.modified, str(log_file)) is False, "Should filter .log files"
    assert watch_filter(Change.modified, str(py_file)) is True, "Should allow .py files"


def test_create_watch_filter_respects_negation_patterns(
    set_project_root: pathlib.Path,
) -> None:
    """Watch filter should respect negation patterns (!pattern)."""
    # Create .pivotignore with negation
    pivotignore = set_project_root / ".pivotignore"
    pivotignore.write_text("*.log\n!important.log\n")

    # Create test files
    debug_log = set_project_root / "debug.log"
    debug_log.write_text("debug data")
    important_log = set_project_root / "important.log"
    important_log.write_text("important data")

    ignore_filter = ignore.IgnoreFilter(project_root=set_project_root)
    watch_filter = _watch_utils.create_watch_filter(ignore_filter=ignore_filter)

    assert watch_filter(Change.modified, str(debug_log)) is False, "Should filter debug.log"
    assert watch_filter(Change.modified, str(important_log)) is True, "Should allow important.log"


def test_create_watch_filter_ignores_temp_directories(
    set_project_root: pathlib.Path,
) -> None:
    """Watch filter should ignore directories in .pivotignore."""
    # Create .pivotignore with directory pattern
    pivotignore = set_project_root / ".pivotignore"
    pivotignore.write_text("temp/\n")

    # Create test directory and file
    temp_dir = set_project_root / "temp"
    temp_dir.mkdir()
    temp_file = temp_dir / "cache.txt"
    temp_file.write_text("cached")

    ignore_filter = ignore.IgnoreFilter(project_root=set_project_root)
    watch_filter = _watch_utils.create_watch_filter(ignore_filter=ignore_filter)

    assert watch_filter(Change.modified, str(temp_file)) is False, "Should filter files in temp/"


def test_create_watch_filter_without_ignore_filter_uses_hardcoded_patterns(
    set_project_root: pathlib.Path,
) -> None:
    """Without ignore_filter, should still filter Python bytecode (hardcoded)."""
    # No .pivotignore, no ignore_filter passed
    pyc_file = set_project_root / "module.pyc"

    watch_filter = _watch_utils.create_watch_filter()

    # Hardcoded bytecode filtering should still work
    assert watch_filter(Change.modified, str(pyc_file)) is False


# =============================================================================
# OutputFilter tests
# =============================================================================


def test_output_filter_filters_outputs_during_execution(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """Outputs should be filtered only while execution is in progress."""
    output_file = set_project_root / "output.txt"
    output_file.write_text("data")

    register_test_stage(_stage_output_txt, name="my_stage")

    output_filter = _watch_utils.OutputFilter(["my_stage"])
    watch_filter = create_filter_for_stages(["my_stage"], output_filter=output_filter)

    # Before execution - outputs should NOT be filtered (external changes detected)
    assert watch_filter(Change.modified, str(output_file)) is True, (
        "External modifications before first execution should be detected"
    )

    # During execution - outputs must be filtered to prevent infinite loops
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(output_file)) is False, (
        "Outputs must be filtered during execution"
    )

    # After execution - outputs should NOT be filtered (external changes detected)
    output_filter.end_execution()
    assert watch_filter(Change.modified, str(output_file)) is True, (
        "External modifications after execution should be detected"
    )


def test_output_filter_allows_non_outputs_regardless_of_state(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """Non-output files should always pass through, regardless of execution state."""
    other_file = set_project_root / "input.txt"
    other_file.write_text("data")

    register_test_stage(_stage_output_txt, name="my_stage")

    output_filter = _watch_utils.OutputFilter(["my_stage"])
    watch_filter = create_filter_for_stages(["my_stage"], output_filter=output_filter)

    # Non-outputs should always be allowed
    assert watch_filter(Change.modified, str(other_file)) is True, "Before execution"
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(other_file)) is True, "During execution"
    output_filter.end_execution()
    assert watch_filter(Change.modified, str(other_file)) is True, "After execution"


def test_output_filter_handles_deleted_outputs_during_execution(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """Deleted output files should be filtered during execution.

    Pivot deletes outputs before running a stage. These deletions must be filtered
    to prevent spurious re-execution of downstream stages.
    """
    output_file = set_project_root / "output.txt"
    # File exists initially
    output_file.write_text("data")

    register_test_stage(_stage_output_txt, name="my_stage")

    output_filter = _watch_utils.OutputFilter(["my_stage"])
    watch_filter = create_filter_for_stages(["my_stage"], output_filter=output_filter)

    output_filter.start_execution()

    # Delete the file (simulating Pivot clearing outputs before stage runs)
    output_file.unlink()

    # The deletion event should be filtered during execution
    # NOTE: This test fails with time-window approach because stat() fails on deleted files
    assert watch_filter(Change.deleted, str(output_file)) is False, (
        "Output deletions during execution must be filtered"
    )


def test_output_filter_intermediate_file_external_modification(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """External modifications to intermediate files should be detected after execution.

    Intermediate files are outputs of one stage and inputs to another.
    After execution completes, external modifications should trigger re-execution.
    """
    input_file = set_project_root / "input.csv"
    intermediate_file = set_project_root / "intermediate.csv"

    input_file.write_text("a,b\n1,2")
    intermediate_file.write_text("x,y\n3,4")

    # stage_a produces intermediate.csv, stage_b consumes it
    register_test_stage(_stage_intermediate, name="stage_a")
    register_test_stage(_stage_final, name="stage_b")

    output_filter = _watch_utils.OutputFilter(["stage_a", "stage_b"])
    watch_filter = create_filter_for_stages(["stage_a", "stage_b"], output_filter=output_filter)

    # Run the pipeline
    output_filter.start_execution()
    # ... stages execute ...
    output_filter.end_execution()

    # After execution, external modification to intermediate file should be detected
    intermediate_file.write_text("modified externally")
    assert watch_filter(Change.modified, str(intermediate_file)) is True, (
        "External modifications to intermediate files should be detected after execution"
    )


def test_output_filter_multiple_execution_cycles(
    set_project_root: pathlib.Path, create_filter_for_stages: CreateFilterForStages
) -> None:
    """OutputFilter should correctly handle multiple execution cycles."""
    output_file = set_project_root / "output.txt"
    output_file.write_text("data")

    register_test_stage(_stage_output_txt, name="my_stage")

    output_filter = _watch_utils.OutputFilter(["my_stage"])
    watch_filter = create_filter_for_stages(["my_stage"], output_filter=output_filter)

    # First cycle
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(output_file)) is False
    output_filter.end_execution()
    assert watch_filter(Change.modified, str(output_file)) is True

    # Second cycle - should behave the same
    output_filter.start_execution()
    assert watch_filter(Change.modified, str(output_file)) is False
    output_filter.end_execution()
    assert watch_filter(Change.modified, str(output_file)) is True
