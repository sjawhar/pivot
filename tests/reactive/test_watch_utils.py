"""Tests for _watch_utils module."""

import pathlib

import pytest
from watchfiles import Change

from pivot import project, registry
from pivot.reactive import _watch_utils


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
    output = set_project_root / "output.txt"

    registry.REGISTRY.register(
        _noop,
        name="my_stage",
        deps=[str(dep_file)],
        outs=[str(output)],
    )

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
    output = set_project_root / "output.txt"

    registry.REGISTRY.register(
        _noop,
        name="my_stage",
        deps=[str(dep_dir)],
        outs=[str(output)],
    )

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
    output = set_project_root / "output.txt"

    registry.REGISTRY.register(
        _noop,
        name="my_stage",
        deps=["nonexistent/file.csv"],
        outs=[str(output)],
    )

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
    output_file = set_project_root / "output.txt"

    registry.REGISTRY.register(
        _noop,
        name="my_stage",
        deps=[],
        outs=[str(output_file)],
    )

    result = _watch_utils.get_output_paths_for_stages(["my_stage"])

    assert str(output_file) in result


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
    output1 = set_project_root / "output1.txt"
    output2 = set_project_root / "output2.txt"

    registry.REGISTRY.register(_noop, name="stage1", deps=[], outs=[str(output1)])
    registry.REGISTRY.register(_noop, name="stage2", deps=[], outs=[str(output2)])

    result = _watch_utils.get_output_paths_for_stages(["stage1", "stage2"])

    assert str(output1) in result
    assert str(output2) in result


# =============================================================================
# create_watch_filter tests
# =============================================================================


def test_create_watch_filter_excludes_outputs(
    set_project_root: pathlib.Path,
) -> None:
    """Should filter out output files of watched stages."""
    output_file = set_project_root / "output.txt"
    output_file.write_text("data")

    registry.REGISTRY.register(_noop, name="my_stage", deps=[], outs=[str(output_file)])

    watch_filter = _watch_utils.create_watch_filter(["my_stage"])

    assert watch_filter(Change.modified, str(output_file)) is False


def test_create_watch_filter_allows_non_outputs(
    set_project_root: pathlib.Path,
) -> None:
    """Should allow files that are not stage outputs."""
    output_file = set_project_root / "output.txt"
    other_file = set_project_root / "other.txt"
    other_file.write_text("data")

    registry.REGISTRY.register(_noop, name="my_stage", deps=[], outs=[str(output_file)])

    watch_filter = _watch_utils.create_watch_filter(["my_stage"])

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
    watch_filter = _watch_utils.create_watch_filter([])

    assert watch_filter(Change.modified, path) is False


def test_create_watch_filter_excludes_files_in_output_directories(
    set_project_root: pathlib.Path,
) -> None:
    """Should filter out files inside output directories."""
    output_dir = set_project_root / "output_dir"
    output_dir.mkdir()
    file_in_output = output_dir / "file.txt"
    file_in_output.write_text("data")

    registry.REGISTRY.register(_noop, name="my_stage", deps=[], outs=[str(output_dir)])

    watch_filter = _watch_utils.create_watch_filter(["my_stage"])

    assert watch_filter(Change.modified, str(file_in_output)) is False


def test_create_watch_filter_allows_unresolvable_paths(
    set_project_root: pathlib.Path,
) -> None:
    """Should allow paths that can't be resolved."""
    watch_filter = _watch_utils.create_watch_filter([])

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

    watch_filter = _watch_utils.create_watch_filter([], watch_globs=["*.py"])

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
    watch_filter = _watch_utils.create_watch_filter([], watch_globs=["*/src/*.py"])

    assert watch_filter(Change.modified, str(nested_file)) is True


def test_create_watch_filter_glob_with_no_match(
    set_project_root: pathlib.Path,
) -> None:
    """Should reject files that don't match any glob pattern."""
    csv_file = set_project_root / "data.csv"
    csv_file.write_text("x,y\n1,2")

    watch_filter = _watch_utils.create_watch_filter([], watch_globs=["*.py"])

    assert watch_filter(Change.modified, str(csv_file)) is False


def test_create_watch_filter_filters_all_outputs_including_intermediate(
    set_project_root: pathlib.Path,
) -> None:
    """All outputs are filtered to prevent infinite loops, even intermediate files.

    This is a known limitation: external modifications to intermediate files
    (outputs that are also deps of downstream stages) are not detected.
    Workaround: modify an input file to trigger a full re-run.
    """
    input_file = set_project_root / "input.csv"
    intermediate_file = set_project_root / "intermediate.csv"
    final_file = set_project_root / "final.csv"

    input_file.write_text("a,b\n1,2")
    intermediate_file.write_text("x,y\n3,4")

    # stage_a produces intermediate.csv
    registry.REGISTRY.register(
        _noop, name="stage_a", deps=[str(input_file)], outs=[str(intermediate_file)]
    )
    # stage_b consumes intermediate.csv
    registry.REGISTRY.register(
        _noop, name="stage_b", deps=[str(intermediate_file)], outs=[str(final_file)]
    )

    watch_filter = _watch_utils.create_watch_filter(["stage_a", "stage_b"])

    # Both intermediate and final outputs are filtered to prevent infinite loops
    assert watch_filter(Change.modified, str(intermediate_file)) is False, (
        "Intermediate outputs are filtered (known limitation)"
    )
    final_file.write_text("result")
    assert watch_filter(Change.modified, str(final_file)) is False, "Terminal outputs are filtered"
