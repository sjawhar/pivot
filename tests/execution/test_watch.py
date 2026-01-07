"""Tests for watch mode functionality."""

import pathlib
from unittest import mock

import pytest
from watchfiles import Change

from pivot import project, watch
from pivot.registry import REGISTRY, stage


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pivot.yaml").write_text("version: 1\n")
    REGISTRY.clear()
    return tmp_path


# _collect_watch_paths tests


def test_collect_watch_paths_includes_project_root(pipeline_dir: pathlib.Path) -> None:
    """Project root should always be in watch paths."""
    paths = watch._collect_watch_paths([])
    assert pipeline_dir in paths


def test_collect_watch_paths_includes_dependency_directories(
    pipeline_dir: pathlib.Path,
) -> None:
    """Dependency file directories should be included."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b\n1,2")

    @stage(deps=["data/input.csv"], outs=["output.txt"])
    def process() -> None:
        pass

    paths = watch._collect_watch_paths(["process"])
    assert data_dir in paths


def test_collect_watch_paths_handles_missing_stage(pipeline_dir: pathlib.Path) -> None:
    """Should handle missing stage gracefully."""
    paths = watch._collect_watch_paths(["nonexistent_stage"])
    assert pipeline_dir in paths


def test_collect_watch_paths_handles_nonexistent_dep_path(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should not include paths for dependencies that don't exist."""

    @stage(deps=["nonexistent/file.txt"], outs=["output.txt"])
    def process() -> None:
        pass

    paths = watch._collect_watch_paths(["process"])
    assert len(paths) == 1
    assert pipeline_dir in paths


# _create_output_filter tests


def test_output_filter_filters_exact_output_match(pipeline_dir: pathlib.Path) -> None:
    """Should filter out exact output file paths."""
    output_path = pipeline_dir / "output.txt"

    @stage(deps=[], outs=["output.txt"])
    def process() -> None:
        pass

    watch_filter = watch._create_output_filter(["process"])

    assert watch_filter(Change.modified, str(output_path)) is False


def test_output_filter_filters_files_under_output_directory(
    pipeline_dir: pathlib.Path,
) -> None:
    """Should filter out files under output directories."""
    output_dir = pipeline_dir / "models"
    output_dir.mkdir()

    @stage(deps=[], outs=["models/"])
    def train() -> None:
        pass

    watch_filter = watch._create_output_filter(["train"])

    assert watch_filter(Change.modified, str(output_dir / "checkpoint.pt")) is False


def test_output_filter_filters_directory_itself(pipeline_dir: pathlib.Path) -> None:
    """Should filter out the output directory itself (without trailing slash)."""
    output_dir = pipeline_dir / "models"
    output_dir.mkdir()

    @stage(deps=[], outs=["models/"])
    def train() -> None:
        pass

    watch_filter = watch._create_output_filter(["train"])

    # Directory reported without trailing slash should still be filtered
    assert watch_filter(Change.modified, str(output_dir)) is False


def test_output_filter_allows_source_files(pipeline_dir: pathlib.Path) -> None:
    """Should allow source files that are not outputs."""
    source_path = pipeline_dir / "src" / "main.py"

    @stage(deps=[], outs=["output.txt"])
    def process() -> None:
        pass

    watch_filter = watch._create_output_filter(["process"])

    assert watch_filter(Change.modified, str(source_path)) is True


@pytest.mark.parametrize(
    "path",
    [
        "/some/path/file.pyc",
        "/some/path/__pycache__/file.py",
        "/some/path/file.pyo",
    ],
)
def test_output_filter_filters_python_bytecode(pipeline_dir: pathlib.Path, path: str) -> None:
    """Should filter out .pyc, .pyo, and __pycache__ files."""
    watch_filter = watch._create_output_filter([])
    assert watch_filter(Change.modified, path) is False


def test_output_filter_resolves_symlinks(pipeline_dir: pathlib.Path) -> None:
    """Should filter symlinked paths that resolve to outputs."""
    # Create output directory
    output_dir = pipeline_dir / "real_output"
    output_dir.mkdir()

    # Create symlink to output directory
    link_path = pipeline_dir / "link_to_output"
    link_path.symlink_to(output_dir)

    @stage(deps=[], outs=["real_output/"])
    def process() -> None:
        pass

    watch_filter = watch._create_output_filter(["process"])

    # Path via symlink should be filtered (resolves to same location)
    assert watch_filter(Change.modified, str(link_path / "file.txt")) is False
    # Direct path should also be filtered
    assert watch_filter(Change.modified, str(output_dir / "file.txt")) is False


# Glob filter tests


def test_output_filter_with_glob_matches_filename(pipeline_dir: pathlib.Path) -> None:
    """Glob filter should match by filename."""
    watch_filter = watch._create_output_filter([], watch_globs=["*.py"])

    assert watch_filter(Change.modified, "/some/path/script.py") is True
    assert watch_filter(Change.modified, "/some/path/data.csv") is False


def test_output_filter_with_glob_matches_path_pattern(pipeline_dir: pathlib.Path) -> None:
    """Glob filter should match full path patterns."""
    watch_filter = watch._create_output_filter([], watch_globs=["*/src/*"])

    assert watch_filter(Change.modified, "/project/src/main.py") is True
    assert watch_filter(Change.modified, "/project/tests/test.py") is False


def test_output_filter_with_multiple_globs(pipeline_dir: pathlib.Path) -> None:
    """Multiple globs should be OR'd together."""
    watch_filter = watch._create_output_filter([], watch_globs=["*.py", "*.txt"])

    assert watch_filter(Change.modified, "/path/script.py") is True
    assert watch_filter(Change.modified, "/path/readme.txt") is True
    assert watch_filter(Change.modified, "/path/data.csv") is False


def test_output_filter_glob_still_filters_outputs(pipeline_dir: pathlib.Path) -> None:
    """Glob filter should still exclude stage outputs."""
    output_path = pipeline_dir / "output.py"

    @stage(deps=[], outs=["output.py"])
    def process() -> None:
        pass

    watch_filter = watch._create_output_filter(["process"], watch_globs=["*.py"])

    # Even though it matches *.py, it should be filtered as an output
    assert watch_filter(Change.modified, str(output_path)) is False


def test_output_filter_excludes_intermediate_files(pipeline_dir: pathlib.Path) -> None:
    """Intermediate files (output of one stage, dependency of another) should be filtered."""
    intermediate_path = pipeline_dir / "intermediate.txt"

    @stage(deps=[], outs=["intermediate.txt"])
    def stage_a() -> None:
        pass

    @stage(deps=["intermediate.txt"], outs=["final.txt"])
    def stage_b() -> None:
        pass

    # Both stages running - intermediate.txt is filtered as output of stage_a
    watch_filter = watch._create_output_filter(["stage_a", "stage_b"])

    # intermediate.txt is an OUTPUT of stage_a, so it should be filtered
    # even though it's also a dependency of stage_b
    assert watch_filter(Change.modified, str(intermediate_path)) is False


# project.resolve_path tests (used by watch module)


def test_resolve_path_returns_realpath(pipeline_dir: pathlib.Path) -> None:
    """resolve_path should resolve symlinks."""
    real_file = pipeline_dir / "real.txt"
    real_file.write_text("content")
    link_file = pipeline_dir / "link.txt"
    link_file.symlink_to(real_file)

    resolved = project.resolve_path(str(link_file))
    assert resolved == real_file.resolve()


def test_resolve_path_handles_nonexistent_path() -> None:
    """resolve_path should handle non-existent paths gracefully."""
    result = project.resolve_path("/nonexistent/path/file.txt")
    # Should return a Path, not crash
    assert isinstance(result, pathlib.Path)


# run_watch_loop tests


def test_watch_loop_exits_on_keyboard_interrupt_during_wait(
    pipeline_dir: pathlib.Path,
) -> None:
    """Watch loop should exit cleanly on KeyboardInterrupt while waiting."""
    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    def mock_wait(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    with (
        mock.patch.object(watch, "executor") as mock_executor,
        mock.patch.object(watch, "_wait_for_changes", mock_wait),
    ):
        reloads = watch.run_watch_loop()

        assert reloads == 0
        assert mock_executor.run.call_count == 1


def test_watch_loop_exits_on_keyboard_interrupt_during_pipeline(
    pipeline_dir: pathlib.Path,
) -> None:
    """Watch loop should exit cleanly on KeyboardInterrupt during pipeline execution."""
    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pass

    with mock.patch.object(watch, "executor") as mock_executor:
        mock_executor.run.side_effect = KeyboardInterrupt

        reloads = watch.run_watch_loop()

        assert reloads == 0
        assert mock_executor.run.call_count == 1


def test_watch_loop_continues_on_pipeline_error(pipeline_dir: pathlib.Path) -> None:
    """Watch loop should continue watching after pipeline error."""
    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pass

    call_count = 0

    def run_side_effect(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Pipeline failed")
        raise KeyboardInterrupt

    def mock_wait(*args: object, **kwargs: object) -> set[tuple[Change, str]]:
        return {(Change.modified, str(pipeline_dir / "input.txt"))}

    def mock_quiet(*args: object, **kwargs: object) -> None:
        pass

    with (
        mock.patch.object(watch, "executor") as mock_executor,
        mock.patch.object(watch, "_wait_for_changes", mock_wait),
        mock.patch.object(watch, "_wait_for_quiet_period", mock_quiet),
    ):
        mock_executor.run.side_effect = run_side_effect

        reloads = watch.run_watch_loop()

        assert mock_executor.run.call_count == 2
        assert reloads == 1, "Should have reloaded once after detecting changes"


def test_watch_loop_passes_globs_to_filter(pipeline_dir: pathlib.Path) -> None:
    """Watch loop should pass watch_globs to the output filter."""
    (pipeline_dir / "input.txt").write_text("data")

    @stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pass

    def mock_wait(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    with (
        mock.patch.object(watch, "executor"),
        mock.patch.object(watch, "_wait_for_changes", mock_wait),
        mock.patch.object(watch, "_create_output_filter") as mock_create_filter,
    ):
        mock_create_filter.return_value = lambda c, p: True  # pyright: ignore[reportUnknownLambdaType]

        watch.run_watch_loop(watch_globs=["*.py", "*.txt"])

        # First arg is stages_to_run, second is watch_globs
        mock_create_filter.assert_called_with(["process"], ["*.py", "*.txt"])


# get_all_output_paths tests


def test_registry_get_all_output_paths(pipeline_dir: pathlib.Path) -> None:
    """REGISTRY.get_all_output_paths() should return all registered output paths."""

    @stage(deps=[], outs=["output1.txt"])
    def stage1() -> None:
        pass

    @stage(deps=[], outs=["models/"])
    def stage2() -> None:
        pass

    paths = REGISTRY.get_all_output_paths()

    assert len(paths) == 2
    assert any("output1.txt" in p for p in paths)
    assert any("models" in p for p in paths)
