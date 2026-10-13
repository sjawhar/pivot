from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

import pandas as pd
import pytest

from pivot import cli, compose, discovery, executor, loaders, project
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    import click.testing
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture

    from conftest import GitRepo

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def make_csv(source: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return source


@compose.stage
def make_output() -> str:
    return "stage output content"


# make_csv writes its output here (WorkspaceStore layout data/<pipeline>/<stage>.<ext>)
_CSV_REL = "data/test/make_csv.csv"

_GET_STAGE_RED = pytest.mark.xfail(
    strict=True,
    reason=(
        "red test for cli-fixes (Task 9): get resolves only slash-free stage names "
        "and restores to identity keys, not workspace paths (storage/restore.py:154-164)"
    ),
)

_DIFF_HEAD_RED = pytest.mark.xfail(
    strict=True,
    reason=(
        "red test for cli-fixes (Task 9): diff cannot read HEAD hashes of single-output "
        "stages; v2 lock outs have key=None and extract_output_hashes_from_lock maps them "
        "to 'unknown' (show/common.py:72-91)"
    ),
)


def _run_csv_stage(
    pipeline: compose.Pipeline, commit_fn: Callable[[str], str], initial: str
) -> pathlib.Path:
    """Register make_csv, run it with `initial` as input, commit, and return the output path."""
    raw = pipeline.root / "data" / "raw" / "source.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(initial)
    with pipeline:
        make_csv(pipeline.input("source.csv", t=pd.DataFrame))
    results = executor.run(pipeline=pipeline)
    assert results["test/make_csv"]["status"] == "ran", results
    output = pipeline.root / _CSV_REL
    assert output.read_text() == initial, "CSV round-trip should reproduce the input"
    commit_fn("Initial CSV output")
    return output


def _rewrite(path: pathlib.Path, content: str) -> None:
    # Remove first: the workspace file may be a link into the cache
    path.unlink()
    path.write_text(content)


# =============================================================================
# Diff Help Tests
# =============================================================================


def test_diff_help(runner: click.testing.CliRunner) -> None:
    """Diff command should show help."""
    result = runner.invoke(cli.cli, ["diff", "--help"])
    assert result.exit_code == 0
    assert "TARGETS" in result.output
    assert "--key" in result.output
    assert "--positional" in result.output
    assert "--no-tui" in result.output
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--summary" in result.output
    assert "--max-rows" in result.output


def test_diff_in_main_help(runner: click.testing.CliRunner) -> None:
    """diff command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "diff" in result.output


def test_get_in_main_help(runner: click.testing.CliRunner) -> None:
    """get command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "get" in result.output


# =============================================================================
# Diff - No Stage Tests
# =============================================================================


def test_diff_no_stages(runner: click.testing.CliRunner, mock_discovery: compose.Pipeline) -> None:
    """Diff with no registered stages should report no data files."""
    result = runner.invoke(cli.cli, ["diff", "--no-tui", "data.csv"])
    assert result.exit_code == 0
    assert "No data files found" in result.output


# =============================================================================
# Diff - Conflicting Options
# =============================================================================


def test_diff_key_and_positional_conflict(
    runner: click.testing.CliRunner, mock_discovery: compose.Pipeline
) -> None:
    """Diff should error when both --key and --positional are specified."""
    (mock_discovery.root / "data.csv").write_text("id,name\n1,alice\n")
    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--key", "id", "--positional", "data.csv"])
    assert result.exit_code != 0
    assert "Cannot use both --key and --positional" in result.output


# =============================================================================
# Diff - Required Arguments
# =============================================================================


def test_diff_requires_targets(
    runner: click.testing.CliRunner, mock_discovery: compose.Pipeline
) -> None:
    """Diff requires at least one target."""
    result = runner.invoke(cli.cli, ["diff", "--no-tui"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output or "required" in result.output.lower()


# =============================================================================
# Diff - CSV File Tests
# =============================================================================


@_DIFF_HEAD_RED
def test_diff_csv_file(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """Diff CSV files against HEAD."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n2,20\n")

    _rewrite(csv_file, "id,value\n1,10\n2,25\n3,30\n")  # Changed row 2, added row 3

    result = runner.invoke(cli.cli, ["diff", "--no-tui", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output
    assert "Rows:" in result.output
    assert "(added)" not in result.output, (
        "File existed at HEAD; it must be diffed as modified, not reported as added"
    )


@_DIFF_HEAD_RED
def test_diff_json_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """--json outputs structured diff."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    _rewrite(csv_file, "id,value\n1,99\n")

    result = runner.invoke(cli.cli, ["diff", "--json", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    data: list[dict[str, object]] = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["path"] == _CSV_REL
    assert data[0]["old_rows"] == 1, "HEAD version (1 row) must be restored from cache"


def test_diff_key_columns(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """--key uses columns for row matching."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,name,value\n1,alice,10\n2,bob,20\n")

    # Modify: update alice's value, add charlie
    _rewrite(csv_file, "id,name,value\n1,alice,15\n2,bob,20\n3,charlie,30\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--key", "id", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output


def test_diff_positional(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """--positional uses row position matching."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n2,20\n")

    # Reorder rows (positional diff will see changes, key-based might not)
    _rewrite(csv_file, "id,value\n2,20\n1,10\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--positional", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output


@_DIFF_HEAD_RED
def test_diff_no_changes_message(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """No changes shows explicit message, not empty output."""
    _, commit = git_repo
    _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    # Don't modify - workspace same as HEAD
    result = runner.invoke(cli.cli, ["diff", "--no-tui", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "No data file changes" in result.output


@_DIFF_HEAD_RED
def test_diff_json_empty_returns_valid_json(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """Empty diff returns valid JSON, not empty string."""
    _, commit = git_repo
    _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    # No changes - same content as HEAD
    result = runner.invoke(cli.cli, ["diff", "--json", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # With --json flag, should always return valid JSON (empty list when no changes)
    data: list[dict[str, object]] = json.loads(result.output)
    assert data == []


# =============================================================================
# Get - Stage and Mode Tests
# =============================================================================


@_GET_STAGE_RED
def test_get_stage_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """get with stage name target restores the stage's outputs at their workspace paths."""
    _, commit = git_repo
    with mock_discovery:
        make_output()
    results = executor.run(pipeline=mock_discovery)
    assert results["test/make_output"]["status"] == "ran", results
    output_file = mock_discovery.root / "data" / "test" / "make_output.txt"
    sha = commit("Stage output")

    # Delete output file
    output_file.unlink()

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "test/make_output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert output_file.read_text() == "stage output content"


def test_get_checkout_mode(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """--checkout-mode affects file restoration."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create file and cache it
    data_file = repo_path / "data.txt"
    data_file.write_text("cached content")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(data_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: data.txt
hash: {output_hash["hash"]}
size: 14
"""
    pvt_path = repo_path / "data.txt.pvt"
    pvt_path.write_text(pvt_content)

    sha = commit("Track data file")

    # Delete data file
    data_file.unlink()
    assert not data_file.exists()

    # Test copy mode
    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--checkout-mode", "copy", "data.txt"],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert data_file.exists()
    # With copy mode, file should not be a symlink
    assert not data_file.is_symlink()
    assert data_file.read_text() == "cached content"


# =============================================================================
# Diff/Get - Pipeline Discovery Tests
# =============================================================================


def test_diff_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """pivot diff discovers pipeline without needing prior command."""
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)
    (repo_path / ".pivot").mkdir()

    # Mock discovery to return a pipeline rooted at the git repo
    pipeline = compose.Pipeline("test", root=repo_path)
    discover = mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    # IMPORTANT: Do NOT mock get_pipeline_from_context - we want to test that
    # ensure_stages_registered() is called and populates the context

    (repo_path / "output.csv").write_text("x,y\n3,4")

    commit("Initial setup")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "output.csv"])

    # Should not raise NoPipelineError - discovery should have been called
    assert "No pipeline found" not in result.output
    discover.assert_called()
    # With no registered stages it reports no data files
    assert result.exit_code == 0, result.output
    assert "No data files found" in result.output


def test_get_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: MonkeyPatch,
) -> None:
    """pivot get works in a project with a pipeline without needing prior command."""
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    # Reset cache so discovery finds this project root
    monkeypatch.setattr(project, "_project_root_cache", None)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    (repo_path / "pipeline.py").write_text(
        "from pivot.compose import Pipeline\n\nwith Pipeline('test') as pipeline:\n    pass\n"
    )

    # Create output and cache it
    output_file = repo_path / "output.csv"
    output_file.write_text("x,y\n3,4")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(output_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: output.csv
hash: {output_hash["hash"]}
size: 8
"""
    (repo_path / "output.csv.pvt").write_text(pvt_content)

    sha = commit("Initial setup with tracked output")

    # Delete output file
    output_file.unlink()
    assert not output_file.exists()

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "output.csv"])

    # Should not raise NoPipelineError - may fail for other reasons but not discovery
    assert "No pipeline" not in result.output
    # Should restore the file successfully
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert output_file.read_text() == "x,y\n3,4"


# =============================================================================
# Diff - Additional Output Format Tests
# =============================================================================


def test_diff_summary_flag(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """--summary flag shows only schema and counts, not detailed rows."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n2,20\n")

    _rewrite(csv_file, "id,value,new_col\n1,10,100\n2,25,200\n3,30,300\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--summary", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output
    # Summary should show counts but not detailed row-by-row changes
    assert "Rows:" in result.output or "Schema" in result.output


def test_diff_markdown_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """--md outputs Markdown-formatted diff."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n1,10\n")

    _rewrite(csv_file, "id,value\n1,99\n")

    result = runner.invoke(cli.cli, ["diff", "--md", _CSV_REL])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # --md produces summary output (same format as plain for now)
    assert _CSV_REL in result.output
    assert "Rows:" in result.output or "Row changes:" in result.output


def test_diff_max_rows_parameter(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """--max-rows limits comparison rows."""
    _, commit = git_repo
    rows = "\n".join([f"{i},{i * 10}" for i in range(1, 101)])
    csv_file = _run_csv_stage(mock_discovery, commit, f"id,value\n{rows}\n")

    # Modify one row
    rows_modified = "\n".join([f"{i},{i * 10 if i != 50 else 9999}" for i in range(1, 101)])
    _rewrite(csv_file, f"id,value\n{rows_modified}\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--max-rows", "10", _CSV_REL])

    # Should succeed without error even with max-rows limit
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output


def test_diff_empty_csv_file(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mock_discovery: compose.Pipeline,
) -> None:
    """Diff handles empty CSV files gracefully."""
    _, commit = git_repo
    csv_file = _run_csv_stage(mock_discovery, commit, "id,value\n")  # Header only

    _rewrite(csv_file, "id,value\n1,10\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", _CSV_REL])

    # Should handle empty file without crashing
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _CSV_REL in result.output


# =============================================================================
# Get - Multiple Targets Tests
# =============================================================================


def test_get_multiple_files(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get multiple files in single invocation."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create and commit multiple files
    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    sha = commit("Multiple files")

    # Modify files
    (repo_path / "file1.txt").write_text("modified1")
    (repo_path / "file2.txt").write_text("modified2")

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "--force", "file1.txt", "file2.txt"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    # Verify files restored
    assert (repo_path / "file1.txt").read_text() == "content1"
    assert (repo_path / "file2.txt").read_text() == "content2"


def test_get_output_with_multiple_targets_fails(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get with -o output path and multiple targets should fail with clear error."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    sha = commit("Multiple files")

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "-o", "output.txt", "file1.txt", "file2.txt"],
    )

    # Should fail - can't use -o with multiple targets
    assert result.exit_code != 0
    assert "output" in result.output.lower() or "multiple" in result.output.lower()
    assert not (repo_path / "output.txt").exists()


@pytest.mark.parametrize(
    ("mode", "expect_symlink"),
    [
        pytest.param("symlink", True, id="symlink"),
        pytest.param("hardlink", False, id="hardlink"),
    ],
)
def test_get_link_checkout_modes(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: MonkeyPatch,
    mode: str,
    expect_symlink: bool,
) -> None:
    """--checkout-mode symlink/hardlink restores via the requested link type."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create file and cache it
    data_file = repo_path / "data.txt"
    data_file.write_text("cached content")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(data_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: data.txt
hash: {output_hash["hash"]}
size: 14
"""
    (repo_path / "data.txt.pvt").write_text(pvt_content)

    sha = commit("Track data file")

    # Delete data file
    data_file.unlink()
    assert not data_file.exists()

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--checkout-mode", mode, "data.txt"],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert data_file.is_symlink() is expect_symlink
    assert data_file.read_text() == "cached content"
