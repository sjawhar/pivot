from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from conftest import GitRepo, init_git_repo
from pivot import cli

if TYPE_CHECKING:
    import pathlib

    import click.testing

    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# plots group Tests
# =============================================================================


def test_plots_group_help(runner: click.testing.CliRunner) -> None:
    """plots group should show available commands."""
    result = runner.invoke(cli.cli, ["plots", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "diff" in result.output


def test_plots_in_main_help(runner: click.testing.CliRunner) -> None:
    """plots command should appear in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "plots" in result.output


# =============================================================================
# plots show Tests
# =============================================================================


def test_plots_show_help(runner: click.testing.CliRunner) -> None:
    """plots show should show its help."""
    result = runner.invoke(cli.cli, ["plots", "show", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--open" in result.output


def test_plots_show_no_plots(
    mock_discovery: Pipeline, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """plots show with no registered plots should report empty."""
    _ = mock_discovery
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["plots", "show"])
        assert result.exit_code == 0
        assert "No plots found" in result.output


def test_plots_show_explicit_file_no_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #62: plots show TARGET should work with explicit file when no stages registered."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    # Create a plot file
    plot_file = tmp_path / "results.png"
    plot_file.write_bytes(b"fake png data")

    # Should work even with no stages registered
    result = runner.invoke(cli.cli, ["plots", "show", "results.png"])

    # Should not fail with "stage not found" error
    assert result.exit_code == 0
    assert "Rendered 1 plot(s)" in result.output


def test_plots_show_creates_html(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plots show creates HTML output file with explicit file target."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    result = runner.invoke(cli.cli, ["plots", "show", "plot.png"])
    assert result.exit_code == 0
    assert "Rendered 1 plot(s)" in result.output


def test_plots_show_custom_output_path(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plots show with custom output path."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    plot_file = tmp_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    result = runner.invoke(cli.cli, ["plots", "show", "plot.png", "-o", "custom/output.html"])
    assert result.exit_code == 0
    assert "custom/output.html" in result.output


# =============================================================================
# plots diff Tests
# =============================================================================


def test_plots_diff_help(runner: click.testing.CliRunner) -> None:
    """plots diff should show its help."""
    result = runner.invoke(cli.cli, ["plots", "diff", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--no-path" in result.output


def test_plots_diff_no_plots(
    mock_discovery: Pipeline, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """plots diff with no registered plots should report empty."""
    _ = mock_discovery
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["plots", "diff"])
        assert result.exit_code == 0
        assert "No plots found" in result.output


def test_plots_diff_explicit_file_no_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #62: plots diff TARGET should work with explicit file when no stages registered."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    init_git_repo(tmp_path, monkeypatch)
    # Create a plot file
    plot_file = tmp_path / "results.png"
    plot_file.write_bytes(b"fake png data")

    # Should work even with no stages registered
    result = runner.invoke(cli.cli, ["plots", "diff", "results.png"])

    # Should not fail with "stage not found" error
    assert result.exit_code == 0


def test_plots_diff_json_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plots diff --json returns valid JSON with explicit file target."""
    _ = mock_discovery
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)

    plot_file = repo_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    # Commit the initial plot file
    commit("initial")

    # Modify the plot file
    plot_file.write_bytes(b"modified png data")

    result = runner.invoke(cli.cli, ["plots", "diff", "--json", "plot.png"])
    assert result.exit_code == 0
    # Output should be valid JSON
    try:
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
    except json.JSONDecodeError:
        pytest.fail(f"Output is not valid JSON: {result.output}")


def test_plots_diff_md_format(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plots diff --md returns markdown table with explicit file target."""
    _ = mock_discovery
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)

    plot_file = repo_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    # Commit the initial plot file
    commit("initial")

    # Modify the plot file
    plot_file.write_bytes(b"modified png data")

    result = runner.invoke(cli.cli, ["plots", "diff", "--md", "plot.png"])
    assert result.exit_code == 0
    assert "|" in result.output  # Markdown table uses pipes


def test_plots_diff_no_path_flag(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plots diff --no-path hides path column with explicit file target."""
    _ = mock_discovery
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)

    plot_file = repo_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    # Commit the initial plot file
    commit("initial")

    # Modify the plot file
    plot_file.write_bytes(b"modified png data")

    result = runner.invoke(cli.cli, ["plots", "diff", "--no-path", "plot.png"])
    assert result.exit_code == 0
    assert "Path" not in result.output
