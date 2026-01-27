from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pytest

from conftest import GitRepo, init_git_repo
from pivot import cli

if TYPE_CHECKING:
    import click.testing


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


def test_plots_show_no_plots(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """plots show with no registered plots should report empty."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["plots", "show"])
        assert result.exit_code == 0
        assert "No plots found" in result.output


def test_plots_show_explicit_file_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Issue #62: plots show TARGET should work with explicit file when no stages registered."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Create a plot file
        plot_file = pathlib.Path("results.png")
        plot_file.write_bytes(b"fake png data")

        # Should work even with no stages registered
        result = runner.invoke(cli.cli, ["plots", "show", str(plot_file)])

        # Should not fail with "stage not found" error
        assert result.exit_code == 0
        assert "Rendered 1 plot(s)" in result.output


def test_plots_show_creates_html(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots show creates HTML output file with explicit file target."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    with runner.isolated_filesystem(temp_dir=set_project_root):
        result = runner.invoke(cli.cli, ["plots", "show", str(plot_file)])
        assert result.exit_code == 0
        assert "Rendered 1 plot(s)" in result.output


def test_plots_show_custom_output_path(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """plots show with custom output path."""
    plot_file = set_project_root / "plot.png"
    plot_file.write_bytes(b"fake png data")

    with runner.isolated_filesystem(temp_dir=set_project_root):
        result = runner.invoke(
            cli.cli, ["plots", "show", str(plot_file), "-o", "custom/output.html"]
        )
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


def test_plots_diff_no_plots(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """plots diff with no registered plots should report empty."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["plots", "diff"])
        assert result.exit_code == 0
        assert "No plots found" in result.output


def test_plots_diff_explicit_file_no_stages(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #62: plots diff TARGET should work with explicit file when no stages registered."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_git_repo(pathlib.Path("."), monkeypatch)
        # Create a plot file
        plot_file = pathlib.Path("results.png")
        plot_file.write_bytes(b"fake png data")

        # Should work even with no stages registered
        result = runner.invoke(cli.cli, ["plots", "diff", str(plot_file)])

        # Should not fail with "stage not found" error
        assert result.exit_code == 0


def test_plots_diff_json_format(runner: click.testing.CliRunner, git_repo: GitRepo) -> None:
    """plots diff --json returns valid JSON with explicit file target."""
    repo_path, commit = git_repo

    plot_file = repo_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    # Commit the initial plot file
    commit("initial")

    # Modify the plot file
    plot_file.write_bytes(b"modified png data")

    with runner.isolated_filesystem(temp_dir=repo_path):
        result = runner.invoke(cli.cli, ["plots", "diff", "--json", str(plot_file)])
        assert result.exit_code == 0
        # Output should be valid JSON
        try:
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pytest.fail(f"Output is not valid JSON: {result.output}")


def test_plots_diff_md_format(runner: click.testing.CliRunner, git_repo: GitRepo) -> None:
    """plots diff --md returns markdown table with explicit file target."""
    repo_path, commit = git_repo

    plot_file = repo_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    # Commit the initial plot file
    commit("initial")

    # Modify the plot file
    plot_file.write_bytes(b"modified png data")

    with runner.isolated_filesystem(temp_dir=repo_path):
        result = runner.invoke(cli.cli, ["plots", "diff", "--md", str(plot_file)])
        assert result.exit_code == 0
        assert "|" in result.output  # Markdown table uses pipes


def test_plots_diff_no_path_flag(runner: click.testing.CliRunner, git_repo: GitRepo) -> None:
    """plots diff --no-path hides path column with explicit file target."""
    repo_path, commit = git_repo

    plot_file = repo_path / "plot.png"
    plot_file.write_bytes(b"fake png data")

    # Commit the initial plot file
    commit("initial")

    # Modify the plot file
    plot_file.write_bytes(b"modified png data")

    with runner.isolated_filesystem(temp_dir=repo_path):
        result = runner.invoke(cli.cli, ["plots", "diff", "--no-path", str(plot_file)])
        assert result.exit_code == 0
        assert "Path" not in result.output
