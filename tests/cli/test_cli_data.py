from __future__ import annotations

import pathlib

import click.testing
import pytest

from pivot import cli


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Data Diff Help Tests
# =============================================================================


def test_data_diff_help(runner: click.testing.CliRunner) -> None:
    """Data diff command should show help."""
    result = runner.invoke(cli.cli, ["data", "diff", "--help"])
    assert result.exit_code == 0
    assert "TARGETS" in result.output
    assert "--key" in result.output
    assert "--positional" in result.output
    assert "--no-tui" in result.output
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--summary" in result.output
    assert "--max-rows" in result.output


def test_data_group_help(runner: click.testing.CliRunner) -> None:
    """Data group shows subcommands."""
    result = runner.invoke(cli.cli, ["data", "--help"])
    assert result.exit_code == 0
    assert "diff" in result.output
    assert "get" in result.output


def test_data_in_main_help(runner: click.testing.CliRunner) -> None:
    """Data command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output


# =============================================================================
# Data Diff - No Stage Tests
# =============================================================================


def test_data_diff_no_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Data diff with no registered stages should report no data files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        result = runner.invoke(cli.cli, ["data", "diff", "--no-tui", "data.csv"])
    assert result.exit_code == 0
    assert "No data files found" in result.output


# =============================================================================
# Data Diff - Conflicting Options
# =============================================================================


def test_data_diff_key_and_positional_conflict(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Data diff should error when both --key and --positional are specified."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Need to create a file so the targets validation passes
        pathlib.Path("data.csv").write_text("id,name\n1,alice\n")
        result = runner.invoke(
            cli.cli, ["data", "diff", "--no-tui", "--key", "id", "--positional", "data.csv"]
        )
    assert result.exit_code != 0
    assert "Cannot use both --key and --positional" in result.output


# =============================================================================
# Data Diff - Required Arguments
# =============================================================================


def test_data_diff_requires_targets(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Data diff requires at least one target."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        result = runner.invoke(cli.cli, ["data", "diff", "--no-tui"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output or "required" in result.output.lower()
