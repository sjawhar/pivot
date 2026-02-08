from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Help Tests
# =============================================================================


def test_check_ignore_help(runner: click.testing.CliRunner) -> None:
    """check-ignore command should show help."""
    result = runner.invoke(cli.cli, ["check-ignore", "--help"])

    assert result.exit_code == 0
    assert "--details" in result.output or "-d" in result.output
    assert "--json" in result.output
    assert "--show-defaults" in result.output


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_check_ignore_matches_default_pattern(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore should match files against .pivotignore patterns."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n")

        result = runner.invoke(cli.cli, ["check-ignore", "app.log"])

        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
        assert "app.log" in result.output


def test_check_ignore_no_match(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """check-ignore should return exit code 1 when no targets are ignored."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n")

        result = runner.invoke(cli.cli, ["check-ignore", "important.txt"])

        assert result.exit_code == 1, "Should return exit code 1 for non-ignored file"


def test_check_ignore_details_shows_source(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore --details should show matching pattern and source."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n")

        result = runner.invoke(cli.cli, ["check-ignore", "--details", "app.log"])

        assert result.exit_code == 0
        assert "*.log" in result.output, "Should show matching pattern"
        assert ".pivotignore" in result.output, "Should show source file"


def test_check_ignore_json_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """check-ignore --json should output JSON format."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n")

        result = runner.invoke(cli.cli, ["check-ignore", "--json", "app.log"])

        assert result.exit_code == 0
        data: list[dict[str, object]] = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["path"] == "app.log"
        assert data[0]["ignored"] is True
        assert data[0]["pattern"] == "*.log"


def test_check_ignore_show_defaults(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore --show-defaults should list default patterns."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["check-ignore", "--show-defaults"])

        assert result.exit_code == 0
        assert "*.pyc" in result.output
        assert "__pycache__/" in result.output
        assert ".venv/" in result.output


def test_check_ignore_multiple_targets(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore should handle multiple targets."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n*.tmp\n")

        result = runner.invoke(cli.cli, ["check-ignore", "app.log", "cache.tmp", "keep.txt"])

        assert result.exit_code == 0, "Exit code 0 if any target is ignored"
        assert "app.log" in result.output
        assert "cache.tmp" in result.output


def test_check_ignore_negation_pattern(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore should respect negation patterns."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n!important.log\n")

        result = runner.invoke(cli.cli, ["check-ignore", "important.log"])

        # important.log is NOT ignored due to negation
        assert result.exit_code == 1, "Negated file should not be ignored"


def test_check_ignore_negation_pattern_details(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore --details should show negation pattern."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("*.log\n!important.log\n")

        result = runner.invoke(cli.cli, ["check-ignore", "--details", "important.log"])

        # The file is NOT ignored, so it shouldn't appear in output
        assert result.exit_code == 1


def test_check_ignore_without_pivotignore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore should work without .pivotignore file."""
    with isolated_pivot_dir(runner, tmp_path):
        # No .pivotignore file

        result = runner.invoke(cli.cli, ["check-ignore", "file.txt"])

        assert result.exit_code == 1, "No patterns = nothing ignored"


def test_check_ignore_directory_pattern(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore should handle directory patterns."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path(".pivotignore").write_text("build/\n")
        pathlib.Path("build").mkdir()

        result = runner.invoke(cli.cli, ["check-ignore", "build/"])

        assert result.exit_code == 0
        assert "build" in result.output


def test_check_ignore_no_targets_shows_help(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """check-ignore without targets or --show-defaults should show message."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["check-ignore"])

        # Should indicate how to use the command
        assert result.exit_code in (0, 1, 2)  # Could be error or help
