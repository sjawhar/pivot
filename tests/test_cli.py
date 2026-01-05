"""Tests for CLI module."""

import pathlib

import click.testing
import pytest

from pivot import cli, stage
from pivot.registry import REGISTRY


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


@pytest.fixture
def clean_registry() -> None:
    """Clear the registry before each test."""
    REGISTRY.clear()


def test_cli_help_shows_commands(runner: click.testing.CliRunner) -> None:
    """CLI should show available commands in help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output


def test_cli_run_help(runner: click.testing.CliRunner) -> None:
    """Run subcommand should show its own help."""
    result = runner.invoke(cli.cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--stages" in result.output
    assert "--dry-run" in result.output


def test_cli_run_no_stages_registered(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, clean_registry: None
) -> None:
    """Run with no stages should report empty pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        (pathlib.Path.cwd() / ".git").mkdir()
        result = runner.invoke(cli.cli, ["run"])
        assert result.exit_code == 0
        assert "0 ran, 0 skipped" in result.output


def test_cli_verbose_accepted(runner: click.testing.CliRunner) -> None:
    """Verbose flag should be accepted."""
    result = runner.invoke(cli.cli, ["--verbose", "run", "--help"])
    assert result.exit_code == 0


def test_cli_status_command_exists(runner: click.testing.CliRunner) -> None:
    """Status subcommand should be available."""
    result = runner.invoke(cli.cli, ["status", "--help"])
    assert result.exit_code == 0


def test_cli_status_shows_registered_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, clean_registry: None
) -> None:
    """Status should show registered stages."""
    output_file = tmp_path / "out.txt"

    @stage(deps=[], outs=[str(output_file)])
    def my_stage() -> None:
        pass

    result = runner.invoke(cli.cli, ["status"])
    assert result.exit_code == 0
    assert "my_stage" in result.output


def test_cli_status_verbose_shows_details(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, clean_registry: None
) -> None:
    """Status --verbose should show deps and outputs."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"

    @stage(deps=[str(input_file)], outs=[str(output_file)])
    def my_stage() -> None:
        pass

    result = runner.invoke(cli.cli, ["--verbose", "status"])
    assert result.exit_code == 0
    assert "my_stage" in result.output
    assert "deps:" in result.output
    assert "outs:" in result.output
