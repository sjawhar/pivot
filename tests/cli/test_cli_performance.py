from __future__ import annotations

import time

import pytest
from click.testing import CliRunner

from pivot.cli import cli

# Maximum allowed time for --help (in seconds)
# This is generous to account for CI variability
MAX_HELP_TIME_SECONDS = 2.0


def _get_all_commands() -> list[str]:
    """Get all top-level CLI commands."""
    return list(cli.commands.keys())


# Get commands at module load time for parametrization
ALL_COMMANDS = _get_all_commands()


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_command_help_performance(runner: CliRunner, command: str) -> None:
    """Each command's --help should complete within the time limit."""
    start = time.perf_counter()
    result = runner.invoke(cli, [command, "--help"])
    elapsed = time.perf_counter() - start

    assert result.exit_code == 0, f"{command} --help failed: {result.output}"
    assert elapsed < MAX_HELP_TIME_SECONDS, (
        f"{command} --help took {elapsed:.2f}s (max: {MAX_HELP_TIME_SECONDS}s)"
    )


def test_main_help_performance(runner: CliRunner) -> None:
    """Main pivot --help should complete within the time limit."""
    start = time.perf_counter()
    result = runner.invoke(cli, ["--help"])
    elapsed = time.perf_counter() - start

    assert result.exit_code == 0, f"pivot --help failed: {result.output}"
    assert elapsed < MAX_HELP_TIME_SECONDS, (
        f"pivot --help took {elapsed:.2f}s (max: {MAX_HELP_TIME_SECONDS}s)"
    )
