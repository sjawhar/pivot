from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from pivot.cli import _LAZY_COMMANDS, cli

if TYPE_CHECKING:
    from click.testing import CliRunner

# Maximum allowed time for --help (in seconds)
# With lazy command loading, --help should complete in ~0.2s
MAX_HELP_TIME_SECONDS = 0.25


def _get_all_commands() -> list[str]:
    """Get all top-level CLI commands."""
    return sorted(_LAZY_COMMANDS.keys())


# Get commands at module load time for parametrization
ALL_COMMANDS = _get_all_commands()


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
