import click.testing
import pytest

from pivot import exceptions
from pivot.cli import decorators as cli_decorators


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# pivot_command Tests
# =============================================================================


def test_pivot_command_creates_click_command() -> None:
    """pivot_command creates a click.Command."""

    @cli_decorators.pivot_command()
    def my_command() -> None:
        pass

    assert isinstance(my_command, click.Command)


def test_pivot_command_with_name() -> None:
    """pivot_command accepts custom command name."""

    @cli_decorators.pivot_command("custom-name")
    def my_command() -> None:
        pass

    assert my_command.name == "custom-name"


def test_pivot_command_handles_pivot_error(runner: click.testing.CliRunner) -> None:
    """pivot_command converts PivotError to ClickException with suggestion."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise exceptions.StageNotFoundError("Unknown stage: foo")

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "Unknown stage: foo" in result.output
    assert "pivot list" in result.output


def test_pivot_command_handles_generic_exception(runner: click.testing.CliRunner) -> None:
    """pivot_command converts generic exceptions using repr."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise ValueError("something went wrong")

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "ValueError" in result.output
    assert "something went wrong" in result.output


def test_pivot_command_passes_through_click_exception(runner: click.testing.CliRunner) -> None:
    """pivot_command passes through ClickException unchanged."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise click.ClickException("Custom click error")

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "Custom click error" in result.output


def test_pivot_command_preserves_function_behavior(runner: click.testing.CliRunner) -> None:
    """pivot_command preserves normal function behavior."""

    @cli_decorators.pivot_command()
    def echo_command() -> None:
        click.echo("Hello, world!")

    result = runner.invoke(echo_command)

    assert result.exit_code == 0
    assert "Hello, world!" in result.output


# =============================================================================
# with_error_handling Tests
# =============================================================================


def test_with_error_handling_handles_pivot_error(runner: click.testing.CliRunner) -> None:
    """with_error_handling converts PivotError to ClickException."""

    @click.command()
    @cli_decorators.with_error_handling
    def failing_command() -> None:
        raise exceptions.RemoteNotFoundError("Unknown remote: origin")

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "Unknown remote: origin" in result.output
    assert "pivot remote list" in result.output


def test_with_error_handling_uses_repr_for_generic_exceptions(
    runner: click.testing.CliRunner,
) -> None:
    """with_error_handling uses repr for generic exceptions to avoid empty messages."""

    @click.command()
    @cli_decorators.with_error_handling
    def failing_command() -> None:
        raise RuntimeError()

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "RuntimeError" in result.output


def test_with_error_handling_with_group_command(runner: click.testing.CliRunner) -> None:
    """with_error_handling works with group subcommands."""

    @click.group()
    def my_group() -> None:
        pass

    @my_group.command("sub")
    @cli_decorators.with_error_handling
    def subcommand() -> None:
        raise exceptions.CyclicGraphError("Cycle detected")

    result = runner.invoke(my_group, ["sub"])

    assert result.exit_code != 0
    assert "Cycle detected" in result.output
    assert "circular" in result.output
