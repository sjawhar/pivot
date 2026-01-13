from __future__ import annotations

from typing import TYPE_CHECKING

import click

from pivot import discovery, exceptions
from pivot.cli import decorators as cli_decorators

if TYPE_CHECKING:
    from click.testing import CliRunner
    from pytest_mock import MockerFixture

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


def test_pivot_command_handles_pivot_error(runner: CliRunner) -> None:
    """pivot_command converts PivotError to ClickException with suggestion."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise exceptions.StageNotFoundError(["foo"])

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "Unknown stage(s): foo" in result.output
    assert "pivot list" in result.output


def test_pivot_command_handles_generic_exception(runner: CliRunner) -> None:
    """pivot_command converts generic exceptions using repr."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise ValueError("something went wrong")

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "ValueError" in result.output
    assert "something went wrong" in result.output


def test_pivot_command_passes_through_click_exception(runner: CliRunner) -> None:
    """pivot_command passes through ClickException unchanged."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise click.ClickException("Custom click error")

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "Custom click error" in result.output


def test_pivot_command_preserves_function_behavior(runner: CliRunner) -> None:
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


def test_with_error_handling_handles_pivot_error(runner: CliRunner) -> None:
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
    runner: CliRunner,
) -> None:
    """with_error_handling uses repr for generic exceptions to avoid empty messages."""

    @click.command()
    @cli_decorators.with_error_handling
    def failing_command() -> None:
        raise RuntimeError()

    result = runner.invoke(failing_command)

    assert result.exit_code != 0
    assert "RuntimeError" in result.output


def test_with_error_handling_with_group_command(runner: CliRunner) -> None:
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


# =============================================================================
# auto_discover Tests
# =============================================================================


def test_pivot_command_auto_discover_calls_discovery_when_no_stages(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover=True calls discover_and_register when no stages registered."""
    mock_has_stages = mocker.patch.object(discovery, "has_registered_stages", return_value=False)
    mock_discover = mocker.patch.object(discovery, "discover_and_register")

    @cli_decorators.pivot_command()
    def my_command() -> None:
        click.echo("Command executed")

    result = runner.invoke(my_command)

    assert result.exit_code == 0
    mock_has_stages.assert_called_once()
    mock_discover.assert_called_once()
    assert "Command executed" in result.output


def test_pivot_command_auto_discover_skips_when_stages_exist(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover=True skips discovery when stages already registered."""
    mock_has_stages = mocker.patch.object(discovery, "has_registered_stages", return_value=True)
    mock_discover = mocker.patch.object(discovery, "discover_and_register")

    @cli_decorators.pivot_command()
    def my_command() -> None:
        click.echo("Command executed")

    result = runner.invoke(my_command)

    assert result.exit_code == 0
    mock_has_stages.assert_called_once()
    mock_discover.assert_not_called()
    assert "Command executed" in result.output


def test_pivot_command_auto_discover_false_skips_discovery(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover=False skips discovery entirely."""
    mock_has_stages = mocker.patch.object(discovery, "has_registered_stages")
    mock_discover = mocker.patch.object(discovery, "discover_and_register")

    @cli_decorators.pivot_command(auto_discover=False)
    def my_command() -> None:
        click.echo("Command executed")

    result = runner.invoke(my_command)

    assert result.exit_code == 0
    mock_has_stages.assert_not_called()
    mock_discover.assert_not_called()
    assert "Command executed" in result.output


def test_pivot_command_auto_discover_converts_discovery_error(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover converts DiscoveryError to ClickException."""
    mocker.patch.object(discovery, "has_registered_stages", return_value=False)
    mocker.patch.object(
        discovery,
        "discover_and_register",
        side_effect=discovery.DiscoveryError("No pivot.yaml found"),
    )

    @cli_decorators.pivot_command()
    def my_command() -> None:
        click.echo("Should not reach here")

    result = runner.invoke(my_command)

    assert result.exit_code != 0
    assert "No pivot.yaml found" in result.output
    assert "Should not reach here" not in result.output
