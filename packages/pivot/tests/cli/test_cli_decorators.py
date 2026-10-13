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


def test_pivot_command_auto_discover_calls_discovery(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover=True calls discover_pipeline."""
    mock_discover_pipeline = mocker.patch.object(discovery, "discover_pipeline", return_value=None)

    @cli_decorators.pivot_command()
    def my_command() -> None:
        click.echo("Command executed")

    result = runner.invoke(my_command)

    assert result.exit_code == 0
    mock_discover_pipeline.assert_called_once()
    assert "Command executed" in result.output


def test_pivot_command_auto_discover_false_skips_discovery(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover=False skips discovery entirely."""
    mock_discover = mocker.patch.object(discovery, "discover_pipeline")

    @cli_decorators.pivot_command(auto_discover=False)
    def my_command() -> None:
        click.echo("Command executed")

    result = runner.invoke(my_command)

    assert result.exit_code == 0
    mock_discover.assert_not_called()
    assert "Command executed" in result.output


def test_pivot_command_auto_discover_converts_discovery_error(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """auto_discover converts DiscoveryError to ClickException."""
    mocker.patch.object(
        discovery,
        "discover_pipeline",
        side_effect=discovery.DiscoveryError("No pivot.yaml found"),
    )

    @cli_decorators.pivot_command()
    def my_command() -> None:
        click.echo("Should not reach here")

    result = runner.invoke(my_command)

    assert result.exit_code != 0
    assert "No pivot.yaml found" in result.output
    assert "Should not reach here" not in result.output


# =============================================================================
# allow_all (--all flag) Tests
# =============================================================================


def test_pivot_command_all_flag_when_allowed(runner: CliRunner, mocker: MockerFixture) -> None:
    """Commands with allow_all=True accept --all flag."""
    mocker.patch.object(discovery, "discover_pipeline", return_value=None)

    @cli_decorators.pivot_command(allow_all=True)
    def test_cmd() -> None:
        click.echo("ok")

    result = runner.invoke(test_cmd, ["--all"])

    assert result.exit_code == 0
    assert "ok" in result.output


def test_pivot_command_all_flag_when_not_allowed(runner: CliRunner) -> None:
    """Commands without allow_all=True do not accept --all flag."""

    @cli_decorators.pivot_command(auto_discover=False)
    def test_cmd() -> None:
        click.echo("ok")

    result = runner.invoke(test_cmd, ["--all"])

    assert result.exit_code != 0


def test_pivot_command_all_flag_passes_to_discovery(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """--all flag causes discover_pipeline(all_pipelines=True) to be called."""
    mock_discover = mocker.patch.object(discovery, "discover_pipeline", return_value=None)

    @cli_decorators.pivot_command(allow_all=True)
    def test_cmd() -> None:
        click.echo("ok")

    runner.invoke(test_cmd, ["--all"])

    mock_discover.assert_called_once_with(all_pipelines=True)


def test_pivot_command_without_all_flag_normal_discovery(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """Without --all flag, discover_pipeline is called without all_pipelines."""
    mock_discover = mocker.patch.object(discovery, "discover_pipeline", return_value=None)

    @cli_decorators.pivot_command(allow_all=True)
    def test_cmd() -> None:
        click.echo("ok")

    runner.invoke(test_cmd)

    mock_discover.assert_called_once_with(all_pipelines=False)


def test_pivot_command_all_pipelines_kwarg_not_leaked_to_command(
    runner: CliRunner, mocker: MockerFixture
) -> None:
    """--all flag is consumed by the decorator and not passed to the command function.

    The decorator pops all_pipelines from kwargs before calling the wrapped function.
    If this pop is removed or broken, the command function receives an unexpected
    kwarg and raises TypeError.
    """
    mocker.patch.object(discovery, "discover_pipeline", return_value=None)
    received_kwargs = dict[str, object]()

    @cli_decorators.pivot_command(allow_all=True)
    def test_cmd(**kwargs: object) -> None:
        received_kwargs.update(kwargs)
        click.echo("ok")

    result = runner.invoke(test_cmd, ["--all"])

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "all_pipelines" not in received_kwargs, "all_pipelines kwarg leaked to command function"
