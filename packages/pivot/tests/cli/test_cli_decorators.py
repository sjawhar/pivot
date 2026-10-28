from __future__ import annotations

import errno
import logging
import os
from typing import TYPE_CHECKING

import click
import pytest

from pivot import discovery, exceptions
from pivot.cli import decorators as cli_decorators

if TYPE_CHECKING:
    import pathlib

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
    """pivot_command words an unexpected exception by its type and message, not its repr."""

    @cli_decorators.pivot_command()
    def failing_command() -> None:
        raise ValueError("something went wrong")

    result = runner.invoke(failing_command)

    assert result.exit_code == 1
    assert result.output == "Error: Unexpected ValueError: something went wrong\n"


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


def test_with_error_handling_names_a_messageless_exception_by_type(
    runner: CliRunner,
) -> None:
    """An exception with no message is named by its type, so the error is never empty."""

    @click.command()
    @cli_decorators.with_error_handling
    def failing_command() -> None:
        raise RuntimeError()

    result = runner.invoke(failing_command)

    assert result.exit_code == 1
    assert result.output == "Error: Unexpected RuntimeError\n"


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


def _helper_raising(error: BaseException) -> click.Command:
    """A command under with_error_handling that raises ``error``."""

    @click.command()
    @cli_decorators.with_error_handling
    def failing_command() -> None:
        raise error

    return failing_command


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            PermissionError(errno.EACCES, os.strerror(errno.EACCES), "/p/.pivot/cache/files/c9"),
            f"{os.strerror(errno.EACCES)}: /p/.pivot/cache/files/c9",
            id="one-path",
        ),
        pytest.param(
            OSError(errno.EXDEV, os.strerror(errno.EXDEV), "/p/a b", None, "/q/b"),
            f"{os.strerror(errno.EXDEV)}: /p/a b -> /q/b",
            id="two-paths",
        ),
        pytest.param(
            OSError(errno.EIO, os.strerror(errno.EIO)),
            os.strerror(errno.EIO),
            id="no-path",
        ),
        pytest.param(
            FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "/p/a\nb"),
            f"{os.strerror(errno.ENOENT)}: '/p/a\\nb'",
            id="path-that-would-break-the-line-is-quoted",
        ),
        pytest.param(
            FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), b"/p/\xff"),
            f"{os.strerror(errno.ENOENT)}: '/p/\\udcff'",
            id="undecodable-bytes-path-is-quoted",
        ),
        pytest.param(
            OSError("Filesystem error for stage output 'out'"),
            "Filesystem error for stage output 'out'",
            id="own-message-without-errno",
        ),
    ],
)
def test_with_error_handling_words_an_os_error_as_its_message_and_paths(
    runner: CliRunner, error: OSError, expected: str
) -> None:
    """An OSError prints as one line: the OS message and the path or paths, with no exception
    class, errno or quotes, and the command exits 1."""
    result = runner.invoke(_helper_raising(error))

    assert result.exit_code == 1
    assert result.output == f"Error: {expected}\n"


def test_with_error_handling_words_a_real_two_path_os_error(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """The OS fills in both paths of a failed rename; both print, source first."""
    source, destination = tmp_path / "missing", tmp_path / "dest"

    @click.command()
    @cli_decorators.with_error_handling
    def rename() -> None:
        os.rename(source, destination)

    result = runner.invoke(rename)

    assert result.exit_code == 1
    assert result.output == f"Error: {os.strerror(errno.ENOENT)}: {source} -> {destination}\n"


def test_with_error_handling_logs_the_traceback_of_an_unexpected_exception(
    runner: CliRunner, caplog: pytest.LogCaptureFixture
) -> None:
    """The message is one line; the traceback goes to the debug log, which --verbose shows."""
    error = KeyError("stage")

    with caplog.at_level(logging.DEBUG, logger="pivot.cli.decorators"):
        result = runner.invoke(_helper_raising(error))

    assert result.exit_code == 1
    assert result.output == "Error: Unexpected KeyError: 'stage'\n"
    assert _helper_logged_exceptions(caplog) == [error]


def test_with_error_handling_keeps_a_pivot_error_text(runner: CliRunner) -> None:
    """A PivotError prints its own message and tip, exactly as before."""
    error = exceptions.RemoteNotFoundError("Unknown remote: origin")

    result = runner.invoke(_helper_raising(error))

    assert result.exit_code == 1
    assert result.output == (
        "Error: Unknown remote: origin\n\nTip: Run 'pivot remote list' to see available remotes\n"
    )


def test_with_error_handling_words_each_exception_of_a_group(
    runner: CliRunner, caplog: pytest.LogCaptureFixture
) -> None:
    """An exception group (an async TaskGroup's) prints each exception it holds, nested groups
    included: a PivotError with its tip, an OSError as its message and path, never the group's
    repr. Only a group holding something other than a PivotError logs a traceback."""
    pivot_error = exceptions.RemoteNotFoundError("Unknown remote: origin")
    os_error = PermissionError(errno.EACCES, os.strerror(errno.EACCES), "/p/c9")
    mixed = ExceptionGroup("errors", [pivot_error, ExceptionGroup("inner", [os_error])])

    with caplog.at_level(logging.DEBUG, logger="pivot.cli.decorators"):
        mixed_result = runner.invoke(_helper_raising(mixed))
        logged_for_mixed = _helper_logged_exceptions(caplog)
        caplog.clear()
        pivot_only = runner.invoke(_helper_raising(ExceptionGroup("g", [pivot_error])))

    assert mixed_result.exit_code == 1
    assert mixed_result.output == (
        "Error: Unknown remote: origin\n\nTip: Run 'pivot remote list' to see available remotes\n"
        + f"{os.strerror(errno.EACCES)}: /p/c9\n"
    )
    assert pivot_only.exit_code == 1
    assert pivot_only.output == (
        "Error: Unknown remote: origin\n\nTip: Run 'pivot remote list' to see available remotes\n"
    )
    assert logged_for_mixed == [mixed]
    assert _helper_logged_exceptions(caplog) == [], "a group of PivotErrors only logs nothing"


def test_pivot_command_exits_with_the_ctx_exit_code_and_prints_no_exit_repr(
    runner: CliRunner,
) -> None:
    """ctx.exit(1) after a command reported its own failure exits 1, with no `Error: Exit(1)`."""

    @cli_decorators.pivot_command()
    @click.pass_context
    def exiting_command(ctx: click.Context) -> None:
        click.echo("Failed 1 stage(s)")
        ctx.exit(1)

    result = runner.invoke(exiting_command)

    assert result.exit_code == 1
    assert "Exit(" not in result.output
    assert result.output == "Failed 1 stage(s)\n"


def _helper_logged_exceptions(caplog: pytest.LogCaptureFixture) -> list[BaseException | None]:
    """The exceptions whose tracebacks the CLI error handler logged."""
    return [
        record.exc_info[1]
        for record in caplog.records
        if record.name == "pivot.cli.decorators" and record.exc_info
    ]


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
