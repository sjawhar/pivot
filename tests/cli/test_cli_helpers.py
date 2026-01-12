from __future__ import annotations

import json

import click
import click.testing
import pytest

from pivot import exceptions, registry
from pivot.cli import CliContext
from pivot.cli import helpers as cli_helpers

# =============================================================================
# validate_stages_exist Tests
# =============================================================================


def test_validate_stages_exist_with_none() -> None:
    """validate_stages_exist returns early for None input."""
    cli_helpers.validate_stages_exist(None)


def test_validate_stages_exist_with_empty_list() -> None:
    """validate_stages_exist returns early for empty list."""
    cli_helpers.validate_stages_exist([])


def test_validate_stages_exist_with_valid_stages() -> None:
    """validate_stages_exist passes for registered stages."""
    registry.REGISTRY.register(lambda: None, name="stage_a", deps=[], outs=["a.txt"])
    registry.REGISTRY.register(lambda: None, name="stage_b", deps=[], outs=["b.txt"])

    cli_helpers.validate_stages_exist(["stage_a", "stage_b"])


def test_validate_stages_exist_raises_for_unknown_stage() -> None:
    """validate_stages_exist raises StageNotFoundError for unknown stages."""
    registry.REGISTRY.register(lambda: None, name="known_stage", deps=[], outs=["out.txt"])

    with pytest.raises(exceptions.StageNotFoundError) as exc_info:
        cli_helpers.validate_stages_exist(["known_stage", "unknown_stage"])

    assert "unknown_stage" in str(exc_info.value)


def test_validate_stages_exist_raises_for_multiple_unknown() -> None:
    """validate_stages_exist includes all unknown stages in error."""
    registry.REGISTRY.register(lambda: None, name="valid", deps=[], outs=["out.txt"])

    with pytest.raises(exceptions.StageNotFoundError) as exc_info:
        cli_helpers.validate_stages_exist(["invalid1", "invalid2"])

    error_msg = str(exc_info.value)
    assert "invalid1" in error_msg
    assert "invalid2" in error_msg


# =============================================================================
# make_progress_callback Tests
# =============================================================================


def test_make_progress_callback_returns_callable() -> None:
    """make_progress_callback returns a callable."""
    callback = cli_helpers.make_progress_callback("Uploaded")
    assert callable(callback)


def test_make_progress_callback_echoes_progress(
    runner: click.testing.CliRunner,
) -> None:
    """make_progress_callback creates callback that echoes progress."""

    @click.command()
    def test_cmd() -> None:
        callback = cli_helpers.make_progress_callback("Downloaded")
        callback(5)
        callback(10)

    result = runner.invoke(test_cmd)

    assert result.exit_code == 0
    assert "Downloaded 5 files" in result.output
    assert "Downloaded 10 files" in result.output


def test_make_progress_callback_uses_action_text(
    runner: click.testing.CliRunner,
) -> None:
    """make_progress_callback uses provided action text."""

    @click.command()
    def test_cmd() -> None:
        callback = cli_helpers.make_progress_callback("Processed")
        callback(42)

    result = runner.invoke(test_cmd)

    assert "Processed 42 files" in result.output


# =============================================================================
# print_transfer_errors Tests
# =============================================================================


def test_print_transfer_errors_with_no_errors(
    runner: click.testing.CliRunner,
) -> None:
    """print_transfer_errors does nothing with empty list."""

    @click.command()
    def test_cmd() -> None:
        cli_helpers.print_transfer_errors([])
        click.echo("done")

    result = runner.invoke(test_cmd)

    assert result.exit_code == 0
    assert "Error:" not in result.output
    assert "done" in result.output


def test_print_transfer_errors_shows_all_when_few(
    runner: click.testing.CliRunner,
) -> None:
    """print_transfer_errors shows all errors when under max_shown."""

    @click.command()
    def test_cmd() -> None:
        errors = ["error1", "error2", "error3"]
        cli_helpers.print_transfer_errors(errors)

    result = runner.invoke(test_cmd)

    assert "Error: error1" in result.output
    assert "Error: error2" in result.output
    assert "Error: error3" in result.output
    assert "more errors" not in result.output


def test_print_transfer_errors_truncates_when_many(
    runner: click.testing.CliRunner,
) -> None:
    """print_transfer_errors truncates when exceeding max_shown."""

    @click.command()
    def test_cmd() -> None:
        errors = ["err1", "err2", "err3", "err4", "err5", "err6", "err7"]
        cli_helpers.print_transfer_errors(errors, max_shown=3)

    result = runner.invoke(test_cmd)

    assert "Error: err1" in result.output
    assert "Error: err2" in result.output
    assert "Error: err3" in result.output
    assert "Error: err4" not in result.output
    assert "4 more errors" in result.output


def test_print_transfer_errors_uses_default_max_shown(
    runner: click.testing.CliRunner,
) -> None:
    """print_transfer_errors defaults to showing 5 errors."""

    @click.command()
    def test_cmd() -> None:
        errors = [f"error{i}" for i in range(8)]
        cli_helpers.print_transfer_errors(errors)

    result = runner.invoke(test_cmd)

    assert "Error: error0" in result.output
    assert "Error: error4" in result.output
    assert "Error: error5" not in result.output
    assert "3 more errors" in result.output


def test_print_transfer_errors_outputs_to_stderr(
    runner: click.testing.CliRunner,
) -> None:
    """print_transfer_errors outputs to stderr."""

    @click.command()
    def test_cmd() -> None:
        cli_helpers.print_transfer_errors(["test error"])

    result = runner.invoke(test_cmd, catch_exceptions=False)

    assert result.exit_code == 0


# =============================================================================
# emit_jsonl Tests
# =============================================================================


def test_emit_jsonl_serializes_dict(
    runner: click.testing.CliRunner, capsys: pytest.CaptureFixture[str]
) -> None:
    """emit_jsonl serializes dict as JSON line."""

    @click.command()
    def test_cmd() -> None:
        cli_helpers.emit_jsonl({"type": "test", "value": 42})

    result = runner.invoke(test_cmd)

    assert result.exit_code == 0
    output = result.output.strip()
    parsed = json.loads(output)
    assert parsed == {"type": "test", "value": 42}


def test_emit_jsonl_flushes_output(runner: click.testing.CliRunner) -> None:
    """emit_jsonl flushes output for streaming."""

    @click.command()
    def test_cmd() -> None:
        cli_helpers.emit_jsonl({"event": "start"})
        cli_helpers.emit_jsonl({"event": "end"})

    result = runner.invoke(test_cmd)

    lines = result.output.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"event": "start"}
    assert json.loads(lines[1]) == {"event": "end"}


def test_emit_jsonl_handles_nested_structures(runner: click.testing.CliRunner) -> None:
    """emit_jsonl handles nested dicts and lists."""

    @click.command()
    def test_cmd() -> None:
        cli_helpers.emit_jsonl({"items": [1, 2, 3], "nested": {"a": "b"}})

    result = runner.invoke(test_cmd)

    parsed = json.loads(result.output.strip())
    assert parsed == {"items": [1, 2, 3], "nested": {"a": "b"}}


# =============================================================================
# get_cli_context Tests
# =============================================================================


def test_get_cli_context_returns_existing_context(runner: click.testing.CliRunner) -> None:
    """get_cli_context returns existing context when set."""
    ctx_obj = CliContext(verbose=True, quiet=False)

    @click.command()
    @click.pass_context
    def test_cmd(ctx: click.Context) -> None:
        ctx.obj = ctx_obj
        result = cli_helpers.get_cli_context(ctx)
        assert result["verbose"] is True
        assert result["quiet"] is False

    result = runner.invoke(test_cmd)
    assert result.exit_code == 0


def test_get_cli_context_returns_defaults_when_none(runner: click.testing.CliRunner) -> None:
    """get_cli_context returns defaults when ctx.obj is None."""

    @click.command()
    @click.pass_context
    def test_cmd(ctx: click.Context) -> None:
        ctx.obj = None
        result = cli_helpers.get_cli_context(ctx)
        assert result["verbose"] is False
        assert result["quiet"] is False

    result = runner.invoke(test_cmd)
    assert result.exit_code == 0


def test_get_cli_context_returns_defaults_when_not_set(runner: click.testing.CliRunner) -> None:
    """get_cli_context returns defaults when ctx.obj was never set."""

    @click.command()
    @click.pass_context
    def test_cmd(ctx: click.Context) -> None:
        # Don't set ctx.obj at all
        result = cli_helpers.get_cli_context(ctx)
        assert result["verbose"] is False
        assert result["quiet"] is False

    result = runner.invoke(test_cmd)
    assert result.exit_code == 0


# =============================================================================
# stages_to_list Tests
# =============================================================================


def test_stages_to_list_converts_tuple_to_list() -> None:
    """stages_to_list converts non-empty tuple to list."""
    result = cli_helpers.stages_to_list(("stage1", "stage2", "stage3"))
    assert result == ["stage1", "stage2", "stage3"]


def test_stages_to_list_returns_none_for_empty_tuple() -> None:
    """stages_to_list returns None for empty tuple."""
    result = cli_helpers.stages_to_list(())
    assert result is None


def test_stages_to_list_preserves_order() -> None:
    """stages_to_list preserves stage order."""
    result = cli_helpers.stages_to_list(("c", "a", "b"))
    assert result == ["c", "a", "b"]


def test_stages_to_list_single_stage() -> None:
    """stages_to_list handles single-element tuple."""
    result = cli_helpers.stages_to_list(("only_stage",))
    assert result == ["only_stage"]
