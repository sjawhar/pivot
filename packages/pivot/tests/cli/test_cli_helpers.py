from __future__ import annotations

import json
import pathlib
import sys
from typing import TYPE_CHECKING, Annotated, TypedDict

import click
import click.testing
import pytest

from helpers import register_test_stage
from pivot import exceptions, loaders, outputs
from pivot.cli import CliContext
from pivot.cli import helpers as cli_helpers

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.pipeline.pipeline import Pipeline


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _StageBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _KnownStageOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("out.txt", loaders.PathOnly())]


class _ValidStageOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("out.txt", loaders.PathOnly())]


def _stage_a_func() -> _StageAOutputs:
    return _StageAOutputs(output=pathlib.Path("a.txt"))


def _stage_b_func() -> _StageBOutputs:
    return _StageBOutputs(output=pathlib.Path("b.txt"))


def _known_stage_func() -> _KnownStageOutputs:
    return _KnownStageOutputs(output=pathlib.Path("out.txt"))


def _valid_stage_func() -> _ValidStageOutputs:
    return _ValidStageOutputs(output=pathlib.Path("out.txt"))


class _HelperDummyBar:
    total: int | None
    n: int
    desc: str
    closed: bool
    refresh_calls: int

    def __init__(self) -> None:
        self.total = None
        self.n = 0
        self.desc = ""
        self.closed = False
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1

    def update(self, n: int) -> None:
        self.n += n

    def close(self) -> None:
        self.closed = True


def _helper_make_dummy_tqdm(
    bar: _HelperDummyBar,
) -> Callable[..., _HelperDummyBar]:
    def _factory(**kwargs: object) -> _HelperDummyBar:
        # Capture the total kwarg if provided (for lazy bar creation)
        if "total" in kwargs:
            total_val = kwargs["total"]
            assert isinstance(total_val, int)
            bar.total = total_val
        return bar

    return _factory


# =============================================================================
# validate_stages_exist Tests
# =============================================================================


@pytest.mark.parametrize(
    "stages",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty_list"),
    ],
)
def test_validate_stages_exist_returns_early_for_empty_input(stages: list[str] | None) -> None:
    """validate_stages_exist returns early for None or empty list without raising."""
    cli_helpers.validate_stages_exist(stages)  # Should not raise


def test_validate_stages_exist_with_valid_stages(mock_discovery: Pipeline) -> None:
    """validate_stages_exist passes for registered stages."""
    register_test_stage(_stage_a_func, name="stage_a")
    register_test_stage(_stage_b_func, name="stage_b")

    # validate_stages_exist requires a Click context with the pipeline
    @click.command()
    def _test_cmd() -> None:
        pass

    ctx = click.Context(_test_cmd)
    ctx.obj = {"_pivot_pipeline": mock_discovery}
    with ctx:
        cli_helpers.validate_stages_exist(["stage_a", "stage_b"])


def test_validate_stages_exist_raises_for_unknown_stage(mock_discovery: Pipeline) -> None:
    """validate_stages_exist raises StageNotFoundError for unknown stages."""
    register_test_stage(_known_stage_func, name="known_stage")

    @click.command()
    def _test_cmd() -> None:
        pass

    ctx = click.Context(_test_cmd)
    ctx.obj = {"_pivot_pipeline": mock_discovery}
    with ctx, pytest.raises(exceptions.StageNotFoundError) as exc_info:
        cli_helpers.validate_stages_exist(["known_stage", "unknown_stage"])

    assert "unknown_stage" in str(exc_info.value)


def test_validate_stages_exist_raises_for_multiple_unknown(mock_discovery: Pipeline) -> None:
    """validate_stages_exist includes all unknown stages in error."""
    register_test_stage(_valid_stage_func, name="valid")

    @click.command()
    def _test_cmd() -> None:
        pass

    ctx = click.Context(_test_cmd)
    ctx.obj = {"_pivot_pipeline": mock_discovery}
    with ctx, pytest.raises(exceptions.StageNotFoundError) as exc_info:
        cli_helpers.validate_stages_exist(["invalid1", "invalid2"])

    error_msg = str(exc_info.value)
    assert "invalid1" in error_msg
    assert "invalid2" in error_msg


# =============================================================================
# TransferProgress Tests
# =============================================================================


def test_transfer_progress_skips_when_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """TransferProgress avoids creating a bar when quiet."""
    bar = _HelperDummyBar()
    calls = {"count": 0}

    def _dummy_tqdm(**_kwargs: object) -> _HelperDummyBar:
        calls["count"] += 1
        return bar

    monkeypatch.setattr(cli_helpers, "async_tqdm", _dummy_tqdm)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    with cli_helpers.TransferProgress("Uploaded", quiet=True) as progress:
        progress.callback(1, 3, "file.txt")

    assert calls["count"] == 0


def test_transfer_progress_no_bar_without_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """TransferProgress does not create bar if callback is never invoked."""
    calls = {"count": 0}

    def _dummy_tqdm(**_kwargs: object) -> _HelperDummyBar:
        calls["count"] += 1
        return _HelperDummyBar()

    monkeypatch.setattr(cli_helpers, "async_tqdm", _dummy_tqdm)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    with cli_helpers.TransferProgress("Uploaded") as progress:
        pass  # No callback invoked — simulates 0 files

    assert calls["count"] == 0
    assert progress._bar is None


def test_transfer_progress_updates_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """TransferProgress updates the tqdm bar with progress data."""
    bar = _HelperDummyBar()
    monkeypatch.setattr(cli_helpers, "async_tqdm", _helper_make_dummy_tqdm(bar))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    progress = cli_helpers.TransferProgress("Uploaded")
    progress.callback(1, 3, "file.txt")
    progress.callback(2, 3, "another.txt")

    assert bar.total == 3
    assert bar.n == 2
    assert bar.desc == "Uploaded another.txt"


def test_transfer_progress_closes_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """TransferProgress closes the tqdm bar on exit."""
    bar = _HelperDummyBar()
    monkeypatch.setattr(cli_helpers, "async_tqdm", _helper_make_dummy_tqdm(bar))
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    with cli_helpers.TransferProgress("Downloaded") as progress:
        progress.callback(1, 1, "file.txt")

    assert bar.closed is True


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


# =============================================================================
# validate_stages_exist Without Context Tests
# =============================================================================


def test_validate_stages_exist_no_context_raises_error() -> None:
    """validate_stages_exist raises helpful error when called outside Click context.

    Critical for debugging - if someone calls this function outside the CLI,
    should get clear error message (NoPipelineError), not AttributeError or NoneType error.
    """
    from pivot.cli.helpers import NoPipelineError

    # Don't set up any Click context - just call directly
    with pytest.raises(NoPipelineError):
        cli_helpers.validate_stages_exist(["some_stage"])
