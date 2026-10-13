"""Every stage-taking command resolves a bare stage name in a single-pipeline project (names.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import StageParams, cli, compose, executor

if TYPE_CHECKING:
    import pathlib

    import click.testing


class _HelperParams(StageParams):
    factor: int = 3


@compose.stage
def _helper_upper(text: str) -> str:
    return text.upper()


@compose.stage
def _helper_repeat(text: str, params: _HelperParams) -> str:
    return text * params.factor


def _register(pipeline: compose.Pipeline) -> None:
    raw = pipeline.root / "data" / "raw" / "words.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("ab")
    with pipeline:
        words = pipeline.input("words.txt", t=str)
        _helper_upper(words)
        _helper_repeat(words, params=_HelperParams())


def _output(pipeline: compose.Pipeline, stage: str) -> pathlib.Path:
    return pipeline.root / "data" / "test" / f"{stage}.txt"


def test_run_resolves_bare_stage_name(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "_helper_upper"])

    assert result.exit_code == 0, result.output
    assert _output(mock_discovery, "_helper_upper").read_text() == "AB"
    assert not _output(mock_discovery, "_helper_repeat").exists()


def test_status_resolves_bare_stage_name(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)

    result = runner.invoke(cli.cli, ["status", "--stages-only", "--verbose", "_helper_upper"])

    assert result.exit_code == 0, result.output
    assert "test/_helper_upper" in result.output
    assert "test/_helper_repeat" not in result.output


def test_verify_resolves_bare_stage_name(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "_helper_upper"])

    assert result.exit_code == 0, result.output
    assert "test/_helper_upper" in result.output
    assert "test/_helper_repeat" not in result.output


def test_commit_resolves_bare_stage_name(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)
    executor.run(pipeline=mock_discovery, no_commit=True)

    result = runner.invoke(cli.cli, ["commit", "_helper_upper"])

    assert result.exit_code == 0, result.output
    assert "Committed 1 stage(s):\n  test/_helper_upper" in result.output


def test_params_show_resolves_bare_stage_name(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)

    result = runner.invoke(cli.cli, ["params", "show", "_helper_repeat"])

    assert result.exit_code == 0, result.output
    assert "test/_helper_repeat" in result.output
    assert "factor" in result.output


def test_checkout_resolves_bare_stage_name(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)
    executor.run(pipeline=mock_discovery)
    output = _output(mock_discovery, "_helper_upper")
    output.unlink()

    result = runner.invoke(cli.cli, ["checkout", "_helper_upper"])

    assert result.exit_code == 0, result.output
    assert output.read_text() == "AB"


def test_unknown_bare_stage_name_still_errors(
    mock_discovery: compose.Pipeline, runner: click.testing.CliRunner
) -> None:
    _register(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "no_such_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): no_such_stage" in result.output
