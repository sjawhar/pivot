"""Tests for --explain CLI flag."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs, stage_def
from pivot.registry import REGISTRY

if TYPE_CHECKING:
    from click.testing import CliRunner


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _ATxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def _helper_stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxtOutputs:
    _ = input_file
    return _ATxtOutputs(output=pathlib.Path("a.txt"))


def _helper_stage_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxtOutputs:
    _ = input_file
    return _BTxtOutputs(output=pathlib.Path("b.txt"))


def _helper_process_v1(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("v1")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def _helper_process_v2(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("v2 - different code")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def _helper_process_writer(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


class TrainParams(stage_def.StageParams):
    learning_rate: float = 0.01


def _helper_train(
    params: TrainParams,
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    _ = params
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


# =============================================================================
# Basic --explain flag tests
# =============================================================================


def test_explain_flag_in_help(runner: CliRunner) -> None:
    """--explain flag should appear in help output."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--explain" in result.output


def test_explain_no_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain with no stages shows appropriate message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "No stages" in result.output


def test_explain_flag_works(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain produces output for stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "WILL RUN" in result.output


def test_explain_specific_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain can target specific stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")

        result = runner.invoke(cli.cli, ["run", "--explain", "stage_a"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


# =============================================================================
# Change type display tests
# =============================================================================


def test_explain_shows_code_changes(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows code changes when code differs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process_v1, name="process")

        executor.run()

        # Re-register with different implementation (simulates code change)
        REGISTRY._stages.clear()

        register_test_stage(_helper_process_v2, name="process")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "WILL RUN" in result.output
        assert "Code" in result.output


def test_explain_shows_param_changes(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows param changes when params differ."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_train, name="train", params=TrainParams)

        executor.run()

        # Change params via params.yaml
        pathlib.Path("params.yaml").write_text("train:\n  learning_rate: 0.001\n")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "WILL RUN" in result.output
        assert "Param" in result.output


def test_explain_shows_dep_changes(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows dependency changes when deps differ."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("original data")

        register_test_stage(_helper_process_writer, name="process")

        executor.run()

        # Modify the input file
        pathlib.Path("input.txt").write_text("modified data")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "WILL RUN" in result.output
        assert "Dep" in result.output or "input" in result.output.lower()


def test_explain_shows_unchanged(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows stages as unchanged when nothing differs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process_writer, name="process")

        executor.run()

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "unchanged" in result.output.lower() or "skip" in result.output.lower()


def test_explain_shows_no_previous_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows 'No previous run' for never-run stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "No previous run" in result.output


# =============================================================================
# Short flag tests
# =============================================================================


def test_explain_short_flag(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """-e short flag works like --explain."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "-e"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "WILL RUN" in result.output


# =============================================================================
# Error handling tests
# =============================================================================


def test_explain_unknown_stage_errors(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain with unknown stage shows error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--explain", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "unknown" in result.output.lower()
