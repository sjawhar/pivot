"""Tests for --explain CLI flag."""

import pathlib

import click.testing
import pydantic
import pytest

from pivot import cli, executor, stage


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Basic --explain flag tests
# =============================================================================


def test_explain_flag_in_help(runner: click.testing.CliRunner) -> None:
    """--explain flag should appear in help output."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--explain" in result.output


def test_explain_no_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain with no stages shows appropriate message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "No stages" in result.output


def test_explain_flag_works(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain produces output for stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "WILL RUN" in result.output


def test_explain_specific_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain can target specific stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["a.txt"])
        def stage_a() -> None:
            pass

        @stage(deps=["input.txt"], outs=["b.txt"])
        def stage_b() -> None:
            pass

        result = runner.invoke(cli.cli, ["run", "--explain", "stage_a"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


# =============================================================================
# Change type display tests
# =============================================================================


def test_explain_shows_code_changes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain shows code changes when code differs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("v1")

        executor.run(show_output=False)

        # Re-register with different implementation (simulates code change)
        from pivot.registry import REGISTRY

        REGISTRY._stages.clear()

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("v2 - different code")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "WILL RUN" in result.output
        assert "Code" in result.output


def test_explain_shows_param_changes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain shows param changes when params differ."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        class TrainParams(pydantic.BaseModel):
            learning_rate: float = 0.01

        @stage(deps=["input.txt"], outs=["output.txt"], params=TrainParams)
        def train(params: TrainParams) -> None:
            pathlib.Path("output.txt").write_text("done")

        executor.run(show_output=False)

        # Change params via params.yaml
        pathlib.Path("params.yaml").write_text("train:\n  learning_rate: 0.001\n")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "WILL RUN" in result.output
        assert "Param" in result.output


def test_explain_shows_dep_changes(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows dependency changes when deps differ."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("original data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("done")

        executor.run(show_output=False)

        # Modify the input file
        pathlib.Path("input.txt").write_text("modified data")

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "WILL RUN" in result.output
        assert "Dep" in result.output or "input" in result.output.lower()


def test_explain_shows_unchanged(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--explain shows stages as unchanged when nothing differs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("done")

        executor.run(show_output=False)

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "unchanged" in result.output.lower() or "skip" in result.output.lower()


def test_explain_shows_no_previous_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain shows 'No previous run' for never-run stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

        result = runner.invoke(cli.cli, ["run", "--explain"])

        assert result.exit_code == 0
        assert "No previous run" in result.output


# =============================================================================
# Short flag tests
# =============================================================================


def test_explain_short_flag(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """-e short flag works like --explain."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

        result = runner.invoke(cli.cli, ["run", "-e"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "WILL RUN" in result.output


# =============================================================================
# Error handling tests
# =============================================================================


def test_explain_unknown_stage_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain with unknown stage shows error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--explain", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "unknown" in result.output.lower()
