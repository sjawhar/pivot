"""Tests for --explain CLI flag (repro command)."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli

if TYPE_CHECKING:
    from click.testing import CliRunner


def _write_input(content: str = "data") -> None:
    """Write the pipeline's raw input at the path ``pipeline.input("input.txt")`` reads."""
    raw_dir = pathlib.Path("data", "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text(content)


def _write_pipeline(content: str) -> None:
    pathlib.Path("pipeline.py").write_text(content)


# Common pipeline definitions
_PROCESS_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return "done"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""

_PROCESS_PIPELINE_V1 = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return "v1"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""

_PROCESS_PIPELINE_V2 = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return "v2 - different code"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""

_STAGES_A_B_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def stage_a(input_file: str) -> str:
    return "a"


@stage
def stage_b(input_file: str) -> str:
    return "b"


with Pipeline("test") as pipeline:
    raw = pipeline.input("input.txt")
    stage_a(raw)
    stage_b(raw)
"""

_TRAIN_PIPELINE_TEMPLATE = """\
from __future__ import annotations

from pivot import StageParams
from pivot.compose import Pipeline, stage


class TrainParams(StageParams):
    learning_rate: float = 0.01


@stage
def train(input_file: str, params: TrainParams) -> str:
    return "done"


with Pipeline("test") as pipeline:
    train(pipeline.input("input.txt"), params=TrainParams(learning_rate=LEARNING_RATE))
"""


# =============================================================================
# Basic --explain flag tests (repro command)
# =============================================================================


def test_explain_flag_in_help(runner: CliRunner) -> None:
    """--explain flag should appear in repro help output."""
    result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--explain" in result.output


def test_explain_no_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain with no pipeline errors with appropriate message."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code != 0
        # Either "No pipeline found" or "No Pipeline in context" depending on discovery path
        assert "pipeline" in result.output.lower()


def test_explain_flag_works(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain produces output for stages."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_PROCESS_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0, f"Failed with: {result.output}"
        assert "process" in result.output
        assert "WILL RUN" in result.output


def test_explain_specific_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain can target specific stages."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_STAGES_A_B_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--explain", "stage_a"])

        assert result.exit_code == 0, result.output
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


# =============================================================================
# Change type display tests
# =============================================================================


def test_explain_shows_code_changes(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain shows code changes when code differs."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_PROCESS_PIPELINE_V1)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, f"First run failed: {first.output}"

        _write_pipeline(_PROCESS_PIPELINE_V2)
        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0, result.output
        assert "WILL RUN" in result.output
        assert "Code Changes:" in result.output, "Should list the changed stage code"


def test_explain_shows_param_changes(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain shows param changes when params differ."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_TRAIN_PIPELINE_TEMPLATE.replace("LEARNING_RATE", "0.01"))
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, f"First run failed: {first.output}"

        _write_pipeline(_TRAIN_PIPELINE_TEMPLATE.replace("LEARNING_RATE", "0.02"))
        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0, result.output
        assert "WILL RUN" in result.output
        assert "Param Changes:" in result.output, "Should list the changed param"
        assert "learning_rate" in result.output
        assert "0.02" in result.output


def test_explain_shows_dep_changes(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain shows dependency changes when deps differ."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input("original")
        _write_pipeline(_PROCESS_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, f"First run failed: {first.output}"

        _write_input("modified")
        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0, result.output
        assert "WILL RUN" in result.output
        assert "Dependency Changes:" in result.output, "Should list the changed input"
        assert "input.txt" in result.output


def test_explain_shows_unchanged(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain shows stages as unchanged when nothing differs."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_PROCESS_PIPELINE)
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, f"First run failed: {first.output}"

        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0, result.output
        assert "SKIP" in result.output
        assert "WILL RUN" not in result.output
        assert "0 will run, 1 unchanged" in result.output


def test_explain_shows_no_previous_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain shows 'No previous run' for never-run stages."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_PROCESS_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0, result.output
        assert "No previous run" in result.output


# =============================================================================
# Short flag tests
# =============================================================================


def test_explain_short_flag(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro -e short flag works like --explain."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_PROCESS_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "-e"])

        assert result.exit_code == 0, result.output
        assert "process" in result.output
        assert "WILL RUN" in result.output


# =============================================================================
# Error handling tests
# =============================================================================


def test_explain_unknown_stage_errors(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --explain with unknown stage shows error."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_pipeline(_PROCESS_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--explain", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "unknown" in result.output.lower()
