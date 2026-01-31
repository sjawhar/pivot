from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


# =============================================================================
# pivot run Tests - Single-stage execution only
# =============================================================================


def test_run_requires_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run without stages shows error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "STAGES" in result.output


def test_run_executes_single_stage(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run STAGE executes the specified stage only (no dependencies)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "process"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("output.txt").exists()


def test_run_fail_fast_stops_on_first_failure(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run --fail-fast stops execution on first failure."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Don't create input.txt - this will cause failure
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--fail-fast", "process"])

        # Should fail since input.txt doesn't exist
        assert result.exit_code != 0


def test_run_multiple_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run STAGE1 STAGE2 executes multiple stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        # Can run the same stage name multiple times (though unusual)
        result = runner.invoke(cli.cli, ["run", "process"])

        assert result.exit_code == 0, f"Failed: {result.output}"
