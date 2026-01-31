from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def test_repro_runs_all_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot repro runs all stages by default."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("output.txt").exists()


def test_repro_runs_specific_stage(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot repro STAGE runs the specified stage and its dependencies."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["repro", "process"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("output.txt").exists()


# =============================================================================
# repro --dry-run --allow-missing Tests
# =============================================================================


def test_repro_dry_run_allow_missing_uses_pvt_hash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --dry-run --allow-missing uses .pvt hash when dep file is missing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create and run
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")
        executor.run()

        # Track input
        input_hash = cache.hash_file(pathlib.Path("input.txt"))
        pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
        track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

        # Delete input (simulating CI)
        pathlib.Path("input.txt").unlink()

        result = runner.invoke(cli.cli, ["repro", "--dry-run", "--allow-missing"])

        # Should show "would skip" not "Missing deps"
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert "would skip" in result.output.lower(), f"Got: {result.output}"


def test_repro_dry_run_explain_allow_missing_uses_pvt_hash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --dry-run --explain --allow-missing uses .pvt hash when dep file is missing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create and run
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")
        executor.run()

        # Track input
        input_hash = cache.hash_file(pathlib.Path("input.txt"))
        pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
        track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

        # Delete input (simulating CI)
        pathlib.Path("input.txt").unlink()

        result = runner.invoke(cli.cli, ["repro", "--dry-run", "--explain", "--allow-missing"])

        # Should NOT show error about missing deps
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert result.exit_code == 0, f"Expected success, got: {result.output}"


def test_repro_allow_missing_requires_dry_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --allow-missing without --dry-run errors."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["repro", "--allow-missing"])

        assert result.exit_code != 0
        assert "--allow-missing" in result.output
        assert "--dry-run" in result.output


# =============================================================================
# Removed standalone dry-run command
# =============================================================================


def test_dry_run_standalone_removed(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot dry-run standalone command no longer exists (use repro -n)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["dry-run"])

        assert result.exit_code != 0
        assert "No such command" in result.output
