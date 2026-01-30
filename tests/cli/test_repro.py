"""Tests for pivot repro CLI command."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _StageBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _StageCOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("c.txt", loaders.PathOnly())]


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_stage_a() -> _StageAOutputs:
    pathlib.Path("a.txt").write_text("a")
    return _StageAOutputs(output=pathlib.Path("a.txt"))


def _helper_stage_b(
    dep: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _StageBOutputs:
    _ = dep
    pathlib.Path("b.txt").write_text("b")
    return _StageBOutputs(output=pathlib.Path("b.txt"))


def _helper_stage_c(
    dep: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
) -> _StageCOutputs:
    _ = dep
    pathlib.Path("c.txt").write_text("c")
    return _StageCOutputs(output=pathlib.Path("c.txt"))


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


# =============================================================================
# Basic Command Tests
# =============================================================================


def test_repro_runs_entire_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro without arguments runs entire pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")
        register_test_stage(_helper_stage_c, name="stage_c")

        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("a.txt").exists()
        assert pathlib.Path("b.txt").exists()
        assert pathlib.Path("c.txt").exists()


def test_repro_runs_stage_with_dependencies(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro STAGE runs stage and its dependencies."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")
        register_test_stage(_helper_stage_c, name="stage_c")

        # Run only stage_b - should also run stage_a
        result = runner.invoke(cli.cli, ["repro", "stage_b"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("a.txt").exists(), "Dependency should have been created"
        assert pathlib.Path("b.txt").exists(), "Target stage should have run"
        assert not pathlib.Path("c.txt").exists(), "Downstream stage should NOT run"


# =============================================================================
# Dry Run Tests
# =============================================================================


def test_repro_dry_run_shows_what_would_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --dry-run shows what would run without executing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        assert result.exit_code == 0
        assert "Would run:" in result.output
        assert "stage_a" in result.output
        assert "stage_b" in result.output
        assert not pathlib.Path("a.txt").exists(), "Should not create files"


def test_repro_dry_run_json_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --dry-run --json outputs JSON format."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert "stage_a" in data["stages"]


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


# =============================================================================
# Explain Mode Tests
# =============================================================================


def test_repro_explain_shows_detailed_breakdown(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --explain shows detailed stage explanations."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")

        result = runner.invoke(cli.cli, ["repro", "--explain"])

        assert result.exit_code == 0
        # Explain mode shows more detailed output than dry-run
        assert "stage_a" in result.output
        assert "stage_b" in result.output


def test_repro_explain_allow_missing_uses_pvt_hash(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --explain --allow-missing uses .pvt hash when dep file is missing."""
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

        result = runner.invoke(cli.cli, ["repro", "--explain", "--allow-missing"])

        # Should NOT show error about missing deps
        assert "Missing deps" not in result.output, f"Got: {result.output}"
        assert result.exit_code == 0, f"Expected success, got: {result.output}"


# =============================================================================
# Option Validation Tests
# =============================================================================


def test_repro_allow_missing_requires_dry_run_or_explain(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --allow-missing without --dry-run or --explain errors."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["repro", "--allow-missing"])

        assert result.exit_code != 0
        assert "--allow-missing" in result.output
        assert "--dry-run" in result.output or "--explain" in result.output


def test_repro_serve_requires_watch(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --serve without --watch errors."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--serve"])

        assert result.exit_code != 0
        assert "--serve" in result.output
        assert "--watch" in result.output


def test_repro_serve_requires_tui(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """repro --serve --watch without --tui errors."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--serve", "--watch"])

        assert result.exit_code != 0
        assert "--serve" in result.output
        assert "--tui" in result.output


def test_repro_debounce_requires_watch(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --debounce without --watch errors."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--debounce", "500"])

        assert result.exit_code != 0
        assert "--debounce" in result.output
        assert "--watch" in result.output


def test_repro_tui_log_cannot_use_with_json(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --tui-log cannot be used with --json."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--tui-log", "test.log", "--json"])

        assert result.exit_code != 0
        assert "--tui-log" in result.output
        assert "--json" in result.output


def test_repro_tui_log_cannot_use_with_dry_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --tui-log cannot be used with --dry-run."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        # Note: --tui is required for --tui-log, so we include it to test the --dry-run validation
        result = runner.invoke(cli.cli, ["repro", "--tui", "--tui-log", "test.log", "--dry-run"])

        assert result.exit_code != 0
        assert "--tui-log" in result.output
        assert "--dry-run" in result.output


def test_repro_unknown_stage_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro with unknown stage name errors with helpful message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "nonexistent_stage"])

        assert result.exit_code != 0
        assert "nonexistent_stage" in result.output


# =============================================================================
# Force Mode Tests
# =============================================================================


def test_repro_force_reruns_cached_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --force re-runs stages even if cached."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        # First run
        result = runner.invoke(cli.cli, ["repro"])
        assert result.exit_code == 0

        # Second run without force - should skip
        result = runner.invoke(cli.cli, ["repro", "--dry-run"])
        assert "would skip" in result.output.lower()

        # Run with force - should run
        result = runner.invoke(cli.cli, ["repro", "--dry-run", "--force"])
        assert "would run" in result.output.lower()


# =============================================================================
# Keep-Going Mode Tests
# =============================================================================


def test_repro_keep_going_option_accepted(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --keep-going option is accepted."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--keep-going"])

        # Command should run successfully (or fail gracefully)
        # Main check is that the option is recognized
        assert "--keep-going" not in result.output or result.exit_code == 0


# =============================================================================
# No Single-Stage Option Tests
# =============================================================================


def test_repro_does_not_have_single_stage_option(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro does not have --single-stage option."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--single-stage"])

        # Should error because option doesn't exist
        assert result.exit_code != 0
        assert "No such option" in result.output or "no such option" in result.output.lower()


# =============================================================================
# Empty Pipeline Tests
# =============================================================================


def test_repro_empty_pipeline_shows_message(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro with no stages shows appropriate message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Don't register any stages
        # Create minimal pivot.yaml with empty stages dict
        pathlib.Path("pivot.yaml").write_text("stages: {}\n")

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        # Should show "no stages" message or succeed with empty output
        assert result.exit_code == 0
        assert "No stages to run" in result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_repro_json_output_streams_jsonl(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --json streams JSONL output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--json"])

        assert result.exit_code == 0
        # Check that output is valid JSONL (multiple JSON lines)
        lines = [ln for ln in result.output.strip().split("\n") if ln]
        assert len(lines) > 0
        for line in lines:
            json.loads(line)  # Should not raise


# =============================================================================
# Cache Options Tests
# =============================================================================


def test_repro_no_commit_option_accepted(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --no-commit option is accepted."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--no-commit"])

        assert result.exit_code == 0


def test_repro_no_cache_option_accepted(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --no-cache option is accepted."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_stage_a, name="stage_a")

        result = runner.invoke(cli.cli, ["repro", "--no-cache"])

        assert result.exit_code == 0
