"""Tests for pivot repro CLI command."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from conftest import isolated_pivot_dir
from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing
    import pytest

    from pivot.pipeline.pipeline import Pipeline

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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro without arguments runs entire pipeline."""
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")
    register_test_stage(_helper_stage_c, name="stage_c")

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").exists()
    assert (tmp_path / "c.txt").exists()


def test_repro_runs_stage_with_dependencies(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro STAGE runs stage and its dependencies."""
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")
    register_test_stage(_helper_stage_c, name="stage_c")

    # Run only stage_b - should also run stage_a
    result = runner.invoke(cli.cli, ["repro", "stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (tmp_path / "a.txt").exists(), "Dependency should have been created"
    assert (tmp_path / "b.txt").exists(), "Target stage should have run"
    assert not (tmp_path / "c.txt").exists(), "Downstream stage should NOT run"


# =============================================================================
# Dry Run Tests
# =============================================================================


def test_repro_dry_run_shows_what_would_run(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --dry-run shows what would run without executing."""
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    result = runner.invoke(cli.cli, ["repro", "--dry-run"])

    assert result.exit_code == 0
    assert "Would run:" in result.output
    assert "stage_a" in result.output
    assert "stage_b" in result.output
    assert not (tmp_path / "a.txt").exists(), "Should not create files"


def test_repro_dry_run_json_output(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --dry-run --json outputs JSON format."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--dry-run", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "stages" in data
    assert "stage_a" in data["stages"]


def test_repro_dry_run_allow_missing_uses_pvt_hash(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --dry-run --allow-missing uses .pvt hash when dep file is missing."""

    # Create and run
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")
    executor.run()

    # Track input
    input_hash = cache.hash_file(tmp_path / "input.txt")
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(tmp_path / "input.txt.pvt", pvt_data)

    # Delete input (simulating CI)
    (tmp_path / "input.txt").unlink()

    result = runner.invoke(cli.cli, ["repro", "--dry-run", "--allow-missing"])

    # Should show "would skip" not "Missing deps"
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert "would skip" in result.output.lower(), f"Got: {result.output}"


# =============================================================================
# Explain Mode Tests
# =============================================================================


def test_repro_explain_shows_detailed_breakdown(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --explain shows detailed stage explanations."""
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    result = runner.invoke(cli.cli, ["repro", "--explain"])

    assert result.exit_code == 0
    # Explain mode shows more detailed output than dry-run
    assert "stage_a" in result.output
    assert "stage_b" in result.output


def test_repro_explain_allow_missing_uses_pvt_hash(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --explain --allow-missing uses .pvt hash when dep file is missing."""

    # Create and run
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")
    executor.run()

    # Track input
    input_hash = cache.hash_file(tmp_path / "input.txt")
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(tmp_path / "input.txt.pvt", pvt_data)

    # Delete input (simulating CI)
    (tmp_path / "input.txt").unlink()

    result = runner.invoke(cli.cli, ["repro", "--explain", "--allow-missing"])

    # Should NOT show error about missing deps
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected success, got: {result.output}"


# =============================================================================
# Option Validation Tests
# =============================================================================


def test_repro_allow_missing_requires_dry_run_or_explain(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --allow-missing without --dry-run or --explain errors."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["repro", "--allow-missing"])

    assert result.exit_code != 0
    assert "--allow-missing" in result.output
    assert "--dry-run" in result.output or "--explain" in result.output


def test_repro_serve_requires_watch(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --serve without --watch errors."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--serve"])

    assert result.exit_code != 0
    assert "--serve" in result.output
    assert "--watch" in result.output


def test_repro_serve_with_watch_uses_headless_mode(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --serve --watch without --tui runs headless daemon mode.

    This test just verifies the combination is accepted. We don't test full
    daemon functionality here since it would require socket interactions.
    The error we get is from trying to run watch mode in test environment.
    """
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--serve", "--watch"])

    # We expect it to try to start (and fail in test environment)
    # The key thing is that the combination is accepted, not rejected
    # If the combination were invalid, we'd see a validation error message
    assert "--serve requires --watch" not in result.output


def test_repro_debounce_requires_watch(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --debounce without --watch errors."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--debounce", "500"])

    assert result.exit_code != 0
    assert "--debounce" in result.output
    assert "--watch" in result.output


def test_repro_tui_log_cannot_use_with_json(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --tui-log cannot be used with --json."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--tui-log", "test.log", "--json"])

    assert result.exit_code != 0
    assert "--tui-log" in result.output
    assert "--json" in result.output


def test_repro_tui_log_cannot_use_with_dry_run(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --tui-log cannot be used with --dry-run."""
    register_test_stage(_helper_stage_a, name="stage_a")

    # Note: --tui is required for --tui-log, so we include it to test the --dry-run validation
    result = runner.invoke(cli.cli, ["repro", "--tui", "--tui-log", "test.log", "--dry-run"])

    assert result.exit_code != 0
    assert "--tui-log" in result.output
    assert "--dry-run" in result.output


def test_repro_unknown_stage_errors(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro with unknown stage name errors with helpful message."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "nonexistent_stage" in result.output


# =============================================================================
# Force Mode Tests
# =============================================================================


def test_repro_force_reruns_cached_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --force re-runs stages even if cached."""
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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --keep-going option is accepted."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    # Command should run successfully (or fail gracefully)
    # Main check is that the option is recognized
    assert "--keep-going" not in result.output or result.exit_code == 0


# =============================================================================
# No Single-Stage Option Tests
# =============================================================================


def test_repro_does_not_have_single_stage_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro does not have --single-stage option."""
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
    with isolated_pivot_dir(runner, tmp_path):
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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --json streams JSONL output."""
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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --no-commit option is accepted."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["repro", "--no-commit"])

    assert result.exit_code == 0


# =============================================================================
# Show Output Flag Tests
# =============================================================================


def test_repro_show_output_mutually_exclusive_with_tui(
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --tui are mutually exclusive."""
    with runner.isolated_filesystem():
        pathlib.Path("pivot.yaml").write_text("stages: {}")

        result = runner.invoke(cli.cli, ["repro", "--show-output", "--tui"])

        assert result.exit_code != 0
        assert "--show-output and --tui are mutually exclusive" in result.output


def test_repro_show_output_mutually_exclusive_with_json(
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --json are mutually exclusive."""
    with runner.isolated_filesystem():
        pathlib.Path("pivot.yaml").write_text("stages: {}")

        result = runner.invoke(cli.cli, ["repro", "--show-output", "--json"])

        assert result.exit_code != 0
        assert "--show-output and --json are mutually exclusive" in result.output


def test_repro_show_output_mutually_exclusive_with_quiet(
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --quiet are mutually exclusive."""
    with runner.isolated_filesystem():
        pathlib.Path("pivot.yaml").write_text("stages: {}")

        result = runner.invoke(cli.cli, ["--quiet", "repro", "--show-output"])

        assert result.exit_code != 0
        assert "--show-output and --quiet are mutually exclusive" in result.output


def test_repro_show_output_streams_stage_logs(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stage stdout to terminal."""
    import sys

    class _PrintOutputs(TypedDict):
        output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

    def _helper_printing_stage() -> _PrintOutputs:
        # This will be captured and should appear in output
        sys.stdout.write("Processing data...\n")
        sys.stdout.flush()
        pathlib.Path("output.txt").write_text("done")
        return _PrintOutputs(output=pathlib.Path("output.txt"))

    register_test_stage(_helper_printing_stage, name="printer")

    result = runner.invoke(cli.cli, ["repro", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should contain the log line with stage prefix
    assert "printer" in result.output
    assert "Processing data..." in result.output


def test_repro_show_output_streams_stderr_in_red(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stderr with red formatting."""
    import sys

    class _StderrOutputs(TypedDict):
        output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

    def _helper_stderr_stage() -> _StderrOutputs:
        sys.stderr.write("Warning: something happened\n")
        sys.stderr.flush()
        pathlib.Path("output.txt").write_text("done")
        return _StderrOutputs(output=pathlib.Path("output.txt"))

    register_test_stage(_helper_stderr_stage, name="warner")

    result = runner.invoke(cli.cli, ["repro", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should contain stderr line
    assert "warner" in result.output
    assert "Warning: something happened" in result.output


def test_repro_without_show_output_hides_logs(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Default behavior (no --show-output) doesn't show stage logs."""
    import sys

    class _QuietOutputs(TypedDict):
        output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

    def _helper_quiet_stage() -> _QuietOutputs:
        sys.stdout.write("This should not appear\n")
        sys.stdout.flush()
        pathlib.Path("output.txt").write_text("done")
        return _QuietOutputs(output=pathlib.Path("output.txt"))

    register_test_stage(_helper_quiet_stage, name="quiet")

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should NOT contain the log line (only stage completion message)
    assert "This should not appear" not in result.output
    # But should show completion
    assert "quiet" in result.output
