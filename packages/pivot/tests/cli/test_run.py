"""Tests for pivot run CLI command (single-stage executor)."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from pivot import cli, compose

if TYPE_CHECKING:
    import pathlib

    import click.testing

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def stage_a() -> str:
    return "a"


@compose.stage
def stage_b(dep: str) -> str:
    return "b"


@compose.stage
def process(input_file: str) -> str:
    return "done"


@compose.stage
def failing() -> str:
    raise RuntimeError("Intentional failure")


@compose.stage
def printer() -> str:
    sys.stdout.write("Processing data...\n")
    sys.stdout.flush()
    return "done"


@compose.stage
def warner() -> str:
    sys.stderr.write("Warning: something happened\n")
    sys.stderr.flush()
    return "done"


@compose.stage
def quiet() -> str:
    sys.stdout.write("This should not appear\n")
    sys.stdout.flush()
    return "done"


def _output(pipeline: compose.Pipeline, stage_name: str) -> pathlib.Path:
    return pipeline.root / "data" / "test" / f"{stage_name}.txt"


def _register_a(pipeline: compose.Pipeline) -> None:
    with pipeline:
        stage_a()


def _register_a_b(pipeline: compose.Pipeline) -> None:
    with pipeline:
        stage_b(stage_a())


def _register_process(pipeline: compose.Pipeline) -> None:
    raw_dir = pipeline.root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text("data")
    with pipeline:
        process(pipeline.input("input.txt", t=str))


# =============================================================================
# Basic Command Tests
# =============================================================================


def test_run_requires_stage_argument(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run without arguments errors with usage message."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run"])

    assert result.exit_code != 0
    assert "STAGES" in result.output


def test_run_executes_single_stage(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run STAGE executes only that stage."""
    _register_a_b(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "test/stage_a"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").read_text() == "a"
    # stage_b should NOT run (no dependency resolution)
    assert not _output(mock_discovery, "stage_b").exists()


def test_run_bare_stage_name_resolves(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run accepts a bare stage name in a single-pipeline project (names.py design)."""
    _register_a_b(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "stage_a"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").read_text() == "a"


def test_run_executes_multiple_stages_in_order(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run STAGE1 STAGE2 executes stages in specified order."""
    _register_a_b(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "test/stage_a", "test/stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").exists()
    assert _output(mock_discovery, "stage_b").exists()


def test_run_does_not_resolve_dependencies(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run STAGE does not automatically run dependencies."""
    _register_a_b(mock_discovery)
    # Create the dependency file manually
    a_out = _output(mock_discovery, "stage_a")
    a_out.parent.mkdir(parents=True, exist_ok=True)
    a_out.write_text("manual")

    # Run only stage_b - should NOT run stage_a
    result = runner.invoke(cli.cli, ["run", "test/stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # stage_a should not have been re-run (file still has "manual" content)
    assert a_out.read_text() == "manual"
    assert _output(mock_discovery, "stage_b").exists()


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_run_unknown_stage_errors(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run with unknown stage name errors with helpful message."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "nonexistent_stage" in result.output


def test_run_default_fails_fast(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run defaults to fail-fast mode - stops on first failure."""
    with mock_discovery:
        failing()
        stage_a()

    # Run failing then stage_a - default fail-fast should stop
    result = runner.invoke(cli.cli, ["run", "test/failing", "test/stage_a"])

    # Exit 1 on failure: #452 commit 378abd0 made run return non-zero on stage failure
    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output


def test_run_fail_fast_option_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run --fail-fast option is accepted and affects execution mode."""
    _register_a(mock_discovery)

    # Verify --fail-fast is accepted without error
    result = runner.invoke(cli.cli, ["run", "--fail-fast", "test/stage_a"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").exists()


# =============================================================================
# Force Mode Tests
# =============================================================================


def test_run_force_reruns_cached_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run --force re-runs stages even if cached."""
    _register_a(mock_discovery)

    # First run
    result = runner.invoke(cli.cli, ["run", "test/stage_a"])
    assert result.exit_code == 0, result.output

    # Second run with force - should re-run (not skip)
    result = runner.invoke(cli.cli, ["run", "--force", "--json", "test/stage_a"])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.strip().split("\n") if line]
    completes = [e for e in events if e["type"] == "stage_complete"]
    assert [e["status"] for e in completes] == ["ran"], "--force must re-run, not skip"


# =============================================================================
# Option Validation Tests
# =============================================================================


def test_run_tui_log_cannot_use_with_json(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run --tui-log cannot be used with --json."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--tui-log", "test.log", "--json", "test/stage_a"])

    assert result.exit_code != 0
    assert "--tui-log" in result.output
    assert "--jsonl" in result.output


def test_run_tui_and_json_mutually_exclusive(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--tui and --json are mutually exclusive."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--tui", "--json", "test/stage_a"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_run_tui_log_requires_tui(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--tui-log requires --tui flag."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--tui-log", "log.jsonl", "test/process"])

    assert result.exit_code != 0
    assert "--tui-log requires --tui" in result.output


def test_run_help_includes_tui_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """--tui flag appears in help text."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--tui" in result.output
    # Help text is case-sensitive for "TUI"
    assert "TUI display" in result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_run_json_output_streams_jsonl(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run --json streams JSONL output."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--json", "test/stage_a"])

    assert result.exit_code == 0, result.output
    # Check that output is valid JSONL (multiple JSON lines)
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise


def test_run_json_flag_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--json flag should work without --tui (plain is now default)."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--json", "test/process"])

    assert result.exit_code == 0, result.output
    # JSONL output should start with schema version
    assert '"type": "schema_version"' in result.output


def test_run_jsonl_flag_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--jsonl flag works and streams JSONL output."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--jsonl", "test/stage_a"])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise


def test_run_help_shows_jsonl(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """run help shows --jsonl flag with JSONL description."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--jsonl" in result.output
    assert "JSONL" in result.output


def test_run_uses_plain_mode_by_default(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Plain text output is the default (no --tui flag needed)."""
    _register_process(mock_discovery)

    # Run without any display flags
    result = runner.invoke(cli.cli, ["run", "test/process"])

    assert result.exit_code == 0, result.output
    # Plain mode shows stage status in simple text format
    # (not JSONL format which has "type":)
    assert '"type":' not in result.output
    # Should contain stage name in output
    assert "test/process" in result.output


# =============================================================================
# Cache Options Tests
# =============================================================================


def test_run_no_commit_option_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """run --no-commit option is accepted."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--no-commit", "test/stage_a"])

    assert result.exit_code == 0, result.output
    assert _output(mock_discovery, "stage_a").exists()


# =============================================================================
# Removed Options Tests (should not exist on run)
# =============================================================================


@pytest.mark.parametrize(
    "option",
    [
        pytest.param("--single-stage", id="single_stage"),
        pytest.param("--dry-run", id="dry_run"),
        pytest.param("--explain", id="explain"),
        pytest.param("--watch", id="watch"),
        pytest.param("--allow-missing", id="allow_missing"),
    ],
)
def test_run_does_not_have_repro_only_option(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    option: str,
) -> None:
    """run does not have repro-only options (run is always single-stage)."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", option, "test/stage_a"])

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_run_tui_with_tui_log_validation_passes(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--tui --tui-log passes validation (log path is writable)."""
    _register_process(mock_discovery)

    log_path = mock_discovery.root / "tui.jsonl"

    # This will attempt to run TUI which fails in non-TTY test environment,
    # but it should NOT fail on validation errors about --tui-log
    result = runner.invoke(cli.cli, ["run", "--tui", "--tui-log", str(log_path), "test/process"])

    # Should NOT have validation errors
    assert "--tui-log requires --tui" not in result.output
    assert "Cannot write to" not in result.output
    # The log file should have been created during validation (touch())
    assert log_path.exists()


# =============================================================================
# --show-output Tests
# =============================================================================


def test_run_show_output_mutually_exclusive_with_tui(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --tui are mutually exclusive."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "test/stage_a", "--show-output", "--tui"])

    assert result.exit_code != 0
    assert "--show-output and --tui are mutually exclusive" in result.output


def test_run_show_output_mutually_exclusive_with_json(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --json are mutually exclusive."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "test/stage_a", "--show-output", "--json"])

    assert result.exit_code != 0
    assert "--show-output and --jsonl/--json are mutually exclusive" in result.output


def test_run_show_output_mutually_exclusive_with_quiet(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --quiet are mutually exclusive."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["--quiet", "run", "test/stage_a", "--show-output"])

    assert result.exit_code != 0
    assert "--show-output and --quiet are mutually exclusive" in result.output


def test_run_show_output_streams_stage_logs(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stage stdout to terminal."""
    with mock_discovery:
        printer()

    result = runner.invoke(cli.cli, ["run", "test/printer", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "printer" in result.output
    assert "Processing data..." in result.output


def test_run_show_output_streams_stderr(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stderr with red formatting."""
    with mock_discovery:
        warner()

    result = runner.invoke(cli.cli, ["run", "test/warner", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "warner" in result.output
    assert "Warning: something happened" in result.output


def test_run_without_show_output_hides_logs(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Default behavior (no --show-output) doesn't show stage logs."""
    with mock_discovery:
        quiet()

    result = runner.invoke(cli.cli, ["run", "test/quiet"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "This should not appear" not in result.output
    assert "quiet" in result.output
