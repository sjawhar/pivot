from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli, compose
from pivot.cli import console

if TYPE_CHECKING:
    from click.testing import CliRunner


# =============================================================================
# Module-level stage functions for testing (required for pickling)
# =============================================================================


@compose.stage
def _stage_my_stage() -> str:
    return "out"


@compose.stage
def _stage_with_input(input_file: str) -> str:
    return input_file


# =============================================================================
# pipeline.py templates
# =============================================================================

_EMPTY_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline

with Pipeline("test") as pipeline:
    pass
"""

# One stage named STAGE_NAME that reads data/raw/input.txt and writes data/test/STAGE_NAME.txt
_SINGLE_STAGE_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def STAGE_NAME(input_file: str) -> str:
    return "done"


with Pipeline("test") as pipeline:
    STAGE_NAME(pipeline.input(INPUT_NAME))
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


def _write_input(content: str = "data", name: str = "input.txt") -> None:
    """Write a raw input where ``pipeline.input(name)`` reads it (data/raw/<name>)."""
    raw_dir = pathlib.Path("data", "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / name).write_text(content)


def _write_single_stage_pipeline(stage_name: str, input_name: str = "input.txt") -> None:
    pathlib.Path("pipeline.py").write_text(
        _SINGLE_STAGE_PIPELINE.replace("STAGE_NAME", stage_name).replace(
            "INPUT_NAME", repr(input_name)
        )
    )


def _write_empty_pipeline() -> None:
    pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)


def test_cli_help_shows_commands(runner: CliRunner) -> None:
    """CLI should show available commands in help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "list" in result.output


def test_cli_run_help(runner: CliRunner) -> None:
    """Run subcommand should show its own help."""
    result = runner.invoke(cli.cli, ["run", "--help"])
    assert result.exit_code == 0
    # run always uses single-stage mode, no --single-stage or --dry-run options
    assert "--fail-fast" in result.output
    assert "--force" in result.output


def test_cli_verbose_accepted(runner: CliRunner) -> None:
    """Verbose flag should be accepted."""
    result = runner.invoke(cli.cli, ["--verbose", "run", "--help"])
    assert result.exit_code == 0


def test_cli_list_command_exists(runner: CliRunner) -> None:
    """List subcommand should be available."""
    result = runner.invoke(cli.cli, ["list", "--help"])
    assert result.exit_code == 0


def test_cli_list_shows_registered_stages(
    runner: CliRunner,
    mock_discovery: compose.Pipeline,
) -> None:
    """List should show registered stages."""
    with mock_discovery:
        _stage_my_stage()

    result = runner.invoke(cli.cli, ["list"])
    assert result.exit_code == 0
    assert "test/_stage_my_stage" in result.output


def test_cli_list_verbose_shows_details(
    runner: CliRunner,
    mock_discovery: compose.Pipeline,
) -> None:
    """List --verbose should show deps and outputs."""
    with mock_discovery:
        _stage_with_input(mock_discovery.input("input.txt", t=str))

    result = runner.invoke(cli.cli, ["--verbose", "list"])
    assert result.exit_code == 0
    assert "test/_stage_with_input" in result.output
    assert "deps:" in result.output
    assert "input.txt" in result.output
    assert "outs:" in result.output


# =============================================================================
# Dry-Run Command Tests (repro command)
# =============================================================================


def test_cli_dry_run_shows_what_would_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run shows stages that would run."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("process")

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Would run:" in result.output
        assert "process" in result.output
        assert "would run" in result.output
        assert not pathlib.Path("data", "test", "process.txt").exists(), (
            "Dry-run must not execute the stage"
        )


def test_cli_dry_run_shows_unchanged_as_skip(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run shows unchanged stages as 'would skip'."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("process")

        # First, actually run to create lock file
        result1 = runner.invoke(cli.cli, ["run", "test/process"])
        assert result1.exit_code == 0, f"First run failed: {result1.output}"

        # Now dry-run should show as unchanged
        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "would skip" in result.output or "unchanged" in result.output


def test_cli_force_dry_run_shows_forced(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with --force shows stages as 'would run (forced)'."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("process")

        # First, actually run to create lock file
        result1 = runner.invoke(cli.cli, ["run", "test/process"])
        assert result1.exit_code == 0, f"First run failed: {result1.output}"

        # Now dry-run with --force should show as 'would run (forced)'
        result = runner.invoke(cli.cli, ["repro", "--dry-run", "--force"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "would run" in result.output
        assert "forced" in result.output


def test_cli_dry_run_missing_deps_errors(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run fails when dependencies don't exist and aren't produced by other stages."""
    with isolated_pivot_dir(runner, tmp_path):
        # Don't create missing_input.txt - it's missing and not produced by any stage
        _write_single_stage_pipeline("process", input_name="missing_input.txt")

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        # This should fail because the dependency doesn't exist
        assert result.exit_code != 0
        assert "missing_input.txt" in result.output


def test_cli_dry_run_no_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with no stages reports empty pipeline."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        assert result.exit_code == 0
        assert "No stages" in result.output


def test_cli_dry_run_specific_stage(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with stage argument only shows specified stage and dependencies."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        pathlib.Path("pipeline.py").write_text(_STAGES_A_B_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--dry-run", "test/stage_a"])

        assert result.exit_code == 0, result.output
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


def test_cli_dry_run_json_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with --json outputs valid JSON with stage information."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("process")

        result = runner.invoke(cli.cli, ["--quiet", "repro", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data, "JSON output should have 'stages' key"
        assert "test/process" in data["stages"], "Stage 'test/process' should be in output"
        assert data["stages"]["test/process"]["would_run"] is True
        assert "reason" in data["stages"]["test/process"]


def test_cli_dry_run_json_empty_pipeline(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run --json with no stages outputs empty stages dict."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["--quiet", "repro", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == {"stages": {}}


def test_cli_dry_run_json_with_force(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run --json --force shows forced status."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("process")

        # First, actually run to create lock file
        result1 = runner.invoke(cli.cli, ["run", "test/process"])
        assert result1.exit_code == 0, f"First run failed: {result1.output}"

        # Now dry-run with --force --json (--quiet to suppress logging)
        result = runner.invoke(cli.cli, ["--quiet", "repro", "--dry-run", "--force", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert "test/process" in data["stages"]
        assert data["stages"]["test/process"]["would_run"] is True
        assert "forced" in data["stages"]["test/process"]["reason"]


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_cli_run_exception_shows_error(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command shows error when exception occurs."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["run", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "error" in result.output.lower()


def test_cli_dry_run_exception_shows_error(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run command shows error when exception occurs."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["repro", "--dry-run", "nonexistent"])

        assert result.exit_code != 0


# =============================================================================
# Results Printing Tests
# =============================================================================


def test_cli_run_prints_results(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command prints results for each stage."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["run", "test/my_stage"])

        assert result.exit_code == 0, result.output
        assert "my_stage" in result.output
        assert "ran" in result.output.lower()
        assert "Summary:" in result.output
        assert pathlib.Path("data", "test", "my_stage.txt").read_text() == "done"


def test_cli_run_prints_skipped_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command correctly shows skipped stages."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        # First run via CLI
        result1 = runner.invoke(cli.cli, ["run", "test/my_stage"])
        assert result1.exit_code == 0
        assert "ran" in result1.output.lower()

        # Reset console singleton (CliRunner closes streams)
        console._console = None

        # Second run via CLI - should skip
        result2 = runner.invoke(cli.cli, ["run", "test/my_stage"])

        assert result2.exit_code == 0
        assert "cached" in result2.output.lower()


# =============================================================================
# List Command Tests
# =============================================================================


def test_cli_list_no_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """List with no stages shows appropriate message."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["list"])

        assert result.exit_code == 0
        assert "No stages registered" in result.output


# =============================================================================
# Categorized Help Tests
# =============================================================================


def test_cli_help_shows_categorized_commands(runner: CliRunner) -> None:
    """CLI help should show commands grouped by category."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "Pipeline Commands:" in result.output
    assert "Inspection Commands:" in result.output
    assert "Sync Commands:" in result.output
    assert "Other Commands:" in result.output


def test_cli_help_contains_pipeline_commands(runner: CliRunner) -> None:
    """Help output should contain pipeline commands (run, status)."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0

    # Test that commands appear in output (without relying on section positions)
    assert "run" in result.output, "Should show 'run' command"
    assert "status" in result.output, "Should show 'status' command"


def test_cli_help_contains_inspection_commands(runner: CliRunner) -> None:
    """Help output should contain inspection commands."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0

    # Test that inspection commands appear in output
    assert "list" in result.output, "Should show 'list' command"
    assert "metrics" in result.output, "Should show 'metrics' command"
    assert "params" in result.output, "Should show 'params' command"
    assert "plots" in result.output, "Should show 'plots' command"
    assert "diff" in result.output, "Should show 'diff' command"


# =============================================================================
# Error Message with Suggestion Tests
# =============================================================================


def test_cli_run_unknown_stage_shows_suggestion(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Running unknown stage shows error with suggestion to run pivot list."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["run", "nonexistent_stage"])

        assert result.exit_code != 0
        assert "nonexistent_stage" in result.output
        assert "pivot list" in result.output, "Should suggest running pivot list"


# =============================================================================
# JSONL Streaming Tests (pivot run --json)
# =============================================================================


def test_cli_run_json_emits_schema_version(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits schema_version event first."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["--quiet", "run", "--json", "test/my_stage"])

        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) >= 1, "Should have at least one event"

        first_event = json.loads(lines[0])
        assert first_event["type"] == "schema_version"
        assert first_event["version"] == 2


def test_cli_run_json_emits_stage_events(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits stage_start and stage_complete events."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["--quiet", "run", "--json", "test/my_stage"])

        assert result.exit_code == 0, result.output
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        event_types = [e["type"] for e in events]
        assert "stage_start" in event_types, "Should emit stage_start event"
        assert "stage_complete" in event_types, "Should emit stage_complete event"


def test_cli_run_json_emits_execution_result(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits execution_result event at end."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["--quiet", "run", "--json", "test/my_stage"])

        assert result.exit_code == 0, result.output
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        last_event = events[-1]
        assert last_event["type"] == "execution_result"
        assert "ran" in last_event
        assert "skipped" in last_event
        assert "failed" in last_event
        assert "total_duration_ms" in last_event


def test_cli_repro_json_no_stages_emits_events(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot repro --json emits events even with no stages (run requires stages)."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        # Use repro for "no stages" case since run requires at least one stage
        result = runner.invoke(cli.cli, ["--quiet", "repro", "--json"])

        assert result.exit_code == 0
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        assert events[0]["type"] == "schema_version"
        assert events[-1]["type"] == "execution_result"
        assert events[-1]["ran"] == 0


def test_cli_run_json_stage_complete_has_duration(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run --json stage_complete events include duration_ms."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["--quiet", "run", "--json", "test/my_stage"])

        assert result.exit_code == 0, result.output
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        complete_events = [e for e in events if e["type"] == "stage_complete"]
        assert len(complete_events) >= 1
        assert "duration_ms" in complete_events[0]
        assert complete_events[0]["duration_ms"] >= 0


# =============================================================================
# Quiet Flag Tests
# =============================================================================


def test_cli_quiet_flag_accepted(runner: CliRunner) -> None:
    """--quiet flag should be accepted."""
    result = runner.invoke(cli.cli, ["--quiet", "--help"])
    assert result.exit_code == 0


def test_cli_quiet_verbose_mutually_exclusive(runner: CliRunner) -> None:
    """--quiet and --verbose are mutually exclusive."""
    result = runner.invoke(cli.cli, ["--quiet", "--verbose", "list"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_cli_list_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet list produces no output."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["--quiet", "list"])

        assert result.exit_code == 0
        # Output should be empty or minimal (just newlines)
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_run_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet run produces no output when stages run successfully."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text("""\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def test_stage() -> str:
    return "hello"


with Pipeline("test") as pipeline:
    test_stage()
""")

        result = runner.invoke(cli.cli, ["--quiet", "run", "test/test_stage"])

        assert result.exit_code == 0, f"Run failed: {result.output}"
        assert result.output.strip() == "", "Quiet mode should suppress all output"
        # Verify stage actually ran
        assert pathlib.Path("data", "test", "test_stage.txt").read_text() == "hello"


def test_cli_track_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet track produces no output."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("data.txt").write_text("test data")
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["--quiet", "track", "data.txt"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"
        # Verify file was tracked
        assert pathlib.Path("data.txt.pvt").exists()


def test_cli_checkout_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet checkout produces no output."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        # Track a file first
        pathlib.Path("data.txt").write_text("test data")
        track_result = runner.invoke(cli.cli, ["track", "data.txt"])
        assert track_result.exit_code == 0, track_result.output

        # Remove original and checkout
        pathlib.Path("data.txt").unlink()

        result = runner.invoke(cli.cli, ["--quiet", "checkout"])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == "", "Quiet mode should suppress output"
        # Verify file was restored
        assert pathlib.Path("data.txt").read_text() == "test data"


def test_cli_commit_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet commit produces no output."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["--quiet", "commit"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_doctor_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet doctor produces no output."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["--quiet", "doctor"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_history_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet history produces no output."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["--quiet", "history"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_dry_run_quiet_suppresses_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet repro --dry-run should suppress output."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("process")

        result = runner.invoke(cli.cli, ["--quiet", "repro", "--dry-run"])

        assert result.exit_code == 0
        assert result.output.strip() == "", (
            f"Expected empty output with --quiet, got: {result.output}"
        )


# =============================================================================
# Metrics Output Tests
# =============================================================================


def test_cli_run_metrics_env_var(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """PIVOT_METRICS=1 enables metrics display to stderr."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["run", "test/my_stage"], env={"PIVOT_METRICS": "1"})

        assert result.exit_code == 0, result.output
        assert "Metrics:" in result.output
        # Metrics are collected and displayed (cli.total may not be present in
        # in-process tests since it requires the env var set at module import time)
        assert "ms" in result.output


def test_cli_run_no_metrics_by_default(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run does not show metrics by default."""
    with isolated_pivot_dir(runner, tmp_path):
        _write_input()
        _write_single_stage_pipeline("my_stage")

        result = runner.invoke(cli.cli, ["run", "test/my_stage"])

        assert result.exit_code == 0, result.output
        assert "Metrics:" not in result.output


def test_cli_run_unknown_stage_in_empty_pipeline_shows_error(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """Running unknown stage in empty pipeline shows clear error.

    Edge case: combines two error conditions (no stages + unknown target).
    Should prioritize the stage-not-found error with helpful suggestion.
    """
    with isolated_pivot_dir(runner, tmp_path):
        _write_empty_pipeline()

        result = runner.invoke(cli.cli, ["run", "nonexistent_stage"])

        assert result.exit_code != 0
        assert "nonexistent_stage" in result.output
        # Should suggest listing stages or note that no stages are registered
        assert (
            "pivot list" in result.output
            or "No stages" in result.output
            or "not found" in result.output.lower()
        ), f"Got: {result.output}"
