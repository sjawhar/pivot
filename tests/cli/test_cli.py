from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import pandas  # noqa: TC002 - needed for type hint resolution

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.tui import console

if TYPE_CHECKING:
    from click.testing import CliRunner


# =============================================================================
# Module-level stage functions for testing (required for pickling)
# =============================================================================


class _IntegrationTestOutputs(TypedDict):
    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]


def integration_process_data(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())],
) -> _IntegrationTestOutputs:
    """Module-level stage function for integration testing."""
    return {"result": {"count": len(data), "sum_a": int(data["a"].sum())}}


# Stage with no deps, one output (out.txt)
class _OutTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("out.txt", loaders.PathOnly())]


def _stage_my_stage() -> _OutTxtOutputs:
    return {"output": pathlib.Path("out.txt")}


# Stage with input.txt dep and output.txt out
class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _stage_with_input(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    return {"output": pathlib.Path("output.txt")}


def _stage_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    return {"output": pathlib.Path("output.txt")}


def _stage_process_creates_file(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


# Stage with missing_input.txt dep and output.txt out
def _stage_missing_input(
    missing_input: Annotated[pathlib.Path, outputs.Dep("missing_input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = missing_input
    return {"output": pathlib.Path("output.txt")}


# Stages for specific-stage testing: stage_a and stage_b
class _ATxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


def _stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxtOutputs:
    _ = input_file
    return {"output": pathlib.Path("a.txt")}


def _stage_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxtOutputs:
    _ = input_file
    return {"output": pathlib.Path("b.txt")}


# Named stage (my_stage) with input.txt dep and output.txt out
def _stage_my_stage_with_input(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


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
    assert "--single-stage" in result.output
    assert "--dry-run" in result.output


def test_cli_run_no_stages_registered(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run with no stages should report empty pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        (pathlib.Path.cwd() / ".git").mkdir()
        result = runner.invoke(cli.cli, ["run"])
        assert result.exit_code == 0
        assert "No stages to run" in result.output


def test_cli_verbose_accepted(runner: CliRunner) -> None:
    """Verbose flag should be accepted."""
    result = runner.invoke(cli.cli, ["--verbose", "run", "--help"])
    assert result.exit_code == 0


def test_cli_list_command_exists(runner: CliRunner) -> None:
    """List subcommand should be available."""
    result = runner.invoke(cli.cli, ["list", "--help"])
    assert result.exit_code == 0


def test_cli_list_shows_registered_stages(
    runner: CliRunner, set_project_root: pathlib.Path
) -> None:
    """List should show registered stages."""
    register_test_stage(_stage_my_stage, name="my_stage")

    result = runner.invoke(cli.cli, ["list"])
    assert result.exit_code == 0
    assert "my_stage" in result.output


def test_cli_list_verbose_shows_details(runner: CliRunner, set_project_root: pathlib.Path) -> None:
    """List --verbose should show deps and outputs."""
    register_test_stage(_stage_with_input, name="my_stage")

    result = runner.invoke(cli.cli, ["--verbose", "list"])
    assert result.exit_code == 0
    assert "my_stage" in result.output
    assert "deps:" in result.output
    assert "outs:" in result.output


# =============================================================================
# Dry-Run Command Tests
# =============================================================================


def test_cli_dry_run_shows_what_would_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run shows stages that would run."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--dry-run"])

        assert result.exit_code == 0
        assert "Would run:" in result.output
        assert "process" in result.output
        assert "would run" in result.output


def test_cli_dry_run_shows_unchanged_as_skip(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run shows unchanged stages as 'would skip'."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_process_creates_file, name="process")

        # First, actually run to create lock file
        executor.run()

        # Now dry-run should show as unchanged
        result = runner.invoke(cli.cli, ["run", "--dry-run"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "would skip" in result.output or "unchanged" in result.output


def test_cli_force_dry_run_shows_forced(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with --force shows stages as 'would run (forced)'."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_process_creates_file, name="process")

        # First, actually run to create lock file
        executor.run()

        # Now dry-run with --force should show as 'would run (forced)'
        result = runner.invoke(cli.cli, ["run", "--dry-run", "--force"])

        assert result.exit_code == 0
        assert "process" in result.output
        assert "would run" in result.output
        assert "forced" in result.output


def test_cli_dry_run_missing_deps_errors(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run fails when dependencies don't exist and aren't produced by other stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Don't create input.txt - it's missing and not produced by any stage

        register_test_stage(_stage_missing_input, name="process")

        result = runner.invoke(cli.cli, ["run", "--dry-run"])

        # This should fail because the dependency doesn't exist
        assert result.exit_code != 0
        assert "missing_input.txt" in result.output


def test_cli_dry_run_no_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with no stages reports empty pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--dry-run"])

        assert result.exit_code == 0
        assert "No stages" in result.output


def test_cli_dry_run_specific_stage(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with stage argument only shows specified stage and dependencies."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_a, name="stage_a")
        register_test_stage(_stage_b, name="stage_b")

        result = runner.invoke(cli.cli, ["run", "--dry-run", "stage_a"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


def test_cli_dry_run_json_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with --json outputs valid JSON with stage information."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data, "JSON output should have 'stages' key"
        assert "process" in data["stages"], "Stage 'process' should be in output"
        assert data["stages"]["process"]["would_run"] is True
        assert "reason" in data["stages"]["process"]


def test_cli_dry_run_json_empty_pipeline(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run --json with no stages outputs empty stages dict."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == {"stages": {}}


def test_cli_dry_run_json_with_force(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run --json --force shows forced status."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_process_creates_file, name="process")

        # First, actually run to create lock file
        executor.run()

        # Now dry-run with --force --json
        result = runner.invoke(cli.cli, ["run", "--dry-run", "--force", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert "process" in data["stages"]
        assert data["stages"]["process"]["would_run"] is True
        assert "forced" in data["stages"]["process"]["reason"]


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_cli_run_exception_shows_error(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command shows error when exception occurs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "error" in result.output.lower()


def test_cli_dry_run_exception_shows_error(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run command shows error when exception occurs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "nonexistent"])

        assert result.exit_code != 0


# =============================================================================
# Results Printing Tests
# =============================================================================


def test_cli_run_prints_results(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command prints results for each stage."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code == 0
        assert "my_stage" in result.output
        assert "ran" in result.output.lower()
        assert "Summary:" in result.output


def test_cli_run_prints_skipped_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command correctly shows skipped stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        # First run via CLI
        result1 = runner.invoke(cli.cli, ["run"])
        assert result1.exit_code == 0
        assert "ran" in result1.output.lower()

        # Reset console singleton (CliRunner closes streams)
        console._console = None

        # Second run via CLI - should skip
        result2 = runner.invoke(cli.cli, ["run"])

        assert result2.exit_code == 0
        assert "cached" in result2.output.lower()


# =============================================================================
# List Command Tests
# =============================================================================


def test_cli_list_no_stages(runner: CliRunner) -> None:
    """List with no stages shows appropriate message."""
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
    assert "Versioning Commands:" in result.output
    assert "Remote Commands:" in result.output


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
    assert "data" in result.output, "Should show 'data' command"


# =============================================================================
# Error Message with Suggestion Tests
# =============================================================================


def test_cli_run_unknown_stage_shows_suggestion(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Running unknown stage shows error with suggestion to run pivot list."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run", "nonexistent_stage"])

        assert result.exit_code != 0
        assert "nonexistent_stage" in result.output
        assert "pivot list" in result.output, "Should suggest running pivot list"


# =============================================================================
# JSONL Streaming Tests (pivot run --json)
# =============================================================================


def test_cli_run_json_emits_schema_version(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits schema_version event first."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run", "--json"])

        assert result.exit_code == 0
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) >= 1, "Should have at least one event"

        first_event = json.loads(lines[0])
        assert first_event["type"] == "schema_version"
        assert first_event["version"] == 1


def test_cli_run_json_emits_stage_events(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits stage_start and stage_complete events."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run", "--json"])

        assert result.exit_code == 0
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        event_types = [e["type"] for e in events]
        assert "stage_start" in event_types, "Should emit stage_start event"
        assert "stage_complete" in event_types, "Should emit stage_complete event"


def test_cli_run_json_emits_execution_result(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits execution_result event at end."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run", "--json"])

        assert result.exit_code == 0
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        last_event = events[-1]
        assert last_event["type"] == "execution_result"
        assert "ran" in last_event
        assert "skipped" in last_event
        assert "failed" in last_event
        assert "total_duration_ms" in last_event


def test_cli_run_json_no_stages_emits_events(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run --json emits events even with no stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--json"])

        assert result.exit_code == 0
        events = [json.loads(line) for line in result.output.strip().split("\n") if line]

        assert events[0]["type"] == "schema_version"
        assert events[-1]["type"] == "execution_result"
        assert events[-1]["ran"] == 0


def test_cli_run_json_stage_complete_has_duration(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run --json stage_complete events include duration_ms."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run", "--json"])

        assert result.exit_code == 0
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
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["--quiet", "list"])

        assert result.exit_code == 0
        # Output should be empty or minimal (just newlines)
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_run_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet run produces no output when stages run successfully."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create a simple pipeline using pipeline.py
        pathlib.Path("pipeline.py").write_text(
            """\
from typing import Annotated, TypedDict
from pivot.registry import REGISTRY
from pivot import outputs, loaders
import pathlib

class _TestOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

def test_stage() -> _TestOutputs:
    pathlib.Path("output.txt").write_text("hello")
    return {"output": pathlib.Path("output.txt")}

REGISTRY.register(test_stage)
"""
        )

        result = runner.invoke(cli.cli, ["--quiet", "run"])

        assert result.exit_code == 0, f"Run failed: {result.output}"
        assert result.output.strip() == "", "Quiet mode should suppress all output"
        # Verify stage actually ran
        assert pathlib.Path("output.txt").exists()


def test_cli_track_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet track produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("data.txt").write_text("test data")

        result = runner.invoke(cli.cli, ["--quiet", "track", "data.txt"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"
        # Verify file was tracked
        assert pathlib.Path("data.txt.pvt").exists()


def test_cli_checkout_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet checkout produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Track a file first
        pathlib.Path("data.txt").write_text("test data")
        runner.invoke(cli.cli, ["track", "data.txt"])

        # Remove original and checkout
        pathlib.Path("data.txt").unlink()

        result = runner.invoke(cli.cli, ["--quiet", "checkout"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"
        # Verify file was restored
        assert pathlib.Path("data.txt").exists()


def test_cli_commit_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet commit produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # No pending stages, but --list should still be quiet
        result = runner.invoke(cli.cli, ["--quiet", "commit", "--list"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_export_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet export produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Create a simple pipeline using pipeline.py
        pathlib.Path("pipeline.py").write_text(
            """\
from pivot.registry import REGISTRY

def test_stage():
    pass

REGISTRY.register(test_stage)
"""
        )

        result = runner.invoke(cli.cli, ["--quiet", "export"])

        assert result.exit_code == 0, f"Export failed: {result.output}"
        assert result.output.strip() == "", "Quiet mode should suppress output"
        # Verify file was created
        assert pathlib.Path("dvc.yaml").exists()


def test_cli_doctor_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet doctor produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["--quiet", "doctor"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_cli_history_quiet_produces_no_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot --quiet history produces no output."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["--quiet", "history"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output"


# =============================================================================
# Metrics Output Tests
# =============================================================================


def test_cli_run_metrics_env_var(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """PIVOT_METRICS=1 enables metrics display to stderr."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run"], env={"PIVOT_METRICS": "1"})

        assert result.exit_code == 0
        assert "Metrics:" in result.output
        # Metrics are collected and displayed (cli.total may not be present in
        # in-process tests since it requires the env var set at module import time)
        assert "ms" in result.output


def test_cli_run_no_metrics_by_default(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot run does not show metrics by default."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_stage_my_stage_with_input, name="my_stage")

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code == 0
        assert "Metrics:" not in result.output
