from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

from pivot import cli, executor, stage
from pivot.registry import REGISTRY
from pivot.tui import console

if TYPE_CHECKING:
    from click.testing import CliRunner


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
    output_file = set_project_root / "out.txt"

    @stage(deps=[], outs=[str(output_file)])
    def my_stage() -> None:
        pass

    result = runner.invoke(cli.cli, ["list"])
    assert result.exit_code == 0
    assert "my_stage" in result.output


def test_cli_list_verbose_shows_details(runner: CliRunner, set_project_root: pathlib.Path) -> None:
    """List --verbose should show deps and outputs."""
    input_file = set_project_root / "input.txt"
    output_file = set_project_root / "output.txt"

    @stage(deps=[str(input_file)], outs=[str(output_file)])
    def my_stage() -> None:
        pass

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

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

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

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("done")

        # First, actually run to create lock file
        executor.run(show_output=False)

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

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("done")

        # First, actually run to create lock file
        executor.run(show_output=False)

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

        @stage(deps=["missing_input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

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

        @stage(deps=["input.txt"], outs=["a.txt"])
        def stage_a() -> None:
            pass

        @stage(deps=["input.txt"], outs=["b.txt"])
        def stage_b() -> None:
            pass

        result = runner.invoke(cli.cli, ["run", "--dry-run", "stage_a"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


def test_cli_dry_run_json_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with --json outputs valid JSON with stage information."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

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

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("done")

        # First, actually run to create lock file
        executor.run(show_output=False)

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

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code == 0
        assert "my_stage" in result.output
        assert "ran" in result.output.lower()
        assert "Total:" in result.output


def test_cli_run_prints_skipped_stages(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command correctly shows skipped stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

        # First run via CLI
        result1 = runner.invoke(cli.cli, ["run"])
        assert result1.exit_code == 0
        assert "ran" in result1.output.lower()

        # Reset console singleton (CliRunner closes streams)
        console._console = None

        # Second run via CLI - should skip
        result2 = runner.invoke(cli.cli, ["run"])

        assert result2.exit_code == 0
        assert "skipped" in result2.output.lower()


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
    """Help output should contain pipeline commands (run, explain)."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0

    # Test that commands appear in output (without relying on section positions)
    assert "run" in result.output, "Should show 'run' command"
    assert "explain" in result.output, "Should show 'explain' command"


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

        @stage(deps=["input.txt"], outs=["output.txt"])
        def my_stage() -> None:
            pass

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

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

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

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

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

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

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

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

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
