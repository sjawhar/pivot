import pathlib

import click.testing
import pytest

from pivot import cli, executor, project, stage
from pivot.registry import REGISTRY
from pivot.tui import console


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


def test_cli_help_shows_commands(runner: click.testing.CliRunner) -> None:
    """CLI should show available commands in help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "list" in result.output


def test_cli_run_help(runner: click.testing.CliRunner) -> None:
    """Run subcommand should show its own help."""
    result = runner.invoke(cli.cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--single-stage" in result.output
    assert "--dry-run" in result.output


def test_cli_run_no_stages_registered(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Run with no stages should report empty pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        (pathlib.Path.cwd() / ".git").mkdir()
        result = runner.invoke(cli.cli, ["run"])
        assert result.exit_code == 0
        assert "No stages to run" in result.output


def test_cli_verbose_accepted(runner: click.testing.CliRunner) -> None:
    """Verbose flag should be accepted."""
    result = runner.invoke(cli.cli, ["--verbose", "run", "--help"])
    assert result.exit_code == 0


def test_cli_list_command_exists(runner: click.testing.CliRunner) -> None:
    """List subcommand should be available."""
    result = runner.invoke(cli.cli, ["list", "--help"])
    assert result.exit_code == 0


def test_cli_list_shows_registered_stages(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
    """List should show registered stages."""
    output_file = set_project_root / "out.txt"

    @stage(deps=[], outs=[str(output_file)])
    def my_stage() -> None:
        pass

    result = runner.invoke(cli.cli, ["list"])
    assert result.exit_code == 0
    assert "my_stage" in result.output


def test_cli_list_verbose_shows_details(
    runner: click.testing.CliRunner, set_project_root: pathlib.Path
) -> None:
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


def test_cli_dry_run_shows_what_would_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
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


def test_cli_dry_run_shows_unchanged_as_skip(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
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


def test_cli_dry_run_missing_deps_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
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


def test_cli_dry_run_no_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Dry-run with no stages reports empty pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--dry-run"])

        assert result.exit_code == 0
        assert "No stages" in result.output


def test_cli_dry_run_specific_stage(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
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


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_cli_run_exception_shows_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Run command shows error when exception occurs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "error" in result.output.lower()


def test_cli_dry_run_exception_shows_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Dry-run command shows error when exception occurs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "nonexistent"])

        assert result.exit_code != 0


# =============================================================================
# Results Printing Tests
# =============================================================================


def test_cli_run_prints_results(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Run command prints results for each stage."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        # Clear cache after entering isolated filesystem
        project._project_root_cache = None

        def stage_fn() -> None:
            pathlib.Path("output.txt").write_text("done")

        REGISTRY.register(stage_fn, name="my_stage", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code == 0
        assert "my_stage" in result.output
        assert "ran" in result.output.lower()
        assert "Total:" in result.output


def test_cli_run_prints_skipped_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Run command correctly shows skipped stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        # Clear cache after entering isolated filesystem
        project._project_root_cache = None

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


def test_cli_list_no_stages(runner: click.testing.CliRunner) -> None:
    """List with no stages shows appropriate message."""
    result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code == 0
    assert "No stages registered" in result.output


# =============================================================================
# Categorized Help Tests
# =============================================================================


def test_cli_help_shows_categorized_commands(runner: click.testing.CliRunner) -> None:
    """CLI help should show commands grouped by category."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "Pipeline Commands:" in result.output
    assert "Inspection Commands:" in result.output
    assert "Versioning Commands:" in result.output
    assert "Remote Commands:" in result.output


def test_cli_help_contains_pipeline_commands(runner: click.testing.CliRunner) -> None:
    """Help output should contain pipeline commands (run, explain)."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0

    # Test that commands appear in output (without relying on section positions)
    assert "run" in result.output, "Should show 'run' command"
    assert "explain" in result.output, "Should show 'explain' command"


def test_cli_help_contains_inspection_commands(runner: click.testing.CliRunner) -> None:
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


def test_cli_run_unknown_stage_shows_suggestion(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
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
