"""Tests for CLI module."""

import pathlib

import click.testing
import pytest

from pivot import cli, console, executor, project, stage
from pivot.registry import REGISTRY


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


def test_cli_help_shows_commands(runner: click.testing.CliRunner) -> None:
    """CLI should show available commands in help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output


def test_cli_run_help(runner: click.testing.CliRunner) -> None:
    """Run subcommand should show its own help."""
    result = runner.invoke(cli.cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--stages" in result.output
    assert "--dry-run" in result.output


def test_cli_run_no_stages_registered(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Run with no stages should report empty pipeline."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        (pathlib.Path.cwd() / ".git").mkdir()
        result = runner.invoke(cli.cli, ["run"])
        assert result.exit_code == 0
        assert "0 ran, 0 skipped" in result.output


def test_cli_verbose_accepted(runner: click.testing.CliRunner) -> None:
    """Verbose flag should be accepted."""
    result = runner.invoke(cli.cli, ["--verbose", "run", "--help"])
    assert result.exit_code == 0


def test_cli_status_command_exists(runner: click.testing.CliRunner) -> None:
    """Status subcommand should be available."""
    result = runner.invoke(cli.cli, ["status", "--help"])
    assert result.exit_code == 0


def test_cli_status_shows_registered_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status should show registered stages."""
    output_file = tmp_path / "out.txt"

    @stage(deps=[], outs=[str(output_file)])
    def my_stage() -> None:
        pass

    result = runner.invoke(cli.cli, ["status"])
    assert result.exit_code == 0
    assert "my_stage" in result.output


def test_cli_status_verbose_shows_details(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status --verbose should show deps and outputs."""
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"

    @stage(deps=[str(input_file)], outs=[str(output_file)])
    def my_stage() -> None:
        pass

    result = runner.invoke(cli.cli, ["--verbose", "status"])
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


def test_cli_dry_run_specific_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Dry-run with --stages only shows specified stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["a.txt"])
        def stage_a() -> None:
            pass

        @stage(deps=["input.txt"], outs=["b.txt"])
        def stage_b() -> None:
            pass

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--stages", "stage_a"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" not in result.output


def test_cli_dry_run_command_directly(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """dry-run command can be invoked directly."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pass

        result = runner.invoke(cli.cli, ["dry-run"])

        assert result.exit_code == 0
        assert "Would run:" in result.output


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_cli_run_exception_shows_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Run command shows error when exception occurs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--stages", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "error" in result.output.lower()


def test_cli_dry_run_exception_shows_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Dry-run command shows error when exception occurs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--dry-run", "--stages", "nonexistent"])

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
# Status Command Tests
# =============================================================================


def test_cli_status_no_stages(runner: click.testing.CliRunner) -> None:
    """Status with no stages shows appropriate message."""
    result = runner.invoke(cli.cli, ["status"])

    assert result.exit_code == 0
    assert "No stages registered" in result.output
