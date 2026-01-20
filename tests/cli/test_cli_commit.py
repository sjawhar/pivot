from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import lock

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level helper functions for stages
# =============================================================================


class _ProcessOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ProcessOutputs:
    """Helper stage that writes output.txt."""
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


# =============================================================================
# commit --list tests
# =============================================================================


def test_commit_list_empty(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """commit --list shows no pending stages when none exist."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot/pending/stages").mkdir(parents=True)

        result = runner.invoke(cli.cli, ["commit", "--list"])

        assert result.exit_code == 0
        assert "No pending stages" in result.output


def test_commit_list_shows_pending(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """commit --list shows stages pending from --no-commit runs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run with --no-commit
        executor.run(show_output=False, no_commit=True)

        result = runner.invoke(cli.cli, ["commit", "--list"])

        assert result.exit_code == 0
        assert "Pending stages:" in result.output
        assert "process" in result.output


# =============================================================================
# commit tests
# =============================================================================


def test_commit_nothing_to_commit(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """commit with no pending stages shows nothing to commit."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot/pending/stages").mkdir(parents=True)
        pathlib.Path(".pivot/cache/stages").mkdir(parents=True)

        result = runner.invoke(cli.cli, ["commit"])

        assert result.exit_code == 0
        assert "Nothing to commit" in result.output


def test_commit_promotes_pending_to_production(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """commit promotes pending locks to production."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run with --no-commit
        executor.run(show_output=False, no_commit=True)

        # Verify pending lock exists
        project_root = pathlib.Path.cwd()
        pending_lock = lock.get_pending_lock("process", project_root)
        assert pending_lock.path.exists(), "Pending lock should exist after --no-commit run"

        # Commit
        result = runner.invoke(cli.cli, ["commit"])

        assert result.exit_code == 0
        assert "Committed 1 stage(s)" in result.output
        assert "process" in result.output

        # Verify pending lock removed
        assert not pending_lock.path.exists(), "Pending lock should be removed after commit"


# =============================================================================
# commit --discard tests
# =============================================================================


def test_commit_discard_nothing(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """commit --discard with no pending stages shows nothing to discard."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot/pending/stages").mkdir(parents=True)

        result = runner.invoke(cli.cli, ["commit", "--discard"])

        assert result.exit_code == 0
        assert "No pending stages to discard" in result.output


def test_commit_discard_removes_pending(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """commit --discard removes pending locks without committing."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run with --no-commit
        executor.run(show_output=False, no_commit=True)

        # Verify pending lock exists
        project_root = pathlib.Path.cwd()
        pending_lock = lock.get_pending_lock("process", project_root)
        assert pending_lock.path.exists()

        # Discard
        result = runner.invoke(cli.cli, ["commit", "--discard"])

        assert result.exit_code == 0
        assert "Discarded 1 pending stage(s)" in result.output

        # Verify pending lock removed
        assert not pending_lock.path.exists()


# =============================================================================
# run --no-commit integration tests
# =============================================================================


def test_run_no_commit_creates_pending_lock(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """run --no-commit creates pending lock instead of production lock."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["run", "--no-commit"])

        assert result.exit_code == 0

        # Verify pending lock exists
        project_root = pathlib.Path.cwd()
        pending_lock = lock.get_pending_lock("process", project_root)
        assert pending_lock.path.exists(), "Pending lock should exist"

        # Verify production lock does NOT exist
        cache_dir = project_root / ".pivot" / "cache"
        production_lock = lock.StageLock("process", cache_dir)
        assert not production_lock.path.exists(), "Production lock should NOT exist"


def test_run_no_commit_second_run_skips(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Second run --no-commit skips unchanged stages (uses pending lock)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # First run via executor to set up pending lock
        executor.run(show_output=False, no_commit=True)

        # Second run via CLI should use cache
        result = runner.invoke(cli.cli, ["run", "--no-commit"])
        assert result.exit_code == 0, f"Failed with output: {result.output}"
        assert "cached" in result.output.lower() or "unchanged" in result.output.lower()


def test_run_no_commit_then_commit_workflow(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Full workflow: run --no-commit, then commit, then run uses production lock."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run with --no-commit via executor
        executor.run(show_output=False, no_commit=True)

        # Commit via CLI
        result1 = runner.invoke(cli.cli, ["commit"])
        assert result1.exit_code == 0
        assert "Committed 1 stage(s)" in result1.output

        # Now a normal run via CLI should use cache (uses production lock)
        result2 = runner.invoke(cli.cli, ["run"])
        assert result2.exit_code == 0, f"Failed with output: {result2.output}"
        assert "cached" in result2.output.lower() or "unchanged" in result2.output.lower()
