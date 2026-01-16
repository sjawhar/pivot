from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

from pivot import cli, stage

if TYPE_CHECKING:
    from click.testing import CliRunner


# =============================================================================
# --keep-going CLI Integration Tests
# =============================================================================


def test_keep_going_flag_continues_after_failure(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--keep-going continues independent stages after failure."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["failing.txt"])
        def failing() -> None:
            raise RuntimeError("Intentional failure")

        @stage(deps=["input.txt"], outs=["succeeding.txt"])
        def succeeding() -> None:
            pathlib.Path("succeeding.txt").write_text("success")

        result = runner.invoke(cli.cli, ["run", "--keep-going"])

        assert result.exit_code == 0
        assert "failing: failed" in result.output
        assert "succeeding: ran" in result.output
        assert pathlib.Path("succeeding.txt").read_text() == "success"


def test_keep_going_flag_skips_downstream(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--keep-going skips stages downstream of failed stage."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["first.txt"])
        def first() -> None:
            raise RuntimeError("First failed")

        @stage(deps=["first.txt"], outs=["second.txt"])
        def second() -> None:
            pathlib.Path("second.txt").write_text("should not run")

        @stage(deps=["input.txt"], outs=["independent.txt"])
        def independent() -> None:
            pathlib.Path("independent.txt").write_text("runs fine")

        result = runner.invoke(cli.cli, ["run", "--keep-going"])

        assert result.exit_code == 0
        assert "first: failed" in result.output
        assert "second: skipped" in result.output
        assert "upstream" in result.output  # Reason should mention upstream failed
        assert "independent: ran" in result.output


def test_keep_going_short_flag(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """-k short flag works the same as --keep-going."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["failing.txt"])
        def failing() -> None:
            raise RuntimeError("Failure")

        @stage(deps=["input.txt"], outs=["succeeding.txt"])
        def succeeding() -> None:
            pathlib.Path("succeeding.txt").write_text("ok")

        result = runner.invoke(cli.cli, ["run", "-k"])

        assert result.exit_code == 0
        assert "failing: failed" in result.output
        assert "succeeding: ran" in result.output


def test_without_keep_going_stops_on_failure(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """Default behavior stops pipeline on first failure (downstream stages skipped)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        # Use dependent stages to test deterministically:
        # failing runs first, downstream depends on its output
        @stage(deps=["input.txt"], outs=["failing.txt"])
        def failing() -> None:
            raise RuntimeError("Intentional failure")

        @stage(deps=["failing.txt"], outs=["downstream.txt"])
        def downstream() -> None:
            pathlib.Path("downstream.txt").write_text("ran")

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code == 0
        assert "failing: failed" in result.output
        # Without --keep-going, downstream stages are skipped due to upstream failure
        assert "downstream: skipped" in result.output
        assert not pathlib.Path("downstream.txt").exists()


def test_keep_going_flag_shown_in_help(runner: CliRunner) -> None:
    """--keep-going flag is documented in help."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--keep-going" in result.output
    assert "-k" in result.output


def test_keep_going_with_json_output(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--keep-going works with --json output mode."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["failing.txt"])
        def failing() -> None:
            raise RuntimeError("Failure")

        @stage(deps=["input.txt"], outs=["succeeding.txt"])
        def succeeding() -> None:
            pathlib.Path("succeeding.txt").write_text("ok")

        result = runner.invoke(cli.cli, ["run", "--keep-going", "--json"])

        assert result.exit_code == 0
        # Parse JSONL output - look for the execution result event
        lines = result.output.strip().split("\n")
        events = [json.loads(line) for line in lines if line.strip()]

        # Should have both stage completions
        stage_complete_events = [e for e in events if e.get("type") == "stage_complete"]
        assert len(stage_complete_events) == 2

        statuses = {e["stage"]: e["status"] for e in stage_complete_events}
        assert statuses["failing"] == "failed"
        assert statuses["succeeding"] == "ran"


def test_keep_going_with_dry_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--keep-going is accepted with --dry-run (flag is no-op since nothing executes)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("processed")

        result = runner.invoke(cli.cli, ["run", "--keep-going", "--dry-run"])

        assert result.exit_code == 0
        # Dry run shows what would run without executing
        assert "would run" in result.output.lower() or "Would run" in result.output
        # The output file should NOT exist (dry run doesn't execute)
        assert not pathlib.Path("output.txt").exists()


def test_keep_going_with_dry_run_json(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--keep-going works with --dry-run --json combination."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("processed")

        result = runner.invoke(cli.cli, ["run", "--keep-going", "--dry-run", "--json"])

        assert result.exit_code == 0
        # Should produce valid JSON output
        output = json.loads(result.output)
        assert "stages" in output
        # The stage should be listed as "would_run"
        assert output["stages"]["process"]["would_run"] is True
