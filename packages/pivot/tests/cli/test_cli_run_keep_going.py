"""Tests for keep-going / fail-fast behavior in CLI commands.

Both run and repro default to fail-fast.
Use --keep-going / -k to continue after failures.

Any stage failure makes run/repro exit 1 (#452 commit 378abd0, "return
non-zero exit on repro/run stage failure").
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pivot import cli, compose

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner

# =============================================================================
# Module-level stage functions for testing (required for pickling)
# =============================================================================


@compose.stage
def failing(input_file: str) -> str:
    raise RuntimeError("Intentional failure")


@compose.stage
def succeeding(input_file: str) -> str:
    return "success"


@compose.stage
def first(input_file: str) -> str:
    raise RuntimeError("First failed")


@compose.stage
def second(first_output: str) -> str:
    return "should not run"


@compose.stage
def independent(input_file: str) -> str:
    return "runs fine"


@compose.stage
def downstream(failing_output: str) -> str:
    return "ran"


@compose.stage
def process(input_file: str) -> str:
    return "processed"


def _output(pipeline: compose.Pipeline, stage_name: str) -> pathlib.Path:
    return pipeline.root / "data" / "test" / f"{stage_name}.txt"


def _input(pipeline: compose.Pipeline) -> str:
    raw_dir = pipeline.root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text("data")
    return pipeline.input("input.txt", t=str)


def _register_failing_and_succeeding(pipeline: compose.Pipeline) -> None:
    with pipeline:
        raw = _input(pipeline)
        failing(raw)
        succeeding(raw)


def _register_failing_and_downstream(pipeline: compose.Pipeline) -> None:
    with pipeline:
        downstream(failing(_input(pipeline)))


def _register_process(pipeline: compose.Pipeline) -> None:
    with pipeline:
        process(_input(pipeline))


# =============================================================================
# repro --keep-going CLI Integration Tests
# =============================================================================


def test_repro_keep_going_flag_continues_after_failure(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --keep-going continues independent stages after failure."""
    _register_failing_and_succeeding(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output
    assert "test/succeeding" in result.output and "done" in result.output
    assert _output(mock_discovery, "succeeding").read_text() == "success"


def test_repro_keep_going_flag_skips_downstream(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --keep-going skips stages downstream of failed stage."""
    with mock_discovery:
        raw = _input(mock_discovery)
        second(first(raw))
        independent(raw)

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    assert result.exit_code == 1, result.output
    assert "test/first" in result.output and "FAILED" in result.output
    # Blocked stages now show as "blocked" in console output
    assert "test/second" in result.output and "blocked" in result.output
    assert "test/independent" in result.output and "done" in result.output
    assert not _output(mock_discovery, "second").exists()


def test_repro_keep_going_short_flag(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro -k short flag works the same as --keep-going."""
    _register_failing_and_succeeding(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "-k"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output
    assert "test/succeeding" in result.output and "done" in result.output


def test_repro_without_keep_going_stops_on_failure(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro default behavior stops pipeline on first failure (downstream stages blocked)."""
    # Dependent stages make this deterministic: downstream consumes failing's output
    _register_failing_and_downstream(mock_discovery)

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output
    # Without --keep-going, downstream stages show as blocked
    assert "test/downstream" in result.output and "blocked" in result.output
    assert not _output(mock_discovery, "downstream").exists()


def test_repro_keep_going_flag_shown_in_help(runner: CliRunner) -> None:
    """repro --keep-going flag is documented in help."""
    result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--keep-going" in result.output
    assert "-k" in result.output


def test_repro_keep_going_with_json_output(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --keep-going works with --json output mode."""
    _register_failing_and_succeeding(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--keep-going", "--json"])

    assert result.exit_code == 1, result.output
    # Parse JSONL output - look for the execution result event
    lines = result.output.strip().split("\n")
    events = [json.loads(line) for line in lines if line.strip()]

    # Should have both stage completions (JSONL uses stage_complete)
    stage_completed_events = [e for e in events if e["type"] == "stage_complete"]
    assert len(stage_completed_events) == 2

    statuses = {e["stage"]: e["status"] for e in stage_completed_events}
    assert statuses["test/failing"] == "failed"
    assert statuses["test/succeeding"] == "ran"


def test_repro_keep_going_with_dry_run(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --keep-going is accepted with --dry-run (flag is no-op since nothing executes)."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--keep-going", "--dry-run"])

    assert result.exit_code == 0, result.output
    # Dry run shows what would run without executing
    assert "would run" in result.output.lower()
    # The output file should NOT exist (dry run doesn't execute)
    assert not _output(mock_discovery, "process").exists()


def test_repro_keep_going_with_dry_run_json(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --keep-going works with --dry-run --json combination."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--keep-going", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    # Should produce valid JSON output
    output = json.loads(result.output)
    assert "stages" in output
    # The stage should be listed as "would_run"
    assert output["stages"]["test/process"]["would_run"] is True


# =============================================================================
# run --fail-fast CLI Integration Tests
# =============================================================================


def test_run_fail_fast_flag_shown_in_help(runner: CliRunner) -> None:
    """run --fail-fast flag is documented in help."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--fail-fast" in result.output


def test_run_default_fails_fast(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run defaults to fail-fast mode (stops on first failure)."""
    _register_failing_and_succeeding(mock_discovery)

    # Run both stages - default should be fail-fast
    result = runner.invoke(cli.cli, ["run", "test/failing", "test/succeeding"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output


def test_run_fail_fast_stops_early(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run --fail-fast prevents subsequent stages from executing after a failure."""
    _register_failing_and_succeeding(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--fail-fast", "test/failing", "test/succeeding"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output


def test_run_keep_going_continues_after_failure(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run --keep-going continues after failures."""
    _register_failing_and_succeeding(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--keep-going", "test/failing", "test/succeeding"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output
    assert "test/succeeding" in result.output and "done" in result.output
    assert _output(mock_discovery, "succeeding").read_text() == "success"


def test_run_keep_going_short_flag(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run -k short flag works the same as --keep-going."""
    _register_failing_and_succeeding(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "-k", "test/failing", "test/succeeding"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output
    assert "test/succeeding" in result.output and "done" in result.output


def test_run_keep_going_flag_shown_in_help(runner: CliRunner) -> None:
    """run --keep-going flag is documented in help."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--keep-going" in result.output
    assert "-k" in result.output


def test_repro_fail_fast_flag_shown_in_help(runner: CliRunner) -> None:
    """repro --fail-fast flag is documented in help."""
    result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--fail-fast" in result.output


def test_repro_fail_fast_stops_on_failure(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --fail-fast stops pipeline on first failure (same as default)."""
    _register_failing_and_downstream(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--fail-fast"])

    assert result.exit_code == 1, result.output
    assert "test/failing" in result.output and "FAILED" in result.output
    assert "test/downstream" in result.output and "blocked" in result.output
    assert not _output(mock_discovery, "downstream").exists()


def test_run_fail_fast_and_keep_going_mutually_exclusive(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """run --fail-fast and --keep-going are mutually exclusive."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["run", "--fail-fast", "--keep-going", "test/process"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_repro_fail_fast_and_keep_going_mutually_exclusive(
    mock_discovery: compose.Pipeline,
    runner: CliRunner,
) -> None:
    """repro --fail-fast and --keep-going are mutually exclusive."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--fail-fast", "--keep-going"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()
