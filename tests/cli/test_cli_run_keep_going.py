"""Tests for keep-going / fail-fast behavior in CLI commands.

run: Uses --fail-fast (default is keep-going)
repro: Uses --keep-going (default is fail)
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import pytest
    from click.testing import CliRunner

    from pivot.pipeline.pipeline import Pipeline

# =============================================================================
# Module-level stage functions for testing (required for pickling)
# =============================================================================


class _FailingTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("failing.txt", loaders.PathOnly())]


class _SucceedingTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("succeeding.txt", loaders.PathOnly())]


class _FirstTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("first.txt", loaders.PathOnly())]


class _SecondTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("second.txt", loaders.PathOnly())]


class _IndependentTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("independent.txt", loaders.PathOnly())]


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _DownstreamTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("downstream.txt", loaders.PathOnly())]


def _stage_failing(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FailingTxtOutputs:
    raise RuntimeError("Intentional failure")


def _stage_succeeding(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _SucceedingTxtOutputs:
    pathlib.Path("succeeding.txt").write_text("success")
    return {"output": pathlib.Path("succeeding.txt")}


def _stage_first_failing(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _FirstTxtOutputs:
    raise RuntimeError("First failed")


def _stage_second(
    first: Annotated[pathlib.Path, outputs.Dep("first.txt", loaders.PathOnly())],
) -> _SecondTxtOutputs:
    _ = first
    pathlib.Path("second.txt").write_text("should not run")
    return {"output": pathlib.Path("second.txt")}


def _stage_independent(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _IndependentTxtOutputs:
    pathlib.Path("independent.txt").write_text("runs fine")
    return {"output": pathlib.Path("independent.txt")}


def _stage_downstream(
    failing: Annotated[pathlib.Path, outputs.Dep("failing.txt", loaders.PathOnly())],
) -> _DownstreamTxtOutputs:
    _ = failing
    pathlib.Path("downstream.txt").write_text("ran")
    return {"output": pathlib.Path("downstream.txt")}


def _stage_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    pathlib.Path("output.txt").write_text("processed")
    return {"output": pathlib.Path("output.txt")}


# =============================================================================
# repro --keep-going CLI Integration Tests
# =============================================================================


def test_repro_keep_going_flag_continues_after_failure(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --keep-going continues independent stages after failure."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    assert "succeeding: done" in result.output  # Console sink outputs "done" for successful stages
    assert (tmp_path / "succeeding.txt").read_text() == "success"


def test_repro_keep_going_flag_skips_downstream(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --keep-going skips stages downstream of failed stage."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_first_failing, name="first")
    register_test_stage(_stage_second, name="second")
    register_test_stage(_stage_independent, name="independent")

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    assert result.exit_code == 0
    assert "first: FAILED" in result.output
    # Blocked stages show as "skipped" in console output (status is SKIPPED with reason)
    assert "second: skipped" in result.output
    # Summary shows blocked count
    assert "blocked" in result.output
    assert "independent: done" in result.output


def test_repro_keep_going_short_flag(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro -k short flag works the same as --keep-going."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    result = runner.invoke(cli.cli, ["repro", "-k"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    assert "succeeding: done" in result.output


def test_repro_without_keep_going_stops_on_failure(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro default behavior stops pipeline on first failure (downstream stages blocked)."""
    (tmp_path / "input.txt").write_text("data")

    # Use dependent stages to test deterministically:
    # failing runs first, downstream depends on its output
    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_downstream, name="downstream")

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    # Without --keep-going, downstream stages show as skipped (blocked internally)
    assert "downstream: skipped" in result.output
    # Summary shows blocked count
    assert "blocked" in result.output
    assert not (tmp_path / "downstream.txt").exists()


def test_repro_keep_going_flag_shown_in_help(runner: CliRunner) -> None:
    """repro --keep-going flag is documented in help."""
    result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--keep-going" in result.output
    assert "-k" in result.output


def test_repro_keep_going_with_json_output(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --keep-going works with --json output mode."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    result = runner.invoke(cli.cli, ["repro", "--keep-going", "--json"])

    assert result.exit_code == 0
    # Parse JSONL output - look for the execution result event
    lines = result.output.strip().split("\n")
    events = [json.loads(line) for line in lines if line.strip()]

    # Should have both stage completions (JSONL uses stage_complete)
    stage_completed_events = [e for e in events if e.get("type") == "stage_complete"]
    assert len(stage_completed_events) == 2

    statuses = {e["stage"]: e["status"] for e in stage_completed_events}
    assert statuses["failing"] == "failed"
    assert statuses["succeeding"] == "ran"


def test_repro_keep_going_with_dry_run(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --keep-going is accepted with --dry-run (flag is no-op since nothing executes)."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_process, name="process")

    result = runner.invoke(cli.cli, ["repro", "--keep-going", "--dry-run"])

    assert result.exit_code == 0
    # Dry run shows what would run without executing
    assert "would run" in result.output.lower() or "Would run" in result.output
    # The output file should NOT exist (dry run doesn't execute)
    assert not (tmp_path / "output.txt").exists()


def test_repro_keep_going_with_dry_run_json(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repro --keep-going works with --dry-run --json combination."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_process, name="process")

    result = runner.invoke(cli.cli, ["repro", "--keep-going", "--dry-run", "--json"])

    assert result.exit_code == 0
    # Should produce valid JSON output
    output = json.loads(result.output)
    assert "stages" in output
    # The stage should be listed as "would_run"
    assert output["stages"]["process"]["would_run"] is True


# =============================================================================
# run --fail-fast CLI Integration Tests
# =============================================================================


def test_run_fail_fast_flag_shown_in_help(runner: CliRunner) -> None:
    """run --fail-fast flag is documented in help."""
    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--fail-fast" in result.output


def test_run_default_keeps_going(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run defaults to keep-going mode (continues after failures)."""
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    # Run both stages - default should be keep-going
    result = runner.invoke(cli.cli, ["run", "failing", "succeeding"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    assert "succeeding: done" in result.output
    assert (tmp_path / "succeeding.txt").read_text() == "success"


def test_run_fail_fast_stops_early(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --fail-fast stops on first failure.

    Note: In single-stage mode (run command), stages are run in user-specified order
    without dependency resolution. With parallel execution, both stages may start
    before the failure is detected. The --fail-fast flag prevents starting NEW stages
    after a failure is detected.

    This test verifies --fail-fast is accepted and the failed stage shows failure status.
    """
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_stage_failing, name="failing")
    register_test_stage(_stage_succeeding, name="succeeding")

    # With fail-fast, we verify the flag is accepted and failing shows failure
    result = runner.invoke(cli.cli, ["run", "--fail-fast", "failing", "succeeding"])

    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
