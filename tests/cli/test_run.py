"""Tests for pivot run CLI command (single-stage executor)."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import click.testing
    import pytest

    from pivot.pipeline.pipeline import Pipeline

# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _StageBOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _StageCOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("c.txt", loaders.PathOnly())]


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_stage_a() -> _StageAOutputs:
    pathlib.Path("a.txt").write_text("a")
    return _StageAOutputs(output=pathlib.Path("a.txt"))


def _helper_stage_b(
    dep: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _StageBOutputs:
    _ = dep
    pathlib.Path("b.txt").write_text("b")
    return _StageBOutputs(output=pathlib.Path("b.txt"))


def _helper_stage_c(
    dep: Annotated[pathlib.Path, outputs.Dep("b.txt", loaders.PathOnly())],
) -> _StageCOutputs:
    _ = dep
    pathlib.Path("c.txt").write_text("c")
    return _StageCOutputs(output=pathlib.Path("c.txt"))


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


class _FailingOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("failing.txt", loaders.PathOnly())]


def _helper_failing_stage() -> _FailingOutputs:
    raise RuntimeError("Intentional failure")


# =============================================================================
# Basic Command Tests
# =============================================================================


def test_run_requires_stage_argument(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run without arguments errors with usage message."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run"])

    assert result.exit_code != 0
    assert "STAGES" in result.output


def test_run_executes_single_stage(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run STAGE executes only that stage."""
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    result = runner.invoke(cli.cli, ["run", "stage_a"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (tmp_path / "a.txt").exists()
    # stage_b should NOT run (no dependency resolution)
    assert not (tmp_path / "b.txt").exists()


def test_run_executes_multiple_stages_in_order(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run STAGE1 STAGE2 executes stages in specified order."""
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    result = runner.invoke(cli.cli, ["run", "stage_a", "stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").exists()


def test_run_does_not_resolve_dependencies(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run STAGE does not automatically run dependencies."""
    # Create the dependency file manually
    (tmp_path / "a.txt").write_text("manual")
    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    # Run only stage_b - should NOT run stage_a
    result = runner.invoke(cli.cli, ["run", "stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # stage_a should not have been re-run (file still has "manual" content)
    assert (tmp_path / "a.txt").read_text() == "manual"
    assert (tmp_path / "b.txt").exists()


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_run_unknown_stage_errors(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run with unknown stage name errors with helpful message."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "nonexistent_stage" in result.output


def test_run_default_keeps_going_continues_after_failure(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run defaults to keep-going mode - continues to next stage after failure."""
    register_test_stage(_helper_failing_stage, name="failing")
    register_test_stage(_helper_stage_a, name="stage_a")

    # Run failing then stage_a - default keep-going should continue to stage_a
    result = runner.invoke(cli.cli, ["run", "failing", "stage_a"])

    # Keep-going mode returns 0 on completion; stage_a should have run
    assert result.exit_code == 0
    assert "failing: FAILED" in result.output
    assert (tmp_path / "a.txt").exists(), "stage_a should run despite failing stage"


def test_run_fail_fast_option_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --fail-fast option is accepted and affects execution mode."""
    register_test_stage(_helper_stage_a, name="stage_a")

    # Verify --fail-fast is accepted without error
    result = runner.invoke(cli.cli, ["run", "--fail-fast", "stage_a"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (tmp_path / "a.txt").exists()


# =============================================================================
# Force Mode Tests
# =============================================================================


def test_run_force_reruns_cached_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --force re-runs stages even if cached."""
    register_test_stage(_helper_stage_a, name="stage_a")

    # First run
    result = runner.invoke(cli.cli, ["run", "stage_a"])
    assert result.exit_code == 0

    # Second run with force - should re-run (not skip)
    result = runner.invoke(cli.cli, ["run", "--force", "stage_a"])
    assert result.exit_code == 0


# =============================================================================
# Option Validation Tests
# =============================================================================


def test_run_tui_log_cannot_use_with_json(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --tui-log cannot be used with --json."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--tui-log", "test.log", "--json", "stage_a"])

    assert result.exit_code != 0
    assert "--tui-log" in result.output
    assert "--json" in result.output


def test_run_tui_and_json_mutually_exclusive(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui and --json are mutually exclusive."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--tui", "--json", "stage_a"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_run_tui_log_requires_tui(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui-log requires --tui flag."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["run", "--tui-log", "log.jsonl", "process"])

    assert result.exit_code != 0
    assert "--tui-log requires --tui" in result.output


def test_run_help_includes_tui_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """--tui flag appears in help text."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".pivot").mkdir()
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--tui" in result.output
    # Help text is case-sensitive for "TUI"
    assert "TUI display" in result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_run_json_output_streams_jsonl(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --json streams JSONL output."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--json", "stage_a"])

    assert result.exit_code == 0
    # Check that output is valid JSONL (multiple JSON lines)
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise


def test_run_json_flag_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json flag should work without --tui (plain is now default)."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["run", "--json", "process"])

    assert result.exit_code == 0
    # JSONL output should start with schema version
    assert '"type": "schema_version"' in result.output


def test_run_uses_plain_mode_by_default(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain text output is the default (no --tui flag needed)."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Run without any display flags
    result = runner.invoke(cli.cli, ["run", "process"])

    assert result.exit_code == 0
    # Plain mode shows stage status in simple text format
    # (not JSONL format which has "type":)
    assert '"type":' not in result.output
    # Should contain stage name in output
    assert "process" in result.output.lower()


# =============================================================================
# Cache Options Tests
# =============================================================================


def test_run_no_commit_option_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --no-commit option is accepted."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--no-commit", "stage_a"])

    assert result.exit_code == 0


def test_run_no_cache_option_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --no-cache option is accepted."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--no-cache", "stage_a"])

    assert result.exit_code == 0


# =============================================================================
# Removed Options Tests (should not exist on run)
# =============================================================================


def test_run_does_not_have_single_stage_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run does not have --single-stage option (it's always single-stage)."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--single-stage", "stage_a"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def test_run_does_not_have_dry_run_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run does not have --dry-run option (use repro instead)."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--dry-run", "stage_a"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def test_run_does_not_have_explain_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run does not have --explain option (use repro instead)."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--explain", "stage_a"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def test_run_does_not_have_watch_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run does not have --watch option (use repro instead)."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--watch", "stage_a"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def test_run_does_not_have_keep_going_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run does not have --keep-going option (use --fail-fast instead)."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--keep-going", "stage_a"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def test_run_does_not_have_allow_missing_option(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run does not have --allow-missing option (use repro instead)."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "--allow-missing", "stage_a"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "no such option" in result.output.lower()


def test_run_tui_with_tui_log_validation_passes(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui --tui-log passes validation (log path is writable)."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    log_path = tmp_path / "tui.jsonl"

    # This will attempt to run TUI which fails in non-TTY test environment,
    # but it should NOT fail on validation errors about --tui-log
    result = runner.invoke(cli.cli, ["run", "--tui", "--tui-log", str(log_path), "process"])

    # Should NOT have validation errors
    assert "--tui-log requires --tui" not in result.output
    assert "Cannot write to" not in result.output
    # The log file should have been created during validation (touch())
    assert log_path.exists()


# =============================================================================
# --show-output Tests
# =============================================================================


def test_run_show_output_mutually_exclusive_with_tui(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --tui are mutually exclusive."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "stage_a", "--show-output", "--tui"])

    assert result.exit_code != 0
    assert "--show-output and --tui are mutually exclusive" in result.output


def test_run_show_output_mutually_exclusive_with_json(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --json are mutually exclusive."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["run", "stage_a", "--show-output", "--json"])

    assert result.exit_code != 0
    assert "--show-output and --json are mutually exclusive" in result.output


def test_run_show_output_mutually_exclusive_with_quiet(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output and --quiet are mutually exclusive."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["--quiet", "run", "stage_a", "--show-output"])

    assert result.exit_code != 0
    assert "--show-output and --quiet are mutually exclusive" in result.output


def test_run_show_output_streams_stage_logs(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stage stdout to terminal."""
    import sys

    class _PrintOutputs(TypedDict):
        output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

    def _helper_printing_stage() -> _PrintOutputs:
        sys.stdout.write("Processing data...\n")
        sys.stdout.flush()
        pathlib.Path("output.txt").write_text("done")
        return _PrintOutputs(output=pathlib.Path("output.txt"))

    register_test_stage(_helper_printing_stage, name="printer")

    result = runner.invoke(cli.cli, ["run", "printer", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "printer" in result.output
    assert "Processing data..." in result.output


def test_run_show_output_streams_stderr(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stderr with red formatting."""
    import sys

    class _StderrOutputs(TypedDict):
        output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

    def _helper_stderr_stage() -> _StderrOutputs:
        sys.stderr.write("Warning: something happened\n")
        sys.stderr.flush()
        pathlib.Path("output.txt").write_text("done")
        return _StderrOutputs(output=pathlib.Path("output.txt"))

    register_test_stage(_helper_stderr_stage, name="warner")

    result = runner.invoke(cli.cli, ["run", "warner", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "warner" in result.output
    assert "Warning: something happened" in result.output


def test_run_without_show_output_hides_logs(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Default behavior (no --show-output) doesn't show stage logs."""
    import sys

    class _QuietOutputs(TypedDict):
        output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]

    def _helper_quiet_stage() -> _QuietOutputs:
        sys.stdout.write("This should not appear\n")
        sys.stdout.flush()
        pathlib.Path("output.txt").write_text("done")
        return _QuietOutputs(output=pathlib.Path("output.txt"))

    register_test_stage(_helper_quiet_stage, name="quiet")

    result = runner.invoke(cli.cli, ["run", "quiet"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "This should not appear" not in result.output
    assert "quiet" in result.output
