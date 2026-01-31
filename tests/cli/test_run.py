from __future__ import annotations

import contextlib
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs, project
from pivot.storage import cache, track

if TYPE_CHECKING:
    from collections.abc import Generator

    import click.testing
    import filelock
    import pytest
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


# =============================================================================
# run --dry-run --allow-missing Tests
# =============================================================================


def test_run_dry_run_allow_missing_uses_pvt_hash(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --dry-run --allow-missing uses .pvt hash when dep file is missing."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    # Create and run
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")
    executor.run(pipeline=mock_discovery)

    # Track input
    input_hash = cache.hash_file(pathlib.Path("input.txt"))
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

    # Delete input (simulating CI)
    pathlib.Path("input.txt").unlink()

    result = runner.invoke(cli.cli, ["run", "--dry-run", "--allow-missing"])

    # Should show "would skip" not "Missing deps"
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert "would skip" in result.output.lower(), f"Got: {result.output}"


def test_run_dry_run_explain_allow_missing_uses_pvt_hash(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --dry-run --explain --allow-missing uses .pvt hash when dep file is missing."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    # Create and run
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")
    executor.run(pipeline=mock_discovery)

    # Track input
    input_hash = cache.hash_file(pathlib.Path("input.txt"))
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(pathlib.Path("input.txt.pvt"), pvt_data)

    # Delete input (simulating CI)
    pathlib.Path("input.txt").unlink()

    result = runner.invoke(cli.cli, ["run", "--dry-run", "--explain", "--allow-missing"])

    # Should NOT show error about missing deps
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected success, got: {result.output}"


def test_run_allow_missing_requires_dry_run(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --allow-missing without --dry-run errors."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["run", "--allow-missing"])

    assert result.exit_code != 0
    assert "--allow-missing" in result.output
    assert "--dry-run" in result.output


# =============================================================================
# --tui and --json Flag Tests
# =============================================================================


def test_run_tui_and_json_mutually_exclusive(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui and --json are mutually exclusive."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    register_test_stage(_helper_process, name="process")
    pathlib.Path("input.txt").write_text("data")

    result = runner.invoke(cli.cli, ["run", "--tui", "--json"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_run_tui_log_requires_tui(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui-log requires --tui flag."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    register_test_stage(_helper_process, name="process")
    pathlib.Path("input.txt").write_text("data")

    result = runner.invoke(cli.cli, ["run", "--tui-log", "log.jsonl"])

    assert result.exit_code != 0
    assert "--tui-log requires --tui" in result.output


def test_run_serve_requires_watch(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--serve requires --watch flag."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    register_test_stage(_helper_process, name="process")
    pathlib.Path("input.txt").write_text("data")

    result = runner.invoke(cli.cli, ["run", "--serve"])

    assert result.exit_code != 0
    assert "--serve requires --watch" in result.output


def test_run_help_includes_tui_flag(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui flag appears in help text."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    result = runner.invoke(cli.cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--tui" in result.output
    # Help text is case-sensitive for "TUI"
    assert "TUI display" in result.output


def test_run_json_flag_accepted(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json flag should work without --tui (plain is now default)."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["run", "--json"])

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
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Run without any display flags
    result = runner.invoke(cli.cli, ["run"])

    assert result.exit_code == 0
    # Plain mode shows stage status in simple text format
    # (not JSONL format which has "type":)
    assert '"type":' not in result.output
    # Should contain stage name in output
    assert "process" in result.output.lower()


def test_run_tui_with_tui_log_validation_passes(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--tui --tui-log passes validation (log path is writable)."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    log_path = tmp_path / "tui.jsonl"

    # This will attempt to run TUI which fails in non-TTY test environment,
    # but it should NOT fail on validation errors about --tui-log
    result = runner.invoke(cli.cli, ["run", "--tui", "--tui-log", str(log_path)])

    # Should NOT have validation errors
    assert "--tui-log requires --tui" not in result.output
    assert "Cannot write to" not in result.output
    # The log file should have been created during validation (touch())
    assert log_path.exists()


# =============================================================================
# Pipeline Discovery Tests
# =============================================================================


def test_cli_run_discovers_pipeline(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """pivot run should discover and use Pipeline from pipeline.py."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)

    # Create pipeline.py that defines a Pipeline
    pipeline_code = """
import pathlib
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test", root=pathlib.Path(__file__).parent)

def my_stage() -> None:
    pass

pipeline.register(my_stage, name="my_stage")
"""
    (tmp_path / "pipeline.py").write_text(pipeline_code)
    (tmp_path / ".pivot").mkdir()

    result = runner.invoke(cli.cli, ["run", "--dry-run"])

    assert result.exit_code == 0, f"Expected success, got: {result.output}"
    assert "my_stage" in result.output


def test_cli_run_no_pipeline_found_error(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """pivot run should error if no pipeline found (non-informational mode)."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)

    # Create .pivot dir but no pipeline.py or pivot.yaml
    (tmp_path / ".pivot").mkdir()

    # Regular run (not --dry-run or --json) should error
    result = runner.invoke(cli.cli, ["run"])

    assert result.exit_code != 0
    assert "pipeline" in result.output.lower() or "no pipeline" in result.output


def test_run_dry_run_allow_missing_no_pvt_shows_would_run(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run --dry-run --allow-missing shows stages without .pvt as would run.

    Critical behavioral test: --allow-missing with --dry-run is for CI scenarios
    where deps don't exist yet. It should show what would run (validation skipped),
    not error. This is different from regular verify which requires all files.
    """
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()

    # Register stage but don't create the dep file OR .pvt file
    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["run", "--dry-run", "--allow-missing"])

    # Should succeed and show stage as "would run" (not error)
    assert result.exit_code == 0, f"Should succeed with --allow-missing, got: {result.output}"
    assert "process" in result.output
    assert "would run" in result.output.lower()
    # Reason should indicate no previous run (not missing deps)
    assert "No previous run" in result.output or "no previous" in result.output.lower()


# =============================================================================
# --no-commit Lock Acquisition Tests
# =============================================================================


def test_run_no_commit_acquires_pending_state_lock(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """run --no-commit acquires pending_state_lock during execution."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Mock pending_state_lock to track calls
    from pivot.storage import project_lock

    lock_context_entered = False
    lock_context_exited = False

    original_pending_state_lock = project_lock.pending_state_lock

    @contextlib.contextmanager
    def mock_pending_state_lock(timeout: float = -1) -> Generator[filelock.BaseFileLock]:
        nonlocal lock_context_entered, lock_context_exited
        lock_context_entered = True
        with original_pending_state_lock(timeout=timeout) as lock:
            yield lock
        lock_context_exited = True

    mocker.patch.object(
        project_lock,
        "pending_state_lock",
        mock_pending_state_lock,
    )

    result = runner.invoke(cli.cli, ["run", "--no-commit"])

    assert result.exit_code == 0, f"Expected success, got: {result.output}"
    assert lock_context_entered, "pending_state_lock should have been acquired"
    assert lock_context_exited, "pending_state_lock should have been released"


def test_run_without_no_commit_does_not_acquire_pending_state_lock(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """run without --no-commit does not acquire pending_state_lock."""
    monkeypatch.chdir(tmp_path)
    pathlib.Path(".git").mkdir()
    pathlib.Path("input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Mock pending_state_lock to track calls
    from pivot.storage import project_lock

    lock_acquired = False

    original_pending_state_lock = project_lock.pending_state_lock

    @contextlib.contextmanager
    def mock_pending_state_lock(timeout: float = -1) -> Generator[filelock.BaseFileLock]:
        nonlocal lock_acquired
        lock_acquired = True
        with original_pending_state_lock(timeout=timeout) as lock:
            yield lock

    mocker.patch.object(
        project_lock,
        "pending_state_lock",
        mock_pending_state_lock,
    )

    result = runner.invoke(cli.cli, ["run"])

    assert result.exit_code == 0, f"Expected success, got: {result.output}"
    assert not lock_acquired, "pending_state_lock should NOT have been acquired without --no-commit"
