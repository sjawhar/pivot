"""Tests for pivot commit command."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import lock

if TYPE_CHECKING:
    import click.testing
    import pytest

    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _ATxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def _helper_stage_a() -> _ATxtOutputs:
    pathlib.Path("a.txt").write_text("output a")
    return _ATxtOutputs(output=pathlib.Path("a.txt"))


# =============================================================================
# Empty Pipeline
# =============================================================================


def test_commit_empty_pipeline(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """commit command with no stages reports nothing to commit."""
    from conftest import isolated_pivot_dir

    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pivot.yaml").write_text("stages: {}")
        result = runner.invoke(cli.cli, ["commit"])

        assert result.exit_code == 0
        assert "Nothing to commit" in result.output


# =============================================================================
# Commit Stale Stages (no args)
# =============================================================================


def test_commit_no_args_commits_stale_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit with no args commits stages whose output exists but has no lock."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Create output on disk manually (simulating a --no-commit run)
    (tmp_path / "output.txt").write_text("done")

    result = runner.invoke(cli.cli, ["commit"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output
    assert "process" in result.output

    # Verify production lock was written with output hashes
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    assert stage_lock.path.exists(), "Production lock should be written"
    lock_data = stage_lock.read()
    assert lock_data is not None, "Lock data should be readable"
    assert len(lock_data["output_hashes"]) == 1, "Lock should have exactly one output hash"
    out_key = next(iter(lock_data["output_hashes"]))
    assert out_key.endswith("output.txt"), (
        f"Output hash key should be for output.txt, got: {out_key}"
    )
    assert lock_data["output_hashes"][out_key]["hash"], "Output hash should not be empty"


# =============================================================================
# Commit Specific Stage (unconditional)
# =============================================================================


def test_commit_specific_stage_unconditionally_commits(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit <stage> re-commits even when lock already matches."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Run normally to create lock
    executor.run(pipeline=mock_discovery)

    # Verify lock exists after normal run and record its mtime
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    assert stage_lock.path.exists(), "Lock should exist after normal run"

    # Explicit commit <stage> should re-commit even though lock matches
    result = runner.invoke(cli.cli, ["commit", "process"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output
    assert "process" in result.output

    # Verify lock was actually re-written (not just left from previous run)
    lock_data = stage_lock.read()
    assert lock_data is not None, "Lock data should be readable after re-commit"
    assert lock_data["output_hashes"], "Lock should have output hashes after re-commit"


# =============================================================================
# Commit Skips Unchanged Stages (no args)
# =============================================================================


def test_commit_skips_unchanged_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit with no args skips stages whose lock already matches."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Run normally - creates lock and caches outputs
    executor.run(pipeline=mock_discovery)

    # Commit with no args should skip (nothing changed)
    result = runner.invoke(cli.cli, ["commit"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Nothing to commit" in result.output


# =============================================================================
# Commit Errors on Missing Output
# =============================================================================


def test_commit_errors_on_missing_output(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit <stage> exits non-zero when output file doesn't exist on disk."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Don't create output.txt on disk - it's missing

    result = runner.invoke(cli.cli, ["commit", "process"])

    assert result.exit_code == 1, f"Expected exit code 1, got: {result.exit_code}\n{result.output}"
    assert "Failed 1 stage(s)" in result.output

    # Verify no lock was written
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    assert not stage_lock.path.exists(), "Lock should NOT be written when output is missing"


# =============================================================================
# Commit After --no-commit Run
# =============================================================================


def test_commit_after_no_commit_run(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run with --no-commit, then pivot commit writes lock and caches outputs."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Run with --no-commit
    executor.run(pipeline=mock_discovery, no_commit=True)

    # Output should exist on disk
    assert (tmp_path / "output.txt").exists(), "Output should exist after --no-commit run"

    # Production lock should NOT exist
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    assert not stage_lock.path.exists(), "Lock should NOT exist after --no-commit run"

    # Cache should NOT have files yet (--no-commit skips cache)
    files_dir = state_dir / "cache" / "files"
    cache_files_before = list(files_dir.rglob("*")) if files_dir.exists() else []
    cache_file_count_before = sum(1 for f in cache_files_before if f.is_file())

    # Now run pivot commit
    result = runner.invoke(cli.cli, ["commit"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output
    assert "process" in result.output

    # Verify production lock was written
    assert stage_lock.path.exists(), "Lock should be written after commit"
    lock_data = stage_lock.read()
    assert lock_data is not None, "Lock data should be readable"
    assert lock_data["output_hashes"], "Lock should have output hashes"

    # Verify cache now has files (commit caches outputs)
    cache_files_after = [f for f in files_dir.rglob("*") if f.is_file()]
    assert len(cache_files_after) > cache_file_count_before, "Cache should have files after commit"


# =============================================================================
# Commit After --no-commit: subsequent run skips
# =============================================================================


def test_commit_after_no_commit_makes_subsequent_run_skip(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After --no-commit + commit, a normal run should skip (unchanged)."""
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")

    # Run with --no-commit
    executor.run(pipeline=mock_discovery, no_commit=True)

    # Commit
    result = runner.invoke(cli.cli, ["commit"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output

    # Now run normally - should skip because commit recorded the state
    results = executor.run(pipeline=mock_discovery)
    assert results["process"]["status"] == "skipped", (
        f"Stage should skip after commit, got: {results['process']}"
    )


# =============================================================================
# Unknown Stage Error
# =============================================================================


def test_commit_unknown_stage_errors(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit with unknown stage name errors with helpful message."""
    register_test_stage(_helper_stage_a, name="stage_a")

    result = runner.invoke(cli.cli, ["commit", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "nonexistent_stage" in result.output
