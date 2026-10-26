"""Tests for pivot commit command."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli, compose, executor, types
from pivot.storage import cache, lock

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def process(input_file: str) -> str:
    return "done"


@compose.stage
def stage_a() -> str:
    return "output a"


_OUTPUT_REL = pathlib.Path("data", "test", "process.txt")


def _register_process(pipeline: compose.Pipeline) -> None:
    raw_dir = pipeline.root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text("data")
    with pipeline:
        process(pipeline.input("input.txt", t=str))


def _process_lock(pipeline: compose.Pipeline) -> lock.StageLock:
    return lock.StageLock("test/process", lock.get_stages_dir(pipeline.state_dir))


def _cache_file_count(pipeline: compose.Pipeline) -> int:
    files_dir = pipeline.state_dir / "cache" / "files"
    if not files_dir.exists():
        return 0
    return sum(1 for f in files_dir.glob("*/*") if cache.is_valid_hash(f.parent.name + f.name))


# =============================================================================
# Empty Pipeline
# =============================================================================


def test_commit_empty_pipeline(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """commit command with no stages reports nothing to commit."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(
            "from pivot.compose import Pipeline\n\nwith Pipeline('test') as pipeline:\n    pass\n"
        )
        result = runner.invoke(cli.cli, ["commit"])

        assert result.exit_code == 0, result.output
        assert "Nothing to commit" in result.output


# =============================================================================
# Commit Stale Stages (no args)
# =============================================================================


def test_commit_no_args_commits_stale_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """commit with no args commits stages whose output exists but has no lock."""
    _register_process(mock_discovery)

    # Create output on disk manually (simulating a --no-commit run)
    output_path = mock_discovery.root / _OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("done")

    result = runner.invoke(cli.cli, ["commit"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output
    assert "test/process" in result.output

    # Verify production lock was written with output hashes
    stage_lock = _process_lock(mock_discovery)
    assert stage_lock.path.exists(), "Production lock should be written"
    lock_data = stage_lock.read()
    assert lock_data is not None, "Lock data should be readable"
    assert list(lock_data["output_hashes"]) == [types.ArtifactIdentity("test/process", None)], (
        "Lock should record exactly the stage's single output"
    )
    out_hash = next(iter(lock_data["output_hashes"].values()))
    assert out_hash["hash"], "Output hash should not be empty"


# =============================================================================
# Commit Specific Stage (unconditional)
# =============================================================================


def test_commit_specific_stage_unconditionally_commits(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """commit <stage> re-commits even when lock already matches."""
    _register_process(mock_discovery)

    # Run normally to create lock
    executor.run(pipeline=mock_discovery)

    stage_lock = _process_lock(mock_discovery)
    assert stage_lock.path.exists(), "Lock should exist after normal run"

    # Explicit commit <stage> should re-commit even though lock matches
    result = runner.invoke(cli.cli, ["commit", "test/process"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output
    assert "test/process" in result.output

    # Verify lock was actually re-written (not just left from previous run)
    lock_data = stage_lock.read()
    assert lock_data is not None, "Lock data should be readable after re-commit"
    assert lock_data["output_hashes"], "Lock should have output hashes after re-commit"


def test_commit_bare_stage_name_resolves(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """commit accepts a bare stage name in a single-pipeline project (names.py design)."""
    _register_process(mock_discovery)
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["commit", "process"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output


# =============================================================================
# Commit Skips Unchanged Stages (no args)
# =============================================================================


def test_commit_skips_unchanged_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """commit with no args skips stages whose lock already matches."""
    _register_process(mock_discovery)

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
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """commit <stage> exits non-zero when output file doesn't exist on disk."""
    _register_process(mock_discovery)

    # Don't create the output on disk - it's missing

    result = runner.invoke(cli.cli, ["commit", "test/process"])

    assert result.exit_code == 1, f"Expected exit code 1, got: {result.exit_code}\n{result.output}"
    assert "Failed 1 stage(s)" in result.output

    # Verify no lock was written
    assert not _process_lock(mock_discovery).path.exists(), (
        "Lock should NOT be written when output is missing"
    )


# =============================================================================
# Commit After --no-commit Run
# =============================================================================


def test_commit_after_no_commit_run_writes_lock(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Run with --no-commit, then pivot commit writes the lock."""
    _register_process(mock_discovery)

    # Run with --no-commit
    executor.run(pipeline=mock_discovery, no_commit=True)

    # Output should exist on disk
    assert (mock_discovery.root / _OUTPUT_REL).exists(), "Output should exist after --no-commit run"

    # Production lock should NOT exist
    stage_lock = _process_lock(mock_discovery)
    assert not stage_lock.path.exists(), "Lock should NOT exist after --no-commit run"

    # Now run pivot commit
    result = runner.invoke(cli.cli, ["commit"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output
    assert "test/process" in result.output

    # Verify production lock was written
    assert stage_lock.path.exists(), "Lock should be written after commit"
    lock_data = stage_lock.read()
    assert lock_data is not None, "Lock data should be readable"
    assert lock_data["output_hashes"], "Lock should have output hashes"


def test_commit_after_no_commit_run_caches_outputs(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Run with --no-commit, then pivot commit caches outputs."""
    _register_process(mock_discovery)

    # Run with --no-commit (--no-commit skips the cache)
    executor.run(pipeline=mock_discovery, no_commit=True)
    cache_file_count_before = _cache_file_count(mock_discovery)

    result = runner.invoke(cli.cli, ["commit"])
    assert result.exit_code == 0, f"Failed: {result.output}"

    # Verify cache now has files (commit caches outputs)
    assert _cache_file_count(mock_discovery) > cache_file_count_before, (
        "Cache should have files after commit"
    )


# =============================================================================
# Commit After --no-commit: subsequent run skips
# =============================================================================


def test_commit_after_no_commit_makes_subsequent_run_skip(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """After --no-commit + commit, a normal run should skip (unchanged)."""
    _register_process(mock_discovery)

    # Run with --no-commit
    executor.run(pipeline=mock_discovery, no_commit=True)

    # Commit
    result = runner.invoke(cli.cli, ["commit"])
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Committed 1 stage(s)" in result.output

    # Now run normally - should skip because commit recorded the state
    results = executor.run(pipeline=mock_discovery)
    assert results["test/process"]["status"] == "cached", (
        f"Stage should skip after commit, got: {results['test/process']}"
    )


# =============================================================================
# Unknown Stage Error
# =============================================================================


def test_commit_unknown_stage_errors(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """commit with unknown stage name errors with helpful message."""
    with mock_discovery:
        stage_a()

    result = runner.invoke(cli.cli, ["commit", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "nonexistent_stage" in result.output
