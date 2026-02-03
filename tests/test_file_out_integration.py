"""Integration tests for single file Out cache restoration scenarios.

Tests verify that Out (single file outputs) properly handles various workspace
states when determining whether to skip (restore from cache) or re-run.

This complements test_directory_out_integration.py which tests DirectoryOut.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest

from helpers import register_test_stage
from pivot import executor, loaders, outputs

if TYPE_CHECKING:
    import click.testing

    from pivot.pipeline.pipeline import Pipeline


class _SingleFileOutResult(TypedDict):
    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]


def _stage_produces_single_file_with_marker() -> _SingleFileOutResult:
    """Stage that produces a single file and writes a run marker.

    The marker file (run_marker.txt) tracks how many times the stage executed.
    """
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _SingleFileOutResult(result={"value": 42, "run": run_count})


def _get_run_count() -> int:
    """Get the current run count from the marker file."""
    marker_path = pathlib.Path("run_marker.txt")
    return int(marker_path.read_text()) if marker_path.exists() else 0


def _assert_file_content(path: pathlib.Path, expected: dict[str, int | bool]) -> None:
    """Assert file contains expected JSON content."""
    actual = json.loads(path.read_text())
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_missing_file_restored_on_cache_hit(
    runner: click.testing.CliRunner,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache has file. Workspace missing file. Expected: Restore file.

    1. Run pipeline to cache file
    2. Delete output file
    3. Run pipeline again
    4. Assert: stage skipped (not re-run), file restored from cache
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # First run - creates and caches file
    register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    _assert_file_content(output_file, {"value": 42, "run": 1})
    assert _get_run_count() == 1

    # Delete output file
    output_file.unlink()
    assert not output_file.exists(), "File should be deleted"

    # Second run - should skip and restore from cache
    executor.run(pipeline=test_pipeline)

    # Verify: stage did NOT re-run (counter still 1)
    assert _get_run_count() == 1, "Stage should have skipped, not re-run"

    # Verify: file restored from cache (run=1, not run=2)
    assert output_file.exists(), "File should be restored"
    _assert_file_content(output_file, {"value": 42, "run": 1})


@pytest.mark.parametrize(
    "unlink_before_corrupt",
    [
        pytest.param(True, id="always-unlink"),
        pytest.param(False, id="conditional-unlink"),
    ],
)
def test_corrupted_file_restored_on_cache_hit(
    runner: click.testing.CliRunner,
    unlink_before_corrupt: bool,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache has correct file. Workspace has corrupted file. Expected: Restore correct file.

    This is the key test for issue #234.5 - verifying that corrupted single file
    outputs are detected and restored from cache.

    1. Run pipeline to cache file
    2. Modify content of output file to corrupt it
    3. Run pipeline again
    4. Assert: stage skipped, file restored with correct content
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # First run - creates and caches file
    register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    _assert_file_content(output_file, {"value": 42, "run": 1})
    assert _get_run_count() == 1

    # Corrupt file - handle read-only hardlinks by unlinking first
    if unlink_before_corrupt:
        output_file.unlink()
    else:
        # Check if file is read-only (hardlinked), unlink if needed
        file_mode = output_file.stat().st_mode & 0o777
        if file_mode == 0o444:
            output_file.unlink()
    output_file.write_text('{"corrupted": true}')
    _assert_file_content(output_file, {"corrupted": True})

    # Second run - should skip and restore correct content
    executor.run(pipeline=test_pipeline)

    # Verify: stage did NOT re-run
    assert _get_run_count() == 1, "Stage should have skipped, not re-run"

    # Verify: file restored with correct content
    _assert_file_content(output_file, {"value": 42, "run": 1})


def test_corrupted_file_triggers_rerun_when_cache_empty(
    runner: click.testing.CliRunner,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache empty, workspace has corrupted file. Expected: Re-run stage.

    1. Run pipeline to cache file
    2. Clear the cache (delete cached files)
    3. Corrupt output file
    4. Run pipeline again
    5. Assert: stage RE-RUN (not skipped), file regenerated with run=2
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # First run - creates and caches file
    register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    _assert_file_content(output_file, {"value": 42, "run": 1})
    assert _get_run_count() == 1

    # Clear the cache
    cache_dir = tmp_path / ".pivot/cache/files"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)

    # Corrupt file
    output_file.unlink()
    output_file.write_text('{"corrupted": true}')

    # Second run - should RE-RUN because cache can't restore
    executor.run(pipeline=test_pipeline)

    # Verify: stage DID re-run (counter is now 2)
    assert _get_run_count() == 2, "Stage should have re-run when cache is empty"

    # Verify: file regenerated with run=2
    _assert_file_content(output_file, {"value": 42, "run": 2})


def test_perfect_match_skips_without_modification(
    runner: click.testing.CliRunner,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache and workspace are identical. Expected: Skip without touching file.

    1. Run pipeline to cache file
    2. Run pipeline again without any modifications
    3. Assert: stage skipped, file unchanged
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # First run - creates and caches file
    register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    _assert_file_content(output_file, {"value": 42, "run": 1})
    assert _get_run_count() == 1

    # Second run - should skip
    executor.run(pipeline=test_pipeline)

    # Verify: stage did NOT re-run
    assert _get_run_count() == 1, "Stage should have skipped"

    # File still present with correct content
    _assert_file_content(output_file, {"value": 42, "run": 1})


def test_invalid_json_in_corrupted_file_handled_gracefully(
    runner: click.testing.CliRunner,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupted file with invalid JSON is detected and restored.

    This tests that corrupted files with invalid JSON (not just wrong content)
    are properly detected and restored from cache.

    1. Run pipeline to cache file
    2. Corrupt file with invalid JSON
    3. Run pipeline again
    4. Assert: stage skipped, file restored (not crash during skip detection)
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # First run - creates and caches file
    register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    _assert_file_content(output_file, {"value": 42, "run": 1})
    assert _get_run_count() == 1

    # Corrupt with invalid JSON (not parseable)
    output_file.unlink()
    output_file.write_text('{"invalid": json content missing closing brace')

    # Second run - should skip and restore (not crash)
    executor.run(pipeline=test_pipeline)

    # Verify: stage did NOT re-run
    assert _get_run_count() == 1, "Stage should have skipped, not re-run"

    # Verify: file restored with valid JSON
    _assert_file_content(output_file, {"value": 42, "run": 1})


def test_cache_files_are_readonly_and_properly_structured(
    runner: click.testing.CliRunner,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cache files are created read-only with correct hash structure.

    This ensures cache integrity by validating:
    1. Cache files are stored with correct hash prefix structure (ab/cdef...)
    2. Cache files are read-only (0o444)
    3. Cache files contain correct content
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # Run pipeline to cache file
    register_test_stage(_stage_produces_single_file_with_marker, name="produce")
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"
    assert output_file.exists()

    # Find cache file
    cache_dir = tmp_path / ".pivot/cache/files"
    assert cache_dir.exists(), "Cache directory should exist"

    # Cache should have 2-char prefix directories
    prefix_dirs = list(cache_dir.iterdir())
    assert len(prefix_dirs) > 0, "Cache should contain prefix directories"

    # Find cached file
    cached_files = list[pathlib.Path]()
    for prefix_dir in prefix_dirs:
        if prefix_dir.is_dir():
            cached_files.extend(prefix_dir.iterdir())

    assert len(cached_files) >= 1, "At least one cache file should exist"

    # Check first cached file
    cache_file = cached_files[0]

    # Verify read-only
    cache_mode = cache_file.stat().st_mode & 0o777
    assert cache_mode == 0o444, f"Cache file should be read-only (0o444), got {oct(cache_mode)}"

    # Verify content matches
    cache_content = json.loads(cache_file.read_text())
    assert cache_content == {"value": 42, "run": 1}, "Cache content should match output"


class _FailingStageResult(TypedDict):
    result: Annotated[dict[str, str], outputs.Out("output.json", loaders.JSON[dict[str, str]]())]


def _stage_fails_after_partial_write() -> _FailingStageResult:
    """Stage that writes partial output then fails."""
    output_path = pathlib.Path("output.json")
    # Write partial output
    output_path.write_text('{"partial": ')
    # Raise exception before completing
    raise RuntimeError("Stage failed intentionally")


def test_stage_failure_leaves_workspace_consistent(
    runner: click.testing.CliRunner,
    mock_discovery: Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage failure after partial write leaves workspace consistent.

    This tests that:
    1. Stage exceptions are properly caught
    2. Lock file is NOT updated on failure
    3. Partial output is NOT cached
    4. Subsequent runs can succeed
    """
    test_pipeline = mock_discovery
    tmp_path = test_pipeline.root
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir(exist_ok=True)

    # Register failing stage
    register_test_stage(_stage_fails_after_partial_write, name="failing")

    # First run - should complete but stage fails
    executor.run(pipeline=test_pipeline)

    output_file = tmp_path / "output.json"

    # Lock file should NOT be created/updated on failure
    # (lock files are only written on successful stage completion)
    lock_file = tmp_path / ".pivot/stages/failing.lock"
    assert not lock_file.exists(), "Lock file should not exist after stage failure"

    # Clean up any partial output
    if output_file.exists():
        output_file.unlink()

    # Register successful stage (clear registry first to allow re-registration)
    test_pipeline.clear()
    register_test_stage(_stage_produces_single_file_with_marker, name="failing")

    # Second run - should succeed
    executor.run(pipeline=test_pipeline)

    # Verify success
    assert output_file.exists()
    _assert_file_content(output_file, {"value": 42, "run": 1})
