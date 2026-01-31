"""Integration tests for DirectoryOut cache restoration scenarios.

Tests verify that DirectoryOut properly handles various workspace states
when determining whether to skip (restore from cache) or re-run.

All tests use real temporary directories and run actual pipelines.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from helpers import register_test_stage
from pivot import executor, loaders, outputs

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Module-level TypedDicts and Stage Functions
# =============================================================================


class _DirectoryOutResult(TypedDict):
    results: Annotated[
        dict[str, dict[str, Any]], outputs.DirectoryOut("results/", loaders.JSON[dict[str, Any]]())
    ]


def _stage_produces_abc_with_marker() -> _DirectoryOutResult:
    """Stage that produces files A, B, C and writes a run marker file.

    The marker file (run_marker.txt) is written outside the DirectoryOut
    and tracks how many times the stage has executed. This works across
    process boundaries since it's file-based.
    """
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _DirectoryOutResult(
        results={
            "a.json": {"value": 1, "run": run_count},
            "b.json": {"value": 2, "run": run_count},
            "c.json": {"value": 3, "run": run_count},
        }
    )


# =============================================================================
# Test Helpers
# =============================================================================


def _get_run_count() -> int:
    """Get the current run count from the marker file."""
    marker_path = pathlib.Path("run_marker.txt")
    return int(marker_path.read_text()) if marker_path.exists() else 0


def _assert_files_exist(results_dir: pathlib.Path, expected: list[str]) -> None:
    """Assert that exactly the expected files exist in results directory."""
    actual = sorted(f.name for f in results_dir.glob("*.json"))
    assert actual == sorted(expected), f"Expected {expected}, got {actual}"


def _assert_file_content(path: pathlib.Path, expected: dict[str, Any]) -> None:
    """Assert file contains expected JSON content."""
    actual = json.loads(path.read_text())
    assert actual == expected, f"Expected {expected}, got {actual}"


# =============================================================================
# H3: Empty directory restored on cache hit
# =============================================================================


def test_h3_empty_directory_restored_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H3: Cache has A,B,C. Workspace has empty dir. Expected: Restore A,B,C.

    1. Run pipeline to cache files A,B,C
    2. Delete all files in output directory but keep the directory
    3. Run pipeline again
    4. Assert: stage skipped (not re-run), files A,B,C restored from cache
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})
        assert _get_run_count() == 1

        # Delete all files but keep directory
        for f in results_dir.glob("*.json"):
            f.unlink()
        assert results_dir.exists(), "Directory should still exist"
        assert not (results_dir / "a.json").exists(), "Files should be deleted"

        # Second run - should skip and restore from cache
        executor.run()

        # Verify: stage did NOT re-run (counter still 1)
        assert _get_run_count() == 1, "Stage should have skipped, not re-run"

        # Verify: files restored from cache (run=1, not run=2)
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


# =============================================================================
# H4: Partial directory restored on cache hit
# =============================================================================


def test_h4_partial_directory_restored_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H4: Cache has A,B,C. Workspace has only A. Expected: Restore B,C, keep A.

    1. Run pipeline to cache files A,B,C
    2. Delete B and C from output directory, keep A
    3. Run pipeline again
    4. Assert: stage skipped, files B,C restored, A still present
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Delete B and C, keep A
        (results_dir / "b.json").unlink()
        (results_dir / "c.json").unlink()
        _assert_files_exist(results_dir, ["a.json"])

        # Second run - should skip and restore missing files
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped, not re-run"

        # Verify: all files present (B, C restored from cache)
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "b.json", {"value": 2, "run": 1})


# =============================================================================
# H5: Extra files removed on cache hit restore
# =============================================================================


def test_h5_extra_files_removed_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H5: Cache has A,B,C. Workspace has A,B,C,D (extra file). Expected: Remove D.

    1. Run pipeline to cache files A,B,C
    2. Add extra file D to output directory
    3. Run pipeline again
    4. Assert: stage skipped, files A,B,C present, D removed
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Add extra file D
        (results_dir / "d.json").write_text('{"extra": true}')
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json", "d.json"])

        # Second run - should skip and remove extra file
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped, not re-run"

        # Verify: only A,B,C present (D removed)
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])


# =============================================================================
# H6: Wrong files replaced on cache hit restore
# =============================================================================


def test_h6_wrong_files_replaced_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H6: Cache has A,B,C. Workspace has A,B,X (wrong file). Expected: Restore C, remove X.

    1. Run pipeline to cache files A,B,C
    2. Delete C, add wrong file X to output directory
    3. Run pipeline again
    4. Assert: stage skipped, files A,B,C present, X removed
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Delete C, add X
        (results_dir / "c.json").unlink()
        (results_dir / "x.json").write_text('{"wrong": true}')
        _assert_files_exist(results_dir, ["a.json", "b.json", "x.json"])

        # Second run - should skip, restore C, remove X
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped, not re-run"

        # Verify: A,B,C present (C restored), X removed
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "c.json", {"value": 3, "run": 1})


# =============================================================================
# H7: Corrupted files restored on cache hit
# =============================================================================


def test_h7_corrupted_files_restored_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H7: Cache has A,B,C. Workspace has A (corrupted). Expected: Restore correct A.

    1. Run pipeline to cache files A,B,C
    2. Modify content of file A to corrupt it
    3. Run pipeline again
    4. Assert: stage skipped, file A restored with correct content
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})
        assert _get_run_count() == 1

        # Corrupt file A (unlink first - cached files are read-only hardlinks)
        (results_dir / "a.json").unlink()
        (results_dir / "a.json").write_text('{"corrupted": true}')
        _assert_file_content(results_dir / "a.json", {"corrupted": True})

        # Second run - should skip and restore correct content
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped, not re-run"

        # Verify: A restored with correct content
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


# =============================================================================
# H8: Corrupted file triggers re-run when cache is empty
# =============================================================================


def test_h8_corrupted_file_triggers_rerun_when_cache_empty(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H8: Cache empty, workspace has corrupted file. Expected: Re-run stage.

    1. Run pipeline to cache files A,B,C
    2. Clear the cache (delete cached files)
    3. Corrupt file A in output directory
    4. Run pipeline again
    5. Assert: stage RE-RUN (not skipped), files A,B,C regenerated with run=2
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})
        assert _get_run_count() == 1

        # Clear the cache (delete cached files)
        cache_dir = pathlib.Path(".pivot/cache/files")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)

        # Corrupt file A (unlink first - may be hardlinked to cache)
        (results_dir / "a.json").unlink()
        (results_dir / "a.json").write_text('{"corrupted": true}')

        # Second run - should RE-RUN because cache can't restore
        executor.run()

        # Verify: stage DID re-run (counter is now 2)
        assert _get_run_count() == 2, "Stage should have re-run when cache is empty"

        # Verify: files regenerated with run=2
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 2})


# =============================================================================
# H9: Missing directory path (not just empty) restored on cache hit
# =============================================================================


def test_h9_missing_directory_restored_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H9: Cache has A,B,C. Directory does not exist. Expected: Create dir, restore A,B,C.

    1. Run pipeline to cache files A,B,C
    2. Delete the entire output directory
    3. Run pipeline again
    4. Assert: stage skipped, directory recreated, files A,B,C restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Delete entire directory
        shutil.rmtree(results_dir)
        assert not results_dir.exists(), "Directory should be deleted"

        # Second run - should skip and restore directory + files
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped, not re-run"

        # Verify: directory and files restored
        assert results_dir.exists(), "Directory should be restored"
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


# =============================================================================
# H10: Partial cache corruption triggers re-run
# =============================================================================


def test_h10_partial_cache_corruption_triggers_rerun(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H10: Cache missing some files. Workspace has corrupted files. Expected: Re-run.

    1. Run pipeline to cache files A,B,C
    2. Delete some cached files (partial cache corruption)
    3. Corrupt workspace files
    4. Run pipeline again
    5. Assert: stage RE-RUN because cache cannot fully restore
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Partially corrupt cache (delete some hash files)
        cache_files_dir = pathlib.Path(".pivot/cache/files")
        if cache_files_dir.exists():
            # Delete half the cache files
            cache_files = list(cache_files_dir.rglob("*"))
            files_to_delete = [f for f in cache_files if f.is_file()][:2]
            for f in files_to_delete:
                f.unlink()

        # Corrupt workspace file A
        (results_dir / "a.json").unlink()
        (results_dir / "a.json").write_text('{"corrupted": true}')

        # Second run - should RE-RUN because cache can't fully restore
        executor.run()

        # Verify: stage DID re-run
        assert _get_run_count() == 2, "Stage should have re-run with corrupted cache"

        # Verify: files regenerated with run=2
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 2})


# =============================================================================
# H11: Nested subdirectories
# =============================================================================


class _NestedDirectoryOutResult(TypedDict):
    results: Annotated[
        dict[str, dict[str, Any]], outputs.DirectoryOut("results/", loaders.JSON[dict[str, Any]]())
    ]


def _stage_produces_nested_with_marker() -> _NestedDirectoryOutResult:
    """Stage that produces files in nested subdirectories."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _NestedDirectoryOutResult(
        results={
            "top.json": {"level": "top", "run": run_count},
            "sub/nested.json": {"level": "nested", "run": run_count},
            "sub/deep/deeper.json": {"level": "deeper", "run": run_count},
        }
    )


def _assert_nested_files_exist(results_dir: pathlib.Path) -> None:
    """Assert that nested files exist."""
    assert (results_dir / "top.json").exists()
    assert (results_dir / "sub" / "nested.json").exists()
    assert (results_dir / "sub" / "deep" / "deeper.json").exists()


def test_h11_nested_directories_restored_on_cache_hit(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H11: Cache has nested files. Workspace empty. Expected: Restore all nested.

    1. Run pipeline to cache nested files
    2. Delete all files in output directory
    3. Run pipeline again
    4. Assert: stage skipped, nested structure restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates nested structure
        register_test_stage(_stage_produces_nested_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_nested_files_exist(results_dir)
        assert _get_run_count() == 1

        # Delete all files but keep root directory
        shutil.rmtree(results_dir)
        results_dir.mkdir()
        assert results_dir.exists()

        # Second run - should skip and restore nested structure
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped"

        # Verify: nested structure restored
        _assert_nested_files_exist(results_dir)
        _assert_file_content(
            results_dir / "sub" / "deep" / "deeper.json", {"level": "deeper", "run": 1}
        )


def test_h11_nested_file_corrupted_restored(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H11b: Nested file corrupted. Expected: Restore correct nested file.

    1. Run pipeline to cache nested files
    2. Corrupt a deeply nested file
    3. Run pipeline again
    4. Assert: stage skipped, corrupted file restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates nested structure
        register_test_stage(_stage_produces_nested_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_nested_files_exist(results_dir)
        assert _get_run_count() == 1

        # Corrupt deeply nested file
        nested_file = results_dir / "sub" / "deep" / "deeper.json"
        nested_file.unlink()
        nested_file.write_text('{"corrupted": true}')

        # Second run - should skip and restore
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped"

        # Verify: nested file restored
        _assert_file_content(nested_file, {"level": "deeper", "run": 1})


# =============================================================================
# H12: Large directory (many files)
# =============================================================================


class _LargeDirectoryOutResult(TypedDict):
    results: Annotated[
        dict[str, dict[str, Any]], outputs.DirectoryOut("results/", loaders.JSON[dict[str, Any]]())
    ]


def _stage_produces_many_files_with_marker() -> _LargeDirectoryOutResult:
    """Stage that produces many files."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    # Create 50 files (enough to test bulk operations without being slow)
    return _LargeDirectoryOutResult(
        results={f"file_{i:03d}.json": {"index": i, "run": run_count} for i in range(50)}
    )


def test_h12_large_directory_partial_restore(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H12: Cache has 50 files. Workspace missing half. Expected: Restore missing.

    1. Run pipeline to cache many files
    2. Delete half the files
    3. Run pipeline again
    4. Assert: stage skipped, missing files restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates many files
        register_test_stage(_stage_produces_many_files_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        all_files = list(results_dir.glob("*.json"))
        assert len(all_files) == 50
        assert _get_run_count() == 1

        # Delete half the files
        for i in range(25):
            (results_dir / f"file_{i:03d}.json").unlink()
        remaining = list(results_dir.glob("*.json"))
        assert len(remaining) == 25

        # Second run - should skip and restore missing files
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped"

        # Verify: all 50 files present
        all_files = list(results_dir.glob("*.json"))
        assert len(all_files) == 50


# =============================================================================
# H13: Perfect match skips without restoration
# =============================================================================


def test_h13_perfect_match_skips_without_modification(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H13: Cache and workspace are identical. Expected: Skip without touching files.

    1. Run pipeline to cache files A,B,C
    2. Run pipeline again without any modifications
    3. Assert: stage skipped, files not touched (mtime preserved)
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Second run - should skip
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped"

        # Files still present
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])


# =============================================================================
# H14: Multiple DirectoryOut outputs
# =============================================================================


class _MultiDirectoryOutResult(TypedDict):
    first: Annotated[
        dict[str, dict[str, Any]], outputs.DirectoryOut("first/", loaders.JSON[dict[str, Any]]())
    ]
    second: Annotated[
        dict[str, dict[str, Any]], outputs.DirectoryOut("second/", loaders.JSON[dict[str, Any]]())
    ]


def _stage_produces_multi_directory_with_marker() -> _MultiDirectoryOutResult:
    """Stage that produces files in two separate DirectoryOut outputs."""
    marker_path = pathlib.Path("run_marker.txt")
    run_count = int(marker_path.read_text()) + 1 if marker_path.exists() else 1
    marker_path.write_text(str(run_count))

    return _MultiDirectoryOutResult(
        first={"a.json": {"dir": "first", "run": run_count}},
        second={"b.json": {"dir": "second", "run": run_count}},
    )


def test_h14_multiple_directory_outs_restored(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H14: Stage has multiple DirectoryOut. One corrupted. Expected: Both restored.

    1. Run pipeline with two DirectoryOut outputs
    2. Corrupt file in one directory
    3. Run pipeline again
    4. Assert: stage skipped, both directories properly restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates files in both directories
        register_test_stage(_stage_produces_multi_directory_with_marker, name="produce")
        executor.run()

        first_dir = pathlib.Path("first")
        second_dir = pathlib.Path("second")
        assert (first_dir / "a.json").exists()
        assert (second_dir / "b.json").exists()
        assert _get_run_count() == 1

        # Corrupt file in first directory
        (first_dir / "a.json").unlink()
        (first_dir / "a.json").write_text('{"corrupted": true}')

        # Second run - should skip and restore
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped"

        # Verify: both directories properly restored
        _assert_file_content(first_dir / "a.json", {"dir": "first", "run": 1})
        _assert_file_content(second_dir / "b.json", {"dir": "second", "run": 1})


# =============================================================================
# H15: Run cache skip with DirectoryOut
# =============================================================================


def test_h15_run_cache_restores_directory_out(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H15: Run cache restores DirectoryOut when lock file is deleted.

    1. Run pipeline to cache files A,B,C and record in run cache
    2. Delete both the lock file AND the output directory
    3. Run pipeline again
    4. Assert: stage skipped via run cache, files A,B,C restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run - creates and caches files, records in run cache
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})
        assert _get_run_count() == 1

        # Delete lock file to force run cache path
        lock_file = pathlib.Path(".pivot/produce.lock")
        if lock_file.exists():
            lock_file.unlink()

        # Delete output directory
        shutil.rmtree(results_dir)
        assert not results_dir.exists(), "Directory should be deleted"

        # Second run - should skip via run cache and restore
        executor.run()

        # Verify: stage did NOT re-run (counter still 1)
        assert _get_run_count() == 1, "Stage should have skipped via run cache"

        # Verify: files restored from run cache (run=1, not run=2)
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


def test_h15b_run_cache_restores_corrupted_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H15b: Run cache detects and restores corrupted DirectoryOut.

    1. Run pipeline to cache files A,B,C and record in run cache
    2. Delete lock file and corrupt file A in output directory
    3. Run pipeline again
    4. Assert: stage skipped via run cache, corrupted file A restored
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Delete lock file to force run cache path
        lock_file = pathlib.Path(".pivot/produce.lock")
        if lock_file.exists():
            lock_file.unlink()

        # Corrupt file A (unlink first - may be hardlinked)
        (results_dir / "a.json").unlink()
        (results_dir / "a.json").write_text('{"corrupted": true}')

        # Second run - should skip via run cache and fix corruption
        executor.run()

        # Verify: stage did NOT re-run
        assert _get_run_count() == 1, "Stage should have skipped via run cache"

        # Verify: corrupted file restored
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 1})


# =============================================================================
# H16: Error paths - cache missing triggers re-run
# =============================================================================


def test_h16_missing_cache_triggers_rerun(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """H16: When both cache and lock file are missing, stage re-runs.

    1. Run pipeline to cache files
    2. Delete cache, lock file, and corrupt output directory
    3. Run pipeline again
    4. Assert: stage RE-RAN (no cache available to restore)
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # First run
        register_test_stage(_stage_produces_abc_with_marker, name="produce")
        executor.run()

        results_dir = pathlib.Path("results")
        _assert_files_exist(results_dir, ["a.json", "b.json", "c.json"])
        assert _get_run_count() == 1

        # Delete lock file
        lock_file = pathlib.Path(".pivot/produce.lock")
        if lock_file.exists():
            lock_file.unlink()

        # Delete entire cache
        cache_dir = pathlib.Path(".pivot/cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        # Also clear run cache (in state.db) by removing state.db
        state_db = pathlib.Path(".pivot/state.db")
        state_db_lock = pathlib.Path(".pivot/state.db-lock")
        if state_db.exists():
            state_db.unlink()
        if state_db_lock.exists():
            state_db_lock.unlink()

        # Corrupt output
        (results_dir / "a.json").unlink()
        (results_dir / "a.json").write_text('{"corrupted": true}')

        # Second run - must RE-RUN because no cache or run cache available
        executor.run()

        # Verify: stage DID re-run (counter is now 2)
        assert _get_run_count() == 2, "Stage should have re-run when cache is missing"

        # Verify: files regenerated with run=2
        _assert_file_content(results_dir / "a.json", {"value": 1, "run": 2})
