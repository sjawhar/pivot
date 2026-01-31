from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs, project
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _ProcessOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _StageAOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions for stage registration
# =============================================================================


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ProcessOutputs:
    _ = input_file  # deps tracked but not loaded in this simple test
    pathlib.Path("output.txt").write_text("processed output")
    return {"output": pathlib.Path("output.txt")}


def _helper_stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _StageAOutputs:
    _ = input_file  # deps tracked but not loaded in this simple test
    pathlib.Path("a.txt").write_text("output a")
    return {"output": pathlib.Path("a.txt")}


# =============================================================================
# Help and Basic Tests
# =============================================================================


def test_checkout_help(runner: click.testing.CliRunner) -> None:
    """Checkout command should show help."""
    result = runner.invoke(cli.cli, ["checkout", "--help"])

    assert result.exit_code == 0
    assert "--checkout-mode" in result.output
    assert "--force" in result.output
    assert "symlink" in result.output
    assert "hardlink" in result.output
    assert "copy" in result.output


# =============================================================================
# Tracked File Tests
# =============================================================================


def test_checkout_tracked_file(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Checkout restores a .pvt tracked file from cache."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create a file, save to cache, then track it
        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None

        # Create .pvt tracking file
        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        # Delete original file (save_to_cache replaced it with symlink/hardlink)
        data_file.unlink()
        assert not data_file.exists()

        # Checkout should restore it
        result = runner.invoke(cli.cli, ["checkout", "data.txt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert data_file.exists()
        assert data_file.read_text() == "tracked content"
        assert "Restored" in result.output


def test_checkout_accepts_pvt_file_path(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout accepts .pvt file paths and restores the corresponding data file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create a file, save to cache, then track it
        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None

        # Create .pvt tracking file
        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        # Delete original file
        data_file.unlink()
        assert not data_file.exists()

        # Checkout using .pvt path should restore the data file
        result = runner.invoke(cli.cli, ["checkout", "data.txt.pvt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert data_file.exists()
        assert data_file.read_text() == "tracked content"
        assert "Restored" in result.output


def test_checkout_all_tracked_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout with no targets restores all tracked files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create and track two files
        for name in ["data1.txt", "data2.txt"]:
            path = pathlib.Path(name)
            path.write_text(f"content of {name}")
            output_hash = cache.save_to_cache(path, cache_dir)
            assert output_hash is not None
            pvt_data = track.PvtData(
                path=name, hash=output_hash["hash"], size=len(f"content of {name}")
            )
            track.write_pvt_file(pathlib.Path(f"{name}.pvt"), pvt_data)
            # Remove the symlink/hardlink
            path.unlink()

        # Checkout all
        result = runner.invoke(cli.cli, ["checkout"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("data1.txt").exists()
        assert pathlib.Path("data2.txt").exists()
        assert pathlib.Path("data1.txt").read_text() == "content of data1.txt"
        assert pathlib.Path("data2.txt").read_text() == "content of data2.txt"


# =============================================================================
# Tracked Directory Tests
# =============================================================================


def test_checkout_tracked_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout restores a tracked directory from cache."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create directory with files
        data_dir = pathlib.Path("images")
        data_dir.mkdir()
        (data_dir / "cat.jpg").write_bytes(b"cat image")
        (data_dir / "dog.jpg").write_bytes(b"dog image")

        # Track the directory (creates .pvt and caches content)
        result = runner.invoke(cli.cli, ["track", "images"])
        assert result.exit_code == 0, f"Track failed: {result.output}"

        # Delete the directory
        shutil.rmtree(data_dir)
        assert not data_dir.exists()

        # Checkout should restore it
        result = runner.invoke(cli.cli, ["checkout", "images"])

        assert result.exit_code == 0, f"Checkout failed: {result.output}"
        assert data_dir.exists(), "Directory should be restored"
        assert (data_dir / "cat.jpg").read_bytes() == b"cat image"
        assert (data_dir / "dog.jpg").read_bytes() == b"dog image"


# =============================================================================
# Stage Output Tests
# =============================================================================


def test_checkout_stage_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Checkout restores a stage output from cache using lock file hash."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run to generate output
        executor.run()
        assert pathlib.Path("output.txt").exists()

        # Delete output
        pathlib.Path("output.txt").unlink()
        assert not pathlib.Path("output.txt").exists()

        # Checkout should restore it
        result = runner.invoke(cli.cli, ["checkout", "output.txt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("output.txt").exists()
        assert pathlib.Path("output.txt").read_text() == "processed output"


# =============================================================================
# Checkout Mode Tests
# =============================================================================


def test_checkout_mode_symlink(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--checkout-mode symlink creates symlinks."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None and "hash" in output_hash

        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)
        data_file.unlink()

        result = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "symlink", "data.txt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert data_file.exists()
        assert data_file.is_symlink(), "Should be symlink with symlink mode"


def test_checkout_mode_copy(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--checkout-mode copy creates independent copies."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None and "hash" in output_hash

        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)
        data_file.unlink()

        result = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "copy", "data.txt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert data_file.exists()
        assert not data_file.is_symlink(), "Should not be symlink with copy mode"


# =============================================================================
# Skip and Force Tests
# =============================================================================


def test_checkout_errors_on_existing_by_default(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout errors on existing files by default."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None and "hash" in output_hash

        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        # File exists (as symlink after save_to_cache), checkout without flags should error
        result = runner.invoke(cli.cli, ["checkout", "data.txt"])

        assert result.exit_code == 1
        assert "already exists" in result.output
        assert "--force" in result.output
        assert "--only-missing" in result.output


def test_checkout_only_missing_skips_existing(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--only-missing skips existing files and shows 'Skipped' message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None and "hash" in output_hash

        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        # File exists, --only-missing should skip it
        result = runner.invoke(cli.cli, ["checkout", "--only-missing", "data.txt"])

        assert result.exit_code == 0
        assert "Skipped" in result.output


def test_checkout_force_overwrites(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--force replaces existing files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        data_file = pathlib.Path("data.txt")
        data_file.write_text("original content")
        # Use copy mode so we can modify the file
        output_hash = cache.save_to_cache(
            data_file, cache_dir, checkout_mode=cache.CheckoutMode.COPY
        )
        assert output_hash is not None and "hash" in output_hash

        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=16)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        # Modify file locally (possible since it's a copy)
        data_file.write_text("modified content")
        assert data_file.read_text() == "modified content"

        # Force checkout should restore original
        result = runner.invoke(cli.cli, ["checkout", "--force", "data.txt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Restored" in result.output
        assert data_file.read_text() == "original content"


# =============================================================================
# Error Handling and UX Tests
# =============================================================================


def test_checkout_cache_miss_suggests_pull(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Missing cache entry error suggests 'pivot pull'."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create pvt file pointing to non-existent cache entry
        pvt_data = track.PvtData(path="data.txt", hash="deadbeef12345678", size=100)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        result = runner.invoke(cli.cli, ["checkout", "data.txt"])

        assert result.exit_code != 0
        assert "pivot pull" in result.output


def test_checkout_unknown_target_suggests_list_and_track(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Unknown target suggests 'pivot list' and 'pivot track'."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["checkout", "unknown_file.txt"])

        assert result.exit_code != 0
        assert "pivot list" in result.output
        assert "pivot track" in result.output


def test_checkout_uncached_output_suggests_run_or_pull(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Uncached stage output suggests 'pivot run' or 'pivot pull'."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        # Register stage but don't run it (so no cached output)
        register_test_stage(_helper_stage_a, name="stage_a")

        # Try to checkout the output that was never produced
        result = runner.invoke(cli.cli, ["checkout", "a.txt"])

        assert result.exit_code != 0
        # Should mention either run or pull as remediation
        assert "pivot" in result.output.lower()


def test_checkout_path_traversal_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout rejects paths with traversal components."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["checkout", "../outside.txt"])

        assert result.exit_code != 0
        assert "traversal" in result.output.lower()


def test_checkout_no_targets_no_files_shows_nothing_restored(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout with nothing to restore completes without error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # No tracked files, no stages - should complete successfully
        result = runner.invoke(cli.cli, ["checkout"])

        # Should complete without error (no targets is valid)
        assert result.exit_code == 0


# =============================================================================
# Partial Success and Parallel Checkout Tests
# =============================================================================


def test_checkout_partial_success_some_missing(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout continues with partial success, reports cache misses at end."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create first file - will be in cache
        data1 = pathlib.Path("data1.txt")
        data1.write_text("content1")
        output_hash1 = cache.save_to_cache(data1, cache_dir)
        assert output_hash1 is not None
        pvt_data1 = track.PvtData(path="data1.txt", hash=output_hash1["hash"], size=8)
        track.write_pvt_file(pathlib.Path("data1.txt.pvt"), pvt_data1)
        data1.unlink()

        # Create second file - will NOT be in cache (fake hash)
        pvt_data2 = track.PvtData(path="data2.txt", hash="deadbeef12345678", size=100)
        track.write_pvt_file(pathlib.Path("data2.txt.pvt"), pvt_data2)

        # Checkout all (no targets) should restore data1, fail on data2
        result = runner.invoke(cli.cli, ["checkout"])

        # Should exit non-zero due to partial failure
        assert result.exit_code == 1, (
            f"Expected exit code 1, got {result.exit_code}: {result.output}"
        )
        # data1 should be restored
        assert data1.exists()
        assert data1.read_text() == "content1"
        # Summary should indicate both success and failure
        assert "Restored 1 file(s)" in result.output
        assert "Missing 1 file(s)" in result.output
        assert "pivot pull" in result.output


def test_checkout_duplicate_targets_deduplicated(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """data.txt and data.txt.pvt resolve to same file, only restored once."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create and track a file
        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")
        output_hash = cache.save_to_cache(data_file, cache_dir)
        assert output_hash is not None
        pvt_data = track.PvtData(path="data.txt", hash=output_hash["hash"], size=15)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)
        data_file.unlink()

        # Checkout both data.txt and data.txt.pvt (duplicate targets)
        result = runner.invoke(cli.cli, ["checkout", "data.txt", "data.txt.pvt"])

        # Should succeed and only restore once
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert data_file.exists()
        assert data_file.read_text() == "tracked content"
        # Only one "Restored" message
        assert result.output.count("Restored") == 1


def test_checkout_quiet_with_failures_shows_summary(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--quiet still shows failure summary when cache misses occur."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create pvt file pointing to non-existent cache entry
        pvt_data = track.PvtData(path="data.txt", hash="deadbeef12345678", size=100)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        result = runner.invoke(cli.cli, ["--quiet", "checkout"])

        # Should fail
        assert result.exit_code == 1
        # Failure summary still shown even with --quiet
        assert "Missing 1 file(s)" in result.output
        assert "pivot pull" in result.output


def test_checkout_multiple_immediate_errors_aggregated(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Multiple 'already exists' errors are aggregated into single message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create and track two files that already exist
        for name in ["data1.txt", "data2.txt"]:
            path = pathlib.Path(name)
            path.write_text(f"content of {name}")
            output_hash = cache.save_to_cache(path, cache_dir)
            assert output_hash is not None
            pvt_data = track.PvtData(
                path=name, hash=output_hash["hash"], size=len(f"content of {name}")
            )
            track.write_pvt_file(pathlib.Path(f"{name}.pvt"), pvt_data)
            # Files exist as symlinks after save_to_cache

        # Try checkout without --force - both should error
        result = runner.invoke(cli.cli, ["checkout"])

        assert result.exit_code == 1
        # Both errors should be shown
        assert "data1.txt" in result.output
        assert "data2.txt" in result.output
        assert "already exists" in result.output


def test_checkout_shows_aggregate_counts(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout shows aggregate counts, not interleaved file names."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create 3 tracked files
        for i in range(1, 4):
            name = f"data{i}.txt"
            path = pathlib.Path(name)
            path.write_text(f"content{i}")
            output_hash = cache.save_to_cache(path, cache_dir)
            assert output_hash is not None
            pvt_data = track.PvtData(path=name, hash=output_hash["hash"], size=8)
            track.write_pvt_file(pathlib.Path(f"{name}.pvt"), pvt_data)
            # Remove the file so it needs restoration
            path.unlink()

        result = runner.invoke(cli.cli, ["checkout"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Should show count, not individual file names
        assert "Restored 3 file(s)" in result.output
        # Individual "Restored: data1.txt" lines should NOT appear
        assert "Restored: data" not in result.output
