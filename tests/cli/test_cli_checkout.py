from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import cli, executor
from pivot.registry import REGISTRY
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing


# Module-level helper for stage functions
def _helper_process() -> None:
    pathlib.Path("output.txt").write_text("processed output")


def _helper_stage_a() -> None:
    pathlib.Path("a.txt").write_text("output a")


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
# Stage Output Tests
# =============================================================================


def test_checkout_stage_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Checkout restores a stage output from cache using lock file hash."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        # Run to generate output
        executor.run(show_output=False)
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


def test_checkout_skips_existing_without_force(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout skips existing files and shows 'Skipped' message."""
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

        # File exists (as symlink after save_to_cache), checkout without force
        result = runner.invoke(cli.cli, ["checkout", "data.txt"])

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
        pvt_data = track.PvtData(path="data.txt", hash="nonexistent_hash", size=100)
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
        REGISTRY.register(_helper_stage_a, name="stage_a", deps=["input.txt"], outs=["a.txt"])

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
