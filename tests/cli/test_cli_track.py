from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs
from pivot.storage import track

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions
# =============================================================================


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_track_single_file(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Track one file creates .pvt file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        data_file = pathlib.Path("data.txt")
        data_file.write_text("tracked content")

        result = runner.invoke(cli.cli, ["track", "data.txt"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Tracked: data.txt" in result.output

        # .pvt file should exist
        pvt_path = pathlib.Path("data.txt.pvt")
        assert pvt_path.exists()

        # Read and verify .pvt content
        pvt_data = track.read_pvt_file(pvt_path)
        assert pvt_data is not None
        assert pvt_data["path"] == "data.txt"
        assert "hash" in pvt_data
        assert "size" in pvt_data


def test_track_directory(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Track directory creates .pvt file with manifest."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create directory with files
        data_dir = pathlib.Path("data_dir")
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("content1")
        (data_dir / "file2.txt").write_text("content2")

        result = runner.invoke(cli.cli, ["track", "data_dir"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Tracked: data_dir" in result.output

        # .pvt file should exist
        pvt_path = pathlib.Path("data_dir.pvt")
        assert pvt_path.exists()

        # Read and verify .pvt content includes manifest
        pvt_data = track.read_pvt_file(pvt_path)
        assert pvt_data is not None
        assert "manifest" in pvt_data
        assert pvt_data.get("num_files") == 2


def test_track_force_overwrites(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--force updates existing .pvt file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        data_file = pathlib.Path("data.txt")
        data_file.write_text("original content")

        # First track
        result = runner.invoke(cli.cli, ["track", "data.txt"])
        assert result.exit_code == 0

        pvt_data_original = track.read_pvt_file(pathlib.Path("data.txt.pvt"))
        original_hash = pvt_data_original["hash"] if pvt_data_original else ""

        # Modify file
        data_file.write_text("modified content")

        # Track again with --force
        result = runner.invoke(cli.cli, ["track", "--force", "data.txt"])

        assert result.exit_code == 0
        pvt_data_updated = track.read_pvt_file(pathlib.Path("data.txt.pvt"))
        assert pvt_data_updated is not None
        # Hash should be different due to modified content
        assert pvt_data_updated["hash"] != original_hash


def test_track_already_tracked_suggests_force(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Already tracked error suggests --force."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        data_file = pathlib.Path("data.txt")
        data_file.write_text("content")

        # First track
        runner.invoke(cli.cli, ["track", "data.txt"])

        # Try to track again without --force
        result = runner.invoke(cli.cli, ["track", "data.txt"])

        assert result.exit_code != 0
        assert "--force" in result.output


# =============================================================================
# Security Tests
# =============================================================================


def test_track_path_traversal_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Rejects paths with ../ components."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["track", "../outside.txt"])

        assert result.exit_code != 0
        assert "traversal" in result.output.lower()


def test_track_broken_symlink_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Rejects broken symlinks with clear error message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create broken symlink
        broken_link = pathlib.Path("broken_link")
        broken_link.symlink_to("nonexistent_target")

        result = runner.invoke(cli.cli, ["track", "broken_link"])

        assert result.exit_code != 0
        # Should mention broken symlink or target not existing
        assert (
            "broken symlink" in result.output.lower() or "does not exist" in result.output.lower()
        )


def test_track_overlap_with_stage_output_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Rejects paths that overlap with declared stage outputs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()
        pathlib.Path("input.txt").write_text("input")

        # Register a stage with output
        register_test_stage(_helper_process, name="process")

        # Create the output file
        pathlib.Path("output.txt").write_text("output")

        # Try to track the stage output
        result = runner.invoke(cli.cli, ["track", "output.txt"])

        assert result.exit_code != 0
        assert "stage output" in result.output.lower() or "overlap" in result.output.lower()


# =============================================================================
# UX Tests
# =============================================================================


def test_track_echoes_user_path_in_output(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track echoes the path as user provided it."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        pathlib.Path("data.csv").write_text("a,b\n1,2")

        result = runner.invoke(cli.cli, ["track", "./data.csv"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # Output shows what user typed
        assert "Tracked: ./data.csv" in result.output
        # .pvt file should be created with normalized name
        assert pathlib.Path("data.csv.pvt").exists()


def test_track_file_not_found_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track nonexistent file shows clear error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        result = runner.invoke(cli.cli, ["track", "nonexistent.txt"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()


def test_track_multiple_files(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Track multiple files at once."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        pathlib.Path("file1.txt").write_text("content1")
        pathlib.Path("file2.txt").write_text("content2")

        result = runner.invoke(cli.cli, ["track", "file1.txt", "file2.txt"])

        assert result.exit_code == 0
        assert "Tracked: file1.txt" in result.output
        assert "Tracked: file2.txt" in result.output
        assert pathlib.Path("file1.txt.pvt").exists()
        assert pathlib.Path("file2.txt.pvt").exists()
