"""Tests for pivot track CLI command."""

import pathlib

import click.testing
import pytest

from pivot import cli, project, pvt


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Basic Track Command Tests
# =============================================================================


def test_track_help(runner: click.testing.CliRunner) -> None:
    """Track command shows help."""
    result = runner.invoke(cli.cli, ["track", "--help"])
    assert result.exit_code == 0
    assert "Track files" in result.output or "track" in result.output.lower()


def test_track_file_creates_pvt_and_caches(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking a file creates .pvt file and caches content."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create a file to track
        data_file = pathlib.Path("data.csv")
        data_file.write_text("col1,col2\n1,2\n3,4\n")

        result = runner.invoke(cli.cli, ["track", "data.csv"])

        assert result.exit_code == 0, f"Failed with: {result.output}"

        # .pvt file should exist
        pvt_file = pathlib.Path("data.csv.pvt")
        assert pvt_file.exists(), ".pvt file should be created"

        # Read and verify .pvt content
        pvt_data = pvt.read_pvt_file(pvt_file)
        assert pvt_data is not None
        assert pvt_data["path"] == "data.csv"
        assert pvt_data["hash"], "Hash should be set"
        assert pvt_data["size"] > 0, "Size should be positive"

        # Cache should contain the file
        cache_dir = pathlib.Path(".pivot/cache/files")
        assert cache_dir.exists(), "Cache directory should be created"


def test_track_file_shows_confirmation(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking shows confirmation message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("content")

        result = runner.invoke(cli.cli, ["track", "data.csv"])

        assert result.exit_code == 0
        assert "data.csv" in result.output


def test_track_multiple_files(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Can track multiple files at once."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("a.csv").write_text("a")
        pathlib.Path("b.csv").write_text("b")

        result = runner.invoke(cli.cli, ["track", "a.csv", "b.csv"])

        assert result.exit_code == 0
        assert pathlib.Path("a.csv.pvt").exists()
        assert pathlib.Path("b.csv.pvt").exists()


def test_track_directory_with_manifest(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking a directory creates .pvt with manifest."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create directory with files
        data_dir = pathlib.Path("images")
        data_dir.mkdir()
        (data_dir / "cat.jpg").write_bytes(b"fake image 1")
        (data_dir / "dog.jpg").write_bytes(b"fake image 2")

        result = runner.invoke(cli.cli, ["track", "images"])

        assert result.exit_code == 0, f"Failed with: {result.output}"

        # .pvt file should exist (not images.pvt/ but images.pvt)
        pvt_file = pathlib.Path("images.pvt")
        assert pvt_file.exists(), ".pvt file should be created for directory"

        # Read and verify .pvt content
        pvt_data = pvt.read_pvt_file(pvt_file)
        assert pvt_data is not None
        assert pvt_data["path"] == "images"
        assert pvt_data["hash"], "Tree hash should be set"
        assert pvt_data.get("num_files") == 2, "Should have 2 files"
        manifest = pvt_data.get("manifest")
        assert manifest is not None, "Should have manifest"
        assert len(manifest) == 2, "Manifest should have 2 entries"

        # Verify manifest entries
        relpaths = {e["relpath"] for e in manifest}
        assert "cat.jpg" in relpaths
        assert "dog.jpg" in relpaths


def test_track_nested_file(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Track file in nested directory creates .pvt next to file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        nested = pathlib.Path("data/raw")
        nested.mkdir(parents=True)
        (nested / "train.csv").write_text("data")

        result = runner.invoke(cli.cli, ["track", "data/raw/train.csv"])

        assert result.exit_code == 0
        pvt_file = pathlib.Path("data/raw/train.csv.pvt")
        assert pvt_file.exists(), ".pvt should be next to tracked file"


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_track_missing_file_errors(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Tracking non-existent file fails with error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        result = runner.invoke(cli.cli, ["track", "missing.csv"])

        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "not found" in result.output.lower()


def test_track_no_arguments_errors(runner: click.testing.CliRunner) -> None:
    """Track without arguments shows error."""
    result = runner.invoke(cli.cli, ["track"])

    assert result.exit_code != 0


def test_track_duplicate_file_raises_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking same file twice raises error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("content")

        # First track should succeed
        result1 = runner.invoke(cli.cli, ["track", "data.csv"])
        assert result1.exit_code == 0

        # Second track should fail
        result2 = runner.invoke(cli.cli, ["track", "data.csv"])
        assert result2.exit_code != 0, "Duplicate tracking should fail"
        assert "already tracked" in result2.output.lower() or "duplicate" in result2.output.lower()


def test_track_file_in_stage_output_dir_raises_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking file within stage output directory raises error."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create output directory for a stage
        output_dir = pathlib.Path("outputs")
        output_dir.mkdir()
        (output_dir / "model.pkl").write_bytes(b"model")

        # Register a stage with directory output
        @stage(deps=[], outs=["outputs/"])
        def train() -> None:
            pass

        # Try to track a file within that output directory
        result = runner.invoke(cli.cli, ["track", "outputs/model.pkl"])

        # Should fail because path overlaps with stage output
        assert result.exit_code != 0, "Tracking file in stage output should fail"
        assert "output" in result.output.lower() or "conflict" in result.output.lower()

        # Clear registry for other tests
        REGISTRY.clear()


def test_track_stage_output_raises_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking a path that is a stage output raises error."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create the file
        pathlib.Path("model.pkl").write_bytes(b"model")

        # Register a stage with this file as output
        @stage(deps=[], outs=["model.pkl"])
        def train() -> None:
            pass

        # Try to track the stage output
        result = runner.invoke(cli.cli, ["track", "model.pkl"])

        assert result.exit_code != 0
        assert "output" in result.output.lower() or "conflict" in result.output.lower()

        REGISTRY.clear()


# =============================================================================
# Update Tracking Tests
# =============================================================================


def test_track_update_existing_with_force(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Re-tracking with --force updates the .pvt file."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        data_file = pathlib.Path("data.csv")
        data_file.write_text("original")

        # First track
        result1 = runner.invoke(cli.cli, ["track", "data.csv"])
        assert result1.exit_code == 0
        pvt_data1 = pvt.read_pvt_file(pathlib.Path("data.csv.pvt"))
        assert pvt_data1 is not None
        original_hash = pvt_data1["hash"]

        # Modify file
        data_file.write_text("modified content")

        # Re-track with --force should succeed and update hash
        result2 = runner.invoke(cli.cli, ["track", "--force", "data.csv"])
        assert result2.exit_code == 0

        pvt_data2 = pvt.read_pvt_file(pathlib.Path("data.csv.pvt"))
        assert pvt_data2 is not None
        assert pvt_data2["hash"] != original_hash, "Hash should be updated"


# =============================================================================
# Path Handling Tests
# =============================================================================


def test_track_relative_path(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Track accepts relative paths."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("content")

        result = runner.invoke(cli.cli, ["track", "./data.csv"])

        assert result.exit_code == 0


def test_track_path_traversal_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Paths with .. are rejected."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create file outside project
        parent = pathlib.Path("..").resolve()
        (parent / "outside.txt").write_text("secret")

        result = runner.invoke(cli.cli, ["track", "../outside.txt"])

        assert result.exit_code != 0


# =============================================================================
# Symlink Overlap Detection Tests
# =============================================================================


def test_track_symlink_to_stage_output_file_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking symlink that points to stage output file is rejected."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create stage output file
        pathlib.Path("model.pkl").write_bytes(b"model")

        # Register stage with this file as output
        @stage(deps=[], outs=["model.pkl"])
        def train() -> None:
            pass

        # Create symlink pointing to the stage output
        pathlib.Path("model_link.pkl").symlink_to("model.pkl")

        # Try to track the symlink
        result = runner.invoke(cli.cli, ["track", "model_link.pkl"])

        assert result.exit_code != 0, "Should reject symlink to stage output"
        assert "overlap" in result.output.lower() or "output" in result.output.lower(), (
            "Error should mention overlap with stage output"
        )
        assert "resolves to" in result.output.lower(), (
            "Error should show resolved path for debugging"
        )

        REGISTRY.clear()


def test_track_symlink_to_stage_output_directory_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking symlink that points to stage output directory is rejected."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create stage output directory
        output_dir = pathlib.Path("outputs")
        output_dir.mkdir()
        (output_dir / "results.txt").write_text("data")

        # Register stage with directory as output
        @stage(deps=[], outs=["outputs/"])
        def process() -> None:
            pass

        # Create symlink pointing to the output directory
        pathlib.Path("output_link").symlink_to("outputs")

        # Try to track the symlink
        result = runner.invoke(cli.cli, ["track", "output_link"])

        assert result.exit_code != 0, "Should reject symlink to stage output directory"
        assert "overlap" in result.output.lower() or "output" in result.output.lower()

        REGISTRY.clear()


def test_track_file_inside_symlinked_stage_output_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking file inside symlinked directory that's a stage output is rejected."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create actual directory with files
        real_dir = pathlib.Path("real_outputs")
        real_dir.mkdir()
        (real_dir / "model.pkl").write_bytes(b"model")

        # Create symlink to directory
        pathlib.Path("outputs_link").symlink_to("real_outputs")

        # Register stage with symlinked path as output
        @stage(deps=[], outs=["outputs_link/"])
        def train() -> None:
            pass

        # Try to track file via the real path
        result = runner.invoke(cli.cli, ["track", "real_outputs/model.pkl"])

        assert result.exit_code != 0, "Should detect overlap through symlink aliasing"
        assert "overlap" in result.output.lower() or "output" in result.output.lower(), (
            "Error should mention overlap with stage output"
        )

        REGISTRY.clear()


def test_track_broken_symlink_rejected_with_clear_error(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking broken symlink fails with clear error message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create broken symlink (target doesn't exist)
        pathlib.Path("broken_link.csv").symlink_to("nonexistent.csv")

        result = runner.invoke(cli.cli, ["track", "broken_link.csv"])

        assert result.exit_code != 0, "Should reject broken symlink"
        assert "broken" in result.output.lower() or "does not exist" in result.output.lower(), (
            "Error should clearly indicate broken symlink"
        )


def test_track_symlink_aliasing_same_file_different_paths(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking same file via different symlink paths detects aliasing."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create real file
        pathlib.Path("real_data.csv").write_text("data")

        # Create two symlinks to same file
        pathlib.Path("link1.csv").symlink_to("real_data.csv")
        pathlib.Path("link2.csv").symlink_to("real_data.csv")

        # Register stage with one symlink as output
        @stage(deps=[], outs=["link1.csv"])
        def produce() -> None:
            pass

        # Try to track the other symlink (points to same file)
        result = runner.invoke(cli.cli, ["track", "link2.csv"])

        assert result.exit_code != 0, "Should detect that link2 and link1 point to same file"
        assert "overlap" in result.output.lower() or "output" in result.output.lower()

        REGISTRY.clear()


def test_track_parent_directory_with_symlinked_stage_output(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Tracking parent directory when child is symlinked stage output is rejected."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create directory structure
        parent_dir = pathlib.Path("data")
        parent_dir.mkdir()
        real_file = parent_dir / "real_model.pkl"
        real_file.write_bytes(b"model")

        # Create symlink inside parent directory
        symlink = parent_dir / "model_link.pkl"
        symlink.symlink_to("real_model.pkl")

        # Register stage with symlink as output
        @stage(deps=[], outs=["data/model_link.pkl"])
        def train() -> None:
            pass

        # Try to track parent directory (contains stage output)
        result = runner.invoke(cli.cli, ["track", "data"])

        assert result.exit_code != 0, "Should reject tracking directory containing stage output"
        assert "overlap" in result.output.lower() or "output" in result.output.lower()

        REGISTRY.clear()


def test_track_with_normalized_vs_resolved_paths(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Error messages show both normalized and resolved paths for debugging."""
    from pivot import stage
    from pivot.registry import REGISTRY

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create nested directory structure with symlink
        pathlib.Path("real").mkdir()
        pathlib.Path("real/data.csv").write_text("data")
        pathlib.Path("link_to_real").symlink_to("real")

        # Register stage with real path
        @stage(deps=[], outs=["real/data.csv"])
        def process() -> None:
            pass

        # Try to track via symlinked path
        result = runner.invoke(cli.cli, ["track", "link_to_real/data.csv"])

        assert result.exit_code != 0, "Should detect overlap via symlink"
        # Both paths should appear in error for clarity
        assert "link_to_real" in result.output, "Should show user's path"
        assert "real" in result.output, "Should show resolved path"

        REGISTRY.clear()
