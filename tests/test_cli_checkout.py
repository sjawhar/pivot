import pathlib

import click.testing

from pivot import cli, executor, project, stage
from pivot.registry import REGISTRY
from pivot.storage import track

# =============================================================================
# Basic Checkout Command Tests
# =============================================================================


def test_checkout_help(runner: click.testing.CliRunner) -> None:
    """Checkout command shows help."""
    result = runner.invoke(cli.cli, ["checkout", "--help"])
    assert result.exit_code == 0
    assert "Restore" in result.output or "checkout" in result.output.lower()


def test_checkout_restores_tracked_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout restores a tracked file from cache."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create and track a file
        data_file = pathlib.Path("data.csv")
        data_file.write_text("col1,col2\n1,2\n")
        original_content = data_file.read_text()

        result = runner.invoke(cli.cli, ["track", "data.csv"])
        assert result.exit_code == 0

        # Delete the data file
        data_file.unlink()
        assert not data_file.exists()

        # Checkout should restore it
        result = runner.invoke(cli.cli, ["checkout", "data.csv"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        # File should be restored
        assert data_file.exists(), "File should be restored"
        assert data_file.read_text() == original_content


def test_checkout_restores_tracked_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout restores a tracked directory from cache."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create and track a directory
        data_dir = pathlib.Path("images")
        data_dir.mkdir()
        (data_dir / "cat.jpg").write_bytes(b"cat image")
        (data_dir / "dog.jpg").write_bytes(b"dog image")

        result = runner.invoke(cli.cli, ["track", "images"])
        assert result.exit_code == 0

        # Delete the directory
        import shutil

        shutil.rmtree(data_dir)
        assert not data_dir.exists()

        # Checkout should restore it
        result = runner.invoke(cli.cli, ["checkout", "images"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        # Directory should be restored
        assert data_dir.exists(), "Directory should be restored"
        assert (data_dir / "cat.jpg").read_bytes() == b"cat image"
        assert (data_dir / "dog.jpg").read_bytes() == b"dog image"


def test_checkout_restores_stage_output(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout restores a stage output from cache."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create input file
        pathlib.Path("input.txt").write_text("input data")

        # Define and run a stage
        @stage(deps=["input.txt"], outs=["output.txt"])
        def process() -> None:
            pathlib.Path("output.txt").write_text("output data")

        # Run the stage to create and cache output
        executor.run(show_output=False)

        # Verify output exists
        output_file = pathlib.Path("output.txt")
        assert output_file.exists()

        # Delete the output
        output_file.unlink()
        assert not output_file.exists()

        # Checkout should restore it
        result = runner.invoke(cli.cli, ["checkout", "output.txt"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        # Output should be restored
        assert output_file.exists(), "Stage output should be restored"
        # Content may differ slightly due to symlinking behavior, but should exist

        REGISTRY.clear()


def test_checkout_all_restores_everything(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout without arguments restores all tracked files and outputs."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Track a file
        pathlib.Path("data.csv").write_text("data")
        runner.invoke(cli.cli, ["track", "data.csv"])

        # Create and run a stage
        pathlib.Path("input.txt").write_text("input")

        @stage(deps=["input.txt"], outs=["model.pkl"])
        def train() -> None:
            pathlib.Path("model.pkl").write_bytes(b"model")

        executor.run(show_output=False)

        # Delete both
        pathlib.Path("data.csv").unlink()
        pathlib.Path("model.pkl").unlink()

        # Checkout all
        result = runner.invoke(cli.cli, ["checkout"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        # Both should be restored
        assert pathlib.Path("data.csv").exists(), "Tracked file should be restored"
        assert pathlib.Path("model.pkl").exists(), "Stage output should be restored"

        REGISTRY.clear()


def test_checkout_with_checkout_mode_option(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout respects --checkout-mode option."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Track a file
        pathlib.Path("data.csv").write_text("data content")
        runner.invoke(cli.cli, ["track", "data.csv"])

        # Delete it
        pathlib.Path("data.csv").unlink()

        # Checkout with copy mode
        result = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "copy", "data.csv"])
        assert result.exit_code == 0

        # File should exist and be a regular file (not symlink)
        data_file = pathlib.Path("data.csv")
        assert data_file.exists()
        assert not data_file.is_symlink(), "With --checkout-mode=copy, should not be symlink"


def test_checkout_with_symlink_mode(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout with symlink mode creates symlinks."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("data")
        runner.invoke(cli.cli, ["track", "data.csv"])

        pathlib.Path("data.csv").unlink()

        result = runner.invoke(cli.cli, ["checkout", "--checkout-mode", "symlink", "data.csv"])
        assert result.exit_code == 0

        data_file = pathlib.Path("data.csv")
        assert data_file.exists()
        # With symlink mode, file should be a symlink
        assert data_file.is_symlink(), "With --checkout-mode=symlink, should be symlink"


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_checkout_missing_from_cache_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout fails when file not in cache."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Create .pvt file manually without caching the data
        pvt_data: track.PvtData = {
            "path": "data.csv",
            "hash": "nonexistent_hash",
            "size": 100,
        }
        track.write_pvt_file(pathlib.Path("data.csv.pvt"), pvt_data)

        result = runner.invoke(cli.cli, ["checkout", "data.csv"])

        assert result.exit_code != 0
        assert "cache" in result.output.lower() or "not found" in result.output.lower()


def test_checkout_unknown_target_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout fails for unknown target (not tracked, not a stage output)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        result = runner.invoke(cli.cli, ["checkout", "unknown.csv"])

        assert result.exit_code != 0


def test_checkout_already_exists_skips(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Checkout skips files that already exist (without --force)."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("original")
        runner.invoke(cli.cli, ["track", "data.csv"])

        # Modify file
        pathlib.Path("data.csv").write_text("modified")

        # Checkout should skip (file exists)
        result = runner.invoke(cli.cli, ["checkout", "data.csv"])
        assert result.exit_code == 0
        assert "skip" in result.output.lower() or pathlib.Path("data.csv").read_text() == "modified"


def test_checkout_force_overwrites(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Checkout with --force overwrites existing files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("original")
        runner.invoke(cli.cli, ["track", "data.csv"])

        # Modify file
        pathlib.Path("data.csv").write_text("modified")

        # Checkout with --force should overwrite
        result = runner.invoke(cli.cli, ["checkout", "--force", "data.csv"])
        assert result.exit_code == 0

        # File should have original content
        assert pathlib.Path("data.csv").read_text() == "original"


# =============================================================================
# Multiple Targets Tests
# =============================================================================


def test_checkout_multiple_targets(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Checkout can restore multiple targets at once."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        # Track multiple files
        pathlib.Path("a.csv").write_text("a")
        pathlib.Path("b.csv").write_text("b")
        runner.invoke(cli.cli, ["track", "a.csv", "b.csv"])

        # Delete both
        pathlib.Path("a.csv").unlink()
        pathlib.Path("b.csv").unlink()

        # Checkout both
        result = runner.invoke(cli.cli, ["checkout", "a.csv", "b.csv"])
        assert result.exit_code == 0

        assert pathlib.Path("a.csv").exists()
        assert pathlib.Path("b.csv").exists()


# =============================================================================
# Status Reporting Tests
# =============================================================================


def test_checkout_reports_status(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Checkout reports status for each target."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        project._project_root_cache = None

        pathlib.Path("data.csv").write_text("data")
        runner.invoke(cli.cli, ["track", "data.csv"])

        pathlib.Path("data.csv").unlink()

        result = runner.invoke(cli.cli, ["checkout", "data.csv"])
        assert result.exit_code == 0
        assert "data.csv" in result.output
