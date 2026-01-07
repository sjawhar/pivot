"""Integration tests for executor with .pvt tracked files."""

import pathlib

import click.testing
import pytest

import pivot
from pivot import cli, exceptions, executor, project, pvt, registry


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Reset registry before each test."""
    registry.REGISTRY.clear()


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None
    return tmp_path


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Tracked File Verification Tests
# =============================================================================


def test_run_succeeds_with_existing_tracked_file(pipeline_dir: pathlib.Path) -> None:
    """Pipeline runs successfully when tracked file exists."""
    # Create and track a data file
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("col1,col2\n1,2\n")

    pvt.write_pvt_file(
        pipeline_dir / "data.csv.pvt",
        {"path": "data.csv", "hash": "placeholder", "size": 100},
    )

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("data.csv").read_text()
        pathlib.Path("output.txt").write_text(f"processed: {len(data)} bytes")

    results = executor.run(show_output=False)

    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").exists()


def test_run_fails_when_tracked_file_missing(pipeline_dir: pathlib.Path) -> None:
    """Pipeline fails if tracked file is missing (with helpful error message)."""
    # Create .pvt file but NOT the data file
    pvt.write_pvt_file(
        pipeline_dir / "data.csv.pvt",
        {"path": "data.csv", "hash": "abc123", "size": 100},
    )

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should fail with error message mentioning checkout
    with pytest.raises(
        exceptions.TrackedFileMissingError,
        match=r"checkout|restore|missing",
    ):
        executor.run(show_output=False)


def test_run_succeeds_with_hash_mismatch(
    pipeline_dir: pathlib.Path,
) -> None:
    """Pipeline runs successfully when tracked file hash doesn't match .pvt."""
    # Create data file with some content
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original content")

    # Track it properly via CLI to get correct hash
    result = click.testing.CliRunner().invoke(cli.cli, ["track", "data.csv"])
    assert result.exit_code == 0

    # Now modify the data file (hash mismatch)
    data_file.write_text("modified content that changes the hash")

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should succeed (warning logged but execution continues)
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").exists()


# =============================================================================
# Dependency Change Detection Tests
# =============================================================================


def test_tracked_file_change_triggers_downstream_rerun(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Changing a tracked file triggers re-execution of dependent stages."""
    # Create and track initial data
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original")

    result = runner.invoke(cli.cli, ["track", "data.csv"])
    assert result.exit_code == 0

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("data.csv").read_text()
        pathlib.Path("output.txt").write_text(f"processed: {data}")

    # First run
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "processed: original"

    # Modify tracked file and re-track
    data_file.write_text("modified")
    result = runner.invoke(cli.cli, ["track", "--force", "data.csv"])
    assert result.exit_code == 0

    # Second run - should re-execute due to tracked file change
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "processed: modified"


def test_unchanged_tracked_file_allows_skip(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Unchanged tracked file allows stage to be skipped."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("content")

    result = runner.invoke(cli.cli, ["track", "data.csv"])
    assert result.exit_code == 0

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("data.csv").read_text()
        pathlib.Path("output.txt").write_text(f"processed: {data}")

    # First run
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"

    # Second run - should skip (nothing changed)
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "skipped"


# =============================================================================
# Directory Tracking Tests
# =============================================================================


def test_tracked_directory_change_triggers_rerun(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Changing a tracked directory triggers re-execution."""
    # Create and track directory
    data_dir = pipeline_dir / "images"
    data_dir.mkdir()
    (data_dir / "a.jpg").write_bytes(b"image1")
    (data_dir / "b.jpg").write_bytes(b"image2")

    result = runner.invoke(cli.cli, ["track", "images"])
    assert result.exit_code == 0

    @pivot.stage(deps=["images"], outs=["count.txt"])
    def count_images() -> None:
        images = list(pathlib.Path("images").iterdir())
        pathlib.Path("count.txt").write_text(str(len(images)))

    # First run
    results = executor.run(show_output=False)
    assert results["count_images"]["status"] == "ran"
    assert (pipeline_dir / "count.txt").read_text() == "2"

    # Add file to directory and re-track
    (data_dir / "c.jpg").write_bytes(b"image3")
    result = runner.invoke(cli.cli, ["track", "--force", "images"])
    assert result.exit_code == 0

    # Second run - should re-execute
    results = executor.run(show_output=False)
    assert results["count_images"]["status"] == "ran"
    assert (pipeline_dir / "count.txt").read_text() == "3"


def test_run_fails_when_tracked_directory_missing(pipeline_dir: pathlib.Path) -> None:
    """Pipeline fails if tracked directory is missing (with helpful error message)."""
    pvt.write_pvt_file(
        pipeline_dir / "images.pvt",
        {
            "path": "images",
            "hash": "abc123",
            "size": 1000,
            "num_files": 2,
            "manifest": [
                {"relpath": "a.jpg", "hash": "h1", "size": 500},
                {"relpath": "b.jpg", "hash": "h2", "size": 500},
            ],
        },
    )

    @pivot.stage(deps=["images"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    # Should fail with error message mentioning checkout
    with pytest.raises(
        exceptions.TrackedFileMissingError,
        match=r"checkout|restore|missing",
    ):
        executor.run(show_output=False)


# =============================================================================
# Mixed Dependency Tests
# =============================================================================


def test_mixed_tracked_and_regular_dependencies(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Stages can depend on both tracked files and regular files."""
    # Create tracked file
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("tracked data")
    result = runner.invoke(cli.cli, ["track", "data.csv"])
    assert result.exit_code == 0

    # Create regular file
    config_file = pipeline_dir / "config.txt"
    config_file.write_text("setting=1")

    @pivot.stage(deps=["data.csv", "config.txt"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("data.csv").read_text()
        config = pathlib.Path("config.txt").read_text()
        pathlib.Path("output.txt").write_text(f"{data}|{config}")

    # First run
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"

    # Change regular file - should trigger rerun
    config_file.write_text("setting=2")
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"


# =============================================================================
# Checkpoint and Restore Tests
# =============================================================================


def test_checkout_then_run_succeeds(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """After checkout, pipeline can run successfully."""
    # Create, track, then delete a file
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("important data")

    result = runner.invoke(cli.cli, ["track", "data.csv"])
    assert result.exit_code == 0

    # Delete the file
    data_file.unlink()
    assert not data_file.exists()

    # Checkout should restore it
    result = runner.invoke(cli.cli, ["checkout", "data.csv"])
    assert result.exit_code == 0
    assert data_file.exists()

    @pivot.stage(deps=["data.csv"], outs=["output.txt"])
    def process() -> None:
        data = pathlib.Path("data.csv").read_text()
        pathlib.Path("output.txt").write_text(data.upper())

    # Pipeline should run successfully
    results = executor.run(show_output=False)
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "IMPORTANT DATA"
