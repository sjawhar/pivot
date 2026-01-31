import pathlib
from typing import Annotated, TypedDict

import click.testing
import pytest

from helpers import register_test_stage
from pivot import cli, exceptions, executor, loaders, outputs, project
from pivot.storage import track


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Override pipeline_dir to include pivot.yaml for CLI tests.

    This fixture creates a minimal valid pivot.yaml because the `track` CLI
    command triggers auto-discovery which requires a valid pipeline config.
    """
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "pivot.yaml").write_text("stages: {}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return tmp_path


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _CountTxt(TypedDict):
    count: Annotated[pathlib.Path, outputs.Out("count.txt", loaders.PathOnly())]


# =============================================================================
# Tracked File Verification Tests
# =============================================================================


def _process_data(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    content = data.read_text()
    pathlib.Path("output.txt").write_text(f"processed: {len(content)} bytes")
    return {"output": pathlib.Path("output.txt")}


def _process_simple(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    _ = data
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _process_data_content(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    content = data.read_text()
    pathlib.Path("output.txt").write_text(f"processed: {content}")
    return {"output": pathlib.Path("output.txt")}


def _count_images(
    images: Annotated[pathlib.Path, outputs.Dep("images", loaders.PathOnly())],
) -> _CountTxt:
    image_list = list(images.iterdir())
    pathlib.Path("count.txt").write_text(str(len(image_list)))
    return {"count": pathlib.Path("count.txt")}


def _process_images(
    images: Annotated[pathlib.Path, outputs.Dep("images", loaders.PathOnly())],
) -> _OutputTxt:
    _ = images
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _process_mixed(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
    config: Annotated[pathlib.Path, outputs.Dep("config.txt", loaders.PathOnly())],
) -> _OutputTxt:
    data_content = data.read_text()
    config_content = config.read_text()
    pathlib.Path("output.txt").write_text(f"{data_content}|{config_content}")
    return {"output": pathlib.Path("output.txt")}


def _process_uppercase(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    content = data.read_text()
    pathlib.Path("output.txt").write_text(content.upper())
    return {"output": pathlib.Path("output.txt")}


def test_run_succeeds_with_existing_tracked_file(pipeline_dir: pathlib.Path) -> None:
    """Pipeline runs successfully when tracked file exists."""
    # Create and track a data file
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("col1,col2\n1,2\n")

    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt",
        {"path": "data.csv", "hash": "placeholder", "size": 100},
    )

    register_test_stage(_process_data, name="process")

    results = executor.run()

    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").exists()


def test_run_fails_when_tracked_file_missing(pipeline_dir: pathlib.Path) -> None:
    """Pipeline fails if tracked file is missing (with helpful error message)."""
    # Create .pvt file but NOT the data file
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt",
        {"path": "data.csv", "hash": "abc123", "size": 100},
    )

    register_test_stage(_process_simple, name="process")

    # Should fail with error message mentioning checkout
    with pytest.raises(
        exceptions.TrackedFileMissingError,
        match=r"checkout|restore|missing",
    ):
        executor.run()


def test_run_succeeds_with_hash_mismatch(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Pipeline runs successfully when tracked file hash doesn't match .pvt."""
    # Create data file with some content
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original content")

    # Track it properly via CLI to get correct hash
    result = runner.invoke(cli.cli, ["track", "data.csv"])
    assert result.exit_code == 0

    # Now modify the data file (hash mismatch)
    data_file.write_text("modified content that changes the hash")

    register_test_stage(_process_simple, name="process")

    # Should succeed (warning logged but execution continues)
    results = executor.run()
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

    register_test_stage(_process_data_content, name="process")

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "processed: original"

    # Modify tracked file and re-track
    data_file.write_text("modified")
    result = runner.invoke(cli.cli, ["track", "--force", "data.csv"])
    assert result.exit_code == 0

    # Second run - should re-execute due to tracked file change
    results = executor.run()
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

    register_test_stage(_process_data_content, name="process")

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Second run - should skip (nothing changed)
    results = executor.run()
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

    register_test_stage(_count_images, name="count_images")

    # First run
    results = executor.run()
    assert results["count_images"]["status"] == "ran"
    assert (pipeline_dir / "count.txt").read_text() == "2"

    # Add file to directory and re-track
    (data_dir / "c.jpg").write_bytes(b"image3")
    result = runner.invoke(cli.cli, ["track", "--force", "images"])
    assert result.exit_code == 0

    # Second run - should re-execute
    results = executor.run()
    assert results["count_images"]["status"] == "ran"
    assert (pipeline_dir / "count.txt").read_text() == "3"


def test_run_fails_when_tracked_directory_missing(pipeline_dir: pathlib.Path) -> None:
    """Pipeline fails if tracked directory is missing (with helpful error message)."""
    track.write_pvt_file(
        pipeline_dir / "images.pvt",
        {
            "path": "images",
            "hash": "abc123",
            "size": 1000,
            "num_files": 2,
            "manifest": [
                {"relpath": "a.jpg", "hash": "h1", "size": 500, "isexec": False},
                {"relpath": "b.jpg", "hash": "h2", "size": 500, "isexec": False},
            ],
        },
    )

    register_test_stage(_process_images, name="process")

    # Should fail with error message mentioning checkout
    with pytest.raises(
        exceptions.TrackedFileMissingError,
        match=r"checkout|restore|missing",
    ):
        executor.run()


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

    register_test_stage(_process_mixed, name="process")

    # First run
    results = executor.run()
    assert results["process"]["status"] == "ran"

    # Change regular file - should trigger rerun
    config_file.write_text("setting=2")
    results = executor.run()
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

    register_test_stage(_process_uppercase, name="process")

    # Pipeline should run successfully
    results = executor.run()
    assert results["process"]["status"] == "ran"
    assert (pipeline_dir / "output.txt").read_text() == "IMPORTANT DATA"
