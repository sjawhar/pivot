from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs
from pivot.storage import track

if TYPE_CHECKING:
    import click.testing
    import pytest

    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _OutputsDir(TypedDict):
    output_dir: Annotated[pathlib.Path, outputs.Out("outputs/", loaders.PathOnly())]


class _ModelPkl(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


class _OutputsLinkDir(TypedDict):
    output_dir: Annotated[pathlib.Path, outputs.Out("outputs_link/", loaders.PathOnly())]


class _Link1Csv(TypedDict):
    link: Annotated[pathlib.Path, outputs.Out("link1.csv", loaders.PathOnly())]


class _ModelLinkPkl(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("data/model_link.pkl", loaders.PathOnly())]


class _RealDataCsv(TypedDict):
    data: Annotated[pathlib.Path, outputs.Out("real/data.csv", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions
# =============================================================================


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _train_outputs_dir() -> _OutputsDir:
    pathlib.Path("outputs").mkdir(exist_ok=True)
    return {"output_dir": pathlib.Path("outputs")}


def _train_model_pkl() -> _ModelPkl:
    pathlib.Path("model.pkl").write_bytes(b"model")
    return {"model": pathlib.Path("model.pkl")}


def _process_outputs_link() -> _OutputsLinkDir:
    return {"output_dir": pathlib.Path("outputs_link")}


def _produce_link1() -> _Link1Csv:
    return {"link": pathlib.Path("link1.csv")}


def _train_model_link() -> _ModelLinkPkl:
    return {"model": pathlib.Path("data/model_link.pkl")}


def _process_real_data() -> _RealDataCsv:
    return {"data": pathlib.Path("real/data.csv")}


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_track_single_file(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track one file creates .pvt file."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    data_file = tmp_path / "data.txt"
    data_file.write_text("tracked content")

    result = runner.invoke(cli.cli, ["track", "data.txt"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Tracked: data.txt" in result.output

    # .pvt file should exist
    pvt_path = tmp_path / "data.txt.pvt"
    assert pvt_path.exists()

    # Read and verify .pvt content
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["path"] == "data.txt"
    assert "hash" in pvt_data
    assert "size" in pvt_data


def test_track_directory(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track directory creates .pvt file with manifest."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create directory with files
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")
    (data_dir / "file2.txt").write_text("content2")

    result = runner.invoke(cli.cli, ["track", "data_dir"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Tracked: data_dir" in result.output

    # .pvt file should exist
    pvt_path = tmp_path / "data_dir.pvt"
    assert pvt_path.exists()

    # Read and verify .pvt content includes manifest
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert "manifest" in pvt_data
    assert pvt_data.get("num_files") == 2


def test_track_force_overwrites(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--force updates existing .pvt file."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    data_file = tmp_path / "data.txt"
    data_file.write_text("original content")

    # First track
    result = runner.invoke(cli.cli, ["track", "data.txt"])
    assert result.exit_code == 0

    pvt_data_original = track.read_pvt_file(tmp_path / "data.txt.pvt")
    original_hash = pvt_data_original["hash"] if pvt_data_original else ""

    # Modify file
    data_file.write_text("modified content")

    # Track again with --force
    result = runner.invoke(cli.cli, ["track", "--force", "data.txt"])

    assert result.exit_code == 0
    pvt_data_updated = track.read_pvt_file(tmp_path / "data.txt.pvt")
    assert pvt_data_updated is not None
    # Hash should be different due to modified content
    assert pvt_data_updated["hash"] != original_hash


def test_track_already_tracked_suggests_force(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already tracked error suggests --force."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    data_file = tmp_path / "data.txt"
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
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejects paths with ../ components."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    result = runner.invoke(cli.cli, ["track", "../outside.txt"])

    assert result.exit_code != 0
    assert "traversal" in result.output.lower()


def test_track_broken_symlink_rejected(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejects broken symlinks with clear error message."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create broken symlink
    broken_link = tmp_path / "broken_link"
    broken_link.symlink_to("nonexistent_target")

    result = runner.invoke(cli.cli, ["track", "broken_link"])

    assert result.exit_code != 0
    # Should mention broken symlink or target not existing
    assert "broken symlink" in result.output.lower() or "does not exist" in result.output.lower()


def test_track_overlap_with_stage_output_rejected(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejects paths that overlap with declared stage outputs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("input")

    # Register a stage with output
    register_test_stage(_helper_process, name="process")

    # Create the output file
    (tmp_path / "output.txt").write_text("output")

    # Try to track the stage output
    result = runner.invoke(cli.cli, ["track", "output.txt"])

    assert result.exit_code != 0
    assert "stage output" in result.output.lower() or "overlap" in result.output.lower()


# =============================================================================
# UX Tests
# =============================================================================


def test_track_echoes_user_path_in_output(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track echoes the path as user provided it."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    (tmp_path / "data.csv").write_text("a,b\n1,2")

    result = runner.invoke(cli.cli, ["track", "./data.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Output shows what user typed
    assert "Tracked: ./data.csv" in result.output
    # .pvt file should be created with normalized name
    assert (tmp_path / "data.csv.pvt").exists()


def test_track_file_not_found_error(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track nonexistent file shows clear error."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    result = runner.invoke(cli.cli, ["track", "nonexistent.txt"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_track_multiple_files(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track multiple files at once."""
    _ = mock_discovery
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    result = runner.invoke(cli.cli, ["track", "file1.txt", "file2.txt"])

    assert result.exit_code == 0
    assert "Tracked: file1.txt" in result.output
    assert "Tracked: file2.txt" in result.output
    assert (tmp_path / "file1.txt.pvt").exists()
    assert (tmp_path / "file2.txt.pvt").exists()


# =============================================================================
# Symlink Aliasing Tests - Security Critical
# =============================================================================


def test_track_symlink_to_stage_output_file_rejected(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking symlink that points to stage output file is rejected."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create stage output file
    (tmp_path / "model.pkl").write_bytes(b"model")

    # Register stage with this file as output
    register_test_stage(_train_model_pkl, name="train")

    # Create symlink pointing to the stage output
    (tmp_path / "model_link.pkl").symlink_to("model.pkl")

    # Try to track the symlink
    result = runner.invoke(cli.cli, ["track", "model_link.pkl"])

    assert result.exit_code != 0, "Should reject symlink to stage output"
    assert "overlap" in result.output.lower() or "output" in result.output.lower(), (
        "Error should mention overlap with stage output"
    )
    assert "resolves to" in result.output.lower(), "Error should show resolved path for debugging"


def test_track_symlink_to_stage_output_directory_rejected(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking symlink that points to stage output directory is rejected."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create stage output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "results.txt").write_text("data")

    # Register stage with directory as output (outputs/)
    register_test_stage(_train_outputs_dir, name="process")

    # Create symlink pointing to the output directory
    (tmp_path / "output_link").symlink_to("outputs")

    # Try to track the symlink
    result = runner.invoke(cli.cli, ["track", "output_link"])

    assert result.exit_code != 0, "Should reject symlink to stage output directory"
    assert "overlap" in result.output.lower() or "output" in result.output.lower()


def test_track_file_inside_symlinked_stage_output_rejected(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking file inside symlinked directory that's a stage output is rejected."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create actual directory with files
    real_dir = tmp_path / "real_outputs"
    real_dir.mkdir()
    (real_dir / "model.pkl").write_bytes(b"model")

    # Create symlink to directory
    (tmp_path / "outputs_link").symlink_to("real_outputs")

    # Register stage with symlinked path as output
    register_test_stage(_process_outputs_link, name="train")

    # Try to track file via the real path
    result = runner.invoke(cli.cli, ["track", "real_outputs/model.pkl"])

    assert result.exit_code != 0, "Should detect overlap through symlink aliasing"
    assert "overlap" in result.output.lower() or "output" in result.output.lower(), (
        "Error should mention overlap with stage output"
    )


def test_track_symlink_aliasing_same_file_different_paths(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking same file via different symlink paths detects aliasing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create real file
    (tmp_path / "real_data.csv").write_text("data")

    # Create two symlinks to same file
    (tmp_path / "link1.csv").symlink_to("real_data.csv")
    (tmp_path / "link2.csv").symlink_to("real_data.csv")

    # Register stage with one symlink as output
    register_test_stage(_produce_link1, name="produce")

    # Try to track the other symlink (points to same file)
    result = runner.invoke(cli.cli, ["track", "link2.csv"])

    assert result.exit_code != 0, "Should detect that link2 and link1 point to same file"
    assert "overlap" in result.output.lower() or "output" in result.output.lower()


def test_track_parent_directory_with_symlinked_stage_output(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking parent directory when child is symlinked stage output is rejected."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create directory structure
    parent_dir = tmp_path / "data"
    parent_dir.mkdir()
    real_file = parent_dir / "real_model.pkl"
    real_file.write_bytes(b"model")

    # Create symlink inside parent directory
    symlink = parent_dir / "model_link.pkl"
    symlink.symlink_to("real_model.pkl")

    # Register stage with symlink as output
    register_test_stage(_train_model_link, name="train")

    # Try to track parent directory (contains stage output)
    result = runner.invoke(cli.cli, ["track", "data"])

    assert result.exit_code != 0, "Should reject tracking directory containing stage output"
    assert "overlap" in result.output.lower() or "output" in result.output.lower()


def test_track_with_normalized_vs_resolved_paths(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error messages show both normalized and resolved paths for debugging."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create nested directory structure with symlink
    (tmp_path / "real").mkdir()
    (tmp_path / "real/data.csv").write_text("data")
    (tmp_path / "link_to_real").symlink_to("real")

    # Register stage with real path
    register_test_stage(_process_real_data, name="process")

    # Try to track via symlinked path
    result = runner.invoke(cli.cli, ["track", "link_to_real/data.csv"])

    assert result.exit_code != 0, "Should detect overlap via symlink"
    # Both paths should appear in error for clarity
    assert "link_to_real" in result.output, "Should show user's path"
    assert "real" in result.output, "Should show resolved path"
