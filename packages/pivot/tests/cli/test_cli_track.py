from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from conftest import isolated_pivot_dir
from pivot import cli, compose
from pivot.storage import track

if TYPE_CHECKING:
    import pathlib

    import click.testing
    import pytest

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


class _TrainOutputs(TypedDict):
    model: str
    metrics: str


@compose.stage
def process(input_file: str) -> str:
    return "done"


@compose.stage
def train() -> str:
    return "model"


@compose.stage
def train_multi() -> _TrainOutputs:
    return {"model": "m", "metrics": "x"}


def _output_path(root: pathlib.Path, stage_name: str) -> pathlib.Path:
    """Workspace path pivot assigns to a single-output str stage."""
    return root / "data" / "test" / f"{stage_name}.txt"


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_track_single_file(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Track one file creates .pvt file."""
    data_file = mock_discovery.root / "data.txt"
    data_file.write_text("tracked content")

    result = runner.invoke(cli.cli, ["track", "data.txt"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Tracked: data.txt" in result.output

    # .pvt file should exist
    pvt_path = mock_discovery.root / "data.txt.pvt"
    assert pvt_path.exists()

    # Read and verify .pvt content
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert pvt_data["path"] == "data.txt"
    assert "hash" in pvt_data
    assert pvt_data["size"] == len("tracked content")


def test_track_directory(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Track directory creates .pvt file with manifest."""
    data_dir = mock_discovery.root / "data_dir"
    data_dir.mkdir()
    (data_dir / "file1.txt").write_text("content1")
    (data_dir / "file2.txt").write_text("content2")

    result = runner.invoke(cli.cli, ["track", "data_dir"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Tracked: data_dir" in result.output

    pvt_path = mock_discovery.root / "data_dir.pvt"
    assert pvt_path.exists()

    # Read and verify .pvt content includes manifest
    pvt_data = track.read_pvt_file(pvt_path)
    assert pvt_data is not None
    assert "manifest" in pvt_data
    assert pvt_data.get("num_files") == 2


def test_track_force_overwrites(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--force updates existing .pvt file."""
    data_file = mock_discovery.root / "data.txt"
    data_file.write_text("original content")

    # First track
    result = runner.invoke(cli.cli, ["track", "data.txt"])
    assert result.exit_code == 0, result.output

    pvt_data_original = track.read_pvt_file(mock_discovery.root / "data.txt.pvt")
    assert pvt_data_original is not None
    original_hash = pvt_data_original["hash"]

    # Modify file
    data_file.write_text("modified content")

    # Track again with --force
    result = runner.invoke(cli.cli, ["track", "--force", "data.txt"])

    assert result.exit_code == 0, result.output
    pvt_data_updated = track.read_pvt_file(mock_discovery.root / "data.txt.pvt")
    assert pvt_data_updated is not None
    # Hash should be different due to modified content
    assert pvt_data_updated["hash"] != original_hash


def test_track_already_tracked_suggests_force(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Already tracked error suggests --force."""
    (mock_discovery.root / "data.txt").write_text("content")

    # First track
    first = runner.invoke(cli.cli, ["track", "data.txt"])
    assert first.exit_code == 0, first.output

    # Try to track again without --force
    result = runner.invoke(cli.cli, ["track", "data.txt"])

    assert result.exit_code != 0
    assert "--force" in result.output


# =============================================================================
# Security Tests
# =============================================================================


def test_track_path_traversal_rejected(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Rejects paths with ../ components."""
    _ = mock_discovery

    result = runner.invoke(cli.cli, ["track", "../outside.txt"])

    assert result.exit_code != 0
    assert (
        "outside project root" in result.output.lower() or "traversal" in result.output.lower()
    ), "Error should mention escaping project root or path traversal"


def test_track_broken_symlink_rejected(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Rejects broken symlinks with clear error message."""
    broken_link = mock_discovery.root / "broken_link"
    broken_link.symlink_to("nonexistent_target")

    result = runner.invoke(cli.cli, ["track", "broken_link"])

    assert result.exit_code != 0
    # Should mention broken symlink or target not existing
    assert "broken symlink" in result.output.lower() or "does not exist" in result.output.lower()


def test_track_overlap_with_stage_output_rejected(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Rejects paths that overlap with declared stage outputs."""
    raw_dir = mock_discovery.root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "input.txt").write_text("input")
    with mock_discovery:
        process(mock_discovery.input("input.txt", t=str))

    # Create the stage's output file where pivot writes it
    output = _output_path(mock_discovery.root, "process")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("output")

    # Try to track the stage output
    result = runner.invoke(cli.cli, ["track", "data/test/process.txt"])

    assert result.exit_code != 0, result.output
    assert "stage output" in result.output.lower() or "overlap" in result.output.lower()
    assert not (output.parent / "process.txt.pvt").exists()


# =============================================================================
# UX Tests
# =============================================================================


def test_track_echoes_user_path_in_output(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Track echoes the path as user provided it."""
    (mock_discovery.root / "data.csv").write_text("a,b\n1,2")

    result = runner.invoke(cli.cli, ["track", "./data.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Output shows what user typed
    assert "Tracked: ./data.csv" in result.output
    # .pvt file should be created with normalized name
    assert (mock_discovery.root / "data.csv.pvt").exists()


def test_track_file_not_found_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Track nonexistent file shows clear error."""
    _ = mock_discovery

    result = runner.invoke(cli.cli, ["track", "nonexistent.txt"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_track_multiple_files(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Track multiple files at once."""
    (mock_discovery.root / "file1.txt").write_text("content1")
    (mock_discovery.root / "file2.txt").write_text("content2")

    result = runner.invoke(cli.cli, ["track", "file1.txt", "file2.txt"])

    assert result.exit_code == 0, result.output
    assert "Tracked: file1.txt" in result.output
    assert "Tracked: file2.txt" in result.output
    assert (mock_discovery.root / "file1.txt.pvt").exists()
    assert (mock_discovery.root / "file2.txt.pvt").exists()


# =============================================================================
# Symlink Aliasing Tests - Security Critical
# =============================================================================


def test_track_symlink_to_stage_output_file_rejected(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Tracking symlink that points to stage output file is rejected."""
    with mock_discovery:
        train()
    output = _output_path(mock_discovery.root, "train")
    output.parent.mkdir(parents=True)
    output.write_text("model")

    # Create symlink pointing to the stage output
    (mock_discovery.root / "model_link.txt").symlink_to(output)

    result = runner.invoke(cli.cli, ["track", "model_link.txt"])

    assert result.exit_code != 0, "Should reject symlink to stage output"
    assert "overlap" in result.output.lower() or "output" in result.output.lower(), (
        "Error should mention overlap with stage output"
    )
    assert "resolves to" in result.output.lower(), "Error should show resolved path for debugging"


def test_track_symlink_to_stage_output_directory_rejected(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Tracking symlink that points to a multi-output stage's output directory is rejected."""
    with mock_discovery:
        train_multi()
    # Multi-output stage writes data/test/train_multi/<key>.<ext>
    output_dir = mock_discovery.root / "data" / "test" / "train_multi"
    output_dir.mkdir(parents=True)
    (output_dir / "model.txt").write_text("m")
    (output_dir / "metrics.txt").write_text("x")

    (mock_discovery.root / "output_link").symlink_to(output_dir)

    result = runner.invoke(cli.cli, ["track", "output_link"])

    assert result.exit_code != 0, "Should reject symlink to stage output directory"
    assert "overlap" in result.output.lower() or "output" in result.output.lower()


def test_track_file_inside_symlinked_stage_output_rejected(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Tracking file inside a real directory that a stage's output dir links to is rejected."""
    with mock_discovery:
        train()
    real_dir = mock_discovery.root / "real_outputs"
    real_dir.mkdir()
    (real_dir / "train.txt").write_text("model")
    # Stage outputs live under data/test, which is a symlink to real_outputs
    (mock_discovery.root / "data").mkdir()
    (mock_discovery.root / "data" / "test").symlink_to(real_dir)

    # Try to track the output via the real path
    result = runner.invoke(cli.cli, ["track", "real_outputs/train.txt"])

    assert result.exit_code != 0, "Should detect overlap through symlink aliasing"
    assert "overlap" in result.output.lower() or "output" in result.output.lower(), (
        "Error should mention overlap with stage output"
    )


def test_track_symlink_aliasing_same_file_different_paths(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Tracking same file via different symlink paths detects aliasing."""
    with mock_discovery:
        train()
    real_file = mock_discovery.root / "real_data.txt"
    real_file.write_text("data")

    # The stage output is one symlink to the real file; link2 is another
    output = _output_path(mock_discovery.root, "train")
    output.parent.mkdir(parents=True)
    output.symlink_to(real_file)
    (mock_discovery.root / "link2.txt").symlink_to(real_file)

    result = runner.invoke(cli.cli, ["track", "link2.txt"])

    assert result.exit_code != 0, "Should detect that link2 and the output point to same file"
    assert "overlap" in result.output.lower() or "output" in result.output.lower()


def test_track_parent_directory_with_symlinked_stage_output(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Tracking parent directory when child is symlinked stage output is rejected."""
    with mock_discovery:
        train()
    output = _output_path(mock_discovery.root, "train")
    output.parent.mkdir(parents=True)
    real_file = output.parent / "real_model.txt"
    real_file.write_text("model")
    output.symlink_to(real_file)

    # Try to track the parent directory (contains the stage output)
    result = runner.invoke(cli.cli, ["track", "data"])

    assert result.exit_code != 0, "Should reject tracking directory containing stage output"
    assert "overlap" in result.output.lower() or "output" in result.output.lower()


def test_track_with_normalized_vs_resolved_paths(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Error messages show both normalized and resolved paths for debugging."""
    with mock_discovery:
        process(mock_discovery.input("input.txt", t=str))
    output = _output_path(mock_discovery.root, "process")
    output.parent.mkdir(parents=True)
    output.write_text("data")
    (mock_discovery.root / "link_to_real").symlink_to(output.parent)

    # Try to track via symlinked path
    result = runner.invoke(cli.cli, ["track", "link_to_real/process.txt"])

    assert result.exit_code != 0, "Should detect overlap via symlink"
    # Both paths should appear in error for clarity
    assert "link_to_real" in result.output, "Should show user's path"
    assert "data/test/process.txt" in result.output, "Should show resolved path"


# =============================================================================
# No Pipeline Tests
# =============================================================================


def test_track_works_without_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track should succeed in a Pivot project without a pipeline definition."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data.csv").write_text("a,b,c\n1,2,3\n")
        result = runner.invoke(cli.cli, ["track", "data.csv"])
        assert result.exit_code == 0, f"Track should succeed without pipeline: {result.output}"
        assert "Tracked: data.csv" in result.output, "Should show tracked confirmation"
        assert (cwd / "data.csv.pvt").exists(), "PVT file should be created"


def test_track_force_works_without_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track --force should work without a pipeline definition."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data.csv").write_text("a,b,c\n1,2,3\n")
        # First track
        first = runner.invoke(cli.cli, ["track", "data.csv"])
        assert first.exit_code == 0, first.output
        # Update and re-track with --force
        (cwd / "data.csv").write_text("a,b,c\n4,5,6\n")
        result = runner.invoke(cli.cli, ["track", "--force", "data.csv"])
        assert result.exit_code == 0, (
            f"Track --force should succeed without pipeline: {result.output}"
        )
        assert "Tracked: data.csv" in result.output


# =============================================================================
# CWD-Relative Path Resolution Tests
# =============================================================================


def test_track_resolves_from_cwd(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Track should resolve paths relative to cwd, not project root."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        # Create subdirectory with file
        data_dir = cwd / "data"
        data_dir.mkdir()
        (data_dir / "file.csv").write_text("a,b,c\n1,2,3\n")

        # Change to subdirectory
        monkeypatch.chdir(data_dir)

        # Track file from subdirectory
        result = runner.invoke(cli.cli, ["track", "file.csv"])

        assert result.exit_code == 0, f"Track should succeed from subdir: {result.output}"
        assert "Tracked: file.csv" in result.output, "Should echo user's path"
        # .pvt file should be created next to the tracked file
        assert (data_dir / "file.csv.pvt").exists(), "PVT file should exist at data/file.csv.pvt"


def test_track_dotdot_from_subdir_resolves_correctly(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Track should allow .. in paths when resolved path stays within project root."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        # Create directory structure
        (cwd / "other").mkdir()
        (cwd / "other" / "file.txt").write_text("content")
        (cwd / "data").mkdir()

        # Change to data subdirectory
        monkeypatch.chdir(cwd / "data")

        # Track file using .. from subdirectory
        result = runner.invoke(cli.cli, ["track", "../other/file.txt"])

        assert result.exit_code == 0, (
            f"Track should allow .. when resolved path is within project: {result.output}"
        )
        assert "traversal" not in result.output.lower(), "Should not reject .. in valid paths"
        # .pvt file should exist for the tracked file
        assert (cwd / "other" / "file.txt.pvt").exists(), (
            "PVT file should exist at other/file.txt.pvt"
        )


def test_track_dotdot_escaping_project_root_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track should reject paths that escape project root via .."""
    with isolated_pivot_dir(runner, tmp_path):
        # Try to track file outside project root
        result = runner.invoke(cli.cli, ["track", "../../etc/passwd"])

        assert result.exit_code != 0, "Should reject paths escaping project root"
        assert (
            "outside project root" in result.output.lower() or "traversal" in result.output.lower()
        ), "Error should mention escaping project root or path traversal"


def test_track_absolute_path_still_works(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track should still work with absolute paths."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        # Create file
        (cwd / "data.csv").write_text("a,b,c\n1,2,3\n")

        # Track using absolute path
        result = runner.invoke(cli.cli, ["track", str(cwd / "data.csv")])

        assert result.exit_code == 0, f"Track should work with absolute path: {result.output}"
        # .pvt file should be created
        assert (cwd / "data.csv.pvt").exists(), "PVT file should exist"
