from __future__ import annotations

import json
import pathlib

import click.testing
import pytest

from pivot import cli, executor
from pivot.registry import REGISTRY
from pivot.storage import cache, track


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Help and Basic Tests
# =============================================================================


def test_status_help(runner: click.testing.CliRunner) -> None:
    """Status command should show help."""
    result = runner.invoke(cli.cli, ["status", "--help"])

    assert result.exit_code == 0
    assert "--verbose" in result.output
    assert "--json" in result.output
    assert "--stages-only" in result.output
    assert "--tracked-only" in result.output
    assert "--remote-only" in result.output
    assert "--remote" in result.output


def test_status_no_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status with no stages shows appropriate message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "No stages registered" in result.output


# =============================================================================
# Pipeline Status Tests
# =============================================================================


def _helper_process() -> None:
    pathlib.Path("output.txt").write_text("done")


def _helper_stage_a() -> None:
    pathlib.Path("a.txt").write_text("output a")


def _helper_stage_b() -> None:
    pathlib.Path("b.txt").write_text("output b")


def test_status_shows_stale_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status shows stale stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "stale" in result.output
        assert "process" in result.output


def test_status_shows_cached_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows cached stages after run."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        executor.run(show_output=False)

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "cached" in result.output


def test_status_verbose_shows_all_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Verbose status shows all stages including cached."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_stage_a, name="stage_a", deps=["input.txt"], outs=["a.txt"])
        REGISTRY.register(_helper_stage_b, name="stage_b", deps=["input.txt"], outs=["b.txt"])

        executor.run(show_output=False)

        result = runner.invoke(cli.cli, ["status", "--verbose"])

        assert result.exit_code == 0
        assert "stage_a" in result.output
        assert "stage_b" in result.output


def test_status_specific_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status with stage argument filters to specific stage."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_stage_a, name="stage_a", deps=["input.txt"], outs=["a.txt"])
        REGISTRY.register(_helper_stage_b, name="stage_b", deps=["input.txt"], outs=["b.txt"])

        result = runner.invoke(cli.cli, ["status", "stage_a"])

        assert result.exit_code == 0
        assert "stage_a" in result.output


def test_status_unknown_stage_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status with unknown stage shows error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower()


# =============================================================================
# Tracked Files Tests
# =============================================================================


def test_status_shows_tracked_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows tracked files section with verbose."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        data_file = pathlib.Path("data.txt")
        data_file.write_text("content")
        file_hash = cache.hash_file(data_file)

        pvt_data = track.PvtData(path="data.txt", hash=file_hash, size=7)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        result = runner.invoke(cli.cli, ["status", "--verbose"])

        assert result.exit_code == 0
        assert "Tracked Files" in result.output
        assert "data.txt" in result.output
        assert "clean" in result.output


def test_status_shows_modified_tracked_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows modified tracked files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        data_file = pathlib.Path("data.txt")
        data_file.write_text("original")
        old_hash = cache.hash_file(data_file)

        pvt_data = track.PvtData(path="data.txt", hash=old_hash, size=8)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        data_file.write_text("modified content")

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "modified" in result.output


# =============================================================================
# Filter Options Tests
# =============================================================================


def test_status_stages_only(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--stages-only shows only pipeline status."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        data_file = pathlib.Path("data.txt")
        data_file.write_text("content")
        file_hash = cache.hash_file(data_file)

        pvt_data = track.PvtData(path="data.txt", hash=file_hash, size=7)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["status", "--stages-only"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "Tracked Files" not in result.output


def test_status_tracked_only(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--tracked-only shows only tracked files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        data_file = pathlib.Path("data.txt")
        data_file.write_text("content")
        file_hash = cache.hash_file(data_file)

        pvt_data = track.PvtData(path="data.txt", hash=file_hash, size=7)
        track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["status", "--tracked-only"])

        assert result.exit_code == 0
        assert "Tracked Files" in result.output
        assert "Pipeline Status" not in result.output


def test_status_remote_only_no_remotes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--remote-only without configured remotes shows error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status", "--remote-only"])

        assert result.exit_code != 0
        assert "No remotes configured" in result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_status_json_output(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--json outputs valid JSON."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert len(data["stages"]) == 1
        assert data["stages"][0]["name"] == "process"


def test_status_json_includes_suggestions(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--json includes suggestions when applicable."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "suggestions" in data
        assert any("pivot run" in s for s in data["suggestions"])


# =============================================================================
# Suggestions Tests
# =============================================================================


def test_status_shows_suggestions(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status shows actionable suggestions."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        REGISTRY.register(_helper_process, name="process", deps=["input.txt"], outs=["output.txt"])

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "Suggestions" in result.output
        assert "pivot run" in result.output


# =============================================================================
# Empty Section Behavior Tests
# =============================================================================


def test_status_tracked_only_no_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--tracked-only with no tracked files shows explicit message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status", "--tracked-only"])

        assert result.exit_code == 0
        assert "Tracked Files" in result.output
        assert "No tracked files" in result.output


def test_status_stages_only_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--stages-only with no stages shows explicit message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status", "--stages-only"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "No stages registered" in result.output


def test_status_json_includes_empty_arrays(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--json includes empty arrays for requested sections."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data, "Should include stages key even if empty"
        assert data["stages"] == []
        assert "tracked_files" in data, "Should include tracked_files key even if empty"
        assert data["tracked_files"] == []


def test_status_json_stages_only_empty(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--json --stages-only includes empty stages array."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["status", "--json", "--stages-only"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert data["stages"] == []
        assert "tracked_files" not in data, "Should not include unrequested sections"
