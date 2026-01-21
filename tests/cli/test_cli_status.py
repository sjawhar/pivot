from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot import status as status_mod
from pivot.storage import cache, track
from pivot.types import RemoteSyncInfo

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _ATxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


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


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def _helper_stage_a(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _ATxtOutputs:
    _ = input_file
    pathlib.Path("a.txt").write_text("output a")
    return _ATxtOutputs(output=pathlib.Path("a.txt"))


def _helper_stage_b(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _BTxtOutputs:
    _ = input_file
    pathlib.Path("b.txt").write_text("output b")
    return _BTxtOutputs(output=pathlib.Path("b.txt"))


def test_status_shows_stale_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status shows stale stages."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

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

        register_test_stage(_helper_process, name="process")

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

        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")

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

        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")

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

        register_test_stage(_helper_process, name="process")

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

        register_test_stage(_helper_process, name="process")

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

        register_test_stage(_helper_process, name="process")

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

        register_test_stage(_helper_process, name="process")

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

        register_test_stage(_helper_process, name="process")

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


# =============================================================================
# Quiet Mode Tests
# =============================================================================


def test_status_quiet_no_output_when_clean(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot --quiet status produces no output when all stages are cached."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run once to cache
        executor.run(show_output=False)

        result = runner.invoke(cli.cli, ["--quiet", "status"])

        assert result.exit_code == 0
        assert result.output.strip() == "", "Quiet mode should suppress output when clean"


def test_status_quiet_exits_1_when_stale(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot --quiet status exits 1 when stages are stale."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Don't run - stage is stale
        result = runner.invoke(cli.cli, ["--quiet", "status"])

        assert result.exit_code == 1, "Should exit 1 when stages are stale"
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_status_quiet_exits_1_when_modified(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot --quiet status exits 1 when files are modified."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        # Run to cache
        executor.run(show_output=False)

        # Modify input file
        pathlib.Path("input.txt").write_text("modified data")

        result = runner.invoke(cli.cli, ["--quiet", "status"])

        # Stage should now be stale due to modified input
        assert result.exit_code == 1, "Should exit 1 when files are modified"
        assert result.output.strip() == "", "Quiet mode should suppress output"


# =============================================================================
# Remote Status Tests
# =============================================================================


def test_status_remote_with_configured_remote(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """--remote shows sync status when remote is configured."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Mock the remote status function
        mock_remote_status = RemoteSyncInfo(
            name="default",
            url="s3://mybucket/cache",
            push_count=5,
            pull_count=3,
        )
        mocker.patch.object(
            status_mod, "get_remote_status", autospec=True, return_value=mock_remote_status
        )

        result = runner.invoke(cli.cli, ["status", "--remote"])

        assert result.exit_code == 0
        assert "Remote Status" in result.output
        assert "5 to push" in result.output
        assert "3 to pull" in result.output


def test_status_remote_only_with_remote(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """--remote-only shows only remote sync counts."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")

        register_test_stage(_helper_process, name="process")

        mock_remote_status = RemoteSyncInfo(
            name="myremote",
            url="s3://bucket/path",
            push_count=10,
            pull_count=2,
        )
        mocker.patch.object(
            status_mod, "get_remote_status", autospec=True, return_value=mock_remote_status
        )

        result = runner.invoke(cli.cli, ["status", "--remote-only"])

        assert result.exit_code == 0
        assert "Remote Status" in result.output
        # Should NOT show pipeline or tracked files sections
        assert "Pipeline Status" not in result.output
        assert "Tracked Files" not in result.output


def test_status_json_with_remote(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """JSON output includes remote status when --remote is used."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        mock_remote_status = RemoteSyncInfo(
            name="default",
            url="s3://bucket/cache",
            push_count=7,
            pull_count=4,
        )
        mocker.patch.object(
            status_mod, "get_remote_status", autospec=True, return_value=mock_remote_status
        )

        result = runner.invoke(cli.cli, ["status", "--json", "--remote"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "remote" in data
        assert data["remote"]["name"] == "default"
        assert data["remote"]["push_count"] == 7
        assert data["remote"]["pull_count"] == 4
