from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, executor, loaders, outputs
from pivot.storage import cache, lock

if TYPE_CHECKING:
    import click.testing
    import pytest
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline


def _setup_mock_remote(mocker: MockerFixture, *, files_exist_on_remote: bool) -> None:
    """Set up mocks for remote configuration and S3Remote.bulk_exists."""
    mocker.patch("pivot.remote.config.list_remotes", return_value={"default": "s3://bucket/cache"})
    mocker.patch("pivot.remote.config.get_default_remote", return_value="default")
    mocker.patch("pivot.remote.config.get_remote_url", return_value="s3://bucket/cache")

    mock_remote_class = mocker.patch("pivot.remote.storage.S3Remote")
    mock_remote = mock_remote_class.return_value

    async def mock_bulk_exists(hashes: list[str], concurrency: int = 20) -> dict[str, bool]:
        return dict.fromkeys(hashes, files_exist_on_remote)

    mock_remote.bulk_exists = mock_bulk_exists


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _ATxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


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
    a_file: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _BTxtOutputs:
    _ = a_file
    pathlib.Path("b.txt").write_text("output b")
    return _BTxtOutputs(output=pathlib.Path("b.txt"))


class _DirDepOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_dir_dep_stage(
    data_file: Annotated[pathlib.Path, outputs.Dep("data/file.csv", loaders.PathOnly())],
) -> _DirDepOutputs:
    _ = data_file
    pathlib.Path("output.txt").write_text("done")
    return _DirDepOutputs(output=pathlib.Path("output.txt"))


# =============================================================================
# Help and Basic Tests
# =============================================================================


def test_verify_help(runner: click.testing.CliRunner) -> None:
    """Verify command should show help."""
    result = runner.invoke(cli.cli, ["verify", "--help"])

    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--allow-missing" in result.output


def test_verify_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify with no stages shows appropriate error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    # Create empty pivot.yaml so pipeline can be discovered
    (tmp_path / "pivot.yaml").write_text("stages: {}")

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code != 0
    assert "No stages registered" in result.output


# =============================================================================
# Basic Verification Tests
# =============================================================================


def test_verify_all_cached_exits_0(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify with all stages cached and files present exits 0."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 0
    assert "Verification passed" in result.output


def test_verify_stale_code_exits_1(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify with stale stage (code changed) exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Modify the lock file to simulate code change
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["code_manifest"]["process"] = "changed_hash"
    stage_lock.write(lock_data)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1
    assert "Code changed" in result.output


def test_verify_stale_params_exits_1(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify with stale stage (params changed) exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Modify the lock file to simulate params change
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["params"]["new_param"] = "value"
    stage_lock.write(lock_data)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1
    assert "Params changed" in result.output


def test_verify_stale_deps_exits_1(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify with stale stage (deps changed) exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Modify input file to change deps
    (tmp_path / "input.txt").write_text("modified data")

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1
    assert "Input dependencies changed" in result.output


def test_verify_missing_dep_file_exits_1(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify with missing dependency file exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Delete the dependency file
    (tmp_path / "input.txt").unlink()

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1
    assert "Missing deps" in result.output and "input.txt" in result.output


def test_verify_never_run_exits_1(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify with stage that was never run exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Don't run - no lock file exists
    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1
    assert "No previous run" in result.output


# =============================================================================
# Stage Filtering Tests
# =============================================================================


def test_verify_specific_stage(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify train verifies only the train stage."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    # Run to cache both
    executor.run(pipeline=mock_discovery)

    # Verify only stage_a
    result = runner.invoke(cli.cli, ["verify", "stage_a"])

    assert result.exit_code == 0
    assert "stage_a" in result.output


def test_verify_multiple_stages(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify stage_a stage_b verifies both stages."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_stage_a, name="stage_a")
    register_test_stage(_helper_stage_b, name="stage_b")

    # Run to cache both
    executor.run(pipeline=mock_discovery)

    # Verify both
    result = runner.invoke(cli.cli, ["verify", "stage_a", "stage_b"])

    assert result.exit_code == 0
    assert "stage_a" in result.output
    assert "stage_b" in result.output


def test_verify_nonexistent_stage_errors(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify nonexistent exits non-zero with stage not found error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    result = runner.invoke(cli.cli, ["verify", "nonexistent"])

    assert result.exit_code != 0
    assert "nonexistent" in result.output.lower()


# =============================================================================
# Allow-Missing Mode Tests
# =============================================================================


def test_verify_allow_missing_no_remote_errors(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify --allow-missing with no remote configured exits non-zero."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code != 0
    assert "remote" in result.output.lower()


def test_verify_allow_missing_file_on_remote_passes(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify --allow-missing with missing local file that exists on remote exits 0."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Get the output hash before deleting
    state_dir = tmp_path / ".pivot"
    cache_dir = tmp_path / ".pivot/cache"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    assert lock_data is not None
    output_hash = list(lock_data["output_hashes"].values())[0]
    assert output_hash is not None

    # Delete output file and its cache entry
    (tmp_path / "output.txt").unlink()
    cache_path = cache.get_cache_path(cache_dir / "files", output_hash["hash"])
    if cache_path.exists():
        cache_path.unlink()

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 0


def test_verify_allow_missing_file_not_on_remote_fails(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify --allow-missing with missing local file not on remote exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Delete output file and its cache entry
    (tmp_path / "output.txt").unlink()
    state_dir = tmp_path / ".pivot"
    cache_dir = tmp_path / ".pivot/cache"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    assert lock_data is not None
    output_hash = list(lock_data["output_hashes"].values())[0]
    assert output_hash is not None
    cache_path = cache.get_cache_path(cache_dir / "files", output_hash["hash"])
    if cache_path.exists():
        cache_path.unlink()

    _setup_mock_remote(mocker, files_exist_on_remote=False)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 1
    assert "Missing files:" in result.output


def test_verify_allow_missing_code_changed_still_fails(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot verify --allow-missing with code changes still exits 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Modify the lock file to simulate code change
    state_dir = tmp_path / ".pivot"
    stage_lock = lock.StageLock("process", lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["code_manifest"]["process"] = "changed_hash"
    stage_lock.write(lock_data)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 1
    assert "Code changed" in result.output


def test_verify_allow_missing_without_prior_run(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify --allow-missing works when dep files missing (CI scenario).

    This tests the fix for the bug where DAG validation raised
    DependencyNotFoundError before --allow-missing logic could run.

    Scenario: CI clone has no dep files yet, but they would exist after setup.
    With validate=False, DAG building doesn't error on missing deps.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    register_test_stage(_helper_process, name="process")

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should NOT raise DependencyNotFoundError during DAG building
    assert "does not exist on disk" not in result.output, f"Got error: {result.output}"
    # Verification fails (exit 1) because stage is stale, but command completes
    assert result.exit_code == 1


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_verify_json_output_passed(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json outputs valid JSON with passed=true when all cached."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "passed" in data
    assert data["passed"] is True
    assert "stages" in data
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "process"


def test_verify_json_output_failed(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json outputs valid JSON with passed=false and reasons when stale."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Don't run - stage is stale
    result = runner.invoke(cli.cli, ["verify", "--json"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert "passed" in data
    assert data["passed"] is False
    assert "stages" in data
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "process"
    assert "reason" in data["stages"][0]


def test_verify_json_includes_all_keys(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json output always includes passed, stages, and failure reasons."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    # Modify deps to make stale
    (tmp_path / "input.txt").write_text("modified")

    result = runner.invoke(cli.cli, ["verify", "--json"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert "passed" in data
    assert "stages" in data
    # Each stage should have name, status, and reason
    for stage in data["stages"]:
        assert "name" in stage
        assert "status" in stage
        assert "reason" in stage


# =============================================================================
# Quiet Mode Tests
# =============================================================================


def test_verify_quiet_no_output_when_passed(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot --quiet verify produces no output when all stages are cached."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Run to cache
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["--quiet", "verify"])

    assert result.exit_code == 0
    assert result.output.strip() == "", "Quiet mode should suppress output when passing"


def test_verify_quiet_exits_1_when_failed(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot --quiet verify exits 1 when verification fails."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / "input.txt").write_text("data")

    register_test_stage(_helper_process, name="process")

    # Don't run - stage is stale
    result = runner.invoke(cli.cli, ["--quiet", "verify"])

    assert result.exit_code == 1
    assert result.output.strip() == "", "Quiet mode should suppress output"


# =============================================================================
# PVT Hash Fallback Tests
# =============================================================================


def test_verify_allow_missing_uses_pvt_hash_for_deps(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify --allow-missing uses .pvt hash when dep file is missing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create input.txt and run stage to cache
    (tmp_path / "input.txt").write_text("data")
    register_test_stage(_helper_process, name="process")
    executor.run(pipeline=mock_discovery)

    # Track the input file (create .pvt)
    from pivot.storage import track

    input_hash = cache.hash_file(tmp_path / "input.txt")
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(tmp_path / "input.txt.pvt", pvt_data)

    # Delete the actual input file (simulating CI without data)
    (tmp_path / "input.txt").unlink()

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should NOT fail with "Missing deps" - should use .pvt hash
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected pass, got: {result.output}"


def test_verify_allow_missing_uses_pvt_hash_for_nested_dep(
    mock_discovery: Pipeline,
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify --allow-missing uses directory .pvt manifest for nested file dep."""
    # Change to tmp_path (matches mock_discovery pipeline root)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()

    # Create data directory with file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file.csv").write_text("content")

    register_test_stage(_helper_dir_dep_stage, name="process")
    run_result = runner.invoke(cli.cli, ["repro"])
    assert run_result.exit_code == 0, f"Run failed: {run_result.output}"

    # Track the directory (create .pvt with manifest)
    from pivot.storage import track

    dir_hash, manifest = cache.hash_directory(data_dir)
    pvt_data = track.PvtData(
        path="data",
        hash=dir_hash,
        size=7,
        num_files=1,
        manifest=manifest,
    )
    track.write_pvt_file(tmp_path / "data.pvt", pvt_data)

    # Delete the actual data directory (simulating CI without data)
    import shutil

    shutil.rmtree(data_dir)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should use manifest entry hash for data/file.csv
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected pass, got: {result.output}"
