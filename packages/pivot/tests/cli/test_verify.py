from __future__ import annotations

import json
import pathlib
import shutil
from typing import TYPE_CHECKING

from pivot import cli, compose, exceptions, executor, project
from pivot.storage import cache, lock, track

if TYPE_CHECKING:
    import click.testing
    import pytest
    from pytest_mock import MockerFixture


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
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def process(input_file: str) -> str:
    return "done"


@compose.stage
def stage_a(input_file: str) -> str:
    return "output a"


@compose.stage
def stage_b(a_file: str) -> str:
    return "output b"


# Raw input "input.txt" lives at data/raw/input.txt (Pipeline.input default path)
_INPUT_REL = pathlib.Path("data", "raw", "input.txt")
_OUTPUT_REL = pathlib.Path("data", "test", "process.txt")


def _write_input(root: pathlib.Path, content: str = "data") -> pathlib.Path:
    path = root / _INPUT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _register_process(pipeline: compose.Pipeline) -> None:
    with pipeline:
        process(pipeline.input("input.txt", t=str))


def _register_a_b(pipeline: compose.Pipeline) -> None:
    with pipeline:
        stage_b(stage_a(pipeline.input("input.txt", t=str)))


def _process_lock(pipeline: compose.Pipeline) -> lock.StageLock:
    return lock.StageLock("test/process", lock.get_stages_dir(pipeline.state_dir))


def _run_process(pipeline: compose.Pipeline) -> None:
    """Register `process`, write its input, and run it to completion."""
    _write_input(pipeline.root)
    _register_process(pipeline)
    results = executor.run(pipeline=pipeline)
    assert results["test/process"]["status"] == "ran", results


def _delete_output_and_cache_entry(pipeline: compose.Pipeline) -> None:
    lock_data = _process_lock(pipeline).read()
    assert lock_data is not None
    output_hash = next(iter(lock_data["output_hashes"].values()))
    (pipeline.root / _OUTPUT_REL).unlink()
    cache_path = cache.get_cache_path(
        pipeline.root / ".pivot" / "cache" / "files", output_hash["hash"]
    )
    if cache_path.exists():
        cache_path.unlink()


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
    # Create an empty pipeline.py so the pipeline can be discovered
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "pipeline.py").write_text(
        "from pivot.compose import Pipeline\n\nwith Pipeline('test') as pipeline:\n    pass\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "_project_root_cache", None)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code != 0
    assert "No stages registered" in result.output


# =============================================================================
# Basic Verification Tests
# =============================================================================


def test_verify_all_cached_exits_0(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify with all stages cached and files present exits 0."""
    _run_process(mock_discovery)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 0, result.output
    assert "Verification passed" in result.output


def test_verify_stale_code_exits_1(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify with stale stage (code changed) exits 1."""
    _run_process(mock_discovery)

    # Modify the lock file to simulate code change
    stage_lock = _process_lock(mock_discovery)
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["code_manifest"]["process"] = "changed_hash"
    stage_lock.write(lock_data)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1, result.output
    assert "Code changed" in result.output


def test_verify_stale_params_exits_1(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify with stale stage (params changed) exits 1."""
    _run_process(mock_discovery)

    # Modify the lock file to simulate params change
    stage_lock = _process_lock(mock_discovery)
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["params"]["new_param"] = "value"
    stage_lock.write(lock_data)

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1, result.output
    assert "Params changed" in result.output


def test_verify_stale_deps_exits_1(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify with stale stage (deps changed) exits 1."""
    _run_process(mock_discovery)

    # Modify input file to change deps
    _write_input(mock_discovery.root, "modified data")

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1, result.output
    assert "Input dependencies changed" in result.output


def test_verify_missing_dep_file_exits_1(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify with missing dependency file exits 1."""
    _run_process(mock_discovery)

    # Delete the dependency file
    (mock_discovery.root / _INPUT_REL).unlink()

    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1, result.output
    assert "Missing deps" in result.output and "input.txt" in result.output


def test_verify_never_run_exits_1(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify with stage that was never run exits 1."""
    _write_input(mock_discovery.root)
    _register_process(mock_discovery)

    # Don't run - no lock file exists
    result = runner.invoke(cli.cli, ["verify"])

    assert result.exit_code == 1, result.output
    assert "No previous run" in result.output


# =============================================================================
# Stage Filtering Tests
# =============================================================================


def test_verify_specific_stage(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify train verifies only the train stage."""
    _write_input(mock_discovery.root)
    _register_a_b(mock_discovery)

    # Run to cache both
    executor.run(pipeline=mock_discovery)

    # Verify only stage_a
    result = runner.invoke(cli.cli, ["verify", "test/stage_a"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output
    assert "test/stage_b" not in result.output


def test_verify_bare_stage_name_resolves(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify accepts a bare stage name in a single-pipeline project (names.py design)."""
    _write_input(mock_discovery.root)
    _register_a_b(mock_discovery)
    executor.run(pipeline=mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "stage_a"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output


def test_verify_multiple_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify stage_a stage_b verifies both stages."""
    _write_input(mock_discovery.root)
    _register_a_b(mock_discovery)

    # Run to cache both
    executor.run(pipeline=mock_discovery)

    # Verify both
    result = runner.invoke(cli.cli, ["verify", "test/stage_a", "test/stage_b"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output
    assert "test/stage_b" in result.output


def test_verify_nonexistent_stage_errors(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify nonexistent exits non-zero with stage not found error."""
    _write_input(mock_discovery.root)
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "nonexistent"])

    assert result.exit_code != 0
    assert "nonexistent" in result.output.lower()


# =============================================================================
# Allow-Missing Mode Tests
# =============================================================================


def test_verify_allow_missing_no_remote_errors(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot verify --allow-missing with no remote configured exits non-zero."""
    _run_process(mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code != 0
    assert "remote" in result.output.lower()


def test_verify_allow_missing_file_on_remote_passes(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """pivot verify --allow-missing with missing local file that exists on remote exits 0."""
    _run_process(mock_discovery)

    # Delete output file and its cache entry
    _delete_output_and_cache_entry(mock_discovery)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 0, result.output


def test_verify_allow_missing_file_not_on_remote_fails(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """pivot verify --allow-missing with missing local file not on remote exits 1."""
    _run_process(mock_discovery)

    # Delete output file and its cache entry
    _delete_output_and_cache_entry(mock_discovery)

    _setup_mock_remote(mocker, files_exist_on_remote=False)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 1, result.output
    assert "Missing files:" in result.output
    assert str(_OUTPUT_REL) in result.output


def test_verify_allow_missing_code_changed_still_fails(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """pivot verify --allow-missing with code changes still exits 1."""
    _run_process(mock_discovery)

    # Modify the lock file to simulate code change
    stage_lock = _process_lock(mock_discovery)
    lock_data = stage_lock.read()
    assert lock_data is not None
    lock_data["code_manifest"]["process"] = "changed_hash"
    stage_lock.write(lock_data)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code == 1, result.output
    assert "Code changed" in result.output


def test_verify_allow_missing_without_prior_run(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing works when dep files missing (CI scenario).

    This tests the fix for the bug where DAG validation raised
    DependencyNotFoundError before --allow-missing logic could run.

    Scenario: CI clone has no dep files yet, but they would exist after setup.
    With validate=False, DAG building doesn't error on missing deps.
    """
    _register_process(mock_discovery)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should NOT raise DependencyNotFoundError during DAG building
    assert "does not exist on disk" not in result.output, f"Got error: {result.output}"
    # Verification fails (exit 1) because stage is stale, but command completes
    assert result.exit_code == 1, result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_verify_json_output_passed(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--json outputs valid JSON with passed=true when all cached."""
    _run_process(mock_discovery)

    result = runner.invoke(cli.cli, ["verify", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "passed" in data
    assert data["passed"] is True
    assert "stages" in data
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "test/process"


def test_verify_json_output_failed(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--json outputs valid JSON with passed=false and reasons when stale."""
    _write_input(mock_discovery.root)
    _register_process(mock_discovery)

    # Don't run - stage is stale
    result = runner.invoke(cli.cli, ["verify", "--json"])

    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    assert "passed" in data
    assert data["passed"] is False
    assert "stages" in data
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "test/process"
    assert "reason" in data["stages"][0]


def test_verify_json_includes_all_keys(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--json output always includes passed, stages, and failure reasons."""
    _run_process(mock_discovery)

    # Modify deps to make stale
    _write_input(mock_discovery.root, "modified")

    result = runner.invoke(cli.cli, ["verify", "--json"])

    assert result.exit_code == 1, result.output
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
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot --quiet verify produces no output when all stages are cached."""
    _run_process(mock_discovery)

    result = runner.invoke(cli.cli, ["--quiet", "verify"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "", "Quiet mode should suppress output when passing"


def test_verify_quiet_exits_1_when_failed(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """pivot --quiet verify exits 1 when verification fails."""
    _write_input(mock_discovery.root)
    _register_process(mock_discovery)

    # Don't run - stage is stale
    result = runner.invoke(cli.cli, ["--quiet", "verify"])

    assert result.exit_code == 1
    assert result.output.strip() == "", "Quiet mode should suppress output"


# =============================================================================
# PVT Hash Fallback Tests
# =============================================================================


def test_verify_allow_missing_uses_pvt_hash_for_deps(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing uses .pvt hash when dep file is missing."""
    _run_process(mock_discovery)
    input_path = mock_discovery.root / _INPUT_REL

    # Track the input file (create .pvt)
    input_hash, _ = cache.hash_file(input_path)
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(input_path.parent / "input.txt.pvt", pvt_data)

    # Delete the actual input file (simulating CI without data)
    input_path.unlink()

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should NOT fail with "Missing deps" - should use .pvt hash
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected pass, got: {result.output}"


def test_verify_allow_missing_uses_lock_hash_when_no_pvt(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing uses lock file hash when dep missing and no .pvt exists.

    This is the key CI scenario: dep files don't exist locally, no .pvt tracking,
    but the dep hashes are available in the remote cache.
    """
    _run_process(mock_discovery)

    # Delete the input file (simulating CI without data) - NO .pvt file created
    (mock_discovery.root / _INPUT_REL).unlink()

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should NOT fail with "Missing deps" - should use lock file hash and verify on remote
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected pass, got: {result.output}"


def test_verify_allow_missing_fails_when_dep_not_on_remote(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing fails when dep hash not on remote (no .pvt)."""
    _run_process(mock_discovery)

    # Delete the input file - NO .pvt file
    (mock_discovery.root / _INPUT_REL).unlink()

    _setup_mock_remote(mocker, files_exist_on_remote=False)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should fail - dep hash not on remote
    assert result.exit_code == 1, result.output
    assert "Missing files:" in result.output or "input.txt" in result.output


def test_verify_allow_missing_uses_pvt_hash_for_nested_dep(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing uses directory .pvt manifest for nested file dep."""
    # Raw data directory with one file the stage depends on
    data_dir = mock_discovery.root / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "file.csv").write_text("content")

    with mock_discovery:
        process(mock_discovery.input("file.csv", t=str))
    run_result = runner.invoke(cli.cli, ["repro"])
    assert run_result.exit_code == 0, f"Run failed: {run_result.output}"

    # Track the directory (create .pvt with manifest)
    dir_hash, manifest = cache.hash_directory(data_dir)
    pvt_data = track.PvtData(
        path="raw",
        hash=dir_hash,
        size=7,
        num_files=1,
        manifest=manifest,
    )
    track.write_pvt_file(data_dir.parent / "raw.pvt", pvt_data)

    # Delete the actual data directory (simulating CI without data)
    shutil.rmtree(data_dir)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should use manifest entry hash for data/raw/file.csv
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected pass, got: {result.output}"


# =============================================================================
# Path Resolution and Remote Error Wrapping Tests
# =============================================================================


def test_verify_allow_missing_resolves_paths_relative_to_project_root(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify --allow-missing resolves dep paths against project root, not cwd."""
    _run_process(mock_discovery)

    # Delete the input file to trigger allow-missing path
    (mock_discovery.root / _INPUT_REL).unlink()

    # Change cwd into a subdirectory (away from project root)
    subdir = mock_discovery.root / "sub"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    _setup_mock_remote(mocker, files_exist_on_remote=True)

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    # Should not error about missing deps — path resolution uses project root
    assert "Missing deps" not in result.output, f"Got: {result.output}"


def test_verify_allow_missing_wraps_remote_creation_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing wraps non-RemoteError from create_remote_from_name."""
    _write_input(mock_discovery.root)
    _register_process(mock_discovery)

    mocker.patch("pivot.remote.config.list_remotes", return_value={"default": "s3://bucket/cache"})
    mocker.patch(
        "pivot.remote.sync.create_remote_from_name",
        side_effect=RuntimeError("connection refused"),
    )

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code != 0
    assert "Failed to create remote connection" in result.output


def test_verify_allow_missing_preserves_remote_error_subclass(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    mocker: MockerFixture,
) -> None:
    """verify --allow-missing preserves RemoteError subclasses (not re-wrapped)."""
    _write_input(mock_discovery.root)
    _register_process(mock_discovery)

    mocker.patch("pivot.remote.config.list_remotes", return_value={"default": "s3://bucket/cache"})
    mocker.patch(
        "pivot.remote.sync.create_remote_from_name",
        side_effect=exceptions.RemoteNotFoundError("no-such"),
    )

    result = runner.invoke(cli.cli, ["verify", "--allow-missing"])

    assert result.exit_code != 0
    assert "no-such" in result.output
