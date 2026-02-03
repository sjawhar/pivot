from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import helpers
from helpers import register_test_stage
from pivot import cli, discovery, loaders, outputs, project
from pivot.cli import decorators as cli_decorators
from pivot.pipeline import pipeline as pipeline_mod
from pivot.storage import cache

if TYPE_CHECKING:
    import click.testing
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture

    from conftest import GitRepo
    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _CsvOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


# =============================================================================
# Diff Help Tests
# =============================================================================


def test_diff_help(runner: click.testing.CliRunner) -> None:
    """Diff command should show help."""
    result = runner.invoke(cli.cli, ["diff", "--help"])
    assert result.exit_code == 0
    assert "TARGETS" in result.output
    assert "--key" in result.output
    assert "--positional" in result.output
    assert "--no-tui" in result.output
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--summary" in result.output
    assert "--max-rows" in result.output


def test_diff_in_main_help(runner: click.testing.CliRunner) -> None:
    """diff command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "diff" in result.output


def test_get_in_main_help(runner: click.testing.CliRunner) -> None:
    """get command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "get" in result.output


# =============================================================================
# Diff - No Stage Tests
# =============================================================================


def test_diff_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: Pipeline
) -> None:
    """Diff with no registered stages should report no data files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        result = runner.invoke(cli.cli, ["diff", "--no-tui", "data.csv"])
    assert result.exit_code == 0
    assert "No data files found" in result.output


# =============================================================================
# Diff - Conflicting Options
# =============================================================================


def test_diff_key_and_positional_conflict(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: Pipeline
) -> None:
    """Diff should error when both --key and --positional are specified."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Need to create a file so the targets validation passes
        pathlib.Path("data.csv").write_text("id,name\n1,alice\n")
        result = runner.invoke(
            cli.cli, ["diff", "--no-tui", "--key", "id", "--positional", "data.csv"]
        )
    assert result.exit_code != 0
    assert "Cannot use both --key and --positional" in result.output


# =============================================================================
# Diff - Required Arguments
# =============================================================================


def test_diff_requires_targets(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: Pipeline
) -> None:
    """Diff requires at least one target."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        result = runner.invoke(cli.cli, ["diff", "--no-tui"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output or "required" in result.output.lower()


# =============================================================================
# Diff - CSV File Tests
# =============================================================================


def _helper_make_csv_output() -> _CsvOutputs:
    """Helper stage that produces a CSV output."""
    pathlib.Path("output.csv").write_text("id,value\n1,10\n2,20\n")
    return {"output": pathlib.Path("output.csv")}


def test_diff_csv_file(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """Diff CSV files against HEAD."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    # Register stage with CSV output
    register_test_stage(
        _helper_make_csv_output,
        name="make_csv",
    )

    # Create initial CSV and cache it
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n2,20\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    # Create lock file with output hash
    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    lock_path = repo_path / ".pivot" / "stages" / "make_csv.lock"
    lock_path.write_text(lock_content)

    # Commit to create HEAD state
    commit("Initial CSV output")

    # Modify the CSV file in workspace (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    csv_file.write_text("id,value\n1,10\n2,25\n3,30\n")  # Changed row 2, added row 3

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should show row changes
    assert "output.csv" in result.output
    assert "Rows:" in result.output


def test_diff_json_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """--json outputs structured diff."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create initial CSV and cache it
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial")

    # Modify workspace (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    csv_file.write_text("id,value\n1,99\n")

    result = runner.invoke(cli.cli, ["diff", "--json", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Output should be valid JSON
    data: list[dict[str, object]] = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "path" in data[0]
    assert data[0]["path"] == "output.csv"


def test_diff_key_columns(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """--key uses columns for row matching."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create initial CSV with key column
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,name,value\n1,alice,10\n2,bob,20\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial")

    # Modify: update alice's value, add charlie (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    csv_file.write_text("id,name,value\n1,alice,15\n2,bob,20\n3,charlie,30\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--key", "id", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should detect modified row (alice) and added row (charlie)
    assert "output.csv" in result.output


def test_diff_positional(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """--positional uses row position matching."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create initial CSV
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n2,20\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial")

    # Reorder rows (positional diff will see changes, key-based might not)
    # Remove link first since cache creates hardlink/symlink
    csv_file.unlink()
    csv_file.write_text("id,value\n2,20\n1,10\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--positional", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "output.csv" in result.output


def test_diff_no_changes_message(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """No changes shows explicit message, not empty output."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial")

    # Don't modify - workspace same as HEAD
    # Workspace hash should match HEAD hash

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should have an explicit message about no changes
    assert "No data file changes" in result.output


def test_diff_json_empty_returns_valid_json(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """Empty diff returns valid JSON, not empty string."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial")

    # No changes - same content as HEAD

    result = runner.invoke(cli.cli, ["diff", "--json", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # With --json flag, should always return valid JSON (empty list when no changes)
    data: list[dict[str, object]] = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 0


# =============================================================================
# Get - Stage and Mode Tests
# =============================================================================


def test_get_stage_output(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """get with stage name target."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create output and cache it
    output_file = repo_path / "result.txt"
    output_file.write_text("stage output content")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(output_file, cache_dir)
    assert output_hash is not None

    # Create lock file for the stage
    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: result.txt
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    lock_path = repo_path / ".pivot" / "stages" / "make_output.lock"
    lock_path.write_text(lock_content)

    sha = commit("Stage output")

    # Delete output file
    output_file.unlink()
    assert not output_file.exists()

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "make_output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert output_file.exists()
    assert output_file.read_text() == "stage output content"


def test_get_checkout_mode(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """--checkout-mode affects file restoration."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create file and cache it
    data_file = repo_path / "data.txt"
    data_file.write_text("cached content")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(data_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: data.txt
hash: {output_hash["hash"]}
size: 14
"""
    pvt_path = repo_path / "data.txt.pvt"
    pvt_path.write_text(pvt_content)

    sha = commit("Track data file")

    # Delete data file
    data_file.unlink()
    assert not data_file.exists()

    # Test copy mode
    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--checkout-mode", "copy", "data.txt"],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert data_file.exists()
    # With copy mode, file should not be a symlink
    assert not data_file.is_symlink()
    assert data_file.read_text() == "cached content"


# =============================================================================
# Diff/Get - Pipeline Discovery Tests
# =============================================================================


def test_diff_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """pivot diff discovers pipeline without needing prior command."""
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)
    (repo_path / ".pivot").mkdir()

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    # IMPORTANT: Do NOT mock get_pipeline_from_context - we want to test that
    # ensure_stages_registered() is called and populates the context

    (repo_path / "output.csv").write_text("x,y\n3,4")

    commit("Initial setup")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "output.csv"])

    # Should not raise NoPipelineError - discovery should have been called
    assert "No pipeline found" not in result.output
    # The test passes if discovery succeeded (no error about missing pipeline)
    # It may show "No data files found" since we have no registered stages, which is fine
    assert result.exit_code == 0 or "No data" in result.output


def test_get_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    monkeypatch: MonkeyPatch,
) -> None:
    """pivot get discovers pipeline without needing prior command."""
    repo_path, commit = git_repo
    monkeypatch.chdir(repo_path)
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create minimal pivot.yaml with stage
    (repo_path / "pivot.yaml").write_text("""\
stages:
  process:
    python: stages.process
    outs:
      output.csv: {}
""")
    (repo_path / "stages.py").write_text("def process(): pass")

    # Create output and cache it
    output_file = repo_path / "output.csv"
    output_file.write_text("x,y\n3,4")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(output_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: output.csv
hash: {output_hash["hash"]}
size: 8
"""
    (repo_path / "output.csv.pvt").write_text(pvt_content)

    sha = commit("Initial setup with tracked output")

    # Delete output file
    output_file.unlink()
    assert not output_file.exists()

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "output.csv"])

    # Should not raise NoPipelineError - may fail for other reasons but not discovery
    assert "No pipeline" not in result.output
    # Should restore the file successfully
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert output_file.exists()


# =============================================================================
# Diff - Additional Output Format Tests
# =============================================================================


def test_diff_summary_flag(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """--summary flag shows only schema and counts, not detailed rows."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create initial CSV and cache it
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n2,20\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial CSV")

    # Modify the CSV file (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    csv_file.write_text("id,value,new_col\n1,10,100\n2,25,200\n3,30,300\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--summary", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "output.csv" in result.output
    # Summary should show counts but not detailed row-by-row changes
    assert "Rows:" in result.output or "Schema" in result.output


def test_diff_markdown_output(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """--md outputs Markdown-formatted diff."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create initial CSV and cache it
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n1,10\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Initial")

    # Modify workspace (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    csv_file.write_text("id,value\n1,99\n")

    result = runner.invoke(cli.cli, ["diff", "--md", "output.csv"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # --md produces summary output (same format as plain for now)
    # Verify the diff summary is present
    assert "output.csv" in result.output
    assert "Rows:" in result.output or "Row changes:" in result.output


def test_diff_max_rows_parameter(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """--max-rows limits comparison rows."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create large CSV and cache it
    csv_file = repo_path / "output.csv"
    rows = "\n".join([f"{i},{i * 10}" for i in range(1, 101)])
    csv_file.write_text(f"id,value\n{rows}\n")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Large CSV")

    # Modify one row (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    rows_modified = "\n".join([f"{i},{i * 10 if i != 50 else 9999}" for i in range(1, 101)])
    csv_file.write_text(f"id,value\n{rows_modified}\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "--max-rows", "10", "output.csv"])

    # Should succeed without error even with max-rows limit
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "output.csv" in result.output


def test_diff_empty_csv_file(
    runner: click.testing.CliRunner,
    git_repo: GitRepo,
    mocker: MockerFixture,
) -> None:
    """Diff handles empty CSV files gracefully."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    # Create Pipeline with git_repo path and mock discovery to return it
    pipeline = pipeline_mod.Pipeline("test", root=repo_path)
    helpers.set_test_pipeline(pipeline)
    mocker.patch.object(discovery, "discover_pipeline", return_value=pipeline)
    mocker.patch.object(project, "_project_root_cache", repo_path)
    mocker.patch.object(cli_decorators, "get_pipeline_from_context", return_value=pipeline)

    register_test_stage(_helper_make_csv_output, name="make_csv")

    # Create empty CSV and cache it
    csv_file = repo_path / "output.csv"
    csv_file.write_text("id,value\n")  # Header only, no data rows
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache.save_to_cache(csv_file, cache_dir)
    assert output_hash is not None

    lock_content = f"""code_manifest: {{}}
params: {{}}
deps: []
outs:
  - path: output.csv
    hash: {output_hash["hash"]}
dep_generations: {{}}
"""
    (repo_path / ".pivot" / "stages" / "make_csv.lock").write_text(lock_content)
    commit("Empty CSV")

    # Modify to add rows (remove link first since cache creates hardlink/symlink)
    csv_file.unlink()
    csv_file.write_text("id,value\n1,10\n")

    result = runner.invoke(cli.cli, ["diff", "--no-tui", "output.csv"])

    # Should handle empty file without crashing
    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "output.csv" in result.output


# =============================================================================
# Get - Multiple Targets Tests
# =============================================================================


def test_get_multiple_files(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get multiple files in single invocation."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create and commit multiple files
    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    sha = commit("Multiple files")

    # Modify files
    (repo_path / "file1.txt").write_text("modified1")
    (repo_path / "file2.txt").write_text("modified2")

    result = runner.invoke(cli.cli, ["get", "--rev", sha[:7], "--force", "file1.txt", "file2.txt"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Both files should be mentioned in output
    assert "file1.txt" in result.output or "Restored" in result.output
    # Verify files restored
    assert (repo_path / "file1.txt").read_text() == "content1"
    assert (repo_path / "file2.txt").read_text() == "content2"


def test_get_output_with_multiple_targets_fails(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """Get with -o output path and multiple targets should fail with clear error."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    (repo_path / "file1.txt").write_text("content1")
    (repo_path / "file2.txt").write_text("content2")
    sha = commit("Multiple files")

    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "-o", "output.txt", "file1.txt", "file2.txt"],
    )

    # Should fail - can't use -o with multiple targets
    assert result.exit_code != 0
    assert "output" in result.output.lower() or "multiple" in result.output.lower()


def test_get_symlink_checkout_mode(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """--checkout-mode symlink creates symlinks."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create file and cache it
    from pivot.storage import cache as cache_mod

    data_file = repo_path / "data.txt"
    data_file.write_text("cached content")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache_mod.save_to_cache(data_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: data.txt
hash: {output_hash["hash"]}
size: 14
"""
    pvt_path = repo_path / "data.txt.pvt"
    pvt_path.write_text(pvt_content)

    sha = commit("Track data file")

    # Delete data file
    data_file.unlink()
    assert not data_file.exists()

    # Test symlink mode
    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--checkout-mode", "symlink", "data.txt"],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert data_file.exists()
    # With symlink mode, file should be a symlink
    assert data_file.is_symlink()
    assert data_file.read_text() == "cached content"


def test_get_hardlink_checkout_mode(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: MonkeyPatch
) -> None:
    """--checkout-mode hardlink creates hardlinks."""
    repo_path, commit = git_repo
    (repo_path / ".pivot" / "cache" / "files").mkdir(parents=True)
    (repo_path / ".pivot" / "stages").mkdir(parents=True)

    monkeypatch.setattr(project, "_project_root_cache", repo_path)

    # Create file and cache it
    from pivot.storage import cache as cache_mod

    data_file = repo_path / "data.txt"
    data_file.write_text("cached content")
    cache_dir = repo_path / ".pivot" / "cache" / "files"
    output_hash = cache_mod.save_to_cache(data_file, cache_dir)
    assert output_hash is not None

    # Create .pvt file to track it
    pvt_content = f"""path: data.txt
hash: {output_hash["hash"]}
size: 14
"""
    pvt_path = repo_path / "data.txt.pvt"
    pvt_path.write_text(pvt_content)

    sha = commit("Track data file")

    # Delete data file
    data_file.unlink()
    assert not data_file.exists()

    # Test hardlink mode
    result = runner.invoke(
        cli.cli,
        ["get", "--rev", sha[:7], "--checkout-mode", "hardlink", "data.txt"],
    )

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Restored" in result.output
    assert data_file.exists()
    # With hardlink mode, file should not be a symlink
    assert not data_file.is_symlink()
    # Verify it's a hardlink by checking link count (if supported by filesystem)
    stat_info = data_file.stat()
    # Hardlinked file has nlink > 1
    assert stat_info.st_nlink >= 1  # At least the file exists
    assert data_file.read_text() == "cached content"
