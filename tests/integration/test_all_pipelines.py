"""Integration tests for --all flag across multiple pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import isolated_pivot_dir
from pivot import cli
from pivot.storage import lock

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner


# =============================================================================
# Pipeline code generators
# =============================================================================


def _make_noop_pipeline_code(
    name: str, stage_name: str, out_annotation: str, write_path: str
) -> str:
    """Generate a simple pipeline with one stage that writes an output file.

    Args:
        name: Pipeline name.
        stage_name: Name of the stage function.
        out_annotation: Output path for Out() annotation (pipeline-relative).
        write_path: Path the stage writes to (project-relative, since workers
            chdir to project root).
    """
    return f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("{name}")

class _Output(TypedDict):
    data: Annotated[Path, Out("{out_annotation}", loaders.PathOnly())]

def {stage_name}() -> _Output:
    Path("{write_path}").parent.mkdir(parents=True, exist_ok=True)
    Path("{write_path}").write_text("{stage_name} output")
    return _Output(data=Path("{write_path}"))

pipeline.register({stage_name})
'''


def _setup_multi_pipeline_project(root: pathlib.Path) -> None:
    """Create a project with two sub-pipelines using different state_dirs.

    - alpha/: pipeline with stage_a, uses root .pivot (default state_dir)
    - beta/: pipeline with stage_b, uses beta/.pivot (separate state_dir)
    """
    # Alpha pipeline — uses project root .pivot
    alpha = root / "alpha"
    alpha.mkdir()
    (alpha / "pipeline.py").write_text(
        _make_noop_pipeline_code("alpha", "stage_a", "output_a.txt", "alpha/output_a.txt")
    )

    # Beta pipeline — uses its own .pivot
    beta = root / "beta"
    beta.mkdir()
    (beta / ".pivot").mkdir()
    (beta / "pipeline.py").write_text(
        _make_noop_pipeline_code("beta", "stage_b", "output_b.txt", "beta/output_b.txt")
    )


# =============================================================================
# Tests
# =============================================================================


def test_repro_all_discovers_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all --dry-run discovers stages from all sub-pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--dry-run"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "stage_a" in result.output, "stage_a from alpha pipeline not found"
        assert "stage_b" in result.output, "stage_b from beta pipeline not found"


def test_repro_all_executes_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all runs stages from all discovered pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "stage_a" in result.output, "stage_a not in output"
        assert "stage_b" in result.output, "stage_b not in output"

        # Verify output files were created
        assert (cwd / "alpha" / "output_a.txt").exists(), "Alpha output not created"
        assert (cwd / "beta" / "output_b.txt").exists(), "Beta output not created"


def test_repro_all_writes_locks_to_correct_state_dir(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """Lock files are written to each pipeline's own .pivot/stages/."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all"])
        assert result.exit_code == 0, f"Expected success, got: {result.output}"

        # Beta's lock file should be in beta/.pivot/stages/
        beta_stages_dir = lock.get_stages_dir(cwd / "beta" / ".pivot")
        beta_lock = lock.StageLock("stage_b", beta_stages_dir)
        assert beta_lock.read() is not None, "Lock file not found in beta's state_dir"


def test_status_all_shows_all_pipelines(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """status --all shows stages from all pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "stage_a" in result.output, "stage_a not in status output"
        assert "stage_b" in result.output, "stage_b not in status output"


def test_verify_all_with_mixed_state_dirs(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """verify --all reads lock files from correct per-pipeline state_dirs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        # Run first to create locks
        run_result = runner.invoke(cli.cli, ["repro", "--all"])
        assert run_result.exit_code == 0, f"repro failed: {run_result.output}"

        result = runner.invoke(cli.cli, ["verify", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "stage_a" in result.output, "stage_a not in verify output"
        assert "stage_b" in result.output, "stage_b not in verify output"


def test_commit_all_after_no_commit_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """commit --all promotes pending locks to correct per-stage state_dirs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        # Run with --no-commit to create pending locks
        run_result = runner.invoke(cli.cli, ["repro", "--all", "--no-commit"])
        assert run_result.exit_code == 0, f"repro failed: {run_result.output}"

        # Commit pending locks
        commit_result = runner.invoke(cli.cli, ["commit", "--all"])
        assert commit_result.exit_code == 0, f"commit failed: {commit_result.output}"

        # Verify beta's production lock is in beta/.pivot/stages/
        beta_stages_dir = lock.get_stages_dir(cwd / "beta" / ".pivot")
        beta_lock = lock.StageLock("stage_b", beta_stages_dir)
        assert beta_lock.read() is not None, "Production lock not in beta's state_dir after commit"


def test_all_flag_with_single_pipeline(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--all works gracefully when the project has only one pipeline."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        alpha = cwd / "alpha"
        alpha.mkdir()
        (alpha / "pipeline.py").write_text(
            _make_noop_pipeline_code("alpha", "stage_a", "alpha_out.txt", "alpha_out.txt")
        )

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0, f"status --all failed: {result.output}"
        assert "stage_a" in result.output
