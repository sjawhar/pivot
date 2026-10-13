"""Integration tests for ``--all`` across compositional pipeline.py files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import cli
from pivot.storage import lock
from tests.conftest import isolated_pivot_dir

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner


def _helper_pipeline_code(name: str, stage_name: str) -> str:
    """Return a complete compositional pipeline.py with one value-returning stage."""
    return f'''\
import pathlib

from pivot.compose import Pipeline, stage


@stage
def {stage_name}() -> str:
    return "{stage_name} output"


with Pipeline("{name}", root=pathlib.Path(__file__).parent) as pipeline:
    {stage_name}()
'''


def _setup_multi_pipeline_project(root: pathlib.Path) -> None:
    """Create alpha and beta compositional pipelines with independent state dirs."""
    alpha = root / "alpha"
    alpha.mkdir()
    (alpha / "pipeline.py").write_text(_helper_pipeline_code("alpha", "stage_a"))

    beta = root / "beta"
    beta.mkdir()
    (beta / ".pivot").mkdir()
    (beta / "pipeline.py").write_text(_helper_pipeline_code("beta", "stage_b"))


def test_repro_all_discovers_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all --dry-run discovers stages from all sub-pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--dry-run"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output, "alpha stage not found"
        assert "beta/stage_b" in result.output, "beta stage not found"


def test_repro_all_executes_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all runs stages from all discovered pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output
        assert "beta/stage_b" in result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert (cwd / "data" / "beta" / "stage_b.txt").read_text() == "stage_b output"


def test_repro_all_writes_locks_to_correct_state_dir(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """Lock files are written to each source pipeline's own .pivot/stages/ directory."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all"])
        assert result.exit_code == 0, f"Expected success, got: {result.output}"

        beta_lock = lock.StageLock("beta/stage_b", lock.get_stages_dir(cwd / "beta" / ".pivot"))
        assert beta_lock.read() is not None, "Lock file not found in beta's state_dir"


def test_status_all_shows_all_pipelines(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """status --all shows stages from all pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output
        assert "beta/stage_b" in result.output


def test_verify_all_with_mixed_state_dirs(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """verify --all reads lock files from each stage's state directory."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        run_result = runner.invoke(cli.cli, ["repro", "--all"])
        assert run_result.exit_code == 0, f"repro failed: {run_result.output}"

        result = runner.invoke(cli.cli, ["verify", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output
        assert "beta/stage_b" in result.output


def test_commit_all_after_no_commit_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """commit --all promotes pending locks into each source state directory."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        run_result = runner.invoke(cli.cli, ["repro", "--all", "--no-commit"])
        assert run_result.exit_code == 0, f"repro failed: {run_result.output}"

        commit_result = runner.invoke(cli.cli, ["commit", "--all"])
        assert commit_result.exit_code == 0, f"commit failed: {commit_result.output}"

        beta_lock = lock.StageLock("beta/stage_b", lock.get_stages_dir(cwd / "beta" / ".pivot"))
        assert beta_lock.read() is not None, "Production lock not in beta's state_dir after commit"


def test_all_flag_with_single_pipeline(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--all works gracefully when the project has only one pipeline."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        alpha = cwd / "alpha"
        alpha.mkdir()
        (alpha / "pipeline.py").write_text(_helper_pipeline_code("alpha", "stage_a"))

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0, f"status --all failed: {result.output}"
        assert "alpha/stage_a" in result.output
