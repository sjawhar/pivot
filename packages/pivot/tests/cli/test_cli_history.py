"""Tests for pivot history and show CLI commands.

Verifies both commands read from the same StateDB data source and agree
on run records. Restored after deletion during pipeline-unification.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pivot import cli, project, run_history
from pivot.config import io as config_io
from pivot.storage import state
from pivot.types import StageStatus

if TYPE_CHECKING:
    import pathlib

    import click.testing


@pytest.fixture
def project_with_runs(
    tmp_path: pathlib.Path,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """Create a project directory with run history in StateDB."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".pivot").mkdir()

    monkeypatch.chdir(project_dir)
    project._project_root_cache = None
    config_io.clear_config_cache()

    # Write test runs to the same StateDB that both commands read from
    state_db_path = project_dir / ".pivot" / "state.db"
    with state.StateDB(state_db_path) as db:
        for i in range(3):
            manifest = run_history.RunManifest(
                run_id=f"2025011{i}_143000_abc1234{i}",
                started_at=f"2025-01-1{i}T14:30:00+00:00",
                ended_at=f"2025-01-1{i}T14:35:00+00:00",
                targeted_stages=["train", "eval"],
                execution_order=["train", "eval"],
                stages={
                    "train": run_history.StageRunRecord(
                        input_hash="hash_train",
                        status=StageStatus.RAN if i == 2 else StageStatus.CACHED,
                        reason="Code changed" if i == 2 else "unchanged",
                        duration_ms=5000 if i == 2 else 100,
                    ),
                    "eval": run_history.StageRunRecord(
                        input_hash="hash_eval",
                        status=StageStatus.RAN,
                        reason="Input changed",
                        duration_ms=3000,
                    ),
                },
            )
            db.write_run(manifest)

    return project_dir


# =============================================================================
# pivot history tests
# =============================================================================


def test_history_lists_runs(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """History command should list recent runs."""
    result = runner.invoke(cli.cli, ["history"])

    assert result.exit_code == 0
    assert "Run ID" in result.output
    # Most recent first
    assert "20250112_143000_abc12342" in result.output


def test_history_shows_status_summary(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """History should show ran/skipped/failed counts."""
    result = runner.invoke(cli.cli, ["history"])

    assert result.exit_code == 0
    # All 3 runs should appear
    assert "20250110_143000_abc12340" in result.output
    assert "20250111_143000_abc12341" in result.output
    assert "20250112_143000_abc12342" in result.output


def test_history_respects_limit(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """History --limit should limit number of runs shown."""
    result = runner.invoke(cli.cli, ["history", "--limit", "1"])

    assert result.exit_code == 0
    # Should only show most recent
    assert "20250112_143000_abc12342" in result.output
    assert "20250111_143000_abc12341" not in result.output


def test_history_json_output(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """History --json should output JSON."""
    result = runner.invoke(cli.cli, ["history", "--json"])

    assert result.exit_code == 0
    data: list[dict[str, object]] = json.loads(result.output)
    assert len(data) == 3
    assert data[0]["run_id"] == "20250112_143000_abc12342"


def test_history_empty(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """History should show message when no runs exist."""
    project_dir = tmp_path / "empty_project"
    project_dir.mkdir()
    (project_dir / ".pivot").mkdir()
    monkeypatch.chdir(project_dir)
    project._project_root_cache = None
    config_io.clear_config_cache()

    result = runner.invoke(cli.cli, ["history"])

    assert result.exit_code == 0
    assert "No runs recorded" in result.output


# =============================================================================
# pivot show tests
# =============================================================================


def test_show_latest_run(runner: click.testing.CliRunner, project_with_runs: pathlib.Path) -> None:
    """Show without argument should show latest run."""
    result = runner.invoke(cli.cli, ["show"])

    assert result.exit_code == 0
    assert "20250112_143000_abc12342" in result.output
    assert "Stages:" in result.output
    assert "train" in result.output
    assert "eval" in result.output


def test_show_specific_run(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """Show with run_id should show that specific run."""
    result = runner.invoke(cli.cli, ["show", "20250110_143000_abc12340"])

    assert result.exit_code == 0
    assert "20250110_143000_abc12340" in result.output


def test_show_nonexistent_run(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """Show with nonexistent run_id should show error."""
    result = runner.invoke(cli.cli, ["show", "nonexistent_run_id"])

    assert result.exit_code != 0
    assert "Run not found" in result.output


def test_show_json_output(runner: click.testing.CliRunner, project_with_runs: pathlib.Path) -> None:
    """Show --json should output JSON."""
    result = runner.invoke(cli.cli, ["show", "--json"])

    assert result.exit_code == 0
    data: dict[str, object] = json.loads(result.output)
    assert data["run_id"] == "20250112_143000_abc12342"
    assert "stages" in data
    stages = data["stages"]
    assert isinstance(stages, dict)
    assert "train" in stages


def test_show_empty_project(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Show on empty project should show error."""
    project_dir = tmp_path / "empty_project"
    project_dir.mkdir()
    (project_dir / ".pivot").mkdir()
    monkeypatch.chdir(project_dir)
    project._project_root_cache = None
    config_io.clear_config_cache()

    result = runner.invoke(cli.cli, ["show"])

    assert result.exit_code != 0
    assert "No runs recorded" in result.output


# =============================================================================
# Cross-command consistency: history and show agree
# =============================================================================


def test_show_finds_all_runs_listed_by_history(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """Every run_id listed by history should be found by show."""
    history_result = runner.invoke(cli.cli, ["history", "--json"])
    assert history_result.exit_code == 0
    runs: list[dict[str, object]] = json.loads(history_result.output)

    for run in runs:
        run_id = run["run_id"]
        assert isinstance(run_id, str)
        show_result = runner.invoke(cli.cli, ["show", run_id])
        assert show_result.exit_code == 0, f"show failed for run_id {run_id}: {show_result.output}"
        assert run_id in show_result.output


def test_show_latest_matches_history_first(
    runner: click.testing.CliRunner, project_with_runs: pathlib.Path
) -> None:
    """Show (no args) should return the same run as the first entry in history."""
    history_result = runner.invoke(cli.cli, ["history", "--json"])
    assert history_result.exit_code == 0
    runs: list[dict[str, object]] = json.loads(history_result.output)
    latest_run_id = runs[0]["run_id"]

    show_result = runner.invoke(cli.cli, ["show", "--json"])
    assert show_result.exit_code == 0
    show_data: dict[str, object] = json.loads(show_result.output)
    assert show_data["run_id"] == latest_run_id, (
        "show (no args) should return the same run as the first entry in history"
    )
