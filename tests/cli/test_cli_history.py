"""Tests for pivot history and show CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pivot import cli, run_history
from pivot.storage import state
from pivot.types import StageStatus

if TYPE_CHECKING:
    import pathlib

    import click.testing


@pytest.fixture
def project_with_runs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Create a project directory with some run history."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".pivot").mkdir()

    monkeypatch.chdir(project_dir)

    # Write some test runs
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
                        status=StageStatus.RAN if i == 2 else StageStatus.SKIPPED,
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
    # Most recent run has 1 ran, 1 skipped (train ran, eval ran in run 2)
    # Actually train=ran (i==2), eval=ran → 2 ran, 0 skipped
    assert "ran" in result.output.lower()


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

    result = runner.invoke(cli.cli, ["show"])

    assert result.exit_code != 0
    assert "No runs recorded" in result.output
