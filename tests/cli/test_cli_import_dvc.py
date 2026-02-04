from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING

import pytest
import yaml

from pivot import cli, project

if TYPE_CHECKING:
    from click.testing import CliRunner

FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "dvc_import"


@pytest.fixture
def dvc_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Create minimal project structure for DVC import tests."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None
    return tmp_path


# =============================================================================
# Basic Command Tests
# =============================================================================


def test_import_dvc_help_shows_options(runner: CliRunner) -> None:
    """import-dvc command shows help with options."""
    result = runner.invoke(cli.cli, ["import-dvc", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output or "-i" in result.output
    assert "--output" in result.output or "-o" in result.output
    assert "--force" in result.output
    assert "--dry-run" in result.output


def test_import_dvc_creates_pivot_yaml(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc creates pivot.yaml from dvc.yaml."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")

    result = runner.invoke(cli.cli, ["import-dvc"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (dvc_project / "pivot.yaml").exists()
    assert "Converted 2 stages" in result.output


def test_import_dvc_with_explicit_input(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc with --input flag reads specified file."""
    src = FIXTURES_DIR / "simple" / "dvc.yaml"

    result = runner.invoke(cli.cli, ["import-dvc", "--input", str(src)])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (dvc_project / "pivot.yaml").exists()


def test_import_dvc_custom_output_path(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc with --output writes to specified path."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")

    result = runner.invoke(cli.cli, ["import-dvc", "--output", "custom.yaml"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert (dvc_project / "custom.yaml").exists()
    assert not (dvc_project / "pivot.yaml").exists()


def test_import_dvc_generates_migration_notes(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc creates migration notes file."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")

    result = runner.invoke(cli.cli, ["import-dvc"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    notes_path = dvc_project / ".pivot" / "migration-notes.md"
    assert notes_path.exists()
    content = notes_path.read_text()
    assert "DVC to Pivot Migration Notes" in content


# =============================================================================
# Overwrite Behavior Tests
# =============================================================================


def test_import_dvc_refuses_overwrite_without_force(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc errors on existing pivot.yaml without --force."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")
    (dvc_project / "pivot.yaml").write_text("existing content")

    result = runner.invoke(cli.cli, ["import-dvc"])

    assert result.exit_code != 0
    assert "already exists" in result.output


def test_import_dvc_force_overwrites(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc with --force overwrites existing files."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")
    (dvc_project / "pivot.yaml").write_text("old content")

    result = runner.invoke(cli.cli, ["import-dvc", "--force"])

    assert result.exit_code == 0, f"Failed: {result.output}"

    content = (dvc_project / "pivot.yaml").read_text()
    assert "old content" not in content
    assert "stages:" in content


# =============================================================================
# Dry Run Tests
# =============================================================================


def test_import_dvc_dry_run_no_files_created(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc --dry-run shows output without creating files."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")

    result = runner.invoke(cli.cli, ["import-dvc", "--dry-run"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "Dry run" in result.output
    assert not (dvc_project / "pivot.yaml").exists()
    assert not (dvc_project / ".pivot" / "migration-notes.md").exists()


# =============================================================================
# Auto-Detection Tests
# =============================================================================


def test_import_dvc_auto_detects_files(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc auto-detects dvc.yaml and params.yaml."""
    shutil.copy(FIXTURES_DIR / "with_params" / "dvc.yaml", dvc_project / "dvc.yaml")
    shutil.copy(FIXTURES_DIR / "with_params" / "params.yaml", dvc_project / "params.yaml")

    result = runner.invoke(cli.cli, ["import-dvc"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert "params.yaml" in result.output

    # Check params were inlined
    with open(dvc_project / "pivot.yaml") as f:
        pivot_yaml = yaml.safe_load(f)

    train_params = pivot_yaml["stages"]["train"].get("params", {})
    assert train_params.get("learning_rate") == 0.01


def test_import_dvc_no_dvc_yaml_error(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """import-dvc without dvc.yaml shows error."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None

    result = runner.invoke(cli.cli, ["import-dvc"])

    assert result.exit_code != 0
    assert "No dvc.yaml found" in result.output


# =============================================================================
# Output Validation Tests
# =============================================================================


def test_import_dvc_generated_yaml_has_stages(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """Generated pivot.yaml has correct stage structure."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")

    result = runner.invoke(cli.cli, ["import-dvc"])

    assert result.exit_code == 0, f"Failed: {result.output}"

    with open(dvc_project / "pivot.yaml") as f:
        pivot_yaml = yaml.safe_load(f)

    assert "stages" in pivot_yaml
    assert "preprocess" in pivot_yaml["stages"]
    assert "train" in pivot_yaml["stages"]

    # Check stage has required fields
    preprocess = pivot_yaml["stages"]["preprocess"]
    assert "python" in preprocess
    assert preprocess["python"] == "PLACEHOLDER.preprocess"
    assert "deps" in preprocess
    assert "data/raw.csv" in preprocess["deps"]


def test_import_dvc_quiet_mode(
    runner: CliRunner,
    dvc_project: pathlib.Path,
) -> None:
    """import-dvc with --quiet suppresses output."""
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", dvc_project / "dvc.yaml")

    result = runner.invoke(cli.cli, ["--quiet", "import-dvc"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert result.output.strip() == ""
    assert (dvc_project / "pivot.yaml").exists()


# =============================================================================
# Path Validation Tests
# =============================================================================


def test_import_dvc_output_path_validation(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """import-dvc validates output path stays within project."""
    (tmp_path / ".pivot").mkdir()
    shutil.copy(FIXTURES_DIR / "simple" / "dvc.yaml", tmp_path / "dvc.yaml")
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None

    result = runner.invoke(cli.cli, ["import-dvc", "--output", "/etc/pivot.yaml"])

    assert result.exit_code != 0
    # Should fail path validation
