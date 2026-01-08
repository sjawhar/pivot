from __future__ import annotations

import json
import pathlib

import click.testing
import pytest
import yaml

from pivot import cli


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Metrics Show Tests
# =============================================================================


def test_metrics_show_help(runner: click.testing.CliRunner) -> None:
    """Metrics show command should show help."""
    result = runner.invoke(cli.cli, ["metrics", "show", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--precision" in result.output


def test_metrics_show_file(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show displays file contents."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95, "loss": 0.05}))

    result = runner.invoke(cli.cli, ["metrics", "show", str(metric_file)])

    assert result.exit_code == 0
    assert "accuracy" in result.output
    assert "loss" in result.output


def test_metrics_show_json_format(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show --json outputs valid JSON."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95}))

    result = runner.invoke(cli.cli, ["metrics", "show", "--json", str(metric_file)])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert str(metric_file) in parsed


def test_metrics_show_markdown_format(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Metrics show --md outputs markdown table."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.95}))

    result = runner.invoke(cli.cli, ["metrics", "show", "--md", str(metric_file)])

    assert result.exit_code == 0
    assert "|" in result.output
    assert "---" in result.output


def test_metrics_show_precision(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show respects --precision flag."""
    metric_file = tmp_path / "metrics.json"
    metric_file.write_text(json.dumps({"accuracy": 0.123456789}))

    result = runner.invoke(cli.cli, ["metrics", "show", "--precision", "2", str(metric_file)])

    assert result.exit_code == 0
    assert "0.12" in result.output
    assert "0.123456789" not in result.output


def test_metrics_show_yaml_file(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show handles YAML files."""
    metric_file = tmp_path / "metrics.yaml"
    metric_file.write_text(yaml.dump({"f1_score": 0.88}))

    result = runner.invoke(cli.cli, ["metrics", "show", str(metric_file)])

    assert result.exit_code == 0
    assert "f1_score" in result.output


def test_metrics_show_csv_file(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show handles CSV files."""
    metric_file = tmp_path / "metrics.csv"
    metric_file.write_text("accuracy,0.95\nloss,0.05\n")

    result = runner.invoke(cli.cli, ["metrics", "show", str(metric_file)])

    assert result.exit_code == 0
    assert "accuracy" in result.output
    assert "loss" in result.output


def test_metrics_show_directory(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show handles directory target."""
    (tmp_path / "a.json").write_text(json.dumps({"a": 1}))
    (tmp_path / "b.json").write_text(json.dumps({"b": 2}))

    result = runner.invoke(cli.cli, ["metrics", "show", str(tmp_path)])

    assert result.exit_code == 0
    assert "a.json" in result.output
    assert "b.json" in result.output


def test_metrics_show_recursive(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics show -R searches recursively."""
    (tmp_path / "a.json").write_text(json.dumps({"a": 1}))
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "b.json").write_text(json.dumps({"b": 2}))

    result = runner.invoke(cli.cli, ["metrics", "show", "-R", str(tmp_path)])

    assert result.exit_code == 0
    assert "a.json" in result.output
    assert "b.json" in result.output


def test_metrics_show_file_not_found(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Metrics show with missing file shows error."""
    result = runner.invoke(cli.cli, ["metrics", "show", str(tmp_path / "nonexistent.json")])

    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "Error" in result.output


def test_metrics_show_no_targets_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Metrics show with no targets and no stages shows no metrics."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        result = runner.invoke(cli.cli, ["metrics", "show"])

    assert result.exit_code == 0
    assert "No metrics found" in result.output


# =============================================================================
# Metrics Diff Tests
# =============================================================================


def test_metrics_diff_help(runner: click.testing.CliRunner) -> None:
    """Metrics diff command should show help."""
    result = runner.invoke(cli.cli, ["metrics", "diff", "--help"])
    assert result.exit_code == 0
    assert "TARGETS" in result.output
    assert "--json" in result.output
    assert "--no-path" in result.output
    assert "-R" in result.output


def test_metrics_diff_no_metrics(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Metrics diff with no registered stages should report empty."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        result = runner.invoke(cli.cli, ["metrics", "diff"])
    assert result.exit_code == 0
    assert "No metrics found" in result.output


# =============================================================================
# Command Group Tests
# =============================================================================


def test_metrics_group_help(runner: click.testing.CliRunner) -> None:
    """Metrics group shows subcommands."""
    result = runner.invoke(cli.cli, ["metrics", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "diff" in result.output


def test_metrics_in_main_help(runner: click.testing.CliRunner) -> None:
    """Metrics command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "metrics" in result.output
