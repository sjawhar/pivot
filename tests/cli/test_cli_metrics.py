from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from pivot import cli, git

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


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
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        metric_file = pathlib.Path("metrics.json")
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


def test_metrics_diff_explicit_file_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Issue #62: metrics diff TARGET should work with explicit file when no stages registered."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Create a metrics file
        metrics_file = pathlib.Path("metrics.json")
        metrics_file.write_text(json.dumps({"accuracy": 0.95}))

        # Should work even with no stages registered
        result = runner.invoke(cli.cli, ["metrics", "diff", str(metrics_file)])

        # Should not fail with "stage not found" error
        assert result.exit_code == 0
        # Should show diff output (no prior commit, so shows as added)
        assert (
            "accuracy" in result.output
            or "No changes" in result.output
            or "No metrics found" in result.output
        )


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


# =============================================================================
# Metrics Diff Integration Tests (using explicit file targets)
# =============================================================================


@pytest.mark.parametrize(
    ("head_metrics", "workspace_metrics", "cli_args", "test_id"),
    [
        pytest.param(
            {"accuracy": 0.85, "loss": 0.15},
            {"accuracy": 0.92, "loss": 0.08},
            [],
            "shows_changes",
            id="shows-changes",
        ),
        pytest.param(
            {"accuracy": 0.80},
            {"accuracy": 0.95},
            ["--json"],
            "json_output",
            id="json-output",
        ),
        pytest.param(
            {"accuracy": 0.90},
            {"accuracy": 0.90},
            [],
            "no_changes",
            id="no-changes",
        ),
        pytest.param(
            {"f1": 0.75},
            {"f1": 0.88},
            ["--md"],
            "markdown_format",
            id="markdown-format",
        ),
    ],
)
def test_metrics_diff_integration(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    head_metrics: dict[str, float],
    workspace_metrics: dict[str, float],
    cli_args: list[str],
    test_id: str,
) -> None:
    """Integration test: metrics diff with various formats and scenarios using file targets."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        # Mock git.read_files_from_head to return HEAD version of metrics file
        mocker.patch.object(
            git,
            "read_files_from_head",
            return_value={
                "metrics.json": json.dumps(head_metrics).encode(),
            },
        )

        # Write workspace version of metrics file
        pathlib.Path("metrics.json").write_text(json.dumps(workspace_metrics))

        # Use explicit file target instead of stage name
        result = runner.invoke(cli.cli, ["metrics", "diff", *cli_args, "metrics.json"])

        assert result.exit_code == 0

        match test_id:
            case "shows_changes":
                assert "0.85" in result.output, "Should show old HEAD value"
                assert "0.92" in result.output, "Should show new workspace value"
                assert "modified" in result.output.lower()
            case "json_output":
                parsed = cast("list[dict[str, object]]", json.loads(result.output))
                assert len(parsed) == 1
                assert parsed[0]["change_type"] == "modified"
                assert parsed[0]["old"] == 0.80, "Should include old value"
                assert parsed[0]["new"] == 0.95, "Should include new value"
            case "no_changes":
                assert "No metric changes" in result.output
            case "markdown_format":
                assert "f1" in result.output, "Should show metric name"
                assert "|" in result.output
                assert "---" in result.output
            case _:
                pytest.fail(f"Unknown test_id: {test_id}")
