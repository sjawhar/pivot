import contextlib
import pathlib

import click.testing
import pytest
import yaml
from tests.fixtures.export import pipeline

from pivot import cli, outputs, registry


@pytest.fixture
def runner() -> click.testing.CliRunner:
    """Create a CLI runner for testing."""
    return click.testing.CliRunner()


# =============================================================================
# Export Command Tests
# =============================================================================


def test_export_help_shows_options(runner: click.testing.CliRunner) -> None:
    """Export command should show help with options."""
    result = runner.invoke(cli.cli, ["export", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output or "-o" in result.output


def test_export_default_output_creates_dvc_yaml(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Export without args creates dvc.yaml in current directory."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.preprocess,
        name="preprocess",
        deps=[str(set_project_root / "data.csv")],
        outs=[str(set_project_root / "clean.csv")],
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (set_project_root / "dvc.yaml").exists()
        assert "Exported 1 stages" in result.output


def test_export_custom_output_path(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Export with --output writes to specified path."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.preprocess,
        name="preprocess",
        deps=[],
        outs=[str(set_project_root / "out.txt")],
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export", "--output", "custom.yaml"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (set_project_root / "custom.yaml").exists()


def test_export_specific_stages_only(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Export with stage names exports only those stages."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.preprocess, name="preprocess", deps=[], outs=[str(set_project_root / "a.txt")]
    )
    registry.REGISTRY.register(
        pipeline.evaluate, name="evaluate", deps=[], outs=[str(set_project_root / "b.txt")]
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export", "preprocess"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Exported 1 stages" in result.output

        with open(set_project_root / "dvc.yaml") as f:
            dvc_yaml = yaml.safe_load(f)

        assert "preprocess" in dvc_yaml["stages"]
        assert "evaluate" not in dvc_yaml["stages"]


def test_export_generates_params_yaml(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Export generates params.yaml with Pydantic model defaults."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.train,
        name="train",
        deps=[str(set_project_root / "data.csv")],
        outs=[str(set_project_root / "model.pkl")],
        params=pipeline.TrainParams,
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (set_project_root / "params.yaml").exists()

        with open(set_project_root / "params.yaml") as f:
            params = yaml.safe_load(f)

        assert params["train"]["learning_rate"] == 0.01
        assert params["train"]["epochs"] == 100


def test_export_unknown_stage_error(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Export with unknown stage name shows error."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.preprocess, name="preprocess", deps=[], outs=[str(set_project_root / "a.txt")]
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output


def test_export_no_stages_error(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """Export with no registered stages shows error."""
    (tmp_path / ".git").mkdir()

    with contextlib.chdir(tmp_path):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code != 0
        assert "No stages" in result.output


def test_export_dvc_yaml_structure(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Exported dvc.yaml has correct structure with cmd, deps, outs."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.preprocess,
        name="preprocess",
        deps=[str(set_project_root / "input.csv")],
        outs=[outputs.Out(str(set_project_root / "output.csv"))],
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code == 0, f"Failed: {result.output}"

        with open(set_project_root / "dvc.yaml") as f:
            dvc_yaml = yaml.safe_load(f)

        stage = dvc_yaml["stages"]["preprocess"]
        assert "cmd" in stage
        assert "python -c" in stage["cmd"]
        assert "preprocess" in stage["cmd"]
        assert stage["deps"] == ["input.csv"]
        assert stage["outs"] == ["output.csv"]


def test_export_with_metrics_and_plots(
    runner: click.testing.CliRunner,
    set_project_root: pathlib.Path,
) -> None:
    """Export correctly separates outs, metrics, and plots."""
    (set_project_root / ".git").mkdir()

    registry.REGISTRY.register(
        pipeline.train,
        name="train",
        deps=[],
        outs=[
            outputs.Out(str(set_project_root / "model.pkl")),
            outputs.Metric(str(set_project_root / "metrics.json")),
            outputs.Plot(str(set_project_root / "loss.csv"), x="epoch", y="loss"),
        ],
        params=pipeline.TrainParams,
    )

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code == 0, f"Failed: {result.output}"

        with open(set_project_root / "dvc.yaml") as f:
            dvc_yaml = yaml.safe_load(f)

        stage = dvc_yaml["stages"]["train"]
        assert stage["outs"] == ["model.pkl"]
        assert stage["metrics"] == ["metrics.json"]
        assert stage["plots"] == [{"loss.csv": {"x": "epoch", "y": "loss"}}]
