from __future__ import annotations

import contextlib
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import yaml
from tests.fixtures.export import pipeline

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner

    from pivot.pipeline.pipeline import Pipeline


# =============================================================================
# Module-level TypedDicts and Stage Functions for annotation-based registration
# =============================================================================


class _CleanCsvOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("clean.csv", loaders.PathOnly())]


class _OutTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("out.txt", loaders.PathOnly())]


class _ATxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


class _BTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


class _ModelPklOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


class _OutputCsvOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


def _helper_preprocess_with_data_dep(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _CleanCsvOutputs:
    """Preprocess stage with data dependency."""
    _ = data
    pathlib.Path("clean.csv").write_text("clean")
    return _CleanCsvOutputs(output=pathlib.Path("clean.csv"))


def _helper_preprocess_no_deps() -> _OutTxtOutputs:
    """Preprocess stage without dependencies."""
    pathlib.Path("out.txt").write_text("out")
    return _OutTxtOutputs(output=pathlib.Path("out.txt"))


def _helper_preprocess_a_txt() -> _ATxtOutputs:
    """Stage producing a.txt."""
    pathlib.Path("a.txt").write_text("a")
    return _ATxtOutputs(output=pathlib.Path("a.txt"))


def _helper_evaluate_b_txt() -> _BTxtOutputs:
    """Stage producing b.txt."""
    pathlib.Path("b.txt").write_text("b")
    return _BTxtOutputs(output=pathlib.Path("b.txt"))


def _helper_train_with_params(
    params: pipeline.TrainParams,
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _ModelPklOutputs:
    """Train stage with params and data dependency."""
    _ = data, params
    pathlib.Path("model.pkl").write_text("model")
    return _ModelPklOutputs(output=pathlib.Path("model.pkl"))


def _helper_preprocess_with_input_dep(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.csv", loaders.PathOnly())],
) -> _OutputCsvOutputs:
    """Preprocess stage with input.csv dependency."""
    _ = input_file
    pathlib.Path("output.csv").write_text("output")
    return _OutputCsvOutputs(output=pathlib.Path("output.csv"))


# =============================================================================
# Export Command Tests
# =============================================================================


def test_export_help_shows_options(runner: CliRunner) -> None:
    """Export command should show help with options."""
    result = runner.invoke(cli.cli, ["export", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output or "-o" in result.output


def test_export_default_output_creates_dvc_yaml(
    mock_discovery: Pipeline,
    runner: CliRunner,
    set_project_root: Path,
) -> None:
    """Export without args creates dvc.yaml in current directory."""
    _ = mock_discovery
    (set_project_root / ".git").mkdir()

    register_test_stage(_helper_preprocess_with_data_dep, name="preprocess")

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (set_project_root / "dvc.yaml").exists()
        assert "Exported 1 stages" in result.output


def test_export_custom_output_path(
    mock_discovery: Pipeline,
    runner: CliRunner,
    set_project_root: Path,
) -> None:
    """Export with --output writes to specified path."""
    _ = mock_discovery
    (set_project_root / ".git").mkdir()

    register_test_stage(_helper_preprocess_no_deps, name="preprocess")

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export", "--output", "custom.yaml"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert (set_project_root / "custom.yaml").exists()


def test_export_specific_stages_only(
    mock_discovery: Pipeline,
    runner: CliRunner,
    set_project_root: Path,
) -> None:
    """Export with stage names exports only those stages."""
    _ = mock_discovery
    (set_project_root / ".git").mkdir()

    register_test_stage(_helper_preprocess_a_txt, name="preprocess")
    register_test_stage(_helper_evaluate_b_txt, name="evaluate")

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export", "preprocess"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Exported 1 stages" in result.output

        with open(set_project_root / "dvc.yaml") as f:
            dvc_yaml = yaml.safe_load(f)

        assert "preprocess" in dvc_yaml["stages"]
        assert "evaluate" not in dvc_yaml["stages"]


def test_export_generates_params_yaml(
    mock_discovery: Pipeline,
    runner: CliRunner,
    set_project_root: Path,
) -> None:
    """Export generates params.yaml with Pydantic model defaults."""
    _ = mock_discovery
    (set_project_root / ".git").mkdir()

    register_test_stage(
        _helper_train_with_params,
        name="train",
        params=pipeline.TrainParams(),
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
    mock_discovery: Pipeline,
    runner: CliRunner,
    set_project_root: Path,
) -> None:
    """Export with unknown stage name shows error."""
    _ = mock_discovery
    (set_project_root / ".git").mkdir()

    register_test_stage(_helper_preprocess_a_txt, name="preprocess")

    with contextlib.chdir(set_project_root):
        result = runner.invoke(cli.cli, ["export", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output


def test_export_no_stages_error(
    mock_discovery: Pipeline,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Export with no registered stages shows error."""
    _ = mock_discovery
    (tmp_path / ".git").mkdir()

    with contextlib.chdir(tmp_path):
        result = runner.invoke(cli.cli, ["export"])

        assert result.exit_code != 0
        assert "No stages" in result.output


def test_export_dvc_yaml_structure(
    mock_discovery: Pipeline,
    runner: CliRunner,
    set_project_root: Path,
) -> None:
    """Exported dvc.yaml has correct structure with cmd, deps, outs."""
    _ = mock_discovery
    (set_project_root / ".git").mkdir()

    register_test_stage(_helper_preprocess_with_input_dep, name="preprocess")

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
