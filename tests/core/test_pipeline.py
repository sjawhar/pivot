"""Tests for pivot.Pipeline programmatic API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pydantic
import pytest

from pivot import Pipeline, outputs, registry
from pivot.pipeline import PipelineError

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None]:
    """Clear registry before and after each test."""
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY.clear()


@pytest.fixture
def mock_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> pathlib.Path:
    """Set up a mock project root."""
    mocker.patch("pivot.project._project_root_cache", tmp_path)
    return tmp_path


# =============================================================================
# Basic Registration Tests
# =============================================================================


def test_pipeline_add_stage_registers_function(mock_project_root: pathlib.Path) -> None:
    """add_stage registers a function to the global registry."""

    def my_stage() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(my_stage, deps=["input.txt"], outs=["output.txt"])

    assert "my_stage" in registry.REGISTRY.list_stages()


def test_pipeline_add_stage_with_custom_name(mock_project_root: pathlib.Path) -> None:
    """add_stage with name parameter uses custom name."""

    def process() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(process, name="custom_name", deps=["in.txt"], outs=["out.txt"])

    assert "custom_name" in registry.REGISTRY.list_stages()
    assert "process" not in registry.REGISTRY.list_stages()


def test_pipeline_add_stage_with_deps_and_outs(mock_project_root: pathlib.Path) -> None:
    """add_stage registers deps and outs correctly."""

    def process() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(process, deps=["data/input.csv"], outs=["data/output.csv"])

    info = registry.REGISTRY.get("process")
    assert any("data/input.csv" in d for d in info["deps"])
    assert any("data/output.csv" in o for o in info["outs_paths"])


def test_pipeline_stages_property_returns_registration_order(
    mock_project_root: pathlib.Path,
) -> None:
    """stages property returns stage names in registration order."""

    def stage_a() -> None:
        pass

    def stage_b() -> None:
        pass

    def stage_c() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(stage_a, outs=["a.txt"])
    pipeline.add_stage(stage_b, outs=["b.txt"])
    pipeline.add_stage(stage_c, outs=["c.txt"])

    assert pipeline.stages == ["stage_a", "stage_b", "stage_c"]


def test_pipeline_add_stage_duplicate_name_raises(mock_project_root: pathlib.Path) -> None:
    """add_stage with duplicate name raises PipelineError."""

    def process() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(process, outs=["out1.txt"])

    with pytest.raises(PipelineError, match="already added"):
        pipeline.add_stage(process, outs=["out2.txt"])


# =============================================================================
# Params Tests
# =============================================================================


class TrainParams(pydantic.BaseModel):
    learning_rate: float = 0.01
    epochs: int = 100


def stage_with_params(params: TrainParams) -> None:
    pass


def test_pipeline_add_stage_with_params_instance(mock_project_root: pathlib.Path) -> None:
    """add_stage with BaseModel instance uses it directly."""
    params = TrainParams(learning_rate=0.1, epochs=200)

    pipeline = Pipeline()
    pipeline.add_stage(stage_with_params, outs=["model.pkl"], params=params)

    info = registry.REGISTRY.get("stage_with_params")
    assert info["params"] is params


# =============================================================================
# Metrics and Plots Tests
# =============================================================================


def test_pipeline_add_stage_with_metrics(mock_project_root: pathlib.Path) -> None:
    """add_stage with metrics creates Metric outputs."""

    def train() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(train, outs=["model.pkl"], metrics=["metrics/train.json"])

    info = registry.REGISTRY.get("train")
    metric_outs = [o for o in info["outs"] if isinstance(o, outputs.Metric)]
    assert len(metric_outs) == 1
    assert "metrics/train.json" in metric_outs[0].path


def test_pipeline_add_stage_with_plots(mock_project_root: pathlib.Path) -> None:
    """add_stage with plots creates Plot outputs."""

    def analyze() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(analyze, outs=["report.json"], plots=["plots/curve.json"])

    info = registry.REGISTRY.get("analyze")
    plot_outs = [o for o in info["outs"] if isinstance(o, outputs.Plot)]
    assert len(plot_outs) == 1
    assert "plots/curve.json" in plot_outs[0].path


# =============================================================================
# Mutex and CWD Tests
# =============================================================================


def test_pipeline_add_stage_with_mutex(mock_project_root: pathlib.Path) -> None:
    """add_stage with mutex registers mutex groups."""

    def gpu_stage() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(gpu_stage, outs=["out.txt"], mutex=["gpu"])

    info = registry.REGISTRY.get("gpu_stage")
    assert "gpu" in info["mutex"]


def test_pipeline_add_stage_with_cwd(mock_project_root: pathlib.Path) -> None:
    """add_stage with cwd sets working directory."""
    (mock_project_root / "subdir").mkdir()

    def subdir_stage() -> None:
        pass

    pipeline = Pipeline()
    pipeline.add_stage(subdir_stage, outs=["out.txt"], cwd="subdir")

    info = registry.REGISTRY.get("subdir_stage")
    assert info["cwd"] is not None
    assert "subdir" in str(info["cwd"])
