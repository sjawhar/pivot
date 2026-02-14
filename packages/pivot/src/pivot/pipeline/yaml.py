# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import pydantic
import yaml

from pivot import yaml_config

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.pipeline.pipeline import Pipeline


class PipelineConfigError(Exception):
    """Error loading or processing pivot.yaml configuration."""


class NamedOutputOptions(TypedDict, total=False):
    """Options for named output specifications (includes path)."""

    path: str | list[str]
    cache: bool
    x: str  # For plots
    y: str  # For plots
    template: str  # For plots


NamedOutputValue = str | list[str] | NamedOutputOptions
NamedDepValue = str | list[str]
DepsSpec = dict[str, NamedDepValue]
OutputsSpec = dict[str, NamedOutputValue]


class StageConfig(pydantic.BaseModel):
    """Configuration for a single stage in pivot.yaml."""

    model_config = pydantic.ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    python: str
    deps: DepsSpec = {}
    outs: OutputsSpec = {}
    metrics: OutputsSpec = {}
    plots: OutputsSpec = {}
    params: dict[str, Any] = {}
    mutex: list[str] = []
    matrix: dict[str, list[str | int | float | bool] | dict[str, dict[str, Any] | None]] | None = (
        None
    )
    variants: str | None = None


class PipelineConfig(pydantic.BaseModel):
    """Top-level pivot.yaml configuration."""

    model_config = pydantic.ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    pipeline: str | None = None
    stages: dict[str, StageConfig]
    vars: list[str] = []


def load_pipeline_file(pipeline_file: Path) -> PipelineConfig:
    """Load and parse pivot.yaml pipeline file."""
    if not pipeline_file.exists():
        raise PipelineConfigError(f"Pipeline file not found: {pipeline_file}")

    with open(pipeline_file) as f:
        data = yaml.load(f, Loader=yaml_config.Loader)

    if data is None:
        raise PipelineConfigError(f"Pipeline file is empty: {pipeline_file}")

    try:
        return PipelineConfig.model_validate(data)
    except pydantic.ValidationError as e:
        raise PipelineConfigError(f"Invalid pipeline configuration: {e}") from e


def load_pipeline_from_yaml(pipeline_file: Path) -> Pipeline:
    """Reject pivot.yaml pipelines (annotation parsing removed)."""
    raise PipelineConfigError(
        "pivot.yaml pipelines are no longer supported. "
        + "Use the compose API to build pipelines instead."
    )
