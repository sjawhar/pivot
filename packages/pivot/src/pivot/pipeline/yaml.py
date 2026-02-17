from __future__ import annotations

from typing import TYPE_CHECKING, Never

if TYPE_CHECKING:
    import pathlib


class PipelineConfigError(Exception):
    """Error loading or processing pivot.yaml configuration."""


def load_pipeline_from_yaml(pipeline_file: pathlib.Path) -> Never:
    """Reject pivot.yaml pipelines (annotation parsing removed)."""
    raise PipelineConfigError(
        "pivot.yaml pipelines are no longer supported. "
        + f"Use the compose API to build pipelines instead: {pipeline_file}"
    )
