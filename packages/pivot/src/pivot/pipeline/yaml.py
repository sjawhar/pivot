# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.pipeline.pipeline import Pipeline


class PipelineConfigError(Exception):
    """Error loading or processing pivot.yaml configuration."""


def load_pipeline_from_yaml(_pipeline_file: Path) -> Pipeline:
    """Reject pivot.yaml pipelines (annotation parsing removed)."""
    raise PipelineConfigError(
        "pivot.yaml pipelines are no longer supported. "
        + "Use the compose API to build pipelines instead."
    )
