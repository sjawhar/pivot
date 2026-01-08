from __future__ import annotations

import logging
import runpy
from typing import TYPE_CHECKING

from pivot import pipeline_config, project, registry

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

PIVOT_YAML_NAMES = ("pivot.yaml", "pivot.yml")
PIPELINE_PY_NAME = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


def discover_and_register(project_root: pathlib.Path | None = None) -> str | None:
    """Discover and register pipeline from pivot.yaml or pipeline.py.

    Looks in project root for:
    1. pivot.yaml (or pivot.yml) - uses pipeline_config to register
    2. pipeline.py - imports module which should register stages

    Args:
        project_root: Override project root (default: auto-detect)

    Returns:
        Path to the discovered file, or None if nothing found

    Raises:
        DiscoveryError: If discovery or registration fails
    """
    root = project_root or project.get_project_root()

    # Try pivot.yaml first
    for yaml_name in PIVOT_YAML_NAMES:
        yaml_path = root / yaml_name
        if yaml_path.exists():
            logger.info(f"Discovered {yaml_path}")
            try:
                pipeline_config.register_from_pipeline_file(yaml_path)
                return str(yaml_path)
            except pipeline_config.PipelineConfigError as e:
                raise DiscoveryError(f"Failed to load {yaml_path}: {e}") from e

    # Try pipeline.py
    pipeline_path = root / PIPELINE_PY_NAME
    if pipeline_path.exists():
        logger.info(f"Discovered {pipeline_path}")
        try:
            _import_pipeline_module(pipeline_path)
            return str(pipeline_path)
        except Exception as e:
            raise DiscoveryError(f"Failed to load {pipeline_path}: {e}") from e

    return None


def has_registered_stages() -> bool:
    """Check if any stages are registered."""
    return len(registry.REGISTRY.list_stages()) > 0


def _import_pipeline_module(path: pathlib.Path) -> None:
    """Execute a pipeline.py file, registering its stages."""
    runpy.run_path(str(path), run_name="_pivot_pipeline")
