from __future__ import annotations

import logging
import runpy
from typing import TYPE_CHECKING

from pivot import project, registry
from pivot.pipeline import yaml as pipeline_config

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

PIVOT_YAML_NAMES = ("pivot.yaml", "pivot.yml")
PIPELINE_PY_NAME = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


def discover_and_register(project_root: Path | None = None) -> str | None:
    """Discover and register pipeline from pivot.yaml or pipeline.py.

    Looks in project root for:
    1. pivot.yaml (or pivot.yml) - uses pipeline_config to register
    2. pipeline.py - imports module which should register stages

    Args:
        project_root: Override project root (default: auto-detect)

    Returns:
        Path to the discovered file, or None if nothing found

    Raises:
        DiscoveryError: If discovery or registration fails, or if both config types exist
    """
    root = project_root or project.get_project_root()

    # Check which files exist upfront
    yaml_path = None
    for yaml_name in PIVOT_YAML_NAMES:
        candidate = root / yaml_name
        if candidate.exists():
            yaml_path = candidate
            break

    pipeline_path = root / PIPELINE_PY_NAME
    pipeline_exists = pipeline_path.exists()

    # Error if both exist
    if yaml_path and pipeline_exists:
        raise DiscoveryError(
            f"Found both {yaml_path.name} and {PIPELINE_PY_NAME} in {root}. Remove one to resolve ambiguity."
        )

    # Register from yaml if found
    if yaml_path:
        logger.info(f"Discovered {yaml_path}")
        try:
            pipeline_config.register_from_pipeline_file(yaml_path)
            return str(yaml_path)
        except pipeline_config.PipelineConfigError as e:
            raise DiscoveryError(f"Failed to load {yaml_path}: {e}") from e

    # Try pipeline.py
    if pipeline_exists:
        logger.info(f"Discovered {pipeline_path}")
        try:
            _import_pipeline_module(pipeline_path)
            return str(pipeline_path)
        except SystemExit as e:
            raise DiscoveryError(f"Pipeline {pipeline_path} called sys.exit({e.code})") from e
        except Exception as e:
            raise DiscoveryError(f"Failed to load {pipeline_path}: {e}") from e

    return None


def has_registered_stages() -> bool:
    """Check if any stages are registered."""
    return len(registry.REGISTRY.list_stages()) > 0


def _import_pipeline_module(path: Path) -> None:
    """Execute a pipeline.py file, registering its stages."""
    runpy.run_path(str(path), run_name="_pivot_pipeline")
