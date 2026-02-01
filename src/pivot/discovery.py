from __future__ import annotations

import logging
import runpy
from typing import TYPE_CHECKING

from pivot import fingerprint, metrics, project
from pivot.pipeline import yaml as pipeline_config

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

PIVOT_YAML_NAMES = ("pivot.yaml", "pivot.yml")
PIPELINE_PY_NAME = "pipeline.py"


class DiscoveryError(Exception):
    """Error during pipeline discovery."""


def discover_pipeline(project_root: Path | None = None) -> Pipeline | None:
    """Discover and return Pipeline from pivot.yaml or pipeline.py.

    Looks in project root for:
    1. pivot.yaml (or pivot.yml) - creates implicit Pipeline
    2. pipeline.py - looks for `pipeline` variable (Pipeline instance)

    Args:
        project_root: Override project root (default: auto-detect)

    Returns:
        Pipeline instance, or None if nothing found

    Raises:
        DiscoveryError: If discovery fails, or if both config types exist
    """
    with metrics.timed("discovery.total"):
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

        # Load from yaml if found
        if yaml_path:
            logger.info(f"Discovered {yaml_path}")
            try:
                return pipeline_config.load_pipeline_from_yaml(yaml_path)
            except pipeline_config.PipelineConfigError as e:
                raise DiscoveryError(f"Failed to load {yaml_path}: {e}") from e

        # Try pipeline.py
        if pipeline_exists:
            logger.info(f"Discovered {pipeline_path}")
            try:
                return _load_pipeline_from_module(pipeline_path)
            except SystemExit as e:
                raise DiscoveryError(f"Pipeline {pipeline_path} called sys.exit({e.code})") from e
            except DiscoveryError:
                # Re-raise DiscoveryError without wrapping
                raise
            except Exception as e:
                raise DiscoveryError(f"Failed to load {pipeline_path}: {e}") from e
            finally:
                fingerprint.flush_ast_hash_cache()

        return None


def _load_pipeline_from_module(path: Path) -> Pipeline | None:
    """Load Pipeline instance from a pipeline.py file.

    Returns None if the file doesn't define a 'pipeline' variable.
    Raises DiscoveryError if:
    - 'pipeline' variable exists but isn't a Pipeline instance
    - A Pipeline instance exists under a different variable name (likely typo)
    """
    from pivot.pipeline.pipeline import Pipeline

    module_dict = runpy.run_path(str(path), run_name="_pivot_pipeline")

    # Look for 'pipeline' variable
    pipeline = module_dict.get("pipeline")
    if pipeline is not None:
        if not isinstance(pipeline, Pipeline):
            raise DiscoveryError(
                f"{path} defines 'pipeline' but it's not a Pipeline instance (got {type(pipeline).__name__})"
            )
        return pipeline

    # No 'pipeline' variable - check if there's a Pipeline under a different name
    # This catches cases where user creates a Pipeline but forgets to name it 'pipeline'
    for name, value in module_dict.items():
        if isinstance(value, Pipeline):
            raise DiscoveryError(
                f"{path} does not define a 'pipeline' variable. Found Pipeline instance named '{name}' - rename it to 'pipeline'."
            )

    # No Pipeline found anywhere
    return None
