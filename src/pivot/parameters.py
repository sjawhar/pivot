from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

import deepmerge
import pydantic
import yaml

from pivot import project, yaml_config

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

# Type alias for params overrides: stage_name -> param_name -> param_value
# Inner dict values are Any because config files can contain arbitrary JSON-compatible types
ParamsOverrides = dict[str, dict[str, Any]]


def load_params_yaml(path: pathlib.Path | None = None) -> ParamsOverrides:
    """Load params.yaml from project root or specified path.

    Returns dict of stage_name -> param_dict. Returns empty dict if file missing.
    """
    if path is None:
        path = project.get_project_root() / "params.yaml"

    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data: object = yaml.load(f, Loader=yaml_config.Loader)
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load params.yaml: {e}")
        return {}

    if not isinstance(data, dict):
        logger.warning(f"params.yaml root must be a dict, got {type(data).__name__}")
        return {}

    # YAML dict has unknown key/value types from parsing arbitrary user input
    typed_data = cast("dict[Any, Any]", data)
    result = ParamsOverrides()
    for k, v in typed_data.items():
        if isinstance(v, dict):
            result[str(k)] = v
    return result


def build_params_instance[T: pydantic.BaseModel](
    params_cls: type[T],
    stage_name: str,
    overrides: ParamsOverrides | None = None,
) -> T:
    """Build Pydantic model instance with overrides applied.

    For matrix stages (name contains '@'), overrides are applied in order:
    1. Base stage name overrides (e.g., 'process' applies to all variants)
    2. Full variant name overrides (e.g., 'process@v1' for specific variant)

    Args:
        params_cls: The Pydantic BaseModel class
        stage_name: Name of the stage (for looking up overrides)
        overrides: Dict from load_params_yaml() or None

    Returns:
        Instantiated Pydantic model (preserves the specific type passed in)

    Raises:
        pydantic.ValidationError: If required fields missing or type mismatch
    """
    if overrides is None:
        return params_cls()

    merged_overrides = _get_merged_overrides(stage_name, overrides)
    return params_cls(**merged_overrides)


def _get_merged_overrides(stage_name: str, overrides: ParamsOverrides) -> dict[str, Any]:
    """Get merged overrides for a stage, applying matrix inheritance.

    Deep-copies values from overrides to prevent mutation of shared structures.
    """
    merged = dict[str, Any]()

    # For matrix stages, apply base name overrides first
    if "@" in stage_name:
        base_name = stage_name.split("@")[0]
        if base_name:  # Guard against names starting with @
            base_overrides = copy.deepcopy(overrides.get(base_name, {}))
            merged = deepmerge.always_merger.merge(merged, base_overrides)

    # Apply exact stage name overrides (more specific, takes precedence)
    stage_overrides = copy.deepcopy(overrides.get(stage_name, {}))
    return deepmerge.always_merger.merge(merged, stage_overrides)


def apply_overrides[T: pydantic.BaseModel](
    params_instance: T,
    stage_name: str,
    overrides: ParamsOverrides | None = None,
) -> T:
    """Apply overrides to an existing params instance with deep merging.

    For matrix stages (name contains '@'), overrides are applied in order:
    1. Base stage name overrides (e.g., 'process' applies to all variants)
    2. Full variant name overrides (e.g., 'process@v1' for specific variant)

    Nested BaseModel fields are merged recursively - only specified keys are
    overridden, preserving unspecified nested values.

    Args:
        params_instance: Existing Pydantic model instance
        stage_name: Name of the stage (for looking up overrides)
        overrides: Dict from load_params_yaml() or None

    Returns:
        New instance with overrides applied (original unchanged)
    """
    if overrides is None:
        return params_instance

    merged_overrides = _get_merged_overrides(stage_name, overrides)
    if not merged_overrides:
        return params_instance

    # Deep merge instance values with overrides
    base_dict = params_instance.model_dump()
    merged = deepmerge.always_merger.merge(base_dict, merged_overrides)
    return type(params_instance).model_validate(merged)


def validate_params_cls(params_cls: object) -> bool:
    """Check if params_cls is a valid Pydantic BaseModel subclass."""
    return isinstance(params_cls, type) and issubclass(params_cls, pydantic.BaseModel)


def get_effective_params(
    params_instance: pydantic.BaseModel | None,
    stage_name: str,
    overrides: ParamsOverrides | None = None,
) -> dict[str, Any]:
    """Get effective params dict for change detection, with overrides applied.

    This is the single source of truth for determining what params a stage will use.
    Used by both executor (for actual runs) and CLI (for dry-run predictions).

    Args:
        params_instance: Pydantic model instance from stage registration, or None
        stage_name: Name of the stage (for looking up overrides)
        overrides: Dict from load_params_yaml() or None

    Returns:
        Dict of effective parameter values, or empty dict if no params
    """
    if params_instance is None:
        return {}

    effective = apply_overrides(params_instance, stage_name, overrides)
    return effective.model_dump()
