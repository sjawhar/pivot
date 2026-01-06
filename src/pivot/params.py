"""Parameter loading and validation for Pydantic-based stage parameters.

Supports loading parameters from YAML files with fallback to Pydantic model defaults.
Parameters are injected into stage functions via a single `params` argument.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import yaml
from pydantic import BaseModel

from pivot import project

if TYPE_CHECKING:
    import inspect
    import pathlib

logger = logging.getLogger(__name__)

_Loader: type[yaml.SafeLoader] | type[yaml.CSafeLoader]
try:
    _Loader = yaml.CSafeLoader
except AttributeError:
    _Loader = yaml.SafeLoader

# Type alias for YAML params structure: stage_name -> param_name -> param_value
# Inner dict values are Any because YAML can contain arbitrary JSON-compatible types
YamlParams = dict[str, dict[str, Any]]


def load_params_yaml(path: pathlib.Path | None = None) -> YamlParams:
    """Load params.yaml from project root or specified path.

    Returns dict of stage_name -> param_dict. Returns empty dict if file missing.
    """
    if path is None:
        path = project.get_project_root() / "params.yaml"

    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data: object = yaml.load(f, Loader=_Loader)
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Failed to load params.yaml: {e}")
        return {}

    if not isinstance(data, dict):
        logger.warning(f"params.yaml root must be a dict, got {type(data).__name__}")
        return {}

    # YAML dict has unknown key/value types from parsing arbitrary user input
    typed_data = cast("dict[Any, Any]", data)
    result = YamlParams()
    for k, v in typed_data.items():
        if isinstance(v, dict):
            result[str(k)] = v
    return result


def build_params_instance[T: BaseModel](
    params_cls: type[T],
    stage_name: str,
    yaml_overrides: YamlParams | None = None,
) -> T:
    """Build Pydantic model instance with YAML overrides applied.

    Args:
        params_cls: The Pydantic BaseModel class
        stage_name: Name of the stage (for looking up YAML overrides)
        yaml_overrides: Dict from load_params_yaml() or None

    Returns:
        Instantiated Pydantic model (preserves the specific type passed in)

    Raises:
        pydantic.ValidationError: If required fields missing or type mismatch
    """
    if yaml_overrides is None:
        yaml_overrides = {}

    stage_overrides = yaml_overrides.get(stage_name, {})
    return params_cls(**stage_overrides)


def params_to_dict(params: BaseModel) -> dict[str, Any]:
    """Convert Pydantic model instance to dict for lock file storage."""
    return params.model_dump()


def validate_params_cls(params_cls: object) -> bool:
    """Check if params_cls is a valid Pydantic BaseModel subclass."""
    return isinstance(params_cls, type) and issubclass(params_cls, BaseModel)


def extract_stage_params(
    params_cls: type[BaseModel] | None,
    signature: inspect.Signature | None,
    stage_name: str,
    yaml_overrides: YamlParams | None = None,
) -> tuple[dict[str, Any], BaseModel | None]:
    """Extract params dict and optionally the Pydantic instance for a stage.

    For Pydantic params_cls: builds instance with YAML overrides and converts to dict.
    For legacy stages: extracts defaults from function signature.

    Returns:
        Tuple of (params_dict, params_instance). params_instance is None for legacy stages.
    """
    if params_cls is not None:
        instance = build_params_instance(params_cls, stage_name, yaml_overrides)
        return params_to_dict(instance), instance

    # Legacy: extract from signature defaults
    if not signature:
        return {}, None

    params_dict = {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not param.empty
    }
    return params_dict, None
