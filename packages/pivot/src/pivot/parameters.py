from __future__ import annotations

import copy
import difflib
import logging
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import deepmerge
import pydantic
import yaml

from pivot import exceptions, project, yaml_config

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.registry import PipelineLike

logger = logging.getLogger(__name__)

# Custom merger: replace lists instead of appending (deepmerge default appends)
_params_merger = deepmerge.Merger(  # pyright: ignore[reportPrivateImportUsage]
    type_strategies=[
        (list, ["override"]),  # Replace lists entirely
        (dict, ["merge"]),  # Merge dicts recursively
    ],
    fallback_strategies=["override"],
    type_conflict_strategies=["override"],
)

# Type alias for params overrides: stage_name -> param_name -> param_value
# Inner dict values are Any because config files can contain arbitrary JSON-compatible types
ParamsOverrides = dict[str, dict[str, Any]]


def load_params_yaml(pipeline: PipelineLike, path: Path | None = None) -> ParamsOverrides:
    """Load params.yaml from the project root (or ``path``) and check its keys against ``pipeline``.

    Keys are qualified stage names (``<pipeline>/<stage>``); a key without ``@...`` applies to
    every variant of that stage. Returns stage_name -> param_dict, or an empty dict when the
    file is missing.

    Raises:
        ParamsError: The file cannot be read or parsed, or its root is not a mapping.
        PipelineConfigError: A key has no pipeline part, names no stage of a pipeline loaded
            in this run, or maps to something other than parameter values. Every bad key is
            listed. A key for a pipeline not loaded in this run is skipped and logged at debug.
    """
    if path is None:
        path = project.get_project_root() / "params.yaml"

    if not path.exists():
        return {}

    try:
        with open(path) as f:
            data: object = yaml.load(f, Loader=yaml_config.Loader)
    except yaml.YAMLError as e:
        raise exceptions.ParamsError(f"Failed to parse {path}: {e}") from e
    except OSError as e:
        raise exceptions.ParamsError(f"Failed to read {path}: {e}") from e

    if not isinstance(data, dict):
        raise exceptions.ParamsError(f"params.yaml root must be a dict, got {type(data).__name__}")

    stage_names = pipeline.list_stages()
    valid_keys = set(stage_names) | {name.split("@")[0] for name in stage_names}
    loaded = pipeline.loaded_pipelines

    # YAML dict has unknown key/value types from parsing arbitrary user input
    typed_data = cast("dict[Any, Any]", data)
    result = ParamsOverrides()
    errors = list[str]()
    for k, v in typed_data.items():
        key = str(k)
        if not isinstance(v, dict):
            errors.append(
                f"params.yaml key '{key}' must map parameter names to values, "
                + f"got {type(v).__name__}"
            )
            continue
        result[key] = v
        if key not in valid_keys and (error := _unmatched_key_error(key, valid_keys, loaded)):
            errors.append(error)
    if errors:
        raise exceptions.PipelineConfigError("\n".join(errors))
    return result


def _unmatched_key_error(key: str, valid_keys: set[str], loaded: frozenset[str]) -> str | None:
    """Explain a key that names no registered stage, or return None for an unloaded pipeline.

    A stage name is ``<pipeline>/<function>`` plus ``@variant`` parts, none of which contain
    ``/``, so everything before a key's last ``/`` is its pipeline.
    """
    if "/" not in key:
        examples = sorted(k for k in valid_keys if k.endswith(f"/{key}")) or sorted(valid_keys)[:1]
        example = ", e.g. " + " or ".join(f"'{k}'" for k in examples) if examples else ""
        return f"params.yaml key '{key}' names no stage; keys are '<pipeline>/<stage>'{example}"
    owner = key.rsplit("/", 1)[0]
    if owner not in loaded:
        logger.debug(
            f"params.yaml key '{key}': pipeline '{owner}' is not loaded in this run; skipping"
        )
        return None
    candidates = sorted(k for k in valid_keys if k.startswith(f"{owner}/"))
    matches = difflib.get_close_matches(key, candidates, n=3, cutoff=0.6)
    hint = (
        "did you mean " + " or ".join(f"'{m}'" for m in matches) + "?"
        if matches
        else "run 'pivot list' to see its stages"
    )
    return f"params.yaml key '{key}' names no stage in pipeline '{owner}'; {hint}"


def build_params_instance[T: pydantic.BaseModel](
    params_cls: type[T],
    stage_name: str,
    overrides: ParamsOverrides | None = None,
) -> T:
    """Build Pydantic model instance with overrides applied.

    For variants (stage names containing '@'), overrides are applied in order:
    1. Base stage name overrides (e.g., 'demo/process' applies to all variants)
    2. Full variant name overrides (e.g., 'demo/process@v1' for one variant)

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
    """Get merged overrides for a stage, applying base-name overrides to variants.

    Deep-copies values from overrides to prevent mutation of shared structures.
    """
    merged = dict[str, Any]()

    # For variants, apply base name overrides first
    if "@" in stage_name:
        base_name = stage_name.split("@")[0]
        if base_name:  # Guard against names starting with @
            base_overrides = copy.deepcopy(overrides.get(base_name, {}))
            merged = _params_merger.merge(merged, base_overrides)

    # Apply exact stage name overrides (more specific, takes precedence)
    stage_overrides = copy.deepcopy(overrides.get(stage_name, {}))
    return _params_merger.merge(merged, stage_overrides)


def apply_overrides[T: pydantic.BaseModel](
    params_instance: T,
    stage_name: str,
    overrides: ParamsOverrides | None = None,
) -> T:
    """Apply overrides to an existing params instance with deep merging.

    For variants (stage names containing '@'), overrides are applied in order:
    1. Base stage name overrides (e.g., 'demo/process' applies to all variants)
    2. Full variant name overrides (e.g., 'demo/process@v1' for one variant)

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
    merged = _params_merger.merge(base_dict, merged_overrides)
    return type(params_instance).model_validate(merged)


def validate_params_cls(params_cls: object) -> TypeGuard[type[pydantic.BaseModel]]:
    """Check if params_cls is a valid Pydantic BaseModel subclass.

    This TypeGuard allows the type checker to narrow the type after a check:

        if validate_params_cls(cls):
            # cls is now type[pydantic.BaseModel]
            instance = cls()
    """
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
    return effective.model_dump(mode="json")
