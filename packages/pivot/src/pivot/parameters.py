from __future__ import annotations

import copy
import dataclasses
import difflib
import json
import logging
import typing
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import deepmerge
import pydantic
import pydantic.json_schema
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
    """Load params.yaml from the project root (or ``path``) and check it against ``pipeline``.

    Keys are qualified stage names (``<pipeline>/<stage>``); a key without ``@...`` applies to
    every variant of that stage. Returns stage_name -> param_dict, or an empty dict when the
    file is missing.

    Raises:
        ParamsError: The file cannot be read or parsed, or its root is not a mapping.
        PipelineConfigError: A key has no pipeline part, names no stage of a pipeline loaded
            in this run, or maps to something other than parameter values; or a stage's
            params class rejects the values its keys give it. Every problem is listed, in
            file order. A key for a pipeline not loaded in this run is skipped and logged at
            debug.
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
    errors_by_key = dict[str, list[str]]()
    for k, v in typed_data.items():
        key = str(k)
        key_errors = errors_by_key.setdefault(key, list[str]())
        if not isinstance(v, dict):
            key_errors.append(
                f"params.yaml key '{key}' must map parameter names to values, "
                + f"got {type(v).__name__}"
            )
            continue
        result[key] = v
        if key not in valid_keys and (error := _unmatched_key_error(key, valid_keys, loaded)):
            key_errors.append(error)
    for stage_name in stage_names:
        params = pipeline.get_stage(stage_name)["params"]
        if params is None:
            continue
        for key, message in _override_errors(params, stage_name, result):
            if message not in errors_by_key[key]:
                errors_by_key[key].append(message)
    errors = [message for messages in errors_by_key.values() for message in messages]
    if errors:
        raise exceptions.PipelineConfigError("\n".join(errors))
    return result


def _override_errors(
    params: pydantic.BaseModel, stage_name: str, overrides: ParamsOverrides
) -> list[tuple[str, str]]:
    """Validate a stage's params with its overrides applied, as a run does.

    Returns (params.yaml key, message) for each error, naming the key that gave the field:
    the variant's own key when it sets the field, otherwise the base key.
    """
    try:
        _ = apply_overrides(params, stage_name, overrides)
    except pydantic.ValidationError as exc:
        params_cls = type(params).__name__
        found = list[tuple[str, str]]()
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else None
            key = _override_key(stage_name, field, overrides)
            path = ".".join(str(part) for part in error["loc"])
            found.append(
                (
                    key,
                    f"params.yaml key '{key}': field '{path}' value {error['input']!r} "
                    + f"is invalid for {params_cls}: {error['msg']}",
                )
            )
        return found
    return []


def _override_key(stage_name: str, field: str | None, overrides: ParamsOverrides) -> str:
    """Name the params.yaml key that sets ``field`` for ``stage_name``."""
    base = stage_name.split("@")[0]
    for key in (stage_name, base):
        if key in overrides and (field is None or field in overrides[key]):
            return key
    return stage_name if stage_name in overrides else base


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
    return sort_set_members(effective, effective.model_dump(mode="json"))


def sort_set_members(original: object, encoded: Any) -> Any:
    """Return `encoded`, a JSON-mode dump of `original`, with every set's members sorted.

    Pydantic encodes a set or frozenset as a list in iteration order, and that order follows
    PYTHONHASHSEED for str members. Recorded params and hashed schemas would then differ
    between processes. Members are sorted by their JSON text, after their own nested sets
    are sorted. The walk pairs each encoded node with the object it came from: JSON mode
    iterates the stored set itself, so the pairing is exact. A node that a custom serializer
    reshaped no longer matches its object and is left as the serializer produced it.
    """
    if isinstance(encoded, dict):
        encoded_dict = cast("dict[str, Any]", encoded)
        if isinstance(original, pydantic.BaseModel):
            fields = type(original).model_fields
            attributes = {name: name for name in fields}
            for name, field in fields.items():
                for alias in (field.serialization_alias, field.alias):
                    if alias is not None:
                        attributes.setdefault(alias, name)
            return {
                key: sort_set_members(getattr(original, attributes.get(key, key), None), value)
                for key, value in encoded_dict.items()
            }
        if dataclasses.is_dataclass(original) and not isinstance(original, type):
            return {
                key: sort_set_members(getattr(original, key, None), value)
                for key, value in encoded_dict.items()
            }
        if isinstance(original, dict):
            original_values = list(cast("dict[Any, Any]", original).values())
            if len(original_values) == len(encoded_dict):
                return {
                    key: sort_set_members(item, value)
                    for (key, value), item in zip(
                        encoded_dict.items(), original_values, strict=True
                    )
                }
        return encoded_dict
    if isinstance(encoded, list) and isinstance(original, (list, tuple, set, frozenset)):
        encoded_items = cast("list[Any]", encoded)
        original_items = list(cast("list[Any] | tuple[Any, ...] | set[Any]", original))
        if len(original_items) != len(encoded_items):
            return encoded_items
        members = [
            sort_set_members(item, value)
            for item, value in zip(original_items, encoded_items, strict=True)
        ]
        if isinstance(original, (set, frozenset)):
            members.sort(key=lambda member: json.dumps(member, sort_keys=True))
        return members
    return cast("Any", encoded)


class SortedSetJsonSchema(pydantic.json_schema.GenerateJsonSchema):
    """JSON schema generator that lists a set default's members in sorted order.

    The stock generator lists them in iteration order, which follows PYTHONHASHSEED, so a
    fingerprint hashing the schema would change from one process to the next.
    """

    @typing.override
    def encode_default(self, dft: Any) -> Any:
        return sort_set_members(dft, super().encode_default(dft))
