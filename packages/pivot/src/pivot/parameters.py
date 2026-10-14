from __future__ import annotations

import copy
import dataclasses
import difflib
import hashlib
import json
import logging
import types
import typing
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import deepmerge
import pydantic
import pydantic.fields
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
            params class rejects the values or field names its keys give it. Every problem
            is listed, grouped by key in file order. A key for a pipeline not loaded in this
            run is skipped and logged at debug.
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
    the variant's own key when it sets the field, otherwise the base key. An unknown key is
    refused at any depth unless its model sets ``extra="allow"``: pydantic reports it for a
    model that forbids extra fields, and the walk here for one that would ignore it.
    """
    params_cls = type(params)
    found = list[tuple[str, str]]()
    set_twice = list[tuple[str, str]]()
    owners = dict[tuple[int | str, ...], type[pydantic.BaseModel]]()
    for key in _override_keys(stage_name):
        unknown: _KeyPaths = []
        try:
            _ = _rename_fields(params_cls, overrides.get(key, {}), unknown)
        except exceptions.ParamsError as exc:
            set_twice.append((key, f"params.yaml key '{key}': {exc}"))
            continue
        for path, owner in unknown:
            owners[path] = owner
            if owner.model_config.get("extra") not in ("allow", "forbid"):
                found.append((key, _unknown_field_message(key, path, owner)))
    if set_twice:
        return set_twice
    try:
        _ = apply_overrides(params, stage_name, overrides)
    except pydantic.ValidationError as exc:
        for error in exc.errors():
            loc = error["loc"]
            key = _override_key(params_cls, stage_name, loc[0] if loc else None, overrides)
            if error["type"] == "extra_forbidden" and loc in owners:
                found.append((key, _unknown_field_message(key, loc, owners[loc])))
                continue
            path = ".".join(str(part) for part in loc)
            detail = (
                f"value {error['input']!r} is invalid for {params_cls.__name__}: " + error["msg"]
            )
            found.append((key, f"params.yaml key '{key}': field '{path}' {detail}"))
    return found


def _unknown_field_message(
    key: str, path: tuple[int | str, ...], owner: type[pydantic.BaseModel]
) -> str:
    return (
        f"params.yaml key '{key}': field '{'.'.join(str(part) for part in path)}' is not a "
        + f"field of {owner.__name__}; "
        + _field_hint(str(path[-1]), _field_keys(owner))
    )


def _field_keys(model: type[pydantic.BaseModel]) -> list[str]:
    """The keys params.yaml may use for ``model``'s fields: each field's name, then its aliases."""
    return [
        key for name, field in model.model_fields.items() for key in (name, *_input_aliases(field))
    ]


def _field_hint(name: str, fields: list[str]) -> str:
    matches = difflib.get_close_matches(name, fields, n=3, cutoff=0.6)
    if matches:
        return "did you mean " + " or ".join(f"'{m}'" for m in matches) + "?"
    if fields:
        return "its fields are " + ", ".join(f"'{f}'" for f in fields)
    return "it has no fields"


def _override_keys(stage_name: str) -> list[str]:
    """The params.yaml keys that apply to ``stage_name``, base key first."""
    base = stage_name.split("@")[0]
    return [base, stage_name] if "@" in stage_name and base else [stage_name]


def _override_key(
    model: type[pydantic.BaseModel],
    stage_name: str,
    field: int | str | None,
    overrides: ParamsOverrides,
) -> str:
    """Name the params.yaml key that sets ``field`` (by name or alias) for ``stage_name``."""
    info = model.model_fields.get(field) if isinstance(field, str) else None
    spellings = {field} if info is None else {field, *_input_aliases(info)}
    keys = _override_keys(stage_name)
    for key in reversed(keys):
        if key in overrides and (field is None or spellings & set(overrides[key])):
            return key
    return next((key for key in reversed(keys) if key in overrides), keys[-1])


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

    merged_overrides = _get_merged_overrides(params_cls, stage_name, overrides)
    return params_cls.model_validate(merged_overrides, by_alias=False, by_name=True)


def _get_merged_overrides(
    model: type[pydantic.BaseModel], stage_name: str, overrides: ParamsOverrides
) -> dict[str, Any]:
    """Get merged overrides for a stage, keyed by field name, base-name overrides first.

    Each key's fields are renamed from alias to name before merging, so a variant's
    ``learning_rate:`` replaces its base key's ``lr:``. Deep-copies values from overrides to
    prevent mutation of shared structures.

    Raises:
        ParamsError: One key sets a field twice, by its name and by an alias.
    """
    merged = dict[str, Any]()
    for key in _override_keys(stage_name):
        renamed = _rename_fields(model, copy.deepcopy(overrides.get(key, {})))
        merged = _params_merger.merge(merged, renamed)
    return merged


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

    params_cls = type(params_instance)
    merged_overrides = _get_merged_overrides(params_cls, stage_name, overrides)
    if not merged_overrides:
        return params_instance

    # Everything is keyed by field name and validated by name only, so any alias form works:
    # a string alias, AliasChoices, an AliasPath. The round-trip dump leaves out computed
    # fields, which a model that forbids extra fields would refuse; by_alias=False holds even
    # when the model serializes by alias.
    base_dict = params_instance.model_dump(round_trip=True, by_alias=False)
    merged = _params_merger.merge(base_dict, merged_overrides)
    return params_cls.model_validate(merged, by_alias=False, by_name=True)


def _input_aliases(field: pydantic.fields.FieldInfo) -> list[str]:
    """The keys besides its name that validation may read a field from: its string alias,
    or each string choice of its AliasChoices. An AliasPath is not a key."""
    alias = field.validation_alias
    if isinstance(alias, str):
        return [alias]
    if isinstance(alias, pydantic.AliasChoices):
        return [choice for choice in alias.choices if isinstance(choice, str)]
    return []


def _nested_model(annotation: object) -> type[pydantic.BaseModel] | None:
    """Return the model a field holds as ``M`` or ``M | None``, or None."""
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is types.UnionType or origin is typing.Union:  # pyright: ignore[reportDeprecated] - runtime identity check
        models = [m for arg in typing.get_args(annotation) if (m := _nested_model(arg)) is not None]
        if len(models) == 1:
            return models[0]
    return None


type _KeyPaths = list[tuple[tuple[int | str, ...], type[pydantic.BaseModel]]]


def _rename_fields(
    model: type[pydantic.BaseModel],
    values: dict[str, Any],
    unknown: _KeyPaths | None = None,
    path: tuple[int | str, ...] = (),
) -> dict[str, Any]:
    """Key ``values`` by field name, through nested models.

    A key may be a field's name or one of its input aliases. Keys that name no field are
    kept, so validation sees them, and each is added to ``unknown`` with its path (field
    names, list indexes, dict keys) and the model it was given to.

    Raises:
        ParamsError: ``values`` sets one field under two of its keys.
    """
    fields = model.model_fields
    names = {alias: name for name, field in fields.items() for alias in _input_aliases(field)}
    renamed = dict[str, Any]()
    spelled = dict[str, str]()
    for key, value in values.items():
        name = key if key in fields else names.get(key)
        if name is None:
            renamed[key] = value
            if unknown is not None:
                unknown.append(((*path, key), model))
            continue
        if name in spelled:
            raise exceptions.ParamsError(
                f"field '{name}' is set twice, as '{spelled[name]}' and as '{key}'"
            )
        spelled[name] = key
        renamed[name] = _rename_nested(fields[name].annotation, value, unknown, (*path, name))
    return renamed


def _rename_nested(
    annotation: object, value: Any, unknown: _KeyPaths | None, path: tuple[int | str, ...]
) -> Any:
    """Apply ``_rename_fields`` inside a value held as ``M``, ``M | None``, a sequence of M,
    or a dict of M."""
    model = _nested_model(annotation)
    if model is not None:
        if isinstance(value, dict):
            return _rename_fields(model, cast("dict[str, Any]", value), unknown, path)
        return value
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (list, tuple) and args and isinstance(value, (list, tuple)):
        item_type = args[0] if origin is list or args[1:] == (Ellipsis,) else None
        if item_type is not None:
            items: list[Any] = list(cast("list[Any]", value))
            return [
                _rename_nested(item_type, item, unknown, (*path, index))
                for index, item in enumerate(items)
            ]
    if origin is dict and len(args) == 2 and isinstance(value, dict):
        entries = cast("dict[Any, Any]", value)
        return {
            key: _rename_nested(args[1], item, unknown, (*path, key))
            for key, item in entries.items()
        }
    return cast("Any", value)


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
    return encode_params(effective)


def encode_params(params: pydantic.BaseModel) -> Any:
    """Return ``params`` as the lock file records them: a JSON-mode dump with every set's
    members sorted and every secret replaced by a digest of its value.

    Pydantic encodes a set or frozenset as a list in iteration order, which follows
    PYTHONHASHSEED for str members, so members are sorted by their JSON text. It dumps a
    ``SecretStr``, ``SecretBytes`` or ``Secret[T]`` as ``'**********'``, so a changed secret
    would not change the record and the stage would not re-run; the record must not hold the
    value either, so a secret becomes ``sha256:`` plus the first 16 hex digits of its
    value's SHA-256.
    """
    return _encode(params, params.model_dump(mode="json"), digest_secrets=True)


def _secret_digest(secret: pydantic.SecretStr | pydantic.SecretBytes | pydantic.Secret[Any]) -> str:
    value: object = secret.get_secret_value()
    if isinstance(value, str):
        data = value.encode()
    elif isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()[:16]


def _encode(original: object, encoded: Any, *, digest_secrets: bool) -> Any:
    """Walk `encoded`, a JSON-mode dump of `original`, sorting set members (after their own
    nested sets) and, with ``digest_secrets``, replacing each secret by its digest.

    The walk pairs each encoded node with the object it came from: JSON mode iterates the
    stored set itself, so the pairing is exact. A node that a custom serializer reshaped no
    longer matches its object and is left as the serializer produced it; a secret is
    replaced whatever its serializer produced, so no serializer puts its value in the record.
    """
    if digest_secrets and isinstance(
        original, (pydantic.SecretStr, pydantic.SecretBytes, pydantic.Secret)
    ):
        return _secret_digest(original)
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
                key: _encode(
                    getattr(original, attributes.get(key, key), None),
                    value,
                    digest_secrets=digest_secrets,
                )
                for key, value in encoded_dict.items()
            }
        if dataclasses.is_dataclass(original) and not isinstance(original, type):
            return {
                key: _encode(getattr(original, key, None), value, digest_secrets=digest_secrets)
                for key, value in encoded_dict.items()
            }
        if isinstance(original, dict):
            original_values = list(cast("dict[Any, Any]", original).values())
            if len(original_values) == len(encoded_dict):
                return {
                    key: _encode(item, value, digest_secrets=digest_secrets)
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
            _encode(item, value, digest_secrets=digest_secrets)
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
        return _encode(dft, super().encode_default(dft), digest_secrets=False)
