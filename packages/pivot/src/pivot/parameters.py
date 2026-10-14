from __future__ import annotations

import contextlib
import copy
import dataclasses
import difflib
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import deepmerge
import pydantic
import pydantic.fields
import pydantic.json_schema
import yaml

from pivot import exceptions, project, yaml_config

if TYPE_CHECKING:
    import pathlib

    import pydantic_core

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


def load_params_yaml(pipeline: PipelineLike, path: pathlib.Path | None = None) -> ParamsOverrides:
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
    refused at any depth unless its model sets ``extra="allow"``; the walk in
    ``_merge_overrides`` finds them, so values are then checked with ``extra="ignore"``. When
    that check passes, the walk saw every mapping's model, so it saw every unknown key.
    """
    params_cls = type(params)
    unknown: _UnknownKeys = []
    set_twice = list[tuple[str, str]]()
    data = _merge_overrides(
        params_cls,
        params.model_dump(round_trip=True, by_alias=False),
        params,
        stage_name,
        overrides,
        unknown,
        set_twice,
    )
    if set_twice:
        return set_twice
    found = list[tuple[str, str]]()
    try:
        _ = params_cls.model_validate(data, by_alias=True, by_name=True, extra="ignore")
    except pydantic.ValidationError as exc:
        for error in exc.errors():
            loc = error["loc"]
            key = _override_key(params_cls, stage_name, loc[0] if loc else None, overrides)
            found.append((key, f"params.yaml key '{key}': {_describe_error(params_cls, error)}"))
    found.extend(
        (key, _unknown_field_message(key, path, owner))
        for key, path, owner in unknown
        if owner.model_config.get("extra") != "allow"
    )
    return found


def describe_validation_error(
    params_cls: type[pydantic.BaseModel], exc: pydantic.ValidationError
) -> str:
    """One line per error of validating ``params_cls``, never showing a secret's value."""
    return "\n".join(_describe_error(params_cls, error) for error in exc.errors())


def _describe_error(params_cls: type[pydantic.BaseModel], error: pydantic_core.ErrorDetails) -> str:
    """Describe a validation error of ``params_cls`` as ``field '<path>' ...``.

    The value is shown as ``'**********'`` when the top-level field holds a secret anywhere
    (``_holds_secret``), so a sibling value inside the same field is masked too. A key that
    is not a field is described without its value: it may be a misspelled secret.
    """
    loc = error["loc"]
    path = ".".join(str(part) for part in loc)
    if error["type"] == "extra_forbidden":
        return f"field '{path}' is not a field of {params_cls.__name__}"
    field = _field_named(params_cls, loc[0]) if loc else None
    secret = field is None or _holds_secret(field.annotation, set())
    shown = "'**********'" if secret else repr(error["input"])
    return f"field '{path}' value {shown} is invalid for {params_cls.__name__}: {error['msg']}"


def _field_named(
    model: type[pydantic.BaseModel], key: int | str
) -> pydantic.fields.FieldInfo | None:
    """Return ``model``'s field that ``key`` names by field name or input alias, or None."""
    for name, field in model.model_fields.items():
        if key == name or key in _input_aliases(field):
            return field
    return None


# What a secret is, for the record (``_encode``, from a value) and for errors
# (``_holds_secret``, from an annotation).
_SECRET_TYPES = (pydantic.SecretStr, pydantic.SecretBytes, pydantic.Secret)


def _holds_secret(annotation: object, seen: set[object]) -> bool:
    """Whether a value of ``annotation`` may contain one of ``_SECRET_TYPES``, at any depth:
    through unions, containers, ``Annotated``, nested models, and PEP 695 ``type`` aliases
    (``type Tok = SecretStr``, ``type Box[T] = list[T]``), recursive ones included."""
    if isinstance(annotation, typing.TypeAliasType):
        if annotation in seen:
            return False
        seen.add(annotation)
        return _holds_secret(annotation.__value__, seen)
    origin = typing.get_origin(annotation)
    if isinstance(origin, typing.TypeAliasType) and _holds_secret(origin, seen):
        return True
    for candidate in (annotation, origin):
        if isinstance(candidate, type) and issubclass(candidate, _SECRET_TYPES):
            return True
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        if annotation in seen:
            return False
        seen.add(annotation)
        return any(_holds_secret(f.annotation, seen) for f in annotation.model_fields.values())
    return any(_holds_secret(arg, seen) for arg in typing.get_args(annotation))


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
        ParamsError: One key sets a field twice, by its name and by an alias.
    """
    if overrides is None:
        return params_cls()
    set_twice = list[tuple[str, str]]()
    data = _merge_overrides(params_cls, {}, None, stage_name, overrides, [], set_twice)
    if set_twice:
        raise exceptions.ParamsError("; ".join(message for _, message in set_twice))
    return params_cls.model_validate(data, by_alias=True, by_name=True)


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
    overridden, preserving unspecified nested values. A field may be spelled by its name
    or any of its aliases, at any depth.

    Args:
        params_instance: Existing Pydantic model instance
        stage_name: Name of the stage (for looking up overrides)
        overrides: Dict from load_params_yaml() or None

    Returns:
        New instance with overrides applied (original unchanged)

    Raises:
        pydantic.ValidationError: The merged values fail validation.
        ParamsError: One key sets a field twice, by its name and by an alias.
    """
    if overrides is None or not any(key in overrides for key in _override_keys(stage_name)):
        return params_instance

    params_cls = type(params_instance)
    set_twice = list[tuple[str, str]]()
    # The round-trip dump is keyed by field name (by_alias=False holds even when the model
    # serializes by alias) and leaves out computed fields, which a model that forbids extra
    # fields would refuse.
    data = _merge_overrides(
        params_cls,
        params_instance.model_dump(round_trip=True, by_alias=False),
        params_instance,
        stage_name,
        overrides,
        [],
        set_twice,
    )
    if set_twice:
        raise exceptions.ParamsError("; ".join(message for _, message in set_twice))
    return params_cls.model_validate(data, by_alias=True, by_name=True)


# (params.yaml key, path of field names, list indexes and dict keys, model the key was given to)
type _UnknownKeys = list[tuple[str, tuple[int | str, ...], type[pydantic.BaseModel]]]


def _merge_overrides(
    params_cls: type[pydantic.BaseModel],
    data: dict[str, Any],
    structure: object,
    stage_name: str,
    overrides: ParamsOverrides,
    unknown: _UnknownKeys,
    set_twice: list[tuple[str, str]],
) -> dict[str, Any]:
    """Deep-merge the stage's params.yaml keys, base key first, into ``data``.

    The merge needs one spelling per field, so a variant's ``learning_rate:`` replaces its
    base key's ``lr:``. Each key's value is first merged as written and validated with
    ``extra="ignore"``, only to learn its structure: which model each mapping became, and
    which union arm was taken. ``_normalize`` then walks the value beside that structure.
    When the merged value fails validation, the keys that failed are left out of the
    structure, so one bad value does not hide a typo elsewhere; the final validation
    reports the value. Unknown keys go to ``unknown`` and fields set twice to ``set_twice``.
    Deep-copies values from overrides to prevent mutation of shared structures.
    """
    for key in _override_keys(stage_name):
        if key not in overrides:
            continue
        raw = copy.deepcopy(overrides[key])
        try:
            structure = _structure_of(params_cls, data, raw)
        except pydantic.ValidationError as exc:
            failed = {error["loc"][0] for error in exc.errors() if error["loc"]}
            kept = {name: value for name, value in raw.items() if name not in failed}
            with contextlib.suppress(pydantic.ValidationError):
                structure = _structure_of(params_cls, data, kept)
        twice = list[str]()
        normalized = _normalize(raw, structure, key, (), unknown, twice)
        set_twice.extend((key, f"params.yaml key '{key}': {message}") for message in twice)
        data = _params_merger.merge(data, normalized)
    return data


def _structure_of(
    params_cls: type[pydantic.BaseModel], data: dict[str, Any], raw: dict[str, Any]
) -> pydantic.BaseModel:
    candidate = _params_merger.merge(copy.deepcopy(data), copy.deepcopy(raw))
    return params_cls.model_validate(candidate, by_alias=True, by_name=True, extra="ignore")


def _normalize(
    value: Any,
    structure: object,
    key: str,
    path: tuple[int | str, ...],
    unknown: _UnknownKeys,
    twice: list[str],
) -> Any:
    """Return ``value`` with each mapping that became a model keyed by field name.

    ``structure`` is what ``value`` validated to. A mapping beside a model has each key that
    is a field's alias renamed to the field's name; a key that names no field is kept, so
    validation sees it, and added to ``unknown``. Lists and tuples pair by position, dicts
    by key. Anything else is returned as written.
    """
    if isinstance(value, dict) and isinstance(structure, pydantic.BaseModel):
        model = type(structure)
        fields = model.model_fields
        names = {alias: name for name, info in fields.items() for alias in _input_aliases(info)}
        normalized = dict[str, Any]()
        spelled = dict[str, str]()
        for item_key, item in cast("dict[str, Any]", value).items():
            name = item_key if item_key in fields else names.get(item_key)
            if name is None:
                normalized[item_key] = item
                unknown.append((key, (*path, item_key), model))
                continue
            if name in spelled:
                field_path = ".".join(str(part) for part in (*path, name))
                twice.append(
                    f"field '{field_path}' is set twice, as '{spelled[name]}' and as '{item_key}'"
                )
                continue
            spelled[name] = item_key
            nested = getattr(structure, name, None)
            normalized[name] = _normalize(item, nested, key, (*path, name), unknown, twice)
        return normalized
    if isinstance(value, dict) and isinstance(structure, dict):
        held = cast("dict[Any, Any]", structure)
        return {
            item_key: _normalize(item, held.get(item_key), key, (*path, item_key), unknown, twice)
            for item_key, item in cast("dict[Any, Any]", value).items()
        }
    if isinstance(value, (list, tuple)) and isinstance(structure, (list, tuple)):
        held_items = list(cast("list[Any] | tuple[Any, ...]", structure))
        return [
            _normalize(
                item,
                held_items[index] if index < len(held_items) else None,
                key,
                (*path, index),
                unknown,
                twice,
            )
            for index, item in enumerate(cast("list[Any] | tuple[Any, ...]", value))
        ]
    return cast("Any", value)


def _input_aliases(field: pydantic.fields.FieldInfo) -> list[str]:
    """The keys besides its name that validation may read a field from: its string alias,
    or each string choice of its AliasChoices. An AliasPath is not a key."""
    alias = field.validation_alias
    if isinstance(alias, str):
        return [alias]
    if isinstance(alias, pydantic.AliasChoices):
        return [choice for choice in alias.choices if isinstance(choice, str)]
    return []


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
    members sorted and every secret replaced by a keyed digest of its value.

    Pydantic encodes a set or frozenset as a list in iteration order, which follows
    PYTHONHASHSEED for str members, so members are sorted by their JSON text. It dumps a
    ``SecretStr``, ``SecretBytes`` or ``Secret[T]`` as ``'**********'``, so a changed secret
    would not change the record and the stage would not re-run. Lock files are committed, so
    the record must not let anyone test guesses either: a secret becomes ``hmac:`` plus the
    first 16 hex digits of its value's HMAC-SHA256 under the project's secret key
    (``_secret_key``). The key is read only when a secret is found.
    """
    key: bytes | None = None

    def digest(secret: pydantic.SecretStr | pydantic.SecretBytes | pydantic.Secret[Any]) -> str:
        nonlocal key
        if key is None:
            key = _secret_key()
        return _secret_digest(key, secret)

    return _encode(params, params.model_dump(mode="json"), digest)


_SECRET_KEY_ENV = "PIVOT_SECRET_KEY"
_SECRET_KEY_FILE = "secret.key"


def _secret_key() -> bytes:
    """Return the 32-byte key secret digests are made with.

    ``PIVOT_SECRET_KEY`` (64 hex digits) wins, so CI and teammates can share digests.
    Otherwise the key is ``.pivot/secret.key`` under the project root, created on first use
    with mode 0600 and added to ``.pivot/.gitignore``: it must never be committed. A clone
    without the key makes a new one, so its stages with secrets re-run once.

    Raises:
        ParamsError: ``PIVOT_SECRET_KEY`` or the key file does not hold 32 hex-encoded bytes.
    """
    if (from_env := os.environ.get(_SECRET_KEY_ENV)) is not None:
        return _parse_secret_key(from_env, _SECRET_KEY_ENV)
    pivot_dir = project.get_project_root() / ".pivot"
    key_file = pivot_dir / _SECRET_KEY_FILE
    if not key_file.exists():
        _create_secret_key(pivot_dir, key_file)
    return _parse_secret_key(key_file.read_text(), str(key_file))


def _parse_secret_key(text: str, source: str) -> bytes:
    try:
        key = bytes.fromhex(text.strip())
    except ValueError:
        key = b""
    if len(key) != 32:
        raise exceptions.ParamsError(f"{source} must be 64 hex digits (a 32-byte key)")
    return key


def _create_secret_key(pivot_dir: pathlib.Path, key_file: pathlib.Path) -> None:
    """Write a random key to ``key_file`` with mode 0600, and gitignore it.

    The key is written to a private temporary file and hard-linked into place, so a worker
    racing to create it reads either no file or a whole key, and one key wins. Only the
    worker whose link made the key touches ``.gitignore``, after the link, and it only
    appends the entry when it is missing: the file is never rewritten, so a racing reader
    never sees it truncated.
    """
    descriptor, temporary = tempfile.mkstemp(dir=pivot_dir, prefix=".secret-key-")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(secrets.token_hex(32) + "\n")
        try:
            os.link(temporary, key_file)
        except FileExistsError:
            return
    finally:
        os.unlink(temporary)
    gitignore = pivot_dir / ".gitignore"
    text = gitignore.read_text() if gitignore.exists() else ""
    if f"/{_SECRET_KEY_FILE}" not in text.splitlines():
        with gitignore.open("a") as handle:
            handle.write(
                ("\n" if text and not text.endswith("\n") else "") + f"/{_SECRET_KEY_FILE}\n"
            )


def _secret_digest(
    key: bytes, secret: pydantic.SecretStr | pydantic.SecretBytes | pydantic.Secret[Any]
) -> str:
    value: object = secret.get_secret_value()
    if isinstance(value, str):
        data = value.encode()
    elif isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, sort_keys=True, default=str).encode()
    return "hmac:" + hmac.new(key, data, hashlib.sha256).hexdigest()[:16]


type _Digest = Callable[[pydantic.SecretStr | pydantic.SecretBytes | pydantic.Secret[Any]], str]


def _encode(original: object, encoded: Any, digest: _Digest | None) -> Any:
    """Walk `encoded`, a JSON-mode dump of `original`, sorting set members (after their own
    nested sets) and, given ``digest``, replacing each secret by its digest.

    The walk pairs each encoded node with the object it came from: JSON mode iterates the
    stored set itself, so the pairing is exact. A node that a custom serializer reshaped no
    longer matches its object and is left as the serializer produced it. A secret whose own
    node survives serialization is replaced whatever its serializer produced. A serializer
    on a container that calls ``get_secret_value()`` itself (a ``field_serializer`` joining a
    ``list[SecretStr]`` into one string, a ``model_serializer`` exposing a secret) reshapes
    the node, so its output, the value included, is recorded as written.
    """
    if digest is not None and isinstance(original, _SECRET_TYPES):
        return digest(original)
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
                key: _encode(getattr(original, attributes.get(key, key), None), value, digest)
                for key, value in encoded_dict.items()
            }
        if dataclasses.is_dataclass(original) and not isinstance(original, type):
            return {
                key: _encode(getattr(original, key, None), value, digest)
                for key, value in encoded_dict.items()
            }
        if isinstance(original, dict):
            original_values = list(cast("dict[Any, Any]", original).values())
            if len(original_values) == len(encoded_dict):
                return {
                    key: _encode(item, value, digest)
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
            _encode(item, value, digest)
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
        return _encode(dft, super().encode_default(dft), None)
