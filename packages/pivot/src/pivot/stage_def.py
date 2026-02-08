from __future__ import annotations

import dataclasses
import logging
import pathlib  # noqa: TC003 - used at runtime in _write_output
import unicodedata
from collections.abc import Callable, Mapping  # noqa: TC003 - used in function signatures
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    TypeAliasType,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic
from typing_extensions import is_typeddict

from pivot import exceptions, outputs

if TYPE_CHECKING:
    from pivot import loaders

logger = logging.getLogger(__name__)

# Key used in out_specs for single-output stages (non-TypedDict Annotated[T, Out(...)] returns)
SINGLE_OUTPUT_KEY = "_single"


def _get_type_hints_safe(
    obj: Callable[..., Any] | type,
    name: str,
    *,
    include_extras: bool = False,
) -> dict[str, Any] | None:
    """Get type hints from a function or type, returning None on failure.

    Args:
        obj: Function or type to get hints from
        name: Name for error messages
        include_extras: Whether to preserve Annotated metadata

    Returns:
        Dict of type hints, or None if hints couldn't be resolved
    """
    try:
        return get_type_hints(obj, include_extras=include_extras)
    except (NameError, AttributeError) as e:
        logger.warning("Failed to resolve type hints for %s: %s", name, e)
        return None
    except Exception as e:
        logger.debug("Failed to get type hints for %s: %s", name, e)
        return None


def _unwrap_type_alias(t: Any) -> Any:
    """Unwrap TypeAliasType (Python 3.12+ 'type' keyword aliases) to their value.

    Handles nested aliases like `type Outer = Inner` where `type Inner = Annotated[...]`.
    Note: Accessing __value__ is the documented approach - get_origin()/get_args()
    return None/() for TypeAliasType by design.
    """
    while isinstance(t, TypeAliasType):
        t = t.__value__
    return t


class StageParams(pydantic.BaseModel):
    """Base class for stage parameters (Pydantic model).

    Use as a simple base class for parameter-only stages:

        class TrainParams(StageParams):
            learning_rate: float = 0.01
            batch_size: int = 32

        def train(
            config: TrainParams,
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> TrainOutputs:
            ...

    For testing, just pass the data directly:

        result = train(TrainParams(learning_rate=0.5), test_df)
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict()


# ==============================================================================
# Common validation and write helpers
# ==============================================================================


def _validate_path_not_escaped(path: pathlib.Path, project_root: pathlib.Path) -> None:
    """Validate that resolved path is within project root (no path traversal)."""
    resolved = path.resolve()
    root_resolved = project_root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"Path escapes project root: '{path}' resolves to '{resolved}' which is outside '{root_resolved}'"
        )


def _validate_directory_out_key(key: str, output_name: str) -> str:
    """Validate and normalize a DirectoryOut key (relative path within directory).

    Returns the normalized key (redundant separators removed, Unicode NFC normalized).

    Raises:
        ValueError: If key is invalid (empty, whitespace-only, absolute, contains .., no extension)
    """
    if not key or not key.strip():
        raise ValueError(f"DirectoryOut '{output_name}': empty or whitespace-only key not allowed")

    # Normalize Unicode to NFC for consistent cross-platform behavior
    # (macOS HFS+ uses NFD, which could cause duplicate detection issues)
    key = unicodedata.normalize("NFC", key)

    # Normalize the key (handles "foo//bar.yaml" -> "foo/bar.yaml")
    path_obj = pathlib.PurePosixPath(key)
    normalized = path_obj.as_posix()

    # Check for absolute path
    if normalized.startswith("/"):
        raise ValueError(f"DirectoryOut '{output_name}': absolute path not allowed: {key!r}")

    # Check for path traversal (..)
    if ".." in path_obj.parts:
        raise ValueError(f"DirectoryOut '{output_name}': path traversal not allowed: {key!r}")

    # Extension required to ensure files match the loader's expected format
    # Use suffix instead of "." in name to correctly reject hidden files without extensions
    if not path_obj.suffix:
        raise ValueError(f"DirectoryOut '{output_name}': key must include file extension: {key!r}")

    # Reject filenames where the stem (name without extension) is empty or whitespace-only
    # e.g., "   .json" has stem="   " which would create problematic filenames
    if not path_obj.stem.strip():
        raise ValueError(
            f"DirectoryOut '{output_name}': filename cannot be empty or whitespace-only: {key!r}"
        )

    return normalized


def _collect_directory_out_ops(
    name: str,
    spec: outputs.DirectoryOut[Any],
    value: Any,
    project_root: pathlib.Path,
    write_ops: list[tuple[pathlib.Path, Any, loaders.Writer[Any]]],
) -> None:
    """Collect write operations for a DirectoryOut.

    Validates the value dict and each key, then appends write operations to write_ops.
    """
    # Validate value is a dict
    if not isinstance(value, dict):
        raise RuntimeError(
            f"DirectoryOut '{name}' expects dict[str, T], got {type(value).__name__}"
        )

    # Validate dict is non-empty
    if not value:
        raise ValueError(f"DirectoryOut '{name}': dict must be non-empty")

    # spec.path is guaranteed to be str ending with "/" by DirectoryOut.__post_init__
    dir_path = spec.path

    # Track normalized keys to detect duplicates after normalization
    seen_normalized = dict[str, str]()  # normalized -> original
    # Track lowercased keys to detect case collisions (for case-insensitive filesystems)
    seen_lowercase = dict[str, str]()  # lowercase -> original normalized

    # Cast value to dict[Any, Any] - we validated it's a dict above
    value_dict = cast("dict[Any, Any]", value)

    for key, item_value in value_dict.items():
        # Validate key is a string
        if not isinstance(key, str):
            raise ValueError(
                f"DirectoryOut '{name}': keys must be strings, got {type(key).__name__}"
            )

        # Validate and normalize the key
        normalized_key = _validate_directory_out_key(key, name)

        # Check for duplicates after normalization
        if normalized_key in seen_normalized:
            # Sort keys for deterministic error message
            sorted_keys = sorted([key, seen_normalized[normalized_key]])
            raise ValueError(
                f"DirectoryOut '{name}': duplicate key after normalization: "
                + f"{sorted_keys[0]!r} and {sorted_keys[1]!r} both normalize to {normalized_key!r}"
            )
        seen_normalized[normalized_key] = key

        # Check for case collisions (would conflict on case-insensitive filesystems)
        lowercase_key = normalized_key.lower()
        if lowercase_key in seen_lowercase:
            existing_key = seen_lowercase[lowercase_key]
            if existing_key != normalized_key:
                sorted_keys = sorted([normalized_key, existing_key])
                raise ValueError(
                    f"DirectoryOut '{name}': keys would collide on case-insensitive filesystems: "
                    + f"{sorted_keys[0]!r} and {sorted_keys[1]!r}"
                )
        seen_lowercase[lowercase_key] = normalized_key

        # Build full path: directory + normalized key
        full_path = project_root / dir_path / normalized_key

        # Validate path hasn't escaped project root
        _validate_path_not_escaped(full_path, project_root)

        write_ops.append((full_path, item_value, spec.loader))


# ==============================================================================
# Return output spec extraction
# ==============================================================================


def _extract_typeddict_outputs(
    return_type: type,
    stage_name: str,
) -> dict[str, outputs.BaseOut]:
    """Extract output specs from TypedDict, erroring if any field lacks Out/DirectoryOut/IncrementalOut."""
    field_hints = _get_type_hints_safe(return_type, str(return_type), include_extras=True)
    if field_hints is None:
        raise exceptions.StageDefinitionError(
            f"Stage '{stage_name}': Failed to resolve type hints for TypedDict '{return_type.__name__}'"
        )

    specs = dict[str, outputs.BaseOut]()
    fields_without_out = list[str]()

    for field_name, field_type in field_hints.items():
        field_type = _unwrap_type_alias(field_type)

        if get_origin(field_type) is not Annotated:
            fields_without_out.append(field_name)
            continue

        args = get_args(field_type)
        if len(args) < 2:
            fields_without_out.append(field_name)
            continue

        out_found = False
        for metadata in args[1:]:
            # Check for any output spec type (Out, DirectoryOut, IncrementalOut, and subclasses)
            if isinstance(metadata, (outputs.Out, outputs.DirectoryOut, outputs.IncrementalOut)):
                specs[field_name] = metadata
                out_found = True
                break

        if not out_found:
            fields_without_out.append(field_name)

    if fields_without_out:
        raise exceptions.StageDefinitionError(
            f"Stage '{stage_name}': TypedDict '{return_type.__name__}' has fields without Out annotations: "
            + f"{', '.join(sorted(fields_without_out))}. All fields must have Out annotations."
        )

    if not specs:
        raise exceptions.StageDefinitionError(
            f"Stage '{stage_name}': TypedDict '{return_type.__name__}' has no fields. "
            + "Use None return type for stages with no outputs."
        )

    return specs


def save_return_outputs(
    return_value: Mapping[str, Any],
    specs: dict[str, outputs.BaseOut],
    project_root: pathlib.Path,
) -> None:
    """Save return value outputs to disk.

    Takes the return value from a stage function and saves each output
    to its configured path using its loader.

    Validates all inputs upfront before writing any files. Path overrides are
    already applied to specs at registration time.

    Args:
        return_value: The dict returned by the stage function
        specs: Output specs with paths already resolved (from registration)
        project_root: Root directory for relative paths

    Raises:
        ValueError: If path escapes project root
        KeyError: If output keys are missing from return_value
        RuntimeError: If value/path count mismatch for sequence outputs
    """
    # Validate all output keys exist
    missing = set(specs.keys()) - set(return_value.keys())
    if missing:
        raise KeyError(
            f"Missing return output keys: {sorted(missing)}. Return value keys: {sorted(return_value.keys())}"
        )

    # Warn about extra keys not declared as outputs
    extra = set(return_value.keys()) - set(specs.keys())
    if extra:
        logger.warning("Extra keys in return value not declared as outputs: %s", sorted(extra))

    # Collect all write operations and validate paths upfront
    write_ops: list[tuple[pathlib.Path, Any, loaders.Writer[Any]]] = []
    for name, spec in specs.items():
        path = spec.path
        value = return_value[name]

        if outputs.is_directory_out(spec):
            # DirectoryOut: value is dict[str, T], keys are relative paths within directory
            _collect_directory_out_ops(name, spec, value, project_root, write_ops)
        elif isinstance(path, (list, tuple)):
            if not isinstance(value, (list, tuple)):
                raise RuntimeError(
                    f"Output '{name}' has sequence path but non-sequence value: {type(value).__name__}"
                )
            value_seq = cast("list[Any] | tuple[Any, ...]", value)
            if len(value_seq) != len(path):
                raise RuntimeError(
                    f"Output '{name}' has {len(path)} paths but {len(value_seq)} values"
                )
            for p, v in zip(path, value_seq, strict=True):
                full_path = project_root / p
                # Defense-in-depth: validate path hasn't escaped (e.g., via symlink attack
                # between registration and execution)
                _validate_path_not_escaped(full_path, project_root)
                write_ops.append((full_path, v, spec.loader))
        else:
            full_path = project_root / path
            # Defense-in-depth: validate path hasn't escaped
            _validate_path_not_escaped(full_path, project_root)
            write_ops.append((full_path, value, spec.loader))

    # All validation passed - now write
    for full_path, value, loader in write_ops:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        loader.save(value, full_path)


# ==============================================================================
# Annotation-based dependency injection helpers
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class FuncDepSpec:
    """Specification for a function argument dependency (from Annotated marker).

    Attributes:
        path: The file path(s) for this dependency.
        loader: The loader to use for loading the file(s).
        creates_dep_edge: If True (default), creates a DAG dependency edge.
            Set to False for IncrementalOut used as input (self-referential,
            no DAG edge to avoid circular dependency).
    """

    path: outputs.PathType
    loader: loaders.Reader[Any]
    creates_dep_edge: bool = True


@dataclasses.dataclass(frozen=True)
class StageDefinition:
    """Complete parsed definition of a stage function's annotations.

    Produced once by extract_stage_definition() and consumed by Pipeline/Registry.
    Avoids redundant get_type_hints() calls across registration layers.
    """

    dep_specs: dict[str, FuncDepSpec]
    out_specs: dict[str, outputs.BaseOut]
    single_out_spec: outputs.BaseOut | None
    placeholder_dep_names: frozenset[str]
    params_arg_name: str | None
    params_type: type[StageParams] | None
    hints_resolved: bool


def extract_stage_definition(
    func: Callable[..., Any],
    stage_name: str,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    *,
    strict: bool = True,
) -> StageDefinition:
    """Extract complete stage definition from function annotations in a single pass.

    Calls get_type_hints() once and derives all dep/output/params specs from the
    result. This is the single extraction point -- Pipeline and Registry should
    call this instead of individual extraction functions.

    Args:
        func: Stage function to extract from.
        stage_name: Name for error messages.
        dep_path_overrides: Override paths for PlaceholderDep and Dep annotations.
        strict: If True (default), raise StageDefinitionError when type hints
            can't be resolved. If False, return a definition with hints_resolved=False
            and empty specs.

    Returns:
        StageDefinition with all parsed annotation data.

    Raises:
        StageDefinitionError: If strict=True and type hints can't be resolved.
        ValueError: If PlaceholderDep override is provided but empty.
    """
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        if strict:
            raise exceptions.StageDefinitionError(
                f"Stage '{stage_name}': failed to resolve type hints for '{func.__name__}'. "
                + "Check that all type annotations are importable."
            )
        return StageDefinition(
            dep_specs={},
            out_specs={},
            single_out_spec=None,
            placeholder_dep_names=frozenset(),
            params_arg_name=None,
            params_type=None,
            hints_resolved=False,
        )

    sig = inspect_module.signature(func)

    # --- Extract deps, placeholders, and params from parameters ---
    overrides = dep_path_overrides or {}
    dep_specs = dict[str, FuncDepSpec]()
    placeholder_dep_names = set[str]()
    params_arg_name: str | None = None
    params_type: type[StageParams] | None = None

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = _unwrap_type_alias(hints[param_name])

        # Check for StageParams: strip Annotated wrapper to get the base type.
        # Only match the first StageParams parameter.
        if params_arg_name is None:
            base_type = (
                get_args(param_type)[0] if get_origin(param_type) is Annotated else param_type
            )
            if isinstance(base_type, type) and issubclass(base_type, StageParams):
                params_arg_name = param_name
                params_type = base_type

        # Check for Annotated deps
        if get_origin(param_type) is not Annotated:
            continue

        args = get_args(param_type)
        if len(args) < 2:
            continue

        for metadata in args[1:]:
            if isinstance(metadata, outputs.PlaceholderDep):
                placeholder_dep_names.add(param_name)
                if param_name not in overrides:
                    # Skip resolution -- caller checks placeholder_dep_names
                    break
                override_path = overrides[param_name]
                if isinstance(override_path, (list, tuple)):
                    if not override_path or any(not p for p in override_path):
                        raise ValueError(
                            f"PlaceholderDep '{param_name}' override contains empty path"
                        )
                elif not override_path:
                    raise ValueError(f"PlaceholderDep '{param_name}' override cannot be empty")
                placeholder = cast("outputs.PlaceholderDep[Any]", metadata)
                dep_specs[param_name] = FuncDepSpec(
                    path=override_path,
                    loader=placeholder.loader,
                )
                break
            elif isinstance(metadata, outputs.Dep):
                dep = cast("outputs.Dep[Any]", metadata)
                path = overrides.get(param_name, dep.path)
                dep_specs[param_name] = FuncDepSpec(path=path, loader=dep.loader)
                break
            elif isinstance(metadata, outputs.IncrementalOut):
                inc = cast("outputs.IncrementalOut[Any, Any]", metadata)
                dep_specs[param_name] = FuncDepSpec(
                    path=inc.path,
                    loader=inc.loader,
                    creates_dep_edge=False,
                )
                break

    # --- Extract output specs from return type ---
    out_specs = dict[str, outputs.BaseOut]()
    single_out_spec: outputs.BaseOut | None = None

    return_type = hints.get("return")
    if return_type is not None and return_type is not type(None):
        return_type = _unwrap_type_alias(return_type)

        if is_typeddict(return_type):
            out_specs = _extract_typeddict_outputs(return_type, stage_name)
        elif get_origin(return_type) is Annotated:
            rt_args = get_args(return_type)
            if len(rt_args) >= 2:
                for metadata in rt_args[1:]:
                    if isinstance(
                        metadata, (outputs.Out, outputs.IncrementalOut, outputs.DirectoryOut)
                    ):
                        single_out_spec = cast("outputs.BaseOut", metadata)
                        break

    return StageDefinition(
        dep_specs=dep_specs,
        out_specs=out_specs,
        single_out_spec=single_out_spec,
        placeholder_dep_names=frozenset(placeholder_dep_names),
        params_arg_name=params_arg_name,
        params_type=params_type,
        hints_resolved=True,
    )


def _load_single_dep(
    name: str,
    path: str,
    spec: FuncDepSpec,
    project_root: pathlib.Path,
) -> Any:
    """Load a single dependency file with error context.

    For deps with creates_dep_edge=False (IncrementalOut as input), returns an
    empty instance from the loader if the file doesn't exist (first run).
    """
    from pivot import loaders as loaders_module

    full_path = project_root / path
    if not spec.creates_dep_edge and not full_path.exists():
        # IncrementalOut as input: file doesn't exist yet (first run)
        # IncrementalOut.loader is always a Loader (has empty()), narrow the type
        if not isinstance(spec.loader, loaders_module.Loader):
            raise RuntimeError(
                f"Dependency '{name}' has creates_dep_edge=False but loader is not a Loader"
            )
        # Cast to Loader[Any, Any] - isinstance narrows but basedpyright keeps Unknown params
        loader = cast("loaders_module.Loader[Any, Any]", spec.loader)
        return loader.empty()
    try:
        return spec.loader.load(full_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dependency '{name}' from '{path}': {e}") from e


def load_deps_from_specs(
    specs: dict[str, FuncDepSpec],
    project_root: pathlib.Path,
    path_overrides: Mapping[str, outputs.PathType] | None = None,
) -> dict[str, Any]:
    """Load dependency files based on specs.

    For single-file deps (path is str), loads and returns the single value.
    For multi-file deps (path is list/tuple), loads each file and returns as list/tuple.

    Args:
        specs: Dep specs from extract_stage_definition()
        project_root: Root directory for relative paths
        path_overrides: Optional dict of dep name -> custom path(s)

    Returns:
        Dict of dep name -> loaded data
    """
    loaded = dict[str, Any]()

    for name, spec in specs.items():
        path = path_overrides[name] if path_overrides and name in path_overrides else spec.path
        if isinstance(path, (list, tuple)):
            items = [_load_single_dep(name, p, spec, project_root) for p in path]
            # Preserve tuple type for fixed-length deps
            loaded[name] = tuple(items) if isinstance(path, tuple) else items
        else:
            loaded[name] = _load_single_dep(name, path, spec, project_root)

    return loaded
