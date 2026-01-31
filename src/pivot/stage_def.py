from __future__ import annotations

import contextlib
import dataclasses
import logging
import pathlib  # noqa: TC003 - used at runtime in _write_output
import unicodedata
import weakref
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

# Cache for find_params_in_signature results. Uses WeakKeyDictionary to avoid
# retaining old function objects during watch mode module reloads.
_params_signature_cache: weakref.WeakKeyDictionary[
    Callable[..., Any], tuple[str | None, type[StageParams] | None]
] = weakref.WeakKeyDictionary()


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


def _validate_path_overrides_common(
    overrides: Mapping[str, outputs.PathType],
    spec_paths: Mapping[str, outputs.PathType],
    kind: str,
) -> None:
    """Validate path overrides against spec paths.

    Args:
        overrides: Dict of name -> path overrides
        spec_paths: Dict of name -> spec path (from specs)
        kind: "dependency", "output", or "return output" for error messages

    Raises:
        ValueError: If override keys unknown
        TypeError: If override type (str vs sequence) doesn't match spec type
        ValueError: If tuple spec overridden with different length

    Note:
        - List specs allow variable-length overrides (can change count)
        - Tuple specs require exact length match (fixed-size)
    """
    unknown = set(overrides.keys()) - set(spec_paths.keys())
    if unknown:
        raise ValueError(f"Unknown {kind} names in path_overrides: {unknown}")

    for name, override_path in overrides.items():
        spec_path = spec_paths[name]

        # Check type compatibility: str vs sequence (list/tuple)
        spec_is_seq = isinstance(spec_path, (list, tuple))
        override_is_seq = isinstance(override_path, (list, tuple))

        if spec_is_seq != override_is_seq:
            spec_type = "sequence" if spec_is_seq else "str"
            override_type = "sequence" if override_is_seq else "str"
            raise TypeError(
                f"Path type mismatch for {kind} '{name}': spec is {spec_type}, override is {override_type}"
            )

        # For tuple specs (fixed-length), validate exact length match
        # (spec_is_seq check above guarantees override_path is also a sequence here)
        if isinstance(spec_path, tuple) and len(override_path) != len(spec_path):
            raise ValueError(
                f"Path count mismatch for {kind} '{name}': tuple spec has {len(spec_path)} paths, override has {len(override_path)} (tuple indicates fixed-length, use list for variable-length)"
            )


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
    dir_path = cast("str", spec.path)

    # Track normalized keys to detect duplicates after normalization
    seen_normalized: dict[str, str] = {}  # normalized -> original
    # Track lowercased keys to detect case collisions (for case-insensitive filesystems)
    seen_lowercase: dict[str, str] = {}  # lowercase -> original normalized

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


def get_output_specs_from_return(
    func: Callable[..., Any],
    stage_name: str,
) -> dict[str, outputs.BaseOut]:
    """Extract output specs from a function's return type annotation (TypedDict only).

    For TypedDict returns, extracts Out specs from field annotations.
    For other return types (None, Annotated[T, Out(...)], or plain types), returns {}.

    Single-output stages (Annotated[T, Out(...)]) are handled by get_single_output_spec_from_return.
    Stages without tracked outputs can have any return type.

    Example:
        class ProcessOutputs(TypedDict):
            result: Annotated[dict[str, int], Out("output.json", JSON())]

        def process(params: ProcessParams) -> ProcessOutputs:
            return {"result": {"count": 42}}

        specs = get_output_specs_from_return(process, "process")
        # specs["result"].path == "output.json"
        # specs["result"].loader == JSON()

    Returns empty dict if return type is not a TypedDict with Out annotations.
    Recognizes Out subclasses (Metric, Plot, IncrementalOut).

    Raises:
        StageDefinitionError: If TypedDict has fields without Out annotations
    """
    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return {}

    return_type = hints.get("return")

    # No return annotation or explicit None return
    if return_type is None or return_type is type(None):
        return {}

    # Unwrap type aliases (Python 3.12+ 'type X = ...' syntax)
    return_type = _unwrap_type_alias(return_type)

    # TypedDict with Out annotations - extract specs
    if is_typeddict(return_type):
        return _extract_typeddict_outputs(return_type, stage_name)

    # Everything else (single Annotated[T, Out(...)], plain types, etc.)
    # Returns empty dict - single outputs handled by get_single_output_spec_from_return,
    # and stages without tracked outputs are allowed
    return {}


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


def get_dep_specs_from_signature(
    func: Callable[..., Any],
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
) -> dict[str, FuncDepSpec]:
    """Extract dependency specs from a function's parameter annotations.

    Looks for Annotated type hints containing Dep, PlaceholderDep, or IncrementalOut markers:

        def process(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
            config: Annotated[dict, Dep("config.json", JSON())],
        ) -> OutputType:
            ...

        specs = get_dep_specs_from_signature(process)
        # specs["data"].path == "input.csv"
        # specs["config"].path == "config.json"

    PlaceholderDep requires an override path:

        def compare(
            baseline: Annotated[DataFrame, PlaceholderDep(CSV())],
        ) -> OutputType:
            ...

        specs = get_dep_specs_from_signature(compare, {"baseline": "model/out.csv"})
        # specs["baseline"].path == "model/out.csv"

    IncrementalOut as input creates a FuncDepSpec with creates_dep_edge=False:

        MyCache = Annotated[dict | None, IncrementalOut("cache.json", JSON())]

        def my_stage(existing: MyCache) -> MyCache:
            ...

        specs = get_dep_specs_from_signature(my_stage)
        # specs["existing"].creates_dep_edge == False

    Returns empty dict if no Dep/PlaceholderDep/IncrementalOut annotations found.
    """
    import inspect as inspect_module

    overrides = dep_path_overrides or {}
    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return {}

    sig = inspect_module.signature(func)
    specs = dict[str, FuncDepSpec]()

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = _unwrap_type_alias(hints[param_name])

        # Check if it's an Annotated type
        if get_origin(param_type) is not Annotated:
            continue

        # Get the annotation args (first is the actual type, rest are metadata)
        args = get_args(param_type)
        if len(args) < 2:
            continue

        # Look for PlaceholderDep, Dep, or IncrementalOut in the metadata
        for metadata in args[1:]:
            if isinstance(metadata, outputs.PlaceholderDep):
                # PlaceholderDep requires override
                if param_name not in overrides:
                    raise ValueError(
                        f"PlaceholderDep '{param_name}' requires override in dep_path_overrides"
                    )
                override_path = overrides[param_name]
                # Validate non-empty path
                if isinstance(override_path, (list, tuple)):
                    if not override_path or any(not p for p in override_path):
                        raise ValueError(
                            f"PlaceholderDep '{param_name}' override contains empty path"
                        )
                elif not override_path:
                    raise ValueError(f"PlaceholderDep '{param_name}' override cannot be empty")
                placeholder = cast("outputs.PlaceholderDep[Any]", metadata)
                specs[param_name] = FuncDepSpec(
                    path=override_path,
                    loader=placeholder.loader,
                )
                break
            elif isinstance(metadata, outputs.Dep):
                # Cast to Dep[Any] - isinstance narrows to Dep[Unknown]
                dep = cast("outputs.Dep[Any]", metadata)
                # Use override if provided, otherwise annotation path
                path = overrides.get(param_name, dep.path)
                specs[param_name] = FuncDepSpec(path=path, loader=dep.loader)
                break
            elif isinstance(metadata, outputs.IncrementalOut):
                # IncrementalOut as input: loads file if exists, returns None if not
                # Does NOT create DAG edge (self-referential, avoids circular dependency)
                inc = cast("outputs.IncrementalOut[Any]", metadata)
                specs[param_name] = FuncDepSpec(
                    path=inc.path,
                    loader=inc.loader,
                    creates_dep_edge=False,
                )
                break

    return specs


def get_placeholder_dep_names(func: Callable[..., Any]) -> set[str]:
    """Get parameter names that use PlaceholderDep annotations.

    Scans function parameters for Annotated hints containing PlaceholderDep.
    Used to validate that all placeholders have overrides before registration.

    Returns:
        Set of parameter names that have PlaceholderDep annotations.
    """
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return set()

    sig = inspect_module.signature(func)
    placeholder_names = set[str]()

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = _unwrap_type_alias(hints[param_name])

        if get_origin(param_type) is not Annotated:
            continue

        args = get_args(param_type)
        if len(args) < 2:
            continue

        for metadata in args[1:]:
            if isinstance(metadata, outputs.PlaceholderDep):
                placeholder_names.add(param_name)
                break

    return placeholder_names


def get_single_output_spec_from_return(func: Callable[..., Any]) -> outputs.BaseOut | None:
    """Extract single output spec from a function's return annotation (non-TypedDict).

    For functions with a single output, the return type can be directly annotated:

        def transform(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> Annotated[DataFrame, Out("output.csv", CSV())]:
            return data.dropna()

        spec = get_single_output_spec_from_return(transform)
        # spec.path == "output.csv"

    Returns None if return type is TypedDict (use get_output_specs_from_return instead)
    or if no output annotation found (Out, IncrementalOut, or DirectoryOut).
    """
    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return None

    return_type = hints.get("return")
    if return_type is None:
        return None

    return_type = _unwrap_type_alias(return_type)

    # If it's a TypedDict, return None (use get_output_specs_from_return instead)
    if is_typeddict(return_type):
        return None

    # Check if it's an Annotated type
    if get_origin(return_type) is not Annotated:
        return None

    # Get the annotation args (first is the actual type, rest are metadata)
    args = get_args(return_type)
    if len(args) < 2:
        return None

    # Look for any output spec type in the metadata
    for metadata in args[1:]:
        if isinstance(metadata, (outputs.Out, outputs.IncrementalOut, outputs.DirectoryOut)):
            return metadata

    return None


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
        return spec.loader.empty()
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
        specs: Dep specs from get_dep_specs_from_signature()
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


def apply_dep_path_overrides(
    specs: dict[str, FuncDepSpec],
    overrides: Mapping[str, outputs.PathType],
) -> dict[str, FuncDepSpec]:
    """Apply path overrides to dep specs, returning new specs.

    Args:
        specs: Original dep specs
        overrides: Dict of dep name -> new path(s)

    Returns:
        New dict with overridden paths (original specs unchanged)

    Raises:
        ValueError: If override keys don't match spec keys or tuple lengths mismatch
        TypeError: If override type (str vs sequence) doesn't match spec type
    """
    # Validate overrides (unknown keys, type compatibility, tuple lengths)
    spec_paths = {name: spec.path for name, spec in specs.items()}
    _validate_path_overrides_common(overrides, spec_paths, "dependency")

    result = dict[str, FuncDepSpec]()
    for name, spec in specs.items():
        if name in overrides:
            result[name] = FuncDepSpec(
                path=overrides[name],
                loader=spec.loader,
                creates_dep_edge=spec.creates_dep_edge,
            )
        else:
            result[name] = spec

    return result


def find_params_type_in_signature(func: Callable[..., Any]) -> type[StageParams] | None:
    """Find StageParams subclass type in function signature.

    Scans function parameters for a type hint that's a StageParams subclass.

    Args:
        func: Function to inspect

    Returns:
        The StageParams subclass type, or None if not found
    """
    _, params_type = find_params_in_signature(func)
    return params_type


def find_params_in_signature(
    func: Callable[..., Any],
) -> tuple[str | None, type[StageParams] | None]:
    """Find StageParams argument name and type in function signature.

    Scans function parameters for a type hint that's a StageParams subclass.
    Results are cached using WeakKeyDictionary to avoid retaining old function
    objects during watch mode module reloads.
    """
    # Check cache first (WeakKeyDictionary raises TypeError for non-weakly-referenceable)
    with contextlib.suppress(TypeError):
        cached = _params_signature_cache.get(func)
        if cached is not None:
            return cached

    result = _find_params_in_signature_impl(func)

    with contextlib.suppress(TypeError):
        _params_signature_cache[func] = result

    return result


def _find_params_in_signature_impl(
    func: Callable[..., Any],
) -> tuple[str | None, type[StageParams] | None]:
    """Internal implementation of find_params_in_signature."""
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__)
    if hints is None:
        return None, None

    sig = inspect_module.signature(func)

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = hints[param_name]

        # Check if it's a StageParams subclass
        if isinstance(param_type, type) and issubclass(param_type, StageParams):
            return param_name, param_type

    return None, None
