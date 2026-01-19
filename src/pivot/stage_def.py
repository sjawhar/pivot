from __future__ import annotations

import dataclasses
import logging
import pathlib  # noqa: TC003 - used at runtime in _write_output
from collections.abc import Callable, Mapping  # noqa: TC003 - used in function signatures
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import pydantic

from pivot import outputs

if TYPE_CHECKING:
    from pivot import loaders

logger = logging.getLogger(__name__)


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


# ==============================================================================
# Return output spec extraction
# ==============================================================================


def get_output_specs_from_return(func: Callable[..., Any]) -> dict[str, outputs.Out[Any]]:
    """Extract output specs from a function's return type annotation.

    The return type must be a TypedDict with Annotated fields containing Out markers:

        class ProcessOutputs(TypedDict):
            result: Annotated[dict[str, int], Out("output.json", JSON())]

        def process(params: ProcessParams) -> ProcessOutputs:
            return {"result": {"count": 42}}

        specs = get_output_specs_from_return(process)
        # specs["result"].path == "output.json"
        # specs["result"].loader == JSON()

    Returns empty dict if return type is None or not a TypedDict.
    Recognizes Out subclasses (Metric, Plot, IncrementalOut).

    Raises:
        TypeError: If TypedDict has fields without Out annotations
    """
    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return {}

    return_type = hints.get("return")
    if return_type is None:
        return {}

    # Must be a TypedDict
    if not is_typeddict(return_type):
        return {}

    # Get the TypedDict's field annotations
    field_hints = _get_type_hints_safe(return_type, str(return_type), include_extras=True)
    if field_hints is None:
        return {}

    specs = dict[str, outputs.Out[Any]]()
    unannotated_fields = list[str]()

    for field_name, field_type in field_hints.items():
        # Must be an Annotated type
        if get_origin(field_type) is not Annotated:
            unannotated_fields.append(field_name)
            continue

        args = get_args(field_type)
        if len(args) < 2:
            unannotated_fields.append(field_name)
            continue

        # Look for Out or its subclasses (Metric, Plot, IncrementalOut) in the metadata
        out_spec: outputs.Out[Any] | None = None
        for metadata in args[1:]:
            if isinstance(metadata, outputs.Out):
                out_spec = cast("outputs.Out[Any]", metadata)
                break

        if out_spec is None:
            unannotated_fields.append(field_name)
            continue

        specs[field_name] = out_spec

    if unannotated_fields:
        raise TypeError(
            f"TypedDict return type '{return_type.__name__}' has fields without Out annotations: {sorted(unannotated_fields)}. All fields must be Annotated with Out, Metric, Plot, or IncrementalOut."
        )

    return specs


def save_return_outputs(
    return_value: Mapping[str, Any],
    specs: dict[str, outputs.Out[Any]],
    project_root: pathlib.Path,
    path_overrides: Mapping[str, outputs.PathType] | None = None,
) -> None:
    """Save return value outputs to disk.

    Takes the return value from a stage function and saves each output
    to its configured path using its loader.

    Validates all inputs upfront before writing any files.

    Args:
        return_value: The dict returned by the stage function
        specs: Output specs extracted from the function's return annotation
        project_root: Root directory for relative paths
        path_overrides: Optional dict of output name -> custom path(s) to override defaults

    Raises:
        TypeError: If path override type mismatches
        ValueError: If path override keys are unknown, list lengths mismatch, or path escapes root
        KeyError: If output keys are missing from return_value
        RuntimeError: If value/path count mismatch for sequence outputs
    """
    # Validate path overrides
    if path_overrides:
        spec_paths = {name: spec.path for name, spec in specs.items()}
        _validate_path_overrides_common(path_overrides, spec_paths, "return output")

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
    write_ops: list[tuple[pathlib.Path, Any, loaders.Loader[Any]]] = []
    for name, spec in specs.items():
        path = path_overrides[name] if path_overrides and name in path_overrides else spec.path
        value = return_value[name]

        if isinstance(path, (list, tuple)):
            if not isinstance(value, (list, tuple)):
                raise RuntimeError(
                    f"Output '{name}' has sequence path but non-sequence value: {type(value).__name__}"
                )
            values = cast("list[Any] | tuple[Any, ...]", value)
            if len(values) != len(path):
                raise RuntimeError(
                    f"Output '{name}' has {len(path)} paths but {len(values)} values"
                )
            for p, v in zip(path, values, strict=True):
                full_path = project_root / p
                _validate_path_not_escaped(full_path, project_root)
                write_ops.append((full_path, v, spec.loader))
        else:
            full_path = project_root / path
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
    """Specification for a function argument dependency (from Annotated marker)."""

    path: outputs.PathType
    loader: loaders.Loader[Any]


def get_dep_specs_from_signature(func: Callable[..., Any]) -> dict[str, FuncDepSpec]:
    """Extract dependency specs from a function's parameter annotations.

    Looks for Annotated type hints containing Dep markers:

        def process(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
            config: Annotated[dict, Dep("config.json", JSON())],
        ) -> OutputType:
            ...

        specs = get_dep_specs_from_signature(process)
        # specs["data"].path == "input.csv"
        # specs["config"].path == "config.json"

    Returns empty dict if no Dep annotations found.
    """
    import inspect as inspect_module

    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return {}

    sig = inspect_module.signature(func)
    specs = dict[str, FuncDepSpec]()

    for param_name in sig.parameters:
        if param_name not in hints:
            continue

        param_type = hints[param_name]

        # Check if it's an Annotated type
        if get_origin(param_type) is not Annotated:
            continue

        # Get the annotation args (first is the actual type, rest are metadata)
        args = get_args(param_type)
        if len(args) < 2:
            continue

        # Look for Dep in the metadata
        for metadata in args[1:]:
            if isinstance(metadata, outputs.Dep):
                dep = cast("outputs.Dep[Any]", metadata)
                specs[param_name] = FuncDepSpec(path=dep.path, loader=dep.loader)
                break

    return specs


def get_single_output_spec_from_return(func: Callable[..., Any]) -> outputs.Out[Any] | None:
    """Extract single output spec from a function's return annotation (non-TypedDict).

    For functions with a single output, the return type can be directly annotated:

        def transform(
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> Annotated[DataFrame, Out("output.csv", CSV())]:
            return data.dropna()

        spec = get_single_output_spec_from_return(transform)
        # spec.path == "output.csv"

    Returns None if return type is TypedDict (use get_output_specs_from_return instead)
    or if no Out annotation found.
    """
    hints = _get_type_hints_safe(func, func.__name__, include_extras=True)
    if hints is None:
        return None

    return_type = hints.get("return")
    if return_type is None:
        return None

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

    # Look for Out or its subclasses in the metadata
    for metadata in args[1:]:
        if isinstance(metadata, outputs.Out):
            return cast("outputs.Out[Any]", metadata)

    return None


def _load_single_dep(
    name: str,
    path: str,
    spec: FuncDepSpec,
    project_root: pathlib.Path,
) -> Any:
    """Load a single dependency file with error context."""
    try:
        return spec.loader.load(project_root / path)
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
            result[name] = FuncDepSpec(path=overrides[name], loader=spec.loader)
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

    Args:
        func: Function to inspect

    Returns:
        Tuple of (argument_name, StageParams_type), or (None, None) if not found
    """
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
