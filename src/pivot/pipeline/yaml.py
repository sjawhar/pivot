from __future__ import annotations

import importlib
import itertools
import re
import typing
from collections.abc import Callable  # noqa: TC003 Pydantic needs at runtime
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, TypeVar

import pydantic
import yaml

from pivot import fingerprint, outputs, parameters, path_policy, registry, stage_def, yaml_config

if TYPE_CHECKING:
    from pathlib import Path


class PipelineConfigError(Exception):
    """Error loading or processing pivot.yaml configuration."""


class NamedOutputOptions(TypedDict, total=False):
    """Options for named output specifications (includes path)."""

    path: str | list[str]
    cache: bool
    x: str  # For plots
    y: str  # For plots
    template: str  # For plots


# Named output value: "path", ["path1", "path2"], or {path: "path", cache: false}
NamedOutputValue = str | list[str] | NamedOutputOptions

# Named deps value: "path" or ["path1", "path2"]
NamedDepValue = str | list[str]

# Deps/outs are always named dicts (path overrides)
DepsSpec = dict[str, NamedDepValue]
OutputsSpec = dict[str, NamedOutputValue]


class VariantDict(TypedDict, total=False):
    """Variant dict structure from Python escape hatch functions."""

    name: str
    deps: dict[str, str | list[str]]
    outs: dict[str, NamedOutputValue]
    params: dict[str, Any]
    mutex: list[str]


class DimensionOverrides(pydantic.BaseModel):
    """Overrides that can be applied per matrix dimension value."""

    model_config = pydantic.ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    deps: DepsSpec | None = None
    outs: OutputsSpec | None = None
    metrics: OutputsSpec | None = None
    plots: OutputsSpec | None = None
    params: dict[str, Any] | None = None  # JSON-compatible values from YAML
    mutex: list[str] | None = None


# Primitive types allowed in matrix dimension lists
MatrixPrimitive = str | int | float | bool

# Matrix dimension can be a list ["a", "b", true, 0.5] or dict {"a": {overrides}, "b": {overrides}}
MatrixDimension = list[MatrixPrimitive] | dict[str, DimensionOverrides | None]


class StageConfig(pydantic.BaseModel):
    """Configuration for a single stage in pivot.yaml.

    deps and outs are path overrides for annotation-defined deps/outs.
    metrics and plots are extra outputs not declared in annotations.
    """

    model_config = pydantic.ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    python: str
    deps: DepsSpec = {}
    outs: OutputsSpec = {}
    metrics: OutputsSpec = {}
    plots: OutputsSpec = {}
    params: dict[str, Any] = {}
    mutex: list[str] = []
    matrix: dict[str, MatrixDimension] | None = None
    variants: str | None = None


class PipelineConfig(pydantic.BaseModel):
    """Top-level pivot.yaml configuration."""

    model_config = pydantic.ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    stages: dict[str, StageConfig]
    vars: list[str] = []  # Files to load variables from for ${var} interpolation


def _validate_callable(v: object) -> Callable[..., Any]:
    """Validate that value is callable."""
    if not callable(v):
        raise ValueError("must be callable")
    return v


class ExpandedStage(pydantic.BaseModel):
    """A stage after matrix expansion."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    name: str
    func: Annotated[Callable[..., Any], pydantic.PlainValidator(_validate_callable)]
    dep_path_overrides: dict[str, str | list[str]]
    out_path_overrides: dict[str, registry.OutOverride]
    params: pydantic.BaseModel | None
    mutex: list[str]
    variant: str | None


def load_pipeline_file(pipeline_file: Path) -> PipelineConfig:
    """Load and parse pivot.yaml pipeline file."""
    if not pipeline_file.exists():
        raise PipelineConfigError(f"Pipeline file not found: {pipeline_file}")

    with open(pipeline_file) as f:
        data = yaml.load(f, Loader=yaml_config.Loader)

    if data is None:
        raise PipelineConfigError(f"Pipeline file is empty: {pipeline_file}")

    try:
        return PipelineConfig.model_validate(data)
    except pydantic.ValidationError as e:
        raise PipelineConfigError(f"Invalid pipeline configuration: {e}") from e


def _load_vars_files(vars_paths: list[str], pipeline_dir: Path) -> dict[str, str]:
    """Load variables from YAML files for ${var} interpolation.

    Only top-level string values are included (nested dicts are skipped).
    """
    from pivot import exceptions

    result = dict[str, str]()
    for var_path in vars_paths:
        try:
            full_path = path_policy.require_valid_path(
                var_path,
                path_policy.PathType.VAR,
                pipeline_dir,
                context="vars",
            )
        except exceptions.SecurityValidationError as e:
            raise PipelineConfigError(str(e)) from e
        if not full_path.exists():
            raise PipelineConfigError(f"Vars file not found: {full_path}")
        with open(full_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise PipelineConfigError(f"Vars file must be a dict: {full_path}")
        data_dict = typing.cast("dict[str, Any]", data)
        for key, value in data_dict.items():
            if isinstance(value, (str, int, float, bool)):
                result[key] = str(value)
    return result


def register_from_pipeline_file(pipeline_file: Path) -> None:
    """Load pivot.yaml and register all stages to the global registry."""
    pipeline = load_pipeline_file(pipeline_file)
    pipeline_dir = pipeline_file.parent

    # Load variables from vars files for ${var} interpolation
    global_vars = _load_vars_files(pipeline.vars, pipeline_dir)

    for stage_name, stage_config in pipeline.stages.items():
        _register_stage(stage_name, stage_config, global_vars)

    # Flush pending AST hash writes to persistent cache
    fingerprint.flush_ast_hash_cache()


def _register_stage(
    name: str,
    config: StageConfig,
    global_vars: dict[str, str],
) -> None:
    """Register a single stage from configuration."""
    if config.matrix is not None:
        expanded = _expand_matrix(name, config, global_vars)
        for stage in expanded:
            registry.REGISTRY.register(
                func=stage.func,
                name=stage.name,
                dep_path_overrides=stage.dep_path_overrides or None,
                out_path_overrides=stage.out_path_overrides or None,
                params=stage.params,
                mutex=stage.mutex,
                variant=stage.variant,
            )
    elif config.variants is not None:
        variants_func = _import_function(config.variants)
        variants = variants_func()
        if not isinstance(variants, (list, tuple)):
            raise PipelineConfigError(
                f"Stage '{name}': variants function '{config.variants}' must return a list, "
                + f"got {type(variants).__name__}"
            )
        func = _import_function(config.python)
        for variant in typing.cast("list[VariantDict]", variants):
            _register_variant_from_dict(name, func, variant)
    else:
        _register_simple_stage(name, config, global_vars)


def _validate_output_type(
    out_name: str,
    return_out_specs: dict[str, outputs.Out[Any]],
    expected_type: type[outputs.Out[Any]],
    section_name: str,
    stage_name: str,
) -> None:
    """Validate that a YAML output key matches the expected annotation type.

    Args:
        out_name: The output name from YAML (e.g., "metrics" in metrics: {metrics: path})
        return_out_specs: Output specs from function return type annotation
        expected_type: Expected Out subclass (e.g., Metric or Plot)
        section_name: YAML section name for error messages (e.g., "metrics" or "plots")
        stage_name: Stage name for error messages

    Raises:
        PipelineConfigError: If output name not found or wrong type
    """
    if out_name not in return_out_specs:
        available = list(return_out_specs.keys()) if return_out_specs else []
        raise PipelineConfigError(
            f"Stage '{stage_name}': {section_name} key '{out_name}' not found in function return type annotation. Available outputs: {available}"
        )

    out_spec = return_out_specs[out_name]
    if not isinstance(out_spec, expected_type):
        actual_type = type(out_spec).__name__
        raise PipelineConfigError(
            f"Stage '{stage_name}': {section_name} key '{out_name}' must be annotated with {expected_type.__name__}, but found {actual_type}"
        )


def _process_typed_output_section(
    section: OutputsSpec,
    out_path_overrides: dict[str, registry.OutOverride],
    return_out_specs: dict[str, outputs.Out[Any]],
    expected_type: type[outputs.Out[Any]],
    section_name: str,
    stage_name: str,
    global_vars: dict[str, str],
) -> None:
    """Process metrics or plots section, validating types and adding to out_path_overrides."""
    for out_name, value in section.items():
        _validate_output_type(out_name, return_out_specs, expected_type, section_name, stage_name)
        if out_name in out_path_overrides:
            raise PipelineConfigError(
                f"Stage '{stage_name}': output '{out_name}' specified in both 'outs' and '{section_name}' sections"
            )
        out_path_overrides.update(
            _normalize_out_path_overrides({out_name: value}, global_vars, stage_name)
        )


def _register_simple_stage(
    name: str,
    config: StageConfig,
    global_vars: dict[str, str],
) -> None:
    """Register a simple (non-matrix) stage."""
    func = _import_function(config.python)

    # Interpolate deps path overrides
    dep_path_overrides = _normalize_deps_spec(config.deps, global_vars, name)

    # Interpolate outs path overrides
    out_path_overrides = _normalize_out_path_overrides(config.outs, global_vars, name)

    # Get return output specs for type validation
    return_out_specs = stage_def.get_output_specs_from_return(func, name)

    # Process metrics and plots sections
    _process_typed_output_section(
        config.metrics,
        out_path_overrides,
        return_out_specs,
        outputs.Metric,
        "metrics",
        name,
        global_vars,
    )
    _process_typed_output_section(
        config.plots, out_path_overrides, return_out_specs, outputs.Plot, "plots", name, global_vars
    )

    params_instance = _resolve_params(func, config.params, name)

    registry.REGISTRY.register(
        func=func,
        name=name,
        dep_path_overrides=dep_path_overrides or None,
        out_path_overrides=out_path_overrides or None,
        params=params_instance,
        mutex=config.mutex,
    )


_VARIANT_DICT_KEYS = frozenset({"name", "deps", "outs", "params", "mutex"})


def _register_variant_from_dict(
    base_name: str,
    func: Callable[..., Any],
    variant: VariantDict,
) -> None:
    """Register a variant from Python escape hatch."""
    # Validate no unknown keys (catch typos like "parmas" instead of "params")
    unknown_keys = set(variant.keys()) - _VARIANT_DICT_KEYS
    if unknown_keys:
        raise PipelineConfigError(
            f"Stage '{base_name}': variant dict has unknown keys: {sorted(unknown_keys)}. "
            + f"Valid keys are: {sorted(_VARIANT_DICT_KEYS)}"
        )

    variant_name = variant.get("name", "default")
    full_name = f"{base_name}@{variant_name}"

    deps: DepsSpec = variant.get("deps") or {}
    outs_raw: OutputsSpec = variant.get("outs") or {}
    params_dict = variant.get("params", {})
    mutex = variant.get("mutex", [])

    # YAML deps are path overrides only
    dep_path_overrides: dict[str, str | list[str]] = {}
    for dep_name, dep_value in deps.items():
        dep_path_overrides[dep_name] = dep_value

    # YAML outs are path overrides only
    out_path_overrides = _normalize_out_path_overrides(outs_raw, {}, full_name)

    params_instance = _resolve_params(func, params_dict, full_name)

    registry.REGISTRY.register(
        func=func,
        name=full_name,
        dep_path_overrides=dep_path_overrides or None,
        out_path_overrides=out_path_overrides or None,
        params=params_instance,
        mutex=mutex,
        variant=variant_name,
    )


def _expand_matrix(
    name: str,
    config: StageConfig,
    global_vars: dict[str, str],
) -> list[ExpandedStage]:
    """Expand matrix configuration into individual variant stages."""
    if config.matrix is None:
        raise PipelineConfigError(f"Stage '{name}' missing 'matrix' field")

    func = _import_function(config.python)
    matrix = config.matrix

    base_name, name_template = _parse_stage_name(name, matrix)
    normalized_dims = _normalize_matrix_dimensions(name, matrix)
    dim_names = list(normalized_dims.keys())

    dim_keys = [list(normalized_dims[dim].keys()) for dim in dim_names]
    combinations = list(itertools.product(*dim_keys))

    expanded = list[ExpandedStage]()
    for combo in combinations:
        # combo contains string keys; build both string and typed value dicts
        string_values = dict(zip(dim_names, combo, strict=True))
        typed_values = {
            dim_name: normalized_dims[dim_name][key][0] for dim_name, key in string_values.items()
        }
        variant_name = _generate_variant_name(name_template, dim_names, string_values)
        full_name = f"{base_name}@{variant_name}"

        # Copy deps/outs dicts for this variant
        deps: DepsSpec = dict(config.deps)
        outs_raw: OutputsSpec = dict(config.outs)
        metrics_raw: OutputsSpec = dict(config.metrics)
        plots_raw: OutputsSpec = dict(config.plots)
        params_dict = dict(config.params)
        mutex = list(config.mutex)

        for dim_name, key in string_values.items():
            overrides = normalized_dims[dim_name][key][1]
            deps, outs_raw, metrics_raw, plots_raw, params_dict, mutex = _apply_overrides(
                deps, outs_raw, metrics_raw, plots_raw, params_dict, mutex, overrides
            )

        # Merge global vars with matrix values (matrix takes precedence)
        all_vars = {**global_vars, **string_values}

        # Interpolate path overrides
        dep_path_overrides = _normalize_deps_spec(deps, all_vars, full_name)
        out_path_overrides = _normalize_out_path_overrides(outs_raw, all_vars, full_name)

        # Get return output specs for type validation
        return_out_specs = stage_def.get_output_specs_from_return(func, full_name)

        # Process metrics and plots sections
        _process_typed_output_section(
            metrics_raw,
            out_path_overrides,
            return_out_specs,
            outputs.Metric,
            "metrics",
            full_name,
            all_vars,
        )
        _process_typed_output_section(
            plots_raw,
            out_path_overrides,
            return_out_specs,
            outputs.Plot,
            "plots",
            full_name,
            all_vars,
        )

        # Use typed values for params interpolation (preserves int/float/bool)
        params_dict = {
            k: _interpolate_value(v, typed_values, full_name) for k, v in params_dict.items()
        }
        params_instance = _resolve_params(func, params_dict, full_name)

        expanded.append(
            ExpandedStage(
                name=full_name,
                func=func,
                dep_path_overrides=dep_path_overrides,
                out_path_overrides=out_path_overrides,
                params=params_instance,
                mutex=mutex,
                variant=variant_name,
            )
        )

    return expanded


def _parse_stage_name(name: str, matrix: dict[str, MatrixDimension]) -> tuple[str, str | None]:
    """Parse stage name for template pattern."""
    if "@" not in name:
        return name, None

    if "@{" not in name:
        raise PipelineConfigError(
            f"Stage name '{name}' contains '@' but no template variables. "
            + "Use '@{dim}' syntax or remove '@' for auto-naming."
        )

    base_name, template = name.split("@", 1)
    dim_names = set(matrix.keys())
    template_vars = set(re.findall(r"\{(\w+)\}", template))

    missing = dim_names - template_vars
    if missing:
        raise PipelineConfigError(
            f"Stage '{name}' template missing dimensions: {missing}. "
            + "All matrix dimensions must appear in the name template."
        )

    extra = template_vars - dim_names
    if extra:
        raise PipelineConfigError(
            f"Stage '{name}' template has unknown variables: {extra}. "
            + f"Available dimensions: {dim_names}"
        )

    return base_name, template


def _normalize_matrix_dimensions(
    stage_name: str,
    matrix: dict[str, MatrixDimension],
) -> dict[str, dict[str, tuple[MatrixPrimitive, DimensionOverrides]]]:
    """Normalize matrix dimensions to {dim: {str_key: (typed_value, overrides)}} form."""
    normalized = dict[str, dict[str, tuple[MatrixPrimitive, DimensionOverrides]]]()

    for dim_name, dim_value in matrix.items():
        if isinstance(dim_value, list):
            if not dim_value:
                raise PipelineConfigError(
                    f"Stage '{stage_name}': matrix dimension '{dim_name}' is empty"
                )
            normalized[dim_name] = {str(v): (v, DimensionOverrides()) for v in dim_value}
        else:
            if not dim_value:
                raise PipelineConfigError(
                    f"Stage '{stage_name}': matrix dimension '{dim_name}' is empty"
                )
            normalized[dim_name] = {
                k: (k, v if v is not None else DimensionOverrides()) for k, v in dim_value.items()
            }

    return normalized


def _generate_variant_name(
    template: str | None, dim_names: list[str], dim_values: dict[str, str]
) -> str:
    """Generate variant name from template or auto-generate."""
    if template is not None:
        return template.format(**dim_values)
    return "_".join(dim_values[d] for d in dim_names)


_V = TypeVar("_V")


def _merge_named_dict(base: dict[str, _V], override: dict[str, _V] | None) -> dict[str, _V]:
    """Merge named dicts with per-key replacement."""
    if override is None:
        return base
    return {**base, **override}


def _apply_overrides(
    deps: DepsSpec,
    outs: OutputsSpec,
    metrics: OutputsSpec,
    plots: OutputsSpec,
    params: dict[str, Any],
    mutex: list[str],
    overrides: DimensionOverrides,
) -> tuple[
    DepsSpec,
    OutputsSpec,
    OutputsSpec,
    OutputsSpec,
    dict[str, Any],
    list[str],
]:
    """Apply dimension overrides to stage config."""
    deps = _merge_named_dict(deps, overrides.deps)
    outs = _merge_named_dict(outs, overrides.outs)
    metrics = _merge_named_dict(metrics, overrides.metrics)
    plots = _merge_named_dict(plots, overrides.plots)

    if overrides.params is not None:
        params = {**params, **overrides.params}
    if overrides.mutex is not None:
        mutex = overrides.mutex

    return deps, outs, metrics, plots, params, mutex


def _interpolate(s: str, values: dict[str, str], stage_name: str) -> str:
    """Interpolate ${dim} variables in a string."""
    result = s
    for key, val in values.items():
        result = result.replace(f"${{{key}}}", val)

    remaining = re.findall(r"\$\{(\w+)\}", result)
    if remaining:
        raise PipelineConfigError(
            f"Stage '{stage_name}': unresolved variable(s) in '{s}': {remaining}"
        )
    return result


def _interpolate_value(
    value: Any, values: dict[str, MatrixPrimitive], stage_name: str | None = None
) -> Any:
    """Interpolate ${dim} in any value, including nested structures.

    When the value is exactly "${key}", returns the typed value (preserves int/float/bool).
    When the value contains "${key}" as substring, uses string replacement.

    Args:
        value: Value to interpolate (can be string, dict, list, or primitive).
        values: Variable name -> value mapping for substitution.
        stage_name: Stage name for error messages (if None, unresolved vars won't raise).
    """
    if isinstance(value, str):
        # Check for exact match first (preserves original type)
        for key, val in values.items():
            if value == f"${{{key}}}":
                return val
        # Otherwise do string replacement
        result = value
        for key, val in values.items():
            result = result.replace(f"${{{key}}}", str(val))
        # Check for unresolved variables
        if stage_name is not None:
            remaining = re.findall(r"\$\{(\w+)\}", result)
            if remaining:
                raise PipelineConfigError(
                    f"Stage '{stage_name}': unresolved variable(s) in '{value}': {remaining}"
                )
        return result
    if isinstance(value, dict):
        return {
            k: _interpolate_value(v, values, stage_name)
            for k, v in typing.cast("dict[str, Any]", value).items()
        }
    if isinstance(value, list):
        return [_interpolate_value(v, values, stage_name) for v in typing.cast("list[Any]", value)]
    return value


def _normalize_deps_spec(
    deps: DepsSpec, values: dict[str, str], stage_name: str
) -> dict[str, str | list[str]]:
    """Interpolate deps spec values."""
    result = dict[str, str | list[str]]()
    for name, paths in deps.items():
        if isinstance(paths, list):
            result[name] = [_interpolate(p, values, stage_name) for p in paths]
        else:
            result[name] = _interpolate(paths, values, stage_name)
    return result


def _normalize_out_path_overrides(
    outs: OutputsSpec, values: dict[str, str], stage_name: str
) -> dict[str, registry.OutOverride]:
    """Extract path and option overrides from outputs spec."""
    result = dict[str, registry.OutOverride]()
    for name, value in outs.items():
        if isinstance(value, str):
            result[name] = registry.OutOverride(path=_interpolate(value, values, stage_name))
        elif isinstance(value, list):
            result[name] = registry.OutOverride(
                path=[_interpolate(p, values, stage_name) for p in value]
            )
        else:
            # Dict with options - extract path and options
            path = value.get("path")
            if path is None:
                raise PipelineConfigError(
                    f"Stage '{stage_name}': named output '{name}' missing 'path' field"
                )
            # Handle both string and list paths with interpolation
            if isinstance(path, list):
                interpolated_path: str | list[str] = [
                    _interpolate(p, values, stage_name) for p in path
                ]
            else:
                interpolated_path = _interpolate(path, values, stage_name)
            override = registry.OutOverride(path=interpolated_path)
            if "cache" in value:
                override["cache"] = value["cache"]
            result[name] = override
    return result


def _import_function(import_path: str) -> Callable[..., Any]:
    """Import a function from module.function path."""
    if "." not in import_path:
        raise PipelineConfigError(
            f"Invalid import path '{import_path}': expected 'module.function' format"
        )

    module_path, func_name = import_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise PipelineConfigError(f"Failed to import module '{module_path}': {e}") from e

    if not hasattr(module, func_name):
        raise PipelineConfigError(f"Module '{module_path}' has no function '{func_name}'")

    func = getattr(module, func_name)
    if not callable(func):
        raise PipelineConfigError(f"'{import_path}' is not callable")

    return func


def _resolve_params(
    func: Callable[..., Any],
    overrides: dict[str, Any],
    stage_name: str,
) -> pydantic.BaseModel | None:
    """Resolve params from function signature + config overrides.

    Uses type-based detection to find any StageParams/BaseModel parameter,
    regardless of parameter name.
    """
    params_arg_name, params_type = stage_def.find_params_in_signature(func)

    if params_arg_name is None:
        if overrides:
            raise PipelineConfigError(
                f"Stage '{stage_name}': pivot.yaml has 'params' but function "
                + f"'{func.__name__}' has no StageParams parameter"
            )
        return None

    if params_type is None:
        raise PipelineConfigError(
            f"Stage '{stage_name}': function '{func.__name__}' has '{params_arg_name}' parameter "
            + "but no type hint. Add a type hint like 'config: MyParams'"
        )

    if not parameters.validate_params_cls(params_type):
        raise PipelineConfigError(
            f"Stage '{stage_name}': params type hint must be a Pydantic BaseModel, "
            + f"got {params_type}"
        )

    try:
        return params_type(**overrides)
    except pydantic.ValidationError as e:
        raise PipelineConfigError(f"Stage '{stage_name}': invalid params: {e}") from e
