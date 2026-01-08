from __future__ import annotations

import importlib
import inspect
import itertools
import re
import typing
from collections.abc import Callable  # noqa: TC003 Pydantic needs at runtime
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

import pydantic
import yaml

from pivot import outputs, parameters, registry, yaml_config

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence


class PipelineConfigError(Exception):
    """Error loading or processing pivot.yaml configuration."""


class OutputOptions(TypedDict, total=False):
    """Valid options for output specifications."""

    cache: bool
    persist: bool
    x: str  # For plots
    y: str  # For plots
    template: str  # For plots


# Union type: YAML can have "path.txt" or {"path.txt": {cache: false}}
OutputSpec = str | dict[str, OutputOptions]

# Output specs from Python escape hatch can also include BaseOut directly
VariantOutputSpec = str | dict[str, OutputOptions] | outputs.BaseOut


class VariantDict(TypedDict, total=False):
    """Variant dict structure from Python escape hatch functions."""

    name: str
    deps: list[str]
    outs: list[VariantOutputSpec]
    params: dict[str, Any]
    mutex: list[str]
    cwd: str


class DimensionOverrides(pydantic.BaseModel, extra="forbid", populate_by_name=True):
    """Overrides that can be applied per matrix dimension value."""

    deps: list[str] | None = None
    outs: list[OutputSpec] | None = None
    metrics: list[OutputSpec] | None = None
    plots: list[OutputSpec] | None = None
    params: dict[str, Any] | None = None  # JSON-compatible values from YAML
    mutex: list[str] | None = None
    cwd: str | None = None

    # Append variants (Hydra-style)
    deps_append: list[str] | None = pydantic.Field(default=None, alias="deps+")
    outs_append: list[OutputSpec] | None = pydantic.Field(default=None, alias="outs+")
    metrics_append: list[OutputSpec] | None = pydantic.Field(default=None, alias="metrics+")
    plots_append: list[OutputSpec] | None = pydantic.Field(default=None, alias="plots+")
    mutex_append: list[str] | None = pydantic.Field(default=None, alias="mutex+")


# Primitive types allowed in matrix dimension lists
MatrixPrimitive = str | int | float | bool

# Matrix dimension can be a list ["a", "b", true, 0.5] or dict {"a": {overrides}, "b": {overrides}}
MatrixDimension = list[MatrixPrimitive] | dict[str, DimensionOverrides | None]


class StageConfig(pydantic.BaseModel, extra="forbid"):
    """Configuration for a single stage in pivot.yaml."""

    python: str
    deps: list[str] = []
    outs: list[OutputSpec] = []
    metrics: list[OutputSpec] = []
    plots: list[OutputSpec] = []
    params: dict[str, Any] = pydantic.Field(default_factory=dict)  # JSON-compatible from YAML
    mutex: list[str] = []
    cwd: str | None = None
    matrix: dict[str, MatrixDimension] | None = None
    variants: str | None = None

    @pydantic.field_validator("cwd")
    @classmethod
    def validate_cwd(cls, v: str | None) -> str | None:
        """Validate cwd doesn't contain path traversal."""
        if v is not None and ".." in v:
            raise ValueError(f"cwd cannot contain '..' (got '{v}')")
        return v


class PipelineConfig(pydantic.BaseModel, extra="forbid"):
    """Top-level pivot.yaml configuration."""

    stages: dict[str, StageConfig]


def _validate_callable(v: object) -> Callable[..., Any]:
    """Validate that value is callable."""
    if not callable(v):
        raise ValueError("must be callable")
    return v


class ExpandedStage(pydantic.BaseModel):
    """A stage after matrix expansion."""

    name: str
    func: Annotated[Callable[..., Any], pydantic.PlainValidator(_validate_callable)]
    deps: list[str]
    outs: list[outputs.BaseOut]
    params: pydantic.BaseModel | None
    mutex: list[str]
    cwd: str
    variant: str | None


def load_pipeline_file(pipeline_file: pathlib.Path) -> PipelineConfig:
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


def register_from_pipeline_file(pipeline_file: pathlib.Path) -> None:
    """Load pivot.yaml and register all stages to the global registry."""
    pipeline = load_pipeline_file(pipeline_file)
    pipeline_dir = pipeline_file.parent

    for stage_name, stage_config in pipeline.stages.items():
        _register_stage(stage_name, stage_config, pipeline_dir)


def _register_stage(name: str, config: StageConfig, pipeline_dir: pathlib.Path) -> None:
    """Register a single stage from configuration."""
    if config.matrix is not None:
        expanded = _expand_matrix(name, config, pipeline_dir)
        for stage in expanded:
            registry.REGISTRY.register(
                func=stage.func,
                name=stage.name,
                deps=stage.deps,
                outs=stage.outs,
                params=stage.params,
                mutex=stage.mutex,
                variant=stage.variant,
                cwd=stage.cwd,
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
            _register_variant_from_dict(name, func, variant, pipeline_dir)
    else:
        _register_simple_stage(name, config, pipeline_dir)


def _register_simple_stage(name: str, config: StageConfig, pipeline_dir: pathlib.Path) -> None:
    """Register a simple (non-matrix) stage."""
    func = _import_function(config.python)
    cwd = config.cwd or str(pipeline_dir)
    outs_spec = (
        _normalize_output_specs(config.outs, outputs.Out)
        + _normalize_output_specs(config.metrics, outputs.Metric)
        + _normalize_output_specs(config.plots, outputs.Plot)
    )
    params_instance = _resolve_params(func, config.params, name)

    registry.REGISTRY.register(
        func=func,
        name=name,
        deps=config.deps,
        outs=outs_spec,
        params=params_instance,
        mutex=config.mutex,
        cwd=cwd,
    )


def _register_variant_from_dict(
    base_name: str,
    func: Callable[..., Any],
    variant: VariantDict,
    pipeline_dir: pathlib.Path,
) -> None:
    """Register a variant from Python escape hatch."""
    variant_name = variant.get("name", "default")
    full_name = f"{base_name}@{variant_name}"

    deps = variant.get("deps", [])
    outs_raw = variant.get("outs", [])
    params_dict = variant.get("params", {})
    mutex = variant.get("mutex", [])
    cwd_raw = variant.get("cwd")
    cwd = cwd_raw if cwd_raw is not None else str(pipeline_dir)

    outs_spec = _normalize_output_specs(outs_raw, outputs.Out)
    params_instance = _resolve_params(func, params_dict, full_name)

    registry.REGISTRY.register(
        func=func,
        name=full_name,
        deps=deps,
        outs=outs_spec,
        params=params_instance,
        mutex=mutex,
        variant=variant_name,
        cwd=cwd,
    )


def _expand_matrix(
    name: str, config: StageConfig, pipeline_dir: pathlib.Path
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

        deps = list(config.deps)
        outs_raw = list(config.outs)
        metrics_raw = list(config.metrics)
        plots_raw = list(config.plots)
        params_dict = dict(config.params)
        mutex = list(config.mutex)
        cwd: str | None = config.cwd

        for dim_name, key in string_values.items():
            overrides = normalized_dims[dim_name][key][1]
            deps, outs_raw, metrics_raw, plots_raw, params_dict, mutex, cwd = _apply_overrides(
                deps, outs_raw, metrics_raw, plots_raw, params_dict, mutex, cwd, overrides
            )

        cwd = cwd or str(pipeline_dir)

        # Use string values for path interpolation (deps, outs, cwd)
        deps = [_interpolate(d, string_values, full_name) for d in deps]
        outs_raw = [_interpolate_out(o, string_values, full_name) for o in outs_raw]
        metrics_raw = [_interpolate_out(m, string_values, full_name) for m in metrics_raw]
        plots_raw = [_interpolate_out(p, string_values, full_name) for p in plots_raw]
        cwd = _interpolate(cwd, string_values, full_name)

        # Use typed values for params interpolation (preserves int/float/bool)
        params_dict = {k: _interpolate_value(v, typed_values) for k, v in params_dict.items()}

        outs_spec = (
            _normalize_output_specs(outs_raw, outputs.Out)
            + _normalize_output_specs(metrics_raw, outputs.Metric)
            + _normalize_output_specs(plots_raw, outputs.Plot)
        )
        params_instance = _resolve_params(func, params_dict, full_name)

        expanded.append(
            ExpandedStage(
                name=full_name,
                func=func,
                deps=deps,
                outs=outs_spec,
                params=params_instance,
                mutex=mutex,
                cwd=cwd,
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


def _apply_overrides(
    deps: list[str],
    outs: list[OutputSpec],
    metrics: list[OutputSpec],
    plots: list[OutputSpec],
    params: dict[str, Any],
    mutex: list[str],
    cwd: str | None,
    overrides: DimensionOverrides,
) -> tuple[
    list[str],
    list[OutputSpec],
    list[OutputSpec],
    list[OutputSpec],
    dict[str, Any],
    list[str],
    str | None,
]:
    """Apply dimension overrides to stage config."""
    if overrides.deps is not None:
        deps = overrides.deps
    if overrides.deps_append:
        deps = deps + overrides.deps_append
    if overrides.outs is not None:
        outs = overrides.outs
    if overrides.outs_append:
        outs = outs + overrides.outs_append
    if overrides.metrics is not None:
        metrics = overrides.metrics
    if overrides.metrics_append:
        metrics = metrics + overrides.metrics_append
    if overrides.plots is not None:
        plots = overrides.plots
    if overrides.plots_append:
        plots = plots + overrides.plots_append
    if overrides.params is not None:
        params = {**params, **overrides.params}
    if overrides.mutex is not None:
        mutex = overrides.mutex
    if overrides.mutex_append:
        mutex = mutex + overrides.mutex_append
    if overrides.cwd is not None:
        cwd = overrides.cwd

    return deps, outs, metrics, plots, params, mutex, cwd


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


def _interpolate_out(out: OutputSpec, values: dict[str, str], stage_name: str) -> OutputSpec:
    """Interpolate ${dim} in output spec (string or dict)."""
    if isinstance(out, str):
        return _interpolate(out, values, stage_name)
    return {_interpolate(k, values, stage_name): v for k, v in out.items()}


def _interpolate_value(value: Any, values: dict[str, MatrixPrimitive]) -> Any:
    """Interpolate ${dim} in any value, including nested structures.

    When the value is exactly "${key}", returns the typed value (preserves int/float/bool).
    When the value contains "${key}" as substring, uses string replacement.
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
        return result
    if isinstance(value, dict):
        return {
            k: _interpolate_value(v, values)
            for k, v in typing.cast("dict[str, Any]", value).items()
        }
    if isinstance(value, list):
        return [_interpolate_value(v, values) for v in typing.cast("list[Any]", value)]
    return value


def _normalize_output_specs(
    specs: Sequence[VariantOutputSpec], out_cls: type[outputs.BaseOut]
) -> list[outputs.BaseOut]:
    """Convert output specs (str, dict, or BaseOut) to BaseOut objects."""
    result = list[outputs.BaseOut]()
    for spec in specs:
        if isinstance(spec, str):
            result.append(out_cls(path=spec))
        elif isinstance(spec, dict):
            for path, opts in spec.items():
                result.append(_make_output_with_options(out_cls, path, opts))
        else:
            # spec is outputs.BaseOut (from Python escape hatch variants)
            result.append(spec)
    return result


def _make_output_with_options(
    out_cls: type[outputs.BaseOut], path: str, opts: OutputOptions
) -> outputs.BaseOut:
    """Create an output object, handling class-specific options."""
    if out_cls is outputs.Plot:
        return outputs.Plot(
            path=path,
            cache=opts.get("cache", True),
            persist=opts.get("persist", True),
            x=opts.get("x"),
            y=opts.get("y"),
            template=opts.get("template"),
        )
    return out_cls(
        path=path,
        cache=opts.get("cache", True),
        persist=opts.get("persist", True),
    )


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
    """Resolve params from function signature + config overrides."""
    sig = inspect.signature(func)

    if "params" not in sig.parameters:
        if overrides:
            raise PipelineConfigError(
                f"Stage '{stage_name}': pivot.yaml has 'params' but function "
                + f"'{func.__name__}' has no 'params' parameter"
            )
        return None

    try:
        type_hints = typing.get_type_hints(func)
    except NameError as e:
        raise PipelineConfigError(
            f"Stage '{stage_name}': failed to resolve type hints for '{func.__name__}': {e}"
        ) from e

    if "params" not in type_hints:
        raise PipelineConfigError(
            f"Stage '{stage_name}': function '{func.__name__}' has 'params' parameter "
            + "but no type hint. Add a type hint like 'params: MyParams'"
        )

    type_hint = type_hints["params"]

    if not parameters.validate_params_cls(type_hint):
        raise PipelineConfigError(
            f"Stage '{stage_name}': params type hint must be a Pydantic BaseModel, "
            + f"got {type_hint}"
        )

    try:
        return type_hint(**overrides)
    except pydantic.ValidationError as e:
        raise PipelineConfigError(f"Stage '{stage_name}': invalid params: {e}") from e
