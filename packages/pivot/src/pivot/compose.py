from __future__ import annotations

import contextvars
import dataclasses
import functools
import importlib
import typing
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin, get_type_hints

from typing_extensions import is_typeddict

if TYPE_CHECKING:
    from collections.abc import Callable

    from . import loaders as loaders_mod
    from . import stage_def


def _get_loaders():  # type: ignore[return-value]
    from . import loaders

    return loaders


class _MetricTag:
    pass


class _PlotTag:
    pass


metric = _MetricTag()
plot = _PlotTag()

SINGLE_OUTPUT_KEY = "_single"

__all__ = [
    "ArtifactHandle",
    "SINGLE_OUTPUT_KEY",
    "_InputNode",
    "_OutputSpec",
    "_StageNode",
    "_analyze_return_type",
    "_format_extension",
    "_infer_format",
    "_infer_format_from_extension",
    "metric",
    "plot",
    "stage",
]


class Pipeline:
    pass


_active_pipeline: contextvars.ContextVar[Pipeline | None] = contextvars.ContextVar(
    "_active_pipeline", default=None
)


_DEFAULT_FORMATS: dict[str, str] = {
    "pandas.core.frame.DataFrame": "DataFrameJSONL",
    "builtins.dict": "YAML",
    "builtins.list": "YAML",
    "builtins.str": "Text",
    "pathlib.Path": "PathOnly",
}
_default_formats_resolved: dict[type, str] | None = None


def _resolve_default_formats() -> dict[type, str]:
    global _default_formats_resolved
    if _default_formats_resolved is not None:
        return _default_formats_resolved

    resolved = dict[type, str]()
    for qualname, loader_name in _DEFAULT_FORMATS.items():
        mod_name, cls_name = qualname.rsplit(".", 1)
        try:
            module = importlib.import_module(mod_name)
            resolved[getattr(module, cls_name)] = loader_name
        except (ImportError, AttributeError):
            continue

    _default_formats_resolved = resolved
    return resolved


def _infer_format(
    python_type: type,
) -> loaders_mod.Writer[object]:
    loaders = _get_loaders()
    defaults = _resolve_default_formats()
    loader_name = defaults.get(python_type)
    if loader_name is not None:
        loader_class = typing.cast(
            "type[loaders_mod.Writer[object]]", getattr(loaders, loader_name)
        )
        return loader_class()

    for registered_type, registered_loader in defaults.items():
        if issubclass(python_type, registered_type):
            loader_class = typing.cast(
                "type[loaders_mod.Writer[object]]", getattr(loaders, registered_loader)
            )
            return loader_class()

    raise ValueError(
        f"Cannot infer serialization format for type {python_type.__name__}. "
        + "Use Annotated[T, format] on the return type to specify explicitly."
    )


def _format_extension(
    fmt: object,
) -> str:
    loaders = _get_loaders()
    match fmt:
        case loaders.DataFrameJSONL():
            return "jsonl"
        case loaders.CSV():
            return "csv"
        case loaders.YAML():
            return "yaml"
        case loaders.JSON():
            return "json"
        case loaders.Text():
            return "txt"
        case loaders.Pickle():
            return "pkl"
        case loaders.MatplotlibFigure():
            return "png"
        case _:
            return "dat"


def _infer_format_from_extension(path: str) -> loaders_mod.Reader[object]:
    """Infer loader from file extension for p.input() declarations."""
    loaders = _get_loaders()
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    match ext:
        case "yaml" | "yml":
            return loaders.YAML()  # type: ignore[return-value]
        case "csv":
            return loaders.CSV()  # type: ignore[return-value]
        case "jsonl":
            return loaders.DataFrameJSONL()  # type: ignore[return-value]
        case "json":
            return loaders.JSON()  # type: ignore[return-value]
        case "txt" | "sql" | "jinja":
            return loaders.Text()  # type: ignore[return-value]
        case "pkl" | "pickle":
            return loaders.Pickle()  # type: ignore[return-value]
        case _:
            return loaders.PathOnly()  # type: ignore[return-value]


# --- Return type analysis ---


def _parse_output_type(hint: Any, key: str) -> list[_OutputSpec]:
    """Parse a type hint into OutputSpec(s)."""
    # Check for TypedDict (multi-output)
    if isinstance(hint, type) and is_typeddict(hint):
        specs = list[_OutputSpec]()
        field_hints = get_type_hints(hint, include_extras=True)
        for field_name, field_hint in field_hints.items():
            specs.extend(_parse_output_type(field_hint, field_name))
        return specs

    # Check for Annotated
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        base_type = args[0]
        tag: _MetricTag | _PlotTag | None = None
        fmt: loaders_mod.Writer[object] | None = None
        loaders = _get_loaders()
        for arg in args[1:]:
            if isinstance(arg, (_MetricTag, _PlotTag)):
                tag = arg
            elif isinstance(arg, loaders.Writer):
                fmt = typing.cast("loaders_mod.Writer[object]", arg)

        if fmt is None:
            fmt = _infer_format(base_type)
        return [_OutputSpec(key=key, python_type=base_type, format=fmt, tag=tag)]

    # Plain type
    fmt = _infer_format(hint)
    return [_OutputSpec(key=key, python_type=hint, format=fmt)]


def _analyze_return_type(func: Callable[..., object]) -> list[_OutputSpec]:
    """Extract output specs from a function's return type annotation."""
    hints = get_type_hints(func, include_extras=True)
    return_hint = hints.get("return")
    if return_hint is None:
        msg = f"Stage function {func.__name__} must have a return type annotation"
        raise ValueError(msg)
    return _parse_output_type(return_hint, SINGLE_OUTPUT_KEY)


# --- @stage decorator ---


def stage(func: Callable[..., object]) -> Callable[..., Any]:
    """Mark a function as a pipeline stage.

    In pipeline context (inside ``with Pipeline()``): records a DAG node, returns ArtifactHandle.
    Outside pipeline context: calls the function normally (for tests/notebooks).

    Return type analysis is DEFERRED to first pipeline-context use, not performed
    at decoration time. This allows stage modules to use in-body imports for heavy
    dependencies while still having type annotations that reference those types.
    """
    _cached_specs: list[_OutputSpec] | None = None

    def _get_output_specs() -> list[_OutputSpec]:
        nonlocal _cached_specs
        if _cached_specs is None:
            _cached_specs = _analyze_return_type(func)
        return _cached_specs

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pipeline = _active_pipeline.get()
        if pipeline is None:
            return func(*args, **kwargs)
        return pipeline._record_stage(func, wrapper, _get_output_specs(), args, kwargs)  # pyright: ignore[reportAttributeAccessIssue] - _record_stage will be added in Task 1.3

    wrapper._is_stage = True  # pyright: ignore[reportAttributeAccessIssue] - dynamic attr for stage detection
    wrapper._original_func = func  # pyright: ignore[reportAttributeAccessIssue] - for bridge to unwrap
    return wrapper


@dataclasses.dataclass
class _OutputSpec:
    key: str
    python_type: type
    format: loaders_mod.Writer[object]
    tag: _MetricTag | _PlotTag | None = None


@dataclasses.dataclass
class _StageNode:
    func: Callable[..., object]
    original_func: Callable[..., object]
    name: str
    params: stage_def.StageParams | None
    input_handles: dict[str, ArtifactHandle]
    output_specs: list[_OutputSpec]
    call_index: int


@dataclasses.dataclass
class _InputNode:
    name: str
    python_type: type | None
    path: str
    format: loaders_mod.Reader[object] | loaders_mod.Loader[object, object]


class ArtifactHandle:
    _pipeline: Pipeline
    _source: _StageNode | _InputNode
    _output_key: str | None
    _python_type: type

    def __init__(
        self,
        pipeline: Pipeline,
        source: _StageNode | _InputNode,
        output_key: str | None,
        python_type: type,
    ) -> None:
        self._pipeline = pipeline
        self._source = source
        self._output_key = output_key
        self._python_type = python_type

    def __getattr__(self, name: str) -> ArtifactHandle:
        if name.startswith("_"):
            raise AttributeError(name)
        if not isinstance(self._source, _StageNode):
            raise AttributeError(f"Input '{self._source.name}' has no sub-outputs")
        for spec in self._source.output_specs:
            if spec.key == name:
                return ArtifactHandle(
                    pipeline=self._pipeline,
                    source=self._source,
                    output_key=name,
                    python_type=spec.python_type,
                )
        available = [spec.key for spec in self._source.output_specs]
        raise AttributeError(
            f"Stage '{self._source.name}' has no output '{name}'. Available: {available}"
        )
