from __future__ import annotations

import contextvars
import dataclasses
import functools
import importlib
import inspect
import pathlib
import typing
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin, get_type_hints

from typing_extensions import is_typeddict

from . import outputs, registry, stage_def
from .pipeline import pipeline as pipeline_mod

if TYPE_CHECKING:
    from collections.abc import Callable

    from . import loaders as loaders_mod


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


def _artifact_dir_prefix(tag: _MetricTag | _PlotTag | None) -> str:
    if isinstance(tag, _MetricTag):
        return "metrics"
    if isinstance(tag, _PlotTag):
        return "plots"
    return "data"


def _generate_artifact_path(
    pipeline_name: str,
    stage_name: str,
    output_spec: _OutputSpec,
    is_single_output: bool,
) -> str:
    prefix = _artifact_dir_prefix(output_spec.tag)
    ext = _format_extension(output_spec.format)
    if is_single_output:
        return f"{prefix}/{pipeline_name}/{stage_name}.{ext}"
    return f"{prefix}/{pipeline_name}/{stage_name}/{output_spec.key}.{ext}"


class Pipeline:
    _name: str
    _root: pathlib.Path
    _stages: list[_StageNode]
    _inputs: dict[str, _InputNode]
    _call_counts: dict[str, int]
    _validation_errors: list[str]
    _token: contextvars.Token[Pipeline | None] | None

    def __init__(self, name: str, *, root: pathlib.Path | None = None) -> None:
        self._name = name
        self._stages = []
        self._inputs = {}
        self._call_counts = {}
        self._validation_errors = []
        self._token = None

        if root is not None:
            self._root = root.resolve()
        else:
            frame = inspect.currentframe()
            try:
                if frame is None or frame.f_back is None:
                    raise RuntimeError("Cannot determine caller frame")
                caller_file = frame.f_back.f_globals.get("__file__")
                if caller_file is None:
                    raise RuntimeError("Provide explicit root= for Pipeline")
                self._root = pathlib.Path(caller_file).resolve().parent
            finally:
                del frame

    @property
    def name(self) -> str:
        return self._name

    def __enter__(self) -> Pipeline:
        self._token = _active_pipeline.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            _active_pipeline.reset(self._token)
            self._token = None
        if exc_type is None:
            self._validate()

    def input(
        self,
        name: str,
        path: str,
        format: loaders_mod.Reader[object] | loaders_mod.Loader[object, object] | None = None,
        python_type: type | None = None,
    ) -> ArtifactHandle:
        if format is None:
            format = _infer_format_from_extension(path)
        node = _InputNode(name=name, python_type=python_type, path=path, format=format)
        self._inputs[name] = node
        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=python_type or object,
        )

    def _record_stage(
        self,
        original_func: Callable[..., object],
        wrapper: Callable[..., object],
        output_specs: list[_OutputSpec],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> ArtifactHandle:
        func_name = original_func.__name__

        count = self._call_counts.get(func_name, 0)
        self._call_counts[func_name] = count + 1
        stage_name = func_name if count == 0 else f"{func_name}@{count}"

        sig = inspect.signature(original_func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as exc:
            self._validation_errors.append(f"{stage_name}: {exc}")
            dummy_input = _InputNode(
                name="__error__",
                python_type=None,
                path="__error__",
                format=_get_loaders().PathOnly(),
            )
            return ArtifactHandle(self, dummy_input, None, object)

        input_handles = dict[str, ArtifactHandle]()
        params: stage_def.StageParams | None = None
        for param_name, value in bound.arguments.items():
            if isinstance(value, ArtifactHandle):
                input_handles[param_name] = value
            elif isinstance(value, stage_def.StageParams):
                params = value

        node = _StageNode(
            func=wrapper,
            original_func=original_func,
            name=stage_name,
            params=params,
            input_handles=input_handles,
            output_specs=output_specs,
            call_index=count,
        )
        self._stages.append(node)

        if len(output_specs) == 1:
            return ArtifactHandle(
                pipeline=self,
                source=node,
                output_key=None,
                python_type=output_specs[0].python_type,
            )
        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=dict,
        )

    def _validate(self) -> None:
        if self._validation_errors:
            msg = (
                f'Pipeline "{self._name}" has {len(self._validation_errors)} '
                + "validation error(s):\n\n"
            )
            for err in self._validation_errors:
                msg += f"  {err}\n"
            raise ValueError(msg)

    def build(self) -> pipeline_mod.Pipeline:
        legacy = pipeline_mod.Pipeline(self._name, root=self._root)

        path_map = dict[tuple[int, str | None], str]()
        for node in self._stages:
            is_single_output = len(node.output_specs) == 1
            for output_spec in node.output_specs:
                path = _generate_artifact_path(
                    self._name,
                    node.name,
                    output_spec,
                    is_single_output,
                )
                output_key = None if is_single_output else output_spec.key
                path_map[(id(node), output_key)] = path

        for node in self._stages:
            func = node.original_func
            assert not hasattr(func, "_is_stage")

            deps = dict[str, outputs.PathType]()
            dep_specs = dict[str, stage_def.FuncDepSpec]()
            deps_paths = list[str]()

            for param_name, handle in node.input_handles.items():
                if isinstance(handle._source, _InputNode):
                    path = handle._source.path
                    loader = handle._source.format
                else:
                    source = typing.cast("_StageNode", handle._source)
                    path = path_map[(id(source), handle._output_key)]
                    if len(source.output_specs) == 1:
                        output_spec = source.output_specs[0]
                    else:
                        output_spec = next(
                            spec for spec in source.output_specs if spec.key == handle._output_key
                        )
                    loader = output_spec.format

                deps[param_name] = path
                deps_paths.append(path)
                dep_specs[param_name] = stage_def.FuncDepSpec(
                    path=path,
                    loader=typing.cast("loaders_mod.Reader[object]", loader),
                    creates_dep_edge=True,
                )

            outs = list[outputs.ExpandedOut]()
            outs_paths = list[str]()
            out_specs = dict[str, outputs.BaseOut]()

            is_single_output = len(node.output_specs) == 1
            for output_spec in node.output_specs:
                out_path = _generate_artifact_path(
                    self._name,
                    node.name,
                    output_spec,
                    is_single_output,
                )
                if isinstance(output_spec.tag, _MetricTag):
                    out = outputs.Metric(out_path, loader=output_spec.format)
                elif isinstance(output_spec.tag, _PlotTag):
                    out = outputs.Plot(out_path, loader=output_spec.format)
                else:
                    out = outputs.Out(out_path, loader=output_spec.format)

                outs.append(outputs.require_expanded(out))
                outs_paths.append(out_path)
                out_specs[output_spec.key] = out

            params_arg_name: str | None = None
            hints = get_type_hints(func, include_extras=True)
            for name, hint in hints.items():
                if name == "return":
                    continue
                base_hint = hint
                if get_origin(base_hint) is Annotated:
                    base_hint = get_args(base_hint)[0]
                if isinstance(base_hint, type) and issubclass(base_hint, stage_def.StageParams):
                    params_arg_name = name
                    break

            stage_info = registry.RegistryStageInfo(
                func=func,
                name=node.name,
                deps=deps,
                deps_paths=deps_paths,
                outs=outs,
                outs_paths=outs_paths,
                params=node.params,
                mutex=[],
                variant=None,
                signature=inspect.signature(func),
                fingerprint=None,
                dep_specs=dep_specs,
                out_specs=out_specs,
                params_arg_name=params_arg_name,
                state_dir=self._root / ".pivot",
            )

            legacy._registry.add_existing(stage_info)

        return legacy


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

    # Unwrap generic aliases (e.g. dict[str, Any] → dict) for lookup and issubclass
    origin = get_origin(python_type)
    lookup_type = origin if origin is not None else python_type

    loader_name = defaults.get(lookup_type)
    if loader_name is not None:
        loader_class = typing.cast(
            "type[loaders_mod.Writer[object]]", getattr(loaders, loader_name)
        )
        return loader_class()

    for registered_type, registered_loader in defaults.items():
        if isinstance(lookup_type, type) and issubclass(lookup_type, registered_type):
            loader_class = typing.cast(
                "type[loaders_mod.Writer[object]]", getattr(loaders, registered_loader)
            )
            return loader_class()

    type_name = getattr(python_type, "__name__", str(python_type))
    raise ValueError(
        f"Cannot infer serialization format for type {type_name}. "
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
