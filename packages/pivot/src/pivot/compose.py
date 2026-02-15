# pyright: reportPrivateUsage=false, reportUnusedFunction=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import types as _stdlib_types
import typing
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

from typing_extensions import is_typeddict

from . import project, registry, stage_def, types
from .pipeline import pipeline as pipeline_mod

if TYPE_CHECKING:
    import pathlib
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
    _variant_stack: list[str]

    def __init__(self, name: str, *, root: pathlib.Path | None = None) -> None:
        self._name = name
        self._stages = []
        self._inputs = {}
        self._call_counts = {}
        self._validation_errors = []
        self._token = None
        self._variant_stack = []

        if root is not None:
            self._root = root.resolve()
        else:
            self._root = project.get_project_root()

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

    @typing.overload
    def input(
        self,
        name: str,
        *,
        t: type[_T],
        path: str | None = None,
        external: bool = False,
        format: loaders_mod.Reader[object] | loaders_mod.Loader[object, object] | None = None,
    ) -> _T: ...

    @typing.overload
    def input(
        self,
        name: str,
        *,
        path: str | None = None,
        external: bool = False,
        format: loaders_mod.Reader[object] | loaders_mod.Loader[object, object] | None = None,
    ) -> ArtifactHandle: ...

    def input(
        self,
        name: str,
        *,
        t: type | None = None,
        path: str | None = None,
        external: bool = False,
        format: loaders_mod.Reader[object] | loaders_mod.Loader[object, object] | None = None,
    ) -> Any:
        if path is None:
            prefix = "data/external" if external else "data/raw"
            path = f"{prefix}/{name}"
        if format is None:
            if t is not None:
                format = _infer_input_format_from_type(t)
            if format is None:
                format = _infer_format_from_extension(path)
        node = _InputNode(name=name, python_type=t, path=path, format=format)
        self._inputs[name] = node
        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=t or object,
        )

    @contextlib.contextmanager
    def variant(self, name: str) -> typing.Generator[None]:
        """Context manager for registering stages with a variant suffix.

        All stages registered within the block get @{name} appended to their stage name.
        Nesting is supported: variant("a") inside variant("b") → @b@a suffix.
        """
        self._variant_stack.append(name)
        try:
            yield
        finally:
            self._variant_stack.pop()

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

        if self._variant_stack:
            for variant in self._variant_stack:
                stage_name = f"{stage_name}@{variant}"

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

        variant_str = "@".join(self._variant_stack) if self._variant_stack else None

        node = _StageNode(
            func=wrapper,
            original_func=original_func,
            name=stage_name,
            params=params,
            input_handles=input_handles,
            output_specs=output_specs,
            call_index=count,
            variant=variant_str,
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

        for node in self._stages:
            func = node.original_func
            assert not hasattr(func, "_is_stage")

            deps = dict[str, types.ArtifactRef]()

            for param_name, handle in node.input_handles.items():
                if isinstance(handle._source, _InputNode):
                    identity = types.ArtifactIdentity(producer=handle._source.name, key=None)
                    types.validate_artifact_identity(identity)
                    deps[param_name] = types.ArtifactRef(
                        identity=identity,
                        format=handle._source.format,
                        python_type=handle._python_type,
                        tag=types.ArtifactTag.DATA,
                    )
                    continue

                source = handle._source
                if not source.output_specs:
                    raise TypeError(
                        f"Stage '{node.name}' depends on '{source.name}' which has no outputs "
                        f"(returns None). A stage must produce outputs to be used as a dependency."
                    )
                if len(source.output_specs) == 1:
                    output_spec = source.output_specs[0]
                    output_key = None
                else:
                    output_spec = next(
                        spec for spec in source.output_specs if spec.key == handle._output_key
                    )
                    output_key = output_spec.key

                if isinstance(output_spec.tag, _MetricTag):
                    dep_tag = types.ArtifactTag.METRIC
                elif isinstance(output_spec.tag, _PlotTag):
                    dep_tag = types.ArtifactTag.PLOT
                else:
                    dep_tag = types.ArtifactTag.DATA

                identity = types.ArtifactIdentity(producer=source.name, key=output_key)
                types.validate_artifact_identity(identity)
                deps[param_name] = types.ArtifactRef(
                    identity=identity,
                    format=output_spec.format,
                    python_type=handle._python_type,
                    tag=dep_tag,
                )

            outs = list[types.ArtifactRef]()

            is_single_output = len(node.output_specs) == 1
            for output_spec in node.output_specs:
                output_key = None if is_single_output else output_spec.key
                if isinstance(output_spec.tag, _MetricTag):
                    out_tag = types.ArtifactTag.METRIC
                elif isinstance(output_spec.tag, _PlotTag):
                    out_tag = types.ArtifactTag.PLOT
                else:
                    out_tag = types.ArtifactTag.DATA

                identity = types.ArtifactIdentity(producer=node.name, key=output_key)
                types.validate_artifact_identity(identity)
                outs.append(
                    types.ArtifactRef(
                        identity=identity,
                        format=output_spec.format,
                        python_type=output_spec.python_type,
                        tag=out_tag,
                    )
                )

            params_arg_name: str | None = None
            hints = get_type_hints(func, include_extras=True)
            for name, hint in hints.items():
                if name == "return":
                    continue
                base_hint = hint
                if get_origin(base_hint) is Annotated:
                    base_hint = get_args(base_hint)[0]
                origin = get_origin(base_hint)
                if origin is _stdlib_types.UnionType or origin is typing.Union:  # pyright: ignore[reportDeprecated] - runtime identity check
                    for union_arg in get_args(base_hint):
                        if isinstance(union_arg, type) and issubclass(
                            union_arg, stage_def.StageParams
                        ):
                            raise TypeError(
                                f"Stage '{node.name}' parameter '{name}' has type "
                                f"'{base_hint}' — StageParams must not be in a union. "
                                f"Use 'params: {union_arg.__name__}' directly, with a "
                                f"default if needed (params: {union_arg.__name__} = "
                                f"{union_arg.__name__}())."
                            )
                if isinstance(base_hint, type) and issubclass(base_hint, stage_def.StageParams):
                    params_arg_name = name
                    break

            stage_info = registry.RegistryStageInfo(
                func=func,
                name=node.name,
                deps=deps,
                outs=outs,
                params=node.params,
                mutex=[],
                variant=node.variant,
                signature=inspect.signature(func),
                fingerprint=None,
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


def _infer_input_format_from_type(t: type) -> loaders_mod.Reader[object] | None:
    import pathlib as _pathlib

    loaders = _get_loaders()
    if t is _pathlib.Path:
        return loaders.PathOnly()  # type: ignore[return-value]
    if t is str:
        return loaders.Text()  # type: ignore[return-value]
    return None


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
            if isinstance(tag, _PlotTag):
                fmt = typing.cast("loaders_mod.Writer[object]", loaders.MatplotlibFigure())
            elif isinstance(tag, _MetricTag):
                fmt = typing.cast("loaders_mod.Writer[object]", loaders.YAML())
            else:
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
    if return_hint is type(None):
        return []
    return _parse_output_type(return_hint, SINGLE_OUTPUT_KEY)


# --- @stage decorator ---

_T = TypeVar("_T")


def stage[**P, R](func: Callable[P, R]) -> Callable[P, R]:
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
        return pipeline._record_stage(func, wrapper, _get_output_specs(), args, kwargs)

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
    variant: str | None = None


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

    def __getitem__(self, key: str) -> ArtifactHandle:
        if not isinstance(self._source, _StageNode):
            raise KeyError(f"Input '{self._source.name}' has no sub-outputs")
        for spec in self._source.output_specs:
            if spec.key == key:
                return ArtifactHandle(
                    pipeline=self._pipeline,
                    source=self._source,
                    output_key=key,
                    python_type=spec.python_type,
                )
        available = [spec.key for spec in self._source.output_specs]
        raise KeyError(f"Stage '{self._source.name}' has no output '{key}'. Available: {available}")
