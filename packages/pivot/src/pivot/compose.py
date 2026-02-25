# pyright: reportPrivateUsage=false, reportUnusedFunction=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false, reportImportCycles=false, reportImplicitRelativeImport=false
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import functools
import importlib
import inspect
import logging
import types as _stdlib_types
import typing
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

from typing_extensions import is_typeddict

from . import project, registry, stage_def, types

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import collections.abc
    import pathlib
    from collections.abc import Callable

    from networkx import DiGraph

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

SINGLE_OUTPUT_KEY = stage_def.SINGLE_OUTPUT_KEY


class CollectionKind(enum.StrEnum):
    LIST = "list"
    TUPLE = "tuple"


__all__ = [
    "ArtifactHandle",
    "CollectionKind",
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


def _handle_to_artifact_ref(handle: ArtifactHandle, consumer_name: str) -> types.ArtifactRef:
    if isinstance(handle._source, _InputNode):
        identity = types.ArtifactIdentity(producer=handle._source.name, key=None)
        types.validate_artifact_identity(identity)
        return types.ArtifactRef(
            identity=identity,
            format=handle._source.format,
            python_type=handle._python_type,
            tag=types.ArtifactTag.DATA,
        )

    source = handle._source
    qualified_name = f"{handle._pipeline._name}/{source.name}"
    if not source.output_specs:
        raise TypeError(
            f"Stage '{consumer_name}' depends on '{qualified_name}' which has no outputs "
            f"(returns None). A stage must produce outputs to be used as a dependency."
        )
    if len(source.output_specs) == 1:
        output_spec = source.output_specs[0]
        output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
    else:
        matched = next(
            (spec for spec in source.output_specs if spec.key == handle._output_key), None
        )
        if matched is None:
            available = [spec.key for spec in source.output_specs]
            raise TypeError(
                f"Stage '{consumer_name}' received handle from multi-output stage "
                f"'{qualified_name}' without selecting an output. "
                f"Use handle.key or handle['key']. Available: {available}"
            )
        output_spec = matched
        output_key = output_spec.key

    if isinstance(output_spec.tag, _MetricTag):
        dep_tag = types.ArtifactTag.METRIC
    elif isinstance(output_spec.tag, _PlotTag):
        dep_tag = types.ArtifactTag.PLOT
    else:
        dep_tag = types.ArtifactTag.DATA

    identity = types.ArtifactIdentity(producer=qualified_name, key=output_key)
    types.validate_artifact_identity(identity)
    return types.ArtifactRef(
        identity=identity,
        format=output_spec.format,
        python_type=handle._python_type,
        tag=dep_tag,
    )


def _iter_stage_handles(node: _StageNode) -> list[ArtifactHandle]:
    result = list[ArtifactHandle]()
    for handle in node.input_handles.values():
        if isinstance(handle._source, _StageNode):
            result.append(handle)
    for handles in node.list_input_handles.values():
        for handle in handles:
            if isinstance(handle._source, _StageNode):
                result.append(handle)
    return result


def _iter_all_handles(node: _StageNode) -> list[ArtifactHandle]:
    result = list[ArtifactHandle]()
    for handle in node.input_handles.values():
        result.append(handle)
    for handles in node.list_input_handles.values():
        for handle in handles:
            result.append(handle)
    return result


def _emit_stage_info(
    node: _StageNode,
    qualified_name: str,
    state_dir: pathlib.Path,
) -> registry.RegistryStageInfo:
    func = node.original_func
    assert not hasattr(func, "_is_stage")
    no_fp = getattr(node.func, "__pivot_no_fingerprint__", False)
    if not no_fp:
        no_fp = getattr(func, "__pivot_no_fingerprint__", False)

    deps = dict[str, types.ArtifactRef]()

    for param_name, handle in node.input_handles.items():
        deps[param_name] = _handle_to_artifact_ref(handle, qualified_name)

    for param_name, handles in node.list_input_handles.items():
        for i, handle in enumerate(handles):
            deps[f"{param_name}[{i}]"] = _handle_to_artifact_ref(handle, qualified_name)

    outs = list[types.ArtifactRef]()

    for output_spec in node.output_specs:
        output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
        if isinstance(output_spec.tag, _MetricTag):
            out_tag = types.ArtifactTag.METRIC
        elif isinstance(output_spec.tag, _PlotTag):
            out_tag = types.ArtifactTag.PLOT
        else:
            out_tag = types.ArtifactTag.DATA

        identity = types.ArtifactIdentity(producer=qualified_name, key=output_key)
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
                if isinstance(union_arg, type) and issubclass(union_arg, stage_def.StageParams):
                    raise TypeError(
                        f"Stage '{qualified_name}' parameter '{name}' has type "
                        f"'{base_hint}' — StageParams must not be in a union. "
                        f"Use 'params: {union_arg.__name__}' directly, with a "
                        f"default if needed (params: {union_arg.__name__} = "
                        f"{union_arg.__name__}())."
                    )
        if isinstance(base_hint, type) and issubclass(base_hint, stage_def.StageParams):
            params_arg_name = name
            break

    return registry.RegistryStageInfo(
        func=func,
        name=qualified_name,
        deps=deps,
        outs=outs,
        params=node.params,
        mutex=[],
        variant=node.variant,
        signature=inspect.signature(func),
        fingerprint=None,
        params_arg_name=params_arg_name,
        state_dir=state_dir,
        collection_params={k: str(v) for k, v in node.collection_params.items()},
        no_fingerprint=no_fp,
    )


class Pipeline:
    _name: str
    _root: pathlib.Path
    _stages: list[_StageNode]
    _inputs: dict[str, _InputNode]
    _call_counts: dict[tuple[str, tuple[str, ...]], int]
    _validation_errors: list[str]
    _token: contextvars.Token[Pipeline | None] | None
    _variant_stack: list[str]
    _registry: registry.StageRegistry
    _constructing: bool

    def __init__(self, name: str, *, root: pathlib.Path | None = None) -> None:
        self._name = name
        self._stages = []
        self._inputs = {}
        self._call_counts = {}
        self._validation_errors = []
        self._token = None
        self._variant_stack = []
        self._registry = registry.StageRegistry()
        self._constructing = False

        if root is not None:
            self._root = root.resolve()
        else:
            self._root = project.get_project_root()

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> pathlib.Path:
        return self._root

    @property
    def state_dir(self) -> pathlib.Path:
        return self._root / ".pivot"

    @property
    def input_bindings(self) -> dict[str, str]:
        return {name: node.path for name, node in self._inputs.items()}

    def _require_materialized(self) -> None:
        if self._constructing:
            raise RuntimeError(
                "Cannot call engine-facing methods on a Pipeline that is still under "
                "construction (inside a 'with' block). Exit the 'with' block first."
            )

    def list_stages(self) -> list[str]:
        self._require_materialized()
        return self._registry.list_stages()

    def get_stage(self, name: str) -> registry.RegistryStageInfo:
        self._require_materialized()
        return self._registry.get(name)

    def ensure_fingerprint(self, name: str) -> dict[str, str]:
        self._require_materialized()
        return self._registry.ensure_fingerprint(name)

    def build_dag(self) -> DiGraph[str]:
        self._require_materialized()
        return self._registry.build_dag()

    def invalidate_dag_cache(self) -> None:
        self._registry.invalidate_dag_cache()

    def snapshot(self) -> dict[str, registry.RegistryStageInfo]:
        self._require_materialized()
        return self._registry.snapshot()

    def restore(self, snapshot: dict[str, registry.RegistryStageInfo]) -> None:
        self._require_materialized()
        self._registry.invalidate_dag_cache()
        self._registry.restore(snapshot)

    def include(self, other: registry.PipelineLike) -> None:
        import copy

        if other is self:
            from pivot.exceptions import PipelineConfigError

            raise PipelineConfigError(f"Pipeline '{self.name}' cannot include itself")

        existing_stages = set(self._registry.list_stages())
        for stage_name in other.list_stages():
            if stage_name in existing_stages:
                logger.debug("include: skipping duplicate stage '%s'", stage_name)
                continue
            stage_info = copy.deepcopy(other.get_stage(stage_name))
            self._registry.add_existing(stage_info)

        for name, path in other.input_bindings.items():
            if name in self._inputs:
                existing_path = self._inputs[name].path
                if existing_path != path:
                    from pivot.exceptions import PipelineConfigError

                    raise PipelineConfigError(
                        f"Input binding conflict: input '{name}' is bound to "
                        f"'{existing_path}' and '{path}' in different pipelines."
                    )
            else:
                self._inputs[name] = _InputNode(
                    name=name, python_type=None, path=path, format=_get_loaders().PathOnly()
                )
        self._registry.invalidate_dag_cache()

    def __enter__(self) -> Pipeline:
        if self._token is not None:
            raise RuntimeError(
                f"Pipeline '{self._name}' is already inside a 'with' block. "
                "Nested 'with' on the same Pipeline is not supported."
            )
        self._token = _active_pipeline.set(self)
        self._constructing = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            _active_pipeline.reset(self._token)
            self._token = None
        self._constructing = False
        if exc_type is None:
            self._validate()
            self._materialize_stages()

    def _materialize_stages(self) -> None:
        new_registry = registry.StageRegistry()
        closure = self._upstream_closure()

        for foreign_pipeline, foreign_node in closure:
            qualified_name = f"{foreign_pipeline._name}/{foreign_node.name}"
            stage_info = _emit_stage_info(
                foreign_node, qualified_name, foreign_pipeline._root / ".pivot"
            )
            new_registry.add_existing(stage_info)

        self._merge_foreign_input_bindings(closure)

        for node in self._stages:
            qualified_name = f"{self._name}/{node.name}"
            stage_info = _emit_stage_info(node, qualified_name, self._root / ".pivot")
            new_registry.add_existing(stage_info)

        # Atomic swap — if any step above raises, self._registry is unchanged
        self._registry = new_registry

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
    def variant(self, name: str) -> collections.abc.Generator[None]:
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

        variant_key = tuple(self._variant_stack)
        count_key = (func_name, variant_key)
        count = self._call_counts.get(count_key, 0)
        self._call_counts[count_key] = count + 1
        stage_name = func_name if count == 0 else f"{func_name}@{count}"

        if self._variant_stack:
            for variant in self._variant_stack:
                stage_name = f"{stage_name}@{variant}"

        sig = inspect.signature(original_func)
        try:
            bound = sig.bind(*args, **kwargs)
            explicit_params = set(bound.arguments.keys())
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
        list_input_handles = dict[str, list[ArtifactHandle]]()
        collection_params = dict[str, CollectionKind]()
        params: stage_def.StageParams | None = None
        for param_name, value in bound.arguments.items():
            if isinstance(value, ArtifactHandle):
                input_handles[param_name] = value
            elif isinstance(value, (list, tuple)):
                items = list[Any](value)
                if all(isinstance(v, ArtifactHandle) for v in items):
                    list_input_handles[param_name] = items
                    collection_params[param_name] = (
                        CollectionKind.TUPLE if isinstance(value, tuple) else CollectionKind.LIST
                    )
                elif param_name in explicit_params:
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' has unsupported type "
                        f"'{type(value).__name__}'. Use ArtifactHandle (dependency), "
                        f"StageParams (config), or list/tuple of ArtifactHandle."
                    )
            elif isinstance(value, stage_def.StageParams):
                params = value
            elif param_name in explicit_params:
                self._validation_errors.append(
                    f"{stage_name}: parameter '{param_name}' has unsupported type "
                    f"'{type(value).__name__}'. Use ArtifactHandle (dependency), "
                    f"StageParams (config), or list/tuple of ArtifactHandle."
                )

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
            list_input_handles=list_input_handles,
            collection_params=collection_params,
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

    def _upstream_closure(self) -> list[tuple[Pipeline, _StageNode]]:
        """Collect foreign stages reachable from this pipeline's handles via DFS."""
        visited = set[int]()
        result = list[tuple[Pipeline, _StageNode]]()
        stack = list[tuple[Pipeline, _StageNode]]()

        for node in self._stages:
            for handle in _iter_stage_handles(node):
                if handle._pipeline is not self:
                    source = handle._source
                    assert isinstance(source, _StageNode)  # guaranteed by _iter_stage_handles
                    stack.append((handle._pipeline, source))

        while stack:
            foreign_pipeline, stage_node = stack.pop()
            node_id = id(stage_node)
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append((foreign_pipeline, stage_node))

            for handle in _iter_stage_handles(stage_node):
                if handle._pipeline is not self:
                    source = handle._source
                    assert isinstance(source, _StageNode)  # guaranteed by _iter_stage_handles
                    stack.append((handle._pipeline, source))

        return result

    def _merge_foreign_input_bindings(
        self,
        closure: list[tuple[Pipeline, _StageNode]],
    ) -> None:
        foreign_bindings = dict[str, str]()
        for _foreign_pipeline, foreign_node in closure:
            for handle in _iter_all_handles(foreign_node):
                if isinstance(handle._source, _InputNode):
                    inp = handle._source
                    if inp.name in foreign_bindings and foreign_bindings[inp.name] != inp.path:
                        raise ValueError(
                            f"Input binding conflict: input '{inp.name}' is bound to "
                            f"'{foreign_bindings[inp.name]}' and '{inp.path}' "
                            f"in different pipelines."
                        )
                    foreign_bindings[inp.name] = inp.path

        local_bindings = self.input_bindings
        for name, path in foreign_bindings.items():
            if name in local_bindings and local_bindings[name] != path:
                raise ValueError(
                    f"Input binding conflict: input '{name}' is bound to "
                    f"'{local_bindings[name]}' locally and '{path}' in a foreign pipeline."
                )
            if name not in self._inputs:
                self._inputs[name] = _InputNode(
                    name=name, python_type=None, path=path, format=_get_loaders().PathOnly()
                )


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
    return loaders.format_extension(fmt)


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
    origin = get_origin(hint)
    if origin is typing.Required or origin is typing.NotRequired:
        hint = get_args(hint)[0]

    if isinstance(hint, type) and is_typeddict(hint):
        specs = list[_OutputSpec]()
        field_hints = get_type_hints(hint, include_extras=True)
        for field_name, field_hint in field_hints.items():
            if field_name == SINGLE_OUTPUT_KEY:
                raise ValueError(
                    f"TypedDict field name {SINGLE_OUTPUT_KEY!r} is reserved by Pivot. "
                    f"Rename the field in {hint.__name__}."
                )
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
    list_input_handles: dict[str, list[ArtifactHandle]] = dataclasses.field(default_factory=dict)
    collection_params: dict[str, CollectionKind] = dataclasses.field(default_factory=dict)


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
