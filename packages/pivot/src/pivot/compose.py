from __future__ import annotations

import contextlib
import contextvars
import copy
import dataclasses
import enum
import functools
import importlib
import inspect
import json
import logging
import os
import pathlib
import types as _stdlib_types
import typing
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Never,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import is_typeddict

from . import (
    exceptions,
    fingerprint,
    parameters,
    path_policy,
    project,
    registry,
    stage_def,
    types,
)
from . import loaders as loaders_mod
from . import state_dir as state_dir_mod
from .storage import lock

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import collections.abc
    from collections.abc import Callable, Mapping

    from networkx import DiGraph


class _MetricTag:
    pass


class _PlotTag:
    pass


class _IncrementalTag:
    pass


class _DirectoryTag:
    pass


metric = _MetricTag()
plot = _PlotTag()
incremental = _IncrementalTag()
directory = _DirectoryTag()

_OutputTag = _MetricTag | _PlotTag | _IncrementalTag | _DirectoryTag

SINGLE_OUTPUT_KEY = stage_def.SINGLE_OUTPUT_KEY
_T = TypeVar("_T")


class CollectionKind(enum.StrEnum):
    LIST = "list"
    TUPLE = "tuple"


__all__ = [
    "ArtifactHandle",
    "directory",
    "incremental",
    "metric",
    "plot",
    "_InputNode",
    "_OutputSpec",
    "_StageNode",
    "_analyze_return_type",
    "_format_extension",
    "_infer_format",
    "_infer_format_from_extension",
    "stage",
]


def _artifact_tag(tag: _OutputTag | None) -> types.ArtifactTag:
    """The engine tag for an output's compose tag; a dep and an out of one output share it."""
    if isinstance(tag, _MetricTag):
        return types.ArtifactTag.METRIC
    if isinstance(tag, _PlotTag):
        return types.ArtifactTag.PLOT
    if isinstance(tag, _IncrementalTag):
        return types.ArtifactTag.INCREMENTAL
    if isinstance(tag, _DirectoryTag):
        return types.ArtifactTag.DIRECTORY
    return types.ArtifactTag.DATA


def _handle_to_artifact_ref(handle: ArtifactHandle, consumer_name: str) -> types.ArtifactRef:
    if isinstance(handle._source, _InputNode):  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle; public names would shadow output keys
        identity = types.ArtifactIdentity(producer=handle._source.name, key=None)  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
        types.validate_artifact_identity(identity)
        return types.ArtifactRef(
            identity=identity,
            format=handle._source.format,  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            python_type=handle._python_type,  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            tag=types.ArtifactTag.INPUT,
        )

    source = handle._source  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
    qualified_name = f"{handle._pipeline._name}/{source.name}"  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle and Pipeline
    if not source.output_specs:
        raise TypeError(
            f"Stage '{consumer_name}' depends on '{qualified_name}' which has no outputs "
            + "(returns None). A stage must produce outputs to be used as a dependency."
        )
    if len(source.output_specs) == 1:
        output_spec = source.output_specs[0]
        output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
    else:
        matched = _select_output(source.output_specs, handle._output_key)  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
        if matched is None:
            available = [spec.key for spec in source.output_specs]
            raise TypeError(
                f"Stage '{consumer_name}' received handle from multi-output stage "
                + f"'{qualified_name}' without selecting an output. "
                + f"Use handle.key or handle['key']. Available: {available}"
            )
        output_spec = matched
        output_key = output_spec.key

    if not isinstance(output_spec.format, loaders_mod.Reader):
        raise TypeError(
            f"Stage '{consumer_name}' depends on {_output_label(output_spec.key)} of stage "
            + f"'{qualified_name}', whose loader {type(output_spec.format).__name__} cannot read "
            + "it back: it only writes. Give that output a loaders.Loader, which reads and "
            + "writes, or do not pass it to a stage."
        )

    identity = types.ArtifactIdentity(producer=qualified_name, key=output_key)
    directory: types.ArtifactIdentity | None = None
    member = handle._member  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
    if member is not None:
        directory = identity
        identity = _directory_file_identity(directory, member, consumer_name)
    types.validate_artifact_identity(identity)
    return types.ArtifactRef(
        identity=identity,
        format=output_spec.format,
        python_type=handle._python_type,  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
        tag=_artifact_tag(output_spec.tag),
        directory=directory,
    )


def _directory_file_identity(
    directory: types.ArtifactIdentity, key: str, consumer_name: str
) -> types.ArtifactIdentity:
    """Name one file of a directory output by its normalized path inside it.

    The key is checked by the rules for a directory output's keys, so a consumer can only name
    a file the producer could write.
    """
    try:
        member = stage_def.normalize_directory_out_key(key)
    except ValueError as error:
        raise ValueError(
            f"Stage '{consumer_name}' depends on file {key!r} of stage '{directory.producer}': "
            + f"{error}"
        ) from None
    return types.directory_file_identity(directory, member)


def _iter_stage_handles(node: _StageNode) -> list[ArtifactHandle]:
    result = list[ArtifactHandle]()
    for handle in node.input_handles.values():
        if isinstance(handle._source, _StageNode):  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            result.append(handle)
    for handles in node.list_input_handles.values():
        for handle in handles:
            if isinstance(handle._source, _StageNode):  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
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


def _stage_definition(func: Callable[..., Any]) -> fingerprint.FunctionIdentity:
    """Name the function a stage runs: `fingerprint.function_identity` of the unwrapped function.

    Importing a pipeline file again makes new function objects with the same identity. Two
    closures from one factory (`make_stage(2)`, `make_stage(5)`) share a file and qualname but
    differ in their closure values.
    """
    return fingerprint.function_identity(fingerprint.unwrap(func))


def _shown_file(func: Callable[..., Any], root: pathlib.Path) -> str:
    """The file defining a stage function, relative to the pipeline root when inside it."""
    func = fingerprint.unwrap(func)
    source = inspect.getsourcefile(func)
    if source is None:
        return func.__module__
    path = pathlib.Path(source).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _cell_values(func: Callable[..., Any]) -> dict[str, object]:
    """The live closure cell values of `func`, by free variable name."""
    func = fingerprint.unwrap(func)
    names = func.__code__.co_freevars
    return {
        name: fingerprint._cell_value(cell)  # pyright: ignore[reportPrivateUsage] - the cell reader fingerprinting uses, so sameness matches it
        for name, cell in zip(names, func.__closure__ or (), strict=True)
    }


def _default_values(func: Callable[..., Any]) -> dict[str, object]:
    """The live default values of `func`, by parameter name."""
    return dict(fingerprint._function_defaults(fingerprint.unwrap(func)))  # pyright: ignore[reportPrivateUsage] - the defaults reader fingerprinting uses


def _show_ref(identity: types.ArtifactIdentity, tag: types.ArtifactTag) -> str:
    key = types.identity_key(identity)
    plain = tag in (types.ArtifactTag.DATA, types.ArtifactTag.INPUT)
    return key if plain else f"{key} ({tag})"


def _show_items(items: collections.abc.Mapping[str, object]) -> str:
    return ", ".join(f"{name}={value}" for name, value in sorted(items.items())) or "none"


def _format_key(fmt: loaders_mod.Reader[Any] | loaders_mod.Writer[Any]) -> tuple[str, str, str]:
    """A loader as its class's module and qualname plus its canonical config text.

    The config text is what the fingerprint hashes as ``loader:...:config``, so a user
    function in a field (``JSON(empty_factory=make_empty)``) compares by definition and code.
    """
    return (type(fmt).__module__, type(fmt).__qualname__, fingerprint.loader_config_text(fmt))


def _show_format(fmt: loaders_mod.Reader[Any] | loaders_mod.Writer[Any]) -> str:
    """A loader for an error message: its ``repr``, with each function field shown by qualname
    plus closure and default values (``transform=make_scaler.<locals>.scaler(factor=2)``).

    A function's ``repr`` holds a memory address, so two loaders that differ only in the
    closure values of a field would read the same apart from that address.
    """
    fields = dataclasses.fields(typing.cast("Any", fmt))
    if not any(inspect.isfunction(getattr(fmt, field.name)) for field in fields):
        return repr(fmt)
    shown = list[str]()
    for field in fields:
        value = getattr(fmt, field.name)
        if not field.repr:
            continue
        if inspect.isfunction(value):
            bound = {**_cell_values(value), **_default_values(value)}
            values = ", ".join(f"{name}={item!r}" for name, item in bound.items())
            shown.append(f"{field.name}={value.__qualname__}({values})")
        else:
            shown.append(f"{field.name}={value!r}")
    return f"{type(fmt).__qualname__}({', '.join(shown)})"


def _stage_wiring(info: registry.RegistryStageInfo) -> list[tuple[str, object, str]]:
    """Each part of a stage besides its function, as (label, comparison key, display).

    Every key survives importing the pipeline file again. Params compare by class module and
    qualname plus their canonical JSON text (``parameters.encode_params``, the encoding recorded
    in the lock file: set members sorted so a deep-copied set matches, secrets by digest so two
    secrets are told apart) with NaN written as ``NaN`` so a re-imported NaN matches. Pydantic
    equality needs one class object. Formats compare through ``_format_key``: the caller picks
    an input's format. Python types are left out: nothing at run time reads them, and a
    re-imported user type is a new object.
    """
    deps = {name: (ref.identity, ref.tag) for name, ref in info["deps"].items()}
    dep_formats = {name: _format_key(ref.format) for name, ref in info["deps"].items()}
    outs = [(ref.identity, ref.tag) for ref in info["outs"]]
    out_formats = [_format_key(ref.format) for ref in info["outs"]]
    incremental = info["incremental_inputs"]
    params = info["params"]
    params_key = (
        None
        if params is None
        else (
            type(params).__module__,
            type(params).__qualname__,
            json.dumps(parameters.encode_params(params), sort_keys=True),
        )
    )
    return [
        ("inputs", deps, _show_items({name: _show_ref(*ref) for name, ref in deps.items()})),
        (
            "input formats",
            dep_formats,
            _show_items({name: _show_format(ref.format) for name, ref in info["deps"].items()}),
        ),
        ("input collections", info["collection_params"], _show_items(info["collection_params"])),
        ("outputs", outs, ", ".join(_show_ref(*ref) for ref in outs) or "none"),
        (
            "output formats",
            out_formats,
            ", ".join(_show_format(ref.format) for ref in info["outs"]),
        ),
        (
            "incremental inputs",
            incremental,
            _show_items({name: types.identity_key(out) for name, out in incremental.items()}),
        ),
        ("params", params_key, "none" if params is None else repr(params)),
        ("resources", info["resources"], _show_items(info["resources"])),
        ("variants", info["variant"], info["variant"] or "none"),
        ("no_fingerprint", info["no_fingerprint"], str(info["no_fingerprint"])),
        ("code_deps", info["code_deps"], ", ".join(info["code_deps"]) or "none"),
    ]


def _stage_difference(
    existing: registry.RegistryStageInfo,
    incoming: registry.RegistryStageInfo,
    root: pathlib.Path,
) -> str | None:
    """Describe the first part that makes two infos different stages, or None if they match."""
    first = _stage_definition(existing["func"])
    again = _stage_definition(incoming["func"])
    first_file = _shown_file(existing["func"], root)
    again_file = _shown_file(incoming["func"], root)
    first_name, again_name = first["qualname"], again["qualname"]
    same_file = (first["file"], first["module"]) == (again["file"], again["module"])
    if same_file and first_name != again_name:
        return f"functions: {first_name} and {again_name}, both in {first_file}"
    if not same_file:
        return f"functions: {first_name} in {first_file} and {again_name} in {again_file}"
    both = f"both calling {first_name} in {first_file}"
    if first["code"] != again["code"]:
        return f"function code, {both}"
    # Compare canonical JSON text, as the manifest cache key does: a NaN made by importing the
    # file again is a new float, and `nan != nan`, but both are written `NaN`.
    bound = (
        ("closure values", first["cells"], again["cells"], _cell_values),
        ("default values", first["defaults"], again["defaults"], _default_values),
    )
    for label, first_bound, again_bound, live_values in bound:
        first_values = {name: json.dumps(value) for name, value in first_bound}
        again_values = {name: json.dumps(value) for name, value in again_bound}
        if first_values != again_values:
            changed = sorted(
                name for name in first_values if first_values[name] != again_values.get(name)
            )
            first_live = live_values(existing["func"])
            again_live = live_values(incoming["func"])
            first_shown = ", ".join(f"{name}={first_live[name]!r}" for name in changed)
            again_shown = ", ".join(f"{name}={again_live[name]!r}" for name in changed)
            return f"{label} ({first_shown} vs {again_shown}), {both}"
    for (label, first_key, first_shown), (_, again_key, again_shown) in zip(
        _stage_wiring(existing), _stage_wiring(incoming), strict=True
    ):
        if first_key != again_key:
            if first_shown == again_shown:
                first_shown, again_shown = repr(first_key), repr(again_key)
            return f"{label} ({first_shown} vs {again_shown}), {both}"
    return None


def _require_same_stage(
    stage_name: str,
    existing: registry.RegistryStageInfo,
    incoming: registry.RegistryStageInfo,
    root: pathlib.Path,
) -> None:
    """Allow one stage reached twice; refuse two different stages under one name.

    A stage name is its lock file and output paths, so two different stages under one name
    would share them. Two infos are one stage when they run the same function
    (`fingerprint.function_identity`: file, qualname, code, closure values and defaults) with
    the same wiring (``_stage_wiring``). Including one pipeline twice, a diamond of includes,
    and a pipeline file imported again by ``pivot repro <path>`` all match.
    """
    difference = _stage_difference(existing, incoming, root)
    if difference is None:
        return
    raise exceptions.PipelineConfigError(
        f"Stage '{stage_name}' is defined twice with different {difference}. "
        + "Pipeline names must be unique within a project; rename one of the pipelines."
    )


def _require_same_template(
    name: str, existing: registry.MapTemplate, incoming: registry.MapTemplate, root: pathlib.Path
) -> None:
    """Allow one template reached twice; refuse two different ones under one name, as
    ``_require_same_stage`` does for stages."""
    _require_same_stage(name, existing["info"], incoming["info"], root)
    if existing["source"] != incoming["source"]:
        raise exceptions.PipelineConfigError(
            f"Stage '{name}' is defined twice, mapped over "
            + f"'{types.identity_key(existing['source'])}' and "
            + f"'{types.identity_key(incoming['source'])}'. Pipeline names must be unique "
            + "within a project; rename one of the pipelines."
        )


def _require_one_state_dir(
    including: str,
    included: str,
    root: pathlib.Path,
    holder: str,
    held: state_dir_mod.StateDirResolution,
    incoming: state_dir_mod.StateDirResolution,
) -> None:
    """Refuse to combine pipelines that keep one project's state in two directories.

    Each pipeline resolves ``core.state_dir`` when it loads, the ``.pivot`` fallback for a value
    or a path it names that cannot be used included, and records the outcome on its stages.
    When the value, a link it names, or whether the path it names can be used changed between
    two loads, a run of the combined pipeline would put some stages' locks in one directory and
    its run record and caches in the other. So the outcomes are compared, not the values.
    Inside one discovery or one CLI invocation every pipeline loads under one record and
    resolves one directory, so ``--all`` and ``repro`` of several pipeline files are unaffected.
    """
    held_dir = state_dir_mod.resolved_state_dir(root, held)
    incoming_dir = state_dir_mod.resolved_state_dir(root, incoming)
    if held_dir == incoming_dir:
        return
    raise exceptions.PipelineConfigError(
        f"Pipeline '{including}' cannot include '{included}': '{holder}' keeps the state of "
        + f"the project at {root} in {held_dir}, and '{included}' in {incoming_dir}. "
        + "core.state_dir, or the path it names, changed between their loads; load both "
        + "pipelines after the change."
    )


def _emit_stage_info(
    node: _StageNode,
    qualified_name: str,
    state_dir: pathlib.Path,
) -> registry.RegistryStageInfo:
    func = node.original_func
    deps = dict[str, types.ArtifactRef]()
    input_paths = dict[str, str]()

    def add_dep(param_name: str, handle: ArtifactHandle) -> None:
        ref = _handle_to_artifact_ref(handle, qualified_name)
        deps[param_name] = ref
        if ref.tag is types.ArtifactTag.INPUT:
            source = handle._source  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            assert isinstance(source, _InputNode)
            input_paths[types.identity_key(ref.identity)] = os.path.normpath(source.path)

    for param_name, handle in node.input_handles.items():
        add_dep(param_name, handle)

    for param_name, handles in node.list_input_handles.items():
        for i, handle in enumerate(handles):
            add_dep(f"{param_name}[{i}]", handle)

    outs = list[types.ArtifactRef]()

    for output_spec in node.output_specs:
        output_key = None if output_spec.key == SINGLE_OUTPUT_KEY else output_spec.key
        identity = types.ArtifactIdentity(producer=qualified_name, key=output_key)
        types.validate_artifact_identity(identity)
        outs.append(
            types.ArtifactRef(
                identity=identity,
                format=output_spec.format,
                python_type=output_spec.python_type,
                tag=_artifact_tag(output_spec.tag),
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
                        + f"'{base_hint}' — StageParams must not be in a union. "
                        + f"Use 'params: {union_arg.__name__}' directly, with a "
                        + f"default if needed (params: {union_arg.__name__} = "
                        + f"{union_arg.__name__}())."
                    )
        if isinstance(base_hint, type) and issubclass(base_hint, stage_def.StageParams):
            params_arg_name = name
            if node.params is not None and not isinstance(node.params, base_hint):
                raise exceptions.ParamsError(
                    f"Stage '{qualified_name}': params type {type(node.params).__name__} "
                    + f"does not match parameter '{name}: {base_hint.__name__}'"
                )
            break

    output_identities = {
        spec.key: out.identity for spec, out in zip(node.output_specs, outs, strict=True)
    }
    incremental_inputs = {
        param_name: output_identities[output_key]
        for param_name, output_key in node.incremental_inputs.items()
    }

    return registry.RegistryStageInfo(
        func=func,
        name=qualified_name,
        deps=deps,
        outs=outs,
        incremental_inputs=incremental_inputs,
        params=node.params,
        resources=dict(node.resources),
        variant=node.variant,
        signature=inspect.signature(func),
        fingerprint=None,
        params_arg_name=params_arg_name,
        state_dir=state_dir,
        collection_params={k: str(v) for k, v in node.collection_params.items()},
        no_fingerprint=node.no_fingerprint,
        code_deps=list(node.code_deps),
        input_paths=input_paths,
        unsafe_fingerprint=False,
    )


def _register_node(
    reg: registry.StageRegistry, node: _StageNode, info: registry.RegistryStageInfo
) -> None:
    """Add a stage to ``reg``, or a template when the stage call was mapped."""
    if node.map_source is None:
        reg.add_existing(info)
        return
    mapped_deps = dict[str, str | None]()

    def note(dep_name: str, handle: ArtifactHandle) -> None:
        if handle._map is None:  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            return
        if handle._is_map_entry():  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            mapped_deps[dep_name] = None
        else:
            mapped_deps[dep_name] = f"{handle._pipeline._name}/{handle._source.name}"  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle and Pipeline

    for param_name, handle in node.input_handles.items():
        note(param_name, handle)
    for param_name, handles in node.list_input_handles.items():
        for i, handle in enumerate(handles):
            note(f"{param_name}[{i}]", handle)
    reg.add_template(
        registry.MapTemplate(info=info, source=node.map_source.identity(), mapped_deps=mapped_deps)
    )


class Pipeline:
    _name: str
    _root: pathlib.Path
    _stages: list[_StageNode]
    _inputs: dict[str, _InputNode]
    _input_names_by_path: dict[str, str]
    _call_counts: dict[tuple[str, tuple[str, ...]], int]
    _validation_errors: list[str]
    _token: contextvars.Token[Pipeline | None] | None
    _variant_stack: list[str]
    _registry: registry.StageRegistry
    _included_stages: dict[str, registry.RegistryStageInfo]
    _included_templates: dict[str, registry.MapTemplate]
    _previous_markers: list[_PreviousOutputMarker]
    _constructing: bool
    _loaded_pipelines: set[str]
    # Each project root's resolution: this pipeline's load, then each included pipeline's.
    _resolved_state_dirs: dict[pathlib.Path, state_dir_mod.StateDirResolution]
    # The resolutions the included pipelines brought, with the name of the first to bring each.
    _included_state_dirs: dict[pathlib.Path, tuple[str, state_dir_mod.StateDirResolution]]

    def __init__(self, name: str, *, root: pathlib.Path | None = None) -> None:
        reason = lock.stage_name_error(name)
        if reason is not None:
            raise ValueError(f"Invalid pipeline name {name!r}: {reason}")
        self._name = name
        self._stages = []
        self._inputs = {}
        self._input_names_by_path = dict[str, str]()
        self._call_counts = {}
        self._validation_errors = []
        self._token = None
        self._variant_stack = list[str]()
        self._registry = registry.StageRegistry()
        self._previous_markers = list[_PreviousOutputMarker]()
        self._included_stages = dict[str, registry.RegistryStageInfo]()
        self._included_templates = dict[str, registry.MapTemplate]()
        self._constructing = False
        self._loaded_pipelines = {name}
        self._resolved_state_dirs = dict[pathlib.Path, state_dir_mod.StateDirResolution]()
        self._included_state_dirs = dict[
            pathlib.Path, tuple[str, state_dir_mod.StateDirResolution]
        ]()

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
        return state_dir_mod.ensure_state_dir(self._root)

    @property
    def resolved_state_dirs(self) -> Mapping[pathlib.Path, state_dir_mod.StateDirResolution]:
        """Each project root's ``core.state_dir``, as this pipeline resolved it when it loaded.

        Its stages' locks are recorded in these directories (``state_dir.resolved_state_dir``).
        Its fingerprints and its runs start their state-directory record with them
        (``state_dir.entry_invocation``), so a ``core.state_dir`` edited after the load leaves
        its fingerprints in one directory. A run (``executor.run(pipeline=...)``) refuses once
        the directory the configuration names differs from these
        (``discovery.require_current_state_dirs``): its locks would go where ``pivot gc`` no
        longer reads. A caller who wants the new directory loads the pipeline again. A value that
        could not be used is kept as its error, so the entry points fall back to ``.pivot`` (or
        fail) as at the load. A path that could not become a directory is kept as its fallback
        (``state_dir.kept_by_pipeline``): the stages' locks are in ``.pivot``, so every reader in
        its fingerprints uses ``.pivot`` too, and a run refuses once the path can be used.
        Read-only: ``include()`` alone adds to it.
        """
        return _stdlib_types.MappingProxyType(self._resolved_state_dirs)

    def _stage_state_dir(self) -> pathlib.Path:
        """The state directory recorded on this pipeline's stages; loading never creates it.

        While the project config cannot be read, its ``core.state_dir`` cannot be used, or the
        path it names cannot become a directory, the stages keep 0.5.2's ``.pivot``
        (``state_dir.state_dir_or_default``), so listing them runs as it did there; commands
        that use the configured directory fail on it themselves. The outcome is the load's
        record's, which the pipeline keeps (``resolved_state_dirs``).
        """
        return state_dir_mod.state_dir_or_default(self._root)

    @property
    def input_bindings(self) -> dict[str, str]:
        return {name: node.path for name, node in self._inputs.items()}

    @property
    def loaded_pipelines(self) -> frozenset[str]:
        """Names of the pipelines whose every stage this one holds: itself and each it includes.

        A pipeline reached only through handles contributes just the stages it produces those
        handles with, so it is not listed.
        """
        return frozenset(self._loaded_pipelines)

    def _require_materialized(self) -> None:
        if self._constructing:
            raise RuntimeError(
                "Cannot call engine-facing methods on a Pipeline that is still under "
                + "construction (inside a 'with' block). Exit the 'with' block first."
            )

    def list_stages(self) -> list[str]:
        self._require_materialized()
        return self._registry.list_stages()

    def get_stage(self, name: str) -> registry.RegistryStageInfo:
        self._require_materialized()
        return self._registry.get(name)

    def list_templates(self) -> list[str]:
        """The stages mapped over a directory output (``map``), not yet expanded."""
        self._require_materialized()
        return self._registry.list_templates()

    def get_template(self, name: str) -> registry.MapTemplate:
        self._require_materialized()
        return self._registry.get_template(name)

    def expand_template(
        self, name: str, keys: collections.abc.Sequence[tuple[str, str]]
    ) -> list[str]:
        """Add template ``name``'s instances (``registry.StageRegistry.expand``)."""
        self._require_materialized()
        return self._registry.expand(name, keys)

    def instances_of(self, name: str) -> list[str]:
        self._require_materialized()
        return self._registry.instances_of(name)

    def reset_expansions(self) -> None:
        """Remove every instance: each run expands its templates afresh."""
        self._require_materialized()
        self._registry.reset_expansions()

    def ensure_fingerprint(self, name: str) -> dict[str, str]:
        self._require_materialized()
        with state_dir_mod.entry_invocation(self._resolved_state_dirs):
            return self._registry.ensure_fingerprint(name)

    def build_dag(self) -> DiGraph[str]:
        self._require_materialized()
        return self._registry.build_dag()

    def invalidate_dag_cache(self) -> None:
        self._registry.invalidate_dag_cache()

    def snapshot(self) -> dict[str, registry.RegistryStageInfo]:
        self._require_materialized()
        return self._registry.snapshot()

    def include(self, other: registry.PipelineLike) -> None:
        """Add ``other``'s stages, templates and input bindings to this pipeline.

        All or nothing: the include works on copies of this pipeline's records, and outside a
        construction block on a copy of its registry, which it keeps only once every check has
        passed. A refused include (a conflicting stage, template, input binding, output path or
        state directory) leaves the pipeline as it was.
        """
        if other is self:
            from pivot.exceptions import PipelineConfigError

            raise PipelineConfigError(f"Pipeline '{self.name}' cannot include itself")
        kept = (
            self._included_stages,
            self._included_templates,
            self._inputs,
            self._input_names_by_path,
            self._loaded_pipelines,
            self._included_state_dirs,
            self._resolved_state_dirs,
            self._registry,
        )
        self._included_stages = dict(self._included_stages)
        self._included_templates = dict(self._included_templates)
        self._inputs = dict(self._inputs)
        self._input_names_by_path = dict(self._input_names_by_path)
        self._loaded_pipelines = set(self._loaded_pipelines)
        self._included_state_dirs = dict(self._included_state_dirs)
        self._resolved_state_dirs = dict(self._resolved_state_dirs)
        if not self._constructing:
            self._registry = self._registry.copy()
        try:
            self._include(other)
        except BaseException:
            (
                self._included_stages,
                self._included_templates,
                self._inputs,
                self._input_names_by_path,
                self._loaded_pipelines,
                self._included_state_dirs,
                self._resolved_state_dirs,
                self._registry,
            ) = kept
            raise

    def _include(self, other: registry.PipelineLike) -> None:
        """``include``'s checks and changes, on the copies it made."""
        # One project's state stays in one directory; checked before anything changes.
        for root, resolution in other.resolved_state_dirs.items():
            held = self._held_state_dir(root)
            if held is not None:
                holder, current = held
                _require_one_state_dir(self._name, other.name, root, holder, current, resolution)

        # Own stages are checked against included ones at materialization.
        registered = self._included_stages if self._constructing else self._registry.snapshot()
        for stage_name in other.list_stages():
            incoming = other.get_stage(stage_name)
            existing = registered.get(stage_name)
            if existing is not None:
                _require_same_stage(stage_name, existing, incoming, self._root)
                continue
            stage_info = copy.deepcopy(incoming)
            self._included_stages[stage_name] = stage_info
            if not self._constructing:
                self._registry.add_existing(copy.deepcopy(stage_info))

        known_templates = (
            self._included_templates
            if self._constructing
            else {
                name: self._registry.get_template(name) for name in self._registry.list_templates()
            }
        )
        for template_name in other.list_templates():
            incoming_template = other.get_template(template_name)
            existing_template = known_templates.get(template_name)
            if existing_template is not None:
                _require_same_template(
                    template_name, existing_template, incoming_template, self._root
                )
                continue
            template = copy.deepcopy(incoming_template)
            self._included_templates[template_name] = template
            if not self._constructing:
                self._registry.add_template(copy.deepcopy(template))

        for name, path in other.input_bindings.items():
            if name in self._inputs:
                existing_path = self._inputs[name].path
                if existing_path != path:
                    from pivot.exceptions import PipelineConfigError

                    raise PipelineConfigError(
                        f"Input binding conflict: input '{name}' is bound to "
                        + f"'{existing_path}' and '{path}' in different pipelines."
                    )
            else:
                self._add_input(
                    _InputNode(
                        name=name,
                        python_type=None,
                        path=path,
                        format=loaders_mod.PathOnly(),
                        inherited=True,
                    )
                )
        if not self._constructing:
            self._check_inputs_against_outputs(self._registry)
        self._loaded_pipelines.update(other.loaded_pipelines)
        for root, resolution in other.resolved_state_dirs.items():
            self._included_state_dirs.setdefault(root, (other.name, resolution))
            if not self._constructing:
                self._resolved_state_dirs.setdefault(root, resolution)
        self._registry.invalidate_dag_cache()

    def _held_state_dir(
        self, root: pathlib.Path
    ) -> tuple[str, state_dir_mod.StateDirResolution] | None:
        """The resolution this pipeline already holds for ``root``, and who it came from.

        Inside its ``with`` block the pipeline has not loaded yet, so only the pipelines it
        included hold one; its load checks them against its own (``_materialize_stages``).
        """
        if self._constructing:
            return self._included_state_dirs.get(root)
        if root in self._resolved_state_dirs:
            return self._name, self._resolved_state_dirs[root]
        return None

    def __enter__(self) -> Pipeline:
        if self._token is not None:
            raise RuntimeError(
                f"Pipeline '{self._name}' is already inside a 'with' block. "
                + "Nested 'with' on the same Pipeline is not supported."
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
        # One record for the load: each project root's state directory is read and resolved
        # once, recorded on its stages, and kept for this pipeline's fingerprints and runs. A
        # library caller is told once per load that a path it cannot use sends them to `.pivot`.
        with state_dir_mod.entry_invocation() as resolved:
            stage_state_dirs = {self._root: self._stage_state_dir()}
            for foreign_pipeline, _ in closure:
                if foreign_pipeline._root not in stage_state_dirs:
                    stage_state_dirs[foreign_pipeline._root] = foreign_pipeline._stage_state_dir()
            for root in stage_state_dirs:
                state_dir_mod.report_unusable_path_to_library(root, resolved[root])
            resolved_state_dirs = {
                root: state_dir_mod.kept_by_pipeline(resolved[root]) for root in stage_state_dirs
            }
        for root, (included, resolution) in self._included_state_dirs.items():
            if root in resolved_state_dirs:
                _require_one_state_dir(
                    self._name, included, root, self._name, resolved_state_dirs[root], resolution
                )

        for foreign_pipeline, foreign_node in closure:
            qualified_name = f"{foreign_pipeline._name}/{foreign_node.name}"
            stage_info = _emit_stage_info(
                foreign_node, qualified_name, stage_state_dirs[foreign_pipeline._root]
            )
            _register_node(new_registry, foreign_node, stage_info)

        self._merge_foreign_input_bindings(closure)

        for node in self._stages:
            qualified_name = f"{self._name}/{node.name}"
            stage_info = _emit_stage_info(node, qualified_name, stage_state_dirs[self._root])
            _register_node(new_registry, node, stage_info)

        materialized_stages = set(new_registry.list_stages())
        for stage_name, stage_info in self._included_stages.items():
            if stage_name in materialized_stages:
                _require_same_stage(
                    stage_name, new_registry.get(stage_name), stage_info, self._root
                )
                continue
            new_registry.add_existing(copy.deepcopy(stage_info))
            materialized_stages.add(stage_name)
        materialized_templates = set(new_registry.list_templates())
        for template_name, template in self._included_templates.items():
            if template_name in materialized_templates:
                _require_same_template(
                    template_name, new_registry.get_template(template_name), template, self._root
                )
                continue
            new_registry.add_template(copy.deepcopy(template))

        self._check_inputs_against_outputs(new_registry)

        # Atomic swap — if any step above raises, self._registry is unchanged
        self._registry = new_registry
        for root, (_, resolution) in self._included_state_dirs.items():
            resolved_state_dirs.setdefault(root, resolution)
        self._resolved_state_dirs = resolved_state_dirs

    def _check_inputs_against_outputs(self, reg: registry.StageRegistry) -> None:
        """Refuse an input named like a stage, or whose path overlaps a registered output.

        An input's identity is its name and a stage's single output's identity is the stage
        name, so one name cannot be both. Runs once the registry holds every output: at
        materialization and after an include() outside a construction block (how --all
        combines pipelines).
        """
        stage_names = set(reg.list_stages())
        for node in self._inputs.values():
            if node.name in stage_names:
                raise exceptions.PipelineConfigError(
                    f"Input '{node.name}' has the name of stage '{node.name}'. "
                    + "Rename the input, or pass the stage's handle to read its output."
                )
            normalized = pathlib.Path(os.path.normpath(self._root / node.path))
            if not normalized.is_relative_to(self._root):
                continue  # path_policy keeps inputs in the project; outputs always are
            reg.check_input_path(node.name, normalized.relative_to(self._root).parts)

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
        # The store reads input bindings relative to the project root, so the path must stay
        # inside it: no control characters, no '..', no absolute path elsewhere.
        path_policy.require_valid_path(
            path, path_policy.PathType.INPUT, self._root, context=f"Input '{name}'"
        )
        if format is None:
            if t is not None:
                format = _infer_input_format_from_type(t)
            if format is None:
                format = _infer_format_from_extension(path)
        existing = self._inputs.get(name)
        if existing is not None:
            if existing.path != path:
                raise exceptions.PipelineConfigError(
                    f"Input binding conflict: input '{name}' is bound to '{existing.path}' "
                    + f"and '{path}' in the same pipeline."
                )
            if existing.inherited:
                existing.python_type = t
                existing.format = format
                existing.inherited = False
                return ArtifactHandle(
                    pipeline=self,
                    source=existing,
                    output_key=None,
                    python_type=t or object,
                )
            if existing.format != format:
                raise exceptions.PipelineConfigError(
                    f"Input binding conflict: input '{name}' at '{path}' changes format from "
                    + f"{type(existing.format).__name__} to {type(format).__name__}."
                )
            return ArtifactHandle(
                pipeline=self,
                source=existing,
                output_key=None,
                python_type=existing.python_type or object,
            )
        node = _InputNode(name=name, python_type=t, path=path, format=format)
        self._add_input(node)
        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=t or object,
        )

    def _add_input(self, node: _InputNode) -> None:
        """Bind a new input name, refusing a second name for a file another input reads.

        The workspace store maps each input identity to its path, so two names on one
        file would collide there mid-run; this rejects them while the pipeline is defined.
        """
        path_key = os.path.normpath(self._root / node.path)
        other = self._input_names_by_path.get(path_key)
        if other is not None:
            other_path = self._inputs[other].path
            spelled = "" if other_path == node.path else f" (as '{node.path}')"
            raise exceptions.PipelineConfigError(
                f"Input binding conflict: inputs '{other}' and '{node.name}' are both bound to "
                + f"'{other_path}'{spelled}. Use one input name per file."
            )
        self._inputs[node.name] = node
        self._input_names_by_path[path_key] = node.name

    @contextlib.contextmanager
    def variant(self, name: str) -> collections.abc.Generator[None]:
        """Context manager for registering stages with a variant suffix.

        All stages registered within the block get @{name} appended to their stage name.
        Nesting is supported: variant("a") inside variant("b") → @b@a suffix.
        """
        reason = lock.variant_name_error(name)
        if reason is not None:
            raise ValueError(f"Invalid variant name {name!r}: {reason}")
        self._variant_stack.append(name)
        try:
            yield
        finally:
            self._variant_stack.pop()

    def _record_stage(
        self,
        original_func: Callable[..., object],
        stage_wrapper: Callable[..., object],
        resources: dict[str, int | str],
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

        qualified_name = f"{self._name}/{stage_name}"
        reason = lock.stage_name_error(qualified_name)
        if reason is not None:
            raise ValueError(f"Invalid stage name {qualified_name!r}: {reason}")
        # Refuse a stage with @fingerprint_by_name anywhere on its `__wrapped__` chain (this
        # @stage wrapper, the function @stage received, and each object below that one, read as
        # every chain is, by `fingerprint.wrapper_chain`), before any read of the stage: the mark
        # is for helpers, and a stage's own code is fingerprinted.
        if fingerprint.fingerprinted_by_name(
            fingerprint.unwrap(stage_wrapper, stop=fingerprint.fingerprinted_by_name)
        ):
            self._validation_errors.append(
                f"{qualified_name}: @fingerprint_by_name marks a helper, not a stage. "
                + "Use @no_fingerprint() to opt a stage out of AST fingerprinting."
            )
            return self._error_handle()
        output_specs = _analyze_return_type(original_func, qualified_name)

        sig = inspect.signature(original_func)
        try:
            bound = sig.bind(*args, **kwargs)
            explicit_params = set(bound.arguments.keys())
            bound.apply_defaults()
        except TypeError as exc:
            self._validation_errors.append(f"{stage_name}: {exc}")
            return self._error_handle()

        for parameter in sig.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                self._validation_errors.append(
                    f"{qualified_name}: parameter '{parameter.name}' uses *args; "
                    + "use a list[...] parameter instead of *args."
                )
            elif parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                self._validation_errors.append(
                    f"{qualified_name}: parameter '{parameter.name}' is positional-only; remove the /."
                )
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                self._validation_errors.append(
                    f"{qualified_name}: parameter '{parameter.name}' uses **kwargs; "
                    + "use named parameters instead."
                )

        input_handles = dict[str, ArtifactHandle]()
        list_input_handles = dict[str, list[ArtifactHandle]]()
        collection_params = dict[str, CollectionKind]()
        incremental_inputs = dict[str, str]()
        params: stage_def.StageParams | None = None
        for param_name, value in bound.arguments.items():
            if isinstance(value, _PreviousOutputMarker):
                if value.pipeline is not self:
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' received p.previous() from "
                        + f"Pipeline '{value.pipeline.name}', not this stage call."
                    )
                    continue
                if value.consumed_by is not None:
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' reuses p.previous() already "
                        + f"bound to {value.consumed_by}."
                    )
                    continue
                value.consumed_by = stage_name
                if value.key is None:
                    if len(output_specs) != 1:
                        self._validation_errors.append(
                            f"{stage_name}: parameter '{param_name}' uses p.previous() but "
                            + "the stage has multiple outputs; name an incremental output key."
                        )
                        continue
                    output_spec = output_specs[0]
                else:
                    output_spec = _select_output(output_specs, value.key)
                    if output_spec is None:
                        self._validation_errors.append(
                            f"{stage_name}: parameter '{param_name}' uses p.previous({value.key!r}) "
                            + f"but '{value.key}' is not an output key."
                        )
                        continue
                if not isinstance(output_spec.tag, _IncrementalTag):
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' uses p.previous("
                        + (f"{value.key!r}" if value.key is not None else "")
                        + f") but {_output_label(output_spec.key)} is not tagged incremental."
                    )
                    continue
                existing_param = next(
                    (
                        existing
                        for existing, output_key in incremental_inputs.items()
                        if output_key == output_spec.key
                    ),
                    None,
                )
                if existing_param is not None:
                    label = _output_label(output_spec.key)
                    self._validation_errors.append(
                        f"{stage_name}: parameters '{existing_param}' and '{param_name}' both "
                        + f"bind p.previous() to {label}, an incremental output."
                    )
                    continue
                incremental_inputs[param_name] = output_spec.key
            elif isinstance(value, ArtifactHandle):
                input_handles[param_name] = value
            elif isinstance(value, (list, tuple)):
                collection = typing.cast("list[object] | tuple[object, ...]", value)
                handles = [v for v in collection if isinstance(v, ArtifactHandle)]
                if len(handles) == len(collection):
                    list_input_handles[param_name] = handles
                    collection_params[param_name] = (
                        CollectionKind.TUPLE
                        if isinstance(collection, tuple)
                        else CollectionKind.LIST
                    )
                elif param_name in explicit_params:
                    self._validation_errors.append(
                        f"{stage_name}: parameter '{param_name}' has unsupported type "
                        + f"'{type(collection).__name__}'. Use ArtifactHandle (dependency), "
                        + "StageParams (config), or list/tuple of ArtifactHandle."
                    )
            elif isinstance(value, stage_def.StageParams):
                params = value
            elif param_name in explicit_params:
                self._validation_errors.append(
                    f"{stage_name}: parameter '{param_name}' has unsupported type "
                    + f"'{type(value).__name__}'. Use ArtifactHandle (dependency), "
                    + "StageParams (config), or list/tuple of ArtifactHandle."
                )

        variant_str = "@".join(self._variant_stack) if self._variant_stack else None
        map_source = self._call_map_source(
            qualified_name,
            [*input_handles.values(), *(h for hs in list_input_handles.values() for h in hs)],
        )

        # Read @no_fingerprint marks from this @stage wrapper, never from the shared function:
        # functools.wraps copies marks set below @stage, and marks set above land here directly.
        node = _StageNode(
            original_func=original_func,
            no_fingerprint=getattr(stage_wrapper, "__pivot_no_fingerprint__", False),
            code_deps=list(getattr(stage_wrapper, "__pivot_code_deps__", [])),
            name=stage_name,
            params=params,
            input_handles=input_handles,
            output_specs=output_specs,
            call_index=count,
            variant=variant_str,
            resources=resources,
            list_input_handles=list_input_handles,
            collection_params=collection_params,
            incremental_inputs=incremental_inputs,
            map_source=map_source,
        )
        self._stages.append(node)

        return ArtifactHandle(
            pipeline=self,
            source=node,
            output_key=None,
            python_type=output_specs[0].python_type if len(output_specs) == 1 else dict,
            map_source=map_source,
        )

    def _call_map_source(
        self, qualified_name: str, handles: list[ArtifactHandle]
    ) -> _MapSource | None:
        """The directory a stage call is mapped over: the one its mapped handles share, or None
        when it has none. Handles mapped over two directories are a validation error."""
        sources = [handle._map for handle in handles if handle._map is not None]  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
        if not sources:
            return None
        first = sources[0]
        other = next((source for source in sources if not source.same(first)), None)
        if other is not None:
            self._validation_errors.append(
                f"{qualified_name}: mapped over both '{types.identity_key(first.identity())}' "
                + f"and '{types.identity_key(other.identity())}'; a stage maps over the files "
                + "of one directory output."
            )
        return first

    def _error_handle(self) -> ArtifactHandle:
        """A placeholder handle for a stage call refused with a validation error."""
        dummy_input = _InputNode(
            name="__error__",
            python_type=None,
            path="__error__",
            format=loaders_mod.PathOnly(),
        )
        return ArtifactHandle(self, dummy_input, None, object)

    def previous(self, key: str | None = None) -> Any:
        """Mark a parameter as this stage call's prior incremental output."""
        if not self._constructing:
            raise ValueError("p.previous() may only be used inside a Pipeline stage call.")
        marker = _PreviousOutputMarker(pipeline=self, key=key)
        self._previous_markers.append(marker)
        return marker

    def map(self, handle: object) -> Any:
        """Stand for each file of a directory output, one at a time.

        A stage called with the returned handle is mapped: a run that finds the directory's
        producer current adds one instance of it per file, named ``<stage>@<key>``, where
        ``key`` is the file's path inside the directory without its extension. Each instance
        depends on its own file only. A stage called with a mapped stage's handle is mapped
        over the same files, its instance of each key depending on that key's instance.
        """
        if not isinstance(handle, ArtifactHandle):
            raise exceptions.PipelineConfigError(
                "pipeline.map() takes what a stage call returned (or one of its fields), "
                + f"not a {type(handle).__name__}."
            )
        if not isinstance(handle._source, _StageNode):  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            raise exceptions.PipelineConfigError(
                "pipeline.map() takes a directory output of a stage; "
                + f"'{handle._source.name}' is an input."  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            )
        source = handle._source  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
        qualified_name = f"{handle._pipeline._name}/{source.name}"  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle and Pipeline
        if handle._map is not None:  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            raise exceptions.PipelineConfigError(
                f"pipeline.map() cannot map a handle of stage '{qualified_name}': it is already "
                + "mapped over "
                + f"'{types.identity_key(handle._map.identity())}', and a map over a mapped "  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                + "stage's output is not supported."
            )
        if handle._member is not None:  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            raise exceptions.PipelineConfigError(
                f"pipeline.map() takes a whole directory output; {handle._member!r} is one file "  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                + f"of a directory output of stage '{qualified_name}'."
            )
        spec = handle._selected_directory()  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
        if spec is None:
            key = handle._output_key  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            shown = "the return value" if key is None else _output_label(key)
            raise exceptions.PipelineConfigError(
                f"pipeline.map() needs a directory output: {shown} of stage '{qualified_name}' "
                + "is not a directory output. Tag a dict[str, T] return value with `directory`, "
                + "or select a directory field of a multi-output stage (handle.field)."
            )
        map_source = _MapSource(
            pipeline=handle._pipeline,  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            node=source,
            output_key=None if spec.key == SINGLE_OUTPUT_KEY else spec.key,
        )
        return ArtifactHandle(
            handle._pipeline,  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            source,
            handle._output_key,  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            get_args(spec.python_type)[1],
            map_source=map_source,
        )

    def _validate(self) -> None:
        for marker in self._previous_markers:
            if marker.consumed_by is None:
                self._validation_errors.append(
                    "p.previous() must be passed as an argument to a stage call."
                )
        if self._validation_errors:
            msg = (
                f'Pipeline "{self._name}" has {len(self._validation_errors)} '
                + "validation error(s):\n\n"
            )
            for err in self._validation_errors:
                msg += f"  {err}\n"
            raise exceptions.StageDefinitionError(msg)

    def _upstream_closure(self) -> list[tuple[Pipeline, _StageNode]]:
        """Collect foreign stages reachable from this pipeline's handles via DFS."""
        visited = set[int]()
        result = list[tuple[Pipeline, _StageNode]]()
        stack = list[tuple[Pipeline, _StageNode]]()

        for node in self._stages:
            for handle in _iter_stage_handles(node):
                if handle._pipeline is not self:  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                    source = handle._source  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                    assert isinstance(source, _StageNode)  # guaranteed by _iter_stage_handles
                    stack.append((handle._pipeline, source))  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle

        while stack:
            foreign_pipeline, stage_node = stack.pop()
            node_id = id(stage_node)
            if node_id in visited:
                continue
            visited.add(node_id)
            result.append((foreign_pipeline, stage_node))

            for handle in _iter_stage_handles(stage_node):
                if handle._pipeline is not self:  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                    source = handle._source  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                    assert isinstance(source, _StageNode)  # guaranteed by _iter_stage_handles
                    stack.append((handle._pipeline, source))  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle

        return result

    def _merge_foreign_input_bindings(
        self,
        closure: list[tuple[Pipeline, _StageNode]],
    ) -> None:
        foreign_bindings = dict[str, str]()

        def _add_foreign_input(handle: ArtifactHandle) -> None:
            if not isinstance(handle._source, _InputNode):  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                return
            inp = handle._source  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
            if inp.name in foreign_bindings and foreign_bindings[inp.name] != inp.path:
                raise ValueError(
                    f"Input binding conflict: input '{inp.name}' is bound to "
                    + f"'{foreign_bindings[inp.name]}' and '{inp.path}' "
                    + "in different pipelines."
                )
            foreign_bindings[inp.name] = inp.path

        for local_node in self._stages:
            for handle in _iter_all_handles(local_node):
                if handle._pipeline is not self:  # pyright: ignore[reportPrivateUsage] - compose owns ArtifactHandle
                    _add_foreign_input(handle)

        for _foreign_pipeline, foreign_node in closure:
            for handle in _iter_all_handles(foreign_node):
                _add_foreign_input(handle)

        local_bindings = self.input_bindings
        for name, path in foreign_bindings.items():
            if name in local_bindings and local_bindings[name] != path:
                raise ValueError(
                    f"Input binding conflict: input '{name}' is bound to "
                    + f"'{local_bindings[name]}' locally and '{path}' in a foreign pipeline."
                )
            if name not in self._inputs:
                self._add_input(
                    _InputNode(
                        name=name,
                        python_type=None,
                        path=path,
                        format=loaders_mod.PathOnly(),
                        inherited=True,
                    )
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
    defaults = _resolve_default_formats()

    # Unwrap generic aliases (e.g. dict[str, Any] → dict) for lookup and issubclass
    origin = get_origin(python_type)
    lookup_type = origin if origin is not None else python_type

    loader_name = defaults.get(lookup_type)
    if loader_name is not None:
        loader_class = typing.cast(
            "type[loaders_mod.Writer[object]]", getattr(loaders_mod, loader_name)
        )
        return loader_class()

    for registered_type, registered_loader in defaults.items():
        if isinstance(lookup_type, type) and issubclass(lookup_type, registered_type):
            loader_class = typing.cast(
                "type[loaders_mod.Writer[object]]", getattr(loaders_mod, registered_loader)
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
    return loaders_mod.format_extension(fmt)


def _infer_input_format_from_type(t: type) -> loaders_mod.Reader[object] | None:
    if t is pathlib.Path:
        return loaders_mod.PathOnly()  # type: ignore[return-value]
    if t is str:
        return loaders_mod.Text()  # type: ignore[return-value]
    return None


def _infer_format_from_extension(path: str) -> loaders_mod.Reader[object]:
    """Infer loader from file extension for p.input() declarations."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    match ext:
        case "yaml" | "yml":
            return loaders_mod.YAML()  # type: ignore[return-value]
        case "csv":
            return loaders_mod.CSV()  # type: ignore[return-value]
        case "jsonl":
            return loaders_mod.DataFrameJSONL()  # type: ignore[return-value]
        case "json":
            return loaders_mod.JSON()  # type: ignore[return-value]
        case "txt" | "sql" | "jinja":
            return loaders_mod.Text()  # type: ignore[return-value]
        case "pkl" | "pickle":
            return loaders_mod.Pickle()  # type: ignore[return-value]
        case _:
            return loaders_mod.PathOnly()  # type: ignore[return-value]


# --- Return type analysis ---


def _output_label(key: str) -> str:
    """Name an output in a user-facing message.

    Users never see SINGLE_OUTPUT_KEY: a single output is the stage's return value.
    """
    return "the return value" if key == SINGLE_OUTPUT_KEY else f"output '{key}'"


def _parse_output_type(hint: Any, key: str, stage_name: str) -> list[_OutputSpec]:
    """Parse a stage return type into output specifications."""
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
                    + f"Rename the field in {hint.__name__}."
                )
            specs.extend(_parse_output_type(field_hint, field_name, stage_name))
        _refuse_outputs_named_like_directory_files(specs, stage_name)
        return specs

    if get_origin(hint) is not Annotated:
        return [_OutputSpec(key=key, python_type=hint, format=_infer_format(hint))]

    args = get_args(hint)
    base_type = args[0]
    tag: _OutputTag | None = None
    fmt: loaders_mod.Writer[object] | None = None
    label = _output_label(key)
    for metadata in args[1:]:
        if isinstance(metadata, _OutputTag):
            if tag is not None:
                raise exceptions.StageDefinitionError(
                    f"Stage '{stage_name}': {label} cannot combine "
                    + f"{type(tag).__name__.removeprefix('_').removesuffix('Tag').lower()} "
                    + f"with {type(metadata).__name__.removeprefix('_').removesuffix('Tag').lower()}."
                )
            tag = metadata
        elif isinstance(metadata, loaders_mod.Writer):
            if fmt is not None:
                raise exceptions.StageDefinitionError(
                    f"Stage '{stage_name}': {label} has more than one loader."
                )
            fmt = typing.cast("loaders_mod.Writer[object]", metadata)
        else:
            metadata_name = (
                metadata.__name__ if isinstance(metadata, type) else type(metadata).__name__
            )
            raise exceptions.StageDefinitionError(
                f"Stage '{stage_name}': {label} has unsupported Annotated metadata "
                + f"'{metadata_name}'."
            )

    if isinstance(tag, _DirectoryTag):
        file_type = _directory_file_type(base_type, label, stage_name)
        if fmt is None:
            fmt = _infer_format(file_type)
    elif fmt is None:
        if isinstance(tag, _PlotTag):
            fmt = typing.cast("loaders_mod.Writer[object]", loaders_mod.MatplotlibFigure())
        elif isinstance(tag, _MetricTag):
            fmt = typing.cast("loaders_mod.Writer[object]", loaders_mod.YAML())
        else:
            fmt = _infer_format(base_type)

    if isinstance(tag, _IncrementalTag) and not _implements_empty(fmt):
        raise exceptions.StageDefinitionError(
            f"Stage '{stage_name}': {label} is incremental but loader "
            + f"'{type(fmt).__name__}' does not implement empty()."
        )
    return [_OutputSpec(key=key, python_type=base_type, format=fmt, tag=tag)]


def _directory_file_type(base_type: Any, label: str, stage_name: str) -> type:
    """The type ``T`` of each file of a ``directory`` output, which must be ``dict[str, T]``."""
    file_args = get_args(base_type)
    if get_origin(base_type) is not dict or len(file_args) != 2 or file_args[0] is not str:
        shown = base_type.__name__ if isinstance(base_type, type) else base_type
        raise exceptions.StageDefinitionError(
            f"Stage '{stage_name}': {label} is tagged directory but its type is {shown}. "
            + "A directory output is a dict[str, T] from each file's path inside the "
            + "directory to the value written there."
        )
    return file_args[1]


def _implements_empty(fmt: object) -> bool:
    """Whether a loader can give an incremental output's first-run value (overrides empty())."""
    base: object = loaders_mod.Loader
    return isinstance(fmt, loaders_mod.Loader) and inspect.getattr_static(
        type(fmt), "empty"
    ) is not inspect.getattr_static(base, "empty")


def _refuse_outputs_named_like_directory_files(specs: list[_OutputSpec], stage_name: str) -> None:
    """Refuse an output whose key starts with a directory output's key and ``/``.

    One file of directory output ``parts`` is named ``parts/<path>``, so an output keyed
    ``parts/a.json`` would share its name with file ``a.json`` of ``parts``.
    """
    directories = [spec.key for spec in specs if isinstance(spec.tag, _DirectoryTag)]
    for spec in specs:
        for directory_key in directories:
            if spec.key.startswith(f"{directory_key}/"):
                raise exceptions.StageDefinitionError(
                    f"Stage '{stage_name}': output '{spec.key}' has the name of a file of "
                    + f"directory output '{directory_key}', so a stage depending on one could "
                    + "not be told from a stage depending on the other. Rename the field."
                )


def _analyze_return_type(func: Callable[..., object], stage_name: str) -> list[_OutputSpec]:
    """Extract output specs from a function's return type annotation.

    ``stage_name`` is the qualified name (``pipeline/function``) messages give the stage.
    """
    hints = get_type_hints(func, include_extras=True)
    return_hint = hints.get("return")
    if return_hint is None:
        msg = f"Stage function {func.__name__} must have a return type annotation"
        raise exceptions.StageDefinitionError(msg)
    if return_hint is type(None):
        return []
    return _parse_output_type(return_hint, SINGLE_OUTPUT_KEY, stage_name)


# --- @stage decorator ---


def _normalize_stage_resources(
    resources: collections.abc.Mapping[str, object] | None,
) -> dict[str, int | str]:
    """Normalize and validate a stage's scheduler resource requests.

    Values are checked at runtime, since a decorator argument can be anything: a positive
    int (not bool), or ``"all"`` for cpu only.
    """
    if resources is None:
        return dict[str, int | str]()

    normalized = dict[str, int | str]()
    for name, request in resources.items():
        normalized_name = name.strip().casefold()
        if not normalized_name:
            raise ValueError("Stage resource names must not be empty")
        if normalized_name in normalized:
            raise ValueError(f"Stage resource {name!r} is declared more than once")
        if isinstance(request, str):
            if request.casefold() != "all":
                raise ValueError(
                    f"Stage resource {normalized_name!r} must request a positive integer or 'all'"
                )
            if normalized_name != "cpu":
                raise ValueError("Only the 'cpu' resource supports an 'all' request")
            normalized[normalized_name] = "all"
        elif isinstance(request, int) and not isinstance(request, bool) and request > 0:
            normalized[normalized_name] = request
        else:
            raise ValueError(
                f"Stage resource {normalized_name!r} must request a positive integer or 'all'"
            )
    return normalized


@typing.overload
def stage[**P, R](func: Callable[P, R], /) -> Callable[P, R]: ...


@typing.overload
def stage[**P, R](
    *, resources: dict[str, int | str]
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def stage[**P, R](
    func: Callable[P, R] | None = None,
    /,
    *,
    resources: dict[str, int | str] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as a pipeline stage with optional scheduler resources.

    In pipeline context (inside ``with Pipeline()``): records a DAG node, returns ArtifactHandle.
    Outside pipeline context: calls the function normally (for tests/notebooks).

    Return type analysis is DEFERRED to first pipeline-context use, not performed
    at decoration time. This allows stage modules to use in-body imports for heavy
    dependencies while still having type annotations that reference those types.
    """
    if func is None:
        return lambda decorated: _stage(decorated, resources)
    return _stage(func, resources)


def _stage[**P, R](func: Callable[P, R], resources: dict[str, int | str] | None) -> Callable[P, R]:
    """Build a stage wrapper for a validated resource declaration."""
    normalized_resources = _normalize_stage_resources(resources)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pipeline = _active_pipeline.get()
        if pipeline is None:
            return func(*args, **kwargs)
        return pipeline._record_stage(  # pyright: ignore[reportPrivateUsage] - the @stage wrapper is Pipeline's recording entry point
            func, wrapper, normalized_resources, args, kwargs
        )

    return wrapper


@dataclasses.dataclass
class _OutputSpec:
    key: str
    python_type: type
    format: loaders_mod.Writer[object]
    tag: _OutputTag | None = None


@dataclasses.dataclass
class _StageNode:
    original_func: Callable[..., object]
    name: str
    params: stage_def.StageParams | None
    input_handles: dict[str, ArtifactHandle]
    output_specs: list[_OutputSpec]
    call_index: int
    variant: str | None = None
    list_input_handles: dict[str, list[ArtifactHandle]] = dataclasses.field(default_factory=dict)
    collection_params: dict[str, CollectionKind] = dataclasses.field(default_factory=dict)
    no_fingerprint: bool = False
    code_deps: list[str] = dataclasses.field(default_factory=list)
    incremental_inputs: dict[str, str] = dataclasses.field(default_factory=dict)
    resources: dict[str, int | str] = dataclasses.field(default_factory=dict)
    # Set on a template: a stage called with a mapped handle (``Pipeline.map``)
    map_source: _MapSource | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class _MapSource:
    """The directory output a mapped handle stands for one file of: ``node``'s output keyed
    ``output_key`` (None for its single output), in ``pipeline``."""

    pipeline: Pipeline
    node: _StageNode
    output_key: str | None

    def identity(self) -> types.ArtifactIdentity:
        return types.ArtifactIdentity(f"{self.pipeline._name}/{self.node.name}", self.output_key)  # pyright: ignore[reportPrivateUsage] - compose owns Pipeline

    def same(self, other: _MapSource) -> bool:
        """Whether both stand for one directory: the same node's same output."""
        return self.node is other.node and self.output_key == other.output_key


@dataclasses.dataclass
class _PreviousOutputMarker:
    pipeline: Pipeline
    key: str | None
    consumed_by: str | None = None


@dataclasses.dataclass
class _InputNode:
    name: str
    python_type: type | None
    path: str
    format: loaders_mod.Reader[object] | loaders_mod.Loader[object, object]
    inherited: bool = False


_HANDLE_NOT_ITERABLE = (
    "An artifact handle is not iterable: pipeline.py runs before any stage writes its files. "
    + "Name one file of a directory output, e.g. parts['train.csv'], or pass the handle itself."
)


class ArtifactHandle:
    _pipeline: Pipeline
    _source: _StageNode | _InputNode
    _output_key: str | None
    _python_type: type
    _member: str | None
    _map: _MapSource | None

    def __init__(
        self,
        pipeline: Pipeline,
        source: _StageNode | _InputNode,
        output_key: str | None,
        python_type: type,
        member: str | None = None,
        map_source: _MapSource | None = None,
    ) -> None:
        """``member`` is the path of one file inside a directory output, as the user wrote it.

        ``map_source`` makes the handle mapped (``Pipeline.map``): it stands for each key's file
        of that directory when ``source`` is the directory's producer (a mapped entry), and for
        each key's instance of a template when ``source`` is the template.
        """
        self._pipeline = pipeline
        self._source = source
        self._output_key = output_key
        self._python_type = python_type
        self._member = member
        self._map = map_source

    def _is_map_entry(self) -> bool:
        return self._map is not None and self._source is self._map.node

    def __getattr__(self, name: str) -> ArtifactHandle:
        if name.startswith("_"):
            raise AttributeError(name)
        if not isinstance(self._source, _StageNode):
            raise AttributeError(f"Input '{self._source.name}' has no sub-outputs")
        if self._is_map_entry():
            raise AttributeError(self._map_entry_has_no_parts_message())
        if self._member is not None:
            raise AttributeError(self._member_has_no_parts_message())
        spec = _select_output(self._source.output_specs, name)
        if spec is None:
            raise AttributeError(_no_such_output_message(self._source, name))
        return ArtifactHandle(
            self._pipeline, self._source, name, spec.python_type, map_source=self._map
        )

    def __getitem__(self, key: str) -> ArtifactHandle:
        """Select an output of a multi-output stage, or one file of a directory output."""
        if not isinstance(self._source, _StageNode):
            raise KeyError(f"Input '{self._source.name}' has no sub-outputs")
        if self._is_map_entry():
            raise KeyError(self._map_entry_has_no_parts_message())
        if self._member is not None:
            raise KeyError(self._member_has_no_parts_message())
        directory_spec = self._selected_directory()
        if directory_spec is not None:
            # _parse_output_type checked that a directory output's type is dict[str, T]
            file_type = get_args(directory_spec.python_type)[1]
            return ArtifactHandle(
                self._pipeline,
                self._source,
                self._output_key,
                file_type,
                member=key,
                map_source=self._map,
            )
        spec = _select_output(self._source.output_specs, key)
        if spec is None:
            raise KeyError(_no_such_output_message(self._source, key))
        return ArtifactHandle(
            self._pipeline, self._source, key, spec.python_type, map_source=self._map
        )

    def __iter__(self) -> Never:
        """Refuse ``for``: a handle names outputs that no stage has written yet.

        Without this method Python iterates by calling ``handle[0]``, ``handle[1]``, ..., and a
        directory handle accepts every key, so the loop would never end.
        """
        raise TypeError(_HANDLE_NOT_ITERABLE)

    def __contains__(self, item: object) -> Never:
        """Refuse ``in``. Without this method ``in`` calls ``__iter__``, and Python replaces the
        TypeError it raises with a generic "not iterable" one that does not say what to write.
        """
        raise TypeError(_HANDLE_NOT_ITERABLE)

    def _member_has_no_parts_message(self) -> str:
        assert isinstance(self._source, _StageNode)
        return (
            f"{self._member!r} is one file of a directory output of stage "
            + f"'{self._pipeline.name}/{self._source.name}'; it has no outputs or files of its "
            + "own. Pass the handle itself."
        )

    def _map_entry_has_no_parts_message(self) -> str:
        assert self._map is not None
        return (
            "A mapped handle stands for one file of the directory output of stage "
            + f"'{types.identity_key(self._map.identity())}', the file of each key; it has no "
            + "outputs or files of its own. Pass the handle itself."
        )

    def _selected_directory(self) -> _OutputSpec | None:
        """The directory output this handle stands for, or None when it stands for another.

        A stage's own handle stands for its return value only when that is a single output;
        a TypedDict field is selected by name before its files are.
        """
        assert isinstance(self._source, _StageNode)
        key = SINGLE_OUTPUT_KEY if self._output_key is None else self._output_key
        spec = _select_output(self._source.output_specs, key)
        return spec if spec is not None and isinstance(spec.tag, _DirectoryTag) else None


def _select_output(output_specs: list[_OutputSpec], key: str | None) -> _OutputSpec | None:
    """The output spec named ``key``, or None when the stage has no such output."""
    return next((spec for spec in output_specs if spec.key == key), None)


def _no_such_output_message(source: _StageNode, key: str) -> str:
    if [spec.key for spec in source.output_specs] == [SINGLE_OUTPUT_KEY]:
        return (
            f"Stage '{source.name}' has no output '{key}': it has a single output, the return "
            + "value. Pass the handle itself."
        )
    available = [spec.key for spec in source.output_specs]
    return f"Stage '{source.name}' has no output '{key}'. Available: {available}"
