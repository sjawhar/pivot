from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
import pathlib
import posixpath
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

import pygtrie

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

from pivot import exceptions, fingerprint, loaded_source, project, stage_def, types
from pivot.storage import cache, lock
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    from inspect import Signature

    from networkx import DiGraph

    from pivot.state_dir import StateDirResolution
logger = logging.getLogger(__name__)


class RegistryStageInfo(TypedDict):
    """Metadata for a registered stage.

    Attributes:
        func: The stage function to execute.
        name: Unique stage identifier (function name or custom name).
        deps: Named input dependencies (param name -> ArtifactRef).
        outs: Output specs as artifact references.
        incremental_inputs: Previous-output parameters (param name -> own output identity).
        params: Pydantic model instance with parameter values.
        resources: Scheduler resource requests declared on the stage.
        variant: Variant name for matrix stages (None for regular stages).
        signature: Function signature for parameter injection.
        fingerprint: Code fingerprint mapping (key -> hash), or None if not computed yet.
        params_arg_name: Name of the StageParams parameter in function signature (or None).
        state_dir: Per-stage state directory (None means use default determined at runtime).
        no_fingerprint: Use file-level hashing instead of AST fingerprinting.
        code_deps: Extra project-relative files hashed into a no_fingerprint stage's fingerprint.
        input_paths: Project-relative paths by input artifact identity, for the wiring manifest.
        unsafe_fingerprint: The fingerprint was computed with a refusal downgraded to a
            warning (unsafe fingerprinting), so the run cache neither files nor reuses results.
    """

    func: Callable[..., Any]
    name: str
    deps: dict[str, types.ArtifactRef]
    outs: list[types.ArtifactRef]
    incremental_inputs: dict[str, types.ArtifactIdentity]
    params: stage_def.StageParams | None
    resources: dict[str, int | str]
    variant: str | None
    signature: Signature | None
    fingerprint: dict[str, str] | None
    params_arg_name: str | None
    state_dir: pathlib.Path | None
    collection_params: dict[str, str]
    no_fingerprint: bool
    code_deps: list[str]
    input_paths: dict[str, str]
    unsafe_fingerprint: bool


class MapTemplate(TypedDict):
    """A stage mapped over a directory output (``Pipeline.map``): not a stage itself, but the
    pattern of one instance per file of the directory, ``<name>@<key>``, made when a run finds
    its producer current.

    Attributes:
        info: The stage as declared, named by the template. A mapped dependency reads the
            whole directory (``source``), or an upstream template's output; the rest are
            ordinary dependencies every instance shares.
        source: The directory output whose files expand the template.
        mapped_deps: Each mapped dependency's name, mapped to None for the file of the
            instance's key, or to the name of the upstream template whose instance of that key
            the dependency reads.
    """

    info: RegistryStageInfo
    source: types.ArtifactIdentity
    mapped_deps: dict[str, str | None]


class StageRegistry:
    """Registry for pipeline stages and DAG construction."""

    _stages: dict[str, RegistryStageInfo]
    _templates: dict[str, MapTemplate]
    # Each expanded template's instances, in the order they were added
    _instances: dict[str, list[str]]
    _output_paths: pygtrie.Trie[types.ArtifactIdentity]

    def __init__(self) -> None:
        self._stages = dict[str, RegistryStageInfo]()
        self._templates = dict[str, MapTemplate]()
        self._instances = dict[str, list[str]]()
        self._output_paths = pygtrie.Trie()
        self._cached_dag: DiGraph[str] | None = None

    def add_existing(self, stage_info: RegistryStageInfo) -> None:
        """Add a pre-validated stage info (for pipeline composition).

        Args:
            stage_info: Complete stage info to add.

        Raises:
            ValidationError: If stage name already exists, or is the name an instance of a
                template would have (``<template>@<key>``).
            OutputPathConflictError: If an output's workspace path equals, contains, or lies
                inside the path of an output already registered. Each worker resolves paths
                in its own store, so this is the only place that sees every output at once.
        """
        name = stage_info["name"]
        template = template_of(name, self._templates)
        if template is not None:
            raise exceptions.ValidationError(_instance_name_error(name, template))
        self._add(stage_info)

    def _add(self, stage_info: RegistryStageInfo) -> None:
        """``add_existing`` for a stage that may be a template's instance (``expand``)."""
        name = stage_info["name"]
        if name in self._stages or name in self._templates:
            raise exceptions.ValidationError(f"Stage '{name}' already registered")
        for out in stage_info["outs"]:
            if out.identity.producer != name:
                raise ValueError(
                    f"Output producer '{out.identity.producer}' for stage '{name}' must match stage name"
                )
        added = list[tuple[str, ...]]()
        try:
            for out in stage_info["outs"]:
                parts = store_mod.relative_output_path(out).parts
                _check_output_path_free(self._output_paths, parts, out.identity)
                self._output_paths[parts] = out.identity
                added.append(parts)
        except exceptions.OutputPathConflictError:
            for parts in added:
                del self._output_paths[parts]
            raise
        self._stages[name] = stage_info
        self._cached_dag = None

    def check_input_path(self, name: str, parts: tuple[str, ...]) -> None:
        """Raise if input ``name``'s project-relative path overlaps a registered output.

        An input that reads a stage's output file, a directory holding one, or a file inside
        an output directory would read bytes a stage rewrites with no DAG edge between them.

        Raises:
            OutputPathConflictError: Naming the input, the output and both paths.
        """
        path = "/".join(parts) or "."
        if parts in self._output_paths:
            output = types.identity_key(self._output_paths[parts])
            raise exceptions.OutputPathConflictError(
                f"Output path conflict: input '{name}' reads {path}, which '{output}' writes."
            )
        if self._output_paths.has_subtrie(parts):
            inner_parts = next(iter(self._output_paths.keys(prefix=parts)))
            output = types.identity_key(self._output_paths[inner_parts])
            raise exceptions.OutputPathConflictError(
                f"Output path conflict: input '{name}' reads {path}, which contains "
                + f"{'/'.join(inner_parts)} that '{output}' writes."
            )
        enclosing = self._output_paths.shortest_prefix(parts)
        if enclosing is not None and enclosing.is_set:
            outer = enclosing.value
            assert outer is not None  # is_set guarantees a stored value
            raise exceptions.OutputPathConflictError(
                f"Output path conflict: input '{name}' reads {path}, inside "
                + f"{'/'.join(enclosing.key)} that '{types.identity_key(outer)}' writes."
            )

    def get(self, name: str) -> RegistryStageInfo:
        """Get stage info by name (raises KeyError if not found)."""
        return self._stages[name]

    def list_stages(self) -> list[str]:
        """Get list of all stage names: no template, and each expanded instance."""
        return list(self._stages.keys())

    def add_template(self, template: MapTemplate) -> None:
        """Add a template (``Pipeline.map``). It writes nothing, so it claims no output path;
        each instance's paths are checked when it is added (``expand``).

        Raises:
            ValidationError: If a stage or template already has its name, a stage or template
                has the name of one of its instances, or it has the name of an instance of
                another template.
        """
        name = template["info"]["name"]
        if name in self._stages or name in self._templates:
            raise exceptions.ValidationError(f"Stage '{name}' already registered")
        owner = template_of(name, self._templates)
        if owner is not None:
            raise exceptions.ValidationError(_instance_name_error(name, owner))
        taken = next(
            (other for other in (*self._stages, *self._templates) if template_of(other, [name])),
            None,
        )
        if taken is not None:
            raise exceptions.ValidationError(_instance_name_error(taken, name))
        self._templates[name] = template
        self._cached_dag = None

    def list_templates(self) -> list[str]:
        """The names of the templates, in the order they were added."""
        return list(self._templates)

    def get_template(self, name: str) -> MapTemplate:
        """Get a template by name (raises KeyError if not found)."""
        return self._templates[name]

    def graph_stages(self) -> dict[str, RegistryStageInfo]:
        """Every stage and every template, as the stage graph holds them: a template stands
        for its instances until a run expands it, and reads its whole source directory."""
        return {**self._stages, **{name: t["info"] for name, t in self._templates.items()}}

    def expand(self, name: str, keys: Sequence[tuple[str, str]]) -> list[str]:
        """Add template ``name``'s instance for each (key, file key) of ``keys``
        (``map_instance_keys``), and return their names.

        Each instance is added as any stage is (``add_existing``), so a name or an output path
        another stage holds is refused. The expansion is all or nothing: on a refusal no
        instance of it stays.

        Raises:
            PipelineConfigError: An instance name is not a legal stage name.
            ValidationError: A stage already has an instance's name.
            OutputPathConflictError: An instance's output path overlaps another's.
        """
        template = self._templates[name]
        added = list[str]()
        try:
            for key, file_key in keys:
                info = instantiate(template, key, file_key)
                reason = lock.stage_name_error(info["name"])
                if reason is not None:
                    raise exceptions.PipelineConfigError(
                        f"Mapped stage '{name}' cannot name its instance for file {file_key!r} "
                        + f"of '{types.identity_key(template['source'])}': {reason}"
                    )
                self._add(info)
                added.append(info["name"])
        except exceptions.PivotError:
            for stage_name in added:
                self._remove(stage_name)
            raise
        self._instances.setdefault(name, []).extend(added)
        return added

    def instances_of(self, name: str) -> list[str]:
        """The instances template ``name`` was expanded into since the last reset."""
        return list(self._instances.get(name, []))

    def reset_expansions(self) -> None:
        """Remove every instance, leaving each template unexpanded, as a load leaves it."""
        for instances in self._instances.values():
            for stage_name in instances:
                self._remove(stage_name)
        self._instances.clear()

    def _remove(self, name: str) -> None:
        info = self._stages.pop(name)
        for out in info["outs"]:
            del self._output_paths[store_mod.relative_output_path(out).parts]
        self._cached_dag = None

    def ensure_fingerprint(self, stage_name: str) -> dict[str, str]:
        """Ensure a stage fingerprint is computed and cached."""
        info = self._stages[stage_name]
        if info["fingerprint"] is None:
            info["fingerprint"], info["unsafe_fingerprint"] = _compute_fingerprint(stage_name, info)
        return info["fingerprint"]

    def build_dag(self) -> DiGraph[str]:
        """Build DAG from registered stages and templates (``graph_stages``).

        Returns:
            NetworkX DiGraph with stages as nodes and dependencies as edges
        """
        if self._cached_dag is not None:
            return self._cached_dag

        from pivot.engine import graph as engine_graph

        # Build bipartite graph, extract stage DAG
        bipartite = engine_graph.build_graph(self.graph_stages())
        graph = engine_graph.get_stage_dag(bipartite)

        self._cached_dag = graph

        return graph

    def clear(self) -> None:
        """Clear all registered stages and templates (for testing)."""
        self._stages.clear()
        self._templates.clear()
        self._instances.clear()
        self._output_paths = pygtrie.Trie()
        self._cached_dag = None

    def invalidate_dag_cache(self) -> None:
        """Invalidate cached DAG without clearing stages.

        Call when external state changes (code reload, config change) that
        would affect DAG construction but stage registrations haven't changed yet.
        """
        self._cached_dag = None

    def snapshot(self) -> dict[str, RegistryStageInfo]:
        """Return a shallow copy of the registered stages.

        Engine reload keeps this copy of the previous pipeline's stages to report which
        stages the reloaded pipeline added, removed, or modified.
        """
        return dict(self._stages)

    def copy(self) -> StageRegistry:
        """A registry holding these stages, templates, instances and output paths, which can
        change without changing this one (``Pipeline.include`` tries an include on one). The
        stage and template entries are shared, as ``snapshot`` shares them."""
        copied = StageRegistry()
        copied._stages = dict(self._stages)
        copied._templates = dict(self._templates)
        copied._instances = {name: list(stages) for name, stages in self._instances.items()}
        copied._output_paths = self._output_paths.copy()
        return copied


@runtime_checkable
class PipelineLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def root(self) -> pathlib.Path: ...

    @property
    def state_dir(self) -> pathlib.Path: ...

    @property
    def resolved_state_dirs(self) -> Mapping[pathlib.Path, StateDirResolution]: ...

    @property
    def input_bindings(self) -> dict[str, str]: ...

    @property
    def loaded_pipelines(self) -> frozenset[str]: ...

    def list_stages(self) -> list[str]: ...

    def get_stage(self, name: str) -> RegistryStageInfo: ...

    def list_templates(self) -> list[str]: ...

    def get_template(self, name: str) -> MapTemplate: ...

    def expand_template(self, name: str, keys: Sequence[tuple[str, str]]) -> list[str]: ...

    def instances_of(self, name: str) -> list[str]: ...

    def reset_expansions(self) -> None: ...

    def ensure_fingerprint(self, name: str) -> dict[str, str]: ...

    def build_dag(self) -> DiGraph[str]: ...

    def invalidate_dag_cache(self) -> None: ...

    def snapshot(self) -> dict[str, RegistryStageInfo]: ...

    def include(self, other: PipelineLike) -> None: ...


def map_instance_keys(file_keys: Iterable[str]) -> list[tuple[str, str]]:
    """Each file of a mapped directory as (instance key, file key), sorted by instance key.

    A file's instance key is its path inside the directory without its extension: a template
    ``p/cut`` over ``plan/a.yaml`` has the instance ``p/cut@a``. The key follows the
    variant-name rule (``lock.variant_name_error``), so a key never holds ``/`` or ``@`` and
    no two files map to one instance.

    Raises:
        PipelineConfigError: Naming every file whose key breaks the rule and every two files
            that share a key.
    """
    file_by_key = dict[str, str]()
    errors = list[str]()
    for file_key in sorted(file_keys):
        key = posixpath.splitext(file_key)[0]
        reason = lock.variant_name_error(key)
        if reason is not None:
            errors.append(
                f"file {file_key!r}: its key {key!r} (its path without the extension) cannot "
                + f"name a stage: {reason}"
            )
        elif key in file_by_key:
            errors.append(f"files {file_by_key[key]!r} and {file_key!r} have one key {key!r}")
        else:
            file_by_key[key] = file_key
    if errors:
        raise exceptions.PipelineConfigError("; ".join(errors))
    return sorted(file_by_key.items())


def template_of(stage_name: str, templates: Iterable[str]) -> str | None:
    """The template of ``templates`` whose instance ``stage_name`` names (``<template>@<key>``,
    with a key ``map_instance_keys`` allows), or None. A key holds no ``@``, so at most one
    template matches."""
    for template in templates:
        prefix = f"{template}@"
        if (
            stage_name.startswith(prefix)
            and lock.variant_name_error(stage_name[len(prefix) :]) is None
        ):
            return template
    return None


def _instance_name_error(name: str, template: str) -> str:
    """Why stage or template ``name`` cannot load beside template ``template``: it has the name
    of one of its instances, so a target, a lock and a ``params.yaml`` key could mean either."""
    return (
        f"'{name}' has the name of an instance of mapped stage '{template}' "
        + f"('{template}@<key>'): call one of them inside a pipeline.variant(...) block"
    )


def instantiate(template: MapTemplate, key: str, file_key: str) -> RegistryStageInfo:
    """Template ``template``'s stage for one key: named ``<template>@<key>``, its entry reading
    file ``file_key`` of the source directory, each mapped dependency on an upstream template
    reading that template's instance of the key, and its outputs its own."""
    declared = template["info"]
    name = f"{declared['name']}@{key}"

    def instance_of(identity: types.ArtifactIdentity) -> types.ArtifactIdentity:
        return types.ArtifactIdentity(f"{identity.producer}@{key}", identity.key)

    deps = dict[str, types.ArtifactRef]()
    for dep_name, ref in declared["deps"].items():
        if dep_name not in template["mapped_deps"]:
            deps[dep_name] = ref
        elif template["mapped_deps"][dep_name] is None:
            source = template["source"]
            deps[dep_name] = dataclasses.replace(
                ref, identity=types.directory_file_identity(source, file_key), directory=source
            )
        else:
            deps[dep_name] = dataclasses.replace(
                ref,
                identity=instance_of(ref.identity),
                directory=None if ref.directory is None else instance_of(ref.directory),
            )
    variant = declared["variant"]
    return RegistryStageInfo(
        func=declared["func"],
        name=name,
        deps=deps,
        outs=[
            dataclasses.replace(out, identity=types.ArtifactIdentity(name, out.identity.key))
            for out in declared["outs"]
        ],
        incremental_inputs={
            param: types.ArtifactIdentity(name, identity.key)
            for param, identity in declared["incremental_inputs"].items()
        },
        # Each instance owns its params, as an included stage owns a deep copy of its info
        params=copy.deepcopy(declared["params"]),
        resources=dict(declared["resources"]),
        variant=key if variant is None else f"{variant}@{key}",
        signature=declared["signature"],
        fingerprint=None,
        params_arg_name=declared["params_arg_name"],
        state_dir=declared["state_dir"],
        collection_params=dict(declared["collection_params"]),
        no_fingerprint=declared["no_fingerprint"],
        code_deps=list(declared["code_deps"]),
        input_paths=dict(declared["input_paths"]),
        unsafe_fingerprint=False,
    )


def _check_output_path_free(
    paths: pygtrie.Trie[types.ArtifactIdentity],
    parts: tuple[str, ...],
    identity: types.ArtifactIdentity,
) -> None:
    """Raise if ``parts`` equals, contains, or lies inside a registered output path."""
    new = f"'{types.identity_key(identity)}'"
    path = "/".join(parts)
    if parts in paths:
        other = f"'{types.identity_key(paths[parts])}'"
        raise exceptions.OutputPathConflictError(
            f"Output path conflict: {new} and {other} both write {path}."
        )
    if paths.has_subtrie(parts):
        inner_parts = next(iter(paths.keys(prefix=parts)))
        inner = f"'{types.identity_key(paths[inner_parts])}'"
        raise exceptions.OutputPathConflictError(
            f"Output path conflict: {inner} writes {'/'.join(inner_parts)}, "
            + f"inside the directory {path} that {new} writes."
        )
    enclosing = paths.shortest_prefix(parts)
    if enclosing is not None and enclosing.is_set:
        outer = enclosing.value
        assert outer is not None  # is_set guarantees a stored value
        raise exceptions.OutputPathConflictError(
            f"Output path conflict: {new} writes {path}, inside the directory "
            + f"{'/'.join(enclosing.key)} that '{types.identity_key(outer)}' writes."
        )


def _compute_fingerprint(stage_name: str, info: RegistryStageInfo) -> tuple[dict[str, str], bool]:
    """Compute a stage fingerprint and whether a refusal in it was downgraded; wrap errors."""
    try:
        with fingerprint.stage_fingerprint_scope(stage_name) as scope:
            no_fp = info["no_fingerprint"]
            if no_fp:
                result = _compute_file_fingerprint(info["func"], info["code_deps"])
            else:
                result = fingerprint.get_stage_fingerprint_cached(stage_name, info["func"])
            result.update(_wiring_manifest(info))
            for dep_name, dep in info["deps"].items():
                fingerprint.merge_loader_fingerprint(result, f"dep:{dep_name}:", dep.format)
            outputs = set[str]()
            for out in info["outs"]:
                out_key = types.identity_key(out.identity)
                outputs.add(out_key)
                # The tag places the output (data/, metrics/, plots/, a directory) and decides
                # how it is written, and a type alias can change it without touching the source.
                result[f"out:{out_key}:tag"] = out.tag.value
                fingerprint.merge_loader_fingerprint(result, f"out:{out_key}:", out.format)
            loaded_source.check_worker_imports(result, outputs)
        return fingerprint.without_by_name(result), scope["unsafe"]
    except Exception as exc:
        raise exceptions.PivotError(f"Stage '{stage_name}': fingerprinting failed: {exc}") from exc


def _wiring_manifest(info: RegistryStageInfo) -> dict[str, str]:
    """Return the call-site artifact wiring that gives a stage its input meaning.

    Dependency content hashes intentionally collapse duplicate identities for cache reuse, but
    a stage call does not: ``combine(a, b)`` differs from ``combine(b, a)``, and a list differs
    from a tuple. Inputs also carry their project-relative binding path because a ``PathOnly``
    stage can observe it even when old and new files have identical contents.
    """
    paths = info["input_paths"]
    result = dict[str, str]()
    collection_params = info["collection_params"]
    for parameter, kind in collection_params.items():
        refs = [
            dep
            for _index, dep in sorted(
                (
                    (int(name.removeprefix(f"{parameter}[").removesuffix("]")), dep)
                    for name, dep in info["deps"].items()
                    if name.startswith(f"{parameter}[") and name.endswith("]")
                ),
                key=lambda item: item[0],
            )
        ]
        identities = [types.identity_key(ref.identity) for ref in refs]
        result[f"wire:{parameter}"] = json.dumps(
            {"kind": kind, "identities": identities}, separators=(",", ":")
        )
        input_paths = [
            paths.get(identity) if ref.tag is types.ArtifactTag.INPUT else None
            for ref, identity in zip(refs, identities, strict=True)
        ]
        if any(path is not None for path in input_paths):
            result[f"wire:{parameter}:paths"] = json.dumps(input_paths, separators=(",", ":"))
    for parameter, ref in info["deps"].items():
        if parameter.split("[", 1)[0] in collection_params:
            continue
        identity = types.identity_key(ref.identity)
        result[f"wire:{parameter}"] = identity
        if ref.tag is types.ArtifactTag.INPUT and identity in paths:
            result[f"wire:{parameter}:path"] = paths[identity]
    return result


def _compute_file_fingerprint(func: Callable[..., Any], code_deps: list[str]) -> dict[str, str]:
    """Compute file-hash fingerprint (no AST analysis).

    Returns dict with keys like ``file:path/to/module.py`` -> hash.
    Includes the stage function's source file and each of ``code_deps``.
    Uses ``fingerprint.unwrap`` (``inspect.unwrap``'s reading) to handle stacked decorators.

    A file this process loaded a module from is hashed as it was loaded (``loaded_source``), the
    code a worker receives, not as it is on disk now.
    """
    result = dict[str, str]()

    # Unwrap decorators so inspect.getfile returns the actual stage source
    unwrapped = fingerprint.unwrap(func)
    source_file = pathlib.Path(inspect.getfile(unwrapped))
    file_hash = loaded_source.digest(str(source_file)) or cache.hash_file(source_file)[0]
    rel_path = project.to_relative_path(source_file)
    result[f"file:{rel_path}"] = file_hash

    root = project.get_project_root()
    for dep_path in code_deps:
        abs_path = (root / dep_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"code_deps file not found: {dep_path}")
        dep_hash = loaded_source.digest(str(abs_path)) or cache.hash_file(abs_path)[0]
        # Normalize key to project-relative path for stable lockfiles
        dep_rel = project.to_relative_path(abs_path)
        result[f"file:{dep_rel}"] = dep_hash

    return result


def get_stage_state_dir(stage_info: RegistryStageInfo, default: pathlib.Path) -> pathlib.Path:
    """Return the stage's state_dir, falling back to the given default."""
    return stage_info["state_dir"] or default
