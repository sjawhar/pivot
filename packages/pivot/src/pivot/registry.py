from __future__ import annotations

import inspect
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

import pygtrie

if TYPE_CHECKING:
    from collections.abc import Callable

from pivot import exceptions, fingerprint, project, stage_def, types
from pivot.storage import cache
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    from inspect import Signature

    from networkx import DiGraph
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


class StageRegistry:
    """Registry for pipeline stages and DAG construction."""

    _stages: dict[str, RegistryStageInfo]
    _output_paths: pygtrie.Trie[types.ArtifactIdentity]

    def __init__(self) -> None:
        self._stages = dict[str, RegistryStageInfo]()
        self._output_paths = pygtrie.Trie()
        self._cached_dag: DiGraph[str] | None = None

    def add_existing(self, stage_info: RegistryStageInfo) -> None:
        """Add a pre-validated stage info (for pipeline composition).

        Args:
            stage_info: Complete stage info to add.

        Raises:
            ValidationError: If stage name already exists.
            OutputPathConflictError: If an output's workspace path equals, contains, or lies
                inside the path of an output already registered. Each worker resolves paths
                in its own store, so this is the only place that sees every output at once.
        """
        name = stage_info["name"]
        if name in self._stages:
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
        """Get list of all stage names."""
        return list(self._stages.keys())

    def ensure_fingerprint(self, stage_name: str) -> dict[str, str]:
        """Ensure a stage fingerprint is computed and cached."""
        info = self._stages[stage_name]
        if info["fingerprint"] is None:
            info["fingerprint"] = _compute_fingerprint(stage_name, info)
        return info["fingerprint"]

    def build_dag(self) -> DiGraph[str]:
        """Build DAG from registered stages.

        Returns:
            NetworkX DiGraph with stages as nodes and dependencies as edges
        """
        if self._cached_dag is not None:
            return self._cached_dag

        from pivot.engine import graph as engine_graph

        # Build bipartite graph, extract stage DAG
        bipartite = engine_graph.build_graph(self._stages)
        graph = engine_graph.get_stage_dag(bipartite)

        self._cached_dag = graph

        return graph

    def clear(self) -> None:
        """Clear all registered stages (for testing)."""
        self._stages.clear()
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


@runtime_checkable
class PipelineLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def root(self) -> pathlib.Path: ...

    @property
    def state_dir(self) -> pathlib.Path: ...

    @property
    def input_bindings(self) -> dict[str, str]: ...

    @property
    def loaded_pipelines(self) -> frozenset[str]: ...

    def list_stages(self) -> list[str]: ...

    def get_stage(self, name: str) -> RegistryStageInfo: ...

    def ensure_fingerprint(self, name: str) -> dict[str, str]: ...

    def build_dag(self) -> DiGraph[str]: ...

    def invalidate_dag_cache(self) -> None: ...

    def snapshot(self) -> dict[str, RegistryStageInfo]: ...

    def include(self, other: PipelineLike) -> None: ...


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


def _compute_fingerprint(stage_name: str, info: RegistryStageInfo) -> dict[str, str]:
    """Compute and return a stage fingerprint, wrapping errors."""
    try:
        with fingerprint.stage_fingerprint_scope(stage_name):
            no_fp = info["no_fingerprint"]
            if no_fp:
                result = _compute_file_fingerprint(info["func"], info["code_deps"])
            else:
                result = fingerprint.get_stage_fingerprint_cached(stage_name, info["func"])
            result.update(_wiring_manifest(info))
            for dep_name, dep in info["deps"].items():
                for key, value in fingerprint.get_loader_fingerprint(dep.format).items():
                    result[f"dep:{dep_name}:{key}"] = value
            for out in info["outs"]:
                out_key = types.identity_key(out.identity)
                for key, value in fingerprint.get_loader_fingerprint(out.format).items():
                    result[f"out:{out_key}:{key}"] = value
        return result
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
    Uses ``inspect.unwrap`` to handle stacked decorators.
    """
    result = dict[str, str]()

    # Unwrap decorators so inspect.getfile returns the actual stage source
    unwrapped = inspect.unwrap(func)
    source_file = pathlib.Path(inspect.getfile(unwrapped))
    file_hash, _ = cache.hash_file(source_file)
    rel_path = project.to_relative_path(source_file)
    result[f"file:{rel_path}"] = file_hash

    root = project.get_project_root()
    for dep_path in code_deps:
        abs_path = (root / dep_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"code_deps file not found: {dep_path}")
        dep_hash, _ = cache.hash_file(abs_path)
        # Normalize key to project-relative path for stable lockfiles
        dep_rel = project.to_relative_path(abs_path)
        result[f"file:{dep_rel}"] = dep_hash

    return result


def get_stage_state_dir(stage_info: RegistryStageInfo, default: pathlib.Path) -> pathlib.Path:
    """Return the stage's state_dir, falling back to the given default."""
    return stage_info["state_dir"] or default
