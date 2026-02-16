# pyright: reportImplicitRelativeImport=false, reportMissingModuleSource=false, reportExplicitAny=false, reportAny=false
from __future__ import annotations

import enum
import inspect
import logging
import pathlib
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

from pivot import exceptions, fingerprint, project, stage_def, types
from pivot.storage import cache

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
        params: Pydantic model instance with parameter values.
        mutex: Mutex groups for exclusive execution.
        variant: Variant name for matrix stages (None for regular stages).
        signature: Function signature for parameter injection.
        fingerprint: Code fingerprint mapping (key -> hash), or None if not computed yet.
        params_arg_name: Name of the StageParams parameter in function signature (or None).
        state_dir: Per-stage state directory (None means use default determined at runtime).
    """

    func: Callable[..., Any]
    name: str
    deps: dict[str, types.ArtifactRef]
    outs: list[types.ArtifactRef]
    params: stage_def.StageParams | None
    mutex: list[str]
    variant: str | None
    signature: Signature | None
    fingerprint: dict[str, str] | None
    params_arg_name: str | None
    state_dir: pathlib.Path | None
    collection_params: dict[str, str]


class ValidationMode(enum.StrEnum):
    """Validation strictness levels."""

    ERROR = "error"  # Raise exception on validation failure
    WARN = "warn"  # Log warning, allow registration


class StageRegistry:
    """Registry for pipeline stages and DAG construction."""

    _stages: dict[str, RegistryStageInfo]

    def __init__(self, validation_mode: ValidationMode = ValidationMode.ERROR) -> None:
        self._stages = dict[str, RegistryStageInfo]()
        self._cached_dag: DiGraph[str] | None = None
        self.validation_mode: ValidationMode = validation_mode

    def add_existing(self, stage_info: RegistryStageInfo) -> None:
        """Add a pre-validated stage info (for pipeline composition).

        Args:
            stage_info: Complete stage info to add.

        Raises:
            ValidationError: If stage name already exists.
        """
        name = stage_info["name"]
        if name in self._stages:
            raise exceptions.ValidationError(f"Stage '{name}' already registered")
        self._stages[name] = stage_info
        self._cached_dag = None

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
        self._cached_dag = None

    def invalidate_dag_cache(self) -> None:
        """Invalidate cached DAG without clearing stages.

        Call when external state changes (code reload, config change) that
        would affect DAG construction but stage registrations haven't changed yet.
        """
        self._cached_dag = None

    def snapshot(self) -> dict[str, RegistryStageInfo]:
        """Create a snapshot of current registry state for backup/restore.

        Returns a shallow copy of the internal stages dict. Use with `restore()`
        to implement atomic reload patterns where you want to preserve the previous
        valid state if the reload fails.

        Example:
            backup = registry.snapshot()
            registry.clear()
            try:
                reload_stages()
            except Exception:
                registry.restore(backup)  # Rollback on failure
        """
        return dict(self._stages)

    def restore(self, snapshot: dict[str, RegistryStageInfo]) -> None:
        """Restore registry state from a previous snapshot.

        Replaces all current stages with the snapshot contents. Typically used
        to rollback after a failed reload operation.

        Args:
            snapshot: Previously captured state from `snapshot()`
        """
        self._stages = dict(snapshot)
        self._cached_dag = None

    def get_all_output_paths(self) -> set[str]:
        """Get all registered output paths (for watch mode filtering)."""
        return set[str]()


def _compute_fingerprint(stage_name: str, info: RegistryStageInfo) -> dict[str, str]:
    """Compute and return a stage fingerprint, wrapping errors."""
    try:
        unwrapped = inspect.unwrap(info["func"])
        if getattr(unwrapped, "__pivot_no_fingerprint__", False):
            result = _compute_file_fingerprint(info["func"])
        else:
            result = fingerprint.get_stage_fingerprint_cached(stage_name, info["func"])
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


def _compute_file_fingerprint(func: Callable[..., Any]) -> dict[str, str]:
    """Compute file-hash fingerprint (no AST analysis).

    Returns dict with keys like ``file:path/to/module.py`` -> hash.
    Includes the stage function's source file and any ``code_deps``.
    Uses ``inspect.unwrap`` to handle stacked decorators.
    """
    result = dict[str, str]()

    # Unwrap decorators so inspect.getfile returns the actual stage source
    unwrapped = inspect.unwrap(func)
    source_file = pathlib.Path(inspect.getfile(unwrapped))
    file_hash, _ = cache.hash_file(source_file)
    rel_path = project.to_relative_path(source_file)
    result[f"file:{rel_path}"] = file_hash

    code_deps: list[str] = getattr(func, "__pivot_code_deps__", [])
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
