"""Watch-mode coordinator: owns policy and planning for watch-triggered actions.

The WatchCoordinator decides *what* should happen in response to file changes.
Engine performs the actual *execution* (state mutations, event emission, run lifecycle).

Responsibilities owned by WatchCoordinator:
- Affected-stage computation (which stages to run after a file change)
- Path filtering (should events for this path be deferred/ignored?)
- Worker restart policy (should workers restart after code reload?)

Responsibilities retained by Engine:
- Pipeline reload mechanics (sys.modules, discovery, fingerprint caches)
- Run state machine (cancel/coalesce, run_id generation, task groups)
- Event emission (OutputEvents to sinks)
- Deferred event storage/processing (tied to execution lifecycle)
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import project
from pivot import types as pivot_types
from pivot.engine import graph as engine_graph
from pivot.engine.types import StageExecutionState

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from pivot.registry import RegistryStageInfo
    from pivot.storage.store import WorkspaceStore

__all__ = ["WatchCoordinator", "build_artifact_paths", "get_watch_paths"]


def build_artifact_paths(
    stages: dict[str, RegistryStageInfo],
    store: WorkspaceStore,
) -> dict[pathlib.Path, pivot_types.ArtifactIdentity]:
    """Resolve every pipeline artifact identity to its workspace path."""
    paths = dict[pathlib.Path, pivot_types.ArtifactIdentity]()
    output_identities = set[pivot_types.ArtifactIdentity]()

    for info in stages.values():
        for out in info["outs"]:
            path = project.normalize_path(store.resolve_display_path(out))
            paths[path] = out.identity
            output_identities.add(out.identity)

    for info in stages.values():
        for dep in info["deps"].values():
            if dep.identity in output_identities:
                continue
            path = project.normalize_path(store.checkout(dep))
            paths[path] = dep.identity

    return paths


def get_watch_paths(
    project_root: pathlib.Path,
    artifact_paths: dict[pathlib.Path, pivot_types.ArtifactIdentity],
) -> list[pathlib.Path]:
    """Watch project sources plus existing artifact paths outside the project."""
    root = project.normalize_path(project_root)
    paths = {root}
    for path in artifact_paths:
        if path.is_relative_to(root) or not path.exists():
            continue
        paths.add(path if path.is_dir() else path.parent)
    return sorted(paths)


class WatchCoordinator:
    """Policy and planning coordinator for watch-mode file change handling.

    Stateless with respect to execution — all execution state is accessed
    via callbacks (get_stage_state) rather than owned directly. This enables
    unit testing with synthetic graphs and state maps.
    """

    _graph: nx.DiGraph[str]
    _artifact_paths: dict[pathlib.Path, pivot_types.ArtifactIdentity]

    def __init__(
        self,
        graph: nx.DiGraph[str],
        artifact_paths: dict[pathlib.Path, pivot_types.ArtifactIdentity],
    ) -> None:
        self._graph = graph
        self._artifact_paths = artifact_paths

    @property
    def graph(self) -> nx.DiGraph[str]:
        return self._graph

    @property
    def artifact_paths(self) -> dict[pathlib.Path, pivot_types.ArtifactIdentity]:
        """Filesystem paths resolved for the graph's artifact identities."""
        return self._artifact_paths

    def resolve_identity(self, path: pathlib.Path) -> pivot_types.ArtifactIdentity | None:
        """Resolve a filesystem event path to the matching artifact identity."""
        current = project.normalize_path(path)
        while True:
            identity = self._artifact_paths.get(current)
            if identity is not None:
                return identity
            parent = current.parent
            if parent == current:
                return None
            current = parent

    def _matching_identities(self, path: pathlib.Path) -> list[pivot_types.ArtifactIdentity]:
        """Return identities whose resolved paths contain a filesystem event path."""
        current = project.normalize_path(path)
        identities = list[pivot_types.ArtifactIdentity]()
        while True:
            identity = self._artifact_paths.get(current)
            if identity is not None:
                identities.append(identity)
            parent = current.parent
            if parent == current:
                return identities
            current = parent

    def _matching_resolved_input_identities(
        self, path: pathlib.Path
    ) -> list[pivot_types.ArtifactIdentity]:
        """Return input identities whose resolved paths contain a resolved event path."""
        resolved_path = path.resolve()
        identities = list[pivot_types.ArtifactIdentity]()
        for input_path, identity in self._artifact_paths.items():
            if engine_graph.get_producer(self._graph, identity) is not None:
                continue
            if resolved_path.is_relative_to(input_path.resolve()):
                identities.append(identity)
        return identities

    def should_filter_path(
        self,
        path: pathlib.Path,
        *,
        get_stage_state: Callable[[str], StageExecutionState],
    ) -> bool:
        """Check if a path change should be filtered (produced by an executing stage)."""
        identity = self.resolve_identity(path)
        if identity is None:
            return False
        producer = engine_graph.get_producer(self._graph, identity)
        if producer is None:
            return False
        state = get_stage_state(producer)
        return StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED

    def get_affected_stages(self, paths: list[pathlib.Path]) -> list[str]:
        """Get all stages affected by the changed paths, including downstream."""
        affected = set[str]()
        for path in paths:
            identities = set(self._matching_identities(path))
            identities.update(self._matching_resolved_input_identities(path))
            for identity in identities:
                consumers = engine_graph.get_consumers(self._graph, identity)
                affected.update(consumers)
                for stage in consumers:
                    affected.update(engine_graph.get_downstream_stages(self._graph, stage))
        return sorted(affected)

    def get_producer(self, path: pathlib.Path) -> str | None:
        """Get the stage that produces a given artifact path."""
        identity = self.resolve_identity(path)
        if identity is None:
            return None
        return engine_graph.get_producer(self._graph, identity)

    def should_restart_workers(self, *, parallel: bool) -> bool:
        """Decide whether worker pool should restart after code/config change.

        Workers should restart to pick up reloaded code, but only when
        running in parallel mode (sequential mode has no persistent pool).
        """
        return parallel
