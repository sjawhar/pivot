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

from typing import TYPE_CHECKING

from pivot.engine import graph as engine_graph
from pivot.engine.types import StageExecutionState

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    import networkx as nx

__all__ = ["WatchCoordinator"]


class WatchCoordinator:
    """Policy and planning coordinator for watch-mode file change handling.

    Stateless with respect to execution — all execution state is accessed
    via callbacks (get_stage_state) rather than owned directly. This enables
    unit testing with synthetic graphs and state maps.
    """

    _graph: nx.DiGraph[str]

    def __init__(self, graph: nx.DiGraph[str]) -> None:
        self._graph = graph

    @property
    def graph(self) -> nx.DiGraph[str]:
        return self._graph

    @graph.setter
    def graph(self, g: nx.DiGraph[str]) -> None:
        self._graph = g

    def should_filter_path(
        self,
        path: pathlib.Path,
        *,
        get_stage_state: Callable[[str], StageExecutionState],
    ) -> bool:
        """Check if a path change should be filtered (produced by an executing stage).

        Returns True if the path's producer stage is currently between
        PREPARING and COMPLETED (exclusive) — i.e., PREPARING, WAITING_ON_LOCK,
        or RUNNING.
        """
        producer = engine_graph.get_producer(self._graph, path)
        if producer is None:
            return False
        state = get_stage_state(producer)
        return StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED

    def get_affected_stages(self, paths: list[pathlib.Path]) -> list[str]:
        """Get all stages affected by the given path changes (including downstream).

        Deduplicates across paths. Returns a sorted list for deterministic ordering.
        """
        affected = set[str]()
        for path in paths:
            consumers = engine_graph.get_consumers(self._graph, path)
            affected.update(consumers)
            for stage in consumers:
                downstream = engine_graph.get_downstream_stages(self._graph, stage)
                affected.update(downstream)
        return sorted(affected)

    def get_producer(self, path: pathlib.Path) -> str | None:
        """Get the stage that produces a given artifact path."""
        return engine_graph.get_producer(self._graph, path)

    def should_restart_workers(self, *, parallel: bool) -> bool:
        """Decide whether worker pool should restart after code/config change.

        Workers should restart to pick up reloaded code, but only when
        running in parallel mode (sequential mode has no persistent pool).
        """
        return parallel
