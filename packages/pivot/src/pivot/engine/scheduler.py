# pyright: reportImplicitRelativeImport=false, reportMissingModuleSource=false

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from pivot.engine import graph as engine_graph
from pivot.engine.types import StageExecutionState
from pivot.executor import core as executor_core

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["Scheduler"]


class Scheduler:
    """Synchronous, deterministic scheduler for stage execution.

    Owns all scheduling state (stage states, upstream/downstream maps, mutex
    counts) and produces deterministic decisions without async or IO.

    Determinism guarantee: when multiple stages are eligible to start,
    the tie-breaker is ``_stage_states`` iteration order, which preserves
    the ``execution_order`` passed to ``initialize()``.  This order is
    seeded from the topological sort of the DAG.

    The Engine delegates all scheduling decisions to this class while
    retaining responsibility for async IO, event emission, and worker
    submission.
    """

    _stage_states: dict[str, StageExecutionState]
    _upstream_unfinished: dict[str, set[str]]
    _downstream: dict[str, list[str]]
    _stage_mutex: dict[str, list[str]]
    _mutex_counts: collections.defaultdict[str, int]
    _stop_starting_new: bool

    def __init__(self) -> None:
        self._stage_states = dict[str, StageExecutionState]()
        self._upstream_unfinished = dict[str, set[str]]()
        self._downstream = dict[str, list[str]]()
        self._stage_mutex = dict[str, list[str]]()
        self._mutex_counts = collections.defaultdict(int)
        self._stop_starting_new = False

    @property
    def stage_states(self) -> dict[str, StageExecutionState]:
        return self._stage_states

    @property
    def stop_starting_new(self) -> bool:
        return self._stop_starting_new

    @stop_starting_new.setter
    def stop_starting_new(self, value: bool) -> None:
        self._stop_starting_new = value

    @property
    def stage_mutex(self) -> dict[str, list[str]]:
        return self._stage_mutex

    @property
    def downstream(self) -> dict[str, list[str]]:
        return self._downstream

    def initialize(
        self,
        execution_order: list[str],
        graph: nx.DiGraph[str] | None,
        *,
        stage_mutex: dict[str, list[str]],
    ) -> None:
        """Reset and configure scheduler for a new execution.

        Args:
            execution_order: Stage names in topological order. This order
                determines the determinism tie-breaker: when multiple stages
                are eligible to start simultaneously, they are considered in
                ``execution_order`` sequence (i.e., dict insertion order of
                ``_stage_states``).
            graph: Bipartite artifact-stage graph for deriving upstream/downstream
                relationships. Pass None for single-stage or no-dependency runs.
            stage_mutex: Mapping of stage name to mutex group names. Must contain
                exactly the same keys as ``execution_order``.

        Raises:
            ValueError: If ``stage_mutex`` keys don't match ``execution_order``.
        """
        self._mutex_counts.clear()
        self._upstream_unfinished.clear()
        self._downstream.clear()
        self._stage_mutex.clear()
        self._stage_states.clear()
        self._stop_starting_new = False

        stages_set = set(execution_order)
        mutex_keys = set(stage_mutex.keys())
        missing = stages_set - mutex_keys
        extra = mutex_keys - stages_set
        if missing or extra:
            parts = list[str]()
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"unknown: {sorted(extra)}")
            msg = f"stage_mutex inconsistency — {', '.join(parts)}"
            raise ValueError(msg)

        for stage_name in execution_order:
            if graph is not None:
                upstream = [
                    upstream_name
                    for upstream_name in engine_graph.get_upstream_stages(graph, stage_name)
                    if upstream_name in stages_set
                ]
            else:
                upstream = []
            self._upstream_unfinished[stage_name] = set(upstream)

            if graph is not None:
                downstream = [
                    downstream_name
                    for downstream_name in engine_graph.get_downstream_stages(graph, stage_name)
                    if downstream_name in stages_set
                ]
            else:
                downstream = []
            self._downstream[stage_name] = downstream

            self._stage_mutex[stage_name] = stage_mutex[stage_name]

            initial_state = (
                StageExecutionState.READY if not upstream else StageExecutionState.PENDING
            )
            _ = self.set_state(stage_name, initial_state)

    def get_state(self, stage: str) -> StageExecutionState:
        return self._stage_states.get(stage, StageExecutionState.PENDING)

    def set_state(
        self, stage: str, new_state: StageExecutionState
    ) -> tuple[StageExecutionState, bool]:
        old_state = self._stage_states.get(stage, StageExecutionState.PENDING)
        is_new = stage not in self._stage_states
        if not is_new and old_state == new_state:
            return old_state, False
        self._stage_states[stage] = new_state
        return old_state, True

    def can_start(self, stage: str, *, running_count: int) -> bool:
        if self.get_state(stage) != StageExecutionState.READY:
            return False

        if self._upstream_unfinished.get(stage):
            return False

        stage_mutexes = self._stage_mutex.get(stage, [])
        is_exclusive = executor_core.EXCLUSIVE_MUTEX in stage_mutexes

        for mutex in stage_mutexes:
            if mutex == executor_core.EXCLUSIVE_MUTEX:
                if self._mutex_counts[mutex] > 0 or running_count > 0:
                    return False
            elif self._mutex_counts[mutex] > 0:
                return False

        return is_exclusive or self._mutex_counts[executor_core.EXCLUSIVE_MUTEX] == 0

    def acquire_mutexes(self, stage: str) -> None:
        for mutex in self._stage_mutex.get(stage, []):
            self._mutex_counts[mutex] += 1

    def release_mutexes(self, stage: str) -> None:
        for mutex in self._stage_mutex.get(stage, []):
            if self._mutex_counts[mutex] <= 0:
                msg = f"Mutex '{mutex}' released when not held (stage '{stage}')"
                raise ValueError(msg)
            self._mutex_counts[mutex] -= 1

    def on_stage_completed(
        self, stage: str, failed: bool
    ) -> tuple[list[str], list[tuple[str, StageExecutionState]]]:
        newly_ready = list[str]()
        for downstream_name in self._downstream.get(stage, []):
            unfinished = self._upstream_unfinished.get(downstream_name)
            if unfinished is None:
                continue
            unfinished.discard(stage)
            if (
                not failed
                and not unfinished
                and self.get_state(downstream_name) == StageExecutionState.PENDING
            ):
                _ = self.set_state(downstream_name, StageExecutionState.READY)
                newly_ready.append(downstream_name)

        if failed:
            newly_blocked = self._cascade_failure(stage)
        else:
            newly_blocked = list[tuple[str, StageExecutionState]]()

        return newly_ready, newly_blocked

    def _cascade_failure(self, failed_stage: str) -> list[tuple[str, StageExecutionState]]:
        newly_blocked = list[tuple[str, StageExecutionState]]()
        for downstream_name in self._downstream.get(failed_stage, []):
            state = self.get_state(downstream_name)
            if state in (StageExecutionState.PENDING, StageExecutionState.READY):
                _ = self.set_state(downstream_name, StageExecutionState.BLOCKED)
                newly_blocked.append((downstream_name, state))
        return newly_blocked

    def apply_fail_fast(self) -> list[tuple[str, StageExecutionState]]:
        self._stop_starting_new = True
        blocked = list[tuple[str, StageExecutionState]]()
        for name, state in self._stage_states.items():
            if state in (StageExecutionState.READY, StageExecutionState.PENDING):
                _ = self.set_state(name, StageExecutionState.BLOCKED)
                blocked.append((name, state))
        return blocked

    def apply_cancel(self) -> list[tuple[str, StageExecutionState]]:
        self._stop_starting_new = True
        cancelled = list[tuple[str, StageExecutionState]]()
        for name, state in self._stage_states.items():
            if state in (StageExecutionState.READY, StageExecutionState.PENDING):
                _ = self.set_state(name, StageExecutionState.COMPLETED)
                cancelled.append((name, state))
        return cancelled
