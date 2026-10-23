# pyright: reportImplicitRelativeImport=false, reportMissingModuleSource=false

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from pivot.engine import graph as engine_graph
from pivot.engine.types import StageExecutionState

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["Scheduler"]


class Scheduler:
    """Synchronous, deterministic scheduler for stage execution.

    Owns all scheduling state (stage states, upstream/downstream maps, and
    scheduler-held resource permits) and produces deterministic decisions without
    async or IO.

    Determinism guarantee: when multiple stages are eligible to start, the
    tie-breaker is ``_stage_states`` iteration order, which preserves the
    ``execution_order`` passed to ``initialize()``. This order is seeded from
    the topological sort of the DAG.

    The Engine delegates all scheduling decisions to this class while retaining
    responsibility for async IO, event emission, and worker submission.
    """

    _stage_states: dict[str, StageExecutionState]
    _upstream_unfinished: dict[str, set[str]]
    _downstream: dict[str, list[str]]
    _stage_resources: dict[str, dict[str, int]]
    _resource_capacity: dict[str, int]
    _resource_counts: collections.defaultdict[str, int]
    _stop_starting_new: bool

    def __init__(self) -> None:
        self._stage_states = dict[str, StageExecutionState]()
        self._upstream_unfinished = dict[str, set[str]]()
        self._downstream = dict[str, list[str]]()
        self._stage_resources = dict[str, dict[str, int]]()
        self._resource_capacity = dict[str, int]()
        self._resource_counts = collections.defaultdict(int)
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
    def stage_resources(self) -> dict[str, dict[str, int]]:
        return self._stage_resources

    @property
    def downstream(self) -> dict[str, list[str]]:
        return self._downstream

    def initialize(
        self,
        execution_order: list[str],
        graph: nx.DiGraph[str] | None,
        *,
        stage_resources: dict[str, dict[str, int]],
        resource_capacity: dict[str, int],
    ) -> None:
        """Reset and configure scheduler state for a new execution."""
        self._resource_counts.clear()
        self._upstream_unfinished.clear()
        self._downstream.clear()
        self._stage_resources.clear()
        self._resource_capacity = dict(resource_capacity)
        self._stage_states.clear()
        self._stop_starting_new = False

        stages_set = set(execution_order)
        resource_keys = set(stage_resources)
        missing = stages_set - resource_keys
        extra = resource_keys - stages_set
        if missing or extra:
            parts = list[str]()
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"unknown: {sorted(extra)}")
            msg = f"stage resource inconsistency — {', '.join(parts)}"
            raise ValueError(msg)

        for resource, capacity in self._resource_capacity.items():
            if capacity <= 0:
                raise ValueError(f"Resource {resource!r} has non-positive capacity {capacity}")

        for stage_name in execution_order:
            if graph is not None:
                upstream = [
                    upstream_name
                    for upstream_name in engine_graph.get_upstream_stages(graph, stage_name)
                    if upstream_name in stages_set
                ]
                downstream = [
                    downstream_name
                    for downstream_name in engine_graph.get_downstream_stages(graph, stage_name)
                    if downstream_name in stages_set
                ]
            else:
                upstream = []
                downstream = []
            self._upstream_unfinished[stage_name] = set(upstream)
            self._downstream[stage_name] = downstream
            self._stage_resources[stage_name] = dict(stage_resources[stage_name])

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

    def can_start(self, stage: str) -> bool:
        """Return whether all scheduler-held resources are available for a ready stage."""
        if self.get_state(stage) != StageExecutionState.READY:
            return False
        if self._upstream_unfinished.get(stage):
            return False
        for resource, request in self._stage_resources[stage].items():
            if self._resource_counts[resource] + request > self._resource_capacity[resource]:
                return False
        return True

    def preview_direct_consumers(self, stage: str, *, limit: int) -> int:
        """Return how many direct consumers could start if ``stage`` completed now.

        The preview preserves stage-order admission and resource capacities. It has no scheduler
        side effects, so the Engine can start worker processes while ``stage`` runs without
        admitting its consumers before their inputs exist.
        """
        if limit <= 0:
            return 0
        resource_counts = collections.defaultdict(int, self._resource_counts)
        for resource, request in self._stage_resources[stage].items():
            resource_counts[resource] -= request
        # ``_downstream`` is transitive. The ``_upstream_unfinished`` check below keeps the
        # direct consumers whose only unfinished upstream stage is ``stage``.
        downstream = set(self._downstream.get(stage, []))
        count = 0
        for candidate, state in self._stage_states.items():
            if candidate not in downstream or state != StageExecutionState.PENDING:
                continue
            if self._upstream_unfinished[candidate] != {stage}:
                continue
            request = self._stage_resources[candidate]
            if any(
                resource_counts[resource] + amount > self._resource_capacity[resource]
                for resource, amount in request.items()
            ):
                continue
            for resource, amount in request.items():
                resource_counts[resource] += amount
            count += 1
            if count == limit:
                break
        return count

    def resource_width(self, *, limit: int) -> int:
        """Return the widest point of one replayed schedule of this run, at most ``limit``.

        The replay starts stages in stage order once their upstream stages have finished and
        their resources are free. Running stages finish one at a time, in the order they
        started, each only when nothing else can start. Resource capacities bound the result:
        four grades that share a two-slot resource count as two, and a chain counts as one. A
        schedule whose stages finish in another order can run more stages at once.
        """
        waiting = {stage: set(upstream) for stage, upstream in self._upstream_unfinished.items()}
        counts = collections.defaultdict[str, int](int)
        running = list[str]()
        pending = list(self._stage_states)
        widest = 0
        while pending or running:
            for stage in list(pending):
                if waiting[stage]:
                    continue
                request = self._stage_resources[stage]
                if any(
                    counts[resource] + amount > self._resource_capacity[resource]
                    for resource, amount in request.items()
                ):
                    continue
                for resource, amount in request.items():
                    counts[resource] += amount
                running.append(stage)
                pending.remove(stage)
            widest = max(widest, len(running))
            if widest >= limit:
                return limit
            if not running:
                return widest
            stage = running.pop(0)
            for resource, amount in self._stage_resources[stage].items():
                counts[resource] -= amount
            for consumer in self._downstream[stage]:
                waiting[consumer].discard(stage)
        return widest

    def acquire_resources(self, stage: str) -> None:
        """Reserve a stage's resources immediately before it is admitted."""
        for resource, request in self._stage_resources[stage].items():
            self._resource_counts[resource] += request

    def release_resources(self, stage: str) -> None:
        """Release all resources held for a completed stage."""
        for resource, request in self._stage_resources[stage].items():
            if self._resource_counts[resource] < request:
                msg = f"Resource {resource!r} released when not held (stage {stage!r})"
                raise ValueError(msg)
            self._resource_counts[resource] -= request

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
