# pyright: reportImplicitRelativeImport=false, reportMissingModuleSource=false

from __future__ import annotations

import collections
import heapq
import itertools
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
    _failed: set[str]
    _stop_starting_new: bool

    def __init__(self) -> None:
        self._stage_states = dict[str, StageExecutionState]()
        self._upstream_unfinished = dict[str, set[str]]()
        self._downstream = dict[str, list[str]]()
        self._stage_resources = dict[str, dict[str, int]]()
        self._resource_capacity = dict[str, int]()
        self._resource_counts = collections.defaultdict(int)
        self._failed = set[str]()
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
        """Reset and configure scheduler state for a new execution.

        Raises:
            ValueError: ``stage_resources`` does not name exactly the stages of
                ``execution_order``, a capacity is below 1, a request is below 1, or a stage
                requests a resource with no capacity.
        """
        self._resource_counts.clear()
        self._upstream_unfinished.clear()
        self._downstream.clear()
        self._stage_resources.clear()
        self._resource_capacity = dict(resource_capacity)
        self._stage_states.clear()
        self._failed.clear()
        self._stop_starting_new = False

        stages_set = set(execution_order)
        self._check_requests(stages_set, stage_resources)

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

    def _check_requests(self, stages: set[str], stage_resources: dict[str, dict[str, int]]) -> None:
        """Raise ValueError unless ``stage_resources`` names exactly ``stages``, every capacity
        is at least 1, each request is at least 1, and each requested resource has a capacity."""
        resource_keys = set(stage_resources)
        missing = stages - resource_keys
        extra = resource_keys - stages
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
        for stage_name, request in stage_resources.items():
            for resource, amount in request.items():
                if amount <= 0:
                    msg = (
                        f"Stage {stage_name!r} requests non-positive amount {amount} "
                        + f"of resource {resource!r}"
                    )
                    raise ValueError(msg)
                if resource not in self._resource_capacity:
                    msg = f"Stage {stage_name!r} requests resource {resource!r} with no capacity"
                    raise ValueError(msg)

    def extend(
        self,
        stages: list[str],
        graph: nx.DiGraph[str],
        *,
        stage_resources: dict[str, dict[str, int]],
    ) -> list[tuple[str, StageExecutionState]]:
        """Add ``stages`` to the schedule under way, after every stage it holds, and return each
        with its first state.

        A run adds a template's instances this way (``Pipeline.map``); ``graph`` already holds
        them (``engine_graph.add_stages``). Every stage already scheduled keeps its state. A new
        stage waits for each upstream stage of the run not yet finished, earlier new stages
        included, and is BLOCKED when an upstream stage failed or is blocked. Each upstream
        stage, however far up, counts the new stage downstream, so a later failure blocks it.

        Raises:
            ValueError: As ``initialize`` does for ``stage_resources``.
        """
        self._check_requests(set(stages), stage_resources)
        members = set(self._stage_states) | set(stages)
        added = list[tuple[str, StageExecutionState]]()
        for stage_name in stages:
            upstream = [
                name
                for name in engine_graph.get_upstream_stages(graph, stage_name)
                if name in members
            ]
            for ancestor in self._ancestors(graph, upstream, members):
                self._downstream[ancestor].append(stage_name)
            self._downstream[stage_name] = []
            self._stage_resources[stage_name] = dict(stage_resources[stage_name])
            self._upstream_unfinished[stage_name] = {
                name for name in upstream if self.get_state(name) != StageExecutionState.COMPLETED
            }
            if any(
                name in self._failed or self.get_state(name) == StageExecutionState.BLOCKED
                for name in upstream
            ):
                state = StageExecutionState.BLOCKED
            elif self._upstream_unfinished[stage_name]:
                state = StageExecutionState.PENDING
            else:
                state = StageExecutionState.READY
            _ = self.set_state(stage_name, state)
            added.append((stage_name, state))
        return added

    @staticmethod
    def _ancestors(graph: nx.DiGraph[str], upstream: list[str], members: set[str]) -> set[str]:
        """``upstream`` and every stage of ``members`` above it in ``graph``."""
        found = set(upstream)
        pending = list(upstream)
        while pending:
            for name in engine_graph.get_upstream_stages(graph, pending.pop()):
                if name in members and name not in found:
                    found.add(name)
                    pending.append(name)
        return found

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
        if not downstream:
            return 0
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

        When no resource can fill before ``limit`` stages run, every ready stage starts at once,
        so the replay is a first-in first-out walk of the stages.

        Otherwise a ready stage that does not fit waits in a group: the first resource in its
        request that blocks it, and the amount it needs of that resource. Each resource keeps a
        heap of the amounts its waiting groups need, so a finish reaches only the groups that
        its release opens, and wakes the earliest stage of each. A woken stage that starts or
        waits again wakes the next one while the group stays open. So an admission looks only
        at stages that a release could have let in. The replay also stops once no set of
        unfinished stages can run wider than the widest point so far: per request, at most as
        many stages as its resources hold, and per resource, at most its capacity of its users.

        Below a step budget the result equals a full replay. Above it the method returns
        ``limit``. The Engine passes ``get_width``'s pool size as ``limit``, so a run over the
        budget starts that pool size, the most workers it may hold at once. The budget is
        ``engine_graph._AUGMENT_STEPS`` steps plus ``engine_graph._REPLAY_STEPS_PER_INPUT`` per
        stage, dependency and requested resource. Random DAGs of 400 or more stages, each stage
        requesting several small resources at random amounts, can exceed it: 22 of 299 at 400
        stages and 4 of 30 at 1,000 stages did.

        An exact answer can cost more than ``get_width``. Give each of 2n independent stages
        one-slot resources that encode an Orthogonal Vectors instance, with every first-half
        stage also on one shared one-slot resource and every second-half stage on another. The
        replay then reaches width 2 exactly when some first-half and second-half stages share
        no resource, which takes about n² steps to decide unless the Strong Exponential Time
        Hypothesis fails. With the budget the method takes O((n + e + r) log(n + r)) time for
        n stages, e dependencies and r requested resources.

        Raises:
            ValueError: ``limit`` is below 1.
        """
        if limit < 1:
            raise ValueError(f"resource_width limit must be at least 1, got {limit}")
        capacity = self._resource_capacity
        # initialize() fills _stage_resources and _upstream_unfinished in the same stage order.
        requests = [tuple(request.items()) for request in self._stage_resources.values()]
        kinds = collections.Counter(requests)
        unfinished = [len(upstream) for upstream in self._upstream_unfinished.values()]
        consumers: list[list[int]] | None = None
        if all(limit * amount <= capacity[resource] for kind in kinds for resource, amount in kind):
            started = collections.deque(
                position for position, count in enumerate(unfinished) if not count
            )
            widest = len(started)
            while started and widest < limit:
                if consumers is None:
                    consumers = self._consumer_positions()
                for consumer in consumers[started.popleft()]:
                    unfinished[consumer] -= 1
                    if not unfinished[consumer]:
                        started.append(consumer)
                widest = max(widest, len(started))
            return min(widest, limit)

        stages = len(requests)
        inputs = stages + sum(unfinished) + sum(map(len, requests))
        # The step budgets of get_width and resource_width live together in engine.graph.
        steps_per_input = engine_graph._REPLAY_STEPS_PER_INPUT  # pyright: ignore[reportPrivateUsage]
        budget = engine_graph._AUGMENT_STEPS + steps_per_input * inputs  # pyright: ignore[reportPrivateUsage]
        # Per request: the most stages with it that its resources hold at once. A request that
        # only one stage makes holds at most that stage.
        kind_slots = dict[tuple[tuple[str, int], ...], int]()
        kind_bound = 0
        for kind, count in kinds.items():
            if count > 1:
                for resource, amount in kind:
                    held = capacity[resource] // amount
                    if held < count:
                        count = held
                kind_slots[kind] = count
            kind_bound += count
        # Per resource: its unfinished users beyond its capacity, which never run at once.
        users = collections.Counter(itertools.chain.from_iterable(self._stage_resources.values()))
        excess = {
            resource: count - capacity[resource]
            for resource, count in users.items()
            if count > capacity[resource]
        }
        excess_heap = [(-over, resource) for resource, over in excess.items()]
        heapq.heapify(excess_heap)
        remaining = stages
        bound = min(kind_bound, remaining - (-excess_heap[0][0] if excess_heap else 0))

        # Stages to check, in stage order. Appended in stage order, so already a heap.
        candidates = [position for position, count in enumerate(unfinished) if not count]
        # A group is a resource and an amount of it. Each group's waiting stages, a heap
        waiting = collections.defaultdict[tuple[str, int], list[int]](list)
        # Per resource, a heap of the amounts whose groups wait for a release
        closed = collections.defaultdict[str, list[int]](list)
        woken = set[tuple[str, int]]()  # groups with a stage among the candidates
        woken_from: list[tuple[str, int] | None] = [None] * stages
        counts = dict.fromkeys(capacity, 0)
        running = collections.deque[int]()
        widest = 0
        steps = 0
        while True:
            while candidates:
                position = heapq.heappop(candidates)
                request = requests[position]
                steps += 1 + len(request)
                for resource, amount in request:
                    if counts[resource] + amount > capacity[resource]:
                        group = (resource, amount)
                        queue = waiting[group]
                        if not queue and group not in woken:
                            heapq.heappush(closed[resource], amount)
                        heapq.heappush(queue, position)
                        break
                else:
                    for resource, amount in request:
                        counts[resource] += amount
                    running.append(position)
                    if len(running) >= limit:
                        return limit
                    if len(running) >= bound:
                        return max(widest, len(running))
                group = woken_from[position]
                if group is not None:
                    woken_from[position] = None
                    woken.discard(group)
                    resource, amount = group
                    if waiting[group]:
                        if counts[resource] + amount <= capacity[resource]:
                            nxt = heapq.heappop(waiting[group])
                            woken_from[nxt] = group
                            woken.add(group)
                            heapq.heappush(candidates, nxt)
                        else:
                            heapq.heappush(closed[resource], amount)
                if steps > budget:
                    return limit
            widest = max(widest, len(running))
            if not running or widest >= bound:
                return widest
            if consumers is None:
                consumers = self._consumer_positions()
            finished = running.popleft()
            request = requests[finished]
            for resource, amount in request:
                counts[resource] -= amount
                if resource in excess:
                    excess[resource] -= 1
                # Every waiting group on this resource needs more than was free before the
                # release, so the groups that the release lets in are its smallest amounts
                amounts = closed[resource]
                free = capacity[resource] - counts[resource]
                while amounts and amounts[0] <= free:
                    steps += 1
                    group = (resource, heapq.heappop(amounts))
                    nxt = heapq.heappop(waiting[group])
                    woken_from[nxt] = group
                    woken.add(group)
                    heapq.heappush(candidates, nxt)
            remaining -= 1
            if kinds[request] <= kind_slots.get(request, 1):
                kind_bound -= 1
            kinds[request] -= 1
            while excess_heap and -excess_heap[0][0] != excess[excess_heap[0][1]]:
                _, resource = heapq.heappop(excess_heap)
                if excess[resource] > 0:
                    heapq.heappush(excess_heap, (-excess[resource], resource))
            bound = min(kind_bound, remaining - (-excess_heap[0][0] if excess_heap else 0))
            if widest >= bound:
                return widest
            for consumer in consumers[finished]:
                unfinished[consumer] -= 1
                if not unfinished[consumer]:
                    heapq.heappush(candidates, consumer)
            if steps > budget:
                return limit

    def _consumer_positions(self) -> list[list[int]]:
        """Each stage's direct consumers, by position in stage order."""
        position = dict(zip(self._upstream_unfinished, itertools.count()))
        # A literal: list[int]() makes a generic-alias call per stage, three times the cost.
        consumers: list[list[int]] = [[] for _ in position]
        for index, upstream in enumerate(self._upstream_unfinished.values()):
            for producer in upstream:
                consumers[position[producer]].append(index)
        return consumers

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
            self._failed.add(stage)
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
