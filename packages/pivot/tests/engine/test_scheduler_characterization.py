from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import networkx as nx
import pytest

from pivot.engine import graph as engine_graph
from pivot.engine.scheduler import Scheduler
from pivot.engine.types import NodeType, StageExecutionState

if TYPE_CHECKING:
    import types


def _helper_init_scheduler(
    *,
    execution_order: list[str],
    edges: list[tuple[str, str]] | None = None,
    stage_resources: dict[str, dict[str, int]] | None = None,
    resource_capacity: dict[str, int] | None = None,
) -> Scheduler:
    """Create a Scheduler using only its public resource API."""
    if stage_resources is None:
        stage_resources = {name: {"cpu": 1} for name in execution_order}
    if resource_capacity is None:
        resource_capacity = {"cpu": len(execution_order)}

    if edges:
        g: nx.DiGraph[str] | None = nx.DiGraph()
        for name in execution_order:
            g.add_node(engine_graph.stage_node(name), type=NodeType.STAGE)
        for src, dst in edges:
            art = f"artifact:{src}__{dst}"
            g.add_node(art, type=NodeType.ARTIFACT)
            g.add_edge(engine_graph.stage_node(src), art)
            g.add_edge(art, engine_graph.stage_node(dst))
    else:
        g = None

    scheduler = Scheduler()
    scheduler.initialize(
        execution_order,
        g,
        stage_resources=stage_resources,
        resource_capacity=resource_capacity,
    )
    return scheduler


def _helper_startable_in_order(scheduler: Scheduler) -> list[str]:
    startable: list[str] = []
    for name in list(scheduler.stage_states.keys()):
        if scheduler.can_start(name):
            startable.append(name)
    return startable


def test_can_start_requires_ready_and_no_upstream() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["upstream", "stage"],
        edges=[("upstream", "stage")],
    )

    assert scheduler.can_start("stage") is False, "PENDING stage should not be startable"

    scheduler.set_state("upstream", StageExecutionState.COMPLETED)
    scheduler.on_stage_completed("upstream", failed=False)
    assert scheduler.can_start("stage") is True, "READY stage with no upstream should be startable"

    scheduler = _helper_init_scheduler(
        execution_order=["upstream_a", "upstream_b", "stage"],
        edges=[("upstream_a", "stage"), ("upstream_b", "stage")],
    )
    scheduler.set_state("upstream_a", StageExecutionState.COMPLETED)
    scheduler.on_stage_completed("upstream_a", failed=False)
    assert scheduler.can_start("stage") is False, "READY stage should wait for all upstreams"


def test_can_start_respects_resource_capacity() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["holder", "stage"],
        stage_resources={"holder": {"cpu": 1, "gpu": 1}, "stage": {"cpu": 1, "gpu": 1}},
        resource_capacity={"cpu": 2, "gpu": 1},
    )

    scheduler.acquire_resources("holder")

    assert scheduler.can_start("stage") is False, "stage should wait while GPU is held"

    scheduler.release_resources("holder")
    assert scheduler.can_start("stage") is True, "stage should start after GPU is released"


def test_can_start_respects_full_cpu_resource() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["exclusive", "normal"],
        stage_resources={"exclusive": {"cpu": 2}, "normal": {"cpu": 1}},
        resource_capacity={"cpu": 2},
    )

    assert scheduler.can_start("exclusive") is True
    scheduler.acquire_resources("exclusive")
    assert scheduler.can_start("normal") is False


def test_ready_selection_respects_stage_state_insertion_order() -> None:
    scheduler = _helper_init_scheduler(execution_order=["second", "first", "third"])

    assert _helper_startable_in_order(scheduler) == [
        "second",
        "first",
        "third",
    ], "startable stages should follow dict insertion order"


def test_chain_updates_downstream_after_completion() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["A", "B", "C"],
        edges=[("A", "B"), ("B", "C")],
    )

    newly_ready, newly_blocked = scheduler.on_stage_completed("A", failed=False)
    assert scheduler.get_state("B") == StageExecutionState.READY, (
        "B should become READY after A completes"
    )
    assert scheduler.get_state("C") == StageExecutionState.PENDING, (
        "C should stay PENDING because B hasn't completed"
    )
    assert "B" in newly_ready, "B should be in newly_ready list"
    assert newly_blocked == [], "no stages should be blocked on success"

    newly_ready, newly_blocked = scheduler.on_stage_completed("B", failed=False)
    assert scheduler.get_state("C") == StageExecutionState.READY, (
        "C should become READY after B completes"
    )
    assert "C" in newly_ready, "C should be in newly_ready list"


def test_fan_out_sets_ready_after_upstream_completion() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["A", "B", "C"],
        edges=[("A", "B"), ("A", "C")],
    )

    newly_ready, newly_blocked = scheduler.on_stage_completed("A", failed=False)
    assert scheduler.get_state("B") == StageExecutionState.READY, (
        "B should become READY after A completes"
    )
    assert scheduler.get_state("C") == StageExecutionState.READY, (
        "C should become READY after A completes"
    )
    assert set(newly_ready) == {"B", "C"}, "both B and C should be newly ready"
    assert newly_blocked == [], "no stages should be blocked on success"


def test_preview_direct_consumers_respects_resource_capacity() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["source", "grade_a", "grade_b", "grade_c", "grade_d"],
        edges=[
            ("source", "grade_a"),
            ("source", "grade_b"),
            ("source", "grade_c"),
            ("source", "grade_d"),
        ],
        stage_resources={
            "source": {"cpu": 1},
            "grade_a": {"cpu": 1, "local_grade": 1},
            "grade_b": {"cpu": 1, "local_grade": 1},
            "grade_c": {"cpu": 1, "local_grade": 1},
            "grade_d": {"cpu": 1, "local_grade": 1},
        },
        resource_capacity={"cpu": 8, "local_grade": 2},
    )
    scheduler.acquire_resources("source")
    scheduler.set_state("source", StageExecutionState.RUNNING)

    assert scheduler.preview_direct_consumers("source", limit=8) == 2


def test_preview_direct_consumers_keeps_chain_at_one_worker() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["source", "next"],
        edges=[("source", "next")],
    )
    scheduler.acquire_resources("source")
    scheduler.set_state("source", StageExecutionState.RUNNING)

    assert scheduler.preview_direct_consumers("source", limit=8) == 1


def test_resource_width_respects_resource_capacity() -> None:
    """A source feeding four grades that share a two-slot resource never runs more than two
    stages at once, though four grades are mutually independent."""
    scheduler = _helper_init_scheduler(
        execution_order=["source", "grade_a", "grade_b", "grade_c", "grade_d"],
        edges=[
            ("source", "grade_a"),
            ("source", "grade_b"),
            ("source", "grade_c"),
            ("source", "grade_d"),
        ],
        stage_resources={
            "source": {"cpu": 1},
            "grade_a": {"cpu": 1, "local_grade": 1},
            "grade_b": {"cpu": 1, "local_grade": 1},
            "grade_c": {"cpu": 1, "local_grade": 1},
            "grade_d": {"cpu": 1, "local_grade": 1},
        },
        resource_capacity={"cpu": 8, "local_grade": 2},
    )

    assert scheduler.resource_width(limit=8) == 2


def test_resource_width_keeps_chain_at_one() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c")],
    )

    assert scheduler.resource_width(limit=8) == 1


def test_resource_width_counts_a_fan_up_to_the_limit() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["source", "f0", "f1", "f2", "sink"],
        edges=[
            ("source", "f0"),
            ("source", "f1"),
            ("source", "f2"),
            ("f0", "sink"),
            ("f1", "sink"),
            ("f2", "sink"),
        ],
    )

    assert scheduler.resource_width(limit=8) == 3
    assert scheduler.resource_width(limit=2) == 2


def _helper_resource_width_lines(scheduler: Scheduler, *, limit: int) -> tuple[int, int]:
    """Return resource_width's result and the lines it executed: a count, not a clock, so
    machine load cannot change it."""
    code = Scheduler.resource_width.__code__
    tool = sys.monitoring.PROFILER_ID
    lines = 0

    def _count(_code: types.CodeType, _line: int) -> None:
        nonlocal lines
        lines += 1

    sys.monitoring.use_tool_id(tool, "resource_width step count")
    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, _count)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.LINE)
        width = scheduler.resource_width(limit=limit)
    finally:
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.NO_EVENTS)
        sys.monitoring.register_callback(tool, sys.monitoring.events.LINE, None)
        sys.monitoring.free_tool_id(tool)
    return width, lines


@pytest.mark.parametrize(
    "cpu",
    [
        pytest.param(16, id="no resource fills before the limit"),
        pytest.param(8, id="cpu fills before the limit"),
    ],
)
def test_resource_width_steps_grow_linearly_on_a_long_chain(cpu: int) -> None:
    """The replay executes a bounded number of lines per stage, so a long chain costs steps in
    proportion to its length, not to its square. A cpu capacity below the limit sends the
    replay through its heaps; 16 lets every ready stage start at once."""
    count = 1_000
    names = [f"s{index:04d}" for index in range(count)]
    scheduler = _helper_init_scheduler(
        execution_order=names,
        edges=list(zip(names, names[1:], strict=False)),
        resource_capacity={"cpu": cpu},
    )

    width, lines = _helper_resource_width_lines(scheduler, limit=16)

    assert width == 1
    # The heap replay runs about 54 lines per stage of a chain, the first-in first-out walk
    # about 8. A replay that rescans every pending stage per admission runs about 2,500 per
    # stage at this length, and more as it grows.
    assert lines <= 100 * count, f"resource_width executed {lines} lines for {count} stages"


def _helper_capped_fan(count: int, *, capacity: int) -> Scheduler:
    """A source feeding ``count`` grades that share one resource of ``capacity`` slots."""
    grades = [f"g{index:04d}" for index in range(count)]
    return _helper_init_scheduler(
        execution_order=["source", *grades],
        edges=[("source", grade) for grade in grades],
        stage_resources={"source": {"cpu": 1}}
        | {grade: {"cpu": 1, "shared": 1} for grade in grades},
        resource_capacity={"cpu": 16, "shared": capacity},
    )


def test_resource_width_stops_once_a_shared_resource_is_full() -> None:
    """Once the grades fill their shared resource, no later point of the replay can be wider,
    so the replay stops without finishing the remaining grades one by one."""
    count = 1_000
    scheduler = _helper_capped_fan(count, capacity=2)

    width, lines = _helper_resource_width_lines(scheduler, limit=16)

    assert width == 2
    assert lines <= 20 * count, f"resource_width executed {lines} lines for {count} grades"


def test_resource_width_counts_a_wave_after_a_shared_resource_drains() -> None:
    """The early stop counts stages still waiting behind a full resource: a fan of eight
    stages that runs after two grades on a one-slot resource is still counted."""
    fan = [f"f{index}" for index in range(8)]
    scheduler = _helper_init_scheduler(
        execution_order=["source", "g0", "g1", "join", *fan],
        edges=[
            ("source", "g0"),
            ("source", "g1"),
            ("g0", "join"),
            ("g1", "join"),
            *(("join", stage) for stage in fan),
        ],
        stage_resources={"source": {"cpu": 1}, "g0": {"cpu": 1, "shared": 1}}
        | {"g1": {"cpu": 1, "shared": 1}, "join": {"cpu": 1}}
        | {stage: {"cpu": 1} for stage in fan},
        resource_capacity={"cpu": 16, "shared": 1},
    )

    assert scheduler.resource_width(limit=16) == 8


def test_fan_in_waits_for_all_upstream() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["A", "B", "C"],
        edges=[("A", "C"), ("B", "C")],
    )

    newly_ready, newly_blocked = scheduler.on_stage_completed("A", failed=False)
    assert scheduler.get_state("C") == StageExecutionState.PENDING, (
        "C should stay PENDING when only A has completed"
    )
    assert newly_ready == [], "nothing should be newly ready after just A"
    assert newly_blocked == [], "no stages should be blocked on success"

    newly_ready, newly_blocked = scheduler.on_stage_completed("B", failed=False)
    assert scheduler.get_state("C") == StageExecutionState.READY, (
        "C should become READY after both A and B complete"
    )
    assert "C" in newly_ready, "C should be newly ready"


def test_on_stage_completed_cascades_failure() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["A", "B"],
        edges=[("A", "B")],
    )
    newly_ready, newly_blocked = scheduler.on_stage_completed("A", failed=True)

    assert scheduler.get_state("B") == StageExecutionState.BLOCKED, (
        "B should be BLOCKED when upstream A failed"
    )
    assert newly_ready == [], "B should NOT become ready when upstream failed"
    assert len(newly_blocked) == 1, "one stage should be newly blocked"
    assert newly_blocked[0][0] == "B", "B should be in newly_blocked"


def test_apply_fail_fast_blocks_ready_and_pending() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["blocker", "pending_dep", "failed", "ready", "pending", "blocked"],
        edges=[("blocker", "blocked"), ("pending_dep", "pending")],
    )
    scheduler.set_state("blocker", StageExecutionState.COMPLETED)
    scheduler.on_stage_completed("blocker", failed=True)
    scheduler.set_state("failed", StageExecutionState.COMPLETED)

    blocked = scheduler.apply_fail_fast()

    assert scheduler.stop_starting_new is True, "stop_starting_new should be set"
    assert scheduler.get_state("pending") == StageExecutionState.BLOCKED, (
        "pending stage should be BLOCKED after fail-fast"
    )
    assert scheduler.get_state("ready") == StageExecutionState.BLOCKED, (
        "ready stage should be BLOCKED after fail-fast"
    )

    blocked_names = {name for name, _ in blocked}
    assert "pending" in blocked_names, "pending should appear in blocked return"
    assert "ready" in blocked_names, "ready should appear in blocked return"

    # Verify return value contains correct old states
    old_states = dict(blocked)
    assert old_states["ready"] == StageExecutionState.READY, (
        "previous_state should be READY for a stage that was READY before fail-fast"
    )
    assert old_states["pending"] == StageExecutionState.PENDING, (
        "previous_state should be PENDING for a stage that was PENDING before fail-fast"
    )


def test_apply_cancel_marks_ready_and_pending_completed() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["pending_dep", "ready", "pending", "running"],
        edges=[("pending_dep", "pending")],
    )
    scheduler.set_state("running", StageExecutionState.RUNNING)

    cancelled = scheduler.apply_cancel()

    assert scheduler.stop_starting_new is True, "stop_starting_new should be set"
    assert scheduler.get_state("ready") == StageExecutionState.COMPLETED, (
        "ready stage should be COMPLETED after cancel"
    )
    assert scheduler.get_state("pending") == StageExecutionState.COMPLETED, (
        "pending stage should be COMPLETED after cancel"
    )
    assert scheduler.get_state("running") == StageExecutionState.RUNNING, (
        "running stage should remain RUNNING after cancel"
    )

    # Verify return value contains correct old states
    old_states = dict(cancelled)
    assert old_states["ready"] == StageExecutionState.READY, (
        "previous_state should be READY for a stage that was READY before cancel"
    )
    assert old_states["pending"] == StageExecutionState.PENDING, (
        "previous_state should be PENDING for a stage that was PENDING before cancel"
    )


def test_state_enum_comparison_for_monotonicity_guard() -> None:
    """Verify IntEnum comparison works for the drain thread's monotonicity guard.

    The drain thread uses `current >= new_state` to skip backward transitions.
    This test verifies that StageExecutionState IntEnum values support this comparison.
    """
    # Verify enum values are ordered correctly
    assert StageExecutionState.PENDING < StageExecutionState.BLOCKED
    assert StageExecutionState.BLOCKED < StageExecutionState.READY
    assert StageExecutionState.READY < StageExecutionState.PREPARING
    assert StageExecutionState.PREPARING < StageExecutionState.WAITING_ON_LOCK
    assert StageExecutionState.WAITING_ON_LOCK < StageExecutionState.RUNNING
    assert StageExecutionState.RUNNING < StageExecutionState.COMPLETED

    # Test the guard logic: if current >= new_state, skip the transition
    scheduler = _helper_init_scheduler(execution_order=["stage"])
    scheduler.set_state("stage", StageExecutionState.COMPLETED)

    # Verify get_state returns the current state
    current = scheduler.get_state("stage")
    assert current == StageExecutionState.COMPLETED, "get_state should return COMPLETED"

    # Verify the guard condition: COMPLETED >= RUNNING should be True
    # (meaning we should skip the transition)
    assert current >= StageExecutionState.RUNNING, (
        "COMPLETED should be >= RUNNING for guard to skip"
    )

    # Verify the guard condition: COMPLETED >= WAITING_ON_LOCK should be True
    assert current >= StageExecutionState.WAITING_ON_LOCK, (
        "COMPLETED should be >= WAITING_ON_LOCK for guard to skip"
    )

    # Verify the guard condition: COMPLETED >= COMPLETED should be True
    assert current >= StageExecutionState.COMPLETED, (
        "COMPLETED should be >= COMPLETED for guard to skip duplicate"
    )


def test_release_resources_raises_on_underflow() -> None:
    """Releasing resources that were never acquired fails loudly."""
    scheduler = _helper_init_scheduler(
        execution_order=["stage"],
        stage_resources={"stage": {"cpu": 1, "gpu": 1}},
        resource_capacity={"cpu": 1, "gpu": 1},
    )
    with pytest.raises(ValueError, match="Resource.*released when not held"):
        scheduler.release_resources("stage")


def test_initialize_validates_stage_resource_consistency() -> None:
    """initialize() raises when resource mapping keys don't match execution order."""
    scheduler = Scheduler()
    with pytest.raises(ValueError, match="stage resource"):
        scheduler.initialize(
            execution_order=["A", "B"],
            graph=None,
            stage_resources={"A": {"cpu": 1}, "C": {"cpu": 1}},
            resource_capacity={"cpu": 2},
        )


def test_cascade_failure_blocks_all_transitive_downstream() -> None:
    """When A fails, both B (direct) and C (transitive) get BLOCKED."""
    scheduler = _helper_init_scheduler(
        execution_order=["A", "B", "C"],
        edges=[("A", "B"), ("B", "C")],
    )

    scheduler.set_state("A", StageExecutionState.COMPLETED)
    _newly_ready, newly_blocked = scheduler.on_stage_completed("A", failed=True)

    assert scheduler.get_state("B") == StageExecutionState.BLOCKED, (
        "B (direct downstream) should be BLOCKED"
    )
    assert scheduler.get_state("C") == StageExecutionState.BLOCKED, (
        "C (transitive downstream) should also be BLOCKED"
    )
    blocked_names = {name for name, _ in newly_blocked}
    assert "B" in blocked_names
    assert "C" in blocked_names


def test_initialize_resets_stop_starting_new() -> None:
    """initialize() resets stop_starting_new after fail-fast or cancel."""
    scheduler = _helper_init_scheduler(execution_order=["stage"])
    scheduler.apply_fail_fast()
    assert scheduler.stop_starting_new is True, "fail-fast should set stop_starting_new"

    # Re-initialize should reset
    scheduler.initialize(
        execution_order=["stage"],
        graph=None,
        stage_resources={"stage": {"cpu": 1}},
        resource_capacity={"cpu": 1},
    )
    assert scheduler.stop_starting_new is False, (
        "initialize() should reset stop_starting_new for watch-mode re-runs"
    )
