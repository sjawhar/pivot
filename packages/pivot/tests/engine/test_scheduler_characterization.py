from __future__ import annotations

import networkx as nx
import pytest

from pivot.engine import graph as engine_graph
from pivot.engine.scheduler import Scheduler
from pivot.engine.types import NodeType, StageExecutionState
from pivot.executor import core as executor_core


def _helper_init_scheduler(
    *,
    execution_order: list[str],
    edges: list[tuple[str, str]] | None = None,
    stage_mutex: dict[str, list[str]] | None = None,
) -> Scheduler:
    """Create a Scheduler using only the public API.

    Args:
        execution_order: Stage names in topological order.
        edges: (upstream, downstream) stage dependency pairs.
        stage_mutex: Per-stage mutex list. Defaults to empty list per stage.
    """
    if stage_mutex is None:
        stage_mutex = {name: list[str]() for name in execution_order}

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
    scheduler.initialize(execution_order, g, stage_mutex=stage_mutex)
    return scheduler


def _helper_startable_in_order(scheduler: Scheduler, running_count: int) -> list[str]:
    startable: list[str] = []
    for name in list(scheduler.stage_states.keys()):
        if scheduler.can_start(name, running_count=running_count):
            startable.append(name)
    return startable


def test_can_start_requires_ready_and_no_upstream() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["upstream", "stage"],
        edges=[("upstream", "stage")],
    )

    assert scheduler.can_start("stage", running_count=0) is False, (
        "PENDING stage should not be startable"
    )

    scheduler.set_state("upstream", StageExecutionState.COMPLETED)
    scheduler.on_stage_completed("upstream", failed=False)
    assert scheduler.can_start("stage", running_count=0) is True, (
        "READY stage with no upstream should be startable"
    )

    scheduler = _helper_init_scheduler(
        execution_order=["upstream_a", "upstream_b", "stage"],
        edges=[("upstream_a", "stage"), ("upstream_b", "stage")],
    )
    scheduler.set_state("upstream_a", StageExecutionState.COMPLETED)
    scheduler.on_stage_completed("upstream_a", failed=False)
    assert scheduler.can_start("stage", running_count=0) is False, (
        "READY stage with unfinished upstream should not be startable"
    )


def test_can_start_respects_named_mutex() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["holder", "stage"],
        stage_mutex={"holder": ["mutex"], "stage": ["mutex"]},
    )

    scheduler.acquire_mutexes("holder")

    assert scheduler.can_start("stage", running_count=0) is False, (
        "stage should not start when its mutex is held"
    )

    scheduler.release_mutexes("holder")
    assert scheduler.can_start("stage", running_count=0) is True, (
        "stage should start when its mutex is released"
    )


def test_can_start_respects_exclusive_mutex_and_running() -> None:
    scheduler = _helper_init_scheduler(
        execution_order=["exclusive", "normal"],
        stage_mutex={"exclusive": [executor_core.EXCLUSIVE_MUTEX], "normal": []},
    )

    assert scheduler.can_start("exclusive", running_count=1) is False, (
        "exclusive stage should not start when other stages are running"
    )

    assert scheduler.can_start("exclusive", running_count=0) is True, (
        "exclusive stage should start when nothing else is running"
    )

    scheduler.acquire_mutexes("exclusive")
    assert scheduler.can_start("normal", running_count=0) is False, (
        "normal stage should not start when exclusive mutex is held"
    )


def test_ready_selection_respects_stage_state_insertion_order() -> None:
    scheduler = _helper_init_scheduler(execution_order=["second", "first", "third"])

    assert _helper_startable_in_order(scheduler, running_count=0) == [
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
        stage_mutex={"A": ["mutex"], "B": ["mutex"]},
    )

    scheduler.acquire_mutexes("A")
    scheduler.release_mutexes("A")
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


def test_release_mutexes_raises_on_underflow() -> None:
    """Releasing a mutex that was never acquired should raise ValueError."""
    scheduler = _helper_init_scheduler(
        execution_order=["stage"],
        stage_mutex={"stage": ["my_mutex"]},
    )
    # Release without acquire — should fail loud
    with pytest.raises(ValueError, match="Mutex.*released when not held"):
        scheduler.release_mutexes("stage")


def test_initialize_validates_stage_mutex_consistency() -> None:
    """initialize() raises if stage_mutex keys don't match execution_order."""
    scheduler = Scheduler()
    with pytest.raises(ValueError, match="stage_mutex"):
        scheduler.initialize(
            execution_order=["A", "B"],
            graph=None,
            stage_mutex={"A": [], "C": []},  # Missing B, has extra C
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
        stage_mutex={"stage": []},
    )
    assert scheduler.stop_starting_new is False, (
        "initialize() should reset stop_starting_new for watch-mode re-runs"
    )
