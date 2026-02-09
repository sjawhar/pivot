from __future__ import annotations

# pyright: reportPrivateUsage=false
import collections

from pivot.engine.scheduler import Scheduler
from pivot.engine.types import StageExecutionState
from pivot.executor import core as executor_core


def _helper_make_scheduler(
    *,
    stage_states: dict[str, StageExecutionState],
    upstream_unfinished: dict[str, set[str]],
    downstream: dict[str, list[str]],
    stage_mutex: dict[str, list[str]],
    mutex_counts: dict[str, int] | None = None,
) -> Scheduler:
    scheduler = Scheduler()
    scheduler._stage_states = stage_states
    scheduler._upstream_unfinished = upstream_unfinished
    scheduler._downstream = downstream
    scheduler._stage_mutex = stage_mutex
    scheduler._mutex_counts = collections.defaultdict(int)
    if mutex_counts:
        for name, count in mutex_counts.items():
            scheduler._mutex_counts[name] = count
    return scheduler


def _helper_startable_in_order(scheduler: Scheduler, running_count: int) -> list[str]:
    startable: list[str] = []
    for name in list(scheduler._stage_states.keys()):
        if scheduler.can_start(name, running_count=running_count):
            startable.append(name)
    return startable


def test_can_start_requires_ready_and_no_upstream() -> None:
    scheduler = _helper_make_scheduler(
        stage_states={"stage": StageExecutionState.PENDING},
        upstream_unfinished={"stage": set()},
        downstream={"stage": []},
        stage_mutex={"stage": []},
    )

    assert scheduler.can_start("stage", running_count=0) is False, (
        "PENDING stage should not be startable"
    )

    scheduler._stage_states["stage"] = StageExecutionState.READY
    assert scheduler.can_start("stage", running_count=0) is True, (
        "READY stage with no upstream should be startable"
    )

    scheduler._upstream_unfinished["stage"] = {"upstream"}
    assert scheduler.can_start("stage", running_count=0) is False, (
        "READY stage with unfinished upstream should not be startable"
    )


def test_can_start_respects_named_mutex() -> None:
    scheduler = _helper_make_scheduler(
        stage_states={"stage": StageExecutionState.READY},
        upstream_unfinished={"stage": set()},
        downstream={"stage": []},
        stage_mutex={"stage": ["mutex"]},
        mutex_counts={"mutex": 1},
    )

    assert scheduler.can_start("stage", running_count=0) is False, (
        "stage should not start when its mutex is held"
    )

    scheduler._mutex_counts["mutex"] = 0
    assert scheduler.can_start("stage", running_count=0) is True, (
        "stage should start when its mutex is released"
    )


def test_can_start_respects_exclusive_mutex_and_running() -> None:
    scheduler = _helper_make_scheduler(
        stage_states={"exclusive": StageExecutionState.READY, "normal": StageExecutionState.READY},
        upstream_unfinished={"exclusive": set(), "normal": set()},
        downstream={"exclusive": [], "normal": []},
        stage_mutex={"exclusive": [executor_core.EXCLUSIVE_MUTEX], "normal": []},
    )

    assert scheduler.can_start("exclusive", running_count=1) is False, (
        "exclusive stage should not start when other stages are running"
    )

    assert scheduler.can_start("exclusive", running_count=0) is True, (
        "exclusive stage should start when nothing else is running"
    )

    scheduler._mutex_counts[executor_core.EXCLUSIVE_MUTEX] = 1
    assert scheduler.can_start("normal", running_count=0) is False, (
        "normal stage should not start when exclusive mutex is held"
    )


def test_ready_selection_respects_stage_state_insertion_order() -> None:
    scheduler = _helper_make_scheduler(
        stage_states={
            "second": StageExecutionState.READY,
            "first": StageExecutionState.READY,
            "third": StageExecutionState.READY,
        },
        upstream_unfinished={"second": set(), "first": set(), "third": set()},
        downstream={"second": [], "first": [], "third": []},
        stage_mutex={"second": [], "first": [], "third": []},
    )

    assert _helper_startable_in_order(scheduler, running_count=0) == [
        "second",
        "first",
        "third",
    ], "startable stages should follow dict insertion order"


def test_chain_updates_downstream_after_completion() -> None:
    scheduler = _helper_make_scheduler(
        stage_states={
            "A": StageExecutionState.READY,
            "B": StageExecutionState.PENDING,
            "C": StageExecutionState.PENDING,
        },
        upstream_unfinished={"A": set(), "B": {"A"}, "C": {"B"}},
        downstream={"A": ["B", "C"], "B": ["C"], "C": []},
        stage_mutex={"A": [], "B": [], "C": []},
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
    scheduler = _helper_make_scheduler(
        stage_states={
            "A": StageExecutionState.READY,
            "B": StageExecutionState.PENDING,
            "C": StageExecutionState.PENDING,
        },
        upstream_unfinished={"A": set(), "B": {"A"}, "C": {"A"}},
        downstream={"A": ["B", "C"], "B": [], "C": []},
        stage_mutex={"A": [], "B": [], "C": []},
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
    scheduler = _helper_make_scheduler(
        stage_states={
            "A": StageExecutionState.READY,
            "B": StageExecutionState.READY,
            "C": StageExecutionState.PENDING,
        },
        upstream_unfinished={"A": set(), "B": set(), "C": {"A", "B"}},
        downstream={"A": ["C"], "B": ["C"], "C": []},
        stage_mutex={"A": [], "B": [], "C": []},
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
    scheduler = _helper_make_scheduler(
        stage_states={"A": StageExecutionState.READY, "B": StageExecutionState.READY},
        upstream_unfinished={"A": set(), "B": {"A"}},
        downstream={"A": ["B"], "B": []},
        stage_mutex={"A": ["mutex"], "B": ["mutex"]},
        mutex_counts={"mutex": 1},
    )

    scheduler.release_mutexes("A")
    newly_ready, newly_blocked = scheduler.on_stage_completed("A", failed=True)

    assert scheduler.get_state("B") == StageExecutionState.BLOCKED, (
        "B should be BLOCKED when upstream A failed"
    )
    assert newly_ready == [], "B should NOT become ready when upstream failed"
    assert len(newly_blocked) == 1, "one stage should be newly blocked"
    assert newly_blocked[0][0] == "B", "B should be in newly_blocked"


def test_apply_fail_fast_blocks_ready_and_pending() -> None:
    scheduler = _helper_make_scheduler(
        stage_states={
            "failed": StageExecutionState.COMPLETED,
            "pending": StageExecutionState.PENDING,
            "ready": StageExecutionState.READY,
            "blocked": StageExecutionState.BLOCKED,
        },
        upstream_unfinished={
            "failed": set(),
            "pending": set(),
            "ready": set(),
            "blocked": set(),
        },
        downstream={"failed": [], "pending": [], "ready": [], "blocked": []},
        stage_mutex={"failed": [], "pending": [], "ready": [], "blocked": []},
    )

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
    scheduler = _helper_make_scheduler(
        stage_states={
            "ready": StageExecutionState.READY,
            "pending": StageExecutionState.PENDING,
            "running": StageExecutionState.RUNNING,
        },
        upstream_unfinished={"ready": set(), "pending": set(), "running": set()},
        downstream={"ready": [], "pending": [], "running": []},
        stage_mutex={"ready": [], "pending": [], "running": []},
    )

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
    scheduler = _helper_make_scheduler(
        stage_states={"stage": StageExecutionState.COMPLETED},
        upstream_unfinished={"stage": set()},
        downstream={"stage": []},
        stage_mutex={"stage": []},
    )

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
