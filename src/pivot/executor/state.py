"""Centralized execution state with versioning and event emission.

This module provides ExecutionState, a centralized component for tracking pipeline
execution state across distributed workers. Key features:

- Version-tracked state changes (monotonic counter)
- Event history for client catch-up (ring buffer)
- Pluggable persistence backends (lock files, LMDB)
- Thread-safe operations

The coordinator holds ExecutionState; workers are stateless and communicate
via protocol messages.
"""

from __future__ import annotations

import collections
import dataclasses
import threading
import time
from typing import TYPE_CHECKING, Protocol, final

from pivot.types import (
    DagChangedEvent,
    ExecutionSnapshot,
    StageCompletedEvent,
    StageFailedEvent,
    StageSkippedUpstreamEvent,
    StageStartedEvent,
    StageStateSnapshot,
    StageStatus,
    StateEvent,
    StateEventType,
    WorkerDisconnectedEvent,
    WorkerHeartbeatEvent,
    WorkerRegisteredEvent,
    WorkerSnapshot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.types import LockData, StageResult

# Maximum events to keep in history for client catch-up
_MAX_HISTORY_SIZE = 1000


class StateBackend(Protocol):
    """Pluggable persistence backend for execution state."""

    def load_lock(self, stage_name: str) -> LockData | None:
        """Load lock data for a stage. Returns None if not found."""
        ...

    def save_lock(self, stage_name: str, data: LockData) -> None:
        """Save lock data for a stage."""
        ...


@final
class LockFileBackend:
    """Backend using current lock file system."""

    def __init__(self, stages_dir: Path) -> None:
        from pivot.storage import lock

        self._stages_dir = stages_dir
        self._lock_module = lock

    def load_lock(self, stage_name: str) -> LockData | None:
        """Load lock data from .lock file."""
        stage_lock = self._lock_module.StageLock(stage_name, self._stages_dir)
        return stage_lock.read()

    def save_lock(self, stage_name: str, data: LockData) -> None:
        """Save lock data to .lock file atomically."""
        stage_lock = self._lock_module.StageLock(stage_name, self._stages_dir)
        stage_lock.write(data)


@dataclasses.dataclass
class _StageState:
    """Internal mutable state for a stage."""

    name: str
    status: StageStatus = StageStatus.READY
    reason: str = ""
    worker_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None


@dataclasses.dataclass
class _WorkerState:
    """Internal mutable state for a worker."""

    worker_id: str
    connected_at: float
    last_heartbeat: float
    current_stage: str | None = None


@final
class ExecutionState:
    """Centralized execution state with versioning and event emission.

    Thread-safe: all public methods acquire a lock before modifying state.
    Version is incremented on every state change, and events are recorded
    in a ring buffer for client catch-up via get_events_since().
    """

    def __init__(
        self,
        backend: StateBackend,
        execution_order: list[str],
        run_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._execution_order = list(execution_order)
        self._run_id = run_id

        self._lock = threading.Lock()
        self._version = 0
        self._events: collections.deque[StateEvent] = collections.deque(maxlen=_MAX_HISTORY_SIZE)

        # Initialize stage states
        self._stages: dict[str, _StageState] = {}
        for name in execution_order:
            self._stages[name] = _StageState(name=name)

        # Worker tracking
        self._workers: dict[str, _WorkerState] = {}

        # Counters
        self._ran = 0
        self._skipped = 0
        self._failed = 0

    @property
    def version(self) -> int:
        """Current state version (monotonic)."""
        with self._lock:
            return self._version

    @property
    def run_id(self) -> str | None:
        """Current run ID."""
        with self._lock:
            return self._run_id

    def set_run_id(self, run_id: str) -> None:
        """Set the current run ID."""
        with self._lock:
            self._run_id = run_id

    # -------------------------------------------------------------------------
    # Stage lifecycle
    # -------------------------------------------------------------------------

    def mark_started(self, stage: str, worker_id: str) -> StateEvent:
        """Mark a stage as started by a worker. Returns the emitted event."""
        with self._lock:
            state = self._stages[stage]
            state.status = StageStatus.IN_PROGRESS
            state.worker_id = worker_id
            state.start_time = time.perf_counter()

            event = self._emit_event(
                StageStartedEvent(
                    type=StateEventType.STAGE_STARTED,
                    version=self._version,
                    timestamp=time.time(),
                    stage=stage,
                    worker_id=worker_id,
                )
            )

            # Update worker's current stage
            if worker_id in self._workers:
                self._workers[worker_id].current_stage = stage

            return event

    def mark_completed(self, stage: str, worker_id: str, result: StageResult) -> StateEvent:
        """Mark a stage as completed with result. Returns the emitted event.

        Note: For failed stages, use mark_failed() instead. This method is for
        RAN and SKIPPED statuses only.
        """
        with self._lock:
            state = self._stages[stage]
            state.status = result["status"]
            state.reason = result["reason"]
            state.end_time = time.perf_counter()

            duration_ms = 0.0
            if state.start_time is not None:
                duration_ms = (state.end_time - state.start_time) * 1000

            # Update counters and narrow type for StageCompletedEvent
            status = result["status"]
            if status == StageStatus.RAN:
                self._ran += 1
                completed_status = StageStatus.RAN
            elif status == StageStatus.SKIPPED:
                self._skipped += 1
                completed_status = StageStatus.SKIPPED
            else:
                # FAILED should use mark_failed() instead
                raise ValueError(f"Unexpected status in mark_completed: {status}")

            event = self._emit_event(
                StageCompletedEvent(
                    type=StateEventType.STAGE_COMPLETED,
                    version=self._version,
                    timestamp=time.time(),
                    stage=stage,
                    worker_id=worker_id,
                    status=completed_status,
                    reason=result["reason"],
                    duration_ms=duration_ms,
                )
            )

            # Clear worker's current stage
            if worker_id in self._workers:
                self._workers[worker_id].current_stage = None

            return event

    def mark_failed(self, stage: str, worker_id: str, error: str) -> StateEvent:
        """Mark a stage as failed. Returns the emitted event."""
        with self._lock:
            state = self._stages[stage]
            state.status = StageStatus.FAILED
            state.reason = error
            state.end_time = time.perf_counter()
            self._failed += 1

            event = self._emit_event(
                StageFailedEvent(
                    type=StateEventType.STAGE_FAILED,
                    version=self._version,
                    timestamp=time.time(),
                    stage=stage,
                    worker_id=worker_id,
                    error=error,
                )
            )

            # Clear worker's current stage
            if worker_id in self._workers:
                self._workers[worker_id].current_stage = None

            return event

    def mark_skipped_upstream(self, stage: str, failed_upstream: str) -> StateEvent:
        """Mark a stage as skipped due to upstream failure. Returns the emitted event."""
        with self._lock:
            state = self._stages[stage]
            state.status = StageStatus.SKIPPED
            state.reason = f"upstream '{failed_upstream}' failed"
            self._skipped += 1

            return self._emit_event(
                StageSkippedUpstreamEvent(
                    type=StateEventType.STAGE_SKIPPED,
                    version=self._version,
                    timestamp=time.time(),
                    stage=stage,
                    failed_upstream=failed_upstream,
                )
            )

    def get_stage_status(self, stage: str) -> StageStatus:
        """Get the current status of a stage."""
        with self._lock:
            return self._stages[stage].status

    def get_ready_stages(self) -> list[str]:
        """Get list of stages with READY status."""
        with self._lock:
            return [
                name for name, state in self._stages.items() if state.status == StageStatus.READY
            ]

    # -------------------------------------------------------------------------
    # Worker management
    # -------------------------------------------------------------------------

    def register_worker(self, worker_id: str) -> StateEvent:
        """Register a new worker. Returns the emitted event."""
        with self._lock:
            now = time.time()
            self._workers[worker_id] = _WorkerState(
                worker_id=worker_id,
                connected_at=now,
                last_heartbeat=now,
            )

            return self._emit_event(
                WorkerRegisteredEvent(
                    type=StateEventType.WORKER_REGISTERED,
                    version=self._version,
                    timestamp=now,
                    worker_id=worker_id,
                )
            )

    def worker_heartbeat(self, worker_id: str) -> StateEvent:
        """Update worker heartbeat. Returns the emitted event."""
        with self._lock:
            now = time.time()
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = now

            return self._emit_event(
                WorkerHeartbeatEvent(
                    type=StateEventType.WORKER_HEARTBEAT,
                    version=self._version,
                    timestamp=now,
                    worker_id=worker_id,
                )
            )

    def unregister_worker(self, worker_id: str) -> StateEvent:
        """Unregister a worker. Returns the emitted event."""
        with self._lock:
            now = time.time()
            if worker_id in self._workers:
                del self._workers[worker_id]

            return self._emit_event(
                WorkerDisconnectedEvent(
                    type=StateEventType.WORKER_DISCONNECTED,
                    version=self._version,
                    timestamp=now,
                    worker_id=worker_id,
                )
            )

    def get_worker_count(self) -> int:
        """Get number of connected workers."""
        with self._lock:
            return len(self._workers)

    # -------------------------------------------------------------------------
    # DAG management
    # -------------------------------------------------------------------------

    def update_dag(self, execution_order: list[str]) -> StateEvent:
        """Update the DAG after a registry reload. Returns the emitted event."""
        with self._lock:
            self._execution_order = list(execution_order)

            # Add new stages, keep existing state for stages that still exist
            new_stages: dict[str, _StageState] = {}
            for name in execution_order:
                if name in self._stages:
                    new_stages[name] = self._stages[name]
                else:
                    new_stages[name] = _StageState(name=name)
            self._stages = new_stages

            return self._emit_event(
                DagChangedEvent(
                    type=StateEventType.DAG_CHANGED,
                    version=self._version,
                    timestamp=time.time(),
                    stages=execution_order,
                )
            )

    # -------------------------------------------------------------------------
    # Client sync
    # -------------------------------------------------------------------------

    def get_snapshot(self) -> ExecutionSnapshot:
        """Get a complete snapshot of current execution state."""
        with self._lock:
            stages: dict[str, StageStateSnapshot] = {}
            for name, state in self._stages.items():
                stages[name] = StageStateSnapshot(
                    name=state.name,
                    status=state.status,
                    reason=state.reason,
                    worker_id=state.worker_id,
                    start_time=state.start_time,
                    end_time=state.end_time,
                )

            workers: dict[str, WorkerSnapshot] = {}
            for worker_id, worker in self._workers.items():
                workers[worker_id] = WorkerSnapshot(
                    worker_id=worker.worker_id,
                    connected_at=worker.connected_at,
                    last_heartbeat=worker.last_heartbeat,
                    current_stage=worker.current_stage,
                )

            return ExecutionSnapshot(
                version=self._version,
                run_id=self._run_id,
                stages=stages,
                workers=workers,
                execution_order=list(self._execution_order),
                ran=self._ran,
                skipped=self._skipped,
                failed=self._failed,
            )

    def get_events_since(self, version: int) -> list[StateEvent]:
        """Get all events since a given version for client catch-up."""
        with self._lock:
            return [event for event in self._events if event["version"] > version]

    # -------------------------------------------------------------------------
    # State reset
    # -------------------------------------------------------------------------

    def reset_for_run(self, run_id: str) -> None:
        """Reset state for a new execution run."""
        with self._lock:
            self._run_id = run_id
            self._ran = 0
            self._skipped = 0
            self._failed = 0

            # Reset all stages to READY
            for state in self._stages.values():
                state.status = StageStatus.READY
                state.reason = ""
                state.worker_id = None
                state.start_time = None
                state.end_time = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _emit_event(self, event: StateEvent) -> StateEvent:
        """Increment version and record event. Must be called with lock held."""
        self._version += 1
        # Update event version to match (events are constructed with old version)
        event["version"] = self._version  # type: ignore[literal-required]
        self._events.append(event)
        return event
