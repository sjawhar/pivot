"""Coordinator for distributed pipeline execution.

The Coordinator manages ExecutionState and distributes work to workers.
It implements greedy parallel scheduling with mutex support, similar to
the existing executor but designed for distributed workers.

Workers connect via Unix socket, request tasks, execute them, and report results.
The coordinator tracks dependencies and ensures proper execution order.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import pathlib  # noqa: TC003 - used at runtime in Path operations
import threading
import uuid
from typing import TYPE_CHECKING, final

from pivot import dag, parameters, registry
from pivot.executor import protocol
from pivot.executor import state as state_mod
from pivot.storage import lock
from pivot.storage import state as storage_state
from pivot.types import (
    OnError,
    StageResult,
    StageStatus,
    WorkerStageTask,
)

if TYPE_CHECKING:
    from pivot.types import DeferredWrites

logger = logging.getLogger(__name__)

# Special mutex that means "run exclusively" - no other stages run concurrently
EXCLUSIVE_MUTEX = "*"


@final
class Coordinator:
    """Manages ExecutionState and distributes work to workers.

    The coordinator implements greedy parallel scheduling:
    - Tracks stage dependencies and completion state
    - Handles mutex exclusivity (including EXCLUSIVE_MUTEX)
    - Distributes ready tasks to requesting workers
    - Processes results and updates state
    """

    def __init__(
        self,
        cache_dir: pathlib.Path,
        execution_order: list[str] | None = None,
        max_workers: int = 8,
        on_error: OnError = OnError.FAIL,
    ) -> None:
        self._cache_dir = cache_dir
        self._max_workers = max_workers
        self._on_error = on_error

        # Build DAG from registry
        self._graph = registry.REGISTRY.build_dag()
        if execution_order is None:
            execution_order = dag.get_execution_order(self._graph)
        self._execution_order = execution_order

        # Initialize state backend
        stages_dir = lock.get_stages_dir(cache_dir)
        stages_dir.mkdir(parents=True, exist_ok=True)
        backend = state_mod.LockFileBackend(stages_dir)

        # Initialize execution state
        self._state = state_mod.ExecutionState(
            backend=backend,
            execution_order=execution_order,
        )

        # Build stage metadata
        self._stage_info: dict[str, _StageInfo] = {}
        for name in execution_order:
            info = registry.REGISTRY.get(name)
            upstream = list(self._graph.predecessors(name))
            downstream = list(self._graph.successors(name))
            mutex = info.get("mutex", [])
            self._stage_info[name] = _StageInfo(
                name=name,
                upstream=upstream,
                upstream_unfinished=set(upstream),
                downstream=downstream,
                mutex=mutex,
            )

        # Mutex tracking
        self._mutex_counts: collections.defaultdict[str, int] = collections.defaultdict(int)

        # Task queue for workers
        self._task_queue: asyncio.Queue[WorkerStageTask] = asyncio.Queue()

        # Stages currently assigned to workers
        self._assigned_stages: dict[str, str] = {}  # stage_name -> worker_id

        # Run configuration
        self._run_id: str | None = None
        self._force = False
        self._no_commit = False
        self._no_cache = False
        self._overrides: parameters.ParamsOverrides = {}

        # StateDB for deferred writes
        self._state_db_path = cache_dir.parent / "state.db"

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

        # Server instance
        self._server: protocol.CoordinatorServer | None = None

        # Lock for thread-safe operations
        self._lock = threading.Lock()

    @property
    def state(self) -> state_mod.ExecutionState:
        """Return the execution state."""
        return self._state

    async def start(self, socket_path: pathlib.Path) -> None:
        """Start the coordinator server."""
        self._server = protocol.CoordinatorServer(
            state=self._state,
            task_queue=self._task_queue,
            result_callback=self._handle_result,
            disconnect_callback=self._handle_worker_disconnect,
            socket_path=socket_path,
        )
        await self._server.start()

    async def stop(self) -> None:
        """Stop the coordinator server."""
        self._shutdown_event.set()
        if self._server is not None:
            await self._server.stop()
            self._server = None

    def start_run(
        self,
        stages: list[str] | None = None,
        force: bool = False,
        no_commit: bool = False,
        no_cache: bool = False,
        overrides: parameters.ParamsOverrides | None = None,
    ) -> str:
        """Start a new execution run. Returns the run ID."""
        self._run_id = str(uuid.uuid4())[:12]
        self._force = force
        self._no_commit = no_commit
        self._no_cache = no_cache
        self._overrides = overrides or {}

        # Reset state for new run
        self._state.reset_for_run(self._run_id)

        # Reset stage tracking
        for info in self._stage_info.values():
            info.upstream_unfinished = set(info.upstream)
        self._mutex_counts.clear()
        self._assigned_stages.clear()

        # Filter stages if specified
        target_stages = set(stages) if stages else set(self._execution_order)

        # Queue initial ready stages
        self._queue_ready_stages(target_stages)

        logger.info(f"Started run {self._run_id} with {len(target_stages)} stages")
        return self._run_id

    def _queue_ready_stages(self, target_stages: set[str] | None = None) -> int:
        """Queue stages that are ready to execute. Returns count of queued stages."""
        if target_stages is None:
            target_stages = set(self._execution_order)

        queued = 0
        for name in self._execution_order:
            if name not in target_stages:
                continue

            # Skip if already queued/assigned or not ready
            if name in self._assigned_stages:
                continue

            status = self._state.get_stage_status(name)
            if status != StageStatus.READY:
                continue

            info = self._stage_info[name]
            if info.upstream_unfinished:
                continue

            # Check mutex availability
            if any(self._mutex_counts[m] > 0 for m in info.mutex):
                continue

            # Exclusive mutex handling
            is_exclusive = EXCLUSIVE_MUTEX in info.mutex
            running_count = len(self._assigned_stages)
            if is_exclusive and running_count > 0:
                continue
            if not is_exclusive and self._mutex_counts[EXCLUSIVE_MUTEX] > 0:
                continue

            # Acquire mutex locks
            for mutex in info.mutex:
                self._mutex_counts[mutex] += 1

            # Track as assigned (worker_id will be set when worker picks up task)
            # Using empty string to indicate "queued but not yet assigned to worker"
            self._assigned_stages[name] = ""

            # Create task
            task = WorkerStageTask(
                stage_name=name,
                run_id=self._run_id or "",
                force=self._force,
                no_commit=self._no_commit,
                no_cache=self._no_cache,
            )

            # Put task in queue (non-blocking since we're controlling flow)
            try:
                self._task_queue.put_nowait(task)
                queued += 1
                logger.debug(f"Queued stage: {name}")
            except asyncio.QueueFull:
                # Release mutex locks and remove from assigned if we couldn't queue
                for mutex in info.mutex:
                    self._mutex_counts[mutex] -= 1
                del self._assigned_stages[name]
                break

        return queued

    def _handle_result(self, worker_id: str, stage_name: str, result: StageResult) -> None:
        """Handle a result reported by a worker."""
        with self._lock:
            info = self._stage_info.get(stage_name)
            if info is None:
                logger.error(f"Unknown stage in result: {stage_name}")
                return

            try:
                # Remove from assigned
                self._assigned_stages.pop(stage_name, None)

                # Update state
                status = result["status"]
                if status == StageStatus.FAILED:
                    self._state.mark_failed(stage_name, worker_id, result["reason"])
                    self._handle_stage_failure(stage_name)
                else:
                    self._state.mark_completed(stage_name, worker_id, result)

                    # Apply deferred writes for successful runs
                    if status == StageStatus.RAN and not self._no_commit:
                        self._apply_deferred_writes(stage_name, result)

                # Update downstream dependencies
                for downstream_name in info.downstream:
                    downstream_info = self._stage_info.get(downstream_name)
                    if downstream_info:
                        downstream_info.upstream_unfinished.discard(stage_name)
            finally:
                # Release mutex locks - always runs even if exception occurs
                for mutex in info.mutex:
                    self._mutex_counts[mutex] -= 1
                    if self._mutex_counts[mutex] < 0:
                        logger.error(f"Mutex '{mutex}' released when not held")
                        self._mutex_counts[mutex] = 0

            # Queue more ready stages
            self._queue_ready_stages()

    def _handle_worker_disconnect(self, worker_id: str) -> None:
        """Handle a worker disconnecting - fail any stages it was running."""
        with self._lock:
            # Find stages assigned to this worker
            orphaned_stages = [
                stage_name
                for stage_name, assigned_worker in self._assigned_stages.items()
                if assigned_worker == worker_id
            ]

            for stage_name in orphaned_stages:
                info = self._stage_info.get(stage_name)
                if info is None:
                    continue

                # Mark stage as failed
                self._state.mark_failed(stage_name, worker_id, f"Worker {worker_id} disconnected")
                self._handle_stage_failure(stage_name)

                # Remove from assigned
                self._assigned_stages.pop(stage_name, None)

                # Release mutex locks
                for mutex in info.mutex:
                    self._mutex_counts[mutex] -= 1
                    if self._mutex_counts[mutex] < 0:
                        self._mutex_counts[mutex] = 0

            if orphaned_stages:
                logger.warning(
                    f"Worker {worker_id} disconnected with {len(orphaned_stages)} "
                    + f"running stages: {orphaned_stages}"
                )

    def _handle_stage_failure(self, failed_stage: str) -> None:
        """Handle stage failure - cascade to downstream if ON_ERROR.FAIL."""
        if self._on_error != OnError.FAIL:
            return

        # Mark all downstream stages as skipped
        visited: set[str] = set()
        to_visit = [failed_stage]

        while to_visit:
            stage_name = to_visit.pop()
            if stage_name in visited:
                continue
            visited.add(stage_name)

            info = self._stage_info.get(stage_name)
            if info is None:
                continue

            for downstream_name in info.downstream:
                if downstream_name in visited:
                    continue
                downstream_status = self._state.get_stage_status(downstream_name)
                if downstream_status == StageStatus.READY:
                    self._state.mark_skipped_upstream(downstream_name, failed_stage)
                to_visit.append(downstream_name)

    def _apply_deferred_writes(self, stage_name: str, result: StageResult) -> None:
        """Apply deferred writes from a successful stage execution."""
        if "deferred_writes" not in result:
            return

        deferred: DeferredWrites = result["deferred_writes"]
        if not deferred:
            return

        try:
            reg_info = registry.REGISTRY.get(stage_name)
            output_paths = [str(out.path) for out in reg_info["outs"]]

            with storage_state.StateDB(self._state_db_path) as state_db:
                state_db.apply_deferred_writes(stage_name, output_paths, deferred)
        except Exception:
            logger.exception(f"Failed to apply deferred writes for {stage_name}")

    def get_run_summary(self) -> dict[str, str]:
        """Get a summary of the current run status."""
        snapshot = self._state.get_snapshot()
        return {
            "run_id": snapshot["run_id"] or "",
            "ran": str(snapshot["ran"]),
            "skipped": str(snapshot["skipped"]),
            "failed": str(snapshot["failed"]),
            "workers": str(len(snapshot["workers"])),
        }

    def is_complete(self) -> bool:
        """Check if all stages are complete (not READY or IN_PROGRESS)."""
        snapshot = self._state.get_snapshot()
        for stage_info in snapshot["stages"].values():
            if stage_info["status"] in (StageStatus.READY, StageStatus.IN_PROGRESS):
                return False
        return True


@final
class _StageInfo:
    """Internal tracking info for a stage."""

    def __init__(
        self,
        name: str,
        upstream: list[str],
        upstream_unfinished: set[str],
        downstream: list[str],
        mutex: list[str],
    ) -> None:
        self.name = name
        self.upstream = upstream
        self.upstream_unfinished = upstream_unfinished
        self.downstream = downstream
        self.mutex = mutex
