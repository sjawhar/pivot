from __future__ import annotations

import atexit
import collections
import concurrent.futures
import contextlib
import dataclasses
import datetime
import functools
import logging
import multiprocessing as mp
import pathlib
import queue
import threading
import time
from typing import TYPE_CHECKING, Literal, TypedDict, final

import loky

from pivot import (
    config,
    dag,
    exceptions,
    explain,
    metrics,
    outputs,
    parameters,
    project,
    registry,
    run_history,
)
from pivot.executor import worker
from pivot.storage import cache, lock, project_lock, track
from pivot.storage import state as state_mod
from pivot.tui import console
from pivot.types import (
    OnError,
    OutputMessage,
    RunEventType,
    RunJsonEvent,
    StageCompleteEvent,
    StageDisplayStatus,
    StageExplanation,
    StageResult,
    StageStartEvent,
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiQueue,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from networkx import DiGraph

logger = logging.getLogger(__name__)

_MAX_WORKERS_DEFAULT = 8

# Special mutex that means "run exclusively" - no other stages run concurrently
EXCLUSIVE_MUTEX = "*"


def _noop() -> None:
    """Module-level for pickling."""
    pass


def _compute_max_workers(stage_count: int, override: int | None = None) -> int:
    """Single source of truth for max_workers calculation."""
    if override is not None:
        return max(1, min(override, stage_count))
    return max(1, min(loky.cpu_count() or 1, _MAX_WORKERS_DEFAULT, stage_count))


def _warm_workers(pool: loky.ProcessPoolExecutor, count: int) -> None:
    """Submit no-op tasks to ensure workers are warm and channels established."""
    futures = [pool.submit(_noop) for _ in range(count)]
    for f in futures:
        f.result()


def prepare_workers(
    stage_count: int, *, parallel: bool = True, max_workers: int | None = None
) -> int:
    """Pre-warm loky worker pool. Returns actual worker count.

    Call before starting Textual TUI to avoid FD inheritance issues.
    Safe to call in non-TUI mode (no downside, slightly faster first execution).
    """
    if not parallel or stage_count <= 0:
        return 1
    workers = _compute_max_workers(stage_count, max_workers)
    pool = loky.get_reusable_executor(max_workers=workers)
    _warm_workers(pool, workers)
    return workers


def restart_workers(stage_count: int, max_workers: int | None = None) -> int:
    """Kill existing workers and spawn fresh ones. For code reload in watch mode.

    Unlike prepare_workers(), this kills existing workers first, then warms the new pool.
    """
    if stage_count <= 0:
        return 1
    workers = _compute_max_workers(stage_count, max_workers)
    pool = loky.get_reusable_executor(max_workers=workers, kill_workers=True)
    _warm_workers(pool, workers)
    return workers


def _cleanup_worker_pool() -> None:
    """Kill loky worker pool on process exit to prevent orphaned workers."""
    with contextlib.suppress(Exception):
        loky.get_reusable_executor(max_workers=1, kill_workers=True)


@functools.cache  # Ensures single atexit registration across threads
def _ensure_cleanup_registered() -> None:
    atexit.register(_cleanup_worker_pool)


class ExecutionSummary(TypedDict):
    """Summary result for a single stage after execution (returned by executor.run)."""

    status: Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED, StageStatus.UNKNOWN]
    reason: str


def count_results(results: dict[str, ExecutionSummary]) -> tuple[int, int, int, int]:
    """Count results by category: ran, cached, blocked, failed.

    Distinguishes between "cached" (skipped because unchanged) and
    "blocked" (skipped because upstream failed).

    Returns:
        Tuple of (ran, cached, blocked, failed) counts
    """
    ran = cached = blocked = failed = 0
    for result in results.values():
        status = result["status"]
        reason = result["reason"]
        if status == StageStatus.RAN:
            ran += 1
        elif status == StageStatus.FAILED:
            failed += 1
        elif status == StageStatus.UNKNOWN:
            # UNKNOWN indicates executor bug - don't count in normal categories
            pass
        elif reason.startswith("upstream"):
            blocked += 1
        else:
            cached += 1
    return ran, cached, blocked, failed


@dataclasses.dataclass
class StageState:
    """Tracks execution state for a single stage."""

    name: str
    index: int  # 1-based position in execution order
    info: registry.RegistryStageInfo
    upstream: list[str]
    upstream_unfinished: set[str]
    downstream: list[str]
    mutex: list[str]
    status: StageStatus = StageStatus.READY
    result: StageResult | None = None
    start_time: float | None = None
    end_time: float | None = None

    def get_duration(self) -> float | None:
        """Calculate elapsed duration from start to end (or current time if still running)."""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


@final
class StageLifecycle:
    """Centralized handler for stage lifecycle events - guarantees notifications.

    All stage state transitions should go through this class to ensure
    TUI, console, and progress callback notifications are always sent.
    """

    def __init__(
        self,
        tui_queue: TuiQueue | None,
        con: console.Console | None,
        progress_callback: Callable[[RunJsonEvent], None] | None,
        total_stages: int,
        run_id: str,
        explain_mode: bool = False,
    ) -> None:
        self.tui_queue = tui_queue
        self.console = con
        self.progress_callback = progress_callback
        self.total_stages = total_stages
        self.run_id = run_id
        self.explain_mode = explain_mode

    def mark_started(self, state: StageState) -> None:
        """Mark stage as in-progress and send all notifications."""
        state.status = StageStatus.IN_PROGRESS
        state.start_time = time.perf_counter()

        if self.console and not self.explain_mode:
            # Use FINGERPRINTING since stage is being checked - not yet confirmed to run
            self.console.stage_start(
                name=state.name,
                index=state.index,
                total=self.total_stages,
                status=StageDisplayStatus.FINGERPRINTING,
            )

        if self.tui_queue:
            self.tui_queue.put(
                TuiStatusMessage(
                    type=TuiMessageType.STATUS,
                    stage=state.name,
                    index=state.index,
                    total=self.total_stages,
                    status=StageStatus.IN_PROGRESS,
                    reason="",
                    elapsed=None,
                    run_id=self.run_id,
                )
            )

        if self.progress_callback:
            self.progress_callback(
                StageStartEvent(
                    type=RunEventType.STAGE_START,
                    stage=state.name,
                    index=state.index,
                    total=self.total_stages,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )

    def mark_completed(self, state: StageState, result: StageResult) -> None:
        """Mark stage completed with result and send all notifications."""
        state.result = result
        state.status = result["status"]
        state.end_time = time.perf_counter()
        self._notify_complete(state, result)

    def mark_failed(self, state: StageState, reason: str) -> None:
        """Mark stage as failed and send all notifications."""
        result = StageResult(status=StageStatus.FAILED, reason=reason, output_lines=[])
        state.result = result
        state.status = StageStatus.FAILED
        state.end_time = time.perf_counter()
        self._notify_complete(state, result)

    def mark_skipped_upstream(self, state: StageState, failed_stage: str) -> None:
        """Mark stage as skipped due to upstream failure and send all notifications.

        Unlike mark_completed, this doesn't set end_time since the stage never started.
        """
        reason = f"upstream '{failed_stage}' failed"
        result = StageResult(status=StageStatus.SKIPPED, reason=reason, output_lines=[])
        state.result = result
        state.status = StageStatus.SKIPPED
        # Don't set end_time - stage never started
        self._notify_complete(state, result)

    def _notify_complete(self, state: StageState, result: StageResult) -> None:
        """Send completion notifications to all channels."""
        result_status = result["status"]
        result_reason = result["reason"]
        duration = state.get_duration()
        logger.debug(f"TUI status: {state.name} -> {result_status}")  # noqa: G004

        if self.console:
            self.console.stage_result(
                name=state.name,
                index=state.index,
                total=self.total_stages,
                status=result_status,
                reason=result_reason,
                duration=duration,
            )

        if self.tui_queue:
            self.tui_queue.put(
                TuiStatusMessage(
                    type=TuiMessageType.STATUS,
                    stage=state.name,
                    index=state.index,
                    total=self.total_stages,
                    status=result_status,
                    reason=result_reason,
                    elapsed=duration,
                    run_id=self.run_id,
                )
            )

        if self.progress_callback:
            self.progress_callback(
                StageCompleteEvent(
                    type=RunEventType.STAGE_COMPLETE,
                    stage=state.name,
                    status=result_status,
                    reason=result_reason,
                    duration_ms=(duration or 0) * 1000,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )


def run(
    stages: list[str] | None = None,
    single_stage: bool = False,
    cache_dir: pathlib.Path | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    on_error: OnError | str = OnError.FAIL,
    show_output: bool = True,
    allow_uncached_incremental: bool = False,
    force: bool = False,
    stage_timeout: float | None = None,
    explain_mode: bool = False,
    tui_queue: TuiQueue | None = None,
    output_queue: mp.Queue[OutputMessage] | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    progress_callback: Callable[[RunJsonEvent], None] | None = None,
    cancel_event: threading.Event | None = None,
    checkout_missing: bool = False,
) -> dict[str, ExecutionSummary]:
    """Execute pipeline stages with greedy parallel execution.

    Args:
        stages: Target stages to run (and their dependencies). If None, runs all.
        single_stage: If True, run only the specified stages without dependencies.
        cache_dir: Directory for lock files. Defaults to .pivot/cache.
        parallel: If True, run independent stages in parallel (default: True).
        max_workers: Max concurrent stages (default: min(cpu_count, 8)).
        on_error: Error handling mode - "fail" or "keep_going".
        show_output: If True, print progress and stage output to console.
        allow_uncached_incremental: If True, skip safety check for uncached IncrementalOut files.
        force: If True, bypass cache and force all stages to re-execute.
        stage_timeout: Max seconds for each stage to complete (default: no timeout).
        explain_mode: If True, show detailed WHY for each stage before execution.
        tui_queue: Queue for TUI messages (status updates and logs).
        output_queue: Queue for worker output streaming. If None, created internally.
            Pass this when running in TUI mode to avoid multiprocessing issues.
        no_commit: If True, defer lock files to pending dir (faster iteration).
        no_cache: If True, skip caching outputs entirely (maximum iteration speed).
        progress_callback: Callback for JSONL progress events (stage start/complete).
        cancel_event: If set, stop starting new stages and mark pending as cancelled.
        checkout_missing: If True, restore missing tracked files from cache before running.

    Returns:
        Dict of stage_name -> {status: "ran"|"skipped"|"failed", reason: str}
    """
    if cache_dir is None:
        cache_dir = project.get_project_root() / ".pivot" / "cache"

    if isinstance(on_error, OnError):
        error_mode = on_error
    else:
        try:
            error_mode = OnError(on_error)
        except ValueError:
            raise ValueError(
                f"Invalid on_error mode: {on_error}. Use 'fail' or 'keep_going'"
            ) from None

    # Verify tracked files before building DAG (provides better error messages)
    project_root = project.get_project_root()
    _verify_tracked_files(project_root, checkout_missing=checkout_missing)

    graph = registry.REGISTRY.build_dag(validate=True)

    if stages:
        registered = set(graph.nodes())
        unknown = [s for s in stages if s not in registered]
        if unknown:
            raise exceptions.StageNotFoundError(unknown, available_stages=list(registered))

    execution_order = dag.get_execution_order(graph, stages, single_stage=single_stage)

    if not execution_order:
        return {}

    # Record start time and generate run_id for run history
    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    run_id = run_history.generate_run_id()
    targeted_stages = stages if stages else list(graph.nodes())

    # Load parameter overrides early to validate and prepare for workers
    overrides = parameters.load_params_yaml()

    # Load checkout mode configuration
    checkout_modes = config.get_checkout_mode_order()

    # Check for uncached IncrementalOut files that would be lost
    if not allow_uncached_incremental:
        uncached = _check_uncached_incremental_outputs(execution_order, cache_dir)
        if uncached:
            files_list = "\n".join(f"  - {stage}: {path}" for stage, path in uncached)
            raise exceptions.UncachedIncrementalOutputError(
                f"The following IncrementalOut files exist but are not in cache:\n{files_list}\n\n"
                + "Running the pipeline will DELETE these files and they cannot be restored.\n"
                + "To proceed anyway, use allow_uncached_incremental=True or back up these files first."
            )

    con = console.get_console() if show_output else None
    total_stages = len(execution_order)

    stage_states = _initialize_stage_states(execution_order, graph)

    max_workers = 1 if not parallel else _compute_max_workers(len(execution_order), max_workers)

    start_time = time.perf_counter()

    # When no_commit=True, acquire lock to prevent commits during execution
    lock_context = project_lock.pending_state_lock() if no_commit else contextlib.nullcontext()
    with lock_context:
        _execute_greedy(
            stage_states=stage_states,
            cache_dir=cache_dir,
            max_workers=max_workers,
            error_mode=error_mode,
            con=con,
            total_stages=total_stages,
            stage_timeout=stage_timeout,
            overrides=overrides,
            explain_mode=explain_mode,
            checkout_modes=checkout_modes,
            tui_queue=tui_queue,
            output_queue=output_queue,
            run_id=run_id,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        results = _build_results(stage_states)

        # Write run history
        ended_at = datetime.datetime.now(datetime.UTC).isoformat()
        retention = config.get_run_history_retention()
        _write_run_history(
            run_id=run_id,
            stage_states=stage_states,
            cache_dir=cache_dir,
            targeted_stages=targeted_stages,
            execution_order=execution_order,
            started_at=started_at,
            ended_at=ended_at,
            retention=retention,
        )

    if con:
        ran, cached, blocked, failed = count_results(results)
        total_duration = time.perf_counter() - start_time
        con.summary(ran, cached, blocked, failed, total_duration)

    return results


def _initialize_stage_states(
    execution_order: list[str],
    graph: DiGraph[str],
) -> dict[str, StageState]:
    """Initialize state tracking for all stages."""
    stages_set = set(execution_order)
    states = dict[str, StageState]()

    for idx, stage_name in enumerate(execution_order, 1):
        stage_info = registry.REGISTRY.get(stage_name)

        upstream = list(graph.successors(stage_name))
        upstream_in_plan = [u for u in upstream if u in stages_set]

        downstream = list(graph.predecessors(stage_name))
        downstream_in_plan = [d for d in downstream if d in stages_set]

        states[stage_name] = StageState(
            name=stage_name,
            index=idx,
            info=stage_info,
            upstream=upstream_in_plan,
            upstream_unfinished=set(upstream_in_plan),
            downstream=downstream_in_plan,
            mutex=stage_info["mutex"],
        )

    return states


def _restore_tracked_file(
    path: pathlib.Path,
    track_data: track.PvtData,
    cache_dir: pathlib.Path,
) -> bool:
    """Restore a tracked file from cache.

    Returns True if successfully restored, False if not in cache.
    """
    output_hash = track.pvt_to_hash_info(track_data)

    # Use default checkout modes (hardlink with copy fallback)
    checkout_modes = config.get_checkout_mode_order()
    return cache.restore_from_cache(path, output_hash, cache_dir, checkout_modes=checkout_modes)


def _verify_tracked_files(project_root: pathlib.Path, checkout_missing: bool = False) -> None:
    """Verify all .pvt tracked files exist and warn on hash mismatches.

    Args:
        project_root: Project root directory.
        checkout_missing: If True, restore missing tracked files from cache before validating.
            Only restores files that don't exist - never overwrites existing files.
    """
    tracked_files = track.discover_pvt_files(project_root)
    if not tracked_files:
        return

    with metrics.timed("core.verify_tracked_files"):
        missing = list[str]()
        state_db_path = project_root / ".pivot" / "state.db"
        cache_dir = project_root / ".pivot" / "cache" / "files"

        with state_mod.StateDB(state_db_path) as state_db:
            for data_path, track_data in tracked_files.items():
                path = pathlib.Path(data_path)

                # Try to hash the file - handles race conditions where file disappears
                try:
                    if path.is_file():
                        current_hash = cache.hash_file(path, state_db)
                    elif path.is_dir():
                        current_hash, _ = cache.hash_directory(path, state_db)
                    else:
                        # Path doesn't exist
                        raise FileNotFoundError(data_path)
                except FileNotFoundError:
                    if checkout_missing:
                        if _restore_tracked_file(path, track_data, cache_dir):
                            logger.info(f"Restored tracked file: {data_path}")
                        else:
                            logger.debug(f"Failed to restore tracked file from cache: {data_path}")
                            missing.append(data_path)
                    else:
                        missing.append(data_path)
                    continue

                # Check hash mismatch (file exists but content changed)
                if current_hash != track_data["hash"]:
                    logger.warning(
                        f"Tracked file '{data_path}' has changed since tracking. "
                        + f"Run 'pivot track --force {track_data['path']}' to update."
                    )

        if missing:
            raise exceptions.TrackedFileMissingError(missing, checkout_attempted=checkout_missing)


def _warn_single_stage_mutex_groups(stage_states: dict[str, StageState]) -> None:
    """Warn if any mutex group contains only one stage (likely a typo)."""
    groups: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
    for name, state in stage_states.items():
        for mutex in state.mutex:
            groups[mutex].append(name)

    for group, members in groups.items():
        # Skip EXCLUSIVE_MUTEX - it's intentionally used for single stages
        if group == EXCLUSIVE_MUTEX:
            continue
        if len(members) == 1:
            logger.warning(f"Mutex group '{group}' only contains stage '{members[0]}'")


def _create_executor(max_workers: int) -> concurrent.futures.Executor:
    """Get reusable loky executor - workers persist across calls for efficiency."""
    _ensure_cleanup_registered()
    return loky.get_reusable_executor(max_workers=max_workers)


def _execute_greedy(
    stage_states: dict[str, StageState],
    cache_dir: pathlib.Path,
    max_workers: int,
    error_mode: OnError,
    con: console.Console | None,
    total_stages: int,
    stage_timeout: float | None = None,
    overrides: parameters.ParamsOverrides | None = None,
    explain_mode: bool = False,
    checkout_modes: list[cache.CheckoutMode] | None = None,
    tui_queue: TuiQueue | None = None,
    output_queue: mp.Queue[OutputMessage] | None = None,
    run_id: str = "",
    force: bool = False,
    no_commit: bool = False,
    no_cache: bool = False,
    progress_callback: Callable[[RunJsonEvent], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Execute stages with greedy parallel scheduling using loky ProcessPoolExecutor."""
    overrides = overrides or {}
    checkout_modes = checkout_modes or config.get_checkout_mode_order()
    futures: dict[concurrent.futures.Future[StageResult], str] = {}
    mutex_counts: collections.defaultdict[str, int] = collections.defaultdict(int)

    # Centralized lifecycle handler for state transitions + notifications
    lifecycle = StageLifecycle(
        tui_queue=tui_queue,
        con=con,
        progress_callback=progress_callback,
        total_stages=total_stages,
        run_id=run_id,
        explain_mode=explain_mode,
    )

    _warn_single_stage_mutex_groups(stage_states)

    executor = _create_executor(max_workers)
    # Create output queue if not provided (for TUI mode, pass pre-created queue to avoid mp issues)
    # Track manager so we can shut it down - only created when no queue is passed in
    local_manager = None
    if output_queue is None:
        # Use spawn context to avoid fork-in-multithreaded-context issues (Python 3.13+ deprecation)
        spawn_ctx = mp.get_context("spawn")
        local_manager = spawn_ctx.Manager()
        # Manager().Queue() returns AutoProxy[Queue] which is incompatible with Queue type stubs
        output_queue = local_manager.Queue()  # pyright: ignore[reportAssignmentType]

    # Type narrowing: output_queue is guaranteed to be non-None after the block above
    assert output_queue is not None

    state_db_path = cache_dir.parent / "state.db"
    output_thread: threading.Thread | None = None

    try:
        # Start output thread inside try block to ensure Manager cleanup on failure
        if con or tui_queue:
            output_thread = threading.Thread(
                target=_output_queue_reader,
                args=(output_queue, con, tui_queue),
                daemon=True,
            )
            output_thread.start()
        with executor, state_mod.StateDB(state_db_path) as state_db:
            _start_ready_stages(
                stage_states=stage_states,
                executor=executor,
                futures=futures,
                cache_dir=cache_dir,
                output_queue=output_queue,
                max_stages=max_workers,
                mutex_counts=mutex_counts,
                overrides=overrides,
                lifecycle=lifecycle,
                checkout_modes=checkout_modes,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                cancel_event=cancel_event,
            )

            while futures:
                # Calculate wait timeout based on oldest running stage
                wait_timeout: float | None = None
                if stage_timeout is not None:
                    now = time.perf_counter()
                    for _future, stage_name in futures.items():
                        state = stage_states[stage_name]
                        if state.start_time:
                            elapsed = now - state.start_time
                            remaining = stage_timeout - elapsed
                            if wait_timeout is None or remaining < wait_timeout:
                                wait_timeout = max(0.1, remaining)  # At least 0.1s

                done, _ = concurrent.futures.wait(
                    futures.keys(),
                    timeout=wait_timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Check for timed-out stages if nothing completed
                if not done and stage_timeout is not None:
                    now = time.perf_counter()
                    timed_out = list[tuple[concurrent.futures.Future[StageResult], str]]()
                    for future, stage_name in futures.items():
                        state = stage_states[stage_name]
                        if state.start_time and (now - state.start_time) >= stage_timeout:
                            timed_out.append((future, stage_name))
                    for future, stage_name in timed_out:
                        futures.pop(future)
                        future.cancel()
                        state = stage_states[stage_name]
                        timeout_reason = f"Stage timed out after {stage_timeout}s"
                        # _mark_stage_failed uses lifecycle.mark_failed() which sends all notifications
                        _mark_stage_failed(
                            state,
                            timeout_reason,
                            stage_states,
                            lifecycle,
                        )
                        for mutex in state.mutex:
                            mutex_counts[mutex] -= 1
                    continue

                for future in done:
                    stage_name = futures.pop(future)
                    state = stage_states[stage_name]

                    try:
                        result = future.result()

                        # Aggregate metrics from worker process (before state change)
                        if "metrics" in result:
                            metrics.add_entries(result["metrics"])

                        # Use lifecycle to set state AND send all notifications
                        lifecycle.mark_completed(state, result)

                        # Handle downstream cascade for failed stages
                        if result["status"] == StageStatus.FAILED:
                            _handle_stage_failure(stage_name, stage_states, lifecycle)

                        # Apply deferred writes for successful stages (only in commit mode)
                        if result["status"] == StageStatus.RAN and not no_commit:
                            # Registry always stores single-file outputs (multi-file are expanded)
                            output_paths = [str(out.path) for out in state.info["outs"]]
                            _apply_deferred_writes(stage_name, output_paths, result, state_db)

                    except concurrent.futures.BrokenExecutor as e:
                        _mark_stage_failed(state, f"Worker died: {e}", stage_states, lifecycle)
                        logger.error(f"Worker process died while running '{stage_name}'")

                    except Exception as e:
                        _mark_stage_failed(state, str(e), stage_states, lifecycle)

                    for downstream_name in state.downstream:
                        downstream_state = stage_states.get(downstream_name)
                        if downstream_state:
                            downstream_state.upstream_unfinished.discard(stage_name)

                    # Release mutex locks (regardless of success/failure)
                    for mutex in state.mutex:
                        mutex_counts[mutex] -= 1
                        if mutex_counts[mutex] < 0:
                            logger.error(
                                f"Mutex '{mutex}' released when not held (bug in mutex tracking)"
                            )
                            mutex_counts[mutex] = 0  # Reset to valid state

                if error_mode == OnError.FAIL:
                    failed_stages = [
                        name for name, s in stage_states.items() if s.status == StageStatus.FAILED
                    ]
                    if failed_stages:
                        failed_stage_name = failed_stages[0]  # Use first failure for reason

                        # Cancel all pending futures (no-op for already-running workers)
                        for f in futures:
                            f.cancel()

                        # Mark all unfinished stages as skipped due to upstream failure
                        # Explicit check for READY (waiting) and IN_PROGRESS (running) -
                        # don't use "not in finished" as UNKNOWN indicates a bug state
                        unfinished = {StageStatus.READY, StageStatus.IN_PROGRESS}
                        for state in stage_states.values():
                            if state.status in unfinished:
                                lifecycle.mark_skipped_upstream(state, failed_stage_name)
                        return

                # Check for cancellation - mark remaining READY stages as cancelled
                if cancel_event is not None and cancel_event.is_set():
                    for state in stage_states.values():
                        if state.status == StageStatus.READY:
                            result = StageResult(
                                status=StageStatus.SKIPPED,
                                reason="cancelled",
                                output_lines=[],
                            )
                            lifecycle.mark_completed(state, result)
                    if not futures:
                        # No running stages - exit immediately
                        return
                    # Otherwise let running stages complete before exiting

                slots_available = max_workers - len(futures)
                if slots_available > 0:
                    _start_ready_stages(
                        stage_states=stage_states,
                        executor=executor,
                        futures=futures,
                        cache_dir=cache_dir,
                        output_queue=output_queue,
                        max_stages=slots_available,
                        mutex_counts=mutex_counts,
                        overrides=overrides,
                        lifecycle=lifecycle,
                        checkout_modes=checkout_modes,
                        force=force,
                        no_commit=no_commit,
                        no_cache=no_cache,
                        cancel_event=cancel_event,
                    )
    finally:
        # Signal output thread to stop - may fail if queue is broken
        with contextlib.suppress(OSError, ValueError):
            output_queue.put(None)
        if output_thread:
            output_thread.join(timeout=1.0)
        # Clean up manager if we created one (prevents orphaned subprocess)
        if local_manager is not None:
            local_manager.shutdown()


def _output_queue_reader(
    output_q: mp.Queue[OutputMessage],
    con: console.Console | None,
    tui_queue: TuiQueue | None = None,
) -> None:
    """Read output messages from worker processes and display/forward them."""
    while True:
        try:
            msg = output_q.get(timeout=0.02)
            if msg is None:
                break
            stage_name, line, is_stderr = msg
            if con:
                con.stage_output(stage_name, line, is_stderr)
            if tui_queue:
                tui_queue.put(
                    TuiLogMessage(
                        type=TuiMessageType.LOG,
                        stage=stage_name,
                        line=line,
                        is_stderr=is_stderr,
                        timestamp=time.time(),
                    )
                )
        except queue.Empty:
            continue
        except (EOFError, OSError, BrokenPipeError):
            # Queue was closed or broken - exit gracefully
            logger.debug("Output queue reader exiting: queue closed or broken")
            # Notify TUI that log streaming was interrupted (tui_queue is reliable since it's stdlib queue.Queue)
            if tui_queue:
                tui_queue.put_nowait(
                    TuiWatchMessage(
                        type=TuiMessageType.WATCH,
                        status=WatchStatus.ERROR,
                        message="Log streaming interrupted - logs may be incomplete",
                    )
                )
            break


def _start_ready_stages(
    stage_states: dict[str, StageState],
    executor: concurrent.futures.Executor,
    futures: dict[concurrent.futures.Future[StageResult], str],
    cache_dir: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    max_stages: int,
    mutex_counts: collections.defaultdict[str, int],
    overrides: parameters.ParamsOverrides,
    lifecycle: StageLifecycle,
    checkout_modes: list[cache.CheckoutMode] | None = None,
    force: bool = False,
    no_commit: bool = False,
    no_cache: bool = False,
    cancel_event: threading.Event | None = None,
) -> None:
    """Find and start stages that are ready to execute."""
    # Check cancellation - stages can become READY after the main loop's check
    # (e.g., when a running stage completes and unblocks downstream)
    if cancel_event is not None and cancel_event.is_set():
        return

    checkout_modes = checkout_modes or config.get_checkout_mode_order()
    started = 0

    for stage_name, state in stage_states.items():
        if started >= max_stages:
            break

        if state.status != StageStatus.READY:
            continue

        if state.upstream_unfinished:
            continue

        # Check mutex availability - skip if any mutex group is held
        if any(mutex_counts[m] > 0 for m in state.mutex):
            continue

        # Exclusive mutex handling:
        # - If this stage is exclusive, wait until no other stages are running
        # - If any exclusive stage is running, no other stages can start
        is_exclusive = EXCLUSIVE_MUTEX in state.mutex
        if is_exclusive and len(futures) > 0:
            continue  # Exclusive stage must wait for all others to finish
        if not is_exclusive and mutex_counts[EXCLUSIVE_MUTEX] > 0:
            continue  # Non-exclusive stage can't start while exclusive is running

        # Show explanation before starting if in explain mode
        if lifecycle.explain_mode and lifecycle.console:
            explanation = _get_stage_explanation(state.info, cache_dir, overrides)
            lifecycle.console.explain_stage(explanation)

        # Acquire mutex locks before changing status
        for mutex in state.mutex:
            mutex_counts[mutex] += 1

        worker_info = _prepare_worker_info(
            state.info, overrides, checkout_modes, lifecycle.run_id, force, no_commit, no_cache
        )

        try:
            future = executor.submit(
                worker.execute_stage,
                stage_name,
                worker_info,
                cache_dir,
                output_queue,
            )
            futures[future] = stage_name
            started += 1

            # Mark as in-progress and send all notifications via lifecycle
            lifecycle.mark_started(state)
        except Exception as e:
            # Rollback mutex acquisition on submission failure
            for mutex in state.mutex:
                mutex_counts[mutex] -= 1
            _mark_stage_failed(state, f"Failed to submit: {e}", stage_states, lifecycle)


def _prepare_worker_info(
    stage_info: registry.RegistryStageInfo,
    overrides: parameters.ParamsOverrides,
    checkout_modes: list[cache.CheckoutMode],
    run_id: str,
    force: bool,
    no_commit: bool,
    no_cache: bool,
) -> worker.WorkerStageInfo:
    """Prepare stage info for pickling to worker process."""
    return worker.WorkerStageInfo(
        func=stage_info["func"],
        fingerprint=stage_info["fingerprint"],
        deps=stage_info["deps_paths"],
        outs=stage_info["outs"],
        signature=stage_info["signature"],
        params=stage_info["params"],
        variant=stage_info["variant"],
        overrides=overrides,
        checkout_modes=checkout_modes,
        run_id=run_id,
        force=force,
        no_commit=no_commit,
        no_cache=no_cache,
        dep_specs=stage_info["dep_specs"],
        out_specs=stage_info["out_specs"],
        params_arg_name=stage_info["params_arg_name"],
    )


def _apply_deferred_writes(
    stage_name: str,
    output_paths: list[str],
    result: StageResult,
    state_db: state_mod.StateDB,
) -> None:
    """Apply deferred StateDB writes from worker result."""
    if "deferred_writes" not in result:
        return
    state_db.apply_deferred_writes(stage_name, output_paths, result["deferred_writes"])


def _mark_stage_failed(
    state: StageState,
    reason: str,
    stage_states: dict[str, StageState],
    lifecycle: StageLifecycle | None = None,
) -> None:
    """Mark a stage as failed and handle downstream effects.

    Uses lifecycle.mark_failed() when available to send notifications atomically
    with state updates. Then handles downstream cascade via _handle_stage_failure.
    """
    if lifecycle:
        # Use lifecycle to set state AND send notifications
        lifecycle.mark_failed(state, reason)
    else:
        # Fallback for non-TUI mode
        state.result = StageResult(status=StageStatus.FAILED, reason=reason, output_lines=[])
        state.status = StageStatus.FAILED
        state.end_time = time.perf_counter()
    _handle_stage_failure(state.name, stage_states, lifecycle)


def _handle_stage_failure(
    failed_stage: str,
    stage_states: dict[str, StageState],
    lifecycle: StageLifecycle | None = None,
) -> None:
    """Handle stage failure by marking downstream stages as skipped.

    This handles both FAIL and KEEP_GOING modes: downstream stages are always
    skipped when their upstream fails. The difference between modes is whether
    independent stages continue (KEEP_GOING) or the pipeline stops (FAIL).
    """
    to_skip = set[str]()
    bfs_queue = collections.deque([failed_stage])
    visited = set[str]()

    while bfs_queue:
        current = bfs_queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        state = stage_states.get(current)
        if not state:
            continue

        for downstream in state.downstream:
            if downstream not in visited:
                to_skip.add(downstream)
                bfs_queue.append(downstream)

    for stage_name in to_skip:
        state = stage_states.get(stage_name)
        if state and state.status == StageStatus.READY:
            if lifecycle:
                # Use lifecycle to update state AND send notifications
                lifecycle.mark_skipped_upstream(state, failed_stage)
            else:
                # Fallback for non-TUI mode (no notifications needed)
                state.status = StageStatus.SKIPPED
                state.result = StageResult(
                    status=StageStatus.SKIPPED,
                    reason=f"upstream '{failed_stage}' failed",
                    output_lines=[],
                )


def _build_results(stage_states: dict[str, StageState]) -> dict[str, ExecutionSummary]:
    """Build results dict from stage states."""
    results = dict[str, ExecutionSummary]()
    for name, state in stage_states.items():
        if state.result:
            results[name] = ExecutionSummary(
                status=state.result["status"],
                reason=state.result["reason"],
            )
        elif state.status == StageStatus.SKIPPED:
            results[name] = ExecutionSummary(status=StageStatus.SKIPPED, reason="upstream failed")
        else:
            results[name] = ExecutionSummary(status=StageStatus.UNKNOWN, reason="never executed")
    return results


def _get_stage_explanation(
    stage_info: registry.RegistryStageInfo,
    cache_dir: pathlib.Path,
    overrides: parameters.ParamsOverrides,
) -> StageExplanation:
    """Compute explanation for a single stage."""
    return explain.get_stage_explanation(
        stage_info["name"],
        stage_info["fingerprint"],
        stage_info["deps_paths"],
        stage_info["params"],
        overrides,
        cache_dir,
    )


def _check_uncached_incremental_outputs(
    execution_order: list[str],
    cache_dir: pathlib.Path,
) -> list[tuple[str, str]]:
    """Check for IncrementalOut files that exist but aren't cached.

    Returns list of (stage_name, output_path) tuples for uncached files.
    """
    uncached = list[tuple[str, str]]()

    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)
        stage_outs = stage_info["outs"]

        # Read lock file to get cached output hashes
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(cache_dir))
        lock_data = stage_lock.read()
        output_hashes = lock_data.get("output_hashes", {}) if lock_data else {}

        for out in stage_outs:
            if isinstance(out, outputs.IncrementalOut):
                # Registry always stores single-file outputs (multi-file are expanded)
                out_path = str(out.path)
                path = pathlib.Path(out_path)
                # File exists on disk but has no cache entry
                if path.exists() and out_path not in output_hashes:
                    uncached.append((stage_name, out_path))

    return uncached


def _write_run_history(
    run_id: str,
    stage_states: dict[str, StageState],
    cache_dir: pathlib.Path,
    targeted_stages: list[str],
    execution_order: list[str],
    started_at: str,
    ended_at: str,
    retention: int,
) -> None:
    """Build and write run manifest to StateDB."""

    stages_records = dict[str, run_history.StageRunRecord]()
    for name, state in stage_states.items():
        stage_lock = lock.StageLock(name, lock.get_stages_dir(cache_dir))
        lock_data = stage_lock.read()

        if lock_data:
            stage_info = registry.REGISTRY.get(name)
            # Registry always stores single-file outputs (multi-file are expanded)
            out_paths = [str(out.path) for out in stage_info["outs"]]
            input_hash = run_history.compute_input_hash_from_lock(lock_data, out_paths)
        else:
            input_hash = "<no-lock>"

        duration_ms = int((state.get_duration() or 0) * 1000)
        status = state.result["status"] if state.result else StageStatus.UNKNOWN
        reason = state.result["reason"] if state.result else ""

        stages_records[name] = run_history.StageRunRecord(
            input_hash=input_hash,
            status=status,
            reason=reason,
            duration_ms=duration_ms,
        )

    manifest = run_history.RunManifest(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        targeted_stages=targeted_stages,
        execution_order=execution_order,
        stages=stages_records,
    )

    state_db_path = project.get_project_root() / ".pivot" / "state.db"
    with state_mod.StateDB(state_db_path) as state_db:
        state_db.write_run(manifest)
        state_db.prune_runs(retention)
