from __future__ import annotations

import atexit
import contextlib
import functools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Literal, TypedDict

import loky

from pivot import config, discovery, exceptions, outputs, parameters
from pivot.executor import worker
from pivot.storage import cache, lock, track
from pivot.storage import state as state_mod
from pivot.types import (
    DisplayCategory,
    OnError,
    StageResult,
    StageStatus,
    categorize_stage_result,
)

if TYPE_CHECKING:
    import concurrent.futures
    import threading
    from collections.abc import Callable

    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo
    from pivot.types import RunJsonEvent

logger = logging.getLogger(__name__)

# Special mutex that means "run exclusively" - no other stages run concurrently
EXCLUSIVE_MUTEX = "*"


def _noop() -> None:
    """Module-level for pickling."""
    pass


def compute_max_workers(stage_count: int, override: int | None = None) -> int:
    """Single source of truth for max_workers calculation.

    Args:
        stage_count: Number of stages to run
        override: CLI override for max_workers (takes precedence over config)

    The effective max_workers is determined by:
    1. CLI override (if provided)
    2. Config value (core.max_workers)

    Negative values mean "cpu_count + value" (e.g., -2 means cpu_count - 2).
    The result is always clamped to [1, stage_count].
    """
    cpu_count = loky.cpu_count() or 1
    max_workers = override if override is not None else config.get_max_workers()

    # Negative values are relative to CPU count
    if max_workers < 0:
        max_workers = cpu_count + max_workers

    return max(1, min(max_workers, stage_count))


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
    workers = compute_max_workers(stage_count, max_workers)
    _ensure_deterministic_environment()
    _ensure_cleanup_registered()
    pool = loky.get_reusable_executor(max_workers=workers)
    _warm_workers(pool, workers)
    return workers


def restart_workers(stage_count: int, max_workers: int | None = None) -> int:
    """Kill existing workers and spawn fresh ones. For code reload in watch mode.

    Unlike prepare_workers(), this kills existing workers first, then warms the new pool.
    """
    if stage_count <= 0:
        return 1
    workers = compute_max_workers(stage_count, max_workers)
    _ensure_deterministic_environment()
    _ensure_cleanup_registered()
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

    Uses shared categorization for consistent treatment across TUI and plain mode.

    Returns:
        Tuple of (ran, cached, blocked, failed) counts
    """
    ran = cached = blocked = failed = 0
    for result in results.values():
        category = categorize_stage_result(result["status"], result["reason"])
        match category:
            case DisplayCategory.SUCCESS:
                ran += 1
            case DisplayCategory.FAILED:
                failed += 1
            case DisplayCategory.BLOCKED:
                blocked += 1
            case DisplayCategory.CACHED | DisplayCategory.CANCELLED:
                # Cancelled stages are counted with cached (both are skipped, not failed)
                cached += 1
            case DisplayCategory.UNKNOWN | DisplayCategory.PENDING | DisplayCategory.RUNNING:
                # UNKNOWN/PENDING/RUNNING shouldn't appear in final results
                pass
    return ran, cached, blocked, failed


def run(
    stages: list[str] | None = None,
    single_stage: bool = False,
    cache_dir: pathlib.Path | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    on_error: OnError | str = OnError.FAIL,
    allow_uncached_incremental: bool = False,
    force: bool = False,
    no_commit: bool = False,
    no_cache: bool = False,
    progress_callback: Callable[[RunJsonEvent], None] | None = None,
    cancel_event: threading.Event | None = None,
    checkout_missing: bool = False,
    pipeline: Pipeline | None = None,
) -> dict[str, ExecutionSummary]:
    """Execute pipeline stages via Engine.

    This function creates an Engine with OneShotSource and ResultCollectorSink,
    runs the event loop until completion, and returns the collected results.
    All execution orchestration is handled by Engine's event-driven architecture.

    Args:
        stages: Target stages to run (and their dependencies). If None, runs all.
        single_stage: If True, run only the specified stages without dependencies.
        cache_dir: Directory for lock files. Defaults to .pivot/cache.
        parallel: If True, run independent stages in parallel (default: True).
        max_workers: Max concurrent stages (default: min(cpu_count, 8)).
        on_error: Error handling mode - "fail" or "keep_going".
        allow_uncached_incremental: If True, skip safety check for uncached IncrementalOut files.
        force: If True, bypass cache and force all stages to re-execute.
        no_commit: If True, defer lock files to pending dir (faster iteration).
        no_cache: If True, skip caching outputs entirely (maximum iteration speed).
        progress_callback: Callback for JSONL progress events (stage start/complete).
        cancel_event: If set, stop starting new stages and mark pending as cancelled.
        checkout_missing: If True, restore missing tracked files from cache before running.

    Returns:
        Dict of stage_name -> {status: "ran"|"skipped"|"failed", reason: str}
    """
    # Import project_lock early to acquire lock BEFORE importing Engine.
    # Engine import is slow (~500ms) and we need to hold the lock during that time
    # to prevent race conditions with concurrent commit operations.
    from pivot.storage import project_lock

    # Acquire lock FIRST if in no_commit mode - before any slow imports
    lock_context = project_lock.pending_state_lock() if no_commit else contextlib.nullcontext()

    with lock_context:
        return _run_inner(
            stages=stages,
            single_stage=single_stage,
            cache_dir=cache_dir,
            parallel=parallel,
            max_workers=max_workers,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            checkout_missing=checkout_missing,
            pipeline=pipeline,
        )


def _run_inner(
    stages: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    parallel: bool,
    max_workers: int | None,
    on_error: OnError | str,
    allow_uncached_incremental: bool,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    progress_callback: Callable[[RunJsonEvent], None] | None,
    cancel_event: threading.Event | None,
    checkout_missing: bool,
    pipeline: Pipeline | None,
) -> dict[str, ExecutionSummary]:
    """Inner implementation of run(), called with lock already held if needed."""
    # Import here to avoid circular import (engine imports executor_core)
    from pivot.engine.engine import Engine
    from pivot.engine.sinks import ResultCollectorSink
    from pivot.engine.sources import OneShotSource

    if not isinstance(on_error, OnError):
        try:
            on_error = OnError(on_error)
        except ValueError:
            raise ValueError(
                f"Invalid on_error mode: {on_error}. Use 'fail' or 'keep_going'"
            ) from None

    # Use provided pipeline or discover from project root
    if pipeline is None:
        pipeline = discovery.discover_pipeline()
        if pipeline is None:
            raise exceptions.PipelineNotFoundError(
                "No pipeline found. Create pivot.yaml or pipeline.py to define stages."
            )

    # Create engine with event-driven execution
    with Engine(pipeline=pipeline) as engine:
        # Pass cancel event to engine if provided
        if cancel_event is not None:
            engine.set_cancel_event(cancel_event)

        # Set progress callback on engine for JSONL output
        if progress_callback is not None:
            engine.set_progress_callback(progress_callback)

        # Add ResultCollectorSink to collect results
        result_sink = ResultCollectorSink()
        engine.add_sink(result_sink)

        # Create OneShotSource with all orchestration parameters
        source = OneShotSource(
            stages=stages,
            force=force,
            reason="cli",
            single_stage=single_stage,
            parallel=parallel,
            max_workers=max_workers,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            cache_dir=cache_dir,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
        engine.add_source(source)

        # Run event loop until all stages complete
        engine.run(exit_on_completion=True)

        # Return collected results
        return result_sink.get_execution_summaries()


# =========================================================================
# Functions used by Engine for orchestration
# =========================================================================


@functools.cache
def _ensure_deterministic_environment() -> None:
    """Set environment variables for deterministic worker execution.

    Must be called before ANY loky executor creation, as workers inherit
    parent environment at spawn time. Using setdefault respects user overrides.
    """
    os.environ.setdefault("PYTHONHASHSEED", "0")


def create_executor(max_workers: int) -> concurrent.futures.Executor:
    """Get reusable loky executor - workers persist across calls for efficiency."""
    _ensure_deterministic_environment()
    _ensure_cleanup_registered()
    return loky.get_reusable_executor(max_workers=max_workers)


def prepare_worker_info(
    stage_info: RegistryStageInfo,
    overrides: parameters.ParamsOverrides,
    checkout_modes: list[cache.CheckoutMode],
    run_id: str,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
) -> worker.WorkerStageInfo:
    """Prepare stage info for pickling to worker process.

    Uses stage's state_dir if set, otherwise falls back to default_state_dir.
    """
    # Use stage's state_dir if set, otherwise fall back to default
    state_dir = stage_info["state_dir"] or default_state_dir

    # Ensure state directory exists (workers open StateDB in readonly mode)
    state_dir.mkdir(parents=True, exist_ok=True)

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
        project_root=project_root,
        state_dir=state_dir,
    )


def apply_deferred_writes(
    stage_name: str,
    output_paths: list[str],
    result: StageResult,
    state_db: state_mod.StateDB,
) -> None:
    """Apply deferred StateDB writes from worker result."""
    if "deferred_writes" not in result:
        return
    # result["deferred_writes"] is DeferredWrites from worker
    state_db.apply_deferred_writes(stage_name, output_paths, result["deferred_writes"])


# =========================================================================
# Safety checks used before execution
# =========================================================================


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


def verify_tracked_files(project_root: pathlib.Path, checkout_missing: bool = False) -> None:
    """Verify all .pvt tracked files exist and warn on hash mismatches.

    Args:
        project_root: Project root directory.
        checkout_missing: If True, restore missing tracked files from cache before validating.
            Only restores files that don't exist - never overwrites existing files.
    """
    from pivot import metrics

    tracked_files = track.discover_pvt_files(project_root)
    if not tracked_files:
        return

    with metrics.timed("core.verify_tracked_files"):
        missing = list[str]()
        cache_dir = config.get_cache_dir() / "files"

        with state_mod.StateDB(config.get_state_db_path()) as state_db:
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


def check_uncached_incremental_outputs(
    execution_order: list[str],
    all_stages: dict[str, RegistryStageInfo],
) -> list[tuple[str, str]]:
    """Check for IncrementalOut files that exist but aren't cached.

    Returns list of (stage_name, output_path) tuples for uncached files.
    """
    uncached = list[tuple[str, str]]()
    state_dir = config.get_state_dir()

    for stage_name in execution_order:
        stage_info = all_stages[stage_name]
        stage_outs = stage_info["outs"]

        # Read lock file to get cached output hashes
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
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
