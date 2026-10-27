from __future__ import annotations

import atexit
import contextlib
import functools
import logging
import multiprocessing as mp
import multiprocessing.resource_tracker
import os
import pathlib
import signal
import threading
import typing
from multiprocessing import managers
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import loky
import loky.backend.resource_tracker
import loky.reusable_executor

from pivot import config, discovery, exceptions, parameters, project, registry, types
from pivot import state_dir as state_dir_mod
from pivot.executor import worker
from pivot.storage import cache, lock, track
from pivot.storage import state as state_mod
from pivot.storage import store as store_mod
from pivot.types import OnError, StageResult, StageStatus

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Mapping

    from pivot.engine import types as engine_types
    from pivot.registry import RegistryStageInfo

logger = logging.getLogger(__name__)

ONE_SHOT_WORKER_IDLE_TIMEOUT_S: Final[float] = 10.0


def start_resource_trackers() -> None:
    """Start resource trackers before Textual replaces ``sys.stderr`` with an invalid fd."""
    multiprocessing.resource_tracker.ensure_running()
    loky.backend.resource_tracker.ensure_running()


def _noop() -> None:
    """Module-level for pickling."""
    pass


def _exit_when_parent_dies(parent_pid: int) -> None:
    """Exit the output-queue manager after its owning Pivot process dies.

    Workers use ``worker.unwind_when_stopped`` instead, which lets the running stage unwind.
    """

    def _watch_parent() -> None:
        poll = threading.Event()
        while os.getppid() == parent_pid:
            poll.wait(1)
        os._exit(0)

    worker._start_pivot_thread(  # pyright: ignore[reportPrivateUsage] - worker owns coverage bypass
        threading.Thread(target=_watch_parent, daemon=True, name="pivot-parent-watch")
    )


def _init_manager(parent_pid: int) -> None:
    """Initialize the output-queue manager process.

    A service manager's stop (systemd, GNU ``timeout``) signals the whole process group. The
    manager ignores SIGTERM, as multiprocessing already makes it ignore SIGINT, so it stays up
    until its owner shuts it down and its finalizer removes its ``pymp-*`` directory. If the
    owner dies instead, the parent-death watcher exits it.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _exit_when_parent_dies(parent_pid)


def configured_worker_count(override: int | None = None) -> int:
    """The configured worker count: the CLI override, else ``core.max_workers``.

    Negative values mean "cpu_count + value" (e.g., -2 means cpu_count - 2). The result is at
    least 1. This is the ``cpu`` resource capacity.
    """
    cpu_count = loky.cpu_count() or 1
    max_workers = override if override is not None else config.get_max_workers()
    if max_workers < 0:
        max_workers = cpu_count + max_workers
    return max(1, max_workers)


def _current_reusable_executor() -> tuple[
    loky.ProcessPoolExecutor | None, Mapping[str, object] | None
]:
    """Return loky's current reusable executor and the arguments that created it."""
    if hasattr(loky.reusable_executor, "_executor"):
        return (
            loky.reusable_executor._executor,  # pyright: ignore[reportPrivateUsage]
            loky.reusable_executor._executor_kwargs,  # pyright: ignore[reportPrivateUsage]
        )

    storage = getattr(loky.reusable_executor, "_executor_storage", None)
    if (
        storage is None
        or not hasattr(storage, "executor")
        or not hasattr(storage, "executor_kwargs")
    ):
        version = getattr(loky, "__version__", "unknown")
        raise RuntimeError(f"Unsupported loky {version}: no reusable-executor singleton")
    return storage.executor, storage.executor_kwargs


def is_executor_usable(
    executor: loky.ProcessPoolExecutor | None,
    max_workers: int,
    *,
    current_kwargs: Mapping[str, object] | None = None,
    requested_kwargs: Mapping[str, object] | None = None,
) -> bool:
    """Whether ``executor`` can serve a pool with ``max_workers``.

    When callers provide argument mappings, the test also matches loky's ``reuse="auto"``
    predicate: an executor whose initializer, timeout or another creation argument differs is
    replaced and must be created at the requested width.
    """
    return (
        executor is not None
        and not executor._flags.shutdown
        and executor._flags.broken is None
        and executor.queue_size >= max_workers
        and (requested_kwargs is None or current_kwargs == requested_kwargs)
    )


def _requested_reusable_executor_kwargs(
    idle_timeout_s: float | None, user_code_root: pathlib.Path | None
) -> dict[str, object]:
    """Return the constructor arguments loky compares to decide whether to reuse its executor."""
    return {
        "context": None,
        "timeout": idle_timeout_s,
        "job_reducers": None,
        "result_reducers": None,
        "initializer": _init_worker,
        "initargs": (os.getpid(), None if user_code_root is None else str(user_code_root)),
        "env": None,
    }


def _reusable_worker_target(
    max_workers: int,
    *,
    kill_workers: bool,
    requested_kwargs: Mapping[str, object] | None = None,
) -> int:
    """Keep reuse from invoking loky's blocking ``_resize`` between runs.

    The live executor's own width is passed back only when loky will reuse it. When loky will
    replace it (missing, shut down, broken, argument mismatch or a call queue too small for
    ``max_workers``), the requested width is passed, so the new executor's queue fits it.
    """
    executor, current_kwargs = _current_reusable_executor()
    if kill_workers or not is_executor_usable(
        executor,
        max_workers,
        current_kwargs=current_kwargs,
        requested_kwargs=requested_kwargs,
    ):
        return max_workers
    assert executor is not None
    return executor._max_workers


def _get_executor(
    max_workers: int,
    *,
    idle_timeout_s: float | None,
    kill_workers: bool = False,
) -> loky.ProcessPoolExecutor:
    """Return Pivot's reusable executor without resizing warm workers between runs.

    Every worker starts with ``_init_worker``. While a discovery's import session lasts, it
    also imports user code as that session does (``discovery.install_user_code_loader``), so a
    worker never runs an edited module's previous code; loky replaces a live pool whose
    initializer arguments differ.
    """
    _ensure_deterministic_environment()
    _ensure_cleanup_registered()
    root = discovery.import_session_root()
    requested_kwargs = _requested_reusable_executor_kwargs(idle_timeout_s, root)
    return loky.get_reusable_executor(
        max_workers=_reusable_worker_target(
            max_workers,
            kill_workers=kill_workers,
            requested_kwargs=requested_kwargs,
        ),
        timeout=idle_timeout_s,
        kill_workers=kill_workers,
        reuse=False if kill_workers else "auto",
        initializer=_init_worker,
        initargs=(os.getpid(), None if root is None else str(root)),
    )


def submit[T](
    pool: loky.ProcessPoolExecutor,
    max_workers: int,
    fn: typing.Callable[..., T],
    /,
    *args: object,
    prestart_target: int = 0,
    keepalive_pings: int = 0,
    **kwargs: object,
) -> concurrent.futures.Future[T]:
    """Submit work and set how many workers loky keeps: the larger of the unfinished work count
    and ``prestart_target``, at most ``max_workers``.

    A worker starts when every live worker already has work, or to reach ``prestart_target``.
    ``prestart_target`` is the run's resource width at its first submission. Later, it counts
    the direct consumers that the scheduler predicts will become resource-admissible when this
    stage completes. ``keepalive_pings`` counts submitted keep-alive pings that have not
    finished. They last milliseconds, so they are not work.
    """
    pending = len(pool._pending_work_items) - keepalive_pings
    target_workers = max(pending + 1, prestart_target)
    pool._max_workers = min(max_workers, target_workers)
    return pool.submit(fn, *args, **kwargs)


def submit_to_live_workers[T](
    pool: loky.ProcessPoolExecutor, fn: typing.Callable[..., T], /, *args: object
) -> concurrent.futures.Future[T]:
    """Submit work that only a worker already alive may take; it never starts a worker."""
    pool._max_workers = len(pool._processes)
    return pool.submit(fn, *args)


def idle_worker_count(pool: loky.ProcessPoolExecutor) -> int:
    """Return how many live workers have no submitted work."""
    return max(0, len(pool._processes) - len(pool._pending_work_items))


def _init_worker(parent_pid: int, user_code_root: str | None) -> None:
    """Loky worker initializer: stop the worker when it is told to or ``parent_pid`` dies
    (``worker.unwind_when_stopped``), record the user modules each stage imports
    (``worker.install_import_hooks``), and import user code through discovery's checked loader
    when a discovery's import session for ``user_code_root`` lasts."""
    worker.unwind_when_stopped(parent_pid)
    worker.install_import_hooks()
    if user_code_root is not None:
        discovery.install_user_code_loader(user_code_root)


def create_manager() -> managers.SyncManager:
    """Start a spawn-context manager that exits when this process dies."""
    manager = managers.SyncManager(ctx=mp.get_context("spawn"))
    manager.start(initializer=_init_manager, initargs=(os.getpid(),))
    return manager


def restart_workers(
    max_workers: int, *, warm_workers: int = 1, idle_timeout_s: float | None
) -> None:
    """Replace stale workers and start ``warm_workers`` fresh ones together for the next run.

    Every warm worker starts at once, and the call returns when one of them has run a no-op;
    the rest finish starting in parallel, so the next run waits at most one worker start-up.

    If the warm-up fails, the executor this call created is shut down, its workers killed,
    before the error propagates. Loky 3.7 keeps it in the calling thread's slot, which no
    ``WorkerPool`` holds, so nothing else would stop its workers.
    """
    pool = create_fresh_executor(max_workers, idle_timeout_s=idle_timeout_s)
    try:
        submit(pool, max_workers, _noop, prestart_target=warm_workers).result()
    except BaseException:
        pool.shutdown(wait=True, kill_workers=True)
        raise


def _cleanup_worker_pool() -> None:
    """Kill loky worker pool on process exit to prevent orphaned workers."""
    with contextlib.suppress(Exception):
        create_fresh_executor(1, idle_timeout_s=None)


@functools.cache  # Ensures single atexit registration across threads
def _ensure_cleanup_registered() -> None:
    atexit.register(_cleanup_worker_pool)


class ExecutionSummary(TypedDict):
    """Summary result for a single stage after execution (returned by executor.run)."""

    status: Literal[
        StageStatus.RAN,
        StageStatus.CACHED,
        StageStatus.BLOCKED,
        StageStatus.CANCELLED,
        StageStatus.FAILED,
        StageStatus.UNKNOWN,
    ]
    reason: str
    input_hash: str | None


def count_results(results: dict[str, ExecutionSummary]) -> tuple[int, int, int, int, int]:
    """Count results by category.

    Returns:
        Tuple of (ran, cached, blocked, cancelled, failed) counts.
    """
    ran = cached = blocked = cancelled = failed = 0
    for result in results.values():
        match result["status"]:
            case StageStatus.RAN:
                ran += 1
            case StageStatus.FAILED:
                failed += 1
            case StageStatus.BLOCKED:
                blocked += 1
            case StageStatus.CACHED:
                cached += 1
            case StageStatus.CANCELLED:
                cancelled += 1
            case _:
                pass
    return ran, cached, blocked, cancelled, failed


def run(
    stages: list[str] | None = None,
    single_stage: bool = False,
    cache_dir: pathlib.Path | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    resource_capacity: dict[str, int] | None = None,
    on_error: OnError | str = OnError.FAIL,
    allow_uncached_incremental: bool = False,
    force: bool = False,
    no_commit: bool = False,
    checkout_missing: bool = False,
    pipeline: registry.PipelineLike | None = None,
) -> dict[str, ExecutionSummary]:
    """Execute pipeline stages via Engine.

    This function creates an Engine with OneShotSource and ResultCollectorSink,
    runs the event loop until completion, and returns the collected results.
    All execution orchestration is handled by Engine's event-driven architecture.

    Args:
        stages: Target stages to run (and their dependencies). If None, runs all.
        single_stage: If True, run only the specified stages without dependencies.
        cache_dir: Cache directory, defaulting to `<state dir>/cache`.
        parallel: If True, run independent stages in parallel (default: True).
        max_workers: Most workers, and the ``cpu`` resource capacity (default: ``core.max_workers``).
        resource_capacity: Per-run resource capacity overrides.
        on_error: Error handling mode - "fail" or "keep_going".
        allow_uncached_incremental: If True, skip the uncached incremental-output safety check.
        force: If True, bypass cache and force all stages to re-execute.
        no_commit: If True, skip writing lock files (faster iteration).
        checkout_missing: If True, restore missing tracked files from cache before running.

    Returns:
        Dict of stage_name -> {status: "ran"|"cached"|"blocked"|"cancelled"|"failed", reason: str}
    """
    return _run_inner(
        stages=stages,
        single_stage=single_stage,
        cache_dir=cache_dir,
        parallel=parallel,
        max_workers=max_workers,
        resource_capacity=resource_capacity,
        on_error=on_error,
        allow_uncached_incremental=allow_uncached_incremental,
        force=force,
        no_commit=no_commit,
        checkout_missing=checkout_missing,
        pipeline=pipeline,
    )


def _run_inner(
    stages: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    parallel: bool,
    max_workers: int | None,
    resource_capacity: dict[str, int] | None,
    on_error: OnError | str,
    allow_uncached_incremental: bool,
    force: bool,
    no_commit: bool,
    checkout_missing: bool,
    pipeline: registry.PipelineLike | None,
) -> dict[str, ExecutionSummary]:
    """Inner implementation of run(), called with lock already held if needed."""
    # Import here to avoid circular import (engine imports executor_core)
    import anyio

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
                "No pipeline found. Create pipeline.py to define stages."
            )

    # Run async execution
    async def execute() -> dict[str, ExecutionSummary]:
        async with Engine(pipeline=pipeline) as eng:
            # Add ResultCollectorSink to collect results
            result_sink = ResultCollectorSink()
            eng.add_sink(cast("engine_types.EventSink", result_sink))

            # Create OneShotSource with all orchestration parameters
            source = OneShotSource(
                stages=stages,
                force=force,
                reason="cli",
                single_stage=single_stage,
                parallel=parallel,
                max_workers=max_workers,
                resource_capacity=resource_capacity,
                no_commit=no_commit,
                on_error=on_error,
                cache_dir=cache_dir,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
            eng.add_source(source)

            # Run event loop until all stages complete
            await eng.run(exit_on_completion=True)

            # Convert results from StageCompleted to ExecutionSummary
            raw_results = await result_sink.get_results()
            return {
                name: ExecutionSummary(
                    status=event["status"],
                    reason=event["reason"],
                    input_hash=event["input_hash"],
                )
                for name, event in raw_results.items()
            }

    try:
        return anyio.run(execute)
    except BaseExceptionGroup as eg:
        # Unwrap single-exception groups for cleaner error handling
        # anyio's task groups wrap exceptions in ExceptionGroup
        if len(eg.exceptions) == 1:
            raise eg.exceptions[0] from None
        raise


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


def create_executor(
    max_workers: int, *, idle_timeout_s: float | None = None
) -> loky.ProcessPoolExecutor:
    """Get Pivot's reusable executor without growing it before work is submitted."""
    return _get_executor(max_workers, idle_timeout_s=idle_timeout_s)


def create_fresh_executor(
    max_workers: int, *, idle_timeout_s: float | None = None
) -> loky.ProcessPoolExecutor:
    """Force loky to replace its reusable executor without waiting for a worker to warm."""
    return _get_executor(max_workers, idle_timeout_s=idle_timeout_s, kill_workers=True)


def prepare_worker_info(
    stage_info: RegistryStageInfo,
    pipeline: registry.PipelineLike,
    overrides: parameters.ParamsOverrides,
    run_id: str,
    force: bool,
    no_commit: bool,
    project_root: pathlib.Path,
    default_state_dir: pathlib.Path,
    store_spec: store_mod.StoreSpec,
) -> worker.WorkerStageInfo:
    """Prepare stage info for pickling to worker process.

    Uses stage's state_dir if set, otherwise falls back to default_state_dir. The worker gets
    this process's resolution of the project's ``core.state_dir``, not a directory derived from
    it, so it reads every state file where this process does.
    """
    # Use stage's state_dir if set, otherwise fall back to default
    state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)

    # Create the state directory through the common path so a newly configured directory has
    # its state ignore file before a worker writes under it.
    state_dir_mod.make_state_dir(state_dir)

    # The fingerprint is computed first: it sets `unsafe_fingerprint`.
    fingerprint = pipeline.ensure_fingerprint(stage_info["name"])
    return worker.WorkerStageInfo(
        func=stage_info["func"],
        fingerprint=fingerprint,
        deps=stage_info["deps"],
        outs=stage_info["outs"],
        store_spec=store_spec,
        incremental_inputs=stage_info["incremental_inputs"],
        signature=stage_info["signature"],
        params=stage_info["params"],
        variant=stage_info["variant"],
        overrides=overrides,
        run_id=run_id,
        force=force,
        no_commit=no_commit,
        params_arg_name=stage_info["params_arg_name"],
        project_root=project_root,
        state_dir=state_dir,
        state_dir_resolution=state_dir_mod.resolution_for(project_root),
        collection_params=stage_info["collection_params"],
        unsafe_fingerprint=stage_info["unsafe_fingerprint"],
        no_fingerprint=stage_info["no_fingerprint"],
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
    return cache.restore_from_cache(
        path,
        output_hash,
        cache_dir,
        checkout_modes=checkout_modes,
        state_dir=config.get_state_dir(),
    )


def restore_missing_tracked_files(project_root: pathlib.Path) -> None:
    """Restore each tracked file missing from the workspace from the cache.

    Files that exist are neither checked nor touched; a file the cache lacks stays missing for
    ``verify_tracked_files`` to report.
    """
    cache_dir = config.get_cache_dir() / "files"
    for data_path, track_data in track.discover_pvt_files(project_root).items():
        path = pathlib.Path(data_path)
        if not path.exists() and _restore_tracked_file(path, track_data, cache_dir):
            logger.info(f"Restored tracked file: {data_path}")


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

    _t = metrics.start()
    missing = list[str]()
    cache_dir = config.get_cache_dir() / "files"

    with state_mod.StateDB(config.get_state_dir()) as state_db:
        for data_path, track_data in tracked_files.items():
            path = pathlib.Path(data_path)

            # Try to hash the file - handles race conditions where file disappears
            try:
                if path.is_file():
                    current_hash, _ = cache.hash_file(path, state_db)
                elif path.is_dir():
                    current_hash = cache.hash_directory(path, state_db, cache_dir=cache_dir)["hash"]
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

    metrics.end("core.verify_tracked_files", _t)
    if missing:
        raise exceptions.TrackedFileMissingError(missing, checkout_attempted=checkout_missing)


def check_uncached_incremental_outputs(
    execution_order: list[str],
    all_stages: dict[str, RegistryStageInfo],
) -> list[tuple[str, str]]:
    """Check for incremental artifacts that exist without a recorded cache hash."""
    uncached = list[tuple[str, str]]()
    default_state_dir = config.get_state_dir()
    workspace = store_mod.WorkspaceStore(project.get_project_root(), "", {})

    for stage_name in execution_order:
        stage_info = all_stages[stage_name]
        stage_state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
        lock_data = stage_lock.read()
        output_hashes = lock_data["output_hashes"] if lock_data else {}

        for out in stage_info["outs"]:
            if out.tag is not types.ArtifactTag.INCREMENTAL:
                continue
            out_path = workspace.resolve_display_path(out)
            if out_path.exists() and out.identity not in output_hashes:
                uncached.append((stage_name, str(out_path)))

    return uncached
