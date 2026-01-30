from __future__ import annotations

import collections
import contextlib
import importlib
import json
import linecache
import logging
import os
import pathlib
import queue
import runpy
import sys
import threading
import time
from typing import TYPE_CHECKING

import watchfiles
import yaml

from pivot import dag, executor, ignore, project, registry, types
from pivot.pipeline import yaml as pipeline_yaml
from pivot.types import (
    AgentCancelResult,
    AgentRunRejection,
    AgentRunStartResult,
    AgentState,
    AgentStatusResult,
    OnError,
    OutputMessage,
    WatchAffectedStagesEvent,
    WatchEventType,
    WatchExecutionResultEvent,
    WatchFilesChangedEvent,
    WatchStageResult,
    WatchStatusEvent,
)
from pivot.watch import _watch_utils

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable

    import networkx as nx

    from pivot.registry import RegistryStageInfo
    from pivot.types import TuiQueue, WatchJsonEvent

logger = logging.getLogger(__name__)


_MAX_PENDING_CHANGES = 10000  # Threshold for "full rebuild" sentinel
_FULL_REBUILD_SENTINEL = pathlib.Path("__PIVOT_FULL_REBUILD__")

# File patterns that trigger code reload (worker restart + cache invalidation)
_CODE_FILE_SUFFIXES = (".py",)
_CONFIG_FILE_NAMES = (
    "pivot.yaml",
    "pivot.yml",
    "pipeline.py",
    "params.yaml",
    "params.yml",
    ".pivotignore",
)


class CombinedEvent:
    """Event wrapper that is set when any component event is set.

    Used to make watchfiles.watch() exit on either shutdown or restart signals.
    """

    _events: tuple[threading.Event, ...]

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)


def _clear_project_modules(root: pathlib.Path) -> int:
    """Remove all project modules from sys.modules and their bytecode caches.

    This ensures transitive dependencies are also reimported, not just stage modules.
    Without this, if stages.py imports helpers.py, reloading stages.py still uses
    the old cached helpers.py because importlib.reload() doesn't reimport dependencies.

    Also removes .pyc bytecode cache files to prevent Python from loading stale
    compiled bytecode instead of re-parsing the modified source files.

    Returns the count of cleared modules.
    """
    root_str = str(root)
    to_remove = list[str]()
    pyc_files = list[pathlib.Path]()

    # Copy to list to avoid RuntimeError if another thread imports during iteration
    for name, module in list(sys.modules.items()):
        # sys.modules values can be None for failed imports (Python docs: "A key can map
        # to None if the module is found to not exist") - type stubs don't reflect this
        if module is None:  # pyright: ignore[reportUnnecessaryComparison]
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            if module_file.startswith(root_str):
                to_remove.append(name)
                # Track corresponding .pyc file for removal
                pyc_path = _get_pyc_path(module_file)
                if pyc_path is not None:
                    pyc_files.append(pyc_path)
        except (TypeError, AttributeError):
            continue

    for name in to_remove:
        del sys.modules[name]
        logger.debug(f"Cleared module from cache: {name}")

    # Remove bytecode cache files to force re-parsing of source
    for pyc_path in pyc_files:
        try:
            pyc_path.unlink(missing_ok=True)
            logger.debug(f"Removed bytecode cache: {pyc_path}")
        except OSError:
            pass  # Best effort - file may be locked or already removed

    # Invalidate import machinery caches after removing modules and .pyc files
    importlib.invalidate_caches()

    return len(to_remove)


def _get_pyc_path(source_path: str) -> pathlib.Path | None:
    """Get the __pycache__/*.pyc path for a source file, or None if not determinable."""
    try:
        source = pathlib.Path(source_path)
        if source.suffix != ".py":
            return None
        # Python stores bytecode in __pycache__/<name>.cpython-<version>.pyc
        cache_dir = source.parent / "__pycache__"
        # Match any version tag - we want to remove all cached versions
        # The glob pattern matches e.g. "helpers.cpython-313.pyc"
        stem = source.stem
        for pyc in cache_dir.glob(f"{stem}.*.pyc"):
            return pyc  # Return first match - typically only one exists
        return None
    except Exception:
        return None


class WatchEngine:
    """Watch mode execution engine with file watching and automatic re-execution."""

    _stages: list[str] | None
    _single_stage: bool
    _cache_dir: pathlib.Path | None
    _max_workers: int | None
    _debounce_ms: int
    _force_first_run: bool
    _first_run_done: bool
    _json_output: bool
    _no_commit: bool
    _no_cache: bool
    _change_queue: queue.Queue[set[pathlib.Path]]
    _shutdown: threading.Event
    _tui_queue: TuiQueue | None
    _output_queue: mp.Queue[OutputMessage] | None
    _watcher_thread: threading.Thread | None
    _cached_dag: nx.DiGraph[str] | None
    _cached_file_index: dict[pathlib.Path, set[str]] | None
    _pipeline_errors: list[str] | None
    _ignore_filter: ignore.IgnoreFilter
    _keep_going_event: threading.Event
    _toggle_lock: threading.Lock
    _output_filter: _watch_utils.OutputFilter
    _cancel_event: threading.Event
    _restart_event: threading.Event
    _current_watch_paths: list[pathlib.Path]

    # Agent RPC state
    _agent_state: AgentState
    _agent_run_id: str | None
    _agent_stages_completed: list[str]
    _agent_stages_pending: list[str]
    _agent_last_ran: int
    _agent_last_skipped: int
    _agent_last_failed: int
    _agent_request_queue: queue.Queue[tuple[str, list[str] | None, bool]]
    _agent_lock: threading.Lock

    def __init__(
        self,
        stages: list[str] | None = None,
        single_stage: bool = False,
        cache_dir: pathlib.Path | None = None,
        max_workers: int | None = None,
        debounce_ms: int = 300,
        force_first_run: bool = False,
        json_output: bool = False,
        no_commit: bool = False,
        no_cache: bool = False,
        on_error: OnError = OnError.FAIL,
    ) -> None:
        if debounce_ms < 0:
            raise ValueError(f"debounce_ms must be non-negative, got {debounce_ms}")
        self._stages = list(stages) if stages is not None else None
        self._single_stage = single_stage
        self._cache_dir = cache_dir
        self._max_workers = max_workers
        self._debounce_ms = debounce_ms
        self._force_first_run = force_first_run
        self._first_run_done = False
        self._json_output = json_output
        self._no_commit = no_commit
        self._no_cache = no_cache

        self._change_queue = queue.Queue(maxsize=100)
        self._shutdown = threading.Event()
        self._tui_queue = None
        self._output_queue = None
        self._watcher_thread = None
        self._cached_dag = None
        self._cached_file_index = None
        self._pipeline_errors = None
        self._ignore_filter = ignore.IgnoreFilter(project_root=project.get_project_root())
        self._keep_going_event = threading.Event()
        self._toggle_lock = threading.Lock()
        # Initialized with empty stages; updated in run() with actual stages
        self._output_filter = _watch_utils.OutputFilter([])
        self._cancel_event = threading.Event()
        self._restart_event = threading.Event()
        self._current_watch_paths = list[pathlib.Path]()
        if on_error == OnError.KEEP_GOING:
            self._keep_going_event.set()

        # Agent RPC state initialization
        self._agent_state = AgentState.IDLE
        self._agent_run_id = None
        self._agent_stages_completed = list[str]()
        self._agent_stages_pending = list[str]()
        self._agent_last_ran = 0
        self._agent_last_skipped = 0
        self._agent_last_failed = 0
        self._agent_request_queue = queue.Queue(maxsize=10)
        self._agent_lock = threading.Lock()

    def run(
        self,
        tui_queue: TuiQueue | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
    ) -> None:
        """Start watch engine with watcher and coordinator."""
        self._tui_queue = tui_queue
        self._output_queue = output_queue

        # Build DAG and get execution order for determining watch scope
        graph = registry.REGISTRY.build_dag(validate=True)
        stages_to_run = dag.get_execution_order(
            graph, self._stages, single_stage=self._single_stage
        )

        # Update output filter BEFORE starting watcher thread (fixes race condition)
        # The filter combines output paths with execution state for atomic filtering
        self._output_filter.update_outputs(stages_to_run)

        # Collect watch paths and store for later comparison during restarts
        self._current_watch_paths = _watch_utils.collect_watch_paths(stages_to_run)

        # Start watcher thread (non-daemon - we have proper cleanup via _shutdown event)
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(list(self._current_watch_paths),),
        )
        self._watcher_thread.start()

        try:
            # Run initial execution
            self._send_message("Running initial pipeline...", status=types.WatchStatus.DETECTING)
            try:
                results = self._execute_stages(self._stages)
                if self._json_output and results:
                    self._emit_json(
                        WatchExecutionResultEvent(
                            type=WatchEventType.EXECUTION_RESULT,
                            stages={
                                name: WatchStageResult(
                                    status=result["status"], reason=result["reason"]
                                )
                                for name, result in results.items()
                            },
                        )
                    )
            except Exception as e:
                self._send_message(f"Initial execution failed: {e}", status=types.WatchStatus.ERROR)
            self._send_message("Watching for changes...")

            # Run coordinator (blocks until shutdown)
            self._coordinator_loop()
        finally:
            # Ensure clean shutdown regardless of how we exit
            # Note: This handles all normal exit paths including KeyboardInterrupt and SystemExit.
            # Edge cases not covered: os._exit(), SIGKILL, segfaults - but nothing can help those.
            self._shutdown.set()
            self._watcher_thread.join(timeout=3.0)
            if self._watcher_thread.is_alive():
                logger.warning(
                    "Watcher thread did not exit within 3s timeout. This may indicate "
                    + "watchfiles.watch() is blocked on I/O. The thread will be abandoned "
                    + "(non-daemon by design for clean shutdown)."
                )

            # Send shutdown sentinel to TUI queue
            # Note: tui_queue is stdlib queue.Queue, so put() doesn't raise OSError/ValueError
            if self._tui_queue is not None:
                self._tui_queue.put(None)

    def shutdown(self) -> None:
        """Signal graceful shutdown."""
        self._shutdown.set()

    @property
    def keep_going(self) -> bool:
        """Return whether keep-going mode is enabled."""
        return self._keep_going_event.is_set()

    def toggle_keep_going(self) -> bool:
        """Toggle keep-going mode. Returns new state (True=enabled)."""
        with self._toggle_lock:  # Prevent race on rapid toggling
            if self._keep_going_event.is_set():
                self._keep_going_event.clear()
                return False
            self._keep_going_event.set()
            return True

    # =========================================================================
    # Agent RPC Methods
    # =========================================================================

    def try_start_agent_run(
        self, run_id: str, stages: list[str] | None, force: bool
    ) -> AgentRunStartResult | AgentRunRejection:
        """Atomically try to start an agent run.

        Returns AgentRunStartResult if started, AgentRunRejection if rejected.
        Thread-safe - called from asyncio thread via run_in_executor.

        This method performs an atomic check-and-set to prevent race conditions:
        - Checks if state is WATCHING (ready to accept runs)
        - Sets state to RUNNING and queues work in a single lock acquisition
        """
        # Compute stages outside lock for O(1) lock hold time
        stages_to_queue = stages if stages else list(registry.REGISTRY.list_stages())

        with self._agent_lock:
            # Only accept when actively watching (not IDLE, RUNNING, COMPLETED, FAILED)
            if self._agent_state != AgentState.WATCHING:
                return AgentRunRejection(
                    reason="not_ready",
                    current_state=self._agent_state.value,
                    current_run_id=self._agent_run_id,
                )

            # Atomic state transition
            self._agent_state = AgentState.RUNNING
            self._agent_run_id = run_id
            self._agent_stages_pending = list(stages_to_queue)
            self._agent_stages_completed = []

            try:
                self._agent_request_queue.put_nowait((run_id, stages, force))
            except queue.Full:
                # Roll back state to prevent deadlock (critical fix from design review)
                self._agent_state = AgentState.WATCHING
                self._agent_run_id = None
                self._agent_stages_pending = []
                return AgentRunRejection(reason="queue_full", current_state="watching")

            return AgentRunStartResult(
                run_id=run_id,
                status="started",
                stages_queued=stages_to_queue,
            )

    def get_agent_status(self, run_id: str | None = None) -> AgentStatusResult:
        """Get current agent execution status.

        Called from asyncio event loop thread, must be thread-safe.
        """
        with self._agent_lock:
            result = AgentStatusResult(state=self._agent_state)

            if run_id is not None and run_id != self._agent_run_id:
                # Requested specific run that's not current
                return result

            if self._agent_run_id is not None:
                result["run_id"] = self._agent_run_id

            if self._agent_stages_completed:
                result["stages_completed"] = list(self._agent_stages_completed)

            if self._agent_stages_pending:
                result["stages_pending"] = list(self._agent_stages_pending)

            # Include stats for completed/failed/watching states (watching includes last run stats)
            # Only include if there was a previous run (at least one stat is non-zero)
            has_stats = self._agent_last_ran or self._agent_last_skipped or self._agent_last_failed
            if (
                self._agent_state in (AgentState.COMPLETED, AgentState.FAILED, AgentState.WATCHING)
                and has_stats
            ):
                result["ran"] = self._agent_last_ran
                result["skipped"] = self._agent_last_skipped
                result["failed"] = self._agent_last_failed

            return result

    def request_agent_cancel(self) -> AgentCancelResult:
        """Request cancellation of current agent execution.

        Called from asyncio event loop thread, must be thread-safe.

        Cancellation is stage-level: running stages complete normally, but no new
        stages are started. Pending stages are marked as skipped with reason "cancelled".
        """
        with self._agent_lock:
            if self._agent_state == AgentState.RUNNING:
                self._cancel_event.set()
                logger.info("Agent cancellation requested - pending stages will be skipped")
                return AgentCancelResult(cancelled=True)
            return AgentCancelResult(cancelled=False)

    def _check_agent_requests(self) -> tuple[str, list[str] | None, bool] | None:
        """Check for pending agent execution requests. Returns (run_id, stages, force) or None."""
        try:
            return self._agent_request_queue.get_nowait()
        except queue.Empty:
            return None

    def _update_agent_state(
        self,
        state: AgentState,
        *,
        run_id: str | None = None,
        stages_completed: list[str] | None = None,
        stages_pending: list[str] | None = None,
        ran: int = 0,
        skipped: int = 0,
        failed: int = 0,
    ) -> None:
        """Update agent state (thread-safe)."""
        with self._agent_lock:
            self._agent_state = state
            if run_id is not None:
                self._agent_run_id = run_id
            if stages_completed is not None:
                self._agent_stages_completed = stages_completed
            if stages_pending is not None:
                self._agent_stages_pending = stages_pending
            if state in (AgentState.COMPLETED, AgentState.FAILED):
                self._agent_last_ran = ran
                self._agent_last_skipped = skipped
                self._agent_last_failed = failed
            # Clear run tracking when returning to WATCHING state
            if state == AgentState.WATCHING:
                self._agent_run_id = None
                self._agent_stages_pending = []

    def _handle_agent_request(self, request: tuple[str, list[str] | None, bool]) -> None:
        """Handle an execution request from the agent RPC server.

        Note: State is already set to RUNNING by try_start_agent_run() before this is called.
        """
        _run_id, stages, force = request  # run_id already set by try_start_agent_run
        stages_to_queue = stages or list(registry.REGISTRY.list_stages())

        self._send_message(
            f"Agent running {len(stages_to_queue)} stage(s)...",
            status=types.WatchStatus.DETECTING,
        )

        # Track execution progress for exception handling
        results = dict[str, executor.ExecutionSummary]()

        try:
            # Store original force settings and override if requested
            original_force = self._force_first_run
            original_first_run_done = self._first_run_done
            if force:
                self._force_first_run = True
                self._first_run_done = False

            try:
                results = self._execute_stages(stages)
            finally:
                # Restore original force settings
                self._force_first_run = original_force
                self._first_run_done = original_first_run_done

            # Calculate stats from results
            ran = sum(1 for r in results.values() if r["status"] == types.StageStatus.RAN)
            skipped = sum(1 for r in results.values() if r["status"] == types.StageStatus.SKIPPED)
            failed = sum(1 for r in results.values() if r["status"] == types.StageStatus.FAILED)

            final_state = AgentState.FAILED if failed > 0 else AgentState.COMPLETED
            self._update_agent_state(
                final_state,
                ran=ran,
                skipped=skipped,
                failed=failed,
                stages_completed=list(results.keys()),
                stages_pending=list[str](),
            )

        except Exception:
            logger.exception("Agent execution failed")
            # Capture partial progress from results collected before the exception
            ran = sum(1 for r in results.values() if r["status"] == types.StageStatus.RAN)
            skipped = sum(1 for r in results.values() if r["status"] == types.StageStatus.SKIPPED)
            failed = sum(1 for r in results.values() if r["status"] == types.StageStatus.FAILED)
            self._update_agent_state(
                AgentState.FAILED,
                ran=ran,
                skipped=skipped,
                failed=failed + 1,  # +1 for the exception itself
                stages_completed=list(results.keys()),
                stages_pending=list[str](),
            )

        # Transition back to WATCHING state after execution completes
        self._update_agent_state(AgentState.WATCHING)
        self._send_message("Watching for changes...")

    def _watch_loop(self, watch_paths: list[pathlib.Path]) -> None:
        """Pure producer - monitors files, enqueues changes."""
        try:
            # OutputFilter handles both output paths and execution state atomically
            watch_filter = _watch_utils.create_watch_filter(
                ignore_filter=self._ignore_filter,
                output_filter=self._output_filter,
            )
            pending = set[pathlib.Path]()

            # Combined event exits on shutdown OR restart request
            stop_event = CombinedEvent(self._shutdown, self._restart_event)

            logger.info(f"Watching paths: {watch_paths}")

            for changes in watchfiles.watch(
                *watch_paths,
                watch_filter=watch_filter,
                stop_event=stop_event,
            ):
                pending.update(pathlib.Path(path) for _, path in changes)

                # Prevent unbounded memory growth - use sentinel for "full rebuild"
                if len(pending) > _MAX_PENDING_CHANGES:
                    logger.warning(
                        f"Pending changes ({len(pending)}) exceeded threshold, signaling full rebuild"
                    )
                    pending = {_FULL_REBUILD_SENTINEL}

                try:
                    self._change_queue.put_nowait(pending)
                    pending = set[pathlib.Path]()
                except queue.Full:
                    pass  # Keep accumulating, will send next iteration
        except Exception as e:
            logger.critical(f"Watcher thread failed: {e}")
            self._send_message(f"File watcher failed: {e}", status=types.WatchStatus.ERROR)
            self.shutdown()  # Signal coordinator to exit

    def _restart_watcher_if_paths_changed(self, stages: list[str]) -> None:
        """Restart watcher thread if watch paths have changed after registry reload.

        Note: There's a brief window (typically <100ms) between the old watcher stopping
        and the new one starting where file changes could theoretically be missed. This is
        acceptable because: (1) code changes already triggered this restart, so the user
        just made a change, (2) the executor's lockfile-based change detection will catch
        any missed changes on the next execution cycle.
        """
        new_paths = _watch_utils.collect_watch_paths(stages)
        new_paths_set = set(new_paths)
        current_paths_set = set(self._current_watch_paths)

        if new_paths_set == current_paths_set:
            return  # No change

        added = new_paths_set - current_paths_set
        removed = current_paths_set - new_paths_set
        logger.info(f"Watch paths changed: +{len(added)} -{len(removed)} paths")
        logger.debug(f"Watch paths added: {added}")
        logger.debug(f"Watch paths removed: {removed}")

        # Signal the old watcher to stop (combined event will trigger)
        self._restart_event.set()

        # Wait for watcher thread to exit
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=3.0)
            if self._watcher_thread.is_alive():
                logger.warning("Watcher thread did not exit within timeout during restart")

        # Create a fresh restart event for the new watcher thread.
        # Don't clear the old event - if the old thread is still running, it needs
        # to see the stop signal. The new thread gets a fresh, unset event.
        self._restart_event = threading.Event()
        self._current_watch_paths = list(new_paths)

        # Start new watcher thread with updated paths
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(list(self._current_watch_paths),),
        )
        self._watcher_thread.start()
        logger.info(f"Watcher restarted with {len(new_paths)} paths")

    def _coordinator_loop(self) -> None:
        """Orchestrate execution waves based on file changes and agent requests."""
        # Update agent state to watching once coordinator starts
        self._update_agent_state(AgentState.WATCHING)

        while not self._shutdown.is_set():
            # Check for agent execution requests (higher priority than file changes)
            agent_request = self._check_agent_requests()
            if agent_request is not None:
                self._handle_agent_request(agent_request)
                continue

            changes = self._collect_and_debounce()
            if not changes:
                continue

            code_changed = _is_code_or_config_change(changes)

            # Emit files_changed event for JSON output
            if self._json_output:
                # Filter out sentinel path and convert to strings
                paths = [str(p) for p in changes if p != _FULL_REBUILD_SENTINEL]
                self._emit_json(
                    WatchFilesChangedEvent(
                        type=WatchEventType.FILES_CHANGED,
                        paths=paths,
                        code_changed=code_changed,
                    )
                )

            if code_changed:
                self._send_message("Reloading code...", status=types.WatchStatus.RESTARTING)
                self._invalidate_caches()
                reload_ok = self._reload_registry()
                self._restart_worker_pool()

                if reload_ok:
                    # Atomically update outputs in filter - safe even if watcher is iterating
                    stages = list(self._stages or registry.REGISTRY.list_stages())
                    self._output_filter.update_outputs(stages)
                    # Restart watcher if new stages have dependencies outside current watch paths
                    self._restart_watcher_if_paths_changed(stages)

                if not reload_ok:
                    # Pipeline is invalid - show error banner and wait for fix
                    error_summary = "; ".join(self._pipeline_errors or [])
                    self._send_message(
                        f"Pipeline invalid - fix errors to continue: {error_summary}",
                        status=types.WatchStatus.ERROR,
                    )
                    self._send_message("Watching for changes...")
                    continue

            # Skip execution if pipeline is still invalid from a previous reload
            if self._pipeline_errors:
                self._send_message("Watching for changes...")
                continue

            affected = self._get_affected_stages(changes, code_changed=code_changed)
            if not affected:
                self._send_message("Watching for changes...")
                continue

            # Emit affected_stages event for JSON output
            if self._json_output:
                self._emit_json(
                    WatchAffectedStagesEvent(
                        type=WatchEventType.AFFECTED_STAGES,
                        stages=affected,
                        count=len(affected),
                    )
                )

            self._send_message(
                f"Running {len(affected)} affected stage(s)...",
                status=types.WatchStatus.DETECTING,
            )

            try:
                results = self._execute_stages(affected)
                # Emit execution results for JSON output
                if self._json_output and results:
                    self._emit_json(
                        WatchExecutionResultEvent(
                            type=WatchEventType.EXECUTION_RESULT,
                            stages={
                                name: WatchStageResult(
                                    status=result["status"], reason=result["reason"]
                                )
                                for name, result in results.items()
                            },
                        )
                    )
            except Exception as e:
                self._send_message(f"Execution failed: {e}", status=types.WatchStatus.ERROR)
            self._send_message("Watching for changes...")

    def _collect_and_debounce(self, max_wait_s: float = 5.0) -> set[pathlib.Path]:
        """Collect changes with quiet period, max wait prevents infinite block."""
        if max_wait_s <= 0:
            raise ValueError(f"max_wait_s must be positive, got {max_wait_s}")
        changes = set[pathlib.Path]()
        deadline = time.monotonic() + max_wait_s
        quiet_period_s = self._debounce_ms / 1000
        last_change = time.monotonic()

        while time.monotonic() < deadline:
            # Check shutdown between queue waits
            if self._shutdown.is_set():
                return set()

            try:
                batch = self._change_queue.get(timeout=0.1)
                changes.update(batch)
                last_change = time.monotonic()
            except queue.Empty:
                if changes and (time.monotonic() - last_change) >= quiet_period_s:
                    return changes

        return changes

    def _invalidate_caches(self) -> None:
        """Invalidate all caches atomically. Call when code/config changes."""
        linecache.clearcache()
        importlib.invalidate_caches()
        self._ignore_filter.invalidate()
        # Atomic replacement - build new caches only when needed via lazy getters
        self._cached_dag = None
        self._cached_file_index = None
        # Also invalidate registry's cached DAG since code may have changed
        registry.REGISTRY.invalidate_dag_cache()

    def _reload_registry(self) -> bool:
        """Reload the registry by re-importing modules that define stages.

        Returns True if reload succeeded, False if pipeline is now invalid.
        On failure, the old registry is preserved.

        Supports three registration patterns:
        1. pivot.yaml-based: Re-runs register_from_pipeline_file()
        2. pipeline.py-based: Re-runs the script via runpy.run_path()
        3. Stage modules: Reloads the modules containing stage definitions
        """
        old_stages = registry.REGISTRY.snapshot()

        root = project.get_project_root()

        # Check for pivot.yaml-based registration
        pipeline_yaml_file = _find_pipeline_file(root)
        if pipeline_yaml_file is not None:
            return self._reload_from_pipeline_file(pipeline_yaml_file, old_stages)

        # Check for pipeline.py-based registration
        pipeline_py = root / "pipeline.py"
        if pipeline_py.exists():
            return self._reload_from_pipeline_py(pipeline_py, old_stages)

        return self._reload_from_decorators(old_stages)

    def _send_reload_notification(self) -> None:
        """Send TuiReloadMessage with current stage list."""
        if self._tui_queue is not None:
            new_stages = list(registry.REGISTRY.list_stages())
            msg = types.TuiReloadMessage(
                type=types.TuiMessageType.RELOAD,
                stages=new_stages,
            )
            with contextlib.suppress(queue.Full):
                self._tui_queue.put_nowait(msg)

    def _reload_with_registration(
        self,
        register_fn: Callable[[], object],
        source_name: str,
        old_stages: dict[str, RegistryStageInfo],
    ) -> bool:
        """Reload registry using provided registration function.

        Clears all project modules from sys.modules before registration to ensure
        transitive dependencies are properly reimported (not just stage modules).
        """
        registry.REGISTRY.clear()
        try:
            root = project.get_project_root()
            cleared = _clear_project_modules(root)
            logger.debug(f"Cleared {cleared} project modules from cache")
            register_fn()
            self._pipeline_errors = None
            new_stages = list(registry.REGISTRY.list_stages())
            logger.info(f"Registry reloaded from {source_name} with {len(new_stages)} stages")
            self._send_reload_notification()
            return True
        except Exception as e:
            registry.REGISTRY.restore(old_stages)
            self._pipeline_errors = [str(e)]
            logger.warning(f"Pipeline invalid: {e}")
            return False

    def _reload_from_pipeline_file(
        self, pipeline_file: pathlib.Path, old_stages: dict[str, RegistryStageInfo]
    ) -> bool:
        """Reload registry from pivot.yaml file."""
        return self._reload_with_registration(
            lambda: pipeline_yaml.register_from_pipeline_file(pipeline_file),
            pipeline_file.name,
            old_stages,
        )

    def _reload_from_pipeline_py(
        self, pipeline_py: pathlib.Path, old_stages: dict[str, RegistryStageInfo]
    ) -> bool:
        """Reload registry from pipeline.py file."""
        return self._reload_with_registration(
            lambda: runpy.run_path(str(pipeline_py), run_name="_pivot_pipeline"),
            "pipeline.py",
            old_stages,
        )

    def _reload_from_decorators(self, old_stages: dict[str, RegistryStageInfo]) -> bool:
        """Reload registry by reimporting stage modules.

        Clears ALL project modules from sys.modules and reimports stage modules.
        Using import_module (not reload) after clearing ensures transitive
        dependencies are also freshly imported.
        """
        stage_modules = _collect_stage_modules(old_stages)
        if not stage_modules:
            logger.warning("No stage modules found to reload")
            return True

        registry.REGISTRY.clear()
        root = project.get_project_root()
        cleared = _clear_project_modules(root)
        logger.debug(f"Cleared {cleared} project modules from cache")

        errors = list[str]()
        for module_name in stage_modules:
            try:
                importlib.import_module(module_name)
                logger.debug(f"Reimported module: {module_name}")
            except Exception as e:
                errors.append(f"{module_name}: {e}")
                logger.error(f"Failed to import module {module_name}: {e}")

        if errors:
            registry.REGISTRY.restore(old_stages)
            self._pipeline_errors = errors
            logger.warning(f"Pipeline invalid, keeping previous registry ({len(errors)} error(s))")
            return False

        self._pipeline_errors = None
        new_stages = list(registry.REGISTRY.list_stages())
        logger.info(f"Registry reloaded with {len(new_stages)} stages: {new_stages}")
        self._send_reload_notification()
        return True

    def _get_affected_stages(self, changes: set[pathlib.Path], *, code_changed: bool) -> list[str]:
        """Determine which stages need to run based on changes."""
        if code_changed:
            # Code/config changed - run all stages, let executor's change detection
            # skip stages that don't actually need to run
            return list(self._stages or registry.REGISTRY.list_stages())

        # Data file changed - find affected stages and their downstream
        affected = self._get_stages_matching_changes(changes)
        return list(self._add_downstream_stages(affected))

    def _get_stages_matching_changes(self, changes: set[pathlib.Path]) -> set[str]:
        """Find stages whose dependencies match changed files (exact or containment)."""
        affected: set[str] = set()
        file_index = self._get_file_index()

        # Pre-compute directory dependencies once for containment checks
        dir_deps = [(dep, stages) for dep, stages in file_index.items() if _is_existing_dir(dep)]

        for path in changes:
            resolved = _resolve_path_for_matching(path)

            # Direct match
            if resolved in file_index:
                affected.update(file_index[resolved])

            # Containment match (file inside a dependency directory)
            for dep_path, stages in dir_deps:
                try:
                    if resolved.is_relative_to(dep_path):
                        affected.update(stages)
                except ValueError:
                    # is_relative_to raises ValueError if paths aren't comparable
                    continue

        return affected

    def _get_file_index(self) -> dict[pathlib.Path, set[str]]:
        """Get cached file-to-stages index, building if needed."""
        if self._cached_file_index is None:
            self._cached_file_index = self._build_file_to_stages_index()
        return self._cached_file_index

    def _build_file_to_stages_index(self) -> dict[pathlib.Path, set[str]]:
        """Map file paths to stages that depend on them."""
        index: collections.defaultdict[pathlib.Path, set[str]] = collections.defaultdict(set)

        for stage_name in registry.REGISTRY.list_stages():
            info = registry.REGISTRY.get(stage_name)
            for dep in info["deps_paths"]:
                dep_path = project.resolve_path(dep)
                index[dep_path].add(stage_name)

        return dict(index)

    def _get_dag(self) -> nx.DiGraph[str]:
        """Get cached DAG, building if needed."""
        if self._cached_dag is None:
            self._cached_dag = registry.REGISTRY.build_dag(validate=True)
        return self._cached_dag

    def _add_downstream_stages(self, stages: set[str]) -> set[str]:
        """Add all stages downstream of the given stages."""
        graph = self._get_dag()
        graph_nodes = set(graph.nodes())
        all_affected: set[str] = set()

        for stage in stages:
            if stage not in graph_nodes:
                logger.warning(f"Stage '{stage}' not found in DAG, skipping")
                continue
            all_affected.add(stage)
            downstream = dag.get_downstream_stages(graph, stage)
            all_affected.update(downstream)

        return all_affected

    def _restart_worker_pool(self) -> None:
        """Kill existing workers, spawn fresh ones with reimported modules."""
        stages = self._stages or list(registry.REGISTRY.list_stages())
        workers = executor.restart_workers(len(stages) if stages else 1, self._max_workers)
        logger.info(f"Worker pool restarted with {workers} workers")

    def _execute_stages(self, stages: list[str] | None) -> dict[str, executor.ExecutionSummary]:
        """Execute stages using the executor."""
        force = self._force_first_run and not self._first_run_done
        # Read keep-going state at execution start (toggle takes effect on next wave)
        on_error = OnError.KEEP_GOING if self._keep_going_event.is_set() else OnError.FAIL

        # Clear cancel event before starting new execution
        self._cancel_event.clear()

        # Track execution window to distinguish Pivot outputs from external modifications
        # Context manager ensures end_execution() is called even on exceptions
        with self._output_filter.executing():
            results = executor.run(
                stages=stages,
                single_stage=self._single_stage,
                cache_dir=self._cache_dir,
                max_workers=self._max_workers,
                force=force,
                no_commit=self._no_commit,
                no_cache=self._no_cache,
                on_error=on_error,
                cancel_event=self._cancel_event,
            )

        self._first_run_done = True
        return results

    def _emit_json(self, event: WatchJsonEvent) -> None:
        """Emit a JSONL event to stdout."""
        print(json.dumps(event), flush=True)

    def _send_message(
        self,
        message: str,
        *,
        status: types.WatchStatus = types.WatchStatus.WAITING,
    ) -> None:
        """Send message to TUI, JSON output, or log."""
        if self._json_output:
            self._emit_json(
                WatchStatusEvent(
                    type=WatchEventType.STATUS,
                    message=message,
                    is_error=status == types.WatchStatus.ERROR,
                )
            )
            return

        if self._tui_queue is not None:
            msg = types.TuiWatchMessage(
                type=types.TuiMessageType.WATCH,
                status=status,
                message=message,
            )
            if status == types.WatchStatus.ERROR:
                # Block briefly for critical messages
                with contextlib.suppress(queue.Full):
                    self._tui_queue.put(msg, timeout=1.0)
            else:
                with contextlib.suppress(queue.Full):
                    self._tui_queue.put_nowait(msg)
            # Don't log when using TUI - messages go to the queue instead
            return

        if status == types.WatchStatus.ERROR:
            logger.error(message)
        else:
            logger.info(message)


def _is_code_or_config_change(changes: set[pathlib.Path]) -> bool:
    """Check if changes include code files, config files, or full rebuild sentinel."""
    if _FULL_REBUILD_SENTINEL in changes:
        return True
    return any(
        path.suffix in _CODE_FILE_SUFFIXES or path.name in _CONFIG_FILE_NAMES for path in changes
    )


def _resolve_path_for_matching(path: pathlib.Path) -> pathlib.Path:
    """Resolve path consistently for file index matching, handling deletions."""
    try:
        return project.resolve_path(path)
    except OSError:
        # File was deleted or inaccessible - use normalized absolute path
        # This allows matching against index entries for deleted files
        return pathlib.Path(os.path.normpath(path.absolute()))


def _is_existing_dir(path: pathlib.Path) -> bool:
    """Check if path is an existing directory, handling errors gracefully."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _find_pipeline_file(root: pathlib.Path) -> pathlib.Path | None:
    """Find pivot.yaml or pivot.yml with stages in project root.

    Only returns a path if the file defines stages (not just a project marker).
    """
    for name in ("pivot.yaml", "pivot.yml"):
        path = root / name
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                if isinstance(config, dict) and "stages" in config:
                    return path
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")
                continue
    return None


def _collect_stage_modules(stages: dict[str, RegistryStageInfo]) -> set[str]:
    """Collect module names from stage functions."""
    modules: set[str] = set()
    for info in stages.values():
        func = info["func"]
        module_name = getattr(func, "__module__", None)
        if module_name and module_name in sys.modules:
            modules.add(module_name)
    return modules
