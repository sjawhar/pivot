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

import loky
import watchfiles
import yaml

from pivot import dag, executor, project, registry, types
from pivot.pipeline import yaml as pipeline_yaml
from pivot.types import (
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
    from pivot.types import TuiMessage, WatchJsonEvent

logger = logging.getLogger(__name__)


_VENV_DIRS = frozenset({".venv", "venv", ".env", "env", "node_modules", ".git", "__pycache__"})


_MAX_PENDING_CHANGES = 10000  # Threshold for "full rebuild" sentinel
_FULL_REBUILD_SENTINEL = pathlib.Path("__PIVOT_FULL_REBUILD__")

# File patterns that trigger code reload (worker restart + cache invalidation)
_CODE_FILE_SUFFIXES = (".py",)
_CONFIG_FILE_NAMES = ("pivot.yaml", "pivot.yml", "pipeline.py", "params.yaml", "params.yml")


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
    _tui_queue: mp.Queue[TuiMessage] | None
    _output_queue: mp.Queue[OutputMessage] | None
    _watcher_thread: threading.Thread | None
    _cached_dag: nx.DiGraph[str] | None
    _cached_file_index: dict[pathlib.Path, set[str]] | None
    _pipeline_errors: list[str] | None

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

    def run(
        self,
        tui_queue: mp.Queue[TuiMessage] | None = None,
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

        # Start watcher thread (non-daemon - we have proper cleanup via _shutdown event)
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(stages_to_run,),
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
            if self._tui_queue is not None:
                with contextlib.suppress(OSError, ValueError):
                    self._tui_queue.put(None)

    def shutdown(self) -> None:
        """Signal graceful shutdown."""
        self._shutdown.set()

    def _watch_loop(self, stages_to_run: list[str]) -> None:
        """Pure producer - monitors files, enqueues changes."""
        try:
            watch_paths = _watch_utils.collect_watch_paths(stages_to_run)
            watch_filter = _watch_utils.create_watch_filter(stages_to_run)
            pending = set[pathlib.Path]()

            logger.info(f"Watching paths: {watch_paths}")

            for changes in watchfiles.watch(
                *watch_paths,
                watch_filter=watch_filter,
                stop_event=self._shutdown,
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

    def _coordinator_loop(self) -> None:
        """Orchestrate execution waves based on file changes."""
        while not self._shutdown.is_set():
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
        # Atomic replacement - build new caches only when needed via lazy getters
        self._cached_dag = None
        self._cached_file_index = None

    def _reload_registry(self) -> bool:
        """Reload the registry by re-importing modules that define stages.

        Returns True if reload succeeded, False if pipeline is now invalid.
        On failure, the old registry is preserved.

        Supports three registration patterns:
        1. pivot.yaml-based: Re-runs register_from_pipeline_file()
        2. pipeline.py-based: Re-runs the script via runpy.run_path()
        3. @stage decorators: Reloads the modules containing decorators
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
        """Reload registry by reimporting modules with @stage decorators.

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
            for dep in info["deps"]:
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
        max_workers = self._max_workers or os.cpu_count() or 1
        loky.get_reusable_executor(max_workers=max_workers, kill_workers=True)
        logger.info(f"Worker pool restarted with {max_workers} workers")

    def _execute_stages(self, stages: list[str] | None) -> dict[str, executor.ExecutionSummary]:
        """Execute stages using the executor."""
        force = self._force_first_run and not self._first_run_done
        # Suppress console output when JSON output is enabled
        show_output = self._tui_queue is None and not self._json_output
        results = executor.run(
            stages=stages,
            single_stage=self._single_stage,
            cache_dir=self._cache_dir,
            max_workers=self._max_workers,
            show_output=show_output,
            tui_queue=self._tui_queue,
            output_queue=self._output_queue,
            force=force,
            no_commit=self._no_commit,
            no_cache=self._no_cache,
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
        return project.resolve_path(str(path))
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
