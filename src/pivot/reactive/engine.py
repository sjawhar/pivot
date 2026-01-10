from __future__ import annotations

import collections
import contextlib
import importlib
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

from pivot import dag, executor, project, registry
from pivot.pipeline import yaml as pipeline_yaml

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable, Iterator

    import networkx as nx

    from pivot.registry import RegistryStageInfo
    from pivot.types import TuiMessage

logger = logging.getLogger(__name__)


_MAX_PENDING_CHANGES = 10000  # Threshold for "full rebuild" sentinel
_FULL_REBUILD_SENTINEL = pathlib.Path("__PIVOT_FULL_REBUILD__")

# File patterns that trigger code reload (worker restart + cache invalidation)
_CODE_FILE_SUFFIXES = (".py",)
_CONFIG_FILE_NAMES = ("pivot.yaml", "pivot.yml", "pipeline.py", "params.yaml", "params.yml")


class ReactiveEngine:
    """Reactive execution engine with file watching and automatic re-execution."""

    _stages: list[str] | None
    _single_stage: bool
    _cache_dir: pathlib.Path | None
    _max_workers: int | None
    _debounce_ms: int
    _change_queue: queue.Queue[set[pathlib.Path]]
    _shutdown: threading.Event
    _tui_queue: mp.Queue[TuiMessage] | None
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
    ) -> None:
        if debounce_ms < 0:
            raise ValueError(f"debounce_ms must be non-negative, got {debounce_ms}")
        self._stages = list(stages) if stages is not None else None
        self._single_stage = single_stage
        self._cache_dir = cache_dir
        self._max_workers = max_workers
        self._debounce_ms = debounce_ms

        self._change_queue = queue.Queue(maxsize=100)
        self._shutdown = threading.Event()
        self._tui_queue = None
        self._watcher_thread = None
        self._cached_dag = None
        self._cached_file_index = None
        self._pipeline_errors = None

    def run(self, tui_queue: mp.Queue[TuiMessage] | None = None) -> None:
        """Start reactive engine with watcher and coordinator."""
        self._tui_queue = tui_queue

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
            self._send_message("Running initial pipeline...")
            try:
                self._execute_stages(self._stages)
            except Exception as e:
                self._send_message(f"Initial execution failed: {e}", is_error=True)
            self._send_message("Watching for changes...")

            # Run coordinator (blocks until shutdown)
            self._coordinator_loop()
        finally:
            # Ensure clean shutdown regardless of how we exit
            self._shutdown.set()
            self._watcher_thread.join(timeout=3.0)
            if self._watcher_thread.is_alive():
                logger.warning("Watcher thread did not exit cleanly")

    def shutdown(self) -> None:
        """Signal graceful shutdown."""
        self._shutdown.set()

    def _watch_loop(self, stages_to_run: list[str]) -> None:
        """Pure producer - monitors files, enqueues changes."""
        try:
            watch_paths = _collect_watch_paths(stages_to_run)
            watch_filter = _create_watch_filter(stages_to_run)
            pending: set[pathlib.Path] = set()

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
                    pending = set()
                except queue.Full:
                    pass  # Keep accumulating, will send next iteration
        except Exception as e:
            logger.critical(f"Watcher thread failed: {e}")
            self.shutdown()  # Signal coordinator to exit

    def _coordinator_loop(self) -> None:
        """Orchestrate execution waves based on file changes."""
        while not self._shutdown.is_set():
            changes = self._collect_and_debounce()
            if not changes:
                continue

            code_changed = _is_code_or_config_change(changes)
            if code_changed:
                self._send_message("Reloading code...")
                self._invalidate_caches()
                reload_ok = self._reload_registry()
                self._restart_worker_pool()

                if not reload_ok:
                    # Pipeline is invalid - show error banner and wait for fix
                    error_summary = "; ".join(self._pipeline_errors or [])
                    self._send_message(
                        f"Pipeline invalid - fix errors to continue: {error_summary}",
                        is_error=True,
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

            self._send_message(f"Running {len(affected)} affected stage(s)...")

            try:
                self._execute_stages(affected)
            except Exception as e:
                self._send_message(f"Execution failed: {e}", is_error=True)

            self._send_message("Watching for changes...")

    def _collect_and_debounce(self, max_wait_s: float = 5.0) -> set[pathlib.Path]:
        """Collect changes with quiet period, max wait prevents infinite block."""
        if max_wait_s <= 0:
            raise ValueError(f"max_wait_s must be positive, got {max_wait_s}")
        changes: set[pathlib.Path] = set()
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
        pipeline_yaml = _find_pipeline_file(root)
        if pipeline_yaml is not None:
            return self._reload_from_pipeline_file(pipeline_yaml, old_stages)

        # Check for pipeline.py-based registration
        pipeline_py = root / "pipeline.py"
        if pipeline_py.exists():
            return self._reload_from_pipeline_py(pipeline_py, old_stages)

        return self._reload_from_decorators(old_stages)

    def _reload_from_pipeline_file(
        self, pipeline_file: pathlib.Path, old_stages: dict[str, RegistryStageInfo]
    ) -> bool:
        """Reload registry from pivot.yaml file."""
        # Clear the registry
        registry.REGISTRY.clear()

        try:
            # Reload stage modules first so functions have fresh code
            stage_modules = _collect_stage_modules(old_stages)
            for module_name in stage_modules:
                if module_name in sys.modules:
                    try:
                        importlib.reload(sys.modules[module_name])
                        logger.debug(f"Reloaded module: {module_name}")
                    except Exception as e:
                        logger.warning(f"Failed to reload module {module_name}: {e}")

            # Re-register stages from pipeline file
            pipeline_yaml.register_from_pipeline_file(pipeline_file)

            self._pipeline_errors = None
            new_stages = list(registry.REGISTRY.list_stages())
            logger.info(
                f"Registry reloaded from {pipeline_file.name} with {len(new_stages)} stages"
            )
            return True

        except Exception as e:
            registry.REGISTRY.restore(old_stages)
            self._pipeline_errors = [str(e)]
            logger.warning(f"Pipeline invalid: {e}")
            return False

    def _reload_from_pipeline_py(
        self, pipeline_py: pathlib.Path, old_stages: dict[str, RegistryStageInfo]
    ) -> bool:
        """Reload registry from pipeline.py file."""
        # Clear the registry
        registry.REGISTRY.clear()

        try:
            # Reload stage modules first so functions have fresh code
            stage_modules = _collect_stage_modules(old_stages)
            for module_name in stage_modules:
                if module_name in sys.modules:
                    try:
                        importlib.reload(sys.modules[module_name])
                        logger.debug(f"Reloaded module: {module_name}")
                    except Exception as e:
                        logger.warning(f"Failed to reload module {module_name}: {e}")

            # Re-run pipeline.py to re-register stages
            runpy.run_path(str(pipeline_py), run_name="_pivot_pipeline")

            self._pipeline_errors = None
            new_stages = list(registry.REGISTRY.list_stages())
            logger.info(f"Registry reloaded from pipeline.py with {len(new_stages)} stages")
            return True

        except Exception as e:
            registry.REGISTRY.restore(old_stages)
            self._pipeline_errors = [str(e)]
            logger.warning(f"Pipeline invalid: {e}")
            return False

    def _reload_from_decorators(self, old_stages: dict[str, RegistryStageInfo]) -> bool:
        """Reload registry by reimporting modules with @stage decorators."""
        stage_modules = _collect_stage_modules(old_stages)

        if not stage_modules:
            logger.warning("No stage modules found to reload")
            return True

        # Clear the registry
        registry.REGISTRY.clear()

        # Re-import each module (this re-runs @stage decorators)
        errors: list[str] = []
        for module_name in stage_modules:
            try:
                module = sys.modules[module_name]
                importlib.reload(module)
                logger.debug(f"Reloaded module: {module_name}")
            except Exception as e:
                errors.append(f"{module_name}: {e}")
                logger.error(f"Failed to reload module {module_name}: {e}")

        if errors:
            registry.REGISTRY.restore(old_stages)
            self._pipeline_errors = errors
            logger.warning(f"Pipeline invalid, keeping previous registry ({len(errors)} error(s))")
            return False

        # Success - clear any previous errors
        self._pipeline_errors = None
        new_stages = list(registry.REGISTRY.list_stages())
        logger.info(f"Registry reloaded with {len(new_stages)} stages: {new_stages}")
        return True

    def _get_affected_stages(self, changes: set[pathlib.Path], *, code_changed: bool) -> list[str]:
        """Determine which stages need to run based on changes."""
        if code_changed:
            # Code/config changed - run all stages, let executor's change detection
            # skip stages that don't actually need to run
            if self._stages is not None:
                return list(self._stages)  # Return copy to prevent mutation
            return list(registry.REGISTRY.list_stages())

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

    def _execute_stages(self, stages: list[str] | None) -> None:
        """Execute stages using the executor."""
        executor.run(
            stages=stages,
            single_stage=self._single_stage,
            cache_dir=self._cache_dir,
            max_workers=self._max_workers,
            show_output=self._tui_queue is None,
            tui_queue=self._tui_queue,
        )

    def _send_message(self, message: str, *, is_error: bool = False) -> None:
        """Send message to TUI or log."""
        if self._tui_queue is not None:
            from pivot.types import TuiReactiveMessage

            with contextlib.suppress(queue.Full):
                self._tui_queue.put_nowait(
                    TuiReactiveMessage(
                        type="reactive",
                        status="error" if is_error else "waiting",
                        message=message,
                    )
                )

        if is_error:
            logger.error(message)
        else:
            logger.info(message)


def _iter_stage_infos(stages: list[str]) -> Iterator[tuple[str, RegistryStageInfo]]:
    """Yield (name, info) pairs for valid stages, logging warnings for missing."""
    for name in stages:
        try:
            yield name, registry.REGISTRY.get(name)
        except KeyError:
            logger.warning(f"Stage '{name}' not found in registry, skipping")


def _collect_watch_paths(stages: list[str]) -> list[pathlib.Path]:
    """Collect paths: project root + dependency directories for specified stages."""
    root = project.get_project_root()
    paths: set[pathlib.Path] = {root}

    for _, info in _iter_stage_infos(stages):
        for dep in info["deps"]:
            dep_path = project.resolve_path(dep)
            if dep_path.exists():
                paths.add(dep_path.parent if dep_path.is_file() else dep_path)

    return list(paths)


def _get_output_paths_for_stages(stages: list[str]) -> set[str]:
    """Get output paths for specific stages only."""
    result: set[str] = set()

    for _, info in _iter_stage_infos(stages):
        for out_path in info["outs_paths"]:
            result.add(str(out_path))

    return result


def _create_watch_filter(
    stages_to_run: list[str],
) -> Callable[[watchfiles.Change, str], bool]:
    """Create filter excluding outputs from stages being run (prevents infinite loops)."""
    outputs_to_filter = {
        project.resolve_path(p) for p in _get_output_paths_for_stages(stages_to_run)
    }

    def watch_filter(change: watchfiles.Change, path: str) -> bool:
        del change  # Unused but required by watchfiles signature

        # Always filter Python bytecode
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
            return False

        # Resolve incoming path for consistent comparison (handles symlinks)
        try:
            resolved_path = project.resolve_path(path)
        except OSError:
            # Can't resolve (symlink loop, permission denied, etc.) - don't filter
            return True

        # Check if path is an output of a stage being run, or inside such an output directory
        for out in outputs_to_filter:
            if resolved_path == out or out in resolved_path.parents:
                return False

        return True

    return watch_filter


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
            except Exception:
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
