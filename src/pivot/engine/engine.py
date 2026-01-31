"""Engine: the central coordinator for event-driven pipeline execution."""

from __future__ import annotations

import collections
import concurrent.futures
import importlib
import linecache
import logging
import multiprocessing as mp
import pathlib
import queue
import runpy
import sys
import threading
import time
from typing import TYPE_CHECKING, Self

import yaml

from pivot import config, dag, parameters, project, registry
from pivot.engine import graph as engine_graph
from pivot.engine.types import (
    CodeOrConfigChanged,
    DataArtifactChanged,
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    InputEvent,
    LogLine,
    NodeType,
    OutputEvent,
    PipelineReloaded,
    RunRequested,
    StageCompleted,
    StageExecutionState,
    StageStarted,
    StageStateChanged,
)
from pivot.executor import core as executor_core
from pivot.executor import worker
from pivot.storage import state as state_mod
from pivot.types import (
    AgentCancelResult,
    AgentRunRejection,
    AgentRunStartResult,
    AgentStatusResult,
    OnError,
    OutputMessage,
    RunEventType,
    StageCompleteEvent,
    StageResult,
    StageStartEvent,
    StageStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from pivot.executor.core import ExecutionSummary
    from pivot.registry import RegistryStageInfo
    from pivot.storage.cache import CheckoutMode
    from pivot.types import RunJsonEvent

__all__ = ["Engine"]

_logger = logging.getLogger(__name__)

# Timeout for draining output queue from worker processes (seconds).
# Short timeout allows responsive shutdown while still batching messages.
_OUTPUT_QUEUE_DRAIN_TIMEOUT = 0.02


class Engine:
    """Central coordinator for pipeline execution.

    The Engine is the single source of truth for execution state. All code paths
    (CLI, watch mode, agent RPC) go through the Engine.
    """

    def __init__(self) -> None:
        """Initialize the engine in IDLE state."""
        self._state: EngineState = EngineState.IDLE
        self._sinks: list[EventSink] = list[EventSink]()
        self._sources: list[EventSource] = list[EventSource]()
        self._event_queue: queue.Queue[InputEvent] = queue.Queue()
        self._shutdown_event: threading.Event = threading.Event()
        self._cancel_event: threading.Event = threading.Event()
        self._graph: nx.DiGraph[str] | None = None
        self._stage_indices: dict[str, tuple[int, int]] = dict[str, tuple[int, int]]()
        # Per-stage execution state tracking
        self._stage_states: dict[str, StageExecutionState] = dict[str, StageExecutionState]()
        self._stage_states_lock: threading.Lock = threading.Lock()
        # Deferred events for filtered paths (stage -> list of events)
        self._deferred_events: dict[str, list[InputEvent]] = dict[str, list[InputEvent]]()
        # Execution orchestration state
        self._futures: dict[concurrent.futures.Future[StageResult], str] = dict[
            concurrent.futures.Future[StageResult], str
        ]()
        self._mutex_counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        self._stage_upstream_unfinished: dict[str, set[str]] = dict[str, set[str]]()
        self._stage_downstream: dict[str, list[str]] = dict[str, list[str]]()
        self._stage_mutex: dict[str, list[str]] = dict[str, list[str]]()
        self._executor: concurrent.futures.Executor | None = None
        self._max_workers: int = 1
        self._error_mode: OnError = OnError.FAIL
        self._stop_starting_new: bool = False
        # Progress callback for backwards compatibility with JSONL output
        self._progress_callback: Callable[[RunJsonEvent], None] | None = None
        # Agent RPC state
        self._run_lock: threading.Lock = threading.Lock()
        self._current_run_id: str | None = None
        # Keep-going mode for watch (continue after failures)
        self._keep_going_event: threading.Event = threading.Event()
        self._toggle_lock: threading.Lock = threading.Lock()
        # Track warned mutex groups to avoid repeated warnings in watch mode
        self._warned_mutex_groups: set[str] = set[str]()

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        return self._state

    @property
    def sinks(self) -> list[EventSink]:
        """Registered event sinks (returns a copy)."""
        return list(self._sinks)

    @property
    def sources(self) -> list[EventSource]:
        """Registered event sources (returns a copy)."""
        return list(self._sources)

    @property
    def cancel_event(self) -> threading.Event:
        """Cancel event for stopping execution.

        Can be passed to executor to enable cancellation.
        """
        return self._cancel_event

    @property
    def keep_going(self) -> bool:
        """Return whether keep-going mode is enabled."""
        return self._keep_going_event.is_set()

    def toggle_keep_going(self) -> bool:
        """Toggle keep-going mode. Returns new state (True=enabled)."""
        with self._toggle_lock:
            if self._keep_going_event.is_set():
                self._keep_going_event.clear()
                return False
            self._keep_going_event.set()
            return True

    def set_keep_going(self, enabled: bool) -> None:
        """Set keep-going mode."""
        if enabled:
            self._keep_going_event.set()
        else:
            self._keep_going_event.clear()

    def set_cancel_event(self, event: threading.Event) -> None:
        """Replace the cancel event for external integration.

        Used by TUI mode to share its cancel event with the Engine,
        allowing the TUI to signal cancellation.
        """
        self._cancel_event = event

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager, closing all sinks."""
        self.close()

    @property
    def graph(self) -> nx.DiGraph[str] | None:
        """Current bipartite artifact-stage graph.

        Returns None until the graph is built (typically on first run or
        after registry reload). Status/verify code can query this graph
        to understand artifact-stage relationships.
        """
        return self._graph

    def _set_stage_state(self, stage: str, new_state: StageExecutionState) -> None:
        """Update stage execution state and emit event.

        Emit is done inside the lock to ensure events are delivered in order.
        """
        with self._stage_states_lock:
            old_state = self._stage_states.get(stage, StageExecutionState.PENDING)
            # Always add the stage to the dict (even if state unchanged) so it's tracked
            # Skip emit only if state is unchanged AND stage was already in dict
            is_new = stage not in self._stage_states
            self._stage_states[stage] = new_state
            if not is_new and old_state == new_state:
                return

            # Emit inside lock to guarantee event ordering
            self.emit(
                StageStateChanged(
                    type="stage_state_changed",
                    stage=stage,
                    state=new_state,
                    previous_state=old_state,
                )
            )

    def get_stage_state(self, stage: str) -> StageExecutionState:
        """Get current execution state for a stage. Thread-safe."""
        with self._stage_states_lock:
            return self._stage_states.get(stage, StageExecutionState.PENDING)

    def get_executing_stages(self) -> list[str]:
        """Get stages currently in PREPARING or RUNNING state. Thread-safe.

        Uses IntEnum ordering: PREPARING=3, RUNNING=4, COMPLETED=5
        """
        with self._stage_states_lock:
            return [
                name
                for name, state in self._stage_states.items()
                if StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED
            ]

    def add_sink(self, sink: EventSink) -> None:
        """Register an event sink to receive output events."""
        self._sinks.append(sink)

    def add_source(self, source: EventSource) -> None:
        """Register an event source to produce input events."""
        self._sources.append(source)

    def submit(self, event: InputEvent) -> None:
        """Submit an event for processing. Thread-safe.

        Events are queued and processed by run_loop().

        Args:
            event: Input event to process.
        """
        self._event_queue.put(event)

    def emit(self, event: OutputEvent) -> None:
        """Emit an event to all registered sinks.

        Exceptions from individual sinks are logged but do not prevent
        other sinks from receiving the event.

        Also forwards stage events to progress_callback if set (for backwards
        compatibility with callers that pass progress_callback to run_once).
        """
        for sink in self._sinks:
            try:
                sink.handle(event)
            except Exception:
                _logger.exception("Sink %s failed to handle event %s", sink, event["type"])

        # Forward stage events to progress_callback for backwards compatibility
        self._forward_to_progress_callback(event)

    def _forward_to_progress_callback(self, event: OutputEvent) -> None:
        """Forward stage events to progress_callback if set."""
        if self._progress_callback is None:
            return

        import datetime

        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        match event["type"]:
            case "stage_started":
                self._progress_callback(
                    StageStartEvent(
                        type=RunEventType.STAGE_START,
                        stage=event["stage"],
                        index=event["index"],
                        total=event["total"],
                        timestamp=timestamp,
                    )
                )
            case "stage_completed":
                # StageCompleteEvent requires Literal[RAN, SKIPPED, FAILED] status
                status = event["status"]
                if status in (StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED):
                    self._progress_callback(
                        StageCompleteEvent(
                            type=RunEventType.STAGE_COMPLETE,
                            stage=event["stage"],
                            status=status,
                            reason=event["reason"],
                            duration_ms=event["duration_ms"],
                            timestamp=timestamp,
                        )
                    )
            case _:
                pass  # Other events not forwarded to progress_callback

    def close(self) -> None:
        """Close all sinks and clean up resources.

        Exceptions from individual sinks are logged but do not prevent
        other sinks from being closed.
        """
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                _logger.exception("Sink %s failed to close", sink)

    def _handle_progress_event(self, event: RunJsonEvent) -> None:
        """Translate executor progress event to engine event and emit to sinks."""
        match event["type"]:
            case RunEventType.STAGE_START:
                self._stage_indices[event["stage"]] = (event["index"], event["total"])
                self.emit(
                    StageStarted(
                        type="stage_started",
                        stage=event["stage"],
                        index=event["index"],
                        total=event["total"],
                    )
                )
            case RunEventType.STAGE_COMPLETE:
                index, total = self._stage_indices.get(event["stage"], (0, 0))
                self.emit(
                    StageCompleted(
                        type="stage_completed",
                        stage=event["stage"],
                        status=event["status"],
                        reason=event["reason"],
                        duration_ms=event["duration_ms"],
                        index=index,
                        total=total,
                    )
                )
            case _:
                pass

    def run_once(
        self,
        stages: list[str] | None = None,
        force: bool = False,
        single_stage: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        no_commit: bool = False,
        no_cache: bool = False,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
        on_error: OnError = OnError.FAIL,
        cache_dir: pathlib.Path | None = None,
        progress_callback: Callable[[RunJsonEvent], None] | None = None,
    ) -> dict[str, ExecutionSummary]:
        """Execute stages once and return.

        This is the primary entry point for 'pivot run' without --watch.
        Uses Engine orchestration for parallel execution.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run all stages.
            single_stage: If True, run only the specified stages (no downstream).
            parallel: If True, run stages in parallel.
            max_workers: Maximum worker processes.
            no_commit: If True, don't update lockfiles.
            no_cache: If True, disable run cache.
            allow_uncached_incremental: Allow incremental outputs without cache (retained for compatibility).
            checkout_missing: Checkout missing dependency files from cache (retained for compatibility).
            on_error: Error handling mode ('fail' or 'keep_going').
            cache_dir: Directory for lock files (defaults to .pivot/cache).
            progress_callback: Callback for JSONL progress events.

        Returns:
            Dict mapping stage name to ExecutionSummary.

        Raises:
            RuntimeError: If the engine is already active (concurrent calls not allowed).
        """
        # Note: If called via executor.run() with no_commit=True, the pending_state_lock
        # is already held by the caller. Direct callers of run_once() should acquire
        # the lock themselves if needed.

        # Prevent concurrent execution
        if self._state == EngineState.ACTIVE:
            msg = "Engine is already active - concurrent run_once() calls are not allowed"
            raise RuntimeError(msg)

        # Clear cancel event from any previous run
        self._cancel_event.clear()

        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        # Reset stage indices for this run
        self._stage_indices.clear()

        # Store progress callback for event forwarding
        # The Engine emits StageStarted/StageCompleted events directly,
        # but we wrap for backwards compatibility with progress_callback users
        self._progress_callback = progress_callback

        try:
            # Use Engine orchestration
            result = self._orchestrate_execution(
                stages=stages,
                force=force,
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
            return result
        finally:
            # Emit state transition: ACTIVE -> IDLE and clear run state
            self._state = EngineState.IDLE
            self._progress_callback = None
            self._current_run_id = None
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def shutdown(self) -> None:
        """Signal the engine to stop processing events."""
        self._shutdown_event.set()

    def run_loop(self) -> None:
        """Process events until shutdown. For 'pivot run --watch'.

        Blocks until shutdown() is called. Processes events from the
        queue and from registered sources.
        """
        # Start all sources
        for source in self._sources:
            source.start(self.submit)

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for an event with timeout to check shutdown
                    event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._handle_input_event(event)
        finally:
            # Stop all sources
            for source in self._sources:
                source.stop()

            # Ensure we're in IDLE state on exit
            if self._state != EngineState.IDLE:
                self._state = EngineState.IDLE
                self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _handle_input_event(self, event: InputEvent) -> None:
        """Process a single input event."""
        match event["type"]:
            case "run_requested":
                self._handle_run_requested(event)
            case "cancel_requested":
                self._handle_cancel_requested()
            case "data_artifact_changed":
                self._handle_data_artifact_changed(event)
            case "code_or_config_changed":
                self._handle_code_or_config_changed(event)

    def _handle_run_requested(self, event: RunRequested) -> None:
        """Handle a RunRequested event by executing stages."""
        # Clear cancel event before starting new execution
        self._cancel_event.clear()

        # Reset stage indices for this run
        self._stage_indices.clear()

        # Emit state transition: IDLE -> ACTIVE
        # (State may already be ACTIVE if entered via try_start_run agent path)
        if self._state != EngineState.ACTIVE:
            self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            self._orchestrate_execution(
                stages=event["stages"],
                force=event["force"],
                single_stage=False,
                parallel=True,
                max_workers=None,
                no_commit=False,
                no_cache=False,
                on_error=OnError.KEEP_GOING,  # Watch mode: continue on error
                cache_dir=None,
            )
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _handle_cancel_requested(self) -> None:
        """Handle a CancelRequested event by setting cancel flag."""
        self._cancel_event.set()

    def _handle_data_artifact_changed(self, event: DataArtifactChanged) -> None:
        """Handle data artifact changes by running affected stages."""
        paths = [pathlib.Path(p) for p in event["paths"]]

        # Filter out paths that are outputs of executing stages
        filtered_paths = list[pathlib.Path]()
        deferred_paths = list[tuple[str, pathlib.Path]]()

        for path in paths:
            if self._should_filter_path(path):
                producer = engine_graph.get_producer(self._graph, path) if self._graph else None
                if producer:
                    deferred_paths.append((producer, path))
                    continue
            filtered_paths.append(path)

        # Defer events for filtered paths
        for producer, path in deferred_paths:
            self._defer_event_for_stage(
                producer,
                DataArtifactChanged(type="data_artifact_changed", paths=[str(path)]),
            )

        if not filtered_paths:
            return

        # Get affected stages
        affected = self._get_affected_stages_for_paths(filtered_paths)

        if not affected:
            return

        _logger.info(
            "Data changed: %d file(s) affect %d stage(s)", len(filtered_paths), len(affected)
        )

        # Execute affected stages
        self._execute_affected_stages(affected)

    def _handle_code_or_config_changed(self, _event: CodeOrConfigChanged) -> None:
        """Handle code/config changes by reloading registry and re-running."""
        _logger.info("Code/config changed - reloading pipeline")

        # Invalidate caches
        self._invalidate_caches()

        # Reload registry
        reload_ok = self._reload_registry()

        if not reload_ok:
            _logger.error("Pipeline invalid - waiting for fix")
            return

        # Rebuild graph
        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        self._graph = engine_graph.build_graph(all_stages)

        # Update watch paths if we have a FilesystemSource
        from pivot.engine.sources import FilesystemSource

        watch_paths = engine_graph.get_watch_paths(self._graph)
        for source in self._sources:
            if isinstance(source, FilesystemSource):
                source.set_watch_paths(watch_paths)

        # Re-run all stages
        stages = list(registry.REGISTRY.list_stages())

        if stages:
            self._execute_affected_stages(stages)

    def _execute_affected_stages(self, stages: list[str]) -> None:
        """Execute the specified stages."""
        self._cancel_event.clear()

        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            self._orchestrate_execution(
                stages=stages,
                force=False,
                single_stage=False,
                parallel=True,
                max_workers=None,
                no_commit=False,
                no_cache=False,
                on_error=OnError.KEEP_GOING,  # Watch mode: continue on error
                cache_dir=None,
            )
        finally:
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _should_filter_path(self, path: pathlib.Path) -> bool:
        """Check if path should be filtered (output of executing stage).

        Uses IntEnum ordering for comparison: filter if state >= PREPARING and < COMPLETED.
        """
        if self._graph is None:
            return False

        # Get the stage that produces this artifact
        producer = engine_graph.get_producer(self._graph, path)
        if producer is None:
            return False

        # Filter if producer is currently executing (PREPARING or RUNNING)
        state = self.get_stage_state(producer)
        return StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED

    def _get_output_paths_for_stage(self, stage: str) -> list[pathlib.Path]:
        """Get all output paths for a stage from the graph."""
        if self._graph is None:
            return []

        node = engine_graph.stage_node(stage)
        if node not in self._graph:
            return []

        paths = list[pathlib.Path]()
        for successor in self._graph.successors(node):
            if self._graph.nodes[successor]["type"] == NodeType.ARTIFACT:
                _, path_str = engine_graph.parse_node(successor)
                paths.append(pathlib.Path(path_str))
        return paths

    def _defer_event_for_stage(self, stage: str, event: InputEvent) -> None:
        """Defer an event until the stage completes."""
        if stage not in self._deferred_events:
            self._deferred_events[stage] = list[InputEvent]()
        self._deferred_events[stage].append(event)

    def _process_deferred_events(self, stage: str) -> None:
        """Process any deferred events for a completed stage.

        Uses iterative approach to avoid recursion if processing defers more events.
        """
        # Iteratively process - if handling an event defers more, they'll be processed
        # in subsequent iterations (when their stage completes), not recursively
        events = self._deferred_events.pop(stage, [])
        for event in events:
            self._handle_input_event(event)

    def _get_affected_stages_for_path(self, path: pathlib.Path) -> list[str]:
        """Get stages affected by a path change using bipartite graph."""
        if self._graph is None:
            return []

        # Use get_consumers() from engine/graph.py
        consumers = engine_graph.get_consumers(self._graph, path)
        if not consumers:
            return []

        # Add downstream stages
        all_affected = set(consumers)
        for stage in consumers:
            downstream = engine_graph.get_downstream_stages(self._graph, stage)
            all_affected.update(downstream)

        return list(all_affected)

    def _get_affected_stages_for_paths(
        self, paths: list[pathlib.Path], *, include_downstream: bool = True
    ) -> list[str]:
        """Get all stages affected by multiple path changes."""
        affected = set[str]()

        for path in paths:
            # Skip if this is an output of an executing stage
            if self._should_filter_path(path):
                _logger.debug("Filtering event for %s (output of executing stage)", path)
                continue

            if include_downstream:
                stage_affected = self._get_affected_stages_for_path(path)
                affected.update(stage_affected)
            else:
                # Only direct consumers, not downstream
                if self._graph:
                    affected.update(engine_graph.get_consumers(self._graph, path))

        return list(affected)

    # =========================================================================
    # Registry Reload
    # =========================================================================

    def _invalidate_caches(self) -> None:
        """Invalidate all caches when code changes."""
        linecache.clearcache()
        importlib.invalidate_caches()
        self._graph = None
        registry.REGISTRY.invalidate_dag_cache()

    def _reload_registry(self) -> bool:
        """Reload the registry by re-importing pipeline definition.

        Returns True if reload succeeded, False if pipeline is invalid.
        """
        old_stages = registry.REGISTRY.snapshot()
        root = project.get_project_root()

        # Clear project modules from sys.modules
        self._clear_project_modules(root)

        # Try pivot.yaml first
        for name in ("pivot.yaml", "pivot.yml"):
            path = root / name
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        yaml_config = yaml.safe_load(f)
                    if isinstance(yaml_config, dict) and "stages" in yaml_config:
                        return self._reload_from_pipeline_file(path, old_stages)
                except Exception:
                    continue

        # Try pipeline.py
        pipeline_py = root / "pipeline.py"
        if pipeline_py.exists():
            return self._reload_from_pipeline_py(pipeline_py, old_stages)

        # Fallback: reimport stage modules
        return self._reload_from_decorators(old_stages)

    def _clear_project_modules(self, root: pathlib.Path) -> None:
        """Remove project modules from sys.modules."""
        root_str = str(root)
        to_remove = list[str]()

        for name, module in list(sys.modules.items()):
            if module is None:  # pyright: ignore[reportUnnecessaryComparison] - sys.modules values can be None
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            try:
                if module_file.startswith(root_str):
                    to_remove.append(name)
            except (TypeError, AttributeError):
                continue

        for name in to_remove:
            del sys.modules[name]

    def _try_reload(
        self,
        action: Callable[[], object],
        old_stages: dict[str, RegistryStageInfo],
    ) -> bool:
        """Execute a reload action with standard error handling.

        Clears registry, runs action, emits reload event on success.
        On failure, restores old stages and returns False.
        """
        registry.REGISTRY.clear()
        try:
            action()
            self._emit_reload_event(old_stages)
            return True
        except Exception as e:
            _logger.warning(f"Pipeline invalid: {e}")
            registry.REGISTRY.restore(old_stages)
            return False

    def _reload_from_pipeline_file(
        self, path: pathlib.Path, old_stages: dict[str, RegistryStageInfo]
    ) -> bool:
        """Reload from pivot.yaml file."""
        from pivot.pipeline import yaml as pipeline_yaml

        return self._try_reload(
            lambda: pipeline_yaml.register_from_pipeline_file(path),
            old_stages,
        )

    def _reload_from_pipeline_py(
        self, path: pathlib.Path, old_stages: dict[str, RegistryStageInfo]
    ) -> bool:
        """Reload from pipeline.py file."""
        return self._try_reload(
            lambda: runpy.run_path(str(path), run_name="_pivot_pipeline"),
            old_stages,
        )

    def _reload_from_decorators(self, old_stages: dict[str, RegistryStageInfo]) -> bool:
        """Reload by reimporting stage modules."""
        modules: set[str] = set()
        for info in old_stages.values():
            func = info["func"]
            module_name = getattr(func, "__module__", None)
            if module_name and module_name in sys.modules:
                modules.add(module_name)

        if not modules:
            return True

        registry.REGISTRY.clear()
        errors = list[str]()

        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                errors.append(f"{module_name}: {e}")

        if errors:
            registry.REGISTRY.restore(old_stages)
            return False

        self._emit_reload_event(old_stages)
        return True

    def _emit_reload_event(self, old_stages: dict[str, RegistryStageInfo]) -> None:
        """Emit PipelineReloaded event with diff information."""
        new_stages = set(registry.REGISTRY.list_stages())
        old_stage_names = set(old_stages.keys())

        added = list(new_stages - old_stage_names)
        removed = list(old_stage_names - new_stages)

        # Detect modified stages by comparing fingerprints
        modified = list[str]()
        for stage_name in old_stage_names & new_stages:
            old_info = old_stages[stage_name]
            new_info = registry.REGISTRY.get(stage_name)
            if old_info["fingerprint"] != new_info["fingerprint"]:
                modified.append(stage_name)

        self.emit(
            PipelineReloaded(
                type="pipeline_reloaded",
                stages_added=added,
                stages_removed=removed,
                stages_modified=modified,
                error=None,
            )
        )

    # =========================================================================
    # Execution Orchestration
    # =========================================================================

    def _initialize_orchestration(
        self,
        execution_order: list[str],
        max_workers: int,
        error_mode: OnError,
    ) -> None:
        """Initialize orchestration state for a new execution.

        Uses bipartite graph (self._graph) to derive stage dependencies via
        engine_graph.get_upstream_stages() and get_downstream_stages().
        """
        self._futures.clear()
        self._mutex_counts.clear()
        self._stage_upstream_unfinished.clear()
        self._stage_downstream.clear()
        self._stage_mutex.clear()
        self._stage_states.clear()
        self._deferred_events.clear()
        self._stop_starting_new = False

        self._max_workers = max_workers
        self._error_mode = error_mode

        stages_set = set(execution_order)

        for stage_name in execution_order:
            stage_info = registry.REGISTRY.get(stage_name)

            # Upstream stages that must complete first (uses bipartite graph)
            if self._graph is not None:
                upstream = [
                    u
                    for u in engine_graph.get_upstream_stages(self._graph, stage_name)
                    if u in stages_set
                ]
            else:
                upstream = []
            self._stage_upstream_unfinished[stage_name] = set(upstream)

            # Downstream stages that depend on this one (uses bipartite graph)
            if self._graph is not None:
                downstream = [
                    d
                    for d in engine_graph.get_downstream_stages(self._graph, stage_name)
                    if d in stages_set
                ]
            else:
                downstream = []
            self._stage_downstream[stage_name] = downstream

            # Mutex groups
            self._stage_mutex[stage_name] = stage_info["mutex"]

            # Initial state: READY if no upstream, else PENDING
            initial_state = (
                StageExecutionState.READY if not upstream else StageExecutionState.PENDING
            )
            self._set_stage_state(stage_name, initial_state)

    def _warn_single_stage_mutex_groups(self) -> None:
        """Warn if any mutex group contains only one stage (likely a typo).

        Only warns once per group to avoid repeated warnings in watch mode.
        """
        groups: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        for stage_name, mutexes in self._stage_mutex.items():
            for mutex in mutexes:
                groups[mutex].append(stage_name)

        for group, members in groups.items():
            # Skip EXCLUSIVE_MUTEX - it's intentionally used for single stages
            if group == executor_core.EXCLUSIVE_MUTEX:
                continue
            if len(members) == 1 and group not in self._warned_mutex_groups:
                self._warned_mutex_groups.add(group)
                _logger.warning(f"Mutex group '{group}' only contains stage '{members[0]}'")

    def _can_start_stage(self, stage_name: str) -> bool:
        """Check if stage is eligible to start (ready and mutex available)."""
        if self.get_stage_state(stage_name) != StageExecutionState.READY:
            return False

        # Check if upstream dependencies are all finished
        if self._stage_upstream_unfinished.get(stage_name):
            return False

        stage_mutexes = self._stage_mutex.get(stage_name, [])
        is_exclusive = executor_core.EXCLUSIVE_MUTEX in stage_mutexes

        # Check mutex availability for this stage's mutexes
        for mutex in stage_mutexes:
            if mutex == executor_core.EXCLUSIVE_MUTEX:
                # Exclusive mutex: no other stages can run
                if self._mutex_counts[mutex] > 0 or len(self._futures) > 0:
                    return False
            elif self._mutex_counts[mutex] > 0:
                return False

        # Non-exclusive stages can't start while an exclusive stage is running
        return is_exclusive or self._mutex_counts[executor_core.EXCLUSIVE_MUTEX] == 0

    def _get_stage_index(self, stage_name: str) -> tuple[int, int]:
        """Get (1-based index, total count) for a stage.

        Used for progress reporting in events.
        """
        stage_index = list(self._stage_states.keys()).index(stage_name) + 1
        total_stages = len(self._stage_states)
        return stage_index, total_stages

    def _emit_skipped_stage(
        self,
        stage_name: str,
        reason: str,
        results: dict[str, executor_core.ExecutionSummary],
    ) -> None:
        """Record and emit a skipped/blocked stage completion.

        Updates results dict and emits StageCompleted event.
        """
        results[stage_name] = executor_core.ExecutionSummary(
            status=StageStatus.SKIPPED,
            reason=reason,
        )
        stage_index, total_stages = self._get_stage_index(stage_name)
        self.emit(
            StageCompleted(
                type="stage_completed",
                stage=stage_name,
                status=StageStatus.SKIPPED,
                reason=reason,
                duration_ms=0.0,
                index=stage_index,
                total=total_stages,
            )
        )

    def _start_ready_stages(
        self,
        cache_dir: pathlib.Path,
        output_queue: mp.Queue[OutputMessage],
        overrides: parameters.ParamsOverrides,
        checkout_modes: list[CheckoutMode],
        force: bool,
        no_commit: bool,
        no_cache: bool,
        stage_start_times: dict[str, float],
        run_id: str,
        project_root: pathlib.Path,
        state_dir: pathlib.Path,
    ) -> None:
        """Start all eligible stages up to max_workers."""
        if self._executor is None or self._stop_starting_new:
            return

        started = 0
        max_to_start = self._max_workers - len(self._futures)
        if max_to_start <= 0:
            return

        # Iterate through stages in state order (deterministic)
        # Re-check eligibility each iteration since starting a stage acquires mutexes
        for stage_name in list(self._stage_states.keys()):
            if started >= max_to_start:
                break

            if not self._can_start_stage(stage_name):
                continue

            # Acquire mutex locks
            for mutex in self._stage_mutex.get(stage_name, []):
                self._mutex_counts[mutex] += 1

            started += 1

            # Transition to PREPARING
            self._set_stage_state(stage_name, StageExecutionState.PREPARING)

            # Get stage info and prepare worker info in main process
            # (worker processes may not have test-registered stages)
            stage_info = registry.REGISTRY.get(stage_name)
            worker_info = executor_core.prepare_worker_info(
                stage_info,
                overrides,
                checkout_modes,
                run_id,
                force,
                no_commit,
                no_cache,
                project_root,
                state_dir,
            )

            # Submit to executor using worker.execute_stage directly
            future = self._executor.submit(
                worker.execute_stage,
                stage_name,
                worker_info,
                cache_dir,
                output_queue,
            )
            self._futures[future] = stage_name

            # Record start time for duration calculation
            stage_start_times[stage_name] = time.perf_counter()

            # Transition to RUNNING and emit StageStarted
            self._set_stage_state(stage_name, StageExecutionState.RUNNING)

            # Emit StageStarted event (sinks use this for display)
            stage_index, total_stages = self._get_stage_index(stage_name)
            self.emit(
                StageStarted(
                    type="stage_started",
                    stage=stage_name,
                    index=stage_index,
                    total=total_stages,
                )
            )

    def _handle_stage_completion(
        self,
        stage_name: str,
        result: StageResult,
        start_time: float,
    ) -> float:
        """Handle a stage completing execution.

        Returns:
            Duration in milliseconds for the stage execution.
        """
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Transition to COMPLETED
        self._set_stage_state(stage_name, StageExecutionState.COMPLETED)

        # Emit StageCompleted event (sinks use this for display)
        stage_index, total_stages = self._get_stage_index(stage_name)
        self.emit(
            StageCompleted(
                type="stage_completed",
                stage=stage_name,
                status=result["status"],
                reason=result["reason"],
                duration_ms=duration_ms,
                index=stage_index,
                total=total_stages,
            )
        )

        # Release mutex locks
        for mutex in self._stage_mutex.get(stage_name, []):
            self._mutex_counts[mutex] -= 1
            if self._mutex_counts[mutex] < 0:
                _logger.error("Mutex '%s' released when not held", mutex)
                self._mutex_counts[mutex] = 0

        # Update downstream stages' upstream_unfinished
        for downstream_name in self._stage_downstream.get(stage_name, []):
            unfinished = self._stage_upstream_unfinished.get(downstream_name)
            if unfinished:
                unfinished.discard(stage_name)
                # If all upstream complete, mark as READY
                if (
                    not unfinished
                    and self.get_stage_state(downstream_name) == StageExecutionState.PENDING
                ):
                    self._set_stage_state(downstream_name, StageExecutionState.READY)

        # Handle failure cascading
        if result["status"] == StageStatus.FAILED:
            self._cascade_failure(stage_name)

        # Process deferred events for this stage's outputs
        self._process_deferred_events(stage_name)

        return duration_ms

    def _cascade_failure(self, failed_stage: str) -> None:
        """Mark downstream stages as blocked due to upstream failure."""
        for downstream_name in self._stage_downstream.get(failed_stage, []):
            state = self.get_stage_state(downstream_name)
            if state in (StageExecutionState.PENDING, StageExecutionState.READY):
                self._set_stage_state(downstream_name, StageExecutionState.BLOCKED)
                # Recursively cascade
                self._cascade_failure(downstream_name)

    def _write_run_history(
        self,
        run_id: str,
        results: dict[str, executor_core.ExecutionSummary],
        stage_durations: dict[str, float],
        targeted_stages: list[str],
        execution_order: list[str],
        started_at: str,
        ended_at: str,
        retention: int,
    ) -> None:
        """Build and write run manifest to StateDB."""
        from pivot import run_history
        from pivot.storage import lock

        state_dir = config.get_state_dir()

        stages_records = dict[str, run_history.StageRunRecord]()
        for name, summary in results.items():
            # Read lock file to compute input hash
            stage_lock = lock.StageLock(name, lock.get_stages_dir(state_dir))
            lock_data = stage_lock.read()

            if lock_data:
                input_hash = run_history.compute_input_hash_from_lock(lock_data)
            else:
                input_hash = "<no-lock>"

            # Use actual duration captured at stage completion time
            duration_ms = int(stage_durations.get(name, 0))

            stages_records[name] = run_history.StageRunRecord(
                input_hash=input_hash,
                status=summary["status"],
                reason=summary["reason"],
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

        with state_mod.StateDB(config.get_state_db_path()) as state_db:
            state_db.write_run(manifest)
            state_db.prune_runs(retention)

    def _orchestrate_execution(
        self,
        stages: list[str] | None,
        force: bool,
        single_stage: bool,
        parallel: bool,
        max_workers: int | None,
        no_commit: bool,
        no_cache: bool,
        on_error: OnError,
        cache_dir: pathlib.Path | None,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
    ) -> dict[str, executor_core.ExecutionSummary]:
        """Orchestrate parallel stage execution with the Engine's event loop.

        Note: If no_commit=True, the pending state lock should be held by the caller
        (run_once acquires it before calling this method).
        """
        import datetime

        from pivot import run_history

        # Record start time for run history
        started_at = datetime.datetime.now(datetime.UTC).isoformat()

        if cache_dir is None:
            cache_dir = config.get_cache_dir()

        # Verify tracked files before building DAG (provides better error messages)
        # This catches missing tracked files before DAG validation throws DependencyNotFoundError
        project_root = project.get_project_root()
        executor_core.verify_tracked_files(project_root, checkout_missing=checkout_missing)

        # Build bipartite graph (single source of truth)
        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        self._graph = engine_graph.build_graph(all_stages)

        # Validate dependencies exist (raises DependencyNotFoundError if missing)
        # This uses dag.build_dag() which checks that all deps either exist on disk,
        # are produced by another stage, or are tracked files
        dag.build_dag(all_stages, validate=True)

        # Extract stage-only DAG for execution order
        stage_dag = engine_graph.get_stage_dag(self._graph)

        if stages:
            registered = set(stage_dag.nodes())
            unknown = [s for s in stages if s not in registered]
            if unknown:
                from pivot import exceptions

                raise exceptions.StageNotFoundError(unknown, available_stages=list(registered))

        execution_order = dag.get_execution_order(stage_dag, stages, single_stage=single_stage)

        if not execution_order:
            return {}

        # Check for uncached IncrementalOut files that would be lost
        if not allow_uncached_incremental:
            uncached = executor_core.check_uncached_incremental_outputs(execution_order)
            if uncached:
                from pivot import exceptions

                files_list = "\n".join(f"  - {stage}: {path}" for stage, path in uncached)
                raise exceptions.UncachedIncrementalOutputError(
                    f"The following IncrementalOut files exist but are not in cache:\n{files_list}\n\n"
                    + "Running the pipeline will DELETE these files and they cannot be restored.\n"
                    + "To proceed anyway, use allow_uncached_incremental=True or back up these files first."
                )

        # Compute max workers
        # Reuse executor private helpers until Task 14 makes them public
        effective_max_workers = (
            1
            if not parallel
            else executor_core.compute_max_workers(len(execution_order), max_workers)
        )

        # Load config
        overrides = parameters.load_params_yaml()
        checkout_modes = config.get_checkout_mode_order()

        # Get project paths for worker info
        project_root = project.get_project_root()
        state_dir = config.get_state_dir()
        run_id = run_history.generate_run_id()

        # Ensure state directory exists (workers open StateDB in readonly mode,
        # which requires the directory to exist)
        state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize orchestration state (uses bipartite graph for dependencies)
        self._initialize_orchestration(execution_order, effective_max_workers, on_error)
        self._warn_single_stage_mutex_groups()

        # Create executor
        self._executor = executor_core.create_executor(effective_max_workers)

        # Create output queue
        spawn_ctx = mp.get_context("spawn")
        local_manager = spawn_ctx.Manager()
        output_queue: mp.Queue[OutputMessage] = local_manager.Queue()  # pyright: ignore[reportAssignmentType]

        # Track results, start times, and actual durations
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_start_times: dict[str, float] = {}
        stage_durations: dict[str, float] = {}

        # Start output reader thread for LogLine events
        output_thread: threading.Thread | None = None
        output_stop_event = threading.Event()

        # Get state DB path (same pattern as executor/core.py)
        state_db_path = config.get_state_db_path()

        try:
            # Start output drain thread to emit LogLine events
            output_thread = threading.Thread(
                target=self._drain_output_queue,
                args=(output_queue, output_stop_event),
                daemon=True,
            )
            output_thread.start()

            # Open StateDB to ensure database exists before workers start (they open readonly)
            # Keep it open throughout execution to apply deferred writes
            with state_mod.StateDB(state_db_path) as state_db:
                # Start initial ready stages
                self._start_ready_stages(
                    cache_dir=cache_dir,
                    output_queue=output_queue,
                    overrides=overrides,
                    checkout_modes=checkout_modes,
                    force=force,
                    no_commit=no_commit,
                    no_cache=no_cache,
                    stage_start_times=stage_start_times,
                    run_id=run_id,
                    project_root=project_root,
                    state_dir=state_dir,
                )

                # Main execution loop
                while self._futures:
                    done, _ = concurrent.futures.wait(
                        self._futures.keys(),
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        stage_name = self._futures.pop(future)
                        start_time = stage_start_times.get(stage_name, time.perf_counter())

                        try:
                            result = future.result()
                            duration_ms = self._handle_stage_completion(
                                stage_name, result, start_time
                            )
                            stage_durations[stage_name] = duration_ms

                            # Apply deferred writes for successful stages (only in commit mode)
                            if result["status"] == StageStatus.RAN and not no_commit:
                                stage_info = registry.REGISTRY.get(stage_name)
                                output_paths = [str(out.path) for out in stage_info["outs"]]
                                executor_core.apply_deferred_writes(
                                    stage_name, output_paths, result, state_db
                                )

                            # Record result
                            results[stage_name] = executor_core.ExecutionSummary(
                                status=result["status"],
                                reason=result["reason"],
                            )

                        except Exception as e:
                            _logger.exception("Stage %s failed with exception", stage_name)
                            failed_result = StageResult(
                                status=StageStatus.FAILED,
                                reason=str(e),
                                output_lines=[],
                            )
                            duration_ms = self._handle_stage_completion(
                                stage_name, failed_result, start_time
                            )
                            stage_durations[stage_name] = duration_ms
                            results[stage_name] = executor_core.ExecutionSummary(
                                status=StageStatus.FAILED,
                                reason=str(e),
                            )

                    # Check error mode
                    if on_error == OnError.FAIL:
                        failed = [
                            n
                            for n, s in self._stage_states.items()
                            if s == StageExecutionState.COMPLETED
                            and results.get(n, {}).get("status") == StageStatus.FAILED
                        ]
                        if failed:
                            self._stop_starting_new = True
                            # Mark remaining READY/PENDING as blocked and add results for
                            # all blocked stages (including those from _cascade_failure)
                            for name, state in self._stage_states.items():
                                if state in (
                                    StageExecutionState.READY,
                                    StageExecutionState.PENDING,
                                ):
                                    self._set_stage_state(name, StageExecutionState.BLOCKED)
                                # Add result and emit event for all blocked stages that don't have one
                                if (
                                    state == StageExecutionState.BLOCKED
                                    or self.get_stage_state(name) == StageExecutionState.BLOCKED
                                ) and name not in results:
                                    self._emit_skipped_stage(
                                        name, f"upstream '{failed[0]}' failed", results
                                    )

                    # Check cancellation
                    if self._cancel_event.is_set():
                        self._stop_starting_new = True
                        for name, state in self._stage_states.items():
                            if state in (StageExecutionState.READY, StageExecutionState.PENDING):
                                self._set_stage_state(name, StageExecutionState.COMPLETED)
                                self._emit_skipped_stage(name, "cancelled", results)

                    # Start more stages if slots available
                    if not self._stop_starting_new:
                        self._start_ready_stages(
                            cache_dir=cache_dir,
                            output_queue=output_queue,
                            overrides=overrides,
                            checkout_modes=checkout_modes,
                            force=force,
                            no_commit=no_commit,
                            no_cache=no_cache,
                            stage_start_times=stage_start_times,
                            run_id=run_id,
                            project_root=project_root,
                            state_dir=state_dir,
                        )

                # After the main loop, handle any blocked stages that weren't processed
                # (happens in keep_going mode when a stage fails and blocks downstream)
                for name, state in self._stage_states.items():
                    if state == StageExecutionState.BLOCKED and name not in results:
                        # Find which upstream stage failed
                        failed_upstream = next(
                            (
                                n
                                for n, r in results.items()
                                if r.get("status") == StageStatus.FAILED
                            ),
                            "unknown",
                        )
                        self._emit_skipped_stage(
                            name, f"upstream '{failed_upstream}' failed", results
                        )
        finally:
            # Signal output thread to stop
            output_stop_event.set()
            if output_thread:
                output_thread.join(timeout=1.0)

            self._executor = None
            local_manager.shutdown()

        # Write run history after execution completes
        ended_at = datetime.datetime.now(datetime.UTC).isoformat()
        targeted_stages = stages if stages else execution_order
        retention = config.get_run_history_retention()

        self._write_run_history(
            run_id=run_id,
            results=results,
            stage_durations=stage_durations,
            targeted_stages=targeted_stages,
            execution_order=execution_order,
            started_at=started_at,
            ended_at=ended_at,
            retention=retention,
        )

        return results

    def _drain_output_queue(
        self,
        output_queue: mp.Queue[OutputMessage],
        stop_event: threading.Event,
    ) -> None:
        """Drain output messages from worker processes and emit LogLine events."""
        while not stop_event.is_set():
            try:
                msg = output_queue.get(timeout=_OUTPUT_QUEUE_DRAIN_TIMEOUT)
                if msg is None:
                    break

                # Emit LogLine event for each output message
                # OutputMessage is tuple[str, str, bool]: (stage_name, line, is_stderr)
                try:
                    stage_name, line, is_stderr = msg
                except (TypeError, ValueError):
                    _logger.debug(f"Malformed output message, skipping: {msg!r}")
                    continue

                self.emit(
                    LogLine(
                        type="log_line",
                        stage=stage_name,
                        line=line,
                        is_stderr=is_stderr,
                    )
                )
            except queue.Empty:
                continue
            except (EOFError, OSError, BrokenPipeError):
                _logger.debug("Output queue drain exiting: queue closed or broken")
                break

    # =========================================================================
    # Agent RPC Methods
    # =========================================================================

    def try_start_run(
        self,
        run_id: str,
        stages: list[str] | None,
        force: bool,
    ) -> AgentRunStartResult | AgentRunRejection:
        """Atomically try to start a run.

        Returns AgentRunStartResult if started, AgentRunRejection if rejected.
        Thread-safe - can be called from asyncio thread.

        Sets state to ACTIVE atomically before submitting to prevent race
        conditions where concurrent calls could both see IDLE state.
        """
        stages_to_run = stages or list(registry.REGISTRY.list_stages())

        with self._run_lock:
            if self._state != EngineState.IDLE:
                return AgentRunRejection(
                    reason="busy",
                    current_state=self._state.value,
                    current_run_id=self._current_run_id,
                )

            # Set state to ACTIVE atomically before releasing lock
            # This prevents race where another try_start_run sees IDLE
            self._state = EngineState.ACTIVE
            self._current_run_id = run_id

            # Submit run request (event loop will handle execution)
            self.submit(
                RunRequested(
                    type="run_requested",
                    stages=stages_to_run,
                    force=force,
                    reason=f"agent:{run_id}",
                )
            )

            return AgentRunStartResult(
                run_id=run_id,
                status="started",
                stages_queued=stages_to_run,
            )

    def get_execution_status(self, run_id: str | None = None) -> AgentStatusResult:
        """Query current execution state.

        Thread-safe - can be called from asyncio thread.
        """
        with self._run_lock:
            status = AgentStatusResult(state=self._state.value)

            if run_id is not None and run_id != self._current_run_id:
                return status

            if self._current_run_id is not None:
                status["run_id"] = self._current_run_id

            # Get stages by state - copy dict under lock to avoid iteration issues
            with self._stage_states_lock:
                stage_states_copy = dict(self._stage_states)

            completed = [
                name
                for name, state in stage_states_copy.items()
                if state == StageExecutionState.COMPLETED
            ]
            pending = [
                name
                for name, state in stage_states_copy.items()
                if state
                in (
                    StageExecutionState.PENDING,
                    StageExecutionState.READY,
                    StageExecutionState.PREPARING,
                    StageExecutionState.RUNNING,
                )
            ]

            if completed:
                status["stages_completed"] = completed
            if pending:
                status["stages_pending"] = pending

            return status

    def request_cancel(self) -> AgentCancelResult:
        """Request cancellation of current execution.

        Thread-safe - can be called from asyncio thread.
        """
        with self._run_lock:
            if self._state == EngineState.ACTIVE:
                self._cancel_event.set()
                return AgentCancelResult(cancelled=True)
            return AgentCancelResult(cancelled=False)
