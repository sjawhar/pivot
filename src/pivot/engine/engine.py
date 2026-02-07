from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import importlib
import linecache
import logging
import multiprocessing as mp
import pathlib
import queue
import sys
import time
from typing import TYPE_CHECKING, Self

import anyio
import anyio.from_thread
import anyio.to_thread

from pivot import config, exceptions, parameters, project
from pivot.engine import agent_rpc
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
from pivot.types import OnError, OutputMessage, StageResult, StageStatus

if TYPE_CHECKING:
    import networkx as nx
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from pivot import registry
    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo
    from pivot.storage.cache import CheckoutMode

__all__ = ["Engine"]

_logger = logging.getLogger(__name__)

# Channel buffer sizes for backpressure
_INPUT_BUFFER_SIZE = 32
_OUTPUT_BUFFER_SIZE = 64


class Engine:
    """Async coordinator for pipeline execution using anyio.

    Thread safety: All state access occurs in the event loop task within run().
    Sources run in separate tasks but only send events to channels - they don't
    access engine state. This design provides implicit serialization through
    the channel, so no explicit lock is needed.
    """

    _pipeline: Pipeline | None
    _state: EngineState
    _sources: list[EventSource]
    _sinks: list[EventSink]
    _event_buffer: agent_rpc.EventBuffer
    _input_send: MemoryObjectSendStream[InputEvent] | None
    _input_recv: MemoryObjectReceiveStream[InputEvent] | None
    _output_send: MemoryObjectSendStream[OutputEvent] | None
    _output_recv: MemoryObjectReceiveStream[OutputEvent] | None

    # Orchestration state
    _graph: nx.DiGraph[str] | None
    _stage_states: dict[str, StageExecutionState]
    _cancel_event: anyio.Event
    _stage_indices: dict[str, tuple[int, int]]
    _deferred_events: dict[str, list[InputEvent]]

    # Execution orchestration state
    _futures: dict[concurrent.futures.Future[StageResult], str]
    _mutex_counts: collections.defaultdict[str, int]
    _stage_upstream_unfinished: dict[str, set[str]]
    _stage_downstream: dict[str, list[str]]
    _stage_mutex: dict[str, list[str]]
    _executor: concurrent.futures.Executor | None
    _max_workers: int
    _error_mode: OnError
    _stop_starting_new: bool
    _warned_mutex_groups: set[str]

    # Stored orchestration params (for watch mode re-runs)
    _stored_no_commit: bool
    _stored_no_cache: bool
    _stored_on_error: OnError

    # Track whether run() has completed to prevent re-use
    _run_completed: bool

    # Event signaling dispatcher has finished draining
    _dispatch_complete: anyio.Event

    def __init__(self, *, pipeline: Pipeline | None = None) -> None:
        """Initialize the async engine in IDLE state."""
        self._pipeline = pipeline
        self._state = EngineState.IDLE
        self._sources = list[EventSource]()
        self._sinks = list[EventSink]()
        self._event_buffer = agent_rpc.EventBuffer(max_events=1000)

        # Channels created on __aenter__
        self._input_send = None
        self._input_recv = None
        self._output_send = None
        self._output_recv = None

        # Orchestration state
        self._graph = None
        self._stage_states = dict[str, StageExecutionState]()
        self._cancel_event = anyio.Event()
        self._stage_indices = dict[str, tuple[int, int]]()
        self._deferred_events = collections.defaultdict(list)

        # Execution orchestration state
        self._futures = dict[concurrent.futures.Future[StageResult], str]()
        self._mutex_counts = collections.defaultdict(int)
        self._stage_upstream_unfinished = dict[str, set[str]]()
        self._stage_downstream = dict[str, list[str]]()
        self._stage_mutex = dict[str, list[str]]()
        self._executor = None
        self._max_workers = 1
        self._error_mode = OnError.FAIL
        self._stop_starting_new = False
        self._warned_mutex_groups = set[str]()

        # Stored orchestration params (for watch mode re-runs)
        self._stored_no_commit = False
        self._stored_no_cache = False
        self._stored_on_error = OnError.FAIL

        # Track whether run() has completed to prevent re-use
        self._run_completed = False

        # Event signaling dispatcher has finished draining (recreated each run())
        self._dispatch_complete = anyio.Event()

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        return self._state

    @property
    def sources(self) -> list[EventSource]:
        """Registered async event sources (returns a copy)."""
        return list(self._sources)

    @property
    def sinks(self) -> list[EventSink]:
        """Registered async event sinks (returns a copy)."""
        return list(self._sinks)

    def add_source(self, source: EventSource) -> None:
        """Register an async event source."""
        self._sources.append(source)

    def add_sink(self, sink: EventSink) -> None:
        """Register an async event sink."""
        self._sinks.append(sink)

    async def __aenter__(self) -> Self:
        """Set up memory channels for event flow."""
        self._input_send, self._input_recv = anyio.create_memory_object_stream[InputEvent](
            _INPUT_BUFFER_SIZE
        )
        self._output_send, self._output_recv = anyio.create_memory_object_stream[OutputEvent](
            _OUTPUT_BUFFER_SIZE
        )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close all sinks and channels.

        Closes send channels first to signal receivers, then receive channels.
        This ordering prevents ClosedResourceError in iterating tasks.
        """
        # Close send channels first to signal end-of-stream to receivers
        if self._input_send:
            await self._input_send.aclose()
        if self._output_send:
            await self._output_send.aclose()

        # Close sinks after output_send is closed (no more events will be dispatched)
        for sink in self._sinks:
            try:
                await sink.close()
            except Exception:
                _logger.exception("Error closing sink %s", sink)

        # Close receive channels last (they may still be iterating)
        if self._input_recv:
            await self._input_recv.aclose()
        if self._output_recv:
            await self._output_recv.aclose()

    async def emit(self, event: OutputEvent) -> None:
        """Emit an output event to all sinks.

        Silently drops events if the output channel is closed (during shutdown).
        """
        if self._output_send:
            with contextlib.suppress(anyio.ClosedResourceError):
                await self._output_send.send(event)

    async def run(self, *, exit_on_completion: bool = True) -> None:
        """Run the async engine with registered sources and sinks.

        Args:
            exit_on_completion: If True, exit after all sources have emitted
                and no more stages are running (one-shot mode). If False,
                continue running until cancelled (watch mode).

        Raises:
            RuntimeError: If run() has already completed on this Engine instance.
                Engine instances cannot be reused after run() returns.
        """
        if self._run_completed:
            msg = (
                "Engine.run() has already completed. Engine instances cannot be reused - "
                "create a new Engine for each run."
            )
            raise RuntimeError(msg)
        if self._input_send is None or self._input_recv is None:
            raise RuntimeError("Engine must be used as async context manager")
        if self._output_send is None or self._output_recv is None:
            raise RuntimeError("Engine must be used as async context manager")

        # Create fresh dispatch completion event (Events can only be set once)
        self._dispatch_complete = anyio.Event()

        async with anyio.create_task_group() as tg:
            # Start all sources with channel cleanup
            for source in self._sources:
                tg.start_soon(self._run_source_with_cleanup, source, self._input_send.clone())

            # Start sink dispatcher
            tg.start_soon(self._dispatch_outputs)

            # Process input events
            async for event in self._input_recv:
                await self._handle_input_event(event)

                if exit_on_completion and self._is_idle():
                    break

            # Close output channel to signal end-of-stream to dispatcher.
            # This lets it drain remaining events before we cancel.
            if self._output_send:
                await self._output_send.aclose()

            # Wait for dispatcher to finish draining all buffered events.
            # The dispatcher sets _dispatch_complete when it exits (after the
            # async for loop ends due to channel closure).
            # Use a timeout to prevent infinite hang if dispatcher gets stuck.
            timed_out = True
            with anyio.move_on_after(5.0):
                await self._dispatch_complete.wait()
                timed_out = False
            if timed_out:
                _logger.warning(
                    "Dispatcher drain timed out after 5s — events may have been dropped"
                )

            # Cancel remaining tasks (sources and possibly stuck dispatcher)
            tg.cancel_scope.cancel()

        # Mark as completed to prevent re-use (channels are closed, state is inconsistent)
        # This applies regardless of exit_on_completion since channels are always closed
        self._run_completed = True

    async def _run_source_with_cleanup(
        self,
        source: EventSource,
        send: MemoryObjectSendStream[InputEvent],
    ) -> None:
        """Run a source and ensure its channel is closed on exit.

        This ensures the cloned send channel is properly closed whether the
        source returns normally, raises an exception, or is cancelled.
        """
        try:
            await source.run(send)
        finally:
            await send.aclose()

    async def _dispatch_outputs(self) -> None:
        """Dispatch output events to all sinks.

        Assumes run() has validated that channels are initialized.
        Errors in individual sinks are logged but don't stop event dispatch.
        Sets _dispatch_complete when finished draining events.
        """
        assert self._output_recv is not None  # Validated by run()
        try:
            async for event in self._output_recv:
                async with anyio.create_task_group() as tg:
                    for sink in self._sinks:
                        tg.start_soon(self._dispatch_to_sink, sink, event)
        finally:
            self._dispatch_complete.set()

    async def _dispatch_to_sink(self, sink: EventSink, event: OutputEvent) -> None:
        """Dispatch event to a single sink, catching errors."""
        try:
            await sink.handle(event)
        except Exception:
            _logger.exception("Error dispatching event to sink %s", sink)

    async def _handle_input_event(self, event: InputEvent) -> None:
        """Process a single input event."""
        match event["type"]:
            case "run_requested":
                await self._handle_run_requested(event)
            case "cancel_requested":
                await self._handle_cancel_requested()
            case "data_artifact_changed":
                await self._handle_data_artifact_changed(event)
            case "code_or_config_changed":
                await self._handle_code_or_config_changed(event)

    async def _handle_run_requested(self, event: RunRequested) -> None:
        """Handle a RunRequested event by executing stages."""
        # Store orchestration params for watch mode re-runs
        self._stored_no_commit = event["no_commit"]
        self._stored_no_cache = event["no_cache"]
        self._stored_on_error = event["on_error"]

        # Clear cancel event before starting new execution
        self._cancel_event = anyio.Event()  # Reset by creating new event

        # Reset stage indices for this run
        self._stage_indices.clear()

        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            # Require pipeline for execution
            self._require_pipeline()

            await self._orchestrate_execution(
                stages=event["stages"],
                force=event["force"],
                single_stage=event["single_stage"],
                parallel=event["parallel"],
                max_workers=event["max_workers"],
                no_commit=event["no_commit"],
                no_cache=event["no_cache"],
                on_error=event["on_error"],
                cache_dir=event["cache_dir"],
                allow_uncached_incremental=event["allow_uncached_incremental"],
                checkout_missing=event["checkout_missing"],
            )
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _is_idle(self) -> bool:
        """Check if engine is idle (no pending work)."""
        return self._state == EngineState.IDLE

    # =========================================================================
    # Pipeline Access
    # =========================================================================

    def _require_pipeline(self) -> Pipeline:
        """Get the pipeline, raising RuntimeError if not set."""
        if self._pipeline is None:
            raise RuntimeError(
                "Engine requires a Pipeline. Pass pipeline= to Engine() constructor."
            )
        return self._pipeline

    def _list_stages(self) -> list[str]:
        """List registered stage names from pipeline."""
        return self._require_pipeline().list_stages()

    def _get_stage(self, name: str) -> RegistryStageInfo:
        """Get stage info from pipeline."""
        return self._require_pipeline().get(name)

    def _get_all_stages(self) -> dict[str, RegistryStageInfo]:
        """Get all stages as a dict from pipeline."""
        pipeline = self._require_pipeline()
        return {name: pipeline.get(name) for name in pipeline.list_stages()}

    # =========================================================================
    # Stage State Management
    # =========================================================================

    async def _set_stage_state(self, stage: str, new_state: StageExecutionState) -> None:
        """Update stage execution state and emit event."""
        old_state = self._stage_states.get(stage, StageExecutionState.PENDING)
        is_new = stage not in self._stage_states
        self._stage_states[stage] = new_state
        if not is_new and old_state == new_state:
            return

        await self.emit(
            StageStateChanged(
                type="stage_state_changed",
                stage=stage,
                state=new_state,
                previous_state=old_state,
            )
        )

    def _get_stage_state(self, stage: str) -> StageExecutionState:
        """Get current execution state for a stage."""
        return self._stage_states.get(stage, StageExecutionState.PENDING)

    def _get_stage_index(self, stage_name: str) -> tuple[int, int]:
        """Get (1-based index, total count) for a stage."""
        stage_keys = list(self._stage_states.keys())
        total_stages = len(stage_keys)
        try:
            stage_index = stage_keys.index(stage_name) + 1
        except ValueError:
            _logger.warning("Stage %s not found in stage_states during index lookup", stage_name)
            return 0, total_stages
        return stage_index, total_stages

    # =========================================================================
    # Execution Orchestration
    # =========================================================================

    async def _orchestrate_execution(
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
        """Orchestrate parallel stage execution with the async event loop."""
        import datetime

        from pivot import run_history

        # Record start time for run history
        started_at = datetime.datetime.now(datetime.UTC).isoformat()

        if cache_dir is None:
            cache_dir = config.get_cache_dir()

        # Verify tracked files before building DAG
        project_root = project.get_project_root()
        executor_core.verify_tracked_files(project_root, checkout_missing=checkout_missing)

        # Build bipartite graph (single source of truth) with validation
        all_stages = self._get_all_stages()
        from pivot.storage import track

        tracked_files = track.discover_pvt_files(project_root)
        self._graph = engine_graph.build_graph(
            all_stages, validate=True, tracked_files=tracked_files
        )

        # Extract stage-only DAG for execution order
        stage_dag = engine_graph.get_stage_dag(self._graph)

        if stages:
            registered = set(stage_dag.nodes())
            unknown = [s for s in stages if s not in registered]
            if unknown:
                from pivot import exceptions

                raise exceptions.StageNotFoundError(unknown, available_stages=list(registered))

        execution_order = engine_graph.get_execution_order(
            stage_dag, stages, single_stage=single_stage
        )

        if not execution_order:
            return {}

        # Check for uncached IncrementalOut files that would be lost
        if not allow_uncached_incremental:
            uncached = executor_core.check_uncached_incremental_outputs(
                execution_order, self._get_all_stages()
            )
            if uncached:
                from pivot import exceptions

                files_list = "\n".join(f"  - {stage}: {path}" for stage, path in uncached)
                raise exceptions.UncachedIncrementalOutputError(
                    f"The following IncrementalOut files exist but are not in cache:\n{files_list}\n\n"
                    + "Running the pipeline will DELETE these files and they cannot be restored.\n"
                    + "To proceed anyway, use allow_uncached_incremental=True or back up these files first."
                )

        # Compute max workers
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

        # Ensure state directory exists
        state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize orchestration state
        await self._initialize_orchestration(execution_order, effective_max_workers, on_error)
        self._warn_single_stage_mutex_groups()

        # Create executor
        self._executor = executor_core.create_executor(effective_max_workers)

        # Create output queue via Manager so the proxy is picklable across loky workers.
        # Plain spawn_ctx.Queue() cannot be pickled for ProcessPoolExecutor.submit().
        spawn_ctx = mp.get_context("spawn")
        local_manager = spawn_ctx.Manager()
        output_queue: mp.Queue[OutputMessage] = local_manager.Queue()  # pyright: ignore[reportAssignmentType]

        # Track results, start times, and actual durations
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_start_times: dict[str, float] = {}
        stage_durations: dict[str, float] = {}

        # Get state DB path
        state_db_path = config.get_state_db_path()

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._drain_output_queue, output_queue)

                # Open StateDB to ensure database exists before workers start
                with state_mod.StateDB(state_db_path) as state_db:
                    # Start initial ready stages
                    await self._start_ready_stages(
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
                        # Snapshot futures keys before passing to thread to avoid
                        # "dictionary changed size during iteration" race condition.
                        # The main async loop can modify _futures while thread waits.
                        futures_snapshot = list(self._futures.keys())
                        # Use default argument to capture snapshot by value, not reference
                        done = await anyio.to_thread.run_sync(
                            lambda fs=futures_snapshot: self._wait_for_futures_snapshot(
                                fs, timeout=0.1
                            )
                        )

                        for future in done:
                            stage_name = self._futures.pop(future)
                            start_time = stage_start_times.get(stage_name, time.perf_counter())

                            try:
                                result = future.result()
                                duration_ms = await self._handle_stage_completion(
                                    stage_name, result, start_time
                                )
                                stage_durations[stage_name] = duration_ms

                                # Apply deferred writes for successful stages
                                if result["status"] == StageStatus.RAN and not no_commit:
                                    stage_info = self._get_stage(stage_name)
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
                                duration_ms = await self._handle_stage_completion(
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
                                and n in results
                                and results[n]["status"] == StageStatus.FAILED
                            ]
                            if failed:
                                self._stop_starting_new = True
                                for name, state in self._stage_states.items():
                                    if state in (
                                        StageExecutionState.READY,
                                        StageExecutionState.PENDING,
                                    ):
                                        await self._set_stage_state(
                                            name, StageExecutionState.BLOCKED
                                        )
                                    if (
                                        state == StageExecutionState.BLOCKED
                                        or self._get_stage_state(name)
                                        == StageExecutionState.BLOCKED
                                    ) and name not in results:
                                        await self._emit_skipped_stage(
                                            name, f"upstream '{failed[0]}' failed", results
                                        )

                        # Check cancellation
                        if self._cancel_event.is_set():
                            self._stop_starting_new = True
                            for name, state in self._stage_states.items():
                                if state in (
                                    StageExecutionState.READY,
                                    StageExecutionState.PENDING,
                                ):
                                    await self._set_stage_state(name, StageExecutionState.COMPLETED)
                                    await self._emit_skipped_stage(name, "cancelled", results)

                        # Start more stages if slots available
                        if not self._stop_starting_new:
                            await self._start_ready_stages(
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

                    # Diagnostic: log stages not in results after main loop
                    missing_by_state: dict[str, list[str]] = {}
                    for name, state in self._stage_states.items():
                        if name not in results:
                            missing_by_state.setdefault(state.name, []).append(name)
                    if missing_by_state:
                        for state_name, stages in missing_by_state.items():
                            _logger.debug(
                                "Stages not in results (state=%s): %s",
                                state_name,
                                stages[:10] if len(stages) > 10 else stages,
                            )

                    # Handle any blocked stages not yet processed
                    for name, state in self._stage_states.items():
                        if state == StageExecutionState.BLOCKED and name not in results:
                            failed_upstream = next(
                                (
                                    n
                                    for n, r in results.items()
                                    if r["status"] == StageStatus.FAILED
                                ),
                                "unknown",
                            )
                            await self._emit_skipped_stage(
                                name, f"upstream '{failed_upstream}' failed", results
                            )

                # Send sentinel to stop blocking drain thread
                output_queue.put(None)

        finally:
            # Ensure drain thread can exit even on exception path.
            # Suppress Full in case the queue is at capacity (e.g., sentinel
            # already enqueued on the happy path).
            with contextlib.suppress(OSError, BrokenPipeError, queue.Full):
                output_queue.put_nowait(None)
            self._executor = None
            # Manager shutdown can fail if the manager process died unexpectedly.
            # Must happen after sentinel send so drain thread exits first.
            with contextlib.suppress(OSError, BrokenPipeError):
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

    def _wait_for_futures_snapshot(
        self, futures: list[concurrent.futures.Future[StageResult]], timeout: float
    ) -> set[concurrent.futures.Future[StageResult]]:
        """Wait for futures with timeout. Called from thread.

        Takes a snapshot of futures to avoid race conditions with the main async
        loop that may modify self._futures concurrently.
        """
        if not futures:
            return set()
        done, _ = concurrent.futures.wait(
            futures,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        return done

    async def _drain_output_queue(
        self,
        output_queue: mp.Queue[OutputMessage],
    ) -> None:
        """Drain output messages from worker processes and emit LogLine events.

        Runs a blocking thread that calls queue.get() without polling.
        The thread exits when it receives a sentinel (None).
        Messages are forwarded into the async event loop via anyio.from_thread.run.
        """

        def _blocking_drain() -> None:
            while True:
                try:
                    # Use a timeout so the thread can exit even if the sentinel
                    # is never delivered (e.g., put() failed on exception path).
                    # The thread is abandoned on task-group cancellation, but a
                    # bounded get prevents it from blocking the interpreter at
                    # shutdown indefinitely.
                    msg = output_queue.get(timeout=5.0)
                except queue.Empty:
                    continue
                except (EOFError, OSError):
                    break

                if msg is None:
                    break

                try:
                    stage_name, line, is_stderr = msg
                except (TypeError, ValueError):
                    continue

                try:
                    anyio.from_thread.run(
                        self.emit,
                        LogLine(
                            type="log_line",
                            stage=stage_name,
                            line=line,
                            is_stderr=is_stderr,
                        ),
                    )
                except Exception:
                    # Event loop closed or cancelled -- stop draining
                    break

        await anyio.to_thread.run_sync(_blocking_drain, abandon_on_cancel=True)

    async def _initialize_orchestration(
        self,
        execution_order: list[str],
        max_workers: int,
        error_mode: OnError,
    ) -> None:
        """Initialize orchestration state for a new execution."""
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
            stage_info = self._get_stage(stage_name)

            # Upstream stages that must complete first
            if self._graph is not None:
                upstream = [
                    u
                    for u in engine_graph.get_upstream_stages(self._graph, stage_name)
                    if u in stages_set
                ]
            else:
                upstream = []
            self._stage_upstream_unfinished[stage_name] = set(upstream)

            # Downstream stages that depend on this one
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
            await self._set_stage_state(stage_name, initial_state)

    def _warn_single_stage_mutex_groups(self) -> None:
        """Warn if any mutex group contains only one stage (likely a typo)."""
        groups: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        for stage_name, mutexes in self._stage_mutex.items():
            for mutex in mutexes:
                groups[mutex].append(stage_name)

        for group, members in groups.items():
            if group == executor_core.EXCLUSIVE_MUTEX:
                continue
            if len(members) == 1 and group not in self._warned_mutex_groups:
                self._warned_mutex_groups.add(group)
                _logger.warning(f"Mutex group '{group}' only contains stage '{members[0]}'")

    def _can_start_stage(self, stage_name: str) -> bool:
        """Check if stage is eligible to start (ready and mutex available)."""
        if self._get_stage_state(stage_name) != StageExecutionState.READY:
            return False

        if self._stage_upstream_unfinished.get(stage_name):
            return False

        stage_mutexes = self._stage_mutex.get(stage_name, [])
        is_exclusive = executor_core.EXCLUSIVE_MUTEX in stage_mutexes

        for mutex in stage_mutexes:
            if mutex == executor_core.EXCLUSIVE_MUTEX:
                if self._mutex_counts[mutex] > 0 or len(self._futures) > 0:
                    return False
            elif self._mutex_counts[mutex] > 0:
                return False

        return is_exclusive or self._mutex_counts[executor_core.EXCLUSIVE_MUTEX] == 0

    async def _start_ready_stages(
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

        pipeline = self._require_pipeline()
        started = 0
        max_to_start = self._max_workers - len(self._futures)
        if max_to_start <= 0:
            return

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
            await self._set_stage_state(stage_name, StageExecutionState.PREPARING)

            # Get stage info and prepare worker info
            stage_info = self._get_stage(stage_name)
            worker_info = executor_core.prepare_worker_info(
                stage_info,
                pipeline._registry,  # pyright: ignore[reportPrivateUsage]
                overrides,
                checkout_modes,
                run_id,
                force,
                no_commit,
                no_cache,
                project_root,
                default_state_dir=state_dir,
            )

            # Submit to executor
            future = self._executor.submit(
                worker.execute_stage,
                stage_name,
                worker_info,
                cache_dir,
                output_queue,
            )
            self._futures[future] = stage_name

            # Record start time
            stage_start_times[stage_name] = time.perf_counter()

            # Transition to RUNNING and emit StageStarted
            await self._set_stage_state(stage_name, StageExecutionState.RUNNING)

            stage_index, total_stages = self._get_stage_index(stage_name)
            await self.emit(
                StageStarted(
                    type="stage_started",
                    stage=stage_name,
                    index=stage_index,
                    total=total_stages,
                )
            )

    async def _handle_stage_completion(
        self,
        stage_name: str,
        result: StageResult,
        start_time: float,
    ) -> float:
        """Handle a stage completing execution. Returns duration in milliseconds."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Transition to COMPLETED
        await self._set_stage_state(stage_name, StageExecutionState.COMPLETED)

        # Emit StageCompleted event
        stage_index, total_stages = self._get_stage_index(stage_name)
        await self.emit(
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
                if (
                    not unfinished
                    and self._get_stage_state(downstream_name) == StageExecutionState.PENDING
                ):
                    await self._set_stage_state(downstream_name, StageExecutionState.READY)

        # Handle failure cascading
        if result["status"] == StageStatus.FAILED:
            await self._cascade_failure(stage_name)

        # Process any deferred events for this stage
        await self._process_deferred_events(stage_name)

        return duration_ms

    async def _cascade_failure(self, failed_stage: str) -> None:
        """Mark downstream stages as blocked due to upstream failure.

        Since _stage_downstream already contains all transitive descendants (computed
        via get_downstream_stages which uses nx.descendants), we simply iterate through
        them once without recursion.
        """
        for downstream_name in self._stage_downstream.get(failed_stage, []):
            state = self._get_stage_state(downstream_name)
            if state in (StageExecutionState.PENDING, StageExecutionState.READY):
                await self._set_stage_state(downstream_name, StageExecutionState.BLOCKED)

    async def _emit_skipped_stage(
        self,
        stage_name: str,
        reason: str,
        results: dict[str, executor_core.ExecutionSummary],
    ) -> None:
        """Record and emit a skipped/blocked stage completion."""
        results[stage_name] = executor_core.ExecutionSummary(
            status=StageStatus.SKIPPED,
            reason=reason,
        )
        stage_index, total_stages = self._get_stage_index(stage_name)
        await self.emit(
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
            stage_lock = lock.StageLock(name, lock.get_stages_dir(state_dir))
            lock_data = stage_lock.read()

            if lock_data:
                input_hash = run_history.compute_input_hash_from_lock(lock_data)
            else:
                input_hash = "<no-lock>"

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

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _handle_cancel_requested(self) -> None:
        """Handle cancel by setting cancel event."""
        self._cancel_event.set()

    async def _handle_data_artifact_changed(self, event: DataArtifactChanged) -> None:
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
        await self._execute_affected_stages(affected)

    async def _handle_code_or_config_changed(self, _event: CodeOrConfigChanged) -> None:
        """Handle code/config changes by reloading registry and re-running."""
        _logger.info("Code/config changed - reloading pipeline")

        # Invalidate caches
        self._invalidate_caches()

        # Reload registry - returns old_stages on success, None on failure
        reload_result = self._reload_registry()

        if reload_result is None:
            _logger.error("Pipeline invalid - waiting for fix")
            return

        # Emit reload event
        old_stages, old_registry = reload_result
        await self._emit_reload_event(old_stages, old_registry)

        # Resolve external deps (e.g. sibling pipelines) before reading stages.
        # Reload bypasses Pipeline.build_dag() so we must resolve explicitly.
        self._require_pipeline().resolve_external_dependencies()
        all_stages = self._get_all_stages()
        self._graph = engine_graph.build_graph(all_stages)

        # Update watch paths if we have a FilesystemSource
        from pivot.engine.sources import FilesystemSource

        watch_paths = engine_graph.get_watch_paths(self._graph)
        for source in self._sources:
            if isinstance(source, FilesystemSource):
                source.set_watch_paths(watch_paths)

        # Re-run all stages
        stages = self._list_stages()

        if stages:
            await self._execute_affected_stages(stages)

    async def _execute_affected_stages(self, stages: list[str]) -> None:
        """Execute the specified stages."""
        self._cancel_event = anyio.Event()  # Reset by creating new event

        self._state = EngineState.ACTIVE
        await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            await self._orchestrate_execution(
                stages=stages,
                force=False,
                single_stage=False,
                parallel=True,
                max_workers=None,
                no_commit=self._stored_no_commit,
                no_cache=self._stored_no_cache,
                on_error=self._stored_on_error,
                cache_dir=None,
            )
        finally:
            self._state = EngineState.IDLE
            await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

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
        state = self._get_stage_state(producer)
        return StageExecutionState.PREPARING <= state < StageExecutionState.COMPLETED

    def _defer_event_for_stage(self, stage: str, event: InputEvent) -> None:
        """Defer an event until the stage completes."""
        self._deferred_events[stage].append(event)

    async def _process_deferred_events(self, stage: str) -> None:
        """Process any deferred events for a completed stage.

        Uses iterative approach to avoid recursion if processing defers more events.
        Errors in individual events are logged but don't block remaining events.
        """
        events = self._deferred_events.pop(stage, [])
        for event in events:
            try:
                await self._handle_input_event(event)
            except Exception:
                _logger.exception(f"Error processing deferred event for stage {stage}: {event}")

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

    def _get_affected_stages_for_paths(self, paths: list[pathlib.Path]) -> list[str]:
        """Get all stages affected by multiple path changes (including downstream)."""
        affected = set[str]()

        for path in paths:
            if self._should_filter_path(path):
                _logger.debug("Filtering event for %s (output of executing stage)", path)
                continue

            stage_affected = self._get_affected_stages_for_path(path)
            affected.update(stage_affected)

        return list(affected)

    # =========================================================================
    # Registry Reload
    # =========================================================================

    def _invalidate_caches(self) -> None:
        """Invalidate all caches when code changes."""
        linecache.clearcache()
        importlib.invalidate_caches()
        self._graph = None
        if self._pipeline is not None:
            self._pipeline.invalidate_dag_cache()

    def _reload_registry(
        self,
    ) -> tuple[dict[str, RegistryStageInfo], registry.StageRegistry | None] | None:
        """Reload the pipeline by re-importing pipeline definition.

        Returns old_stages and old_registry if reload succeeded, None if pipeline is invalid.
        The caller should emit the reload event using the returned old_stages.
        """
        old_pipeline = self._pipeline
        old_stages = old_pipeline.snapshot() if old_pipeline else {}
        old_registry = old_pipeline._registry if old_pipeline else None  # pyright: ignore[reportPrivateUsage]
        root = project.get_project_root()

        # Clear project modules from sys.modules
        self._clear_project_modules(root)

        # Use discovery to reload the pipeline
        try:
            from pivot import discovery

            new_pipeline = discovery.discover_pipeline(root)
            if new_pipeline is None:
                _logger.warning("No pipeline found during reload")
                return None

            self._pipeline = new_pipeline
            return old_stages, old_registry
        except Exception as e:
            _logger.warning(f"Pipeline invalid: {e}")
            # Restore old pipeline on failure
            self._pipeline = old_pipeline
            return None

    def _clear_project_modules(self, root: pathlib.Path) -> None:
        """Remove project modules from sys.modules."""
        to_remove = list[str]()

        for name, module in list(sys.modules.items()):
            if module is None:  # pyright: ignore[reportUnnecessaryComparison] - sys.modules values can be None
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            try:
                module_path = pathlib.Path(module_file)
                if module_path.is_relative_to(root):
                    to_remove.append(name)
            except (TypeError, ValueError):
                continue

        for name in to_remove:
            del sys.modules[name]

    async def _emit_reload_event(
        self,
        old_stages: dict[str, RegistryStageInfo],
        old_registry: registry.StageRegistry | None,
    ) -> None:
        """Emit PipelineReloaded event with diff information."""
        new_stage_names = self._list_stages()
        new_stages_set = set(new_stage_names)
        old_stage_names = set(old_stages.keys())

        added = sorted(new_stages_set - old_stage_names)
        removed = sorted(old_stage_names - new_stages_set)

        # Detect modified stages by comparing fingerprints
        modified = list[str]()
        pipeline = self._require_pipeline()
        new_registry = pipeline._registry  # pyright: ignore[reportPrivateUsage]
        for stage_name in sorted(old_stage_names & new_stages_set):
            had_error = False
            if old_registry is None:
                old_fp = None
                had_error = True
                _logger.warning(
                    "Fingerprinting failed for stage '%s' during reload (old registry unavailable)",
                    stage_name,
                )
            else:
                try:
                    old_fp = old_registry.ensure_fingerprint(stage_name)
                except exceptions.PivotError as exc:
                    old_fp = None
                    had_error = True
                    _logger.warning(
                        "Fingerprinting failed for stage '%s' during reload (old registry): %s",
                        stage_name,
                        exc,
                    )

            try:
                new_fp = new_registry.ensure_fingerprint(stage_name)
            except exceptions.PivotError as exc:
                new_fp = None
                had_error = True
                _logger.warning(
                    "Fingerprinting failed for stage '%s' during reload (new registry): %s",
                    stage_name,
                    exc,
                )

            if had_error or old_fp != new_fp:
                modified.append(stage_name)

        await self.emit(
            PipelineReloaded(
                type="pipeline_reloaded",
                stages=new_stage_names,
                stages_added=added,
                stages_removed=removed,
                stages_modified=modified,
                error=None,
            )
        )
