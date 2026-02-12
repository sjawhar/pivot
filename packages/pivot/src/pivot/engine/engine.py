from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import importlib
import itertools
import linecache
import logging
import pathlib
import queue
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

import anyio
import anyio.from_thread
import anyio.to_thread

from pivot import config, exceptions, fingerprint, parameters, project, registry
from pivot.engine import agent_rpc
from pivot.engine import graph as engine_graph
from pivot.engine import scheduler as engine_scheduler
from pivot.engine import watch as engine_watch
from pivot.engine import worker_pool as worker_pool_mod
from pivot.engine.types import (
    CodeOrConfigChanged,
    DataArtifactChanged,
    EngineDiagnostic,
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    InputEvent,
    LogLine,
    OutputChangeSummary,
    OutputEvent,
    PipelineReloaded,
    RunRequested,
    SinkState,
    SinkStateChanged,
    StageCompleted,
    StageExecutionState,
    StageStarted,
    StageStateChanged,
)
from pivot.executor import core as executor_core
from pivot.executor import worker
from pivot.storage import state as state_mod
from pivot.types import (
    DepEntry,
    HashInfo,
    LockData,
    OnError,
    OutputMessage,
    OutputMessageKind,
    StageExplanation,
    StageResult,
    StageStatus,
)

if TYPE_CHECKING:
    import multiprocessing as mp
    import threading

    import networkx as nx
    from anyio.abc import TaskGroup
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo
    from pivot.storage.cache import CheckoutMode

__all__ = ["Engine"]

_logger = logging.getLogger(__name__)

# Process-lifetime sequence counter for OutputEvent.seq monotonicity across Engine instances
_seq_counter = itertools.count(1)

# Channel buffer sizes for backpressure
_INPUT_BUFFER_SIZE = 32
_OUTPUT_BUFFER_SIZE = 64
_SINK_FAILURE_THRESHOLD = 5
_SINK_QUEUE_SIZE = 1024
_SINK_HANDLE_TIMEOUT_S = 5.0
_SINK_BACKOFF_BASE_S = 1.0
_SINK_BACKOFF_MAX_S = 1800.0
_SINK_SHUTDOWN_GRACE_S = 0.05


class _RunState(Enum):
    """Internal state machine for run lifecycle."""

    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"


@dataclass(slots=True)
class _SinkRuntime:
    sink: EventSink
    sink_id: str
    send: MemoryObjectSendStream[OutputEvent]
    recv: MemoryObjectReceiveStream[OutputEvent]
    failures: int = 0
    enabled: bool = True
    backoff_s: float = _SINK_BACKOFF_BASE_S
    disabled_until: float | None = None
    disabled_event: anyio.Event = field(default_factory=anyio.Event)


class Engine:
    """Async coordinator for pipeline execution using anyio.

    Thread safety: All state access occurs in the event loop task within run().
    Sources run in separate tasks but only send events to channels - they don't
    access engine state. This design provides implicit serialization through
    the channel, so no explicit lock is needed.
    """

    _pipeline: Pipeline | None
    _all_pipelines: bool
    _state: EngineState
    _sources: list[EventSource]
    _sinks: list[EventSink]
    _event_buffer: agent_rpc.EventBuffer
    _input_send: MemoryObjectSendStream[InputEvent] | None
    _input_recv: MemoryObjectReceiveStream[InputEvent] | None
    _output_send: MemoryObjectSendStream[OutputEvent] | None
    _output_recv: MemoryObjectReceiveStream[OutputEvent] | None
    _sink_runtimes: list[_SinkRuntime] | None

    # Orchestration state
    _graph: nx.DiGraph[str] | None
    _watch_coordinator: engine_watch.WatchCoordinator | None
    _scheduler: engine_scheduler.Scheduler
    _cancel_event: anyio.Event
    _stage_indices: dict[str, tuple[int, int]]
    _deferred_events: dict[str, list[InputEvent]]

    # Execution orchestration state
    _futures: dict[concurrent.futures.Future[StageResult], str]
    _worker_pool: worker_pool_mod.WorkerPool | None
    _warned_mutex_groups: set[str]

    # Run state machine (Task 10/11)
    _run_state: _RunState
    _run_task_group: TaskGroup | None
    _restart_pending: RunRequested | None
    _reload_pending: set[str] | None
    _idle_condition: anyio.Condition

    # Stored orchestration params (for watch mode re-runs)
    _stored_no_commit: bool
    _stored_on_error: OnError
    _stored_parallel: bool
    _stored_max_workers: int | None

    # Track whether run() has completed to prevent re-use
    _run_completed: bool

    # Event signaling dispatcher has finished draining
    _dispatch_complete: anyio.Event

    def __init__(self, *, pipeline: Pipeline | None = None, all_pipelines: bool = False) -> None:
        """Initialize the async engine in IDLE state."""
        self._pipeline = pipeline
        self._all_pipelines = all_pipelines
        self._state = EngineState.IDLE
        self._sources = list[EventSource]()
        self._sinks = list[EventSink]()
        self._event_buffer = agent_rpc.EventBuffer(max_events=1000)

        # Channels created on __aenter__
        self._input_send = None
        self._input_recv = None
        self._output_send = None
        self._output_recv = None
        self._sink_runtimes = None

        # Orchestration state
        self._graph = None
        self._watch_coordinator = None
        self._scheduler = engine_scheduler.Scheduler()
        self._cancel_event = anyio.Event()
        self._stage_indices = dict[str, tuple[int, int]]()
        self._deferred_events = collections.defaultdict(list)

        # Execution orchestration state
        self._futures = dict[concurrent.futures.Future[StageResult], str]()
        self._worker_pool = None
        self._warned_mutex_groups = set[str]()
        self._effective_max_workers: int = 1

        # Run state machine (Task 10/11)
        self._run_state = _RunState.IDLE
        self._run_task_group = None
        self._restart_pending = None
        self._reload_pending = None
        self._idle_condition = anyio.Condition()

        # Current run_id for stamping events (Task 12)
        self._current_run_id: str | None = None

        # Stored orchestration params (for watch mode re-runs)
        self._stored_no_commit = False
        self._stored_on_error = OnError.FAIL
        self._stored_parallel = True
        self._stored_max_workers = None

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
                seq = self._next_seq()
                event["seq"] = seq
                event["run_id"] = self._current_run_id or ""
                await self._output_send.send(event)

    def _next_seq(self) -> int:
        return next(_seq_counter)

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
            self._run_task_group = tg

            # Start all sources with channel cleanup
            for source in self._sources:
                tg.start_soon(self._run_source_with_cleanup, source, self._input_send.clone())

            # Start sink dispatcher
            tg.start_soon(self._dispatch_outputs)

            # Process input events.
            # In exit_on_completion mode, we also need to wait for background
            # execution tasks to finish. We interleave checking for new events
            # and checking for idle state using move_on_after to avoid blocking.
            if exit_on_completion:
                saw_run = False
                while True:
                    result: tuple[str, InputEvent | None] = await self._receive_input_or_idle(
                        wait_for_idle=saw_run
                    )  # type: ignore[reportAttributeAccessIssue] - pyright false positive
                    kind, event = result
                    if kind == "idle":
                        break
                    if kind == "eos":
                        if not self._is_idle():
                            await self._wait_for_idle()
                        break

                    if event is None:
                        continue
                    await self._handle_input_event(event)
                    if not self._is_idle():
                        saw_run = True

                    if saw_run and self._is_idle():
                        break
            else:
                async for event in self._input_recv:
                    await self._handle_input_event(event)

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

    async def _emit_sink_state(self, event: SinkStateChanged) -> None:
        event["seq"] = self._next_seq()
        event["run_id"] = self._current_run_id or ""
        if self._output_send:
            try:
                self._output_send.send_nowait(event)
                return
            except (anyio.ClosedResourceError, anyio.WouldBlock):
                pass
        if not self._sink_runtimes:
            return
        for runtime in self._sink_runtimes:
            if not runtime.enabled:
                continue
            try:
                runtime.send.send_nowait(event)
            except anyio.WouldBlock:
                continue
            except anyio.ClosedResourceError:
                # Best-effort direct delivery during shutdown; failures are logged
                # but not recorded (no disable tracking needed when shutting down)
                try:
                    with anyio.fail_after(_SINK_HANDLE_TIMEOUT_S):
                        await runtime.sink.handle(event)
                except Exception:
                    _logger.exception("Error dispatching sink state to sink %s", runtime.sink)

    async def _record_sink_failure(self, runtime: _SinkRuntime, reason: str) -> None:
        runtime.failures += 1
        if runtime.failures < _SINK_FAILURE_THRESHOLD:
            return
        if not runtime.enabled:
            return
        runtime.enabled = False
        runtime.disabled_until = anyio.current_time() + runtime.backoff_s
        await runtime.send.aclose()
        runtime.disabled_event.set()
        await self._emit_sink_state(
            SinkStateChanged(
                type="sink_state_changed",
                sink_id=runtime.sink_id,
                state=SinkState.DISABLED,
                reason=reason,
                failure_count=runtime.failures,
                backoff_s=runtime.backoff_s,
            )
        )

    async def _reenable_sink(self, runtime: _SinkRuntime, stop_event: anyio.Event) -> None:
        if runtime.disabled_until is None:
            return
        delay = max(0.0, runtime.disabled_until - anyio.current_time())
        if delay > 0:
            with anyio.move_on_after(delay):
                await stop_event.wait()
            if stop_event.is_set():
                return
        if stop_event.is_set():
            return
        send, recv = anyio.create_memory_object_stream[OutputEvent](_SINK_QUEUE_SIZE)
        runtime.send = send
        runtime.recv = recv
        runtime.enabled = True
        runtime.failures = 0
        runtime.backoff_s = _SINK_BACKOFF_BASE_S
        runtime.disabled_until = None
        await self._emit_sink_state(
            SinkStateChanged(
                type="sink_state_changed",
                sink_id=runtime.sink_id,
                state=SinkState.ENABLED,
                reason="backoff_elapsed",
                failure_count=runtime.failures,
                backoff_s=None,
            )
        )

    async def _supervise_sink_reenable(
        self,
        runtime: _SinkRuntime,
        tg: TaskGroup,
        stop_event: anyio.Event,
    ) -> None:
        while not stop_event.is_set():
            await runtime.disabled_event.wait()
            if stop_event.is_set():
                return
            if runtime.enabled or runtime.disabled_until is None:
                runtime.disabled_event = anyio.Event()
                continue
            await self._reenable_sink(runtime, stop_event)
            runtime.disabled_event = anyio.Event()
            if stop_event.is_set():
                return
            if runtime.enabled:
                tg.start_soon(self._run_sink_task, runtime)

    async def _dispatch_outputs(self) -> None:
        """Dispatch output events to all sinks via per-sink bounded queues.

        Each sink gets a dedicated long-lived task that reads from its own
        bounded queue (_SINK_QUEUE_SIZE items). This ensures:
        1. Strict per-sink ordering (events processed sequentially per sink)
        2. Slow sinks don't block fast sinks until their buffer fills
        3. Backpressure: engine records failures when any sink queue is full

        Assumes run() has validated that channels are initialized.
        Errors in individual sinks are logged but don't stop event dispatch.
        Sets _dispatch_complete when finished draining events.
        """
        assert self._output_recv is not None  # Validated by run()
        stop_event = anyio.Event()
        try:
            async with anyio.create_task_group() as tg:
                sink_runtimes: list[_SinkRuntime] = []
                for sink in self._sinks:
                    send, recv = anyio.create_memory_object_stream[OutputEvent](_SINK_QUEUE_SIZE)
                    runtime = _SinkRuntime(
                        sink=sink,
                        sink_id=type(sink).__name__,
                        send=send,
                        recv=recv,
                        backoff_s=_SINK_BACKOFF_BASE_S,
                    )
                    sink_runtimes.append(runtime)
                    tg.start_soon(self._run_sink_task, runtime)
                    tg.start_soon(self._supervise_sink_reenable, runtime, tg, stop_event)

                self._sink_runtimes = sink_runtimes

                async for event in self._output_recv:
                    for runtime in sink_runtimes:
                        if not runtime.enabled:
                            continue
                        try:
                            runtime.send.send_nowait(event)
                        except anyio.WouldBlock:
                            await self._record_sink_failure(runtime, reason="queue_full")
                        except anyio.ClosedResourceError:
                            continue

                grace_deadline = anyio.current_time() + _SINK_SHUTDOWN_GRACE_S
                if any(
                    not runtime.enabled
                    and runtime.disabled_until is not None
                    and runtime.disabled_until <= grace_deadline
                    for runtime in sink_runtimes
                ):
                    while any(not runtime.enabled for runtime in sink_runtimes):
                        if anyio.current_time() >= grace_deadline:
                            break
                        await anyio.sleep(0.01)

                stop_event.set()
                for runtime in sink_runtimes:
                    runtime.disabled_event.set()
                # Close all send channels when output stream ends
                for runtime in sink_runtimes:
                    if runtime.enabled:
                        await runtime.send.aclose()
        finally:
            self._sink_runtimes = None
            self._dispatch_complete.set()

    async def _run_sink_task(self, runtime: _SinkRuntime) -> None:
        """Process events for a single sink from its dedicated queue.

        Runs until the send side is closed. Errors in the sink are logged
        but don't stop processing subsequent events.
        """
        async for event in runtime.recv:
            if not runtime.enabled:
                continue
            try:
                with anyio.fail_after(_SINK_HANDLE_TIMEOUT_S):
                    await runtime.sink.handle(event)
            except TimeoutError:
                _logger.warning("Sink %s timed out handling event", runtime.sink)
                await self._record_sink_failure(runtime, reason="timeout")
            except Exception:
                _logger.exception("Error dispatching event to sink %s", runtime.sink)
                await self._record_sink_failure(runtime, reason="exception")

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
        """Handle a RunRequested event by spawning a non-blocking execution task.

        If already running, coalesces into a restart: cancels the current run
        and queues the new event for replay once the current run completes.
        """
        # Store orchestration params for watch mode re-runs
        self._stored_no_commit = event["no_commit"]
        self._stored_on_error = event["on_error"]
        self._stored_parallel = event["parallel"]
        self._stored_max_workers = event["max_workers"]

        if self._run_state == _RunState.RUNNING:
            # Coalesce: only the latest restart request survives
            self._restart_pending = event
            await self._handle_cancel_requested()
            return

        self._run_state = _RunState.RUNNING

        # Clear cancel event before starting new execution
        self._cancel_event = anyio.Event()  # Reset by creating new event

        # Reset stage indices for this run
        self._stage_indices.clear()

        # Generate run_id before ACTIVE event so it's stamped on all run events
        from pivot import run_history

        self._current_run_id = run_history.generate_run_id()

        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        assert self._run_task_group is not None
        self._run_task_group.start_soon(self._run_execution_task, event)

    async def _run_execution_task(self, event: RunRequested) -> None:
        """Execute a run in a background task within the run task group.

        Handles ACTIVE→IDLE transitions, restart coalescing, and deferred reloads.
        """
        try:
            self._require_pipeline()

            await self._orchestrate_execution(
                stages=event["stages"],
                force=event["force"],
                single_stage=event["single_stage"],
                parallel=event["parallel"],
                max_workers=event["max_workers"],
                no_commit=event["no_commit"],
                on_error=event["on_error"],
                cache_dir=event["cache_dir"],
                allow_uncached_incremental=event["allow_uncached_incremental"],
                checkout_missing=event["checkout_missing"],
            )
        finally:
            pending_reload = self._reload_pending is not None
            pending_restart = self._restart_pending is not None

            if pending_reload or pending_restart:
                self._run_state = _RunState.IDLE
                self._state = EngineState.IDLE
                self._current_run_id = None

                # Process deferred reload before restart (reload may change stage list)
                reload_pending = self._reload_pending
                if reload_pending is not None:
                    self._reload_pending = None
                    reload_paths = sorted(reload_pending)
                    await self._handle_code_or_config_changed(
                        CodeOrConfigChanged(type="code_or_config_changed", paths=reload_paths)
                    )

                # Replay coalesced restart request (clear before call for future-safety)
                if self._restart_pending is not None:
                    restart = self._restart_pending
                    self._restart_pending = None
                    await self._handle_run_requested(restart)

                if self._run_state == _RunState.IDLE and not self._has_pending_work():
                    async with self._idle_condition:
                        self._idle_condition.notify_all()
                    await self.emit(
                        EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE)
                    )
            else:
                self._run_state = _RunState.IDLE
                self._state = EngineState.IDLE
                async with self._idle_condition:
                    self._idle_condition.notify_all()
                await self.emit(
                    EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE)
                )
                self._current_run_id = None

    def _has_pending_work(self) -> bool:
        return self._restart_pending is not None or self._reload_pending is not None

    def _is_idle(self) -> bool:
        """Check if engine is idle (no pending work)."""
        return self._run_state == _RunState.IDLE and not self._has_pending_work()

    async def _wait_for_idle(self) -> None:
        async with self._idle_condition:
            while not self._is_idle():
                await self._idle_condition.wait()

    async def _receive_input_or_idle(self, *, wait_for_idle: bool) -> tuple[str, InputEvent | None]:
        if self._input_recv is None:
            raise RuntimeError("Input channel not initialized")
        input_recv = self._input_recv

        kind = "event"
        event: InputEvent | None = None

        async with anyio.create_task_group() as tg:

            async def _recv() -> None:
                nonlocal kind, event
                try:
                    event = await input_recv.receive()
                    kind = "event"
                except anyio.EndOfStream:
                    kind = "eos"
                finally:
                    tg.cancel_scope.cancel()

            async def _idle() -> None:
                nonlocal kind
                await self._wait_for_idle()
                kind = "idle"
                tg.cancel_scope.cancel()

            tg.start_soon(_recv)
            if wait_for_idle:
                tg.start_soon(_idle)

        return kind, event

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
        old_state, changed = self._scheduler.set_state(stage, new_state)
        if not changed:
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
        return self._scheduler.get_state(stage)

    def _get_stage_index(self, stage_name: str) -> tuple[int, int]:
        """Get (1-based index, total count) for a stage."""
        stage_keys = list(self._scheduler.stage_states.keys())
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
        default_state_dir = config.get_state_dir()
        run_id = self._current_run_id or run_history.generate_run_id()
        self._current_run_id = run_id  # May already be set by _handle_run_requested

        # Ensure state directory exists
        default_state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize orchestration state
        await self._initialize_orchestration(execution_order, effective_max_workers)
        self._warn_single_stage_mutex_groups()

        # Worker pool is created lazily on first worker dispatch (see _ensure_worker_pool).
        # If all stages skip in the coordinator, the pool is never started.

        # Track results, start times, and actual durations
        results: dict[str, executor_core.ExecutionSummary] = {}
        stage_start_times: dict[str, float] = {}
        stage_durations: dict[str, float] = {}

        # Per-stage StateDB cache: routes deferred writes to correct database
        state_dbs = dict[pathlib.Path, state_mod.StateDB]()

        def _get_state_db(stage_state_dir: pathlib.Path) -> state_mod.StateDB:
            """Get or open a StateDB for the given state_dir."""
            if stage_state_dir not in state_dbs:
                db_path = stage_state_dir / "state.db"
                state_dbs[stage_state_dir] = state_mod.StateDB(db_path)
            return state_dbs[stage_state_dir]

        try:
            # Ensure default StateDB exists before skip checks
            _get_state_db(default_state_dir)

            # Start initial ready stages (may skip all in coordinator, or lazily create pool)
            await self._start_ready_stages(
                cache_dir=cache_dir,
                overrides=overrides,
                checkout_modes=checkout_modes,
                force=force,
                no_commit=no_commit,
                stage_start_times=stage_start_times,
                run_id=run_id,
                project_root=project_root,
                state_dir=default_state_dir,
                results=results,
            )

            # Only start drain thread and main loop if workers were dispatched.
            # When all stages skip in the coordinator, the pool is never created.
            if self._worker_pool is not None:
                output_queue = self._worker_pool.output_queue()
                shutdown_event = self._worker_pool.shutdown_event()

                # Three-tier shutdown strategy for drain thread:
                # 1. Sentinel message (None) - primary happy-path signal
                # 2. shutdown_event - fallback when sentinel delivery fails or queue stays busy
                # 3. abandon_on_cancel + timeout - final backstop prevents indefinite blocking
                # All three are needed to handle different failure modes (PR #400 proved this).
                async with anyio.create_task_group() as tg:
                    tg.start_soon(self._drain_output_queue, output_queue, shutdown_event)
                    try:
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
                                        stage_name,
                                        result,
                                        start_time,
                                        default_state_dir,
                                        run_id=run_id,
                                    )
                                    stage_durations[stage_name] = duration_ms

                                    # Apply deferred writes for RAN and SKIPPED stages
                                    if (
                                        result["status"] in (StageStatus.RAN, StageStatus.SKIPPED)
                                        and not no_commit
                                    ):
                                        stage_info = self._get_stage(stage_name)
                                        output_paths = [str(out.path) for out in stage_info["outs"]]
                                        stage_state_dir = registry.get_stage_state_dir(
                                            stage_info, default_state_dir
                                        )
                                        stage_db = _get_state_db(stage_state_dir)
                                        executor_core.apply_deferred_writes(
                                            stage_name, output_paths, result, stage_db
                                        )

                                    # Record result
                                    results[stage_name] = executor_core.ExecutionSummary(
                                        status=result["status"],
                                        reason=result["reason"],
                                        input_hash=result["input_hash"],
                                    )

                                except Exception as e:
                                    _logger.exception("Stage %s failed with exception", stage_name)
                                    failed_result = StageResult(
                                        status=StageStatus.FAILED,
                                        reason=str(e),
                                        input_hash=None,
                                        output_lines=[],
                                    )
                                    duration_ms = await self._handle_stage_completion(
                                        stage_name,
                                        failed_result,
                                        start_time,
                                        default_state_dir,
                                        run_id=run_id,
                                    )
                                    stage_durations[stage_name] = duration_ms
                                    results[stage_name] = executor_core.ExecutionSummary(
                                        status=StageStatus.FAILED,
                                        reason=str(e),
                                        input_hash=None,
                                    )

                            # Check error mode
                            if on_error == OnError.FAIL:
                                has_failed = any(
                                    s == StageExecutionState.COMPLETED
                                    and n in results
                                    and results[n]["status"] == StageStatus.FAILED
                                    for n, s in self._scheduler.stage_states.items()
                                )
                                if has_failed:
                                    first_failed = next(
                                        n
                                        for n, s in self._scheduler.stage_states.items()
                                        if s == StageExecutionState.COMPLETED
                                        and n in results
                                        and results[n]["status"] == StageStatus.FAILED
                                    )
                                    blocked = self._scheduler.apply_fail_fast()
                                    for name, old_state in blocked:
                                        await self.emit(
                                            StageStateChanged(
                                                type="stage_state_changed",
                                                stage=name,
                                                state=StageExecutionState.BLOCKED,
                                                previous_state=old_state,
                                            )
                                        )
                                    # Emit skipped for all blocked stages not yet in results
                                    for name, state in self._scheduler.stage_states.items():
                                        if (
                                            state == StageExecutionState.BLOCKED
                                            and name not in results
                                        ):
                                            await self._emit_skipped_stage(
                                                name,
                                                f"upstream '{first_failed}' failed",
                                                results,
                                                run_id=run_id,
                                            )

                            # Check cancellation
                            if self._cancel_event.is_set():
                                cancelled = self._scheduler.apply_cancel()
                                for name, old_state in cancelled:
                                    await self.emit(
                                        StageStateChanged(
                                            type="stage_state_changed",
                                            stage=name,
                                            state=StageExecutionState.COMPLETED,
                                            previous_state=old_state,
                                        )
                                    )
                                    await self._emit_skipped_stage(
                                        name, "cancelled", results, run_id=run_id
                                    )

                            # Start more stages if slots available
                            if not self._scheduler.stop_starting_new:
                                await self._start_ready_stages(
                                    cache_dir=cache_dir,
                                    overrides=overrides,
                                    checkout_modes=checkout_modes,
                                    force=force,
                                    no_commit=no_commit,
                                    stage_start_times=stage_start_times,
                                    run_id=run_id,
                                    project_root=project_root,
                                    state_dir=default_state_dir,
                                    results=results,
                                )

                        # Diagnostic: log stages not in results after main loop
                        missing_by_state: dict[str, list[str]] = {}
                        for name, state in self._scheduler.stage_states.items():
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
                        for name, state in self._scheduler.stage_states.items():
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
                                    name,
                                    f"upstream '{failed_upstream}' failed",
                                    results,
                                    run_id=run_id,
                                )

                        # Send sentinel to stop blocking drain thread without blocking event loop.
                        # Not redundant: this is the primary shutdown signal for the drain thread;
                        # the finally sentinel below is the fallback for exception paths.
                        with contextlib.suppress(queue.Full, OSError, BrokenPipeError):
                            output_queue.put_nowait(None)

                    finally:
                        # Ensure drain thread can exit even on exception path.
                        shutdown_event.set()
                        try:
                            output_queue.put_nowait(None)
                        except (OSError, queue.Full) as exc:
                            _logger.debug(
                                "Sentinel send failed (will rely on shutdown_event): %s", exc
                            )

        finally:
            # Close StateDBs first (lightweight), then shut down worker pool.
            with anyio.CancelScope(shield=True):
                for db in state_dbs.values():
                    db.close()
                state_dbs.clear()

            # WorkerPool shutdown handles executor + manager cleanup.
            # Must happen after sentinel send so drain thread exits first.
            # Wrapped in outer finally to ensure it always executes, even if task group raises.
            if self._worker_pool is not None:
                worker_pool = self._worker_pool
                self._worker_pool = None
                try:
                    with anyio.CancelScope(shield=True):
                        await anyio.to_thread.run_sync(worker_pool.shutdown)
                except (OSError, BrokenPipeError) as exc:
                    _logger.debug("WorkerPool shutdown failed (may already be dead): %s", exc)

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
        shutdown_event: threading.Event,
    ) -> None:
        """Drain output messages from worker processes and emit LogLine events.

        Runs a blocking thread that calls queue.get() without polling.
        The thread exits when it receives a sentinel (None).
        Messages are forwarded into the async event loop via anyio.from_thread.run.
        """

        def _blocking_drain() -> None:
            saw_shutdown = False
            while True:
                # If shutdown requested, continue draining until queue is idle.
                if shutdown_event.is_set():
                    saw_shutdown = True

                try:
                    # Use a 1.0s timeout so the thread can exit even if the
                    # sentinel is never delivered (e.g., put() failed on
                    # exception path). When shutting down, tighten the timeout
                    # to drain any remaining messages without blocking forever.
                    timeout = 0.2 if saw_shutdown else 1.0
                    msg = output_queue.get(timeout=timeout)
                except queue.Empty:
                    if saw_shutdown:
                        break
                    continue
                except (EOFError, OSError):
                    break

                if msg is None:
                    saw_shutdown = True
                    continue

                if msg["kind"] == OutputMessageKind.STATE:
                    try:
                        if msg["state"] == "waiting_on_lock":
                            new_state = StageExecutionState.WAITING_ON_LOCK
                        elif msg["state"] == "running":
                            new_state = StageExecutionState.RUNNING
                        else:
                            continue
                        # Guard: skip if stage already at or past this state.
                        # Late-arriving messages from the output queue must not
                        # rewind a stage that has already completed.
                        current = self._scheduler.get_state(msg["stage"])
                        if current >= new_state:
                            continue
                        anyio.from_thread.run(self._set_stage_state, msg["stage"], new_state)
                    except Exception:
                        break
                    continue

                try:
                    anyio.from_thread.run(
                        self.emit,
                        LogLine(
                            type="log_line",
                            stage=msg["stage"],
                            line=msg["line"],
                            is_stderr=msg["is_stderr"],
                        ),
                    )
                except Exception:
                    break

        await anyio.to_thread.run_sync(_blocking_drain, abandon_on_cancel=True)

    async def _initialize_orchestration(
        self,
        execution_order: list[str],
        max_workers: int,
    ) -> None:
        """Initialize orchestration state for a new execution."""
        self._futures.clear()
        self._deferred_events.clear()

        # Build stage_mutex map from pipeline registry
        stage_mutex = {name: self._get_stage(name)["mutex"] for name in execution_order}

        self._scheduler.initialize(
            execution_order,
            self._graph,
            stage_mutex=stage_mutex,
        )
        self._effective_max_workers = max_workers

        # Emit initial state events for all stages
        for stage_name, initial_state in self._scheduler.stage_states.items():
            await self.emit(
                StageStateChanged(
                    type="stage_state_changed",
                    stage=stage_name,
                    state=initial_state,
                    previous_state=StageExecutionState.PENDING,
                )
            )

    def _warn_single_stage_mutex_groups(self) -> None:
        """Warn if any mutex group contains only one stage (likely a typo)."""
        groups: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        for stage_name, mutexes in self._scheduler.stage_mutex.items():
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
        return self._scheduler.can_start(stage_name, running_count=len(self._futures))

    def _ensure_worker_pool(self) -> worker_pool_mod.WorkerPool:
        """Lazily create and start the worker pool on first worker dispatch."""
        if self._worker_pool is None:
            self._worker_pool = worker_pool_mod.WorkerPool()
            self._worker_pool.start(max_workers=self._effective_max_workers)
        return self._worker_pool

    async def _start_ready_stages(
        self,
        cache_dir: pathlib.Path,
        overrides: parameters.ParamsOverrides,
        checkout_modes: list[CheckoutMode],
        force: bool,
        no_commit: bool,
        stage_start_times: dict[str, float],
        run_id: str,
        project_root: pathlib.Path,
        state_dir: pathlib.Path,
        results: dict[str, executor_core.ExecutionSummary],
    ) -> None:
        """Start all eligible stages up to max_workers."""
        if self._scheduler.stop_starting_new:
            return

        pipeline = self._require_pipeline()
        started = 0
        max_to_start = self._effective_max_workers - len(self._futures)
        if max_to_start <= 0:
            return

        # Shared read-only StateDBs for skip checks (one per state_dir).
        # Closed after the loop to avoid repeated LMDB open/close per stage.
        skip_state_dbs = dict[pathlib.Path, state_mod.StateDB]()
        try:
            for stage_name in list(self._scheduler.stage_states.keys()):
                if started >= max_to_start:
                    break

                if self._cancel_event.is_set():
                    break

                if not self._can_start_stage(stage_name):
                    continue

                # Acquire mutex locks (must happen before next can_start check)
                self._scheduler.acquire_mutexes(stage_name)

                # Transition to PREPARING
                await self._set_stage_state(stage_name, StageExecutionState.PREPARING)

                stage_info = self._get_stage(stage_name)

                try:
                    skipped = await self._try_skip_in_coordinator(
                        stage_name,
                        stage_info,
                        overrides,
                        checkout_modes,
                        force,
                        cache_dir,
                        state_dir,
                        project_root,
                        results,
                        run_id,
                        skip_state_dbs=skip_state_dbs,
                    )
                except Exception:
                    _logger.debug(
                        "Coordinator skip check failed for %s, dispatching to worker",
                        stage_name,
                        exc_info=True,
                    )
                    skipped = False
                if skipped:
                    continue

                # Stage must run — lazily create pool on first dispatch
                started += 1
                pool = self._ensure_worker_pool()
                worker_info = executor_core.prepare_worker_info(
                    stage_info,
                    pipeline._registry,  # pyright: ignore[reportPrivateUsage]
                    overrides,
                    checkout_modes,
                    run_id,
                    force,
                    no_commit,
                    project_root,
                    default_state_dir=state_dir,
                )

                future = pool.submit(
                    worker.execute_stage,
                    stage_name,
                    worker_info,
                    cache_dir,
                    pool.output_queue(),
                )
                self._futures[future] = stage_name

                stage_start_times[stage_name] = time.perf_counter()

                await self._set_stage_state(stage_name, StageExecutionState.RUNNING)

                explanation = await self._compute_explanation(
                    stage_name,
                    worker_info,
                    stage_info,
                    overrides,
                    force,
                )

                stage_index, total_stages = self._get_stage_index(stage_name)
                await self.emit(
                    StageStarted(
                        type="stage_started",
                        stage=stage_name,
                        index=stage_index,
                        total=total_stages,
                        run_id=run_id,
                        explanation=explanation,
                    )
                )
        finally:
            for db in skip_state_dbs.values():
                db.close()

    async def _try_skip_in_coordinator(
        self,
        stage_name: str,
        stage_info: RegistryStageInfo,
        overrides: parameters.ParamsOverrides,
        checkout_modes: list[CheckoutMode],
        force: bool,
        cache_dir: pathlib.Path,
        state_dir: pathlib.Path,
        project_root: pathlib.Path,
        results: dict[str, executor_core.ExecutionSummary],
        run_id: str,
        skip_state_dbs: dict[pathlib.Path, state_mod.StateDB] | None = None,
    ) -> bool:
        from pivot.storage import artifact_lock, lock

        if force:
            return False

        stage_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
        files_cache_dir = cache_dir / "files"

        def _read_lock() -> LockData | None:
            stages_dir = lock.get_stages_dir(stage_state_dir)
            stage_lock = lock.StageLock(stage_name, stages_dir)
            return stage_lock.read()

        lock_data = await anyio.to_thread.run_sync(_read_lock)
        if lock_data is None:
            return False

        def _ensure_fingerprint() -> dict[str, str]:
            pipeline = self._require_pipeline()
            stage_registry = pipeline._registry  # pyright: ignore[reportPrivateUsage]
            return stage_registry.ensure_fingerprint(stage_name)

        current_fingerprint = await anyio.to_thread.run_sync(_ensure_fingerprint)

        try:
            current_params = parameters.get_effective_params(
                stage_info["params"], stage_name, overrides
            )
        except Exception:
            return False

        out_paths = stage_info["outs_paths"]
        out_specs = [
            (path, out.cache) for out, path in zip(stage_info["outs"], out_paths, strict=True)
        ]

        # Acquire artifact locks (READ on deps, WRITE on outs) to prevent
        # concurrent pivot runs from racing during skip check + output restore.
        await self._set_stage_state(stage_name, StageExecutionState.WAITING_ON_LOCK)

        def _acquire_artifact_locks() -> artifact_lock.LockHandle:
            lock_requests = artifact_lock.expand_lock_requests(
                stage_info["deps_paths"], stage_info["outs"], project_root
            )
            lock_service = artifact_lock.LocalFlockLockService(stage_state_dir / "locks")
            return lock_service.acquire_many(lock_requests)

        lock_handle = await anyio.to_thread.run_sync(_acquire_artifact_locks)
        await self._set_stage_state(stage_name, StageExecutionState.PREPARING)

        try:
            return await self._try_skip_with_locks(
                stage_name,
                stage_info,
                lock_data,
                current_fingerprint,
                current_params,
                out_paths,
                out_specs,
                checkout_modes,
                files_cache_dir,
                stage_state_dir,
                state_dir,
                results,
                run_id,
                skip_state_dbs,
            )
        finally:
            lock_handle.release()

    async def _try_skip_with_locks(
        self,
        stage_name: str,
        stage_info: RegistryStageInfo,
        lock_data: LockData,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        out_paths: list[str],
        out_specs: list[tuple[str, bool]],
        checkout_modes: list[CheckoutMode],
        files_cache_dir: pathlib.Path,
        stage_state_dir: pathlib.Path,
        state_dir: pathlib.Path,
        results: dict[str, executor_core.ExecutionSummary],
        run_id: str,
        skip_state_dbs: dict[pathlib.Path, state_mod.StateDB] | None,
    ) -> bool:
        """Perform skip detection while artifact locks are held."""
        from pivot import run_history
        from pivot import skip as skip_mod

        def _get_skip_state_db() -> state_mod.StateDB:
            state_db_path = stage_state_dir / "state.db"
            if skip_state_dbs is not None:
                if state_db_path not in skip_state_dbs:
                    skip_state_dbs[state_db_path] = state_mod.StateDB(state_db_path, readonly=True)
                return skip_state_dbs[state_db_path]
            return state_mod.StateDB(state_db_path, readonly=True)

        def _try_generation_skip() -> tuple[bool, str | None]:
            state_db = _get_skip_state_db()
            try:
                can_skip = worker.can_skip_via_generation(
                    stage_name=stage_name,
                    fingerprint=current_fingerprint,
                    deps=stage_info["deps_paths"],
                    outs_paths=out_paths,
                    current_params=current_params,
                    lock_data=lock_data,
                    state_db=state_db,
                    verify_files=True,
                )
                if not can_skip:
                    return False, None
                deps_list = [
                    DepEntry(path=path, hash=info["hash"])
                    for path, info in lock_data["dep_hashes"].items()
                ]
                input_hash = run_history.compute_input_hash(
                    current_fingerprint, current_params, deps_list, out_specs
                )
                restored = worker.restore_outputs_from_cache(
                    stage_info["outs"],
                    lock_data,
                    files_cache_dir,
                    checkout_modes,
                    state_db=state_db,
                    state_dir=stage_state_dir,
                )
                if not restored:
                    return False, None
                return True, input_hash
            finally:
                if skip_state_dbs is None:
                    state_db.close()

        skipped, generation_input_hash = await anyio.to_thread.run_sync(_try_generation_skip)
        if skipped:
            await self._record_skipped_stage(
                stage_name,
                "unchanged (generation)",
                generation_input_hash,
                state_dir,
                results,
                run_id=run_id,
            )
            return True

        def _hash_deps() -> tuple[
            dict[str, HashInfo], list[str], list[str], list[tuple[str, int, int, int, str]]
        ]:
            state_db = _get_skip_state_db()
            try:
                return worker.hash_dependencies(stage_info["deps_paths"], state_db)
            finally:
                if skip_state_dbs is None:
                    state_db.close()

        dep_hashes, missing, unreadable, _file_hash_entries = await anyio.to_thread.run_sync(
            _hash_deps
        )
        if missing or unreadable:
            return False

        decision = skip_mod.check_stage(
            lock_data,
            current_fingerprint,
            current_params,
            dep_hashes,
            out_paths,
            explain=False,
            force=False,
        )
        if decision["changed"]:
            return False

        deps_list = [DepEntry(path=path, hash=info["hash"]) for path, info in dep_hashes.items()]
        input_hash = run_history.compute_input_hash(
            current_fingerprint,
            current_params,
            deps_list,
            out_specs,
        )

        def _restore_outputs() -> bool:
            state_db = _get_skip_state_db()
            try:
                return worker.restore_outputs_from_cache(
                    stage_info["outs"],
                    lock_data,
                    files_cache_dir,
                    checkout_modes,
                    state_db=state_db,
                    state_dir=stage_state_dir,
                )
            finally:
                if skip_state_dbs is None:
                    state_db.close()

        restored = await anyio.to_thread.run_sync(_restore_outputs)
        if not restored:
            return False

        await self._record_skipped_stage(
            stage_name,
            "unchanged",
            input_hash,
            state_dir,
            results,
            run_id=run_id,
        )
        return True

    async def _record_skipped_stage(
        self,
        stage_name: str,
        reason: str,
        input_hash: str | None,
        state_dir: pathlib.Path,
        results: dict[str, executor_core.ExecutionSummary],
        *,
        run_id: str,
    ) -> None:
        result = StageResult(
            status=StageStatus.SKIPPED,
            reason=reason,
            input_hash=input_hash,
            output_lines=[],
        )
        start_time = time.perf_counter()
        _ = await self._handle_stage_completion(
            stage_name,
            result,
            start_time,
            state_dir,
            run_id=run_id,
        )

        results[stage_name] = executor_core.ExecutionSummary(
            status=result["status"],
            reason=result["reason"],
            input_hash=result["input_hash"],
        )

    async def _compute_explanation(
        self,
        stage_name: str,
        worker_info: worker.WorkerStageInfo,
        stage_info: RegistryStageInfo,
        overrides: parameters.ParamsOverrides,
        force: bool,
    ) -> StageExplanation | None:
        """Compute explanation for why a stage will run, in a worker thread.

        Returns None if computation fails (explanation should never block execution).
        """
        from pivot import explain as explain_mod

        def _compute() -> StageExplanation:
            return explain_mod.get_stage_explanation(
                stage_name=stage_name,
                fingerprint=worker_info["fingerprint"],
                deps=worker_info["deps"],
                outs_paths=stage_info["outs_paths"],
                params_instance=worker_info["params"],
                overrides=overrides,
                state_dir=worker_info["state_dir"],
                force=force,
            )

        try:
            return await anyio.to_thread.run_sync(_compute)
        except Exception:
            _logger.warning("Failed to compute explanation for %s", stage_name, exc_info=True)
            return None

    async def _compute_output_summary(
        self,
        stage_name: str,
        state_dir: pathlib.Path,
    ) -> list[OutputChangeSummary] | None:
        """Compute output summary from the stage's lock file after execution.

        Reads the lock file (just written by the worker) to get output hashes.
        Returns None if computation fails (should never block event emission).
        """
        from pivot import outputs
        from pivot.storage import lock

        def _compute() -> list[OutputChangeSummary]:
            stage_info = self._get_stage(stage_name)
            stage_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
            stages_dir = lock.get_stages_dir(stage_state_dir)
            stage_lock = lock.StageLock(stage_name, stages_dir)
            lock_data = stage_lock.read()

            # Build path -> output_type map from registry
            outs = stage_info["outs"]
            outs_paths = stage_info["outs_paths"]
            path_to_type = dict[str, str]()
            for out, path in zip(outs, outs_paths, strict=True):
                if isinstance(out, outputs.Metric):
                    path_to_type[path] = "metric"
                elif isinstance(out, outputs.Plot):
                    path_to_type[path] = "plot"
                else:
                    path_to_type[path] = "out"

            summary = list[OutputChangeSummary]()
            # Get new hashes from lock file
            new_hashes = dict[str, str | None]()
            if lock_data is not None and "output_hashes" in lock_data:
                for path, hash_info in lock_data["output_hashes"].items():
                    new_hashes[path] = hash_info["hash"]

            for path in outs_paths:
                new_hash = new_hashes.get(path)
                output_type = path_to_type.get(path, "out")
                # change_type is None because we don't have old hashes at this point
                summary.append(
                    OutputChangeSummary(
                        path=path,
                        change_type=None,
                        output_type=output_type,
                        old_hash=None,
                        new_hash=new_hash,
                    )
                )

            return summary

        try:
            return await anyio.to_thread.run_sync(_compute)
        except Exception:
            _logger.debug("Failed to compute output summary for %s", stage_name, exc_info=True)
            return None

    async def _handle_stage_completion(
        self,
        stage_name: str,
        result: StageResult,
        start_time: float,
        state_dir: pathlib.Path | None = None,
        run_id: str = "",
    ) -> float:
        """Handle a stage completing execution. Returns duration in milliseconds."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Transition to COMPLETED
        await self._set_stage_state(stage_name, StageExecutionState.COMPLETED)

        # Compute output summary for stages that ran (not failed)
        output_summary: list[OutputChangeSummary] | None = None
        if result["status"] != StageStatus.FAILED and state_dir is not None:
            output_summary = await self._compute_output_summary(stage_name, state_dir)

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
                run_id=run_id,
                input_hash=result["input_hash"],
                output_summary=output_summary,
            )
        )

        # Release mutex locks — guard against underflow to prevent exception
        # propagating into _orchestrate_execution's try/except, which would call
        # _handle_stage_completion again and corrupt scheduler state.
        try:
            self._scheduler.release_mutexes(stage_name)
        except ValueError:
            _logger.error("Mutex underflow releasing locks for stage %s", stage_name)

        failed = result["status"] == StageStatus.FAILED
        newly_ready, newly_blocked = self._scheduler.on_stage_completed(stage_name, failed)

        # Emit events for newly ready stages (PENDING → READY)
        for ready_name in newly_ready:
            await self.emit(
                StageStateChanged(
                    type="stage_state_changed",
                    stage=ready_name,
                    state=StageExecutionState.READY,
                    previous_state=StageExecutionState.PENDING,
                )
            )

        # Emit events for newly blocked stages
        for blocked_name, old_state in newly_blocked:
            await self.emit(
                StageStateChanged(
                    type="stage_state_changed",
                    stage=blocked_name,
                    state=StageExecutionState.BLOCKED,
                    previous_state=old_state,
                )
            )

        # Process any deferred events for this stage
        await self._process_deferred_events(stage_name)

        return duration_ms

    async def _emit_skipped_stage(
        self,
        stage_name: str,
        reason: str,
        results: dict[str, executor_core.ExecutionSummary],
        run_id: str = "",
    ) -> None:
        """Record and emit a skipped/blocked stage completion."""
        results[stage_name] = executor_core.ExecutionSummary(
            status=StageStatus.SKIPPED,
            reason=reason,
            input_hash=None,
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
                run_id=run_id,
                input_hash=None,
                output_summary=None,
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

        stages_records = dict[str, run_history.StageRunRecord]()
        for name, summary in results.items():
            input_hash = summary["input_hash"]
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
        """Handle cancel with explicit state machine.

        IDLE → no-op (nothing to cancel).
        RUNNING → transition to CANCELLING, set cancel event, stop accepting.
        CANCELLING → no-op (already cancelling, avoid duplicate work).
        """
        if self._run_state == _RunState.CANCELLING:
            return
        if self._run_state == _RunState.IDLE:
            return
        self._run_state = _RunState.CANCELLING
        self._cancel_event.set()
        if self._worker_pool is not None:
            self._worker_pool.stop_accepting()

    async def _handle_data_artifact_changed(self, event: DataArtifactChanged) -> None:
        """Handle data artifact changes by running affected stages."""
        paths = [pathlib.Path(p) for p in event["paths"]]

        # Filter out paths that are outputs of executing stages
        filtered_paths = list[pathlib.Path]()
        deferred_paths = list[tuple[str, pathlib.Path]]()

        for path in paths:
            if self._should_filter_path(path):
                coordinator = self._get_watch_coordinator()
                producer = coordinator.get_producer(path) if coordinator else None
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

    async def _handle_code_or_config_changed(self, event: CodeOrConfigChanged) -> None:
        """Handle code/config changes by reloading registry and re-running.

        If a run is active, defers the reload until the current run completes.
        """
        if self._run_state in (_RunState.RUNNING, _RunState.CANCELLING):
            _logger.info("Code/config changed while running - deferring reload")
            if self._reload_pending is None:
                self._reload_pending = set(event["paths"])
            else:
                self._reload_pending.update(event["paths"])
            return

        _logger.info("Code/config changed - reloading pipeline")

        # Invalidate caches
        self._invalidate_caches()

        # Invalidate manifest cache entries for changed source files
        fingerprint.invalidate_manifests_for_paths(event["paths"])

        # Reload registry - returns old_stages on success, None on failure
        reload_result = self._reload_registry()

        if reload_result is None:
            _logger.error("Pipeline invalid - waiting for fix")
            return

        # Emit reload event
        old_stages, old_registry = reload_result
        await self._emit_reload_event(old_stages, old_registry)

        # Flush manifest cache so cached manifests survive the reload
        fingerprint.flush_manifest_cache()

        # Resolve external deps (e.g. sibling pipelines) before reading stages.
        # Reload bypasses Pipeline.build_dag() so we must resolve explicitly.
        self._require_pipeline().resolve_external_dependencies()
        all_stages = self._get_all_stages()
        self._graph = engine_graph.build_graph(all_stages)

        # Update watch paths if we have a FilesystemSource
        from pivot.engine.sources import FilesystemSource

        watch_paths = engine_graph.get_watch_paths(self._graph)

        # In --all mode, also watch pipeline config directories so creating/removing
        # pipeline.py or pivot.yaml triggers reload.
        if self._all_pipelines:
            from pivot import discovery

            config_paths = discovery.glob_all_pipelines(project.get_project_root())
            for config_path in config_paths:
                watch_paths.append(config_path.parent)

        for source in self._sources:
            if isinstance(source, FilesystemSource):
                source.set_watch_paths(watch_paths)

        # Re-run all stages
        stages = self._list_stages()

        if stages:
            # Restart worker pool so workers pick up reloaded code.
            # Run in thread to avoid blocking the event loop during process kill/spawn.
            # Catch errors so a failed restart doesn't kill the watch session;
            # create_executor() in _orchestrate_execution will retry pool creation.
            coordinator = self._get_watch_coordinator()
            if coordinator is not None and coordinator.should_restart_workers(
                parallel=self._stored_parallel
            ):
                n_stages = len(stages)
                stored_max = self._stored_max_workers
                try:
                    await anyio.to_thread.run_sync(
                        lambda: executor_core.restart_workers(n_stages, stored_max)
                    )
                    _logger.info("Worker pool restarted for code reload (%d stages)", n_stages)
                except Exception:
                    _logger.warning(
                        "Failed to restart worker pool - continuing with existing workers",
                        exc_info=True,
                    )

            await self._execute_affected_stages(stages)

    async def _execute_affected_stages(self, stages: list[str]) -> None:
        """Execute the specified stages via non-blocking run handling.

        Constructs a synthetic RunRequested event from stored params and delegates
        to _handle_run_requested, which manages the run state machine.
        """
        event = RunRequested(
            type="run_requested",
            stages=stages,
            force=False,
            reason="watch:affected",
            single_stage=False,
            parallel=self._stored_parallel,
            max_workers=self._stored_max_workers,
            no_commit=self._stored_no_commit,
            on_error=self._stored_on_error,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )
        await self._handle_run_requested(event)

    def _get_watch_coordinator(self) -> engine_watch.WatchCoordinator | None:
        """Lazily create/update WatchCoordinator from current graph.

        This ensures existing tests that set engine._graph directly still work
        without needing to also set up the coordinator.
        """
        if self._graph is None:
            return None
        if self._watch_coordinator is None:
            self._watch_coordinator = engine_watch.WatchCoordinator(self._graph)
        elif self._watch_coordinator.graph is not self._graph:
            self._watch_coordinator.graph = self._graph
        return self._watch_coordinator

    def _should_filter_path(self, path: pathlib.Path) -> bool:
        """Check if path should be filtered (output of executing stage)."""
        coordinator = self._get_watch_coordinator()
        if coordinator is None:
            return False
        return coordinator.should_filter_path(path, get_stage_state=self._get_stage_state)

    def _defer_event_for_stage(self, stage: str, event: InputEvent) -> None:
        """Defer an event until the stage completes."""
        self._deferred_events[stage].append(event)

    _DEFERRED_MAX_ITERATIONS: int = 100

    async def _process_deferred_events(self, stage: str) -> None:
        """Process deferred events for a completed stage, draining until empty.

        If event handlers defer new events for the same stage during processing,
        those are picked up in subsequent iterations. A max-iterations guard
        prevents infinite loops — on trip, remaining events are dropped and a
        diagnostic event is emitted.
        """
        for _ in range(self._DEFERRED_MAX_ITERATIONS):
            events = self._deferred_events.pop(stage, [])
            if not events:
                return
            for event in events:
                try:
                    await self._handle_input_event(event)
                except Exception:
                    _logger.exception(
                        "Error processing deferred event for stage %s: %s", stage, event
                    )

        # Guard tripped — drop remaining events and emit diagnostic
        remaining = self._deferred_events.pop(stage, [])
        remaining_count = len(remaining)
        message = (
            f"Deferred event loop for stage '{stage}' hit max iterations "
            f"({self._DEFERRED_MAX_ITERATIONS})"
        )
        detail = f"Dropped {remaining_count} remaining event(s)" if remaining_count else ""
        _logger.error("%s. %s", message, detail)
        await self.emit(
            EngineDiagnostic(
                type="engine_diagnostic",
                message=message,
                detail=detail,
            )
        )

    def _get_affected_stages_for_paths(self, filtered_paths: list[pathlib.Path]) -> list[str]:
        """Get all stages affected by multiple path changes (including downstream).

        Args:
            filtered_paths: Paths that have already been filtered through
                ``_should_filter_path`` to exclude executing-stage outputs.
        """
        coordinator = self._get_watch_coordinator()
        if coordinator is None:
            return []
        if not filtered_paths:
            return []
        return coordinator.get_affected_stages(filtered_paths)

    # =========================================================================
    # Registry Reload
    # =========================================================================

    def _invalidate_caches(self) -> None:
        """Invalidate all caches when code changes."""
        linecache.clearcache()
        importlib.invalidate_caches()
        self._graph = None
        self._watch_coordinator = None
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

            new_pipeline = discovery.discover_pipeline(root, all_pipelines=self._all_pipelines)
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
