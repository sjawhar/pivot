from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import contextvars
import importlib
import itertools
import linecache
import logging
import os
import pathlib
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

import anyio
import anyio.from_thread
import anyio.lowlevel
import anyio.to_thread

from pivot import (
    config,
    exceptions,
    fingerprint,
    parameters,
    project,
    registry,
    types,
)
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
from pivot.storage import store as store_mod
from pivot.types import (
    ArtifactTag,
    CompletionType,
    DeferredWrites,
    HashInfo,
    LockData,
    OnError,
    OutputMessage,
    OutputMessageKind,
    StageExplanation,
    StageResult,
    StageStatus,
    ensure_completion_type,
)

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable

    import networkx as nx
    from anyio.abc import TaskGroup
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from pivot.registry import RegistryStageInfo
    from pivot.storage.cache import CheckoutMode

__all__ = ["Engine"]

_logger = logging.getLogger(__name__)

# Names the engine's "Pipeline invalid" log records with the error they report
# (``extra={"pivot_condition": (PIPELINE_ERROR_CONDITION, error)}``), so a TUI session can leave
# out the records of an error a later reload fixed.
PIPELINE_ERROR_CONDITION = "pipeline_error"

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


class _SkipOutcome(Enum):
    """What the coordinator's skip check decided for one stage."""

    RUN = "run"  # dispatch the stage to a worker
    SKIPPED = "skipped"  # recorded cached; its outputs already match the lock
    NEEDS_RESTORE = "needs_restore"  # cached, but outputs must first be written from the cache


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


_WatchOutputSignature = tuple[int, int, int] | None


@dataclass(slots=True)
class _WatchOutputSnapshot:
    """Filesystem state of an output after pivot completed its write."""

    is_directory: bool
    entries: dict[pathlib.Path, _WatchOutputSignature]


@dataclass(frozen=True, slots=True)
class _MissingDependencyRun:
    """A watch run a missing input stopped, and both forms of where it could appear."""

    request: RunRequested
    # Absolute paths from pipeline inputs; empty when only a stage output is missing
    input_paths: frozenset[pathlib.Path]
    # ``realpath`` forms of ``input_paths``, resolving the input's existing symlink prefix
    resolved_input_paths: frozenset[pathlib.Path]

    def awaits(self, path: pathlib.Path) -> bool:
        """Whether either form of ``path`` is a missing input or contains one."""
        resolved_path = pathlib.Path(os.path.realpath(path))
        return any(
            missing.is_relative_to(path) or missing.is_relative_to(resolved_path)
            for missing in itertools.chain(self.input_paths, self.resolved_input_paths)
        )


def _merge_run_requests(earlier: RunRequested, later: RunRequested) -> RunRequested:
    """One request that runs everything ``earlier`` and ``later`` would run.

    A restart replays this instead of ``later`` alone, so it never narrows what was going to run:
    the stages are the union (None, all stages, wins), the merged run is forced if either was, and
    dependencies are resolved unless both ran single stages. Every other field comes from
    ``later``, the newest request.
    """
    stages = (
        None
        if earlier["stages"] is None or later["stages"] is None
        else list(dict.fromkeys([*earlier["stages"], *later["stages"]]))
    )
    return RunRequested(
        type="run_requested",
        stages=stages,
        force=earlier["force"] or later["force"],
        reason=later["reason"],
        single_stage=earlier["single_stage"] and later["single_stage"],
        parallel=later["parallel"],
        max_workers=later["max_workers"],
        resource_capacity=later["resource_capacity"],
        no_commit=later["no_commit"],
        on_error=later["on_error"],
        cache_dir=later["cache_dir"],
        allow_uncached_incremental=later["allow_uncached_incremental"],
        checkout_missing=later["checkout_missing"],
    )


async def _run_in_daemon_thread[T](func: Callable[[], T]) -> T:
    """Run ``func`` on a new daemon thread and wait for its result; cancelling abandons it.

    anyio's worker threads are not daemon threads when the event loop runs on the main thread,
    so interpreter shutdown joins any still running. This is for work the process must not wait
    for when it exits, such as importing an edited pipeline.py that may never return.
    """
    token = anyio.lowlevel.current_token()
    done = anyio.Event()
    results = list[T]()
    failures = list[BaseException]()
    context = contextvars.copy_context()

    def _target() -> None:
        try:
            results.append(context.run(func))
        except BaseException as exc:  # re-raised in the waiting task
            failures.append(exc)
        # Once the event loop has finished or is closing (RunFinishedError, or "Event loop is
        # closed" in the race between the two), nobody waits for the result
        with contextlib.suppress(RuntimeError):
            anyio.from_thread.run_sync(done.set, token=token)

    threading.Thread(target=_target, name="pivot reload", daemon=True).start()
    await done.wait()
    if failures:
        raise failures[0]
    return results[0]


def _deferred_writes_to_apply(result: StageResult, *, no_commit: bool) -> DeferredWrites | None:
    """The StateDB writes a finished stage persists, or None when it persists nothing.

    Only a RAN or CACHED stage writes. Under ``--no-commit`` only the record of what ran is
    kept, for ``pivot commit``; nothing else is persisted.
    """
    if result["status"] not in (StageStatus.RAN, StageStatus.CACHED):
        return None
    if "deferred_writes" not in result:
        return None
    deferred = result["deferred_writes"]
    if not no_commit:
        return deferred
    if "no_commit_record" not in deferred:
        return None
    return DeferredWrites(no_commit_record=deferred["no_commit_record"])


class Engine:
    """Async coordinator for pipeline execution using anyio.

    Thread safety: All state access occurs in the event loop task within run().
    Sources run in separate tasks but only send events to channels - they don't
    access engine state. This design provides implicit serialization through
    the channel, so no explicit lock is needed.
    """

    _pipeline: registry.PipelineLike | None
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
    _watch_output_stats: dict[pathlib.Path, _WatchOutputSnapshot]
    _scheduler: engine_scheduler.Scheduler
    _cancel_event: anyio.Event
    _stage_indices: dict[str, tuple[int, int]]
    _deferred_events: dict[str, list[InputEvent]]
    # Execution orchestration state
    _futures: dict[concurrent.futures.Future[StageResult], str]
    _worker_pool: worker_pool_mod.WorkerPool | None
    _effective_max_workers: int
    _worker_idle_timeout_s: float | None
    # Workers the run's first submission starts: the run's resource width
    # (Scheduler.resource_width). Zero once that submission is made.
    _first_submission_workers: int

    # Run state machine (Task 10/11)
    _run_state: _RunState
    _run_task_group: TaskGroup | None
    # Requests made since the running run began, replayed once it ends
    _restart_pending: RunRequested | None
    # The request of the run in progress; a restart that cancels it replays its stages too
    _running_request: RunRequested | None
    # The request of a run a restart cancelled, merged into the restart only when it replays;
    # an explicit cancel clears it, so that run's own stages stay cancelled
    _cancelled_request: RunRequested | None
    _reload_pending: set[str] | None
    _idle_condition: anyio.Condition

    # Stored orchestration params (for watch mode re-runs)
    _stored_no_commit: bool
    _stored_on_error: OnError
    _stored_parallel: bool
    _stored_max_workers: int | None
    _stored_resource_capacity: dict[str, int]

    # Track whether run() has completed to prevent re-use
    _run_completed: bool

    # Event signaling dispatcher has finished draining
    _dispatch_complete: anyio.Event

    # Set once a watched engine's startup fingerprint snapshot has finished
    _startup_fingerprinted: anyio.Event
    # Why the pipeline is invalid: set by the startup snapshot, a rejected reload, or a watch run
    # that a missing input leaves with no stage to run; cleared by the next good reload, or by
    # the first run that missing inputs no longer stop. While set, no stage runs (RPC run, data
    # change, or startup).
    _pipeline_error: str | None
    # The watch run a missing input stopped. While set, a data change checks the dependencies
    # again instead of being dropped, and runs this run's stages too.
    _missing_dependency_run: _MissingDependencyRun | None
    # Set while no code reload is in flight; replaced by a fresh Event when one starts
    _reload_done: anyio.Event

    def __init__(
        self, *, pipeline: registry.PipelineLike | None = None, all_pipelines: bool = False
    ) -> None:
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
        self._watch_output_stats = dict[pathlib.Path, _WatchOutputSnapshot]()
        self._scheduler = engine_scheduler.Scheduler()
        self._cancel_event = anyio.Event()
        self._stage_indices = dict[str, tuple[int, int]]()
        self._deferred_events = collections.defaultdict(list)

        self._futures = dict[concurrent.futures.Future[StageResult], str]()
        self._worker_pool = None
        self._effective_max_workers = 1
        self._worker_idle_timeout_s = None
        self._first_submission_workers = 0

        # Run state machine (Task 10/11)
        self._run_state = _RunState.IDLE
        self._run_task_group = None
        self._restart_pending = None
        self._running_request = None
        self._cancelled_request = None
        self._reload_pending = None
        self._idle_condition = anyio.Condition()

        # Current run_id for stamping events (Task 12)
        self._current_run_id: str | None = None

        self._stored_no_commit = False
        self._stored_on_error = OnError.FAIL
        self._stored_parallel = True
        self._stored_max_workers = None
        self._stored_resource_capacity = dict[str, int]()
        self._current_overrides: parameters.ParamsOverrides | None = None

        # Track whether run() has completed to prevent re-use
        self._run_completed = False

        # Event signaling dispatcher has finished draining (recreated each run())
        self._dispatch_complete = anyio.Event()

        self._startup_fingerprinted = anyio.Event()
        self._pipeline_error = None
        self._missing_dependency_run = None
        self._reload_done = anyio.Event()
        self._reload_done.set()

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        return self._state

    @property
    def pipeline_error(self) -> str | None:
        """Why the loaded pipeline cannot run, or None when it can."""
        return self._pipeline_error

    @property
    def is_starting(self) -> bool:
        """Whether a watched engine is still taking its startup fingerprint snapshot."""
        return self._watches_filesystem() and not self._startup_fingerprinted.is_set()

    @property
    def is_reloading(self) -> bool:
        """Whether a code reload is in flight: import, fingerprints, then the re-run's start."""
        return not self._reload_done.is_set()

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
        if self._worker_pool is not None:
            worker_pool = self._worker_pool
            self._worker_pool = None
            try:
                with anyio.CancelScope(shield=True):
                    shutdown = (
                        worker_pool.hard_cancel
                        if _exc and _exc[0] is not None
                        else worker_pool.shutdown
                    )
                    await anyio.to_thread.run_sync(shutdown)
            except (OSError, BrokenPipeError) as exc:
                _logger.debug("WorkerPool shutdown failed (may already be dead): %s", exc)

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
        self._worker_idle_timeout_s = (
            executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S if exit_on_completion else None
        )

        async with anyio.create_task_group() as tg:
            self._run_task_group = tg

            # The RPC server needs no fingerprints, so it listens while the startup snapshot
            # runs: clients connect at once, `run` is refused with "Engine is starting" until the
            # snapshot finishes, and queries that fingerprint wait for it. Every other source,
            # the startup run and the file watcher included, starts after the snapshot.
            for source in self._sources:
                if isinstance(source, agent_rpc.AgentRpcSource):
                    tg.start_soon(self._run_source_with_cleanup, source, self._input_send.clone())

            if self._watches_filesystem():
                try:
                    await self._fingerprint_watched_stages()
                finally:
                    self._startup_fingerprinted.set()

            for source in self._sources:
                if not isinstance(source, agent_rpc.AgentRpcSource):
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
                # An explicit cancel: the run's own stages do not replay with a queued restart
                self._cancelled_request = None
                await self._handle_cancel_requested()
            case "data_artifact_changed":
                await self._handle_data_artifact_changed(event)
            case "code_or_config_changed":
                await self._handle_code_or_config_changed(event)

    async def _handle_run_requested(
        self, event: RunRequested, *, recover_missing: bool = False
    ) -> None:
        """Handle a RunRequested event by spawning a non-blocking execution task.

        If already running, coalesces into a restart: cancels the current run and queues the new
        event, replayed with the cancelled run's stages once that run ends. While a run is
        cancelling, the event joins the queued restart. An explicit cancel, before or after,
        keeps the cancelled run's stages out of the replay. A restart waiting behind a deferred
        reload is folded into that reload's re-run. ``recover_missing`` is only set after a
        filesystem event was admitted as a missing-input recovery.
        """
        self._stored_no_commit = event["no_commit"]
        self._stored_on_error = event["on_error"]
        self._stored_parallel = event["parallel"]
        self._stored_max_workers = event["max_workers"]
        self._stored_resource_capacity = dict(event["resource_capacity"])

        if self._pipeline_error is not None:
            missing = self._missing_dependency_run
            if missing is None or not recover_missing:
                _logger.debug("Pipeline invalid - ignoring run request until it is fixed")
                return
            event = _merge_run_requests(missing.request, event)

        if self._run_state == _RunState.RUNNING:
            # Coalesce: hold the cancelled run's request apart, merged in only at replay
            assert self._running_request is not None
            self._cancelled_request = self._running_request
            self._restart_pending = event
            await self._handle_cancel_requested()
            return

        if self._run_state == _RunState.CANCELLING:
            # Wait for the stopping run and join the restart queued behind it
            self._restart_pending = (
                event
                if self._restart_pending is None
                else _merge_run_requests(self._restart_pending, event)
            )
            return

        pending = self._take_restart()
        if pending is not None:
            # Only a deferred reload's re-run starts while a restart waits; it runs both
            pending = self._drop_removed_stages(pending)
            if pending is not None:
                event = _merge_run_requests(pending, event)

        self._run_state = _RunState.RUNNING
        self._running_request = event

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
                resource_capacity=event["resource_capacity"],
                no_commit=event["no_commit"],
                on_error=event["on_error"],
                cache_dir=event["cache_dir"],
                allow_uncached_incremental=event["allow_uncached_incremental"],
                checkout_missing=event["checkout_missing"],
            )
        finally:
            self._running_request = None
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
                    # Its batches' data files were handled as data changes when they arrived
                    await self._handle_code_or_config_changed(
                        CodeOrConfigChanged(
                            type="code_or_config_changed", paths=reload_paths, data_paths=[]
                        )
                    )

                # Replay a restart no reload re-run absorbed (taken before the call)
                restart = self._take_restart()
                if restart is not None and reload_pending is not None:
                    restart = self._drop_removed_stages(restart)
                if restart is not None:
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

    def _take_restart(self) -> RunRequested | None:
        """Remove and return the queued restart, with the cancelled run's stages merged in.

        The cancelled run's request is held apart until now so an explicit cancel made after the
        restart was queued can still drop it.
        """
        restart = self._restart_pending
        cancelled = self._cancelled_request
        self._restart_pending = None
        self._cancelled_request = None
        if restart is None or cancelled is None:
            return restart
        return _merge_run_requests(cancelled, restart)

    def _drop_removed_stages(self, request: RunRequested) -> RunRequested | None:
        """``request`` without stages a reload removed since it was made; None if none remain.

        An empty stage list would mean every stage, so a request left with none is dropped.
        """
        if request["stages"] is None:
            return request
        current = set(self._list_stages())
        stages = [stage for stage in request["stages"] if stage in current]
        if not stages:
            return None
        kept = request.copy()
        kept["stages"] = stages
        return kept

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

    def _require_pipeline(self) -> registry.PipelineLike:
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
        return self._require_pipeline().get_stage(name)

    def _get_all_stages(self) -> dict[str, RegistryStageInfo]:
        """Get all stages as a dict from pipeline."""
        pipeline = self._require_pipeline()
        return {name: pipeline.get_stage(name) for name in pipeline.list_stages()}

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
        resource_capacity: dict[str, int],
        no_commit: bool,
        on_error: OnError,
        cache_dir: pathlib.Path | None,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
    ) -> dict[str, executor_core.ExecutionSummary]:
        """Orchestrate parallel stage execution with the async event loop."""
        import datetime

        from pivot import exceptions as pivot_exceptions
        from pivot import run_history

        # Record start time for run history
        started_at = datetime.datetime.now(datetime.UTC).isoformat()

        if cache_dir is None:
            cache_dir = config.get_cache_dir()

        # Verify tracked files before building DAG
        project_root = project.get_project_root()
        executor_core.verify_tracked_files(project_root, checkout_missing=checkout_missing)

        # Build bipartite graph (single source of truth)
        all_stages = self._get_all_stages()
        graph = engine_graph.build_graph(all_stages)
        pipeline = self._pipeline
        if pipeline is None:
            raise pivot_exceptions.PipelineNotFoundError("No active pipeline registered.")
        store = store_mod.WorkspaceStore(
            project_root=project_root,
            pipeline_name=pipeline.name,
            input_bindings=pipeline.input_bindings,
        )
        runnable, dep_errors = engine_graph.get_runnable_stages(
            all_stages, store, all_pipelines=self._all_pipelines, on_error=on_error
        )
        if runnable is not None:
            # Skip broken stages but continue with healthy pipelines
            for err in dep_errors:
                _logger.warning("Skipping: %s", err.format_user_message())
        if runnable is None or (dep_errors and not runnable):
            missing = (
                dep_errors[0]
                if runnable is None and len(dep_errors) == 1
                else pivot_exceptions.MultipleDependencyError(dep_errors)
            )
            if not self._watches_filesystem():
                raise missing
            # Watch mode waits for the input: a data change, such as the input's creation,
            # checks again and replays this run (see _handle_run_requested)
            if self._running_request is None:
                raise RuntimeError("A watch run stopped by a missing input has no run request")
            input_paths = frozenset(
                err.input_path for err in dep_errors if err.input_path is not None
            )
            self._missing_dependency_run = _MissingDependencyRun(
                request=self._running_request,
                input_paths=input_paths,
                resolved_input_paths=frozenset(
                    pathlib.Path(os.path.realpath(path)) for path in input_paths
                ),
            )
            await self._invalidate_pipeline([missing.format_user_message()])
            return {}
        if self._missing_dependency_run is not None:
            # Missing inputs no longer prevent this run; keep-going may still skip broken stages.
            self._missing_dependency_run = None
            self._pipeline_error = None
            await self._emit_pipeline_error(None)
            if stages and dep_errors:
                # Across all pipelines with keep-going, a stage still missing an input stays out
                stages = [stage for stage in stages if stage in runnable]
                if not stages:
                    return {}
        if dep_errors:
            graph = engine_graph.build_graph(runnable)
        all_stages = runnable
        self._graph = graph

        # Extract stage-only DAG for execution order
        stage_dag = engine_graph.get_stage_dag(graph)

        if stages:
            registered = set(stage_dag.nodes())
            unknown = [s for s in stages if s not in registered]
            if unknown:
                raise pivot_exceptions.StageNotFoundError(
                    unknown,
                    available_stages=list(registered),
                )

        execution_order = engine_graph.get_execution_order(
            stage_dag, stages, single_stage=single_stage
        )

        if not execution_order:
            return {}

        # Check for uncached incremental output files that would be lost.
        if not allow_uncached_incremental:
            uncached = executor_core.check_uncached_incremental_outputs(
                execution_order, self._get_all_stages()
            )
            if uncached:
                files_list = "\n".join(f"  - {stage}: {path}" for stage, path in uncached)
                raise pivot_exceptions.UncachedIncrementalOutputError(
                    "The following incremental output files exist but are not in cache:\n"
                    + f"{files_list}\n\n"
                    + "Running the pipeline will DELETE these files and they cannot be restored.\n"
                    + "To proceed anyway, use --allow-uncached-incremental or back up these "
                    + "files first."
                )

        # The pool never needs more workers than stages that can run at once, but the cpu
        # resource capacity is the configured worker count, whatever this run schedules.
        cpu_capacity = executor_core.configured_worker_count(max_workers) if parallel else 1
        effective_max_workers = (
            1
            if not parallel
            else engine_graph.get_width(stage_dag, execution_order, limit=cpu_capacity)
        )

        try:
            stage_resources, capacities = self._resolve_stage_resources(
                execution_order,
                cpu_capacity,
                resource_capacity,
            )
        except exceptions.ValidationError as exc:
            if not self._watches_filesystem():
                raise
            await self._invalidate_pipeline([str(exc)])
            return {}

        # Load config. Startup and every reload check params.yaml, so in watch mode an error
        # here is an edit the watcher has not reported yet: the pipeline becomes invalid.
        try:
            overrides = parameters.load_params_yaml(pipeline)
        except (pivot_exceptions.ParamsError, pivot_exceptions.PipelineConfigError) as exc:
            if not self._watches_filesystem():
                raise
            await self._invalidate_pipeline([str(exc)])
            return {}
        checkout_modes = config.get_checkout_mode_order()
        self._current_overrides = overrides

        # Get project paths for worker info
        project_root = project.get_project_root()
        default_state_dir = config.get_state_dir()
        run_id = self._current_run_id or run_history.generate_run_id()
        self._current_run_id = run_id  # May already be set by _handle_run_requested

        # Ensure state directory exists
        default_state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize orchestration state after every stage request has been validated.
        await self._initialize_orchestration(
            execution_order,
            effective_max_workers,
            stage_resources=stage_resources,
            resource_capacity=capacities,
        )

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
                state_dbs[stage_state_dir] = state_mod.StateDB(stage_state_dir)
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

            # Only start drain thread and main loop if workers were dispatched in this run.
            if self._worker_pool is not None and self._worker_pool.in_run:
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
                            self._worker_pool.run_due_keepalives()
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
                                    if "user_code_files" in result:
                                        self._worker_pool.record_user_code_files(
                                            result["user_code_files"]
                                        )
                                    duration_ms = await self._handle_stage_completion(
                                        stage_name,
                                        result,
                                        start_time,
                                        default_state_dir,
                                        run_id=run_id,
                                    )
                                    stage_durations[stage_name] = duration_ms

                                    deferred = _deferred_writes_to_apply(
                                        result, no_commit=no_commit
                                    )
                                    if deferred is not None:
                                        stage_info = self._get_stage(stage_name)
                                        output_paths = [
                                            types.identity_key(out.identity)
                                            for out in stage_info["outs"]
                                        ]
                                        stage_state_dir = registry.get_stage_state_dir(
                                            stage_info, default_state_dir
                                        )
                                        stage_db = _get_state_db(stage_state_dir)
                                        stage_db.apply_deferred_writes(
                                            stage_name, output_paths, deferred
                                        )

                                    # Record result
                                    results[stage_name] = executor_core.ExecutionSummary(
                                        status=result["status"],
                                        reason=result["reason"],
                                        input_hash=result["input_hash"],
                                    )

                                except Exception as e:
                                    # A service manager can SIGTERM the worker cgroup at the
                                    # same time it stops pivot. Let cancellation win over a late
                                    # TerminatedWorkerError so a deliberate stop is not logged
                                    # as a stage failure.
                                    await anyio.lowlevel.checkpoint_if_cancelled()
                                    # One line names the stage and why; --verbose shows the
                                    # traceback. The reason recorded below stays str(e), and the
                                    # console prints no second copy under the stage line.
                                    _logger.error(
                                        "Stage %s failed: %s",
                                        stage_name,
                                        exceptions.describe_error(e),
                                    )
                                    _logger.debug(
                                        "Traceback of stage %s's failure", stage_name, exc_info=True
                                    )
                                    self._worker_pool.mark_workers_unverified()
                                    failed_result = StageResult(
                                        display_reason="",
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
                                    # Emit terminal events for blocked stages not yet in results
                                    for name, state in self._scheduler.stage_states.items():
                                        if (
                                            state == StageExecutionState.BLOCKED
                                            and name not in results
                                        ):
                                            await self._emit_terminal_stage(
                                                name,
                                                StageStatus.BLOCKED,
                                                f"upstream '{first_failed}' failed",
                                                results,
                                                run_id=run_id,
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
                                await self._emit_terminal_stage(
                                    name,
                                    StageStatus.BLOCKED,
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

            if self._cancel_event.is_set():
                await self._finalize_cancelled_stages(results, run_id)

        finally:
            # Close StateDBs first, then release this run's output queue after its drain exits.
            with anyio.CancelScope(shield=True):
                for db in state_dbs.values():
                    db.close()
                state_dbs.clear()

            if self._worker_pool is not None and self._worker_pool.in_run:
                self._worker_pool.end_run()

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

        # Materialize presentation symlinks for successful stages
        from pivot.storage import presentation

        try:
            presentation.present(
                project_root=project_root,
                pipeline_name=pipeline.name,
                cache_dir=cache_dir / "files",
                stages=all_stages,
            )
        except Exception:
            _logger.debug("Presentation layer failed", exc_info=True)

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

    def _resolve_stage_resources(
        self,
        execution_order: list[str],
        cpu_capacity: int,
        cli_resource_capacity: dict[str, int],
    ) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        """Resolve configured and CLI resource capacities for a scheduler run.

        ``cpu`` defaults to ``cpu_capacity`` (the worker count) unless config or
        ``--resources`` sets it.
        """
        capacities = {
            name.casefold(): capacity for name, capacity in config.get_resource_capacities().items()
        }
        capacities.update(
            {name.casefold(): capacity for name, capacity in cli_resource_capacity.items()}
        )
        capacities.setdefault("cpu", cpu_capacity)

        requests = dict[str, dict[str, int]]()
        for stage_name in execution_order:
            request = {"cpu": 1}
            # compose validated each request: a positive int, or "all" for cpu only.
            for resource, amount in self._get_stage(stage_name)["resources"].items():
                capacity = capacities.setdefault(resource, 1)
                request[resource] = capacity if amount == "all" else int(amount)
            for resource, amount in request.items():
                capacity = capacities[resource]
                if amount > capacity:
                    msg = f"Stage {stage_name!r} requests {amount} units of resource {resource!r}, but its capacity is {capacity}"
                    raise exceptions.ValidationError(msg)
            requests[stage_name] = request
        return requests, capacities

    async def _initialize_orchestration(
        self,
        execution_order: list[str],
        max_workers: int,
        *,
        stage_resources: dict[str, dict[str, int]] | None = None,
        resource_capacity: dict[str, int] | None = None,
    ) -> None:
        """Initialize orchestration state for a new execution."""
        self._futures.clear()
        self._deferred_events.clear()

        if stage_resources is None:
            stage_resources = {name: {"cpu": 1} for name in execution_order}
        if resource_capacity is None:
            resource_capacity = {"cpu": max_workers}
        self._scheduler.initialize(
            execution_order,
            self._graph,
            stage_resources=stage_resources,
            resource_capacity=resource_capacity,
        )
        self._effective_max_workers = max_workers
        self._first_submission_workers = self._scheduler.resource_width(limit=max_workers)

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

    def _can_start_stage(self, stage_name: str) -> bool:
        """Check whether a ready stage has all requested resources available."""
        return self._scheduler.can_start(stage_name)

    def _ensure_worker_pool(self) -> worker_pool_mod.WorkerPool:
        """Lazily begin this run's worker pool on first worker dispatch."""
        if self._worker_pool is None:
            self._worker_pool = worker_pool_mod.WorkerPool()
        if not self._worker_pool.in_run:
            self._worker_pool.start(
                max_workers=self._effective_max_workers,
                idle_timeout_s=self._worker_idle_timeout_s,
            )
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
        store_spec = store_mod.StoreSpec(
            kind="workspace",
            cache_dir=str(cache_dir / "files"),
            project_root=str(project_root),
            pipeline_name=pipeline.name,
            input_bindings=pipeline.input_bindings,
            checkout_modes=checkout_modes,
        )
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

                # Reserve every resource before checking the next ready stage.
                self._scheduler.acquire_resources(stage_name)

                # Transition to PREPARING
                await self._set_stage_state(stage_name, StageExecutionState.PREPARING)

                stage_info = self._get_stage(stage_name)

                try:
                    skipped = await self._try_skip_in_coordinator(
                        stage_name,
                        stage_info,
                        overrides,
                        force,
                        store_spec,
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
                # A watch event may cancel this run while its coordinator skip check awaited.
                # Do not submit a stage to the pool after cancellation stopped admissions.
                if self._cancel_event.is_set():
                    self._scheduler.release_resources(stage_name)
                    await self._set_stage_state(stage_name, StageExecutionState.COMPLETED)
                    await self._emit_terminal_stage(
                        stage_name,
                        StageStatus.CANCELLED,
                        "cancelled",
                        results,
                        run_id=run_id,
                    )
                    continue

                # Stage must run — lazily create pool on first dispatch
                started += 1
                pool = self._ensure_worker_pool()
                worker_info = executor_core.prepare_worker_info(
                    stage_info,
                    pipeline,
                    overrides,
                    run_id,
                    force,
                    no_commit,
                    project_root,
                    default_state_dir=state_dir,
                    store_spec=store_spec,
                )

                active_after_parent = len(self._futures)
                prestart_limit = self._effective_max_workers - active_after_parent
                direct_consumers = self._scheduler.preview_direct_consumers(
                    stage_name,
                    limit=prestart_limit,
                )
                prestart_target = max(
                    active_after_parent + direct_consumers, self._first_submission_workers
                )
                self._first_submission_workers = 0
                future = pool.submit(
                    worker.execute_stage,
                    stage_name,
                    worker_info,
                    cache_dir,
                    pool.output_queue(),
                    prestart_target=prestart_target,
                )
                self._futures[future] = stage_name
                pool.keep_consumers_warm(stage_name, direct_consumers)

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
        force: bool,
        store_spec: store_mod.StoreSpec,
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
        store = store_mod.store_from_spec(store_spec)

        def _read_lock() -> LockData | None:
            stages_dir = lock.get_stages_dir(stage_state_dir)
            stage_lock = lock.StageLock(stage_name, stages_dir)
            return stage_lock.read()

        lock_data = await anyio.to_thread.run_sync(_read_lock)
        if lock_data is None:
            return False

        def _ensure_fingerprint() -> dict[str, str]:
            pipeline = self._require_pipeline()
            return pipeline.ensure_fingerprint(stage_name)

        current_fingerprint = await anyio.to_thread.run_sync(_ensure_fingerprint)

        try:
            current_params = parameters.get_effective_params(
                stage_info["params"], stage_name, overrides
            )
        except Exception:
            return False

        out_paths = [types.identity_key(ref.identity) for ref in stage_info["outs"]]

        async def _skip_under_locks(outs_mode: artifact_lock.LockMode) -> _SkipOutcome:
            """Take READ on deps and ``outs_mode`` on outs, so no concurrent pivot run writes
            them mid-check, then decide afresh from the lock file as it is now."""
            await self._set_stage_state(stage_name, StageExecutionState.WAITING_ON_LOCK)

            def _acquire_artifact_locks() -> artifact_lock.LockHandle:
                lock_requests = artifact_lock.expand_lock_requests(
                    stage_info["deps"], stage_info["outs"], project_root, outs_mode=outs_mode
                )
                lock_service = artifact_lock.LocalFlockLockService(stage_state_dir / "locks")
                return lock_service.acquire_many(lock_requests)

            lock_handle = await anyio.to_thread.run_sync(_acquire_artifact_locks)
            try:
                locked_data = await anyio.to_thread.run_sync(_read_lock)
                if locked_data is None:
                    return _SkipOutcome.RUN
                await self._set_stage_state(stage_name, StageExecutionState.PREPARING)
                return await self._try_skip_with_locks(
                    stage_name,
                    stage_info,
                    locked_data,
                    current_fingerprint,
                    current_params,
                    out_paths,
                    store,
                    stage_state_dir,
                    state_dir,
                    results,
                    run_id,
                    skip_state_dbs,
                    restore=outs_mode is artifact_lock.LockMode.WRITE,
                )
            finally:
                lock_handle.release()

        # Deciding that a cached stage can skip only reads its outputs, so READ on them is
        # enough, and the check does not wait for runs that read them as deps (a widely read
        # output is held READ by every running consumer). Only an output that must be written
        # from the cache needs WRITE. READ is released before WRITE is requested, never
        # upgraded in place (two upgrading holders would wait on each other), and the whole
        # decision is made again under WRITE: another run may have restored or rebuilt the
        # outputs in between.
        outcome = await _skip_under_locks(artifact_lock.LockMode.READ)
        if outcome is _SkipOutcome.NEEDS_RESTORE:
            outcome = await _skip_under_locks(artifact_lock.LockMode.WRITE)
        return outcome is _SkipOutcome.SKIPPED

    async def _try_skip_with_locks(
        self,
        stage_name: str,
        stage_info: RegistryStageInfo,
        lock_data: LockData,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        out_paths: list[str],
        store: store_mod.Store,
        stage_state_dir: pathlib.Path,
        state_dir: pathlib.Path,
        results: dict[str, executor_core.ExecutionSummary],
        run_id: str,
        skip_state_dbs: dict[pathlib.Path, state_mod.StateDB] | None,
        *,
        restore: bool,
    ) -> _SkipOutcome:
        """Perform skip detection while artifact locks are held.

        With ``restore``, outputs that differ from the lock are written from the cache, which
        needs WRITE on them. Without it, nothing is written: a cached stage whose outputs
        need restoring returns ``NEEDS_RESTORE``, for the caller to decide again under WRITE.
        """
        from pivot import run_history
        from pivot import skip as skip_mod

        deps = stage_info["deps"]

        def _save_file_hashes(entries: list[types.FileHashEntry]) -> None:
            """Record the file hashes this check read or restored, so the next one reads none;
            best-effort, so a write that cannot happen never fails the run."""
            if entries:
                _ = state_mod.save_best_effort(
                    stage_state_dir,
                    lambda writable_db: writable_db.apply_deferred_writes(
                        stage_name, [], DeferredWrites(file_hash_entries=entries)
                    ),
                    what=f"{len(entries)} file hashes",
                )

        def _use_outputs(state_db: state_mod.StateDB) -> _SkipOutcome:
            file_hash_entries = list[types.FileHashEntry]()
            if not restore:
                checked = worker.restore_outputs_to_hashes(
                    store,
                    stage_info["outs"],
                    lock_data["output_hashes"],
                    state_db,
                    state_dir=state_dir,
                    restore=False,
                    report_changes=True,
                    file_hash_entries=file_hash_entries,
                )
                _save_file_hashes(file_hash_entries)
                if not checked.available:
                    return _SkipOutcome.RUN
                return _SkipOutcome.NEEDS_RESTORE if checked.to_restore else _SkipOutcome.SKIPPED
            restored = worker.restore_outputs_to_hashes(
                store,
                stage_info["outs"],
                lock_data["output_hashes"],
                state_db,
                state_dir=state_dir,
                file_hash_entries=file_hash_entries,
            )
            _save_file_hashes(file_hash_entries)
            return _SkipOutcome.SKIPPED if restored else _SkipOutcome.RUN

        def _get_skip_state_db() -> state_mod.StateDB:
            if skip_state_dbs is not None:
                if stage_state_dir not in skip_state_dbs:
                    skip_state_dbs[stage_state_dir] = state_mod.StateDB(
                        stage_state_dir, readonly=True
                    )
                return skip_state_dbs[stage_state_dir]
            return state_mod.StateDB(stage_state_dir, readonly=True)

        def _try_generation_skip() -> tuple[_SkipOutcome, str | None]:
            state_db = _get_skip_state_db()
            try:
                can_skip = worker.can_skip_via_generation(
                    stage_name=stage_name,
                    fingerprint=current_fingerprint,
                    deps=deps,
                    outs_paths=out_paths,
                    current_params=current_params,
                    lock_data=lock_data,
                    state_db=state_db,
                    store=store,
                )
                if not can_skip:
                    return _SkipOutcome.RUN, None
                input_hash = run_history.stage_input_hash(
                    current_fingerprint,
                    current_params,
                    deps.values(),
                    lock_data["dep_hashes"],
                    stage_info["outs"],
                )
                return _use_outputs(state_db), input_hash
            finally:
                if skip_state_dbs is None:
                    state_db.close()

        generation_outcome, generation_input_hash = await anyio.to_thread.run_sync(
            _try_generation_skip
        )
        if generation_outcome is _SkipOutcome.NEEDS_RESTORE:
            return generation_outcome
        if generation_outcome is _SkipOutcome.SKIPPED:
            await self._record_cached_stage(
                stage_name,
                "unchanged (generation)",
                generation_input_hash,
                state_dir,
                results,
                run_id=run_id,
            )
            return generation_outcome

        def _hash_deps() -> tuple[dict[str, HashInfo], list[str], list[str]]:
            state_db = _get_skip_state_db()
            try:
                dep_hashes, missing, unreadable, file_hash_entries = worker.hash_dependencies(
                    deps, store, state_db
                )
            finally:
                if skip_state_dbs is None:
                    state_db.close()
            _save_file_hashes(file_hash_entries)
            return dep_hashes, missing, unreadable

        dep_hashes, missing, unreadable = await anyio.to_thread.run_sync(_hash_deps)
        dep_hashes_by_identity = {
            types.identity_from_key(key): info for key, info in dep_hashes.items()
        }
        if missing or unreadable:
            return _SkipOutcome.RUN

        out_identities = [out.identity for out in stage_info["outs"]]
        decision = skip_mod.check_stage(
            lock_data,
            current_fingerprint,
            current_params,
            dep_hashes_by_identity,
            out_identities,
            explain=False,
            force=False,
        )
        if decision["changed"]:
            return _SkipOutcome.RUN

        input_hash = run_history.stage_input_hash(
            current_fingerprint,
            current_params,
            deps.values(),
            dep_hashes_by_identity,
            stage_info["outs"],
        )

        def _use_lock_outputs() -> _SkipOutcome:
            state_db = _get_skip_state_db()
            try:
                return _use_outputs(state_db)
            finally:
                if skip_state_dbs is None:
                    state_db.close()

        outcome = await anyio.to_thread.run_sync(_use_lock_outputs)
        if outcome is not _SkipOutcome.SKIPPED:
            return outcome

        await self._record_cached_stage(
            stage_name,
            "unchanged",
            input_hash,
            state_dir,
            results,
            run_id=run_id,
        )
        return outcome

    async def _record_cached_stage(
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
            status=StageStatus.CACHED,
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
        from pivot.storage import lock

        def _compute() -> StageExplanation:
            pipeline = self._require_pipeline()
            store = store_mod.WorkspaceStore(
                project_root=worker_info["project_root"],
                pipeline_name=pipeline.name,
                input_bindings=pipeline.input_bindings,
                cache_dir=pathlib.Path(worker_info["store_spec"]["cache_dir"]),
            )
            lock_data = lock.StageLock(
                stage_name, lock.get_stages_dir(worker_info["state_dir"])
            ).read()
            return explain_mod.get_stage_explanation(
                stage_name=stage_name,
                fingerprint=worker_info["fingerprint"],
                deps=stage_info["deps"],
                outs=list(stage_info["outs"]),
                params_instance=worker_info["params"],
                overrides=overrides,
                state_dir=worker_info["state_dir"],
                store=store,
                lock_data=lock_data,
                force=force,
                allow_missing=True,
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
        from pivot.storage import lock

        def _compute() -> list[OutputChangeSummary]:
            stage_info = self._get_stage(stage_name)
            stage_state_dir = registry.get_stage_state_dir(stage_info, state_dir)
            stages_dir = lock.get_stages_dir(stage_state_dir)
            stage_lock = lock.StageLock(stage_name, stages_dir)
            lock_data = stage_lock.read()

            summary = list[OutputChangeSummary]()
            new_hashes = dict[types.ArtifactIdentity, str]()
            if lock_data is not None and "output_hashes" in lock_data:
                for identity, hash_info in lock_data["output_hashes"].items():
                    new_hashes[identity] = hash_info["hash"]

            for out in stage_info["outs"]:
                match out.tag:
                    case ArtifactTag.METRIC:
                        output_type = "metric"
                    case ArtifactTag.PLOT:
                        output_type = "plot"
                    case _:
                        output_type = "out"
                # change_type is None because we don't have old hashes at this point
                summary.append(
                    OutputChangeSummary(
                        identity=out.identity,
                        change_type=None,
                        output_type=output_type,
                        old_hash=None,
                        new_hash=new_hashes.get(out.identity),
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
        # Stop keeping this stage's consumer workers warm; from here they are ordinary idle
        # workers that the consumers reuse or the idle timeout reclaims.
        if self._worker_pool is not None:
            self._worker_pool.release_consumers(stage_name)
        # Record completed output state before replaying deferred file events.
        self._record_watch_output_stats(stage_name)
        await self._set_stage_state(stage_name, StageExecutionState.COMPLETED)

        # Compute output summary for stages that ran (not failed)
        output_summary: list[OutputChangeSummary] | None = None
        if result["status"] != StageStatus.FAILED and state_dir is not None:
            output_summary = await self._compute_output_summary(stage_name, state_dir)

        # Emit StageCompleted event
        stage_index, total_stages = self._get_stage_index(stage_name)
        completed = StageCompleted(
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
        if "display_reason" in result:
            completed["display_reason"] = result["display_reason"]
        await self.emit(completed)

        # Release scheduler-held resources before downstream stages are considered.
        try:
            self._scheduler.release_resources(stage_name)
        except ValueError:
            _logger.error("Resource underflow releasing permits for stage %s", stage_name)

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

    async def _finalize_cancelled_stages(
        self,
        results: dict[str, executor_core.ExecutionSummary],
        run_id: str,
    ) -> None:
        """Mark every ready or pending stage cancelled exactly once."""
        for stage_name, old_state in self._scheduler.apply_cancel():
            await self.emit(
                StageStateChanged(
                    type="stage_state_changed",
                    stage=stage_name,
                    state=StageExecutionState.COMPLETED,
                    previous_state=old_state,
                )
            )
            await self._emit_terminal_stage(
                stage_name,
                StageStatus.CANCELLED,
                "cancelled",
                results,
                run_id=run_id,
            )

    async def _emit_terminal_stage(
        self,
        stage_name: str,
        status: CompletionType,
        reason: str,
        results: dict[str, executor_core.ExecutionSummary],
        run_id: str = "",
    ) -> None:
        """Record and emit a non-executed stage completion (cached/blocked/cancelled)."""
        results[stage_name] = executor_core.ExecutionSummary(
            status=status,
            reason=reason,
            input_hash=None,
        )
        stage_index, total_stages = self._get_stage_index(stage_name)
        await self.emit(
            StageCompleted(
                type="stage_completed",
                stage=stage_name,
                status=status,
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
                status=ensure_completion_type(summary["status"]),
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

        with state_mod.StateDB(config.get_state_dir()) as state_db:
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

    def _watches_filesystem(self) -> bool:
        """Return whether this engine has a filesystem event source."""
        from pivot.engine.sources import FilesystemSource

        return any(isinstance(source, FilesystemSource) for source in self._sources)

    async def wait_for_fingerprints(self) -> None:
        """Wait until neither the startup snapshot nor a code reload is fingerprinting stages.

        RPC queries that fingerprint stages wait here, so no stage is fingerprinted on two
        threads at once. Returns at once for an engine without a filesystem source, which
        takes no snapshot and never reloads.
        """
        if self._watches_filesystem():
            await self._startup_fingerprinted.wait()
        while self.is_reloading:
            await self._reload_done.wait()

    async def _fingerprint_watched_stages(self) -> None:
        """Fingerprint every stage before a watched engine starts its watcher and startup run.

        A code reload diffs the edited pipeline against these cached fingerprints; computing
        the old ones after the edit would read the edited source and report nothing modified.
        Each stage is fingerprinted in a worker thread so the RPC server keeps answering, and
        nothing else fingerprints until this returns (see ``wait_for_fingerprints``),
        so the startup run's skip check reuses the cached values.

        A stage that fails to fingerprint, or a ``params.yaml`` that fails its checks against the
        pipeline, makes the pipeline invalid, as a ``pipeline.py`` syntax error does on reload: the
        errors are logged and no run starts until an edit reloads a valid pipeline.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return
        errors = list[str]()
        # Each worker thread copies this context, so the stages share one fingerprint pass.
        with fingerprint.fingerprint_pass():
            for stage_name in pipeline.list_stages():
                try:
                    _ = await anyio.to_thread.run_sync(pipeline.ensure_fingerprint, stage_name)
                except exceptions.PivotError as exc:
                    errors.append(str(exc))
        try:
            _ = parameters.load_params_yaml(pipeline)
        except (exceptions.ParamsError, exceptions.PipelineConfigError) as exc:
            errors.append(str(exc))
        if errors:
            await self._invalidate_pipeline(errors)

    async def _invalidate_pipeline(self, errors: list[str]) -> None:
        """Mark the loaded pipeline invalid: log and report ``errors``; nothing runs until a fix."""
        self._pipeline_error = "\n".join(errors)
        condition = {"pivot_condition": (PIPELINE_ERROR_CONDITION, self._pipeline_error)}
        for error in errors:
            _logger.warning(f"Pipeline invalid: {error}", extra=condition)
        _logger.error("Pipeline invalid - waiting for fix", extra=condition)
        await self._emit_pipeline_error(self._pipeline_error)

    async def _emit_pipeline_error(self, message: str | None) -> None:
        """Report why the pipeline is invalid, or None once it is valid again without a reload.

        The stage list stays the loaded pipeline's.
        """
        stages = self._pipeline.list_stages() if self._pipeline is not None else list[str]()
        await self.emit(
            PipelineReloaded(
                type="pipeline_reloaded",
                stages=stages,
                stages_added=list[str](),
                stages_removed=list[str](),
                stages_modified=list[str](),
                error=message,
            )
        )

    def _snapshot_watch_output(self, path: pathlib.Path) -> _WatchOutputSnapshot:
        """Capture every currently visible path in one output's filesystem tree."""
        entries = dict[pathlib.Path, _WatchOutputSignature]()

        def _record(entry: pathlib.Path) -> None:
            try:
                stat = entry.stat()
            except FileNotFoundError:
                entries[entry] = None
                return
            entries[entry] = (stat.st_mtime_ns, stat.st_size, stat.st_ino)

        _record(path)
        is_directory = path.is_dir()
        if is_directory:
            for child in path.rglob("*"):
                _record(child)
        return _WatchOutputSnapshot(is_directory=is_directory, entries=entries)

    def _record_watch_output_stats(self, stage_name: str) -> None:
        """Remember output filesystem state written by a completed watched stage."""
        if not self._watches_filesystem():
            return
        coordinator = self._get_watch_coordinator()
        if coordinator is None:
            return
        for path, identity in coordinator.artifact_paths.items():
            if engine_graph.get_producer(coordinator.graph, identity) == stage_name:
                self._watch_output_stats[path] = self._snapshot_watch_output(path)

    def _is_own_watch_output_write(
        self, path: pathlib.Path, verdicts: dict[pathlib.Path, bool]
    ) -> bool:
        """Return whether a changed output still matches the engine's own write.

        The deepest recorded root that is the path itself, or a directory output holding it,
        decides. ``verdicts`` memoizes each root's answer for one event batch, so a batch of K
        files in one directory output snapshots and compares that directory once.
        """
        changed_path = project.normalize_path(path)
        for root in (changed_path, *changed_path.parents):
            recorded = self._watch_output_stats.get(root)
            if recorded is None or (root != changed_path and not recorded.is_directory):
                continue
            if root not in verdicts:
                verdicts[root] = self._snapshot_watch_output(root) == recorded
            return verdicts[root]
        return False

    async def _handle_data_artifact_changed(self, event: DataArtifactChanged) -> None:
        """Handle data artifact changes by running affected stages."""
        if self.is_reloading:
            # Only a reload deferred to the end of a run overlaps the input loop; if it succeeds,
            # the re-run it starts checks every stage against the inputs as they are now.
            _logger.debug(
                "Data changed during a reload - handled by its re-run if the reload succeeds"
            )
            return
        paths = [pathlib.Path(path) for path in event["paths"]]
        filtered_paths = list[pathlib.Path]()
        deferred_paths = list[tuple[str, pathlib.Path]]()
        own_write_verdicts = dict[pathlib.Path, bool]()

        for path in paths:
            if self._should_filter_path(path):
                coordinator = self._get_watch_coordinator()
                producer = coordinator.get_producer(path) if coordinator is not None else None
                if producer is not None:
                    deferred_paths.append((producer, path))
                    continue
            if self._is_own_watch_output_write(path, own_write_verdicts):
                continue
            filtered_paths.append(path)

        for producer, path in deferred_paths:
            self._defer_event_for_stage(
                producer,
                DataArtifactChanged(type="data_artifact_changed", paths=[str(path)]),
            )

        if not filtered_paths:
            return

        missing = self._missing_dependency_run
        if missing is not None and any(
            missing.awaits(project.normalize_path(path)) for path in filtered_paths
        ):
            # A missing input that arrives inside a new directory is reported as the directory
            # alone, which maps to no stage.
            _logger.info("Data changed where a missing input belongs - checking again")
            await self._handle_run_requested(missing.request, recover_missing=True)
            return

        affected = self._get_affected_stages_for_paths(filtered_paths)
        if not affected:
            return

        _logger.info(
            "Data changed: %d file(s) affect %d stage(s)",
            len(filtered_paths),
            len(affected),
        )
        await self._execute_affected_stages(affected)

    async def _handle_code_or_config_changed(self, event: CodeOrConfigChanged) -> None:
        """Handle code/config changes by reloading registry and re-running.

        If a run is active, defers the reload until the current run completes, and handles the
        data files the same watch batch changed as a data change. Otherwise the reload runs
        first: when it starts a re-run, that re-run checks those files, so they start nothing
        of their own; when it starts none, they are handled as a data change once it ends, as
        a separate data event would be. Reloads run one at a time: an edit that arrives while a
        reload deferred to the end of a run is still in flight waits for it, then reloads again.
        """
        while True:
            if self._run_state in (_RunState.RUNNING, _RunState.CANCELLING):
                _logger.info("Code/config changed while running - deferring reload")
                if self._reload_pending is None:
                    self._reload_pending = set(event["paths"])
                else:
                    self._reload_pending.update(event["paths"])
                if event["data_paths"]:
                    await self._handle_data_artifact_changed(
                        DataArtifactChanged(type="data_artifact_changed", paths=event["data_paths"])
                    )
                return
            if not self.is_reloading:
                break
            await self._reload_done.wait()

        self._reload_done = anyio.Event()
        try:
            reran = await self._reload_and_rerun(event)
        finally:
            self._reload_done.set()
        if not event["data_paths"]:
            return
        if reran:
            _logger.debug("Data changed with the code - the reload's re-run covers the change")
            return
        await self._handle_data_artifact_changed(
            DataArtifactChanged(type="data_artifact_changed", paths=event["data_paths"])
        )

    async def _reload_and_rerun(self, event: CodeOrConfigChanged) -> bool:
        """Reload the edited pipeline off the event loop, then re-run every stage.

        Returns whether a re-run started: none does when the edit leaves the pipeline invalid
        or it has no stages.
        """
        _logger.info("Code/config changed - reloading pipeline")

        # Invalidate caches
        self._invalidate_caches()

        # Invalidate manifest cache entries for changed source files
        fingerprint.invalidate_manifests_for_paths(event["paths"])

        # Reload registry - returns the old stages on success, the error message on failure
        reload_result = await self._reload_registry()

        if isinstance(reload_result, str):
            _logger.error(
                "Pipeline invalid - waiting for fix",
                extra={"pivot_condition": (PIPELINE_ERROR_CONDITION, reload_result)},
            )
            await self._emit_pipeline_error(reload_result)
            return False
        old_stages = reload_result

        # Emit reload event
        await self._emit_reload_event(old_stages)

        # Flush manifest cache so cached manifests survive the reload
        fingerprint.flush_manifest_cache()

        all_stages = self._get_all_stages()
        # Hot-reload: build the graph without checking dependencies, which may be missing while
        # the user edits
        self._graph = engine_graph.build_graph(all_stages)

        # Update watch paths if we have a FilesystemSource.
        from pivot.engine.sources import FilesystemSource

        coordinator = self._get_watch_coordinator()
        if coordinator is not None:
            watch_paths = engine_watch.get_watch_paths(
                project.get_project_root(),
                coordinator.artifact_paths,
            )
            for source in self._sources:
                if isinstance(source, FilesystemSource):
                    source.set_watch_paths(watch_paths)

        # Re-run every stage. Across all pipelines with keep-going, leave out the stages a
        # missing dependency breaks, unless none is left; any other run checks dependencies
        # when it starts.
        stages = list(all_stages)
        if engine_graph.skips_missing_dependencies(
            all_pipelines=self._all_pipelines, on_error=self._stored_on_error
        ):
            pipeline = self._require_pipeline()
            store = store_mod.WorkspaceStore(
                project_root=project.get_project_root(),
                pipeline_name=pipeline.name,
                input_bindings=pipeline.input_bindings,
            )
            runnable, _ = engine_graph.get_runnable_stages(
                all_stages, store, all_pipelines=True, on_error=self._stored_on_error
            )
            stages = list(runnable or all_stages)

        if stages:
            # Restart the worker pool with the re-run's width of fresh workers so the next run
            # picks up reloaded code and starts its stages together. Run in thread to avoid
            # blocking the event loop during process kill/spawn.
            # A failed restart leaves the pool without an executor, so the re-run's start()
            # creates new workers; the failure is logged rather than ending the session.
            coordinator = self._get_watch_coordinator()
            if coordinator is not None and coordinator.should_restart_workers(
                parallel=self._stored_parallel
            ):
                width = engine_graph.get_width(
                    engine_graph.get_stage_dag(self._graph),
                    stages,
                    limit=executor_core.configured_worker_count(self._stored_max_workers),
                )
                if self._worker_pool is None:
                    self._worker_pool = worker_pool_mod.WorkerPool()
                worker_pool = self._worker_pool
                try:
                    await anyio.to_thread.run_sync(lambda: worker_pool.restart(max_workers=width))
                    _logger.info("Worker pool restarted for code reload")
                except Exception:
                    _logger.warning(
                        "Failed to restart worker pool - the next run starts new workers",
                        exc_info=True,
                    )

            await self._execute_affected_stages(stages)
            return True
        return False

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
            resource_capacity=dict(self._stored_resource_capacity),
            no_commit=self._stored_no_commit,
            on_error=self._stored_on_error,
            cache_dir=None,
            allow_uncached_incremental=False,
            checkout_missing=False,
        )
        await self._handle_run_requested(event)

    def _get_watch_coordinator(self) -> engine_watch.WatchCoordinator | None:
        """Lazily build watch planning from the current pipeline and graph."""
        if self._pipeline is None:
            return None
        if self._graph is None:
            self._graph = engine_graph.build_graph(self._get_all_stages())

        if self._watch_coordinator is None or self._watch_coordinator.graph is not self._graph:
            pipeline = self._require_pipeline()
            store = store_mod.WorkspaceStore(
                project_root=project.get_project_root(),
                pipeline_name=pipeline.name,
                input_bindings=pipeline.input_bindings,
            )
            self._watch_coordinator = engine_watch.WatchCoordinator(
                self._graph,
                engine_watch.build_artifact_paths(self._get_all_stages(), store),
            )
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
        self._watch_output_stats.clear()
        if self._pipeline is not None:
            self._pipeline.invalidate_dag_cache()

    async def _reload_registry(self) -> dict[str, RegistryStageInfo] | str:
        """Reload the pipeline by re-importing pipeline definition.

        Returns the old pipeline's stages if reload succeeded, or the error message if the
        pipeline is invalid: it fails to load, one of its stages fails to fingerprint, or
        ``params.yaml`` fails its checks against it. On failure the last loaded pipeline stays
        loaded but is marked invalid, so nothing runs until a good reload. The new fingerprints
        stay cached for the reload event's diff. The caller should emit the reload event using
        the returned stages.

        The import, the fingerprints and the ``params.yaml`` checks run in one daemon thread
        (``_load_pipeline``), so the RPC server keeps answering and an exit never waits for an
        import that hangs; the pipeline swap happens here, on the event loop.
        """
        old_pipeline = self._pipeline
        old_stages = old_pipeline.snapshot() if old_pipeline else {}
        # A good reload re-runs every stage and a rejected one waits for its own fix, so neither
        # replays the run a missing input stopped
        self._missing_dependency_run = None
        root = project.get_project_root()
        try:
            new_pipeline, errors = await _run_in_daemon_thread(lambda: self._load_pipeline(root))
        except Exception as e:
            return self._reject_reload(old_pipeline, [str(e)])
        if new_pipeline is None:
            return self._reject_reload(old_pipeline, ["No pipeline found during reload"])
        if errors:
            return self._reject_reload(old_pipeline, errors)

        self._pipeline = new_pipeline
        self._pipeline_error = None
        return old_stages

    def _load_pipeline(self, root: pathlib.Path) -> tuple[registry.PipelineLike | None, list[str]]:
        """Re-import the project's pipeline, fingerprint every stage and check ``params.yaml``.

        Runs in a daemon thread. All of a reload's ``sys.modules`` changes happen in this one
        call, on one thread: discovery's import session drops the project modules the previous
        load imported, then imports the edited ones. Returns the new pipeline, or None if the
        project defines none, and the errors that make it invalid.
        """
        from pivot import discovery

        new_pipeline = discovery.discover_pipeline(root, all_pipelines=self._all_pipelines)
        if new_pipeline is None:
            return None, list[str]()
        errors = list[str]()
        with fingerprint.fingerprint_pass():
            for stage_name in new_pipeline.list_stages():
                try:
                    _ = new_pipeline.ensure_fingerprint(stage_name)
                except exceptions.PivotError as exc:
                    errors.append(str(exc))
        try:
            _ = parameters.load_params_yaml(new_pipeline)
        except (exceptions.ParamsError, exceptions.PipelineConfigError) as exc:
            errors.append(str(exc))
        return new_pipeline, errors

    def _reject_reload(self, old_pipeline: registry.PipelineLike | None, errors: list[str]) -> str:
        """Keep the last loaded pipeline but mark it invalid; return the error message."""
        self._pipeline = old_pipeline
        self._pipeline_error = "\n".join(errors)
        condition = {"pivot_condition": (PIPELINE_ERROR_CONDITION, self._pipeline_error)}
        for error in errors:
            _logger.warning(f"Pipeline invalid: {error}", extra=condition)
        return self._pipeline_error

    async def _emit_reload_event(self, old_stages: dict[str, RegistryStageInfo]) -> None:
        """Emit PipelineReloaded event with diff information.

        Old fingerprints are the ones cached before the edit (see
        ``_fingerprint_watched_stages``); recomputing them now would read the edited source.
        A stage without a cached old fingerprint counts as modified.
        """
        new_stage_names = self._list_stages()
        new_stages_set = set(new_stage_names)
        old_stage_names = set(old_stages.keys())

        added = sorted(new_stages_set - old_stage_names)
        removed = sorted(old_stage_names - new_stages_set)

        # ``_reload_registry`` fingerprinted every new stage: shared ones are compared now, and
        # all of them are the baseline the next reload compares against.
        pipeline = self._require_pipeline()
        modified = list[str]()
        for stage_name in sorted(old_stage_names & new_stages_set):
            old_fp = old_stages[stage_name]["fingerprint"]
            if old_fp is None or old_fp != pipeline.ensure_fingerprint(stage_name):
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
