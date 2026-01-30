"""Engine: the central coordinator for event-driven pipeline execution."""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Self

from pivot import executor
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    InputEvent,
    OutputEvent,
    RunRequested,
    StageCompleted,
    StageStarted,
)
from pivot.types import OnError, RunEventType

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    import networkx as nx

    from pivot.executor.core import ExecutionSummary
    from pivot.types import RunJsonEvent

__all__ = ["Engine"]

_logger = logging.getLogger(__name__)


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
        """
        for sink in self._sinks:
            try:
                sink.handle(event)
            except Exception:
                _logger.exception("Sink %s failed to handle event %s", sink, event["type"])

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
        Delegates to the existing executor while emitting events to sinks.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run all stages.
            single_stage: If True, run only the specified stages (no downstream).
            parallel: If True, run stages in parallel.
            max_workers: Maximum worker processes.
            no_commit: If True, don't update lockfiles.
            no_cache: If True, disable run cache.
            allow_uncached_incremental: Allow incremental outputs without cache.
            checkout_missing: Checkout missing dependency files from cache.
            on_error: Error handling mode ('fail' or 'keep_going').
            cache_dir: Directory for lock files (defaults to .pivot/cache).
            progress_callback: Callback for JSONL progress events.

        Returns:
            Dict mapping stage name to ExecutionSummary.

        Raises:
            RuntimeError: If the engine is already active (concurrent calls not allowed).
        """
        # Prevent concurrent execution
        if self._state == EngineState.ACTIVE:
            msg = "Engine is already active - concurrent run_once() calls are not allowed"
            raise RuntimeError(msg)

        # Clear cancel event from any previous run. This is also done in
        # _handle_run_requested() for event-driven execution via run_loop().
        self._cancel_event.clear()

        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        # Reset stage indices for this run
        self._stage_indices.clear()

        def progress_adapter(event: RunJsonEvent) -> None:
            self._handle_progress_event(event)
            if progress_callback is not None:
                progress_callback(event)

        try:
            # Delegate to existing executor
            result = executor.run(
                stages=stages,
                force=force,
                single_stage=single_stage,
                parallel=parallel,
                max_workers=max_workers,
                no_commit=no_commit,
                no_cache=no_cache,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
                on_error=on_error,
                cache_dir=cache_dir,
                progress_callback=progress_adapter,
                cancel_event=self._cancel_event,
            )
            return result
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
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
                pass  # TODO: implement in Phase 4
            case "code_or_config_changed":
                pass  # TODO: implement in Phase 4

    def _handle_run_requested(self, event: RunRequested) -> None:
        """Handle a RunRequested event by executing stages."""
        # Clear cancel event before starting new execution
        self._cancel_event.clear()

        # Reset stage indices for this run
        self._stage_indices.clear()

        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        try:
            executor.run(
                stages=event["stages"],
                force=event["force"],
                cancel_event=self._cancel_event,
            )
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _handle_cancel_requested(self) -> None:
        """Handle a CancelRequested event by setting cancel flag."""
        self._cancel_event.set()
