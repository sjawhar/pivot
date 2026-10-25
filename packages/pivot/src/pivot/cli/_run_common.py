"""Shared helpers for run and repro CLI commands."""

from __future__ import annotations

import contextlib
import functools
import logging
import sys
import threading
from collections.abc import (  # noqa: TC003 - used in function signatures
    Callable,
    Coroutine,
    Mapping,
)
from typing import TYPE_CHECKING, Any, TextIO, TypedDict, cast, override

import anyio
import click

from pivot import discovery, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.engine import graph as engine_graph
from pivot.executor import core as executor_core
from pivot.types import OnError

if TYPE_CHECKING:
    import concurrent.futures
    import pathlib
    from collections.abc import Generator

    import networkx as nx

    from pivot.engine.types import OutputEvent, StageCompleted
    from pivot.executor import ExecutionSummary
    from pivot.registry import PipelineLike


logger = logging.getLogger(__name__)

_POLLER_CONNECT_ATTEMPTS = 30
_POLLER_CONNECT_RETRY_S = 1.0
_POLLER_STOP_CHECK_S = 0.05
POLLER_THREAD_NAME = "pivot-tui-event-poller"


def _record_key(record: logging.LogRecord) -> tuple[object, ...]:
    """Return the user-visible identity of a log record, uniquely if it cannot format."""
    try:
        exception = logging.Formatter().formatException(record.exc_info) if record.exc_info else ""
        return (record.name, record.levelno, record.getMessage(), exception, record.stack_info)
    except Exception:
        # Handler.handle() reports bad records through handleError. The deferral must never
        # interrupt the thread that logged one, and this unique object cannot be reused.
        return (object(),)


class _HoldWhileTuiRuns(logging.Filter):
    """Filter one console handler before, during and after the TUI owns the terminal."""

    _lock: threading.RLock
    _tui_started: threading.Event
    _printed: set[tuple[object, ...]]
    _held: dict[tuple[object, ...], logging.LogRecord]
    _released: bool

    def __init__(self, tui_started: threading.Event) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._tui_started = tui_started
        self._printed = set[tuple[object, ...]]()
        self._held = dict[tuple[object, ...], logging.LogRecord]()
        self._released = False

    @staticmethod
    def _condition(record: logging.LogRecord) -> tuple[str, object] | None:
        """Return only the condition protocol owned by Pivot's loggers."""
        if record.name == "pivot" or record.name.startswith("pivot."):
            return getattr(record, "pivot_condition", None)
        return None

    def _key(self, record: logging.LogRecord) -> tuple[object, ...]:
        """Include a Pivot-owned condition so a later pipeline error cannot replace an earlier one."""
        key = _record_key(record)
        condition = self._condition(record)
        return (*key, condition) if condition is not None else key

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Pass through before/after the TUI; hold new warning-or-higher records during it."""
        printing = self._released or not self._tui_started.is_set()
        if record.levelno < logging.WARNING:
            return printing
        if record.name.partition(".")[0] == "pivot_tui":
            return printing
        key = self._key(record)
        if printing:
            # The wrapper marks a record only after its actual terminal write. A repeat that
            # arrives after that write but before Textual starts has already reached the user.
            return key not in self._printed
        if key not in self._printed:
            self._held.setdefault(key, record)
        return False

    def _remember_printed(self, record: logging.LogRecord) -> None:
        """Remember a warning only after the original handler completed its terminal write."""
        if (
            not self._released
            and not self._tui_started.is_set()
            and record.levelno >= logging.WARNING
        ):
            self._printed.add(self._key(record))

    def handle_with(
        self,
        original_handle: Callable[[logging.LogRecord], bool],
        record: logging.LogRecord,
    ) -> bool:
        """Run one complete handler cycle under the lock that guards TUI takeover."""
        with self._lock:
            handled = original_handle(record)
            if handled:
                self._remember_printed(record)
            return handled

    @property
    def write_lock(self) -> threading.RLock:
        """The lock that covers this handler's filter decision and terminal write."""
        return self._lock

    def release(
        self,
        original_handle: Callable[[logging.LogRecord], bool],
        conditions: Mapping[str, object],
    ) -> None:
        """Print held records that still describe the engine state at TUI exit."""
        with self._lock:
            self._released = True
            for record in self._held.values():
                condition = self._condition(record)
                if condition is None or conditions.get(condition[0], condition[1]) == condition[1]:
                    original_handle(record)
            self._held.clear()


class _ConsoleDeferral:
    """Attach console filters and make passing writes atomic with starting the Textual TUI."""

    _tui_started: threading.Event
    _conditions: Callable[[], Mapping[str, object]]
    _handlers: list[tuple[logging.Handler, _HoldWhileTuiRuns, Callable[[logging.LogRecord], bool]]]

    def __init__(self, conditions: Callable[[], Mapping[str, object]]) -> None:
        self._tui_started = threading.Event()
        self._conditions = conditions
        self._handlers = list[
            tuple[logging.Handler, _HoldWhileTuiRuns, Callable[[logging.LogRecord], bool]]
        ]()

    def attach(self) -> None:
        """Wrap each console handler so its filter decision and terminal write share one lock."""
        for handler in logging.getLogger().handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            stream_handler = cast("logging.StreamHandler[TextIO]", handler)
            if stream_handler.stream not in (sys.stderr, sys.stdout):
                continue
            hold = _HoldWhileTuiRuns(self._tui_started)
            original_handle = stream_handler.handle

            stream_handler.addFilter(hold)
            stream_handler.handle = functools.partial(
                hold.handle_with, original_handle
            )  # atomic wrapper
            self._handlers.append((stream_handler, hold, original_handle))

    def start_holding(self) -> None:
        """Wait for every passing write, then let Textual take the terminal."""
        with contextlib.ExitStack() as locks:
            for _, hold, _ in self._handlers:
                locks.enter_context(hold.write_lock)
            self._tui_started.set()

    def release(self) -> None:
        """Flush held records, then restore the console handlers exactly as they were."""
        conditions = self._conditions()
        for handler, hold, original_handle in self._handlers:
            hold.release(original_handle, conditions)
            handler.removeFilter(hold)
            handler.handle = original_handle  # pyright: ignore[reportAttributeAccessIssue] - restore original


@contextlib.contextmanager
def defer_console_logging(
    conditions: Callable[[], Mapping[str, object]] = dict,
    *,
    hold_now: bool = True,
) -> Generator[_ConsoleDeferral]:
    """Keep console records off Textual's terminal, then print held diagnostics after exit.

    With ``hold_now=False``, records print until ``start_holding()``. ``run_tui_with_engine``
    uses that so a record printed before the TUI starts is remembered, not printed again later.
    """
    deferral = _ConsoleDeferral(conditions)
    deferral.attach()
    if hold_now:
        deferral.start_holding()
    try:
        yield deferral
    finally:
        deferral.release()


def compute_dag_levels(graph: nx.DiGraph[str]) -> dict[str, int]:
    """Compute DAG level for each stage.

    Level 0: stages with no dependencies
    Level N: stages whose dependencies are all at level < N

    Stages at the same level can run in parallel - there's no ordering between them.
    """
    import networkx as nx

    levels = dict[str, int]()
    # Process in topological order (dependencies before dependents)
    for stage in nx.dfs_postorder_nodes(graph):
        # successors = what this stage depends on (edges go consumer -> producer)
        dep_levels = [levels[dep] for dep in graph.successors(stage) if dep in levels]
        levels[stage] = max(dep_levels, default=-1) + 1
    return levels


def sort_for_display(execution_order: list[str], graph: nx.DiGraph[str]) -> list[str]:
    """Sort stages for TUI display: group matrix variants while respecting DAG structure.

    Uses DAG levels (not arbitrary execution order) so parallel-capable stages
    are treated as equals. Matrix variants are grouped at the level of their
    earliest member.
    """
    from pivot.types import parse_stage_name

    levels = compute_dag_levels(graph)

    # Compute minimum level for each base_name (group position)
    group_min_level: dict[str, int] = {}
    for name in execution_order:
        base, _ = parse_stage_name(name)
        level = levels.get(name, 0)
        if base not in group_min_level or level < group_min_level[base]:
            group_min_level[base] = level

    def display_sort_key(name: str) -> tuple[int, str, int, str]:
        base, variant = parse_stage_name(name)
        individual_level = levels.get(name, 0)
        # Sort by: group level, then base_name (to keep groups together),
        # then individual level, then variant name
        return (group_min_level[base], base, individual_level, variant)

    return sorted(execution_order, key=display_sort_key)


def ensure_stages_registered(*, all_pipelines: bool = False) -> None:
    """Ensure a Pipeline is discovered and in context.

    If no Pipeline is in context, attempts discovery and stores the result.

    Args:
        all_pipelines: If True, discover all pipelines recursively.
    """
    if cli_decorators.get_pipeline_from_context() is not None:
        return
    try:
        pipeline: PipelineLike | None = discovery.discover_pipeline(
            all_pipelines=all_pipelines,
        )
        if pipeline is not None:
            cli_decorators.store_pipeline_in_context(pipeline)
            logger.info(f"Loaded pipeline: {pipeline.name}")
    except discovery.DiscoveryError as e:
        raise click.ClickException(str(e)) from e


def validate_tui_log(
    tui_log: pathlib.Path | None,
    as_json: bool,
    tui_flag: bool,
    dry_run: bool = False,
) -> pathlib.Path | None:
    """Validate --tui-log option and resolve path if valid."""
    if not tui_log:
        return None
    if as_json:
        raise click.ClickException("--tui-log cannot be used with --jsonl/--json")
    if not tui_flag:
        raise click.ClickException("--tui-log requires --tui")
    if dry_run:
        raise click.ClickException("--tui-log cannot be used with --dry-run")
    # Validate path upfront (fail fast)
    resolved = tui_log.expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.touch()  # Verify writable
    except OSError as e:
        raise click.ClickException(f"Cannot write to {resolved}: {e}") from e
    return resolved


def validate_show_output(show_output: bool, tui_flag: bool, as_json: bool, quiet: bool) -> None:
    """Validate --show-output mutual exclusions."""
    if show_output and tui_flag:
        raise click.ClickException("--show-output and --tui are mutually exclusive")
    if show_output and as_json:
        raise click.ClickException("--show-output and --jsonl/--json are mutually exclusive")
    if show_output and quiet:
        raise click.ClickException("--show-output and --quiet are mutually exclusive")


def validate_display_mode(tui_flag: bool, as_json: bool) -> None:
    """Validate --tui and --jsonl are mutually exclusive."""
    if tui_flag and as_json:
        raise click.ClickException("--tui and --jsonl are mutually exclusive")


def resolve_on_error(fail_fast: bool, keep_going: bool) -> OnError:
    """Resolve --fail-fast / --keep-going flags to OnError enum.

    Validates mutual exclusion and returns the appropriate enum value.
    Default (neither flag) is fail-fast.
    """
    if fail_fast and keep_going:
        raise click.ClickException("--fail-fast and --keep-going are mutually exclusive")
    return OnError.KEEP_GOING if keep_going else OnError.FAIL


def tui_prewarm_width(
    stages: list[str] | None,
    *,
    single_stage: bool,
    all_pipelines: bool,
    on_error: OnError,
    checkout_missing: bool,
) -> int:
    """Return the width the engine will compute for this run, for the TUI's worker pre-warm.

    Like the engine, it first restores missing tracked files when asked to, then sizes the
    pool over the stages the engine runs (``engine_graph.get_runnable_stages``), so the pool
    warmed before Textual starts is the pool the engine then uses. Checking the tracked files
    that exist stays the engine's.
    """
    if checkout_missing:
        executor_core.restore_missing_tracked_files(project.get_project_root())
    runnable, _ = engine_graph.get_runnable_stages(
        cli_helpers.get_all_stages(),
        cli_helpers.require_workspace_store(),
        all_pipelines=all_pipelines,
        on_error=on_error,
    )
    if not runnable:
        # The engine stops at the missing dependency before it starts a worker.
        return 1
    stage_dag = engine_graph.get_stage_dag(engine_graph.build_graph(runnable))
    targets = [stage for stage in stages if stage in stage_dag] if stages else None
    if stages and not targets:
        return 1
    order = engine_graph.get_execution_order(stage_dag, targets, single_stage=single_stage)
    return engine_graph.get_width(stage_dag, order, limit=executor_core.configured_worker_count())


class DryRunJsonStageOutput(TypedDict):
    """JSON output for a single stage in dry-run mode."""

    would_run: bool
    reason: str


class DryRunJsonOutput(TypedDict):
    """JSON output for pivot run --dry-run --json."""

    stages: dict[str, DryRunJsonStageOutput]


# ---------------------------------------------------------------------------
# Shared utilities for run.py and repro.py
# ---------------------------------------------------------------------------


class JsonlSink:
    """Async sink that calls a callback for each stage event.

    Converts internal engine events to the documented JSONL schema (pivot/types.py).
    """

    _callback: Callable[[dict[str, object]], None]

    def __init__(self, callback: Callable[[dict[str, object]], None]) -> None:
        self._callback = callback

    async def handle(self, event: OutputEvent) -> None:
        """Convert stage events to JSONL records.

        Note: Engine uses "stage_started"/"stage_completed" internally, but
        JSONL schema (pivot/types.py) uses "stage_start"/"stage_complete".
        We convert to match the documented public API.
        """
        match event["type"]:
            case "stage_started":
                # Convert to documented JSONL schema type
                self._callback(
                    {
                        "type": "stage_start",  # Matches RunEventType.STAGE_START
                        "stage": event["stage"],
                        "index": event["index"],
                        "total": event["total"],
                    }
                )
            case "stage_completed":
                # Convert to documented JSONL schema type
                self._callback(
                    {
                        "type": "stage_complete",  # Matches RunEventType.STAGE_COMPLETE
                        "stage": event["stage"],
                        "status": event["status"].value,
                        "reason": event["reason"],
                        "duration_ms": event["duration_ms"],
                        "index": event["index"],
                        "total": event["total"],
                    }
                )
            case "pipeline_reloaded":
                # All fields are JSON-serializable (strings and lists of strings)
                self._callback(dict(event))
            case "engine_state_changed":
                # Convert state enum to string for JSON serialization
                self._callback(
                    {
                        "type": "engine_state_changed",
                        "state": event["state"].value,
                    }
                )
            case _:
                pass  # Ignore other event types (log_line, stage_state_changed)

    async def close(self) -> None:
        """No cleanup needed."""


def configure_result_collector(eng: engine.Engine) -> sinks.ResultCollectorSink:
    """Add ResultCollectorSink to collect execution results."""
    result_sink = sinks.ResultCollectorSink()
    eng.add_sink(result_sink)
    return result_sink


def configure_output_sink(
    eng: engine.Engine,
    *,
    quiet: bool,
    as_json: bool,
    use_console: bool,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    show_output: bool = False,
) -> None:
    """Configure output sinks based on display mode."""
    import rich.console

    # JSON sink is always added when as_json=True, regardless of quiet mode
    if as_json and jsonl_callback:
        eng.add_sink(JsonlSink(callback=jsonl_callback))
        return

    if quiet:
        return

    if use_console:
        console = rich.console.Console()
        if console.is_terminal:
            eng.add_sink(sinks.LiveConsoleSink(console=console, show_output=show_output))
        else:
            eng.add_sink(sinks.StaticConsoleSink(console=console, show_output=show_output))


def convert_results(
    stage_results: dict[str, StageCompleted],
) -> dict[str, ExecutionSummary]:
    """Convert StageCompleted events to ExecutionSummary.

    This is used to translate engine event results into the ExecutionSummary
    format expected by the CLI commands.
    """
    from pivot.executor import core as executor_core

    return {
        name: executor_core.ExecutionSummary(
            status=event["status"],
            reason=event["reason"],
            input_hash=event["input_hash"],
        )
        for name, event in stage_results.items()
    }


# ---------------------------------------------------------------------------
# TUI + Engine threading helper
# ---------------------------------------------------------------------------


def run_tui_with_engine(
    app: Any,
    engine_fn: Callable[[threading.Event], Coroutine[Any, Any, Any]],
    socket_path: pathlib.Path,
    *,
    result_future: concurrent.futures.Future[Any] | None = None,
    conditions: Callable[[], Mapping[str, object]] = dict,
    oneshot: bool = True,
    suppress_keyboard_interrupt: bool = False,
) -> None:
    """Run the TUI, engine and event poller on their required three threads.

    - Main thread: Textual TUI (required for signal handlers).
    - Engine thread: its own AnyIO loop runs ``engine_fn`` and the RPC server.
    - Poller thread: its own AnyIO loop polls engine events and posts TUI updates.

    ``poller_stop`` is set when the engine finishes or the TUI exits. It stops a connected
    poller and ends retries when a one-shot engine has already removed its socket.

    Args:
        app: PivotApp instance (connects its own client on mount)
        engine_fn: Async function that runs the engine. Receives socket_ready Event.
                   Must call ``socket_ready.set()`` after RPC socket is listening.
        socket_path: Path to Unix socket
        result_future: Optional Future for returning engine results to caller.
                       If provided, engine thread joins after TUI exits.
        conditions: Current engine conditions used to leave out superseded held records.
        oneshot: If True, poller stops on connection close (run mode).
                 If False, poller retries on disconnect (watch mode).
        suppress_keyboard_interrupt: If True, suppress KeyboardInterrupt around app.run().
    """
    import pivot_tui.run as tui_run
    from pivot_tui.event_poller import EventPoller
    from pivot_tui.rpc_client_impl import RpcPivotClient

    socket_ready = threading.Event()
    poller_stop = threading.Event()
    # The exception that ended the engine; raised once the TUI has exited.
    engine_errors = list[BaseException]()

    def engine_thread_target() -> None:
        """Run the async Engine in a background thread with its own event loop."""
        error_text: str | None = None
        try:
            result = anyio.run(engine_fn, socket_ready)
            if result_future is not None and not result_future.done():
                result_future.set_result(result)
        except BaseException as e:
            error_text = f"{type(e).__name__}: {e}"
            engine_errors.append(e)
            if result_future is not None and not result_future.done():
                result_future.set_exception(e)
        finally:
            socket_ready.set()  # Unblock main thread even on failure
            poller_stop.set()
            # Signal TUI to exit when engine completes (success or failure)
            with contextlib.suppress(Exception):
                app.post_message(tui_run.TuiShutdown(error=error_text))

    async def wait_for_poller_stop(timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the stop signal; True once it is set."""
        with anyio.move_on_after(timeout):
            while not poller_stop.is_set():
                await anyio.sleep(_POLLER_STOP_CHECK_S)
        return poller_stop.is_set()

    async def stop_poller_on_signal(poller: EventPoller) -> None:
        while not poller_stop.is_set():
            await anyio.sleep(_POLLER_STOP_CHECK_S)
        poller.stop()

    def poller_thread_target() -> None:
        async def poller_main() -> None:
            poller_client = RpcPivotClient()
            for _attempt in range(_POLLER_CONNECT_ATTEMPTS):
                try:
                    await poller_client.connect(socket_path)
                    break
                except OSError:
                    if await wait_for_poller_stop(_POLLER_CONNECT_RETRY_S):
                        # A one-shot engine can finish and remove its socket first.
                        logger.debug("Engine finished before the event poller connected")
                        return
            else:
                logger.error("Poller failed to connect after %d attempts", _POLLER_CONNECT_ATTEMPTS)
                return
            poller = EventPoller(
                poller_client,
                lambda msg: app.post_message(tui_run.TuiUpdate(msg)),
                oneshot=oneshot,
            )
            app.set_event_poller(poller)
            try:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(stop_poller_on_signal, poller)
                    await poller.run()
                    tg.cancel_scope.cancel()
            finally:
                with contextlib.suppress(Exception):
                    await poller_client.disconnect()

        try:
            anyio.run(poller_main)
        except Exception:
            logger.debug("Event poller stopped", exc_info=True)

    engine_thread = threading.Thread(target=engine_thread_target, daemon=True)
    poller_thread = threading.Thread(
        target=poller_thread_target, name=POLLER_THREAD_NAME, daemon=True
    )
    # Console handlers print as usual until the TUI starts and remember what they printed, so a
    # warning the engine logs both before the TUI starts and while it runs reaches the terminal
    # once. A "Pipeline invalid" record held while the TUI ran prints only if ``conditions``
    # still reports that error when it exits.
    with defer_console_logging(conditions, hold_now=False) as console_logging:
        engine_thread.start()
        if not socket_ready.wait(timeout=10.0):
            logger.warning("Engine socket not ready after 10s, poller will retry connect")
        poller_thread.start()

        # Run TUI in main thread (required for signal handlers). This waits for every console
        # write that passed the filter before Textual redirects stderr.
        console_logging.start_holding()
        try:
            if suppress_keyboard_interrupt:
                with contextlib.suppress(KeyboardInterrupt):
                    app.run()
            else:
                app.run()
        finally:
            poller_stop.set()

    poller_thread.join()
    if result_future is not None:
        engine_thread.join(timeout=5.0)

    # A crashed engine fails the command with its own error, not the TUI's return code.
    if engine_errors:
        raise engine_errors[0]

    # Textual sets a non-zero return code when the app exits on an unhandled exception;
    # surface it so callers (scripts, agents) can tell the TUI died.
    return_code = app.return_code
    if return_code:
        raise SystemExit(return_code)
