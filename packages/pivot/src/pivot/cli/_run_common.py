"""Shared helpers for run and repro CLI commands."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
from collections.abc import Callable, Coroutine  # noqa: TC003 - used in function signatures
from typing import TYPE_CHECKING, Any, TextIO, TypedDict, cast

import anyio
import click

from pivot import discovery
from pivot.cli import decorators as cli_decorators
from pivot.engine import engine, sinks
from pivot.types import OnError

if TYPE_CHECKING:
    import concurrent.futures
    import pathlib
    from collections.abc import Generator

    import networkx as nx

    from pivot.engine.types import OutputEvent, StageCompleted
    from pivot.executor import ExecutionSummary
    from pivot.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stderr_logging() -> Generator[None]:
    """Suppress logging to stdout/stderr while TUI is active.

    Textual takes over the terminal, so stdout/stderr writes appear as garbage
    in the upper-left corner. This temporarily removes StreamHandlers
    that write to stdout or stderr and restores them on exit.
    """
    root = logging.getLogger()
    removed_handlers = list[logging.Handler]()

    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            # Cast to known type - stdlib handlers writing to stderr/stdout use TextIO
            # Use string literal because generic type isn't narrowed by isinstance
            stream_handler = cast("logging.StreamHandler[TextIO]", handler)
            if stream_handler.stream in (sys.stderr, sys.stdout):
                root.removeHandler(stream_handler)
                removed_handlers.append(stream_handler)
    try:
        yield
    finally:
        for handler in removed_handlers:
            root.addHandler(handler)


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


def ensure_stages_registered() -> None:
    """Ensure a Pipeline is discovered and in context.

    If no Pipeline is in context, attempts discovery and stores the result.
    """
    if cli_decorators.get_pipeline_from_context() is not None:
        return
    try:
        pipeline: Pipeline | None = discovery.discover_pipeline()
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
        eng.add_sink(sinks.ConsoleSink(console=rich.console.Console(), show_output=show_output))


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
    oneshot: bool = True,
    suppress_keyboard_interrupt: bool = False,
) -> None:
    """Run TUI with engine in background thread.

    Three-thread model:
    - Main thread: Textual TUI (required for signal handlers)
    - Engine thread: runs engine_fn with its own anyio event loop
    - Poller thread: connects client, polls events, feeds TUI

    Args:
        app: PivotApp instance (connects its own client on mount)
        engine_fn: Async function that runs the engine. Receives socket_ready Event.
                   Must call ``socket_ready.set()`` after RPC socket is listening.
        socket_path: Path to Unix socket
        result_future: Optional Future for returning engine results to caller.
                       If provided, engine thread joins after TUI exits.
        oneshot: If True, poller stops on connection close (run mode).
                 If False, poller retries on disconnect (watch mode).
        suppress_keyboard_interrupt: If True, suppress KeyboardInterrupt around app.run().
    """
    import pivot_tui.run as tui_run
    from pivot_tui.event_poller import EventPoller
    from pivot_tui.rpc_client_impl import RpcPivotClient

    socket_ready = threading.Event()

    def engine_thread_target() -> None:
        """Run the async Engine in a background thread with its own event loop."""
        try:
            result = anyio.run(engine_fn, socket_ready)
            if result_future is not None and not result_future.done():
                result_future.set_result(result)
        except BaseException as e:
            if result_future is not None and not result_future.done():
                result_future.set_exception(e)
        finally:
            socket_ready.set()  # Unblock main thread even on failure
            # Signal TUI to exit when engine completes (success or failure)
            with contextlib.suppress(Exception):
                app.post_message(tui_run.TuiShutdown())

    # Start engine in background thread
    engine_thread = threading.Thread(target=engine_thread_target, daemon=True)
    engine_thread.start()
    if not socket_ready.wait(timeout=10.0):
        logger.warning("Engine socket not ready after 10s, poller will retry connect")

    def poller_thread_target() -> None:
        async def poller_main() -> None:
            poller_client = RpcPivotClient()
            for _attempt in range(30):
                try:
                    await poller_client.connect(socket_path)
                    break
                except OSError:
                    await anyio.sleep(1.0)
            else:
                logger.error("Poller failed to connect after 30 attempts")
                return
            poller = EventPoller(
                poller_client,
                lambda msg: app.post_message(tui_run.TuiUpdate(msg)),
                oneshot=oneshot,
            )
            app.set_event_poller(poller)
            try:
                await poller.run()
            finally:
                with contextlib.suppress(Exception):
                    await poller_client.disconnect()

        try:
            anyio.run(poller_main)
        except Exception:
            logger.debug("Event poller stopped", exc_info=True)

    poller_thread = threading.Thread(target=poller_thread_target, daemon=True)
    poller_thread.start()

    # Run TUI in main thread (required for signal handlers)
    if suppress_keyboard_interrupt:
        with contextlib.suppress(KeyboardInterrupt), suppress_stderr_logging():
            app.run()
    else:
        with suppress_stderr_logging():
            app.run()

    # For oneshot mode, wait for engine thread to finish
    poller_thread.join(timeout=2.0)
    if result_future is not None:
        engine_thread.join(timeout=5.0)
