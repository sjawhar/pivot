"""Pivot run command - single-stage execution without DAG resolution."""

from __future__ import annotations

import contextlib
import datetime
import logging
import pathlib
import sys
import time
from typing import TYPE_CHECKING

import click

from pivot import config
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.executor import prepare_workers
from pivot.types import (
    DisplayMode,
    ExecutionResultEvent,
    OnError,
    RunEventType,
    SchemaVersionEvent,
    StageStatus,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from pivot.executor import ExecutionSummary


@contextlib.contextmanager
def _suppress_stderr_logging() -> Generator[None]:
    """Suppress logging to stderr while TUI is active."""
    root = logging.getLogger()
    removed_handlers = list[logging.Handler]()

    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)  # pyright: ignore[reportUnknownArgumentType]
            if stream in (sys.stderr, sys.stdout):
                root.removeHandler(handler)  # pyright: ignore[reportUnknownArgumentType]
                removed_handlers.append(handler)  # pyright: ignore[reportUnknownArgumentType]
    try:
        yield
    finally:
        for handler in removed_handlers:
            root.addHandler(handler)


# JSONL schema version for forward compatibility
_JSONL_SCHEMA_VERSION = 1


def _run_with_tui(
    stages_list: list[str],
    cache_dir: pathlib.Path | None,
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    on_error: OnError = OnError.KEEP_GOING,
    allow_uncached_incremental: bool = False,
    checkout_missing: bool = False,
) -> dict[str, ExecutionSummary] | None:
    """Run stages with TUI display."""
    import queue as thread_queue
    import threading
    import uuid

    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    resolved_cache_dir = cache_dir or config.get_cache_dir()

    # Pre-warm loky executor before starting Textual TUI.
    prepare_workers(len(stages_list))

    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()
    cancel_event = threading.Event()
    run_id = str(uuid.uuid4())[:8]

    def executor_func() -> dict[str, ExecutionSummary]:
        with engine.Engine() as eng:
            eng.set_cancel_event(cancel_event)
            eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
            return eng.run_once(
                stages=stages_list,
                single_stage=True,
                cache_dir=resolved_cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

    with _suppress_stderr_logging():
        return run_tui.run_with_tui(
            stages_list, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, required=True, shell_complete=completion.complete_stages)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option("--force", "-f", is_flag=True, help="Force re-run of stages, ignoring cache")
@click.option(
    "--display",
    type=click.Choice([e.value for e in DisplayMode]),
    default=None,
    help="Display mode: tui (interactive) or plain (streaming text). Auto-detects if not specified.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--tui-log",
    type=click.Path(path_type=pathlib.Path),
    help="Write TUI messages to JSONL file for monitoring",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Defer lock files to pending dir for faster iteration. Run 'pivot commit' to finalize.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip caching outputs entirely for maximum iteration speed. Outputs won't be cached.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop execution on first failure (default: keep going).",
)
@click.option(
    "--allow-uncached-incremental",
    is_flag=True,
    help="Allow running stages with IncrementalOut files that exist but aren't in cache.",
)
@click.option(
    "--checkout-missing",
    is_flag=True,
    help="Restore tracked files that don't exist on disk from cache before running.",
)
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    cache_dir: pathlib.Path | None,
    force: bool,
    display: str | None,
    as_json: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    fail_fast: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> None:
    """Execute specified stages without dependency resolution.

    STAGES are required - specify which stages to run.
    Unlike 'pivot repro', this command runs stages directly without resolving
    their dependencies. Use this for testing individual stages.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet

    stages_list = list(stages)
    cli_helpers.validate_stages_exist(stages_list)

    # Validate tui_log requires TUI mode
    if tui_log:
        if as_json:
            raise click.ClickException("--tui-log cannot be used with --json")
        if display == DisplayMode.PLAIN.value:
            raise click.ClickException("--tui-log cannot be used with --display=plain")
        tui_log = tui_log.expanduser().resolve()
        try:
            tui_log.parent.mkdir(parents=True, exist_ok=True)
            tui_log.touch()
        except OSError as e:
            raise click.ClickException(f"Cannot write to {tui_log}: {e}") from e

    # Default is KEEP_GOING, --fail-fast changes to FAIL
    on_error = OnError.FAIL if fail_fast else OnError.KEEP_GOING

    # Determine display mode
    display_mode = DisplayMode(display) if display else None

    from pivot.tui import run as run_tui

    use_tui = run_tui.should_use_tui(display_mode) and not as_json

    if use_tui:
        results = _run_with_tui(
            stages_list,
            cache_dir,
            force=force,
            tui_log=tui_log,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    elif as_json:
        # JSONL streaming mode
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

        start_time = time.perf_counter()
        with engine.Engine() as eng:
            eng.add_sink(sinks.JsonlSink(callback=cli_helpers.emit_jsonl))
            results = eng.run_once(
                stages=stages_list,
                single_stage=True,
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        total_duration_ms = (time.perf_counter() - start_time) * 1000

        ran = sum(1 for r in results.values() if r["status"] == StageStatus.RAN)
        skipped = sum(1 for r in results.values() if r["status"] == StageStatus.SKIPPED)
        failed = sum(1 for r in results.values() if r["status"] == StageStatus.FAILED)

        cli_helpers.emit_jsonl(
            ExecutionResultEvent(
                type=RunEventType.EXECUTION_RESULT,
                ran=ran,
                skipped=skipped,
                failed=failed,
                total_duration_ms=total_duration_ms,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            )
        )
    else:
        from pivot.executor import core as executor_core
        from pivot.tui import console as tui_console

        console: tui_console.Console | None = None
        if not quiet:
            console = tui_console.Console()

        start_time = time.perf_counter()
        with engine.Engine() as eng:
            if console:
                eng.add_sink(sinks.ConsoleSink(console))
            results = eng.run_once(
                stages=stages_list,
                single_stage=True,
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

        if console and results:
            ran, cached, blocked, failed = executor_core.count_results(results)
            total_duration = time.perf_counter() - start_time
            console.summary(ran, cached, blocked, failed, total_duration)

    if not results and show_human_output and not use_tui:
        click.echo("No stages to run")
