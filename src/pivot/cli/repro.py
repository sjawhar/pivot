"""Pivot repro command - DAG-aware pipeline reproduction."""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import pathlib
import sys
import time
from typing import TYPE_CHECKING, TypedDict

import click

from pivot import config, discovery, registry
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

    import networkx as nx

    from pivot.executor import ExecutionSummary


@contextlib.contextmanager
def _suppress_stderr_logging() -> Generator[None]:
    """Suppress logging to stderr while TUI is active.

    Textual takes over the terminal, so stderr writes appear as garbage
    in the upper-left corner. This temporarily removes StreamHandlers
    that write to stderr and restores them on exit.
    """
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


logger = logging.getLogger(__name__)


def _compute_dag_levels(graph: nx.DiGraph[str]) -> dict[str, int]:
    """Compute DAG level for each stage.

    Level 0: stages with no dependencies
    Level N: stages whose dependencies are all at level < N

    Stages at the same level can run in parallel - there's no ordering between them.
    """
    import networkx as nx

    levels: dict[str, int] = {}
    # Process in topological order (dependencies before dependents)
    for stage in nx.dfs_postorder_nodes(graph):
        # successors = what this stage depends on (edges go consumer -> producer)
        dep_levels = [levels[dep] for dep in graph.successors(stage) if dep in levels]
        levels[stage] = max(dep_levels, default=-1) + 1
    return levels


def _sort_for_display(execution_order: list[str], graph: nx.DiGraph[str]) -> list[str]:
    """Sort stages for TUI display: group matrix variants while respecting DAG structure.

    Uses DAG levels (not arbitrary execution order) so parallel-capable stages
    are treated as equals. Matrix variants are grouped at the level of their
    earliest member.
    """
    from pivot.tui.types import parse_stage_name

    levels = _compute_dag_levels(graph)

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
    """Auto-discover and register stages if none are registered."""
    if not discovery.has_registered_stages():
        try:
            discovered = discovery.discover_and_register()
            if discovered:
                logger.info(f"Loaded pipeline from {discovered}")
        except discovery.DiscoveryError as e:
            raise click.ClickException(str(e)) from e


def _validate_stages(stages_list: list[str] | None) -> None:
    """Validate stage arguments."""
    cli_helpers.validate_stages_exist(stages_list)


def _output_explain(
    stages_list: list[str] | None,
    force: bool = False,
    allow_missing: bool = False,
) -> None:
    """Output detailed stage explanations using status logic."""
    from pivot import status as status_mod
    from pivot.cli import status as status_cli
    from pivot.engine import graph as engine_graph

    # Validate dependencies exist when allow_missing is False
    if not allow_missing:
        registry.REGISTRY.build_dag(validate=True)

    # Build graph once for all explanations
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage=False,  # repro always runs with dependencies
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )
    status_cli.output_explain_text(explanations)


def _run_with_tui(
    stages_list: list[str] | None,
    cache_dir: pathlib.Path | None,
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    on_error: OnError = OnError.FAIL,
    allow_uncached_incremental: bool = False,
    checkout_missing: bool = False,
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with TUI display."""
    import queue as thread_queue
    import threading

    from pivot import dag
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Get execution order for stage names
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=False)

    if not execution_order:
        return {}

    resolved_cache_dir = cache_dir or config.get_cache_dir()

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    prepare_workers(len(execution_order))

    # tui_queue is inter-thread only (executor -> TUI reader), no cross-process IPC needed.
    # Using stdlib queue.Queue avoids Manager subprocess dependency issues.
    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()

    # Cancel event allows TUI to signal executor to stop scheduling new stages
    cancel_event = threading.Event()

    import uuid

    # Generate run_id for TUI tracking
    run_id = str(uuid.uuid4())[:8]

    # Create executor function using Engine with TuiSink
    def executor_func() -> dict[str, ExecutionSummary]:
        with engine.Engine() as eng:
            # Share TUI's cancel_event so TUI can signal cancellation
            eng.set_cancel_event(cancel_event)
            eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
            return eng.run_once(
                stages=stages_list,
                single_stage=False,  # repro always runs with dependencies
                cache_dir=resolved_cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

    # Sort for display: group matrix variants together while preserving DAG structure
    display_order = _sort_for_display(execution_order, graph)

    with _suppress_stderr_logging():
        return run_tui.run_with_tui(
            display_order, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


def _run_watch_with_tui(
    stages_list: list[str] | None,
    cache_dir: pathlib.Path | None,  # noqa: ARG001 - cache_dir not yet passed to Engine
    debounce: int,  # noqa: ARG001 - debounce not yet used by FilesystemSource
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,  # noqa: ARG001 - not yet supported in Engine watch mode
    no_cache: bool = False,  # noqa: ARG001 - not yet supported in Engine watch mode
    on_error: OnError = OnError.FAIL,
    serve: bool = False,
) -> None:
    """Run watch mode with TUI display.

    Note: Several parameters (cache_dir, debounce, no_commit, no_cache) are
    retained for CLI signature compatibility but not currently used by Engine watch mode.
    """
    # Suppress unused parameter warnings - retained for CLI compatibility
    _ = cache_dir, debounce, no_commit, no_cache

    import queue as thread_queue
    import uuid

    from pivot import dag
    from pivot.engine import graph as engine_graph
    from pivot.engine import sources
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Get execution order to calculate the correct number of workers
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=False)

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    prepare_workers(len(execution_order) if execution_order else 1)

    # tui_queue is inter-thread only (executor -> TUI reader), no cross-process IPC needed.
    # Using stdlib queue.Queue avoids Manager subprocess dependency issues.
    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()

    # Generate run_id for TUI tracking
    run_id = str(uuid.uuid4())[:8]

    # Create the Engine
    with engine.Engine() as eng:
        # Set keep-going mode based on on_error
        eng.set_keep_going(on_error == OnError.KEEP_GOING)

        # Build bipartite graph for watch paths
        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        bipartite_graph = engine_graph.build_graph(all_stages)
        watch_paths = engine_graph.get_watch_paths(bipartite_graph)

        # Add FilesystemSource for watching file changes
        filesystem_source = sources.FilesystemSource(watch_paths)
        eng.add_source(filesystem_source)

        # Add OneShotSource for initial run if force is set
        if force:
            initial_source = sources.OneShotSource(
                stages=stages_list,
                force=True,
                reason="watch:initial:forced",
            )
            eng.add_source(initial_source)

        # Add sinks for TUI updates
        eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
        eng.add_sink(sinks.WatchSink(tui_queue=tui_queue))

        # Sort for display: group matrix variants together while preserving DAG structure
        display_order = _sort_for_display(execution_order, graph) if execution_order else None

        with _suppress_stderr_logging():
            run_tui.run_watch_tui(
                eng,
                tui_queue,
                tui_log=tui_log,
                stage_names=display_order,
                no_commit=False,  # TODO: Support no_commit in Engine watch mode
                serve=serve,
            )


class DryRunJsonStageOutput(TypedDict):
    """JSON output for a single stage in dry-run mode."""

    would_run: bool
    reason: str


class DryRunJsonOutput(TypedDict):
    """JSON output for pivot repro --dry-run --json."""

    stages: dict[str, DryRunJsonStageOutput]


def _dry_run(
    stages_list: list[str] | None,
    force: bool,
    as_json: bool,
    allow_missing: bool,
) -> None:
    """Show what would run without executing."""
    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph

    _validate_stages(stages_list)

    # Validate dependencies exist when allow_missing is False
    if not allow_missing:
        registry.REGISTRY.build_dag(validate=True)

    # Build bipartite graph for consistent execution order with Engine
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list, single_stage=False, force=force, allow_missing=allow_missing, graph=graph
    )

    if not explanations:
        if as_json:
            click.echo(json.dumps(DryRunJsonOutput(stages={})))
        else:
            click.echo("No stages to run")
        return

    if as_json:
        output = DryRunJsonOutput(
            stages={
                exp["stage_name"]: DryRunJsonStageOutput(
                    would_run=exp["will_run"],
                    reason=exp["reason"] or "unchanged",
                )
                for exp in explanations
            }
        )
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo("Would run:")
        for exp in explanations:
            status = "would run" if exp["will_run"] else "would skip"
            reason = exp["reason"] or "unchanged"
            click.echo(f"  {exp['stage_name']}: {status} ({reason})")


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would run without executing")
@click.option(
    "--explain", "-e", is_flag=True, help="Show detailed breakdown of why stages would run"
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-run of stages, ignoring cache (in --watch mode, first run only)",
)
@click.option(
    "--watch",
    "-w",
    is_flag=True,
    help="Watch for file changes and re-run affected stages",
)
@click.option(
    "--debounce",
    type=click.IntRange(min=0),
    default=None,
    help="Debounce delay in milliseconds (for --watch mode)",
)
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
    "--keep-going",
    "-k",
    is_flag=True,
    help="Continue running stages after failures; skip only downstream dependents.",
)
@click.option(
    "--serve",
    is_flag=True,
    help="Start RPC server for agent control (requires --watch). Creates Unix socket at .pivot/agent.sock",
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
@click.option(
    "--allow-missing",
    is_flag=True,
    help="Allow missing dep files if tracked (.pvt exists). Only affects --dry-run.",
)
@click.pass_context
def repro(
    ctx: click.Context,
    stages: tuple[str, ...],
    cache_dir: pathlib.Path | None,
    dry_run: bool,
    explain: bool,
    force: bool,
    watch: bool,
    debounce: int | None,
    display: str | None,  # Click passes string, converted to DisplayMode below
    as_json: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    keep_going: bool,
    serve: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
    allow_missing: bool,
) -> None:
    """Reproduce pipeline stages with their dependencies.

    If STAGES are provided, runs those stages and their dependencies.
    If no stages are provided, runs all stages in the pipeline.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet
    debounce = debounce if debounce is not None else config.get_watch_debounce()

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list)

    # Validate tui_log requires TUI mode
    if tui_log:
        if as_json:
            raise click.ClickException("--tui-log cannot be used with --json")
        if display == DisplayMode.PLAIN.value:
            raise click.ClickException("--tui-log cannot be used with --display=plain")
        if dry_run:
            raise click.ClickException("--tui-log cannot be used with --dry-run")
        # Validate path upfront (fail fast)
        tui_log = tui_log.expanduser().resolve()
        try:
            tui_log.parent.mkdir(parents=True, exist_ok=True)
            tui_log.touch()  # Verify writable
        except OSError as e:
            raise click.ClickException(f"Cannot write to {tui_log}: {e}") from e

    # Validate --serve requires --watch
    if serve and not watch:
        raise click.ClickException("--serve requires --watch mode")

    # Validate --allow-missing requires --dry-run
    if allow_missing and not dry_run:
        raise click.ClickException("--allow-missing can only be used with --dry-run")

    # Handle explain mode (with or without dry-run) - show explanations without execution
    if explain:
        _output_explain(stages_list, force, allow_missing=allow_missing)
        return

    # Handle dry-run mode (without explain) - terse output
    if dry_run:
        _dry_run(stages_list, force, as_json, allow_missing)
        return

    on_error = OnError.KEEP_GOING if keep_going else OnError.FAIL

    if watch:
        from pivot.tui import run as run_tui

        display_mode = DisplayMode(display) if display else None
        use_tui = run_tui.should_use_tui(display_mode) and not as_json

        # Validate --serve requires TUI mode
        if serve and not use_tui:
            raise click.ClickException(
                "--serve requires TUI mode (not compatible with --json or --display=plain)"
            )

        if use_tui:
            try:
                _run_watch_with_tui(
                    stages_list,
                    cache_dir,
                    debounce,
                    force,
                    tui_log=tui_log,
                    no_commit=no_commit,
                    no_cache=no_cache,
                    on_error=on_error,
                    serve=serve,
                )
            except KeyboardInterrupt:
                if show_human_output:
                    click.echo("\nWatch mode stopped.")
        else:
            from pivot.engine import graph as engine_graph
            from pivot.engine import sources

            # Create the Engine
            with engine.Engine() as eng:
                # Set keep-going mode based on on_error
                eng.set_keep_going(on_error == OnError.KEEP_GOING)

                # Build bipartite graph for watch paths
                all_stages = {
                    name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()
                }
                bipartite_graph = engine_graph.build_graph(all_stages)
                watch_paths = engine_graph.get_watch_paths(bipartite_graph)

                # Add FilesystemSource for watching file changes
                filesystem_source = sources.FilesystemSource(watch_paths)
                eng.add_source(filesystem_source)

                # Add OneShotSource for initial run if force is set
                if force:
                    initial_source = sources.OneShotSource(
                        stages=stages_list,
                        force=True,
                        reason="watch:initial:forced",
                    )
                    eng.add_source(initial_source)

                # Add console sink for plain display (unless JSON output)
                if not as_json:
                    from pivot.tui import console as tui_console

                    console = tui_console.Console()
                    eng.add_sink(sinks.ConsoleSink(console))

                try:
                    eng.run_loop()
                except KeyboardInterrupt:
                    pass  # Normal exit via Ctrl+C
                finally:
                    eng.shutdown()
                    if show_human_output:
                        click.echo("\nWatch mode stopped.")
        return

    # Determine display mode
    display_mode = DisplayMode(display) if display else None

    # Normal execution (with optional explain mode)
    from pivot.tui import run as run_tui

    # Disable TUI when JSON output is requested
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
                single_stage=False,  # repro always runs with dependencies
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        total_duration_ms = (time.perf_counter() - start_time) * 1000

        # Emit final execution result
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

        # Add ConsoleSink for stage progress display (unless quiet)
        console: tui_console.Console | None = None
        if not quiet:
            console = tui_console.Console()

        start_time = time.perf_counter()
        with engine.Engine() as eng:
            if console:
                eng.add_sink(sinks.ConsoleSink(console))
            results = eng.run_once(
                stages=stages_list,
                single_stage=False,  # repro always runs with dependencies
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

        # Print summary for plain mode (TUI/JSON modes handle this differently)
        if console and results:
            ran, cached, blocked, failed = executor_core.count_results(results)
            total_duration = time.perf_counter() - start_time
            console.summary(ran, cached, blocked, failed, total_duration)

    if not results and show_human_output and not use_tui:
        click.echo("No stages to run")
