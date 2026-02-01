from __future__ import annotations

import contextlib
import datetime
import json
import logging
import pathlib
import queue as thread_queue
import sys
import threading
import time
import uuid
from typing import TYPE_CHECKING, TypedDict

import click

from pivot import config, discovery
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.engine import sources as engine_sources
from pivot.executor import prepare_workers
from pivot.types import (
    ExecutionResultEvent,
    OnError,
    RunEventType,
    SchemaVersionEvent,
    StageStatus,
    TuiMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import networkx as nx

    from pivot.executor import ExecutionSummary
    from pivot.tui import console as tui_console
    from pivot.types import TuiQueue


def _configure_result_collector(eng: engine.Engine) -> sinks.ResultCollectorSink:
    """Add ResultCollectorSink to collect execution results."""
    result_sink = sinks.ResultCollectorSink()
    eng.add_sink(result_sink)
    return result_sink


def _configure_output_sink(
    eng: engine.Engine,
    *,
    quiet: bool,
    as_json: bool,
    tui: bool,
    tui_queue: TuiQueue | None,
    run_id: str | None,
    console: tui_console.Console | None,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    watch: bool,
) -> None:
    """Configure output sinks based on display mode."""
    # JSON sink is always added when as_json=True, regardless of quiet mode
    # (quiet only suppresses human-readable console output)
    if as_json and jsonl_callback:
        eng.add_sink(sinks.JsonlSink(callback=jsonl_callback))
        return

    if quiet:
        return
    elif tui and tui_queue and run_id:
        eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
        if watch:
            eng.add_sink(sinks.WatchSink(tui_queue=tui_queue))
    elif console:
        eng.add_sink(sinks.ConsoleSink(console))


def _configure_watch_sources(
    eng: engine.Engine,
    watch_paths: list[pathlib.Path],
    debounce: int,
    *,
    force: bool,
    stages: list[str] | None,
) -> None:
    """Configure sources for watch mode."""
    eng.add_source(engine_sources.FilesystemSource(watch_paths, debounce=debounce))
    if force:
        eng.add_source(
            engine_sources.OneShotSource(
                stages=stages,
                force=True,
                reason="watch:initial:forced",
            )
        )


def _configure_oneshot_source(
    eng: engine.Engine,
    stages: list[str] | None,
    *,
    force: bool,
    single_stage: bool,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> None:
    """Configure OneShotSource for non-watch mode."""
    eng.add_source(
        engine_sources.OneShotSource(
            stages=stages,
            force=force,
            reason="cli",
            single_stage=single_stage,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    )


def _run_pipeline(
    stages_list: list[str] | None,
    *,
    watch: bool,
    single_stage: bool,
    force: bool,
    quiet: bool,
    tui: bool,
    as_json: bool,
    debounce: int,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    serve: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with unified watch/non-watch execution.

    Returns execution results for non-watch mode, None for watch mode.
    """
    from pivot import dag
    from pivot.tui import console as tui_console

    # Emit schema version early for JSONL mode (even if no stages to run)
    if as_json:
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

    # Build DAG and get execution order for TUI display and worker pre-warming
    graph = cli_helpers.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order and not watch:
        # Emit execution result for JSONL mode
        if as_json:
            cli_helpers.emit_jsonl(
                ExecutionResultEvent(
                    type=RunEventType.EXECUTION_RESULT,
                    ran=0,
                    skipped=0,
                    failed=0,
                    total_duration_ms=0.0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )
        return {}

    # Pre-warm loky executor before starting Textual TUI
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts
    num_workers = len(execution_order) if execution_order else 1
    if tui:
        prepare_workers(num_workers)

    # Set up TUI queue and run_id if using TUI
    tui_queue: thread_queue.Queue[TuiMessage] | None = None
    run_id: str | None = None
    if tui:
        tui_queue = thread_queue.Queue()
        run_id = str(uuid.uuid4())[:8]

    # Set up console for plain text output
    console: tui_console.Console | None = None
    if not quiet and not as_json and not tui:
        console = tui_console.Console()

    # Set up JSONL callback (schema version already emitted above)
    jsonl_callback: Callable[[dict[str, object]], None] | None = None
    if as_json:
        jsonl_callback = cli_helpers.emit_jsonl

    # Create cancel event for TUI mode
    cancel_event = threading.Event() if tui else None

    if watch:
        return _run_watch_mode(
            stages_list=stages_list,
            execution_order=execution_order,
            graph=graph,
            quiet=quiet,
            tui=tui,
            as_json=as_json,
            debounce=debounce,
            tui_log=tui_log,
            on_error=on_error,
            serve=serve,
            force=force,
            tui_queue=tui_queue,
            run_id=run_id,
            console=console,
            jsonl_callback=jsonl_callback,
            cancel_event=cancel_event,
            no_commit=no_commit,
        )

    return _run_oneshot_mode(
        stages_list=stages_list,
        execution_order=execution_order,
        graph=graph,
        quiet=quiet,
        tui=tui,
        as_json=as_json,
        tui_log=tui_log,
        force=force,
        single_stage=single_stage,
        no_commit=no_commit,
        no_cache=no_cache,
        on_error=on_error,
        allow_uncached_incremental=allow_uncached_incremental,
        checkout_missing=checkout_missing,
        tui_queue=tui_queue,
        run_id=run_id,
        console=console,
        jsonl_callback=jsonl_callback,
        cancel_event=cancel_event,
    )


def _run_watch_mode(
    stages_list: list[str] | None,
    execution_order: list[str],
    graph: nx.DiGraph[str],
    *,
    quiet: bool,
    tui: bool,
    as_json: bool,
    debounce: int,
    tui_log: pathlib.Path | None,
    on_error: OnError,
    serve: bool,
    force: bool,
    tui_queue: TuiQueue | None,
    run_id: str | None,
    console: tui_console.Console | None,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    cancel_event: threading.Event | None,
    no_commit: bool,
) -> None:
    """Run watch mode with unified event-driven execution."""
    from pivot.engine import graph as engine_graph
    from pivot.tui import run as run_tui

    # Build bipartite graph for watch paths
    all_stages = cli_helpers.get_all_stages()
    bipartite_graph = engine_graph.build_graph(all_stages)
    watch_paths = engine_graph.get_watch_paths(bipartite_graph)

    # Sort for display: group matrix variants together while preserving DAG structure
    display_order = _sort_for_display(execution_order, graph) if execution_order else None

    with _create_engine() as eng:
        eng.set_keep_going(on_error == OnError.KEEP_GOING)

        if cancel_event:
            eng.set_cancel_event(cancel_event)

        # Configure sinks
        _configure_result_collector(eng)
        _configure_output_sink(
            eng,
            quiet=quiet,
            as_json=as_json,
            tui=tui,
            tui_queue=tui_queue,
            run_id=run_id,
            console=console,
            jsonl_callback=jsonl_callback,
            watch=True,
        )

        # Configure sources
        _configure_watch_sources(eng, watch_paths, debounce, force=force, stages=stages_list)

        if tui and tui_queue:
            with _suppress_stderr_logging():
                run_tui.run_watch_tui(
                    eng,
                    tui_queue,
                    tui_log=tui_log,
                    stage_names=display_order,
                    no_commit=no_commit,
                    serve=serve,
                )
        else:
            try:
                eng.run(exit_on_completion=False)
            except KeyboardInterrupt:
                pass  # Normal exit via Ctrl+C
            finally:
                eng.shutdown()

    return None


def _run_oneshot_mode(
    stages_list: list[str] | None,
    execution_order: list[str],
    graph: nx.DiGraph[str],
    *,
    quiet: bool,
    tui: bool,
    as_json: bool,
    tui_log: pathlib.Path | None,
    force: bool,
    single_stage: bool,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
    tui_queue: TuiQueue | None,
    run_id: str | None,
    console: tui_console.Console | None,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    cancel_event: threading.Event | None,
) -> dict[str, ExecutionSummary]:
    """Run non-watch (one-shot) mode with unified event-driven execution."""
    from pivot.executor import core as executor_core
    from pivot.tui import run as run_tui

    # TUI mode uses a thread to run executor
    if tui and tui_queue and run_id:
        # Sort for display
        display_order = _sort_for_display(execution_order, graph)

        def executor_func() -> dict[str, ExecutionSummary]:
            with _create_engine() as eng:
                if cancel_event:
                    eng.set_cancel_event(cancel_event)

                result_sink = _configure_result_collector(eng)
                _configure_output_sink(
                    eng,
                    quiet=quiet,
                    as_json=as_json,
                    tui=tui,
                    tui_queue=tui_queue,
                    run_id=run_id,
                    console=console,
                    jsonl_callback=jsonl_callback,
                    watch=False,
                )
                _configure_oneshot_source(
                    eng,
                    stages_list,
                    force=force,
                    single_stage=single_stage,
                    no_commit=no_commit,
                    no_cache=no_cache,
                    on_error=on_error,
                    allow_uncached_incremental=allow_uncached_incremental,
                    checkout_missing=checkout_missing,
                )
                eng.run(exit_on_completion=True)
                return result_sink.get_execution_summaries()

        with _suppress_stderr_logging():
            return run_tui.run_with_tui(
                display_order, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
            )

    # Non-TUI mode runs directly
    start_time = time.perf_counter()
    with _create_engine() as eng:
        result_sink = _configure_result_collector(eng)
        _configure_output_sink(
            eng,
            quiet=quiet,
            as_json=as_json,
            tui=tui,
            tui_queue=tui_queue,
            run_id=run_id,
            console=console,
            jsonl_callback=jsonl_callback,
            watch=False,
        )
        _configure_oneshot_source(
            eng,
            stages_list,
            force=force,
            single_stage=single_stage,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
        eng.run(exit_on_completion=True)
        results = result_sink.get_execution_summaries()

    # Emit JSONL final result
    if as_json:
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

    # Print summary for plain mode
    if console and results:
        ran, cached, blocked, failed = executor_core.count_results(results)
        total_duration = time.perf_counter() - start_time
        console.summary(ran, cached, blocked, failed, total_duration)

    return results


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
            # StreamHandler is generic but handlers list is Handler[]
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
    """Ensure a Pipeline is discovered and in context.

    If no Pipeline is in context, attempts discovery and stores the result.
    """
    if cli_decorators.get_pipeline_from_context() is not None:
        return
    try:
        pipeline = discovery.discover_pipeline()
        if pipeline is not None:
            cli_decorators.store_pipeline_in_context(pipeline)
            logger.info(f"Loaded pipeline: {pipeline.name}")
    except discovery.DiscoveryError as e:
        raise click.ClickException(str(e)) from e


def _create_engine() -> engine.Engine:
    """Create an Engine with the Pipeline from context.

    The Pipeline must have been discovered by the CLI decorator (or ensure_stages_registered).
    """
    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is None:
        raise click.ClickException(
            "No pipeline discovered. Create a pivot.yaml or pipeline.py file."
        )
    return engine.Engine(pipeline=pipeline)


def _validate_stages(stages_list: list[str] | None, single_stage: bool) -> None:
    """Validate stage arguments and options."""
    if single_stage and not stages_list:
        raise click.ClickException("--single-stage requires at least one stage name")
    cli_helpers.validate_stages_exist(stages_list)


def _output_explain(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool = False,
    allow_missing: bool = False,
) -> None:
    """Output detailed stage explanations using status logic."""
    from pivot import status as status_mod
    from pivot.cli import status as status_cli
    from pivot.engine import graph as engine_graph

    # Validate dependencies exist when allow_missing is False
    # (consistent with dry_run_cmd behavior)
    if not allow_missing:
        cli_helpers.build_dag(validate=True)

    # Build graph once for all explanations
    all_stages = cli_helpers.get_all_stages()
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage,
        all_stages,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )
    status_cli.output_explain_text(explanations)


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
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
    "--tui",
    "tui_flag",
    is_flag=True,
    help="Use interactive TUI display (default: plain text)",
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
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    single_stage: bool,
    dry_run: bool,
    explain: bool,
    force: bool,
    watch: bool,
    debounce: int | None,
    tui_flag: bool,
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
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    debounce = debounce if debounce is not None else config.get_watch_debounce()

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    # Validate --tui and --json are mutually exclusive
    if tui_flag and as_json:
        raise click.ClickException("--tui and --json are mutually exclusive")

    # Validate tui_log requires TUI mode
    if tui_log:
        if as_json:
            raise click.ClickException("--tui-log cannot be used with --json")
        if not tui_flag:
            raise click.ClickException("--tui-log requires --tui")
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

    # Check that a pipeline was discovered (either Pipeline object or registered stages)
    # Allow dry-run and JSON modes to proceed with empty pipeline (they'll report "No stages")
    # Explain mode requires a pipeline - it can't explain stages if there are none
    pipeline = cli_decorators.get_pipeline_from_context()
    # Check pipeline exists before checking stages (list_stages() would raise if called on None)
    has_stages = pipeline is not None and bool(pipeline.list_stages())
    if not has_stages and not dry_run and not as_json:
        raise click.ClickException("No pipeline found (pivot.yaml or pipeline.py)")

    # Handle explain mode (with or without dry-run) - show explanations without execution
    if explain:
        _output_explain(stages_list, single_stage, force, allow_missing=allow_missing)
        return

    # Handle dry-run mode (without explain) - terse output
    if dry_run:
        ctx.invoke(
            dry_run_cmd,
            stages=stages,
            single_stage=single_stage,
            force=force,
            as_json=as_json,
            allow_missing=allow_missing,
        )
        return

    on_error = OnError.KEEP_GOING if keep_going else OnError.FAIL

    # Validate --serve requires TUI mode
    if watch and serve and not tui_flag:
        raise click.ClickException("--serve requires --tui")

    show_human_output = not as_json and not quiet

    try:
        results = _run_pipeline(
            stages_list,
            watch=watch,
            single_stage=single_stage,
            force=force,
            quiet=quiet,
            tui=tui_flag,
            as_json=as_json,
            debounce=debounce,
            tui_log=tui_log,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            serve=serve,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    except KeyboardInterrupt:
        if show_human_output:
            click.echo("\nWatch mode stopped." if watch else "\nCancelled.")
        return

    if results is None:
        # Watch mode completed
        if show_human_output:
            click.echo("\nWatch mode stopped.")
        return

    if not results and show_human_output and not tui_flag:
        click.echo("No stages to run")


class DryRunJsonStageOutput(TypedDict):
    """JSON output for a single stage in dry-run mode."""

    would_run: bool
    reason: str


class DryRunJsonOutput(TypedDict):
    """JSON output for pivot run --dry-run --json."""

    stages: dict[str, DryRunJsonStageOutput]


@cli_decorators.pivot_command("dry-run")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Show what would run if forced",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--allow-missing", is_flag=True, help="Allow missing dep files if tracked")
def dry_run_cmd(
    stages: tuple[str, ...],
    single_stage: bool,
    force: bool,
    as_json: bool,
    allow_missing: bool,
) -> None:
    """Show what would run without executing."""
    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    # Validate dependencies exist when allow_missing is False
    # This validation was previously done inside get_pipeline_explanations via build_dag()
    if not allow_missing:
        cli_helpers.build_dag(validate=True)

    # Build bipartite graph for consistent execution order with Engine
    all_stages = cli_helpers.get_all_stages()
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
        all_stages=all_stages,
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
