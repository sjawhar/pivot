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

from pivot import config, discovery, executor, registry
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.types import (
    DisplayMode,
    ExecutionResultEvent,
    OnError,
    OutputMessage,
    RunEventType,
    SchemaVersionEvent,
    StageExplanation,
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
    """Auto-discover and register stages if none are registered."""
    if not discovery.has_registered_stages():
        try:
            discovered = discovery.discover_and_register()
            if discovered:
                logger.info(f"Loaded pipeline from {discovered}")
        except discovery.DiscoveryError as e:
            raise click.ClickException(str(e)) from e


def _validate_stages(stages_list: list[str] | None, single_stage: bool) -> None:
    """Validate stage arguments and options."""
    if single_stage and not stages_list:
        raise click.ClickException("--single-stage requires at least one stage name")
    cli_helpers.validate_stages_exist(stages_list)


def _get_all_explanations(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool = False,
) -> list[StageExplanation]:
    """Get explanations for all stages in execution order."""
    from pivot import dag, explain, parameters

    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order:
        return []

    state_dir = config.get_state_dir()
    overrides = parameters.load_params_yaml()

    explanations = list[StageExplanation]()
    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)
        explanation = explain.get_stage_explanation(
            stage_name,
            stage_info["fingerprint"],
            stage_info["deps_paths"],
            stage_info["outs_paths"],
            stage_info["params"],
            overrides,
            state_dir,
            force=force,
        )
        explanations.append(explanation)

    return explanations


def _run_with_tui(
    stages_list: list[str] | None,
    single_stage: bool,
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
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order:
        return {}

    resolved_cache_dir = cache_dir or config.get_cache_dir()

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    executor.prepare_workers(len(execution_order))

    # tui_queue is inter-thread only (executor -> TUI reader), no cross-process IPC needed.
    # Using stdlib queue.Queue avoids Manager subprocess dependency issues.
    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()

    # Cancel event allows TUI to signal executor to stop scheduling new stages
    cancel_event = threading.Event()

    # Create executor function that passes the TUI queue and cancel event
    def executor_func() -> dict[str, ExecutionSummary]:
        return executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=resolved_cache_dir,
            show_output=False,
            tui_queue=tui_queue,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
            cancel_event=cancel_event,
        )

    # Sort for display: group matrix variants together while preserving DAG structure
    display_order = _sort_for_display(execution_order, graph)

    with _suppress_stderr_logging():
        return run_tui.run_with_tui(
            display_order, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


def _run_watch_with_tui(
    stages_list: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    debounce: int,
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    on_error: OnError = OnError.FAIL,
    serve: bool = False,
) -> None:
    """Run watch mode with TUI display."""
    import multiprocessing as mp
    import queue as thread_queue

    from pivot import dag
    from pivot import watch as watch_module
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Get execution order to calculate the correct number of workers
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    executor.prepare_workers(len(execution_order) if execution_order else 1)

    # tui_queue is inter-thread only (executor -> TUI reader), no cross-process IPC needed.
    # Using stdlib queue.Queue avoids Manager subprocess dependency issues.
    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()

    # output_queue crosses process boundaries (loky workers -> main), requires Manager.Queue.
    # Create Manager BEFORE Textual starts to avoid multiprocessing fd inheritance issues.
    # Use spawn context to avoid fork-in-multithreaded-context issues (Python 3.13+ deprecation)
    spawn_ctx = mp.get_context("spawn")
    manager = spawn_ctx.Manager()
    try:
        output_queue: mp.Queue[OutputMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        engine = watch_module.WatchEngine(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            debounce_ms=debounce,
            force_first_run=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
        )

        # Sort for display: group matrix variants together while preserving DAG structure
        display_order = _sort_for_display(execution_order, graph) if execution_order else None

        with _suppress_stderr_logging():
            run_tui.run_watch_tui(
                engine,
                tui_queue,
                output_queue=output_queue,
                tui_log=tui_log,
                stage_names=display_order,
                no_commit=no_commit,
                serve=serve,
            )
    finally:
        manager.shutdown()


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
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
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    single_stage: bool,
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
) -> None:
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet
    debounce = debounce if debounce is not None else config.get_watch_debounce()

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

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

    # Handle dry-run modes (with or without explain)
    if dry_run:
        if explain:
            # --dry-run --explain: detailed explanation without execution
            ctx.invoke(
                explain_cmd,
                stages=stages,
                single_stage=single_stage,
                force=force,
            )
        else:
            # --dry-run only: terse output
            ctx.invoke(
                dry_run_cmd,
                stages=stages,
                single_stage=single_stage,
                force=force,
                as_json=as_json,
            )
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
                    single_stage,
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
            from pivot import watch as watch_module

            engine = watch_module.WatchEngine(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                debounce_ms=debounce,
                force_first_run=force,
                json_output=as_json,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
            )

            try:
                engine.run(tui_queue=None)
            except KeyboardInterrupt:
                pass  # Normal exit via Ctrl+C
            finally:
                engine.shutdown()
                if show_human_output:
                    click.echo("\nWatch mode stopped.")
        return

    # Determine display mode
    display_mode = DisplayMode(display) if display else None

    # Normal execution (with optional explain mode)
    from pivot.tui import run as run_tui

    # Disable TUI when JSON output is requested
    use_tui = run_tui.should_use_tui(display_mode) and not explain and not as_json
    if use_tui:
        results = _run_with_tui(
            stages_list,
            single_stage,
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
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=False,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            show_output=False,
            progress_callback=cli_helpers.emit_jsonl,
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
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=explain,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            show_output=not quiet,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )

    if not results and show_human_output:
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
def dry_run_cmd(
    stages: tuple[str, ...],
    single_stage: bool,
    force: bool,
    as_json: bool,
) -> None:
    """Show what would run without executing."""
    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, force=force)

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


@cli_decorators.pivot_command("explain")
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
    help="Show explanation as if forced",
)
def explain_cmd(stages: tuple[str, ...], single_stage: bool, force: bool) -> None:
    """Show detailed breakdown of why stages would run."""
    from pivot.tui import console

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, force=force)

    if not explanations:
        click.echo("No stages to run")
        return

    con = console.Console()
    for exp in explanations:
        con.explain_stage(exp)

    will_run = sum(1 for e in explanations if e["will_run"])
    con.explain_summary(will_run, len(explanations) - will_run)
