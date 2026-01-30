"""DAG-aware pipeline execution with full dependency resolution.

The `pivot repro` command runs stages with their dependencies, supporting
watch mode for continuous re-execution on file changes.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import click

from pivot import config, registry
from pivot.cli import _run_common, completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.executor import prepare_workers
from pivot.types import OnError

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary


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
        single_stage=False,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )
    status_cli.output_explain_text(explanations)


def _run_with_tui(
    stages_list: list[str] | None,
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
    import uuid

    from pivot import dag
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Get execution order for stage names
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=False)

    if not execution_order:
        return {}

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    prepare_workers(len(execution_order))

    # tui_queue is inter-thread only (executor -> TUI reader), no cross-process IPC needed.
    # Using stdlib queue.Queue avoids Manager subprocess dependency issues.
    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()

    # Cancel event allows TUI to signal executor to stop scheduling new stages
    cancel_event = threading.Event()

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
                single_stage=False,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

    # Sort for display: group matrix variants together while preserving DAG structure
    display_order = _run_common.sort_for_display(execution_order, graph)

    with _run_common.suppress_stderr_logging():
        return run_tui.run_with_tui(
            display_order, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


def _run_watch_with_tui(
    stages_list: list[str] | None,
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    on_error: OnError = OnError.FAIL,
    serve: bool = False,
    debounce: int = 0,
) -> None:
    """Run watch mode with TUI display."""
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
        eng.add_source(sources.FilesystemSource(watch_paths, debounce=debounce))

        # Add OneShotSource for initial run if force is set
        if force:
            eng.add_source(
                sources.OneShotSource(stages=stages_list, force=True, reason="watch:initial:forced")
            )

        # Add sinks for TUI updates
        eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
        eng.add_sink(sinks.WatchSink(tui_queue=tui_queue))

        # Sort for display: group matrix variants together while preserving DAG structure
        display_order = (
            _run_common.sort_for_display(execution_order, graph) if execution_order else None
        )

        with _run_common.suppress_stderr_logging():
            run_tui.run_watch_tui(
                eng,
                tui_queue,
                tui_log=tui_log,
                stage_names=display_order,
                no_commit=False,  # TODO: Support no_commit in Engine watch mode
                serve=serve,
            )


def _run_watch_plain(
    stages_list: list[str] | None,
    force: bool,
    on_error: OnError,
    as_json: bool,
    quiet: bool,
    debounce: int = 0,
) -> None:
    """Run watch mode with plain (non-TUI) display."""
    from pivot.engine import graph as engine_graph
    from pivot.engine import sources
    from pivot.tui import console as tui_console

    with engine.Engine() as eng:
        eng.set_keep_going(on_error == OnError.KEEP_GOING)

        # Build bipartite graph for watch paths
        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        bipartite_graph = engine_graph.build_graph(all_stages)
        watch_paths = engine_graph.get_watch_paths(bipartite_graph)

        eng.add_source(sources.FilesystemSource(watch_paths, debounce=debounce))

        if force:
            eng.add_source(
                sources.OneShotSource(stages=stages_list, force=True, reason="watch:initial:forced")
            )

        if as_json:
            _run_common.emit_json_schema_version()
            eng.add_sink(sinks.JsonlSink(callback=cli_helpers.emit_jsonl))
        elif not quiet:
            eng.add_sink(sinks.ConsoleSink(tui_console.Console()))

        eng.run_loop()


def _dry_run(
    stages_list: list[str] | None,
    force: bool,
    as_json: bool,
    allow_missing: bool,
) -> None:
    """Show what would run without executing."""
    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph

    if not allow_missing:
        registry.REGISTRY.build_dag(validate=True)

    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage=False,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )

    if not explanations:
        if as_json:
            click.echo(json.dumps(_run_common.DryRunJsonOutput(stages={})))
        else:
            click.echo("No stages to run")
        return

    if as_json:
        output = _run_common.DryRunJsonOutput(
            stages={
                exp["stage_name"]: _run_common.DryRunJsonStageOutput(
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
    help="Debounce delay in milliseconds (requires --watch)",
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
def repro(
    ctx: click.Context,
    stages: tuple[str, ...],
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
    """Reproduce pipeline stages with full dependency resolution.

    If STAGES are provided, runs those stages and all their dependencies.
    Without arguments, runs the entire pipeline.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet

    # Validate --debounce was explicitly provided (for error message)
    debounce_from_cli = debounce is not None
    # Use provided debounce value or fall back to config default
    debounce_ms = debounce if debounce is not None else config.get_watch_debounce()

    stages_list = cli_helpers.stages_to_list(stages)
    cli_helpers.validate_stages_exist(stages_list)

    # Validate --tui and --json are mutually exclusive
    if tui_flag and as_json:
        raise click.ClickException("--tui and --json are mutually exclusive")

    # Validate tui_log
    tui_log = _run_common.validate_tui_log(tui_log, as_json, tui_flag, dry_run=dry_run)

    # Validate --serve requires --watch
    if serve and not watch:
        raise click.ClickException("--serve requires --watch mode")

    # Validate --debounce requires --watch
    if debounce_from_cli and not watch:
        raise click.ClickException("--debounce requires --watch mode")

    # Validate --allow-missing requires --dry-run or --explain
    if allow_missing and not dry_run and not explain:
        raise click.ClickException("--allow-missing can only be used with --dry-run or --explain")

    # Handle explain mode
    if explain:
        _output_explain(stages_list, force, allow_missing=allow_missing)
        return

    # Handle dry-run mode
    if dry_run:
        _dry_run(stages_list, force, as_json, allow_missing)
        return

    on_error = OnError.KEEP_GOING if keep_going else OnError.FAIL

    if watch:
        use_tui = tui_flag

        # Validate --serve requires TUI mode
        if serve and not use_tui:
            raise click.ClickException("--serve requires --tui")

        try:
            if use_tui:
                _run_watch_with_tui(
                    stages_list,
                    force=force,
                    tui_log=tui_log,
                    on_error=on_error,
                    serve=serve,
                    debounce=debounce_ms,
                )
            else:
                _run_watch_plain(stages_list, force, on_error, as_json, quiet, debounce_ms)
        except KeyboardInterrupt:
            pass  # Normal exit via Ctrl+C
        finally:
            if show_human_output:
                click.echo("\nWatch mode stopped.")
        return

    # Normal (non-watch) execution
    use_tui = tui_flag

    if use_tui:
        results = _run_with_tui(
            stages_list,
            force=force,
            tui_log=tui_log,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
        # TUI returns None if user quit early
        if results is None:
            return
    elif as_json:
        results = _run_common.run_json_mode(
            stages_list,
            single_stage=False,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    else:
        results = _run_common.run_plain_mode(
            stages_list,
            single_stage=False,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
            quiet=quiet,
        )

    if not results and show_human_output:
        click.echo("No stages to run")
