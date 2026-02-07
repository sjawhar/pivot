"""DAG-aware pipeline execution with full dependency resolution.

The `pivot repro` command runs stages with their dependencies, supporting
watch mode for continuous re-execution on file changes.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import json
import logging
import pathlib
import threading
import time
import uuid
from collections.abc import Callable  # noqa: TC003 - used in function signatures
from typing import TYPE_CHECKING

import anyio
import click

from pivot import config
from pivot.cli import _run_common, completion
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
)

if TYPE_CHECKING:
    import networkx as nx

    from pivot.executor import ExecutionSummary
    from pivot.tui import console as tui_console


_logger = logging.getLogger(__name__)

# JSONL schema version for forward compatibility
_JSONL_SCHEMA_VERSION = 1


def _configure_watch_sources(
    eng: engine.Engine,
    watch_paths: list[pathlib.Path],
    debounce: int,
    *,
    force: bool,
    stages: list[str] | None,
    no_commit: bool,
    on_error: OnError,
) -> None:
    """Configure sources for watch mode."""
    eng.add_source(engine_sources.FilesystemSource(watch_paths=watch_paths, debounce_ms=debounce))
    if force:
        eng.add_source(
            engine_sources.OneShotSource(
                stages=stages,
                force=True,
                reason="watch:initial:forced",
                no_commit=no_commit,
                on_error=on_error,
            )
        )


def _configure_oneshot_source(
    eng: engine.Engine,
    stages: list[str] | None,
    *,
    force: bool,
    no_commit: bool,
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
            single_stage=False,  # Always resolve deps for repro
            no_commit=no_commit,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    )


def _output_explain(
    stages_list: list[str] | None,
    force: bool = False,
    allow_missing: bool = False,
) -> None:
    """Output detailed stage explanations using status logic."""
    from pivot import project
    from pivot import status as status_mod
    from pivot.cli import status as status_cli
    from pivot.engine import graph as engine_graph
    from pivot.storage import track

    # Resolve cross-pipeline dependencies before getting stages
    cli_helpers.resolve_external_dependencies()

    all_stages = cli_helpers.get_all_stages()

    # Build graph with validation when allow_missing is False
    # When allow_missing=True, tracked files are used for validation
    tracked_files = track.discover_pvt_files(project.get_project_root()) if allow_missing else None
    graph = engine_graph.build_graph(
        all_stages, validate=not allow_missing, tracked_files=tracked_files
    )

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage=False,
        all_stages=all_stages,
        stage_registry=cli_helpers.get_registry(),
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )
    status_cli.output_explain_text(explanations)


def _dry_run(
    stages_list: list[str] | None,
    force: bool,
    as_json: bool,
    allow_missing: bool,
    quiet: bool,
) -> None:
    """Show what would run without executing."""
    from pivot import project
    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph
    from pivot.storage import track

    # Quiet mode suppresses output (except JSON which is always emitted)
    if quiet and not as_json:
        return

    # Resolve cross-pipeline dependencies before getting stages
    # (build_dag() does this automatically, but dry-run builds the graph directly)
    cli_helpers.resolve_external_dependencies()

    all_stages = cli_helpers.get_all_stages()

    # Build graph with validation when allow_missing is False
    # When allow_missing=True, tracked files are used for validation
    tracked_files = track.discover_pvt_files(project.get_project_root()) if allow_missing else None
    graph = engine_graph.build_graph(
        all_stages, validate=not allow_missing, tracked_files=tracked_files
    )

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage=False,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
        all_stages=all_stages,
        stage_registry=cli_helpers.get_registry(),
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


def _run_pipeline(
    stages_list: list[str] | None,
    *,
    watch: bool,
    force: bool,
    quiet: bool,
    tui: bool,
    as_json: bool,
    show_output: bool,
    debounce: int,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    on_error: OnError,
    serve: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with unified watch/non-watch execution.

    Returns execution results for non-watch mode, None for watch mode.
    """
    from pivot.engine import graph as engine_graph
    from pivot.tui import console as tui_console_mod

    # Emit schema version early for JSONL mode (even if no stages to run)
    if as_json:
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

    # Build DAG and get execution order for TUI display and worker pre-warming
    graph = cli_helpers.build_dag(validate=True)
    execution_order = engine_graph.get_execution_order(graph, stages_list, single_stage=False)

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
    num_workers = len(execution_order) if execution_order else 1
    if tui:
        prepare_workers(num_workers)

    # Set up run_id if using TUI
    run_id: str | None = None
    if tui:
        run_id = str(uuid.uuid4())[:8]

    # Set up console for plain text output
    console: tui_console_mod.Console | None = None
    if not quiet and not as_json and not tui:
        console = tui_console_mod.Console()

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
            show_output=show_output,
            debounce=debounce,
            tui_log=tui_log,
            on_error=on_error,
            serve=serve,
            force=force,
            run_id=run_id,
            console=console,
            jsonl_callback=jsonl_callback,
            no_commit=no_commit,
        )

    return _run_oneshot_mode(
        stages_list=stages_list,
        execution_order=execution_order,
        graph=graph,
        quiet=quiet,
        tui=tui,
        as_json=as_json,
        show_output=show_output,
        tui_log=tui_log,
        force=force,
        no_commit=no_commit,
        on_error=on_error,
        allow_uncached_incremental=allow_uncached_incremental,
        checkout_missing=checkout_missing,
        run_id=run_id,
        console=console,
        jsonl_callback=jsonl_callback,
        cancel_event=cancel_event,
    )


def _run_watch_mode(  # noqa: PLR0913 - many params needed for different modes
    stages_list: list[str] | None,
    execution_order: list[str],
    graph: nx.DiGraph[str],
    *,
    quiet: bool,
    tui: bool,
    as_json: bool,
    show_output: bool,
    debounce: int,
    tui_log: pathlib.Path | None,
    on_error: OnError,
    serve: bool,
    force: bool,
    run_id: str | None,
    console: tui_console.Console | None,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    no_commit: bool,
) -> None:
    """Run watch mode with unified event-driven execution."""

    from pivot.engine import graph as engine_graph

    # Use async serve mode for headless daemon
    if serve and not tui:
        return _run_serve_mode(
            stages_list,
            force=force,
            quiet=quiet,
            show_output=show_output,
            debounce=debounce,
            on_error=on_error,
            no_commit=no_commit,
        )

    # Build bipartite graph for watch paths
    all_stages = cli_helpers.get_all_stages()
    bipartite_graph = engine_graph.build_graph(all_stages)
    watch_paths = engine_graph.get_watch_paths(bipartite_graph)

    # Sort for display: group matrix variants together while preserving DAG structure
    display_order = _run_common.sort_for_display(execution_order, graph) if execution_order else []

    pipeline = cli_decorators.get_pipeline_from_context()

    if tui and run_id:
        # TUI mode with async Engine
        # IMPORTANT: Textual must run in the main thread for signal handlers (SIGTSTP, etc.)
        # to work correctly. We run the Engine in a background thread instead.
        from pivot.tui import run as tui_run

        # Create TUI app (will run in main thread)
        app = tui_run.PivotApp(
            stage_names=display_order,
            tui_log=tui_log,
            watch_mode=True,
            no_commit=no_commit,
            serve=serve,
            stage_data_provider=pipeline,
        )

        def engine_thread_target() -> None:
            """Run the async Engine in a background thread with its own event loop."""

            async def engine_main() -> None:
                async with engine.Engine(pipeline=pipeline) as eng:
                    # Configure sinks - TuiSink posts to app (thread-safe)
                    _run_common.configure_result_collector(eng)
                    _run_common.configure_output_sink(
                        eng,
                        quiet=quiet,
                        as_json=as_json,
                        tui=True,
                        app=app,
                        run_id=run_id,
                        use_console=False,
                        jsonl_callback=jsonl_callback,
                    )

                    # Add agent RPC source for TUI (needed for force re-run) or serve mode
                    if serve or tui:
                        from pivot import project
                        from pivot.engine.agent_rpc import (
                            AgentRpcHandler,
                            AgentRpcSource,
                            BroadcastEventSink,
                        )

                        state_dir = project.get_project_root() / ".pivot"
                        socket_path = state_dir / "agent.sock"
                        state_dir.mkdir(parents=True, exist_ok=True)

                        # Add event buffer as sink so it captures events
                        eng.add_sink(eng._event_buffer)  # pyright: ignore[reportPrivateUsage]

                        rpc_handler = AgentRpcHandler(
                            engine=eng,
                            event_buffer=eng._event_buffer,  # pyright: ignore[reportPrivateUsage]
                        )
                        eng.add_source(AgentRpcSource(socket_path=socket_path, handler=rpc_handler))
                        eng.add_sink(BroadcastEventSink())

                    # Configure watch sources
                    _configure_watch_sources(
                        eng,
                        watch_paths,
                        debounce,
                        force=force,
                        stages=stages_list,
                        no_commit=no_commit,
                        on_error=on_error,
                    )

                    await eng.run(exit_on_completion=False)

            try:
                anyio.run(engine_main)
            except Exception:
                # Engine failed - log and signal TUI to exit
                _logger.exception("Engine thread failed in watch mode")
                with contextlib.suppress(Exception):
                    app.post_message(tui_run.TuiShutdown())

        # Start engine in background thread
        engine_thread = threading.Thread(target=engine_thread_target, daemon=True)
        engine_thread.start()

        # Run TUI in main thread (required for signal handlers)
        with contextlib.suppress(KeyboardInterrupt), _run_common.suppress_stderr_logging():
            app.run()
    else:
        # Non-TUI async mode
        async def watch_main() -> None:
            async with engine.Engine(pipeline=pipeline) as eng:
                # Configure sinks
                _run_common.configure_result_collector(eng)
                _run_common.configure_output_sink(
                    eng,
                    quiet=quiet,
                    as_json=as_json,
                    tui=False,
                    app=None,
                    run_id=None,
                    use_console=console is not None,
                    jsonl_callback=jsonl_callback,
                    show_output=show_output,
                )

                # Configure sources
                _configure_watch_sources(
                    eng,
                    watch_paths,
                    debounce,
                    force=force,
                    stages=stages_list,
                    no_commit=no_commit,
                    on_error=on_error,
                )

                await eng.run(exit_on_completion=False)

        with contextlib.suppress(KeyboardInterrupt):
            anyio.run(watch_main)


def _run_serve_mode(
    stages_list: list[str] | None,
    *,
    force: bool,
    quiet: bool,
    show_output: bool,
    debounce: int,
    on_error: OnError,
    no_commit: bool,
) -> None:
    """Run serve mode with Engine and agent RPC.

    This is the headless daemon mode that accepts agent connections
    via Unix socket while watching for file changes.
    """
    import rich.console

    from pivot import project
    from pivot.engine import graph as engine_graph
    from pivot.engine.agent_rpc import AgentRpcHandler, AgentRpcSource, BroadcastEventSink

    # Get project paths
    project_root = project.get_project_root()
    state_dir = project_root / ".pivot"
    socket_path = state_dir / "agent.sock"

    # Ensure state directory exists
    state_dir.mkdir(parents=True, exist_ok=True)

    # Get pipeline for Engine
    pipeline = cli_decorators.get_pipeline_from_context()

    async def serve_main() -> None:
        # Build watch paths
        all_stages = cli_helpers.get_all_stages()
        bipartite_graph = engine_graph.build_graph(all_stages)
        watch_paths = engine_graph.get_watch_paths(bipartite_graph)

        async with engine.Engine(pipeline=pipeline) as eng:
            # Add filesystem watch source
            eng.add_source(
                engine_sources.FilesystemSource(
                    watch_paths=watch_paths,
                    debounce_ms=debounce,
                )
            )

            # Add initial run source if force specified
            if force:
                eng.add_source(
                    engine_sources.OneShotSource(
                        stages=stages_list,
                        force=True,
                        reason="serve:initial",
                        no_commit=no_commit,
                        on_error=on_error,
                    )
                )

            # Add event buffer as sink so it captures events
            eng.add_sink(eng._event_buffer)  # pyright: ignore[reportPrivateUsage]

            # Add agent RPC source with handler for status/stages queries
            rpc_handler = AgentRpcHandler(engine=eng, event_buffer=eng._event_buffer)  # pyright: ignore[reportPrivateUsage]
            eng.add_source(AgentRpcSource(socket_path=socket_path, handler=rpc_handler))

            # Add sinks
            if not quiet:
                serve_console = rich.console.Console()
                eng.add_sink(sinks.ConsoleSink(console=serve_console, show_output=show_output))
            eng.add_sink(sinks.ResultCollectorSink())
            eng.add_sink(BroadcastEventSink())  # Broadcast events to connected agents

            # Run until interrupted (watch mode never exits on its own)
            await eng.run(exit_on_completion=False)

    # Run the async event loop
    with contextlib.suppress(KeyboardInterrupt):
        anyio.run(serve_main)


def _run_oneshot_mode(
    stages_list: list[str] | None,
    execution_order: list[str],
    graph: nx.DiGraph[str],
    *,
    quiet: bool,
    tui: bool,
    as_json: bool,
    show_output: bool,
    tui_log: pathlib.Path | None,
    force: bool,
    no_commit: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
    run_id: str | None,
    console: tui_console.Console | None,
    jsonl_callback: Callable[[dict[str, object]], None] | None,
    cancel_event: threading.Event | None,
) -> dict[str, ExecutionSummary]:
    """Run non-watch (one-shot) mode with unified event-driven execution."""
    from pivot.executor import core as executor_core

    # Sort for display
    display_order = _run_common.sort_for_display(execution_order, graph) if execution_order else []

    pipeline = cli_decorators.get_pipeline_from_context()

    # TUI mode for oneshot
    # IMPORTANT: Textual must run in the main thread for signal handlers (SIGTSTP, etc.)
    # to work correctly. We run the Engine in a background thread instead.
    if tui and run_id:
        from pivot.tui import run as tui_run

        # Create TUI app (will run in main thread)
        app = tui_run.PivotApp(
            stage_names=display_order,
            tui_log=tui_log,
            cancel_event=cancel_event,
            stage_data_provider=pipeline,
        )

        # Use Future for thread-safe result passing
        result_future: concurrent.futures.Future[dict[str, executor_core.ExecutionSummary]] = (
            concurrent.futures.Future()
        )

        def engine_thread_target() -> None:
            """Run the async Engine in a background thread with its own event loop."""

            async def engine_main() -> dict[str, executor_core.ExecutionSummary]:
                async with engine.Engine(pipeline=pipeline) as eng:
                    # Configure sinks - TuiSink posts to app (thread-safe)
                    result_sink = _run_common.configure_result_collector(eng)
                    _run_common.configure_output_sink(
                        eng,
                        quiet=quiet,
                        as_json=as_json,
                        tui=True,
                        app=app,
                        run_id=run_id,
                        use_console=False,
                        jsonl_callback=jsonl_callback,
                    )
                    _configure_oneshot_source(
                        eng,
                        stages_list,
                        force=force,
                        no_commit=no_commit,
                        on_error=on_error,
                        allow_uncached_incremental=allow_uncached_incremental,
                        checkout_missing=checkout_missing,
                    )
                    await eng.run(exit_on_completion=True)
                    stage_results = await result_sink.get_results()
                    return _run_common.convert_results(stage_results)

            try:
                result_future.set_result(anyio.run(engine_main))
            except BaseException as e:
                result_future.set_exception(e)
            finally:
                # Signal TUI to exit when engine completes (success or failure)
                with contextlib.suppress(Exception):
                    app.post_message(tui_run.TuiShutdown())

        # Start engine in background thread
        engine_thread = threading.Thread(target=engine_thread_target, daemon=True)
        engine_thread.start()

        # Run TUI in main thread (required for signal handlers)
        with _run_common.suppress_stderr_logging():
            app.run()

        # Wait for engine thread to finish (should be quick since TUI already exited)
        engine_thread.join(timeout=5.0)

        # Return results (re-raises if engine raised an exception)
        if result_future.done():
            return result_future.result()
        return {}

    # Non-TUI async mode
    start_time = time.perf_counter()

    async def oneshot_main() -> dict[str, executor_core.ExecutionSummary]:
        async with engine.Engine(pipeline=pipeline) as eng:
            result_sink = _run_common.configure_result_collector(eng)
            _run_common.configure_output_sink(
                eng,
                quiet=quiet,
                as_json=as_json,
                tui=False,
                app=None,
                run_id=None,
                use_console=console is not None,
                jsonl_callback=jsonl_callback,
                show_output=show_output,
            )
            _configure_oneshot_source(
                eng,
                stages_list,
                force=force,
                no_commit=no_commit,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
            await eng.run(exit_on_completion=True)
            stage_results = await result_sink.get_results()
            return _run_common.convert_results(stage_results)

    results = anyio.run(oneshot_main)

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
@click.option(
    "--jsonl",
    "--json",
    "as_json",
    is_flag=True,
    help="Stream results as JSONL (one JSON object per line).",
)
@click.option(
    "--show-output",
    is_flag=True,
    help="Stream stage output (stdout/stderr) to terminal",
)
@click.option(
    "--tui-log",
    type=click.Path(path_type=pathlib.Path),
    help="Write TUI messages to JSONL file for monitoring",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Skip writing lock files. Run 'pivot commit' to finalize.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure (default).",
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
    show_output: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    fail_fast: bool,
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

    # Validate mutual exclusions
    _run_common.validate_display_mode(tui_flag, as_json)
    _run_common.validate_show_output(show_output, tui_flag, as_json, quiet)
    tui_log = _run_common.validate_tui_log(tui_log, as_json, tui_flag, dry_run=dry_run)

    # Validate flag prerequisites
    if serve and not watch:
        raise click.ClickException("--serve requires --watch mode")
    if debounce_from_cli and not watch:
        raise click.ClickException("--debounce requires --watch mode")
    if allow_missing and not dry_run and not explain:
        raise click.ClickException("--allow-missing can only be used with --dry-run or --explain")

    # Check that a pipeline was discovered
    pipeline = cli_decorators.get_pipeline_from_context()
    has_stages = pipeline is not None and bool(pipeline.list_stages())
    if not has_stages and not dry_run and not as_json:
        raise click.ClickException("No pipeline found (pivot.yaml or pipeline.py)")

    # Handle explain mode
    if explain:
        _output_explain(stages_list, force, allow_missing=allow_missing)
        return

    on_error = _run_common.resolve_on_error(fail_fast, keep_going)

    # Handle dry-run mode
    if dry_run:
        _dry_run(stages_list, force, as_json, allow_missing, quiet)
        return

    try:
        results = _run_pipeline(
            stages_list,
            watch=watch,
            force=force,
            quiet=quiet,
            tui=tui_flag,
            as_json=as_json,
            show_output=show_output,
            debounce=debounce_ms,
            tui_log=tui_log,
            no_commit=no_commit,
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
