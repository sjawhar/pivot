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

from pivot import discovery, executor, metrics, registry
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
    cache_dir: pathlib.Path | None,
    force: bool = False,
) -> list[StageExplanation]:
    """Get explanations for all stages in execution order."""
    from pivot import dag, explain, parameters, project

    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order:
        return []

    resolved_cache_dir = cache_dir or project.get_project_root() / ".pivot" / "cache"
    overrides = parameters.load_params_yaml()

    explanations = list[StageExplanation]()
    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)
        explanation = explain.get_stage_explanation(
            stage_name,
            stage_info["fingerprint"],
            stage_info["deps"],
            stage_info["params"],
            overrides,
            resolved_cache_dir,
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
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with TUI display."""
    import multiprocessing as mp

    import loky

    from pivot import dag, project
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Get execution order for stage names
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order:
        return {}

    resolved_cache_dir = cache_dir or project.get_project_root() / ".pivot" / "cache"

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    loky.get_reusable_executor(max_workers=1)

    # Create manager and queue (Manager().Queue for loky compatibility)
    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        # Create executor function that passes the TUI queue
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
            )

        with _suppress_stderr_logging():
            return run_tui.run_with_tui(execution_order, tui_queue, executor_func, tui_log=tui_log)
    finally:
        manager.shutdown()


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
    import os

    import loky

    from pivot import dag
    from pivot import watch as watch_module
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Get execution order to calculate the correct number of workers
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    # Calculate max_workers the same way executor does
    max_workers = min(os.cpu_count() or 1, 8, len(execution_order)) if execution_order else 1

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    # IMPORTANT: Must use the same max_workers that executor will use, otherwise
    # loky will try to spawn additional workers after Textual has started.
    # We also submit a no-op task to each worker to ensure all communication
    # channels are fully established before Textual starts.
    pool = loky.get_reusable_executor(max_workers=max_workers)
    # Submit no-op tasks to ensure workers are warm and communication channels exist
    futures = [pool.submit(lambda: None) for _ in range(max_workers)]
    for f in futures:
        f.result()  # Wait for completion to ensure channels are established

    # Create Manager and queues BEFORE Textual starts to avoid multiprocessing
    # fd inheritance issues. Textual manipulates terminal file descriptors which
    # breaks Manager() subprocess spawning if done after Textual starts.
    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]
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

        with _suppress_stderr_logging():
            run_tui.run_watch_tui(
                engine,
                tui_queue,
                output_queue=output_queue,
                tui_log=tui_log,
                stage_names=execution_order,
                no_commit=no_commit,
                serve=serve,
            )
    finally:
        manager.shutdown()


def _print_results(results: dict[str, ExecutionSummary]) -> None:
    """Print execution results in a readable format."""
    ran = 0
    skipped = 0
    failed = 0

    for name, result in results.items():
        result_status = result["status"]
        reason = result["reason"]

        if result_status == StageStatus.RAN:
            ran += 1
            click.echo(f"{name}: ran ({reason})")
        elif result_status == StageStatus.FAILED:
            failed += 1
            click.echo(f"{name}: failed ({reason})")
        else:
            skipped += 1
            if reason:
                click.echo(f"{name}: skipped ({reason})")
            else:
                click.echo(f"{name}: skipped")

    parts = [f"{ran} ran", f"{skipped} skipped"]
    if failed > 0:
        parts.append(f"{failed} failed")
    click.echo(f"\nTotal: {', '.join(parts)}")

    # Print metrics summary if available (when PIVOT_METRICS=1 or metrics.enable() was called)
    metrics_summary = metrics.summary()
    if metrics_summary:
        click.echo("\nMetrics:")
        for name, stats in metrics_summary.items():
            click.echo(
                f"  {name}: {stats['count']}x, total={stats['total_ms']:.1f}ms, avg={stats['avg_ms']:.1f}ms"
            )


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
    default=300,
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
    debounce: int,
    display: str | None,  # Click passes string, converted to DisplayMode below
    as_json: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    keep_going: bool,
    serve: bool,
) -> None:
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
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
                cache_dir=cache_dir,
                force=force,
            )
        else:
            # --dry-run only: terse output
            ctx.invoke(
                dry_run_cmd,
                stages=stages,
                single_stage=single_stage,
                cache_dir=cache_dir,
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
                if not as_json:
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
        )

    if not results and not as_json:
        click.echo("No stages to run")
    elif not explain and not use_tui and not as_json and results:
        _print_results(results)


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
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
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
    cache_dir: pathlib.Path | None,
    force: bool,
    as_json: bool,
) -> None:
    """Show what would run without executing."""
    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, cache_dir, force=force)

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
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Show explanation as if forced",
)
def explain_cmd(
    stages: tuple[str, ...], single_stage: bool, cache_dir: pathlib.Path | None, force: bool
) -> None:
    """Show detailed breakdown of why stages would run."""
    from pivot.tui import console

    stages_list = cli_helpers.stages_to_list(stages)
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, cache_dir, force=force)

    if not explanations:
        click.echo("No stages to run")
        return

    con = console.Console()
    for exp in explanations:
        con.explain_stage(exp)

    will_run = sum(1 for e in explanations if e["will_run"])
    con.explain_summary(will_run, len(explanations) - will_run)
