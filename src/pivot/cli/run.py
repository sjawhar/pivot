from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import click

from pivot import discovery, exceptions, executor, registry
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.types import DisplayMode, StageExplanation, StageStatus

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary

logger = logging.getLogger(__name__)


def _ensure_stages_registered() -> None:
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

    if stages_list:
        graph = registry.REGISTRY.build_dag(validate=True)
        registered = set(graph.nodes())
        unknown = [s for s in stages_list if s not in registered]
        if unknown:
            raise exceptions.StageNotFoundError(f"Unknown stage(s): {', '.join(unknown)}")


def _get_all_explanations(
    stages_list: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
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
        )
        explanations.append(explanation)

    return explanations


def _run_with_tui(
    stages_list: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with TUI display."""
    import multiprocessing as mp

    from pivot import dag, project, run_tui
    from pivot.types import TuiMessage

    # Get execution order for stage names
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order:
        return {}

    resolved_cache_dir = cache_dir or project.get_project_root() / ".pivot" / "cache"

    # Create manager and queue (Manager().Queue for loky compatibility)
    manager = mp.Manager()
    tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

    # Create executor function that passes the TUI queue
    def executor_func() -> dict[str, ExecutionSummary]:
        return executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=resolved_cache_dir,
            show_output=False,
            tui_queue=tui_queue,
        )

    try:
        return run_tui.run_with_tui(execution_order, tui_queue, executor_func)
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
    "--watch",
    "-w",
    is_flag=False,
    flag_value="",
    default=None,
    metavar="GLOBS",
    help="Watch for file changes and re-run. Optionally specify comma-separated glob patterns.",
)
@click.option("--debounce", type=int, default=300, help="Debounce delay in milliseconds")
@click.option(
    "--display",
    type=click.Choice(["tui", "plain"]),
    default=None,
    help="Display mode: tui (interactive) or plain (streaming text). Auto-detects if not specified.",
)
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    dry_run: bool,
    explain: bool,
    watch: str | None,
    debounce: int,
    display: str | None,
) -> None:
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    # Handle dry-run modes (with or without explain)
    if dry_run:
        if explain:
            # --dry-run --explain: detailed explanation without execution
            ctx.invoke(explain_cmd, stages=stages, single_stage=single_stage, cache_dir=cache_dir)
        else:
            # --dry-run only: terse output
            ctx.invoke(dry_run_cmd, stages=stages, single_stage=single_stage, cache_dir=cache_dir)
        return

    if watch is not None:
        from pivot import watch as watch_module

        # Parse comma-separated globs if provided
        watch_globs = [g.strip() for g in watch.split(",") if g.strip()] if watch else None

        watch_module.run_watch_loop(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            watch_globs=watch_globs,
            debounce_ms=debounce,
        )
        return

    # Determine display mode
    display_mode = DisplayMode(display) if display else None

    # Normal execution (with optional explain mode)
    from pivot import run_tui

    use_tui = run_tui.should_use_tui(display_mode) and not explain
    if use_tui:
        results = _run_with_tui(stages_list, single_stage, cache_dir)
    else:
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=explain,
        )

    if not results:
        click.echo("No stages to run")
    elif not explain and not use_tui:
        _print_results(results)


@cli_decorators.pivot_command("dry-run")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
def dry_run_cmd(
    stages: tuple[str, ...], single_stage: bool, cache_dir: pathlib.Path | None
) -> None:
    """Show what would run without executing."""
    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, cache_dir)

    if not explanations:
        click.echo("No stages to run")
        return

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
def explain_cmd(
    stages: tuple[str, ...], single_stage: bool, cache_dir: pathlib.Path | None
) -> None:
    """Show detailed breakdown of why stages would run."""
    from pivot import console

    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, cache_dir)

    if not explanations:
        click.echo("No stages to run")
        return

    con = console.Console()
    for exp in explanations:
        con.explain_stage(exp)

    will_run = sum(1 for e in explanations if e["will_run"])
    con.explain_summary(will_run, len(explanations) - will_run)
