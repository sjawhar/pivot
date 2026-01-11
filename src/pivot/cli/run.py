from __future__ import annotations

import json
import logging
import multiprocessing as mp
import pathlib
from typing import TYPE_CHECKING, Literal, TypedDict, override

import click
import click.shell_completion

from pivot import (
    dag,
    discovery,
    executor,
    explain,
    parameters,
    project,
    reactive,
    registry,
)
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.tui import console
from pivot.tui import run as run_tui
from pivot.types import DisplayMode, StageExplanation, StageStatus, TuiMessage

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary


class RunJsonStageOutput(TypedDict):
    """JSON output for a single stage result."""

    status: Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED, StageStatus.UNKNOWN]
    reason: str


class RunJsonOutput(TypedDict):
    """JSON output for pivot run --json."""

    stages: dict[str, RunJsonStageOutput]


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


class DisplayModeType(click.ParamType):
    """Click parameter type that converts string to DisplayMode enum."""

    name: str = "display_mode"

    @override
    def convert(
        self, value: str | None, param: click.Parameter | None, ctx: click.Context | None
    ) -> DisplayMode | None:
        if value is None:
            return None
        try:
            return DisplayMode(value)
        except ValueError:
            self.fail(f"Invalid display mode: {value}. Choose from: tui, plain", param, ctx)

    @override
    def shell_complete(
        self, ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[click.shell_completion.CompletionItem]:
        return [
            click.shell_completion.CompletionItem(mode.value)
            for mode in DisplayMode
            if mode.value.startswith(incomplete)
        ]


DISPLAY_MODE = DisplayModeType()


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
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with TUI display."""
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
            force=force,
        )

    try:
        return run_tui.run_with_tui(execution_order, tui_queue, executor_func)
    finally:
        manager.shutdown()


def _run_watch_with_tui(
    stages_list: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    debounce: int,
    force: bool = False,
) -> None:
    """Run watch mode with TUI display."""
    manager = mp.Manager()
    tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

    engine = reactive.ReactiveEngine(
        stages=stages_list,
        single_stage=single_stage,
        cache_dir=cache_dir,
        debounce_ms=debounce,
        force_first_run=force,
    )

    try:
        run_tui.run_watch_tui(engine, tui_queue)
    finally:
        manager.shutdown()


def _results_to_json(results: dict[str, ExecutionSummary]) -> RunJsonOutput:
    """Convert execution results to JSON-serializable format."""
    return RunJsonOutput(
        stages={
            name: RunJsonStageOutput(status=result["status"], reason=result["reason"])
            for name, result in results.items()
        }
    )


def _print_results(results: dict[str, ExecutionSummary], as_json: bool = False) -> None:
    """Print execution results in a readable format."""
    if as_json:
        click.echo(json.dumps(_results_to_json(results), indent=2))
        return

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
    type=DISPLAY_MODE,
    default=None,
    help="Display mode: tui (interactive) or plain (streaming text). Auto-detects if not specified.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
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
    display: DisplayMode | None,
    as_json: bool,
) -> None:
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

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
            )
        return

    if watch:
        display_mode = display
        use_tui = run_tui.should_use_tui(display_mode) and not as_json

        if use_tui:
            try:
                _run_watch_with_tui(stages_list, single_stage, cache_dir, debounce, force)
            except KeyboardInterrupt:
                click.echo("\nWatch mode stopped.")
        else:
            engine = reactive.ReactiveEngine(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                debounce_ms=debounce,
                force_first_run=force,
                json_output=as_json,
            )

            try:
                engine.run(tui_queue=None)
            except KeyboardInterrupt:
                pass
            finally:
                engine.shutdown()
                if not as_json:
                    click.echo("\nWatch mode stopped.")
        return

    # Determine display mode
    display_mode = display

    use_tui = run_tui.should_use_tui(display_mode) and not explain and not as_json
    if use_tui:
        results = _run_with_tui(stages_list, single_stage, cache_dir, force=force)
    else:
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=explain,
            force=force,
        )

    if not results:
        if as_json:
            click.echo(json.dumps(RunJsonOutput(stages={})))
        else:
            click.echo("No stages to run")
    elif not explain and not use_tui:
        _print_results(results, as_json=as_json)


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
def dry_run_cmd(
    stages: tuple[str, ...], single_stage: bool, cache_dir: pathlib.Path | None, force: bool
) -> None:
    """Show what would run without executing."""
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    explanations = _get_all_explanations(stages_list, single_stage, cache_dir, force=force)

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
    stages_list = list(stages) if stages else None
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
