"""Command-line interface for Pivot.

Provides commands to run and manage pipelines from the command line.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import click

from pivot import executor, registry

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary

logger = logging.getLogger(__name__)


def _validate_stages(stages_list: list[str] | None, single_stage: bool) -> None:
    """Validate stage arguments and options."""
    if single_stage and not stages_list:
        raise click.ClickException("--single-stage requires at least one stage name")

    if stages_list:
        graph = registry.REGISTRY.build_dag(validate=True)
        registered = set(graph.nodes())
        unknown = [s for s in stages_list if s not in registered]
        if unknown:
            raise click.ClickException(f"Unknown stage(s): {', '.join(unknown)}")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Fast pipeline execution with per-stage caching."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.argument("stages", nargs=-1)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would run without executing")
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
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    dry_run: bool,
    watch: str | None,
    debounce: int,
) -> None:
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.
    """
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    if dry_run:
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

    try:
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
        )
        _print_results(results)
    except Exception as e:
        raise click.ClickException(str(e)) from e


@cli.command("dry-run")
@click.argument("stages", nargs=-1)
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
    from pivot import dag, parameters, project

    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    try:
        graph = registry.REGISTRY.build_dag(validate=True)
        execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

        if not execution_order:
            click.echo("No stages to run")
            return

        cache_dir = cache_dir or project.get_project_root() / ".pivot" / "cache"
        overrides = parameters.load_params_yaml()

        click.echo("Would run:")
        for stage_name in execution_order:
            stage_info = registry.REGISTRY.get(stage_name)

            result = executor.check_stage_changed(
                stage_name,
                stage_info["fingerprint"],
                stage_info["deps"],
                stage_info["params"],
                overrides,
                cache_dir,
            )

            if result["missing_deps"] or result["changed"]:
                click.echo(f"  {stage_name}: would run ({result['reason']})")
            else:
                click.echo(f"  {stage_name}: would skip (unchanged)")

    except Exception as e:
        raise click.ClickException(str(e)) from e


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pipeline status."""
    verbose = ctx.obj.get("verbose", False)
    stage_list = registry.REGISTRY.list_stages()

    if not stage_list:
        click.echo("No stages registered")
        return

    click.echo(f"Registered stages ({len(stage_list)}):")
    for name in stage_list:
        info = registry.REGISTRY.get(name)
        deps = info["deps"]
        outs = info["outs_paths"]
        click.echo(f"  {name}")
        if verbose:
            click.echo(f"    deps: {deps}")
            click.echo(f"    outs: {outs}")


@cli.command()
@click.argument("stages", nargs=-1)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default="dvc.yaml",
    help="Output path for dvc.yaml (default: dvc.yaml)",
)
def export(stages: tuple[str, ...], output: pathlib.Path) -> None:
    """Export pipeline to DVC YAML format."""
    from pivot import dvc_compat

    stages_list = list(stages) if stages else None

    try:
        result = dvc_compat.export_dvc_yaml(output, stages=stages_list)
        click.echo(f"Exported {len(result['stages'])} stages to {output}")
    except Exception as e:
        raise click.ClickException(str(e)) from e


def _print_results(results: dict[str, ExecutionSummary]) -> None:
    """Print execution results in a readable format."""
    ran = 0
    skipped = 0

    for name, result in results.items():
        result_status = result["status"]
        reason = result["reason"]

        if result_status == "ran":
            ran += 1
            click.echo(f"{name}: ran ({reason})")
        else:
            skipped += 1
            click.echo(f"{name}: skipped")

    click.echo(f"\nTotal: {ran} ran, {skipped} skipped")


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
    )


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
