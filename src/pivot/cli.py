"""Command-line interface for Pivot.

Provides commands to run and manage pipelines from the command line.
"""

import logging
import pathlib
from typing import Any

import click

from pivot import executor
from pivot.registry import REGISTRY

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Fast pipeline execution with per-stage caching."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.option("--stages", "-s", multiple=True, metavar="STAGE", help="Specific stages to run")
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would run without executing")
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    cache_dir: pathlib.Path | None,
    dry_run: bool,
) -> None:
    """Execute pipeline stages."""
    if dry_run:
        ctx.invoke(dry_run_cmd, stages=stages, cache_dir=cache_dir)
        return

    try:
        stage_list = list(stages) if stages else None
        results = executor.run(
            stages=stage_list,
            cache_dir=cache_dir,
        )
        _print_results(results)
    except Exception as e:
        raise click.ClickException(str(e)) from e


@cli.command("dry-run")
@click.option("--stages", "-s", multiple=True, metavar="STAGE", help="Specific stages to run")
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
def dry_run_cmd(stages: tuple[str, ...], cache_dir: pathlib.Path | None) -> None:
    """Show what would run without executing."""
    from pivot import dag, lock, project

    try:
        graph = REGISTRY.build_dag(validate=True)
        stage_list = list(stages) if stages else None
        execution_order = dag.get_execution_order(graph, stage_list)

        if not execution_order:
            click.echo("No stages to run")
            return

        cache_path = cache_dir or project.get_project_root() / ".pivot" / "cache"

        click.echo("Would run:")
        for stage_name in execution_order:
            stage_info = REGISTRY.get(stage_name)
            stage_lock = lock.StageLock(stage_name, cache_path)

            current_fingerprint = stage_info["fingerprint"]
            current_params = executor.extract_params(stage_info)
            dep_hashes, missing = executor.hash_dependencies(stage_info.get("deps", []))

            if missing:
                click.echo(f"  {stage_name}: would run (missing deps: {', '.join(missing)})")
                continue

            changed, reason = stage_lock.is_changed(current_fingerprint, current_params, dep_hashes)

            if changed:
                click.echo(f"  {stage_name}: would run ({reason})")
            else:
                click.echo(f"  {stage_name}: would skip (unchanged)")

    except Exception as e:
        raise click.ClickException(str(e)) from e


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pipeline status."""
    verbose = ctx.obj.get("verbose", False)
    stage_list = REGISTRY.list_stages()

    if not stage_list:
        click.echo("No stages registered")
        return

    click.echo(f"Registered stages ({len(stage_list)}):")
    for name in stage_list:
        info = REGISTRY.get(name)
        deps = info.get("deps", [])
        outs = info.get("outs_paths", [])
        click.echo(f"  {name}")
        if verbose:
            click.echo(f"    deps: {deps}")
            click.echo(f"    outs: {outs}")


def _print_results(results: dict[str, dict[str, Any]]) -> None:
    """Print execution results in a readable format."""
    ran = 0
    skipped = 0

    for name, result in results.items():
        result_status = result["status"]
        reason = result.get("reason", "")

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
