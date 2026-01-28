from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from pivot import config
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.storage import state
from pivot.types import StageStatus

if TYPE_CHECKING:
    from pivot.run_history import RunManifest


@cli_decorators.pivot_command(auto_discover=False)
@click.option("--limit", "-n", default=10, help="Number of runs to show")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def history(ctx: click.Context, limit: int, output_json: bool) -> None:
    """List recent pipeline runs."""
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    state_db_path = config.get_state_db_path()

    with state.StateDB(state_db_path) as state_db:
        runs = state_db.list_runs(limit=limit)

    if output_json:
        click.echo(json.dumps(runs, indent=2))
        return

    if quiet:
        return

    if not runs:
        click.echo("No runs recorded")
        return

    click.echo("Run ID                      Ran  Skipped  Failed  Duration")
    click.echo("-" * 60)

    for run in runs:
        _print_run_summary(run)


def _print_run_summary(run: RunManifest) -> None:
    """Print single-line summary of a run as table row."""
    stages = run["stages"]
    ran = sum(1 for s in stages.values() if s["status"] == StageStatus.RAN)
    skipped = sum(1 for s in stages.values() if s["status"] == StageStatus.SKIPPED)
    failed = sum(1 for s in stages.values() if s["status"] == StageStatus.FAILED)

    total_duration_ms = sum(s["duration_ms"] for s in stages.values())
    duration_str = _format_duration(total_duration_ms)

    click.echo(f"{run['run_id']:<26}  {ran:>3}  {skipped:>7}  {failed:>6}  {duration_str:>8}")


@cli_decorators.pivot_command("show", auto_discover=False)
@click.argument("run_id", required=False)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def show_cmd(ctx: click.Context, run_id: str | None, output_json: bool) -> None:
    """Show details of a specific run.

    If RUN_ID is not provided, shows the most recent run.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    state_db_path = config.get_state_db_path()

    with state.StateDB(state_db_path) as state_db:
        if run_id:
            run = state_db.read_run(run_id)
            if run is None:
                raise click.ClickException(f"Run not found: {run_id}")
        else:
            runs = state_db.list_runs(limit=1)
            if not runs:
                raise click.ClickException("No runs recorded")
            run = runs[0]

    if output_json:
        click.echo(json.dumps(run, indent=2))
        return

    if not quiet:
        _print_run_detail(run)


def _print_run_detail(run: RunManifest) -> None:
    """Print detailed view of a run."""
    click.echo(f"Run:     {run['run_id']}")
    click.echo(f"Started: {run['started_at']}")
    click.echo(f"Ended:   {run['ended_at']}")

    if run["targeted_stages"]:
        click.echo("Targets:")
        for target in run["targeted_stages"]:
            click.echo(f"  - {target}")

    click.echo()
    click.echo("Stages:")

    stages = run["stages"]
    if not stages:
        click.echo("  No stages")
        return

    click.echo(f"  {'Stage':<25} {'Status':<8} {'Duration':>8}  Reason")
    click.echo(f"  {'-' * 70}")

    for name in run["execution_order"]:
        record = stages[name]
        status = record["status"]
        reason = record["reason"]
        duration_str = _format_duration(record["duration_ms"])

        match status:
            case StageStatus.RAN:
                icon = "\u2713"
            case StageStatus.SKIPPED:
                icon = "\u2022"
            case StageStatus.FAILED:
                icon = "\u2717"
            case _:
                icon = "?"

        click.echo(f"  {icon} {name:<24} {status:<8} {duration_str:>8}  {reason}")


def _format_duration(duration_ms: int) -> str:
    """Format duration in human-readable form."""
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    if duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) / 1000
    return f"{minutes}m{seconds:.0f}s"
