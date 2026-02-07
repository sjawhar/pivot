from __future__ import annotations

import click

from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.executor import commit


@cli_decorators.pivot_command("commit")
@click.argument("stages", nargs=-1)
@click.pass_context
def commit_command(ctx: click.Context, stages: tuple[str, ...]) -> None:
    """Commit current workspace state for stages.

    Hashes current deps and outputs, writes lock files and cache.
    Without arguments, commits all stale stages.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    stage_names = list(stages) if stages else None
    committed, failed = commit.commit_stages(stage_names)

    if not quiet:
        if committed:
            click.echo(f"Committed {len(committed)} stage(s):")
            for stage_name in committed:
                click.echo(f"  {stage_name}")
        if failed:
            click.echo(f"Failed {len(failed)} stage(s):")
            for stage_name in failed:
                click.echo(f"  {stage_name}")
        if not committed and not failed:
            click.echo("Nothing to commit")

    if failed:
        ctx.exit(1)
