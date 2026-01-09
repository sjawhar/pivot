from __future__ import annotations

import click

from pivot import registry
from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List registered stages."""
    verbose = ctx.obj.get("verbose", False)
    stage_list = registry.REGISTRY.list_stages()

    if not stage_list:
        click.echo("No stages registered.")
        click.echo("Create a pipeline.py with @stage decorators, or a pivot.yaml file.")
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
