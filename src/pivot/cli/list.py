from __future__ import annotations

import json
from typing import TypedDict

import click

from pivot import registry
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers


class StageJsonOutput(TypedDict):
    """JSON output for a single stage."""

    name: str
    deps: list[str]
    outs: list[str]
    mutex: list[str]
    variant: str | None


class ListJsonOutput(TypedDict):
    """JSON output for pivot list --json."""

    stages: list[StageJsonOutput]


def _get_output_sources(stage_list: list[str]) -> dict[str, str]:
    """Build a map from output path to the stage that produces it."""
    return {
        out_path: name
        for name in stage_list
        for out_path in registry.REGISTRY.get(name)["outs_paths"]
    }


@cli_decorators.pivot_command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--deps", "show_deps", is_flag=True, help="Show stage dependencies")
@click.pass_context
def list_cmd(ctx: click.Context, as_json: bool, show_deps: bool) -> None:
    """List registered stages."""
    cli_ctx = cli_helpers.get_cli_context(ctx)
    verbose = cli_ctx["verbose"]
    quiet = cli_ctx["quiet"]
    stage_list = registry.REGISTRY.list_stages()

    if not stage_list:
        if as_json:
            click.echo(json.dumps(ListJsonOutput(stages=[])))
        elif not quiet:
            click.echo("No stages registered.")
            click.echo("Create a pipeline.py with @stage decorators, or a pivot.yaml file.")
        return

    if as_json:
        stages = [
            StageJsonOutput(
                name=name,
                deps=(info := registry.REGISTRY.get(name))["deps"],
                outs=info["outs_paths"],
                mutex=info["mutex"],
                variant=info["variant"],
            )
            for name in stage_list
        ]
        click.echo(json.dumps(ListJsonOutput(stages=stages), indent=2))
        return

    # Quiet mode: no output, just exit code
    if quiet:
        return

    # Build output->stage map for showing dep sources
    output_sources = _get_output_sources(stage_list) if show_deps else {}

    click.echo(f"Registered stages ({len(stage_list)}):")
    for name in stage_list:
        info = registry.REGISTRY.get(name)
        deps = info["deps"]
        outs = info["outs_paths"]
        click.echo(f"  {name}")

        if show_deps or verbose:
            if deps:
                click.echo("    deps:")
                for dep in deps:
                    source = output_sources.get(dep)
                    if source and source != name:
                        click.echo(f"      {dep} (from: {source})")
                    else:
                        click.echo(f"      {dep}")
            if outs:
                click.echo("    outs:")
                for out in outs:
                    click.echo(f"      {out}")
