from __future__ import annotations

import json
from typing import TypedDict

import click

from pivot import project, types
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets


class StageJsonOutput(TypedDict):
    """JSON output for a single stage."""

    name: str
    deps: list[str]
    outs: list[str]
    resources: dict[str, int | str]
    variant: str | None


class ListJsonOutput(TypedDict):
    """JSON output for pivot list --json."""

    stages: list[StageJsonOutput]


def _get_output_sources(stage_list: list[str]) -> dict[types.ArtifactIdentity, str]:
    """Build a map from each output's identity to the stage that produces it."""
    return {
        out.identity: name for name in stage_list for out in cli_helpers.get_stage(name)["outs"]
    }


@cli_decorators.pivot_command("list", allow_all=True)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--deps", "show_deps", is_flag=True, help="Show stage dependencies")
@click.pass_context
def list_cmd(ctx: click.Context, as_json: bool, show_deps: bool) -> None:
    """List registered stages."""
    cli_ctx = cli_helpers.get_cli_context(ctx)
    verbose = cli_ctx["verbose"]
    quiet = cli_ctx["quiet"]
    stage_list = cli_helpers.list_stages()

    if not stage_list:
        if as_json:
            click.echo(json.dumps(ListJsonOutput(stages=[])))
        elif not quiet:
            click.echo("No stages registered.")
            click.echo("Create a pipeline.py with stage functions.")
        return

    if as_json:
        stages = list[StageJsonOutput]()
        for name in stage_list:
            info = cli_helpers.get_stage(name)
            stages.append(
                StageJsonOutput(
                    name=name,
                    deps=[types.identity_key(dep.identity) for dep in info["deps"].values()],
                    outs=[types.identity_key(out.identity) for out in info["outs"]],
                    resources=info["resources"],
                    variant=info["variant"],
                )
            )
        click.echo(json.dumps(ListJsonOutput(stages=stages), indent=2))
        return

    # Quiet mode: no output, just exit code
    if quiet:
        return

    # Build output->stage map for showing dep sources
    output_sources = _get_output_sources(stage_list) if show_deps else {}

    click.echo(f"Registered stages ({len(stage_list)}):")
    store = cli_helpers.require_workspace_store()
    project_root = project.get_project_root()
    for name in stage_list:
        info = cli_helpers.get_stage(name)
        deps = list(info["deps"].values())
        outs = info["outs"]
        click.echo(f"  {name}")
        resources = info["resources"]
        if resources:
            click.echo(
                "    resources: " + ", ".join(f"{key}={value}" for key, value in resources.items())
            )

        if show_deps or verbose:
            if deps:
                click.echo("    deps:")
                for dep in deps:
                    dep_key = types.identity_key(dep.identity)
                    # One file of a directory output comes from its directory's producer
                    source = output_sources.get(types.source_identity(dep))
                    if source and source != name:
                        click.echo(f"      {dep_key} (from: {source})")
                    else:
                        click.echo(f"      {dep_key}")
            if outs:
                click.echo("    outs:")
                for out in outs:
                    output_path = project.to_relative_path(
                        store.resolve_display_path(out), project_root
                    )
                    click.echo(
                        f"      {cli_targets.format_identity_display(out.identity, output_path)}"
                    )
