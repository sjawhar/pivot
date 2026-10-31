from __future__ import annotations

from typing import TypedDict

import click

from pivot import display, expansion, project, types
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


class TemplateJsonOutput(TypedDict):
    """JSON output for one stage mapped over a directory output (``Pipeline.map``)."""

    name: str
    source: str  # the directory output it is mapped over
    expanded: bool
    instances: list[str]  # the instances it expanded into, empty when not expanded
    reason: str | None  # why it is not expanded; None when it is


class ListJsonOutput(TypedDict):
    """JSON output for pivot list --json."""

    stages: list[StageJsonOutput]
    templates: list[TemplateJsonOutput]


def _get_output_sources(stage_list: list[str]) -> dict[types.ArtifactIdentity, str]:
    """Build a map from each output's identity to the stage that produces it."""
    return {
        out.identity: name for name in stage_list for out in cli_helpers.get_stage(name)["outs"]
    }


def _templates_json(result: expansion.Expansion) -> list[TemplateJsonOutput]:
    pipeline = cli_helpers.get_pipeline()
    return [
        TemplateJsonOutput(
            name=name,
            source=types.identity_key(pipeline.get_template(name)["source"]),
            expanded=name in result["expanded"],
            instances=result["expanded"][name] if name in result["expanded"] else [],
            reason=result["not_expanded"][name] if name in result["not_expanded"] else None,
        )
        for name in pipeline.list_templates()
    ]


@cli_decorators.pivot_command("list", allow_all=True)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--deps", "show_deps", is_flag=True, help="Show stage dependencies")
@click.pass_context
def list_cmd(ctx: click.Context, as_json: bool, show_deps: bool) -> None:
    """List registered stages.

    A stage mapped over a directory output lists as its instances when its producer is current,
    and under "Mapped stages, not yet expanded" otherwise.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    verbose = cli_ctx["verbose"]
    quiet = cli_ctx["quiet"]
    result, _ = cli_helpers.expand_mapped_stages()
    stage_list = cli_helpers.list_stages()

    if not stage_list:
        if as_json:
            click.echo(display.to_json(ListJsonOutput(stages=[], templates=[])))
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
        output = ListJsonOutput(stages=stages, templates=_templates_json(result))
        click.echo(display.to_json(output, indent=2))
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

    if result["not_expanded"]:
        click.echo()
        click.echo(f"Mapped stages, not yet expanded ({len(result['not_expanded'])}):")
        for name, reason in result["not_expanded"].items():
            click.echo(f"  {name}")
            click.echo(f"    {display.escape_control(reason)}")
