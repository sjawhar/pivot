from __future__ import annotations

import pathlib

import click

from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default="dvc.yaml",
    help="Output path for dvc.yaml (default: dvc.yaml)",
)
@click.pass_context
def export(ctx: click.Context, stages: tuple[str, ...], output: pathlib.Path) -> None:
    """Export pipeline to DVC YAML format."""
    from pivot import dvc_compat, path_policy, project

    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    # Validate output path stays within project
    proj_root = project.get_project_root()
    path_policy.require_valid_path(
        str(output),
        path_policy.PathType.CLI_OUTPUT,
        proj_root,
        context="export --output",
    )

    stages_list = list(stages) if stages else None

    result = dvc_compat.export_dvc_yaml(output, stages=stages_list)
    if not quiet:
        click.echo(f"Exported {len(result['stages'])} stages to {output}")
