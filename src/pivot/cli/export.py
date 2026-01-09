from __future__ import annotations

import pathlib

import click

from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command()
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

    result = dvc_compat.export_dvc_yaml(output, stages=stages_list)
    click.echo(f"Exported {len(result['stages'])} stages to {output}")
