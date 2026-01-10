from __future__ import annotations

import json

import click

from pivot.cli import decorators as cli_decorators
from pivot.pipeline import yaml as pipeline_yaml


@cli_decorators.pivot_command("schema")
@click.option(
    "--indent",
    type=int,
    default=2,
    help="JSON indentation (0 for compact)",
)
def schema(indent: int) -> None:
    """Output JSON Schema for pivot.yaml configuration."""
    json_schema = pipeline_yaml.PipelineConfig.model_json_schema()
    indent_val = indent if indent > 0 else None
    click.echo(json.dumps(json_schema, indent=indent_val))
