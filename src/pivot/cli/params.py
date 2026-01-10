from __future__ import annotations

from typing import TYPE_CHECKING

import click

from pivot import exceptions
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.show import params as params_mod

if TYPE_CHECKING:
    from pivot.types import OutputFormat


@click.group()
def params() -> None:
    """Display and compare parameters."""


@params.command("show")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--json", "output_format", flag_value="json", default=None, help="Output as JSON")
@click.option("--md", "output_format", flag_value="md", help="Output as Markdown table")
@click.option(
    "--precision", default=5, type=click.IntRange(0, 10), help="Decimal precision for floats"
)
@cli_decorators.with_error_handling
def params_show(
    stages: tuple[str, ...],
    output_format: OutputFormat,
    precision: int,
) -> None:
    """Display current parameter values.

    If STAGES are specified, shows params for those stages only.
    Otherwise, shows params from all registered stages.
    """
    stages_list = list(stages) if stages else None
    result = params_mod.collect_params_from_stages(stages_list)

    if result["unknown_stages"]:
        raise exceptions.StageNotFoundError(
            f"Unknown stages: {', '.join(result['unknown_stages'])}"
        )

    output = params_mod.format_params_table(result["params"], output_format, precision)
    click.echo(output)


@params.command("diff")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--json", "output_format", flag_value="json", default=None, help="Output as JSON")
@click.option("--md", "output_format", flag_value="md", help="Output as Markdown table")
@click.option(
    "--precision", default=5, type=click.IntRange(0, 10), help="Decimal precision for floats"
)
@cli_decorators.with_error_handling
def params_diff(
    stages: tuple[str, ...],
    output_format: OutputFormat,
    precision: int,
) -> None:
    """Compare workspace parameters against git HEAD.

    If STAGES are specified, compares those stages only.
    Otherwise, compares all registered stages.
    """
    stages_list = list(stages) if stages else None

    head_result = params_mod.get_params_from_head(stages_list)
    workspace_result = params_mod.collect_params_from_stages(stages_list)

    if workspace_result["unknown_stages"]:
        raise exceptions.StageNotFoundError(
            f"Unknown stages: {', '.join(workspace_result['unknown_stages'])}"
        )

    if not head_result["git_available"]:
        click.echo("Warning: Not in a git repository or no commits yet.", err=True)

    head_params = head_result["params"]
    workspace_params = workspace_result["params"]

    if not head_params and not workspace_params:
        click.echo("No parameters found in registered stages.")
        return

    diffs = params_mod.diff_params(head_params, workspace_params)
    output = params_mod.format_diff_table(diffs, output_format, precision)
    click.echo(output)
