from __future__ import annotations

from typing import TYPE_CHECKING

import click

from pivot import config, display
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.cli._run_common import ensure_stages_registered
from pivot.show import params as params_mod
from pivot.types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Mapping


def _resolve_targets(stages: tuple[str, ...]) -> tuple[list[str] | None, dict[str, str]]:
    """The stages whose params to read, each mapped stage (``pipeline.map``) expanded into its
    instances when its producer is current, as a run would expand it: a target naming a mapped
    stage stands for its instances, and naming one not yet expanded is refused. Without targets,
    also each mapped stage not yet expanded whose instances take params, with what it waits for:
    the params it hides.

    Raises:
        StageNotFoundError: A target names no stage, mapped stage or instance.
        click.ClickException: A target names a mapped stage not yet expanded.
    """
    targets: tuple[str, ...] | None = stages or None
    pipeline = cli_helpers.get_pipeline()
    waiting = dict[str, str]()
    if pipeline.list_templates():
        result, _ = cli_helpers.expand_mapped_stages()
        if targets is not None:
            targets = cli_helpers.require_expanded_targets(targets, result)
        else:
            waiting = {
                name: why
                for name, why in result["not_expanded"].items()
                if pipeline.get_template(name)["info"]["params"] is not None
            }
    return cli_helpers.resolve_stage_names(list(targets) if targets else None), waiting


def _echo_table(
    table: str, *, has_rows: bool, waiting: Mapping[str, str], output_format: OutputFormat | None
) -> None:
    """Echo ``table``, then the mapped stages not yet expanded whose params it lacks
    (``waiting``), as `pivot list` lists them: on stderr beside JSON, so stdout stays one
    document, and in place of a table with no row, whose empty message they explain."""
    if not waiting:
        click.echo(table)
        return
    lines = [f"Mapped stages, not yet expanded ({len(waiting)}):"]
    for name, why in waiting.items():
        lines.extend((f"  {name}", f"    {display.escape_control(why)}"))
    note = "\n".join(lines)
    if output_format == OutputFormat.JSON:
        click.echo(table)
        click.echo(note, err=True)
    elif has_rows:
        click.echo(f"{table}\n\n{note}")
    else:
        click.echo(note)


@click.group()
def params() -> None:
    """Display and compare parameters."""


@params.command("show")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--all",
    "all_pipelines",
    is_flag=True,
    default=False,
    help="Run across all pipelines in the project.",
)
@click.option(
    "--json", "output_format", flag_value=OutputFormat.JSON, default=None, help="Output as JSON"
)
@click.option("--md", "output_format", flag_value=OutputFormat.MD, help="Output as Markdown table")
@click.option(
    "--precision", default=None, type=click.IntRange(0, 10), help="Decimal precision for floats"
)
@cli_decorators.with_error_handling
def params_show(
    stages: tuple[str, ...],
    all_pipelines: bool,
    output_format: OutputFormat | None,
    precision: int | None,
) -> None:
    """Display current parameter values.

    If STAGES are specified, shows params for those stages only.
    Otherwise, shows params from all registered stages, and from each instance of a mapped
    stage whose producer is current; each other mapped stage that takes params is named as not
    yet expanded.
    """
    ensure_stages_registered(all_pipelines=all_pipelines)
    precision = precision if precision is not None else config.get_display_precision()
    stages_list, waiting = _resolve_targets(stages)
    workspace_params = params_mod.collect_params_from_stages(stages_list)

    output = params_mod.format_params_table(workspace_params, output_format, precision)
    _echo_table(
        output, has_rows=bool(workspace_params), waiting=waiting, output_format=output_format
    )


@params.command("diff")
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--all",
    "all_pipelines",
    is_flag=True,
    default=False,
    help="Run across all pipelines in the project.",
)
@click.option(
    "--json", "output_format", flag_value=OutputFormat.JSON, default=None, help="Output as JSON"
)
@click.option("--md", "output_format", flag_value=OutputFormat.MD, help="Output as Markdown table")
@click.option(
    "--precision", default=None, type=click.IntRange(0, 10), help="Decimal precision for floats"
)
@cli_decorators.with_error_handling
def params_diff(
    stages: tuple[str, ...],
    all_pipelines: bool,
    output_format: OutputFormat | None,
    precision: int | None,
) -> None:
    """Compare workspace parameters against git HEAD.

    If STAGES are specified, compares those stages only.
    Otherwise, compares all registered stages, and each instance of a mapped stage whose
    producer is current; each other mapped stage that takes params is named as not yet expanded.
    """
    ensure_stages_registered(all_pipelines=all_pipelines)
    precision = precision if precision is not None else config.get_display_precision()
    stages_list, waiting = _resolve_targets(stages)

    head_result = params_mod.get_params_from_head(stages_list)
    workspace_params = params_mod.collect_params_from_stages(stages_list)

    if not head_result["git_available"]:
        click.echo("Warning: Not in a git repository or no commits yet.", err=True)

    head_params = head_result["params"]

    if not head_params and not workspace_params:
        _echo_table(
            "No parameters found in registered stages.",
            has_rows=False,
            waiting=waiting,
            output_format=output_format,
        )
        return

    diffs = params_mod.diff_params(head_params, workspace_params)
    output = params_mod.format_diff_table(diffs, output_format, precision)
    _echo_table(output, has_rows=bool(diffs), waiting=waiting, output_format=output_format)
