from __future__ import annotations

from typing import TYPE_CHECKING

import click

from pivot import outputs, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import targets as cli_targets
from pivot.cli.run import ensure_stages_registered
from pivot.show import metrics as metrics_mod

if TYPE_CHECKING:
    from pivot.types import OutputFormat


@click.group()
def metrics() -> None:
    """Display and compare metrics."""


@metrics.command("show")
@click.argument("targets", nargs=-1)
@click.option("--json", "output_format", flag_value="json", default=None, help="Output as JSON")
@click.option("--md", "output_format", flag_value="md", help="Output as Markdown table")
@click.option("-R", "--recursive", is_flag=True, help="Search directories recursively")
@click.option("--precision", default=5, type=int, help="Decimal precision for floats")
@cli_decorators.with_error_handling
def metrics_show(
    targets: tuple[str, ...],
    output_format: OutputFormat,
    recursive: bool,
    precision: int,
) -> None:
    """Display metric values in tabular format.

    TARGETS can be file paths or stage names. If a stage name is provided,
    all its metric outputs are included.

    If no TARGETS are specified, shows metrics from all registered stages.
    """
    proj_root = project.get_project_root()
    ensure_stages_registered()

    paths = cli_targets.resolve_and_validate(targets, proj_root, outputs.Metric)
    if paths is not None:
        all_metrics = metrics_mod.collect_metrics_from_files(list(paths), recursive)
    else:
        all_metrics = metrics_mod.collect_all_stage_metrics_flat()

    output = metrics_mod.format_metrics_table(all_metrics, output_format, precision)
    click.echo(output)


@metrics.command("diff")
@click.argument("targets", nargs=-1)
@click.option("--json", "output_format", flag_value="json", default=None, help="Output as JSON")
@click.option("--md", "output_format", flag_value="md", help="Output as Markdown table")
@click.option("-R", "--recursive", is_flag=True, help="Search directories recursively")
@click.option("--no-path", is_flag=True, help="Hide path column")
@click.option("--precision", default=5, type=int, help="Decimal precision for floats")
@cli_decorators.with_error_handling
def metrics_diff(
    targets: tuple[str, ...],
    output_format: OutputFormat,
    recursive: bool,
    no_path: bool,
    precision: int,
) -> None:
    """Compare workspace metric files against git HEAD.

    TARGETS can be file paths or stage names. If a stage name is provided,
    all its metric outputs are included.

    If no TARGETS are specified, compares all registered stages' Metric outputs.
    """
    proj_root = project.get_project_root()
    ensure_stages_registered()

    paths = cli_targets.resolve_and_validate(targets, proj_root, outputs.Metric)
    if paths is not None:
        head_info = metrics_mod.get_metric_info_from_head()
        head_info = {k: v for k, v in head_info.items() if k in paths}
        head_metrics = metrics_mod.collect_metrics_from_head(list(paths), head_info)
        workspace_metrics = metrics_mod.collect_metrics_from_files(
            list(paths), recursive, tolerant=False
        )
    else:
        head_info = metrics_mod.get_metric_info_from_head()
        if not head_info:
            click.echo("No metrics found in registered stages.")
            return
        all_paths = set(head_info.keys())
        head_metrics = metrics_mod.collect_metrics_from_head(list(all_paths), head_info)
        workspace_metrics = metrics_mod.collect_metrics_from_files(
            list(all_paths), recursive, tolerant=True
        )

    diffs = metrics_mod.diff_metrics(head_metrics, workspace_metrics)
    output = metrics_mod.format_diff_table(diffs, output_format, precision, show_path=not no_path)
    click.echo(output)
