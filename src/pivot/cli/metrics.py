from __future__ import annotations

from typing import TYPE_CHECKING

import click

from pivot import project
from pivot.cli import decorators as cli_decorators
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

    If TARGETS are specified, parses those files/directories directly.
    Otherwise, shows metrics from all registered stages' Metric outputs.
    """
    if targets:
        all_metrics = metrics_mod.collect_metrics_from_files(list(targets), recursive)
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

    If TARGETS are specified, compares those files/directories.
    Otherwise, compares all registered stages' Metric outputs.
    """
    # Get HEAD info (hashes from lock files)
    head_info = metrics_mod.get_metric_info_from_head()

    if not head_info:
        click.echo("No metrics found in registered stages.")
        return

    # Filter to targets if specified
    if targets:
        proj_root = project.get_project_root()
        target_set = {
            project.to_relative_path(project.normalize_path(t), proj_root) for t in targets
        }
        head_info = {k: v for k, v in head_info.items() if k in target_set}
        paths = list(target_set)
    else:
        paths = list(head_info.keys())

    # Get metrics from HEAD (cache-first, git-fallback)
    head_metrics = metrics_mod.collect_metrics_from_head(paths, head_info)

    # Get current workspace metrics
    workspace_metrics = metrics_mod.collect_metrics_from_files(paths, recursive)

    diffs = metrics_mod.diff_metrics(head_metrics, workspace_metrics)
    output = metrics_mod.format_diff_table(diffs, output_format, precision, show_path=not no_path)
    click.echo(output)
