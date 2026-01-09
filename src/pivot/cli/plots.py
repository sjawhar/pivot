from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click

from pivot import project
from pivot.cli import decorators as cli_decorators
from pivot.show import plots as plots_mod

if TYPE_CHECKING:
    from pivot.types import OutputFormat


@click.group()
def plots() -> None:
    """Display and compare plots."""


@plots.command("show")
@click.argument("targets", nargs=-1)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=pathlib.Path),
    default="pivot_plots/index.html",
    help="Output HTML path (default: pivot_plots/index.html)",
)
@click.option("--open", "open_browser", is_flag=True, help="Open browser after rendering")
@cli_decorators.with_error_handling
def plots_show(targets: tuple[str, ...], output: pathlib.Path, open_browser: bool) -> None:
    """Render plots as HTML image gallery."""
    if targets:
        # Filter to specific targets
        all_plots = plots_mod.collect_plots_from_stages()
        target_set = set(targets)
        plot_list = [p for p in all_plots if p["path"] in target_set]
    else:
        plot_list = plots_mod.collect_plots_from_stages()

    if not plot_list:
        click.echo("No plots found.")
        return

    output_path = plots_mod.render_plots_html(plot_list, output)
    click.echo(f"Rendered {len(plot_list)} plot(s) to {output_path}")

    if open_browser:
        import webbrowser

        webbrowser.open(f"file://{output_path.resolve()}")


@plots.command("diff")
@click.argument("targets", nargs=-1)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--md", is_flag=True, help="Output as markdown table")
@click.option("--no-path", "no_path", is_flag=True, help="Hide path column")
@cli_decorators.with_error_handling
def plots_diff(targets: tuple[str, ...], json_output: bool, md: bool, no_path: bool) -> None:
    """Show which plots changed since last commit."""
    # Get old hashes from lock files at Git HEAD
    all_old_hashes = plots_mod.get_plot_hashes_from_head()

    if not all_old_hashes:
        click.echo("No plots found in registered stages.")
        return

    # Filter to targets if specified
    if targets:
        # Normalize user targets to relative paths from project root
        proj_root = project.get_project_root()
        target_set = {
            project.to_relative_path(project.normalize_path(t), proj_root) for t in targets
        }
        old_hashes = {k: v for k, v in all_old_hashes.items() if k in target_set}
        paths = list(target_set)
    else:
        old_hashes = all_old_hashes
        paths = list(old_hashes.keys())

    # Get current hashes from workspace
    new_hashes = plots_mod.get_plot_hashes_from_workspace(paths)

    # Compute diffs
    diffs = plots_mod.diff_plots(old_hashes, new_hashes)

    # Format output
    output_format: OutputFormat = "json" if json_output else ("md" if md else None)
    result = plots_mod.format_diff_table(diffs, output_format, show_path=not no_path)
    click.echo(result)
