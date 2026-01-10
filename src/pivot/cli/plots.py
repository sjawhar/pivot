from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click

from pivot import outputs, path_policy, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import targets as cli_targets
from pivot.cli.run import ensure_stages_registered
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
    """Render plots as HTML image gallery.

    TARGETS can be file paths or stage names. If a stage name is provided,
    all its plot outputs are included.

    If no TARGETS are specified, shows plots from all registered stages.
    """
    proj_root = project.get_project_root()
    ensure_stages_registered()

    # Validate output path stays within project
    path_policy.require_valid_path(
        str(output),
        path_policy.PathType.CLI_OUTPUT,
        proj_root,
        context="plots show --output",
    )

    if targets:
        valid_targets = cli_targets.validate_targets(targets)
        if not valid_targets:
            return

        plot_list, missing = cli_targets.resolve_plot_infos(valid_targets, proj_root)
        if missing:
            raise click.ClickException(f"Unknown targets: {', '.join(missing)}")
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
    """Show which plots changed since last commit.

    TARGETS can be file paths or stage names. If a stage name is provided,
    all its plot outputs are included.

    If no TARGETS are specified, compares all registered stages' Plot outputs.
    """
    proj_root = project.get_project_root()
    ensure_stages_registered()

    paths = cli_targets.resolve_and_validate(targets, proj_root, outputs.Plot)
    if paths is not None:
        # Filter lock file hashes to only requested paths
        all_old_hashes = plots_mod.get_plot_hashes_from_head()
        old_hashes = {k: v for k, v in all_old_hashes.items() if k in paths}
    else:
        old_hashes = plots_mod.get_plot_hashes_from_head()
        if not old_hashes:
            click.echo("No plots found in registered stages.")
            return
        paths = set(old_hashes.keys())

    new_hashes = plots_mod.get_plot_hashes_from_workspace(list(paths))
    diffs = plots_mod.diff_plots(old_hashes, new_hashes)

    output_format: OutputFormat = "json" if json_output else ("md" if md else None)
    result = plots_mod.format_diff_table(diffs, output_format, show_path=not no_path)
    click.echo(result)
