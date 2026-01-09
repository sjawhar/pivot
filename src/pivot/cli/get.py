from __future__ import annotations

import pathlib

import click

from pivot import cache, config, get, project
from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command("get")
@click.argument("targets", nargs=-1, required=True)
@click.option(
    "--rev",
    "-r",
    required=True,
    help="Git revision (SHA, branch, tag) to retrieve files from",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output path for single file target (incompatible with multiple targets or stage names)",
)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def get_cmd(
    targets: tuple[str, ...],
    rev: str,
    output: pathlib.Path | None,
    checkout_mode: str | None,
    force: bool,
) -> None:
    """Retrieve files or stage outputs from a specific git revision.

    TARGETS can be file paths or stage names.

    \b
    Examples:
      pivot get --rev v1.0 model.pkl              # Get file from tag
      pivot get --rev v1.0 model.pkl -o old.pkl   # Get file to alternate location
      pivot get --rev abc123 train                # Get all outputs from stage
    """
    project_root = project.get_project_root()
    cache_dir = project_root / ".pivot" / "cache"

    # Determine checkout modes
    mode_strings = [checkout_mode] if checkout_mode else config.get_checkout_mode_order()
    checkout_modes = [cache.CheckoutMode(m) for m in mode_strings]

    messages = get.restore_targets_from_revision(
        targets=list(targets),
        rev=rev,
        output=output,
        cache_dir=cache_dir,
        checkout_modes=checkout_modes,
        force=force,
    )

    for msg in messages:
        click.echo(msg)
