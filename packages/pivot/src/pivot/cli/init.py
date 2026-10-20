from __future__ import annotations

import os
import pathlib

import click

from pivot import exceptions, ignore, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers


@cli_decorators.pivot_command(auto_discover=False)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite .pivot/.gitignore only when using the default state directory",
)
@click.pass_context
def init(ctx: click.Context, force: bool) -> None:
    """Initialize a new Pivot project."""
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    cwd = pathlib.Path.cwd()
    pivot_dir = cwd / ".pivot"

    if pivot_dir.is_symlink():
        raise exceptions.InitError(f"'{pivot_dir}' is a symlink; refusing to initialize")

    if pivot_dir.exists():
        if not pivot_dir.is_dir():
            raise exceptions.InitError(f"'{pivot_dir}' exists but is not a directory")
        if not force:
            raise exceptions.AlreadyInitializedError(
                f"Pivot already initialized in {pivot_dir.parent}"
            )

    pivot_dir.mkdir(exist_ok=True)
    marker_gitignore = pivot_dir / ".gitignore"
    created_gitignores = list[pathlib.Path]()
    if project.state_dir_is_configured(cwd):
        # .pivot keeps the config and its lock, so it keeps an ignore file; init is its only
        # writer. Neither ignore file is ever overwritten in a configured layout.
        if project.create_gitignore_if_absent(marker_gitignore):
            created_gitignores.append(marker_gitignore)
        state_gitignore = project.state_dir_for(cwd) / ".gitignore"
        state_gitignore_existed = os.path.lexists(state_gitignore)
        state_dir = project.ensure_state_dir(cwd)
        if not state_gitignore_existed:
            created_gitignores.append(state_gitignore)
        elif force and state_gitignore != marker_gitignore and not quiet:
            click.echo(f"Leaving {state_gitignore}; Pivot does not own it.")
    else:
        state_dir = project.ensure_state_dir(cwd)
        # Warn if overwriting existing .gitignore with custom content
        if force and marker_gitignore.exists():
            existing_content = marker_gitignore.read_text()
            if existing_content != project.STATE_GITIGNORE_CONTENT and not quiet:
                click.echo("Warning: Overwriting existing .pivot/.gitignore", err=True)
        marker_gitignore.write_text(project.STATE_GITIGNORE_CONTENT)
        created_gitignores.append(marker_gitignore)

    stages_dir = state_dir / "stages"
    stages_dir.mkdir(exist_ok=True)

    # Create .pivotignore with default patterns if it doesn't exist
    pivotignore_path = pathlib.Path.cwd() / ".pivotignore"
    created_pivotignore = False
    if not pivotignore_path.exists():
        pivotignore_content = "\n".join(ignore.get_default_patterns()) + "\n"
        pivotignore_path.write_text(pivotignore_content)
        created_pivotignore = True

    if not quiet:
        click.echo("Initialized Pivot project.")
        click.echo()
        click.echo("Created:")
        click.echo("  .pivot/")
        click.echo(f"  {_display_path(stages_dir, cwd)}/")
        for gitignore_path in dict.fromkeys(created_gitignores):
            click.echo(f"  {_display_path(gitignore_path, cwd)}")
        if created_pivotignore:
            click.echo("  .pivotignore")
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Create pipeline.py to define your pipeline stages")
        click.echo("  2. Run 'pivot repro' to execute the pipeline")
        click.echo("  3. See 'pivot --help' for more commands")


def _display_path(path: pathlib.Path, cwd: pathlib.Path) -> pathlib.Path:
    """Show ``path`` relative to the project when it lies inside it."""
    try:
        return path.relative_to(cwd)
    except ValueError:
        return path
