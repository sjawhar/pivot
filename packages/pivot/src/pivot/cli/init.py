from __future__ import annotations

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
    state_dir_is_configured = project.state_dir_is_configured(cwd)
    gitignore_path = project.state_dir_for(cwd) / ".gitignore"
    try:
        gitignore_path.lstat()
        gitignore_existed = True
    except FileNotFoundError:
        gitignore_existed = False
    state_dir = project.ensure_state_dir(cwd)

    if force and not state_dir_is_configured:
        existing_content = gitignore_path.read_text()
        if existing_content != project.STATE_GITIGNORE_CONTENT and not quiet:
            click.echo(f"Warning: Overwriting existing {gitignore_path}", err=True)
        gitignore_path.write_text(project.STATE_GITIGNORE_CONTENT)
    elif force and gitignore_existed and not quiet:
        click.echo(f"Leaving {gitignore_path}; Pivot does not own it.")

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
        try:
            display_stages = stages_dir.relative_to(cwd)
            display_gitignore = gitignore_path.relative_to(cwd)
        except ValueError:
            display_stages = stages_dir
            display_gitignore = gitignore_path
        click.echo(f"  {display_stages}/")
        if not state_dir_is_configured or not gitignore_existed:
            click.echo(f"  {display_gitignore}")
        if created_pivotignore:
            click.echo("  .pivotignore")
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Create pipeline.py to define your pipeline stages")
        click.echo("  2. Run 'pivot repro' to execute the pipeline")
        click.echo("  3. See 'pivot --help' for more commands")
