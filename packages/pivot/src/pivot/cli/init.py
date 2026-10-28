from __future__ import annotations

import os
import pathlib

import click

from pivot import exceptions, ignore
from pivot import state_dir as state_dir_mod
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers

# A listing's bytes must hash to the record in its lock or .pvt, so git must never merge two
# listings line by line (the result would hash to neither side's record): a merge that changes
# one on both sides conflicts instead, beside its lock, and both are taken from one side.
_GITATTRIBUTES_LINE = "*.listing -merge"
_GITATTRIBUTES_CONTENT = f"""\
# Pivot listing files: never merge one line by line; take it and its lock from one side
{_GITATTRIBUTES_LINE}
"""



def _ensure_gitattributes(path: pathlib.Path) -> bool:
    """Add the listing merge rule to the project's .gitattributes; whether the file was created
    (an existing one keeps its lines and gains the rule if it lacks it)."""
    try:
        existing = path.read_text()
    except FileNotFoundError:
        path.write_text(_GITATTRIBUTES_CONTENT)
        return True
    if _GITATTRIBUTES_LINE not in (line.strip() for line in existing.splitlines()):
        separator = "" if not existing or existing.endswith("\n") else "\n"
        path.write_text(existing + separator + _GITATTRIBUTES_CONTENT)
    return False


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
    # A config that cannot be read, or a core.state_dir that cannot be used, leaves 0.5.2's
    # default layout after one warning: 0.5.2's init never read the config.
    state_dir = state_dir_mod.state_dir_or_default(cwd)
    if state_dir != pivot_dir:
        # .pivot keeps the config and its lock, so it keeps an ignore file; init is its only
        # writer. Neither ignore file is ever overwritten in a configured layout.
        if state_dir_mod.create_gitignore_if_absent(marker_gitignore):
            created_gitignores.append(marker_gitignore)
        state_gitignore = state_dir / ".gitignore"
        state_gitignore_existed = os.path.lexists(state_gitignore)
        state_dir_mod.ensure_state_dir(cwd)
        if not state_gitignore_existed:
            state_dir_mod.create_gitignore_if_absent(state_gitignore)
            created_gitignores.append(state_gitignore)
        elif force and not quiet:
            click.echo(f"Leaving {state_gitignore}; Pivot does not own it.")
    else:
        # Warn if overwriting existing .gitignore with custom content
        if force and marker_gitignore.exists():
            existing_content = marker_gitignore.read_text()
            if existing_content != state_dir_mod.STATE_GITIGNORE_CONTENT and not quiet:
                click.echo("Warning: Overwriting existing .pivot/.gitignore", err=True)
        marker_gitignore.write_text(state_dir_mod.STATE_GITIGNORE_CONTENT)
        created_gitignores.append(marker_gitignore)

    stages_dir = state_dir / "stages"
    state_dir_mod.make_dirs(stages_dir)

    # Create .pivotignore with default patterns if it doesn't exist
    pivotignore_path = pathlib.Path.cwd() / ".pivotignore"
    created_pivotignore = False
    if not pivotignore_path.exists():
        pivotignore_content = "\n".join(ignore.get_default_patterns()) + "\n"
        pivotignore_path.write_text(pivotignore_content)
        created_pivotignore = True

    created_gitattributes = _ensure_gitattributes(cwd / ".gitattributes")

    if not quiet:
        click.echo("Initialized Pivot project.")
        click.echo()
        click.echo("Created:")
        click.echo("  .pivot/")
        click.echo(f"  {_display_path(stages_dir, cwd)}/")
        for gitignore_path in created_gitignores:
            click.echo(f"  {_display_path(gitignore_path, cwd)}")
        if created_pivotignore:
            click.echo("  .pivotignore")
        if created_gitattributes:
            click.echo("  .gitattributes")
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
