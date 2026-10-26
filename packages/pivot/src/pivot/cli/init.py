from __future__ import annotations

import pathlib

import click

from pivot import exceptions, ignore
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers

# Entries are anchored to .pivot/ so a pipeline named e.g. "cache" or "locks" keeps its
# committed lock files under .pivot/stages/<pipeline>/.
_GITIGNORE_CONTENT = """\
# Cache directory (stage outputs, file hashes)
/cache/

# State database (file hashes, generation counters)
/state.lmdb/

# Artifact and restore locks (flock files for concurrent execution)
/locks/

# Agent RPC socket (left behind by TUI and --serve sessions)
/agent.sock

# Config lock (held while editing config.yaml)
/config.yaml.lock

# Generated plots gallery (pivot plots show)
/plots/

# Key for secret params' digests in lock files; never commit it
/secret.key

# Keep lock and listing files committed under a root pattern such as `lib/` or `build/`
!/stages/**
"""

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
    help="Overwrite existing .pivot/.gitignore",
)
@click.pass_context
def init(ctx: click.Context, force: bool) -> None:
    """Initialize a new Pivot project."""
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    pivot_dir = pathlib.Path.cwd() / ".pivot"

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
    (pivot_dir / "stages").mkdir(exist_ok=True)
    gitignore_path = pivot_dir / ".gitignore"

    # Warn if overwriting existing .gitignore with custom content
    if force and gitignore_path.exists():
        existing_content = gitignore_path.read_text()
        if existing_content != _GITIGNORE_CONTENT and not quiet:
            click.echo("Warning: Overwriting existing .pivot/.gitignore", err=True)

    gitignore_path.write_text(_GITIGNORE_CONTENT)

    # Create .pivotignore with default patterns if it doesn't exist
    pivotignore_path = pathlib.Path.cwd() / ".pivotignore"
    created_pivotignore = False
    if not pivotignore_path.exists():
        pivotignore_content = "\n".join(ignore.get_default_patterns()) + "\n"
        pivotignore_path.write_text(pivotignore_content)
        created_pivotignore = True

    created_gitattributes = _ensure_gitattributes(pathlib.Path.cwd() / ".gitattributes")

    if not quiet:
        click.echo("Initialized Pivot project.")
        click.echo()
        click.echo("Created:")
        click.echo("  .pivot/")
        click.echo("  .pivot/stages/")
        click.echo("  .pivot/.gitignore")
        if created_pivotignore:
            click.echo("  .pivotignore")
        if created_gitattributes:
            click.echo("  .gitattributes")
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Create pipeline.py to define your pipeline stages")
        click.echo("  2. Run 'pivot repro' to execute the pipeline")
        click.echo("  3. See 'pivot --help' for more commands")
