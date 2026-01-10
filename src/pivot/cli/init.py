from __future__ import annotations

import pathlib

import click

from pivot import exceptions
from pivot.cli import decorators as cli_decorators

_GITIGNORE_CONTENT = """\
# Cache directory (stage outputs, file hashes)
cache/

# State database (file hashes, generation counters)
state.db
state.lmdb/

# Lock files
config.yaml.lock
"""


@cli_decorators.pivot_command()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing .pivot/.gitignore",
)
def init(force: bool) -> None:
    """Initialize a new Pivot project."""
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
    (pivot_dir / ".gitignore").write_text(_GITIGNORE_CONTENT)

    click.echo("Initialized Pivot project.")
    click.echo()
    click.echo("Created:")
    click.echo("  .pivot/")
    click.echo("  .pivot/.gitignore")
    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Create pivot.yaml to define your pipeline stages")
    click.echo("  2. Run 'pivot run' to execute the pipeline")
    click.echo("  3. See 'pivot --help' for more commands")
