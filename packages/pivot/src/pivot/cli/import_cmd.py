from __future__ import annotations

import asyncio

import click

from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command(auto_discover=False)
@click.argument("repo_url")
@click.argument("path")
@click.option("--rev", default="main", help="Git ref to import from (branch, tag, commit)")
@click.option("--out", default=None, help="Local output path (default: same as source path)")
@click.option("--force", is_flag=True, help="Overwrite existing files")
@click.option("--no-download", is_flag=True, help="Create .pvt metadata without downloading")
def import_cmd(
    repo_url: str, path: str, rev: str, out: str | None, force: bool, no_download: bool
) -> None:
    """Import an artifact from a remote Pivot repo."""
    from pivot import import_artifact

    result = asyncio.run(
        import_artifact.import_artifact(
            repo_url, path, rev=rev, out=out, force=force, no_download=no_download
        )
    )
    if result["downloaded"]:
        click.echo(f"Imported {result['data_path']}")
    else:
        click.echo(f"Created {result['pvt_path']} (metadata only)")
