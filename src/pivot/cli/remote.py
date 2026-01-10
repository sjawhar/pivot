from __future__ import annotations

import click

from pivot.cli import completion
from pivot.cli import decorators as cli_decorators


@click.group()
def remote() -> None:
    """Manage remote storage for cache synchronization."""


@remote.command("list")
@cli_decorators.with_error_handling
def remote_list() -> None:
    """List configured remote storage locations."""
    from pivot import remote_config

    remotes = remote_config.list_remotes()
    default = remote_config.get_default_remote()

    if not remotes:
        click.echo("No remotes configured.")
        click.echo("Use 'pivot config set remotes.<name> <url>' to add one.")
        return

    for name, url in remotes.items():
        marker = " (default)" if name == default else ""
        click.echo(f"  {name}: {url}{marker}")


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be pushed")
@click.option("-j", "--jobs", type=int, default=20, help="Parallel upload jobs")
def push(targets: tuple[str, ...], remote_name: str | None, dry_run: bool, jobs: int) -> None:
    """Push cached outputs to remote storage.

    TARGETS can be stage names or file paths. If specified, pushes only
    those outputs. Otherwise, pushes all cached files.
    """
    from pivot import state, transfer

    cache_dir = transfer.get_default_cache_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = list(targets) if targets else None

    if targets_list:
        local_hashes = transfer.get_target_hashes(targets_list, cache_dir, include_deps=False)
    else:
        local_hashes = transfer.get_local_cache_hashes(cache_dir)

    if not local_hashes:
        click.echo("No files to push")
        return

    if dry_run:
        click.echo(f"Would push {len(local_hashes)} file(s) to '{resolved_name}'")
        return

    with state.StateDB(cache_dir) as state_db:

        def progress_callback(completed: int) -> None:
            click.echo(f"  Uploaded {completed} files...", nl=False)
            click.echo("\r", nl=False)

        result = transfer.push(
            cache_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            progress_callback,
        )

    click.echo(
        f"Pushed to '{resolved_name}': {result['transferred']} transferred, {result['skipped']} skipped, {result['failed']} failed"
    )

    if result["errors"]:
        for err in result["errors"][:5]:
            click.echo(f"  Error: {err}", err=True)
        if len(result["errors"]) > 5:
            click.echo(f"  ... and {len(result['errors']) - 5} more errors", err=True)


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be pulled")
@click.option("-j", "--jobs", type=int, default=20, help="Parallel download jobs")
def pull(targets: tuple[str, ...], remote_name: str | None, dry_run: bool, jobs: int) -> None:
    """Pull cached outputs from remote storage.

    TARGETS can be stage names or file paths. If specified, pulls those
    outputs (and dependencies for stages). Otherwise, pulls all available
    files from remote.
    """
    from pivot import state, transfer

    cache_dir = transfer.get_default_cache_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = list(targets) if targets else None

    if dry_run:
        if targets_list:
            needed = transfer.get_target_hashes(targets_list, cache_dir, include_deps=True)
        else:
            import asyncio

            needed = asyncio.run(s3_remote.list_hashes())

        local = transfer.get_local_cache_hashes(cache_dir)
        missing = needed - local
        click.echo(f"Would pull {len(missing)} file(s) from '{resolved_name}'")
        return

    with state.StateDB(cache_dir) as state_db:

        def progress_callback(completed: int) -> None:
            click.echo(f"  Downloaded {completed} files...", nl=False)
            click.echo("\r", nl=False)

        result = transfer.pull(
            cache_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            progress_callback,
        )

    click.echo(
        f"Pulled from '{resolved_name}': {result['transferred']} transferred, {result['skipped']} skipped, {result['failed']} failed"
    )

    if result["errors"]:
        for err in result["errors"][:5]:
            click.echo(f"  Error: {err}", err=True)
        if len(result["errors"]) > 5:
            click.echo(f"  ... and {len(result['errors']) - 5} more errors", err=True)
