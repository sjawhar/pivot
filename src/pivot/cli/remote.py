from __future__ import annotations

import asyncio

import click

from pivot import config
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import state


@click.group()
def remote() -> None:
    """Manage remote storage for cache synchronization."""


@remote.command("list")
@cli_decorators.with_error_handling
def remote_list() -> None:
    """List configured remote storage locations."""
    remotes = remote_config.list_remotes()
    default = remote_config.get_default_remote()

    if not remotes:
        click.echo("No remotes configured.")
        click.echo("Use 'pivot config set remotes.<name> <url>' to add one.")
        return

    for name, url in remotes.items():
        marker = " (default)" if name == default else ""
        click.echo(f"  {name}: {url}{marker}")


def _get_targets_list(targets: tuple[str, ...]) -> list[str] | None:
    """Convert targets tuple to list, or None if empty."""
    return list(targets) if targets else None


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be pushed")
@click.option("-j", "--jobs", type=click.IntRange(min=1), default=None, help="Parallel upload jobs")
@click.pass_context
def push(
    ctx: click.Context,
    targets: tuple[str, ...],
    remote_name: str | None,
    dry_run: bool,
    jobs: int | None,
) -> None:
    """Push cached outputs to remote storage.

    TARGETS can be stage names or file paths. If specified, pushes only
    those outputs. Otherwise, pushes all cached files.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    cache_dir = config.get_cache_dir()
    state_dir = config.get_state_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = _get_targets_list(targets)

    if targets_list:
        local_hashes = transfer.get_target_hashes(targets_list, state_dir, include_deps=False)
    else:
        local_hashes = transfer.get_local_cache_hashes(cache_dir)

    if not local_hashes:
        if not quiet:
            click.echo("No files to push")
        return

    if dry_run:
        if not quiet:
            click.echo(f"Would push {len(local_hashes)} file(s) to '{resolved_name}'")
        return

    with state.StateDB(config.get_state_db_path()) as state_db:
        result = transfer.push(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            None if quiet else cli_helpers.make_progress_callback("Uploaded"),
        )

    if not quiet:
        transferred = result["transferred"]
        skipped = result["skipped"]
        failed = result["failed"]
        click.echo(
            f"Pushed to '{resolved_name}': {transferred} transferred, {skipped} skipped, {failed} failed"
        )

    # Always print errors to stderr and exit non-zero on failures
    cli_helpers.print_transfer_errors(result["errors"])
    if result["failed"] > 0:
        raise SystemExit(1)


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be fetched")
@click.option(
    "-j", "--jobs", type=click.IntRange(min=1), default=None, help="Parallel download jobs"
)
@click.pass_context
def fetch(
    ctx: click.Context,
    targets: tuple[str, ...],
    remote_name: str | None,
    dry_run: bool,
    jobs: int | None,
) -> None:
    """Fetch cached outputs from remote storage to local cache.

    TARGETS can be stage names or file paths. If specified, fetches those
    outputs (and dependencies for stages). Otherwise, fetches all available
    files from remote.

    This command only downloads to the local cache. Use 'pivot pull' to also
    restore files to your workspace, or 'pivot checkout' to restore from cache.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    cache_dir = config.get_cache_dir()
    state_dir = config.get_state_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = _get_targets_list(targets)

    if dry_run:
        if targets_list:
            needed = transfer.get_target_hashes(targets_list, state_dir, include_deps=True)
        else:
            needed = asyncio.run(s3_remote.list_hashes())

        local = transfer.get_local_cache_hashes(cache_dir)
        missing = needed - local
        if not quiet:
            click.echo(f"Would fetch {len(missing)} file(s) from '{resolved_name}'")
        return

    with state.StateDB(config.get_state_db_path()) as state_db:
        result = transfer.pull(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            None if quiet else cli_helpers.make_progress_callback("Downloaded"),
        )

    if not quiet:
        transferred = result["transferred"]
        skipped = result["skipped"]
        failed = result["failed"]
        click.echo(
            f"Fetched from '{resolved_name}': {transferred} transferred, {skipped} skipped, {failed} failed"
        )

    cli_helpers.print_transfer_errors(result["errors"])
    if result["failed"] > 0:
        raise SystemExit(1)


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("-r", "--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be pulled")
@click.option(
    "-j", "--jobs", type=click.IntRange(min=1), default=None, help="Parallel download jobs"
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing workspace files")
@click.option(
    "--only-missing",
    is_flag=True,
    help="Only restore files that don't exist in workspace",
)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.pass_context
def pull(
    ctx: click.Context,
    targets: tuple[str, ...],
    remote_name: str | None,
    dry_run: bool,
    jobs: int | None,
    force: bool,
    only_missing: bool,
    checkout_mode: str | None,
) -> None:
    """Pull cached outputs from remote and restore to workspace.

    Combines 'fetch' (download from remote) and 'checkout' (restore to workspace).
    This matches the behavior of 'git pull' and 'dvc pull'.

    TARGETS can be stage names or file paths. If specified, pulls those
    outputs (and dependencies for stages). Otherwise, pulls all available files.
    """
    if force and only_missing:
        raise click.ClickException("--force and --only-missing are mutually exclusive")

    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    cache_dir = config.get_cache_dir()
    state_dir = config.get_state_dir()
    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)

    targets_list = _get_targets_list(targets)

    # Dry-run: show what would be fetched, don't proceed to checkout
    if dry_run:
        if targets_list:
            needed = transfer.get_target_hashes(targets_list, state_dir, include_deps=True)
        else:
            needed = asyncio.run(s3_remote.list_hashes())

        local = transfer.get_local_cache_hashes(cache_dir)
        missing = needed - local
        if not quiet:
            click.echo(f"Would pull {len(missing)} file(s) from '{resolved_name}'")
        return

    # Step 1: Fetch from remote to cache
    with state.StateDB(config.get_state_db_path()) as state_db:
        fetch_result = transfer.pull(
            cache_dir,
            state_dir,
            s3_remote,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            None if quiet else cli_helpers.make_progress_callback("Downloaded"),
        )

    if not quiet:
        transferred = fetch_result["transferred"]
        skipped = fetch_result["skipped"]
        failed = fetch_result["failed"]
        click.echo(
            f"Fetched from '{resolved_name}': {transferred} transferred, {skipped} skipped, {failed} failed"
        )

    cli_helpers.print_transfer_errors(fetch_result["errors"])

    # If fetch had failures, exit without checkout
    if fetch_result["failed"] > 0:
        raise SystemExit(1)

    # Step 2: Checkout from cache to workspace
    # Import here to avoid circular imports at module level
    from pivot.cli import checkout as checkout_mod

    ctx.invoke(
        checkout_mod.checkout,
        targets=targets,
        checkout_mode=checkout_mode,
        force=force,
        only_missing=only_missing,
    )
