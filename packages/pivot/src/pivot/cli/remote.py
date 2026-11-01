from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING

import click

from pivot import config, display, names, project, types
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import state, track

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.registry import MapTemplate, PipelineLike
    from pivot.remote import protocol
    from pivot.storage import listing

# Gets the listings at the records given (by hash): those read for expansion (a dry run's), the
# hashes downloaded into the cache (a fetch's or pull's), and why each one missing was not had
type _ListingFetch = Callable[
    [dict[str, types.DirHash]], tuple[dict[str, listing.DirListing], frozenset[str], list[str]]
]


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
        shown_url = display.redact_url_credentials(display.escape_control(url))
        click.echo(f"  {display.escape_control(name)}: {shown_url}{marker}")


def _get_targets_list(targets: tuple[str, ...]) -> list[str] | None:
    """Convert targets tuple to list, or None if empty."""
    return list(targets) if targets else None


def _needed_templates(targets: tuple[str, ...], pipeline: PipelineLike) -> set[str]:
    """The mapped stages (``Pipeline.map``) ``targets`` need expanded: each one they name, by
    itself or by an instance (``cli_helpers.template_target``), and each one those are mapped
    over, transitively."""
    templates = pipeline.list_templates()
    pending = [
        name
        for target in targets
        if (name := cli_helpers.template_target(target, templates)) is not None
    ]
    needed = set[str]()
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            mapped_deps = pipeline.get_template(name)["mapped_deps"].values()
            pending.extend(dep for dep in mapped_deps if dep is not None)
    return needed


def _expand_mapped_targets(
    targets: tuple[str, ...],
    pipeline: PipelineLike | None,
    *,
    fetch_listings: _ListingFetch | None = None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    """``targets`` with each mapped stage (``Pipeline.map``) named in them replaced by its
    instances, its producer current (``cli_helpers.expand_mapped_stages``). Without targets
    nothing is expanded: every cached object, or every remote one, is transferred.

    ``fetch_listings`` gets the listings this clone lacks of the directories that the mapped
    stages the targets need are mapped over (``_needed_templates``), and no other: a fetch or
    pull downloads them into the cache, a dry run reads them into a temporary directory. Then
    the mapped stages expand again. Returns the targets and the hashes downloaded, which the
    transfer that follows counts (``transfer.pull``'s ``prefetched``).

    Raises:
        click.ClickException: A target names a mapped stage that is not expanded, saying why,
            or one whose listing could not be fetched, naming the listing.
    """
    if not targets or pipeline is None:
        return targets, frozenset()
    result, _ = cli_helpers.expand_mapped_stages()
    needed = _needed_templates(targets, pipeline)
    records = {
        result["listings"][name]["hash"]: result["listings"][name]
        for name in needed
        if name in result["listings"]
    }
    if not records or fetch_listings is None:
        return cli_helpers.require_expanded_targets(targets, result), frozenset()
    fetched, downloaded, errors = fetch_listings(records)
    result, _ = cli_helpers.expand_mapped_stages(fetched=fetched)
    unfetched = sorted(name for name in needed if name in result["listings"])
    if unfetched:
        cli_helpers.print_transfer_errors(errors)
        raise click.ClickException(
            "; ".join(
                f"'{name}' cannot expand: the listing of "
                + f"'{types.identity_key(pipeline.get_template(name)['source'])}' "
                + f"({result['listings'][name]['hash']}) could not be fetched from the remote"
                for name in unfetched
            )
        )
    return cli_helpers.require_expanded_targets(targets, result), downloaded


def _templates(pipeline: PipelineLike | None) -> dict[str, MapTemplate] | None:
    """The pipeline's mapped stages (``Pipeline.map``), by name, for ``transfer``'s target set."""
    if pipeline is None:
        return None
    return {name: pipeline.get_template(name) for name in pipeline.list_templates()}


def _fetch_listings(
    cache_dir: pathlib.Path,
    state_dir: pathlib.Path,
    remote_storage: protocol.RemoteStorage,
    remote_name: str,
    jobs: int,
    records: dict[str, types.DirHash],
) -> tuple[dict[str, listing.DirListing], frozenset[str], list[str]]:
    """Download the directory listings at ``records`` from the remote into the cache, where
    expansion reads them (``_ListingFetch``)."""
    with state.StateDB(state_dir) as state_db:
        summary, downloaded = transfer.fetch_objects(
            cache_dir, remote_storage, state_db, remote_name, set(records), jobs
        )
    return {}, downloaded, summary["errors"]


def _read_listings(
    remote_storage: protocol.RemoteStorage, jobs: int, records: dict[str, types.DirHash]
) -> tuple[dict[str, listing.DirListing], frozenset[str], list[str]]:
    """Read the directory listings at ``records`` from the remote into a temporary directory,
    for a dry run's expansion: nothing is written to the cache (``_ListingFetch``)."""
    listings, errors = transfer.read_remote_listings(remote_storage, records, jobs)
    return listings, frozenset(), errors


def _normalize_cli_targets(
    targets: tuple[str, ...],
    known_stages: set[str] | None = None,
) -> tuple[str, ...]:
    """Normalize CLI file targets by resolving from cwd and stripping .pvt suffixes.

    Stage names (matched against known_stages) are passed through unchanged.
    File paths are resolved relative to cwd (not project root) and .pvt
    suffixes are stripped to get the data file path.
    """
    normalized = list[str]()
    project_root = project.get_project_root()

    for target in targets:
        if known_stages is not None:
            identity = types.identity_from_key(target)
            stage_name = names.resolve_stage_name(identity.producer, dict.fromkeys(known_stages))
            if stage_name in known_stages:
                normalized.append(
                    types.identity_key(types.ArtifactIdentity(stage_name, identity.key))
                )
                continue

        original = target
        target_path = pathlib.Path(target)
        if target_path.suffix == ".pvt":
            target = str(track.get_data_path(target_path))

        relative = project.relative_within(
            project.normalize_path(target, base=pathlib.Path.cwd()), project_root
        )
        if relative is None:
            raise click.ClickException(f"Target '{original}' resolves outside project root")

        normalized.append(str(project_root / relative))

    return tuple(normalized)


@cli_decorators.pivot_command(allow_all=True)
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

    TARGETS can be stage names or file paths. If specified, pushes those outputs: what
    'pivot pull' of the same targets fetches. Otherwise, pushes all cached files. An object the
    remote lacks and the local cache does not hold fails the push, naming it.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    # A dry run writes nothing: it locates the cache and state directories without creating them.
    cache_dir = config.locate_cache_dir() if dry_run else config.get_cache_dir()
    state_dir = config.locate_state_dir() if dry_run else config.get_state_dir()
    remote_storage, resolved_name = transfer.create_remote_from_name(remote_name)

    # Per-stage state_dir lookup when pipeline is available (e.g., --all mode)
    pipeline = cli_decorators.get_pipeline_from_context()
    targets, _ = _expand_mapped_targets(targets, pipeline)
    all_stages = cli_helpers.get_all_stages() if pipeline is not None else None
    templates = _templates(pipeline)

    stage_names = set(all_stages) if all_stages is not None else None
    normalized = _normalize_cli_targets(targets, known_stages=stage_names)
    targets_list = _get_targets_list(normalized)

    if dry_run:
        with state.open_readonly(state_dir, create=False) as readonly_db:
            result = transfer.push_dry_run(
                cache_dir,
                state_dir,
                remote_storage,
                readonly_db,
                resolved_name,
                targets_list,
                jobs,
                all_stages=all_stages,
                templates=templates,
            )
    else:
        # Remote hash tracking is project-level (not per-stage), so use the
        # project-level StateDB regardless of --all mode.
        with (
            state.StateDB(state_dir) as state_db,
            cli_helpers.TransferProgress("Uploaded", quiet=quiet) as progress,
        ):
            result = transfer.push(
                cache_dir,
                state_dir,
                remote_storage,
                state_db,
                resolved_name,
                targets_list,
                jobs,
                progress.callback,
                all_stages=all_stages,
                templates=templates,
            )

    transferred = result["transferred"]
    skipped = result["skipped"]
    failed = result["failed"]
    if not transferred + skipped + failed:
        if not quiet:
            click.echo("No files to push")
        return

    if not quiet:
        shown = display.escape_control(resolved_name)
        if dry_run:
            click.echo(f"Would push {transferred} file(s) to '{shown}'")
        else:
            click.echo(
                f"Pushed to '{shown}': {transferred} transferred, {skipped} skipped, "
                + f"{failed} failed"
            )

    # Always print errors to stderr and exit non-zero on failures
    cli_helpers.print_transfer_errors(result["errors"])
    if failed > 0:
        raise SystemExit(1)


@cli_decorators.pivot_command(allow_all=True)
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
    outputs: what 'pivot checkout' of the same targets restores. Otherwise,
    fetches all available files from remote.

    This command only downloads to the local cache. Use 'pivot pull' to also
    restore files to your workspace, or 'pivot checkout' to restore from cache.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    # A dry run writes nothing: it locates the cache and state directories without creating them,
    # and reads a listing it needs from the remote into a temporary directory (_read_listings).
    cache_dir = config.locate_cache_dir() if dry_run else config.get_cache_dir()
    state_dir = config.locate_state_dir() if dry_run else config.get_state_dir()
    remote_storage, resolved_name = transfer.create_remote_from_name(remote_name)

    # Per-stage state_dir lookup when pipeline is available (e.g., --all mode). A mapped stage
    # whose producer is current but whose listing this clone lacks expands once fetch fetched it.
    pipeline = cli_decorators.get_pipeline_from_context()
    targets, prefetched = _expand_mapped_targets(
        targets,
        pipeline,
        fetch_listings=functools.partial(_read_listings, remote_storage, jobs)
        if dry_run
        else functools.partial(
            _fetch_listings, cache_dir, state_dir, remote_storage, resolved_name, jobs
        ),
    )
    all_stages = cli_helpers.get_all_stages() if pipeline is not None else None
    templates = _templates(pipeline)

    stage_names = set(all_stages) if all_stages is not None else None
    normalized = _normalize_cli_targets(targets, known_stages=stage_names)
    targets_list = _get_targets_list(normalized)

    if dry_run:
        result = transfer.pull_dry_run(
            cache_dir,
            state_dir,
            remote_storage,
            resolved_name,
            targets_list,
            jobs,
            all_stages=all_stages,
            templates=templates,
        )
        if not quiet:
            shown = display.escape_control(resolved_name)
            click.echo(f"Would fetch {result['transferred']} file(s) from '{shown}'")
        cli_helpers.print_transfer_errors(result["errors"])
        if result["failed"] > 0:
            raise SystemExit(1)
        return

    with (
        state.StateDB(state_dir) as state_db,
        cli_helpers.TransferProgress("Downloaded", quiet=quiet) as progress,
    ):
        result = transfer.pull(
            cache_dir,
            state_dir,
            remote_storage,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            progress.callback,
            all_stages=all_stages,
            templates=templates,
            prefetched=prefetched,
        )

    if not quiet:
        transferred = result["transferred"]
        skipped = result["skipped"]
        failed = result["failed"]
        shown = display.escape_control(resolved_name)
        click.echo(
            f"Fetched from '{shown}': {transferred} transferred, {skipped} skipped, {failed} failed"
        )

    cli_helpers.print_transfer_errors(result["errors"])
    if result["failed"] > 0:
        raise SystemExit(1)


@cli_decorators.pivot_command(allow_all=True)
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
@cli_decorators.checkout_mode_option
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
    outputs: what 'pivot checkout' of the same targets restores. Otherwise,
    pulls all available files.
    """
    if force and only_missing:
        raise click.ClickException("--force and --only-missing are mutually exclusive")

    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    jobs = jobs if jobs is not None else config.get_remote_jobs()

    # A dry run writes nothing: it locates the cache and state directories without creating them,
    # and reads a listing it needs from the remote into a temporary directory (_read_listings).
    cache_dir = config.locate_cache_dir() if dry_run else config.get_cache_dir()
    state_dir = config.locate_state_dir() if dry_run else config.get_state_dir()
    remote_storage, resolved_name = transfer.create_remote_from_name(remote_name)

    # Per-stage state_dir lookup when pipeline is available (e.g., --all mode). A mapped stage
    # whose producer is current but whose listing this clone lacks expands once pull fetched it.
    pipeline = cli_decorators.get_pipeline_from_context()
    targets, prefetched = _expand_mapped_targets(
        targets,
        pipeline,
        fetch_listings=functools.partial(_read_listings, remote_storage, jobs)
        if dry_run
        else functools.partial(
            _fetch_listings, cache_dir, state_dir, remote_storage, resolved_name, jobs
        ),
    )
    all_stages = cli_helpers.get_all_stages() if pipeline is not None else None
    templates = _templates(pipeline)

    stage_names = set(all_stages) if all_stages is not None else None
    normalized = _normalize_cli_targets(targets, known_stages=stage_names)
    targets_list = _get_targets_list(normalized)

    # Fail early if no pipeline and no targets specified (unless --all was used)
    use_all = cli_decorators.get_all_pipelines_from_context()
    if pipeline is None and not targets_list and not use_all:
        raise click.UsageError(
            "No pipeline found. Use --all to pull from all pipelines, or specify file targets."
        )

    # Dry-run: show what would be fetched, don't proceed to checkout
    if dry_run:
        result = transfer.pull_dry_run(
            cache_dir,
            state_dir,
            remote_storage,
            resolved_name,
            targets_list,
            jobs,
            all_stages=all_stages,
            templates=templates,
        )
        if not quiet:
            shown = display.escape_control(resolved_name)
            click.echo(f"Would pull {result['transferred']} file(s) from '{shown}'")
        cli_helpers.print_transfer_errors(result["errors"])
        if result["failed"] > 0:
            raise SystemExit(1)
        return

    # Step 1: Fetch from remote to cache
    with (
        state.StateDB(state_dir) as state_db,
        cli_helpers.TransferProgress("Downloaded", quiet=quiet) as progress,
    ):
        fetch_result = transfer.pull(
            cache_dir,
            state_dir,
            remote_storage,
            state_db,
            resolved_name,
            targets_list,
            jobs,
            progress.callback,
            all_stages=all_stages,
            templates=templates,
            prefetched=prefetched,
        )

    if not quiet:
        transferred = fetch_result["transferred"]
        skipped = fetch_result["skipped"]
        failed = fetch_result["failed"]
        shown = display.escape_control(resolved_name)
        click.echo(
            f"Fetched from '{shown}': {transferred} transferred, {skipped} skipped, {failed} failed"
        )

    cli_helpers.print_transfer_errors(fetch_result["errors"])

    # If fetch had failures, exit without checkout
    if fetch_result["failed"] > 0:
        raise SystemExit(1)

    # Step 2: Checkout from cache to workspace
    # Import here to avoid circular imports at module level
    from pivot.cli import checkout as checkout_mod

    # Default to only_missing=True to avoid "already exists" errors
    if not force and not only_missing:
        only_missing = True

    ctx.invoke(
        checkout_mod.checkout,
        targets=normalized,
        checkout_mode=checkout_mode,
        force=force,
        only_missing=only_missing,
    )
