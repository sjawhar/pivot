from __future__ import annotations

import asyncio
import enum
import logging
import pathlib
from typing import TYPE_CHECKING, Literal

import click

from pivot import config, project, registry
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.storage import cache, lock, track
from pivot.types import OutputHash

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

RestoreResult = Literal["restored", "skipped", "missing"]
MAX_CONCURRENT_RESTORES = 32


class CheckoutBehavior(enum.StrEnum):
    """How to handle existing files during checkout."""

    ERROR = "error"  # Error if file already exists (default)
    SKIP_EXISTING = "skip_existing"  # Skip files that already exist (--only-missing)
    FORCE = "force"  # Overwrite existing files (--force)


def _get_stage_output_info(state_dir: pathlib.Path) -> dict[str, OutputHash]:
    """Get output hash info from lock files for all stages."""
    outputs = dict[str, OutputHash]()

    for stage_name in registry.REGISTRY.list_stages():
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data and "output_hashes" in lock_data:
            for out_path, out_hash in lock_data["output_hashes"].items():
                # Normalize paths for backward compatibility with old lock files
                # (old lock files may have resolved paths, new ones have normalized)
                norm_path = str(project.normalize_path(out_path))
                outputs[norm_path] = out_hash

    return outputs


def _restore_path_sync(
    path: pathlib.Path,
    output_hash: OutputHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
) -> tuple[RestoreResult, str]:
    """Restore a file or directory from cache (sync version).

    Returns:
        Tuple of (result, path_name) for the caller to handle output.

    Raises:
        click.ClickException: For immediate failures (path traversal, unknown target,
            "already exists" without --force). Cache misses return ("missing", name)
            instead of raising.
    """
    if path.exists():
        match behavior:
            case CheckoutBehavior.ERROR:
                raise click.ClickException(
                    f"'{path.name}' already exists. "
                    + "Use --force to overwrite or --only-missing to skip existing files."
                )
            case CheckoutBehavior.SKIP_EXISTING:
                # For directories with manifests, don't skip - files inside may be missing.
                # Let restore_from_cache() handle it (does full directory restoration).
                # DirHash has "manifest" key, FileHash does not.
                is_directory = output_hash is not None and "manifest" in output_hash
                if not is_directory:
                    return ("skipped", path.name)
            case CheckoutBehavior.FORCE:
                cache.remove_output(path)
            case _:  # pyright: ignore[reportUnnecessaryComparison] - defensive for future enum values
                raise ValueError(f"Unhandled checkout behavior: {behavior}")  # pyright: ignore[reportUnreachable]

    success = cache.restore_from_cache(path, output_hash, cache_dir, checkout_modes=checkout_modes)
    if not success:
        return ("missing", path.name)

    return ("restored", path.name)


async def _checkout_files_async(
    files: Mapping[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
) -> tuple[list[str], int, int]:
    """Restore files in parallel.

    Returns:
        Tuple of (failures, restored_count, skipped_count) where failures is a list
        of file names that were missing from cache.

    Raises:
        click.ClickException: For immediate failures (aggregated if multiple).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESTORES)
    failures = list[str]()
    restored = 0
    skipped = 0
    immediate_errors = list[click.ClickException]()

    async def restore_one(abs_path_str: str, output_hash: OutputHash) -> None:
        nonlocal restored, skipped
        path = pathlib.Path(abs_path_str)
        try:
            async with semaphore:
                result, name = await asyncio.to_thread(
                    _restore_path_sync, path, output_hash, cache_dir, checkout_modes, behavior
                )
            match result:
                case "missing":
                    failures.append(name)
                case "restored":
                    restored += 1
                case "skipped":
                    skipped += 1
        except click.ClickException as e:
            immediate_errors.append(e)

    try:
        async with asyncio.TaskGroup() as tg:
            for abs_path_str, output_hash in files.items():
                if output_hash is not None:
                    tg.create_task(restore_one(abs_path_str, output_hash))
                else:
                    logger.debug(f"Skipping output with no cached hash: {abs_path_str}")
    except* Exception as eg:
        # Convert unexpected exceptions to friendly error message
        errors = [str(e) for e in eg.exceptions]
        raise click.ClickException("\n".join(errors)) from None

    # Re-raise immediate failures (all of them, aggregated)
    if immediate_errors:
        msgs = [str(e.message) for e in immediate_errors]
        raise click.ClickException("\n".join(msgs))

    return (failures, restored, skipped)


def _dedupe_targets(targets: tuple[str, ...]) -> list[str]:
    """Deduplicate targets by normalized absolute path.

    Handles both data paths (data.txt) and .pvt paths (data.txt.pvt) resolving
    to the same file.
    """
    seen = set[str]()
    unique = list[str]()
    for target in targets:
        # Convert .pvt to data path
        target_path = pathlib.Path(target)
        if target_path.suffix == ".pvt":
            target = str(track.get_data_path(target_path))

        abs_path = str(project.normalize_path(target))
        if abs_path not in seen:
            seen.add(abs_path)
            unique.append(target)
    return unique


def _validate_and_build_files(
    targets: list[str],
    tracked_files: dict[str, track.PvtData],
    stage_outputs: dict[str, OutputHash],
) -> dict[str, OutputHash]:
    """Validate targets and build files dict for checkout.

    Targets should already have .pvt suffixes converted by _dedupe_targets().

    Raises:
        click.ClickException: For path traversal or unknown targets.
    """
    files = dict[str, OutputHash]()

    for target in targets:
        # Validate path doesn't escape project
        if track.has_path_traversal(target):
            raise click.ClickException(f"Path traversal not allowed: {target}")

        # Use normalized path (preserve symlinks) to match keys in tracked_files/stage_outputs
        abs_path = project.normalize_path(target)
        abs_path_str = str(abs_path)

        # Check if it's a tracked file
        if abs_path_str in tracked_files:
            pvt_data = tracked_files[abs_path_str]
            files[abs_path_str] = track.pvt_to_hash_info(pvt_data)
            continue

        # Check if it's a stage output
        if abs_path_str in stage_outputs:
            output_hash = stage_outputs[abs_path_str]
            if output_hash is None:
                raise click.ClickException(
                    f"'{target}' has no cached version. "
                    + "Run the stage first, or use 'pivot pull' to fetch from remote."
                )
            files[abs_path_str] = output_hash
            continue

        # Unknown target
        raise click.ClickException(
            f"'{target}' is not a tracked file or stage output. "
            + "Use 'pivot list' to see stages or 'pivot track' to track files."
        )

    return files


async def _checkout_main_async(
    targets: tuple[str, ...],
    tracked_files: dict[str, track.PvtData],
    stage_outputs: dict[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
) -> tuple[list[str], int, int]:
    """Main async checkout logic.

    Returns:
        Tuple of (failures, restored_count, skipped_count).
    """
    if targets:
        unique_targets = _dedupe_targets(targets)
        files = _validate_and_build_files(unique_targets, tracked_files, stage_outputs)
        return await _checkout_files_async(files, cache_dir, checkout_modes, behavior)
    else:
        # Checkout all tracked files and stage outputs
        tracked_as_hashes: dict[str, OutputHash] = {
            path: track.pvt_to_hash_info(pvt) for path, pvt in tracked_files.items()
        }
        # Run both in parallel
        t1 = asyncio.create_task(
            _checkout_files_async(tracked_as_hashes, cache_dir, checkout_modes, behavior)
        )
        t2 = asyncio.create_task(
            _checkout_files_async(stage_outputs, cache_dir, checkout_modes, behavior)
        )
        (f1, r1, s1), (f2, r2, s2) = await asyncio.gather(t1, t2)
        return (f1 + f2, r1 + r2, s1 + s2)


def _print_summary(failures: list[str], restored: int, skipped: int, quiet: bool) -> bool:
    """Print checkout summary.

    Returns:
        True if all files were restored successfully, False if any were missing.
    """
    if not quiet and (restored or skipped):
        if restored:
            click.echo(f"Restored {restored} file(s)")
        if skipped:
            click.echo(f"Skipped {skipped} file(s) (already exist)")

    if failures:
        if restored or skipped:
            click.echo("")  # Add blank line after success summary
        click.echo(f"Missing {len(failures)} file(s):")
        for name in failures[:15]:
            click.echo(f"  {name}")
        if len(failures) > 15:
            click.echo(f"  ... and {len(failures) - 15} more")
        click.echo("")
        click.echo("Run 'pivot pull' to fetch from remote storage.")
        return False

    return True


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.option(
    "--only-missing",
    is_flag=True,
    help="Only restore files that don't exist on disk (safe for local modifications)",
)
@click.pass_context
def checkout(
    ctx: click.Context,
    targets: tuple[str, ...],
    checkout_mode: str | None,
    force: bool,
    only_missing: bool,
) -> None:
    """Restore tracked files and stage outputs from cache.

    If no targets specified, restores all tracked files and stage outputs.
    Use --only-missing to skip files that already exist (safe for local modifications).
    """
    if force and only_missing:
        raise click.ClickException("--force and --only-missing are mutually exclusive")

    # Convert CLI flags to behavior enum
    if force:
        behavior = CheckoutBehavior.FORCE
    elif only_missing:
        behavior = CheckoutBehavior.SKIP_EXISTING
    else:
        behavior = CheckoutBehavior.ERROR

    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    project_root = project.get_project_root()
    cache_dir = config.get_cache_dir() / "files"
    state_dir = config.get_state_dir()

    # Determine checkout modes - CLI flag overrides config (single mode, no fallback)
    checkout_modes = (
        [cache.CheckoutMode(checkout_mode)] if checkout_mode else config.get_checkout_mode_order()
    )

    # Discover tracked files
    tracked_files = track.discover_pvt_files(project_root)

    # Get stage output info from lock files
    stage_outputs = _get_stage_output_info(state_dir)

    # Run async checkout
    failures, restored, skipped = asyncio.run(
        _checkout_main_async(
            targets, tracked_files, stage_outputs, cache_dir, checkout_modes, behavior
        )
    )

    success = _print_summary(failures, restored, skipped, quiet)
    if not success:
        ctx.exit(1)
