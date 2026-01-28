from __future__ import annotations

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

RestoreResult = Literal["restored", "skipped"]


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


def _restore_path(
    path: pathlib.Path,
    output_hash: OutputHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
) -> tuple[RestoreResult, str]:
    """Restore a file or directory from cache.

    Returns:
        Tuple of (result, path_name) for the caller to handle output.
    """
    if path.exists():
        match behavior:
            case CheckoutBehavior.ERROR:
                raise click.ClickException(
                    f"'{path.name}' already exists. "
                    + "Use --force to overwrite or --only-missing to skip existing files."
                )
            case CheckoutBehavior.SKIP_EXISTING:
                return ("skipped", path.name)
            case CheckoutBehavior.FORCE:
                cache.remove_output(path)
            case _:  # pyright: ignore[reportUnnecessaryComparison] - defensive for future enum values
                raise ValueError(f"Unhandled checkout behavior: {behavior}")  # pyright: ignore[reportUnreachable]

    success = cache.restore_from_cache(path, output_hash, cache_dir, checkout_modes=checkout_modes)
    if not success:
        raise click.ClickException(
            f"Failed to restore '{path.name}': not found in cache. "
            + "Try 'pivot pull' to fetch from remote storage."
        )

    return ("restored", path.name)


def _print_restore_result(result: RestoreResult, name: str) -> None:
    """Print restore result to user."""
    if result == "restored":
        click.echo(f"Restored: {name}")
    else:
        click.echo(f"Skipped: {name} (already exists)")


def _checkout_files(
    files: Mapping[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    quiet: bool,
) -> None:
    """Restore files from cache."""
    for abs_path_str, output_hash in files.items():
        if output_hash is None:
            logger.debug(f"Skipping output with no cached hash: {abs_path_str}")
            continue
        path = pathlib.Path(abs_path_str)
        result, name = _restore_path(path, output_hash, cache_dir, checkout_modes, behavior)
        if not quiet:
            _print_restore_result(result, name)


def _checkout_target(
    target: str,
    tracked_files: dict[str, track.PvtData],
    stage_outputs: dict[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    quiet: bool,
) -> None:
    """Restore a specific target (tracked file or stage output) from cache."""
    # Validate path doesn't escape project
    if track.has_path_traversal(target):
        raise click.ClickException(f"Path traversal not allowed: {target}")

    # Convert .pvt file paths to their corresponding data paths
    target_path = pathlib.Path(target)
    if target_path.suffix == ".pvt":
        target = str(track.get_data_path(target_path))

    # Use normalized path (preserve symlinks) to match keys in tracked_files/stage_outputs
    abs_path = project.normalize_path(target)
    abs_path_str = str(abs_path)

    # Check if it's a tracked file
    if abs_path_str in tracked_files:
        pvt_data = tracked_files[abs_path_str]
        output_hash = track.pvt_to_hash_info(pvt_data)
        result, name = _restore_path(abs_path, output_hash, cache_dir, checkout_modes, behavior)
        if not quiet:
            _print_restore_result(result, name)
        return

    # Check if it's a stage output
    if abs_path_str in stage_outputs:
        output_hash = stage_outputs[abs_path_str]
        if output_hash is None:
            raise click.ClickException(
                f"'{target}' has no cached version. "
                + "Run the stage first, or use 'pivot pull' to fetch from remote."
            )
        result, name = _restore_path(abs_path, output_hash, cache_dir, checkout_modes, behavior)
        if not quiet:
            _print_restore_result(result, name)
        return

    # Unknown target
    raise click.ClickException(
        f"'{target}' is not a tracked file or stage output. "
        + "Use 'pivot list' to see stages or 'pivot track' to track files."
    )


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

    if not targets:
        # Convert tracked files to OutputHash format
        tracked_as_hashes = {
            path: track.pvt_to_hash_info(pvt) for path, pvt in tracked_files.items()
        }
        _checkout_files(tracked_as_hashes, cache_dir, checkout_modes, behavior, quiet)
        _checkout_files(stage_outputs, cache_dir, checkout_modes, behavior, quiet)
    else:
        for target in targets:
            _checkout_target(
                target,
                tracked_files,
                stage_outputs,
                cache_dir,
                checkout_modes,
                behavior,
                quiet,
            )
