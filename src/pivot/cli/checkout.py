from __future__ import annotations

import pathlib

import click

from pivot import cache, config, lock, project, pvt, registry
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.types import OutputHash


def _get_stage_output_info(project_root: pathlib.Path) -> dict[str, OutputHash]:
    """Get output hash info from lock files for all stages."""
    outputs = dict[str, OutputHash]()
    cache_dir = project_root / ".pivot" / "cache"

    for stage_name in registry.REGISTRY.list_stages():
        stage_lock = lock.StageLock(stage_name, cache_dir)
        lock_data = stage_lock.read()
        if lock_data and "output_hashes" in lock_data:
            for out_path, out_hash in lock_data["output_hashes"].items():
                # Normalize paths for backward compatibility with old lock files
                # (old lock files may have resolved paths, new ones have normalized)
                norm_path = str(project.normalize_path(out_path))
                outputs[norm_path] = out_hash

    return outputs


def _checkout_all_tracked(
    tracked_files: dict[str, pvt.PvtData],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore all tracked files."""
    for abs_path_str, pvt_data in tracked_files.items():
        path = pathlib.Path(abs_path_str)
        output_hash = _pvt_to_output_hash(pvt_data)
        _restore_path(path, output_hash, cache_dir, checkout_modes, force)


def _checkout_all_outputs(
    stage_outputs: dict[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore all stage outputs."""
    for abs_path_str, output_hash in stage_outputs.items():
        if output_hash is None:
            continue
        path = pathlib.Path(abs_path_str)
        _restore_path(path, output_hash, cache_dir, checkout_modes, force)


def _checkout_target(
    target: str,
    tracked_files: dict[str, pvt.PvtData],
    stage_outputs: dict[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore a specific target."""
    # Validate path doesn't escape project
    if pvt.has_path_traversal(target):
        raise click.ClickException(f"Path traversal not allowed: {target}")

    # Use normalized path (preserve symlinks) to match keys in tracked_files/stage_outputs
    abs_path = project.normalize_path(target)
    abs_path_str = str(abs_path)

    # Check if it's a tracked file
    if abs_path_str in tracked_files:
        pvt_data = tracked_files[abs_path_str]
        output_hash = _pvt_to_output_hash(pvt_data)
        _restore_path(abs_path, output_hash, cache_dir, checkout_modes, force)
        return

    # Check if it's a stage output
    if abs_path_str in stage_outputs:
        output_hash = stage_outputs[abs_path_str]
        if output_hash is None:
            raise click.ClickException(
                f"'{target}' has no cached version. "
                + "Run the stage first, or use 'pivot pull' to fetch from remote."
            )
        _restore_path(abs_path, output_hash, cache_dir, checkout_modes, force)
        return

    # Unknown target
    raise click.ClickException(
        f"'{target}' is not a tracked file or stage output. "
        + "Use 'pivot list' to see stages or 'pivot track' to track files."
    )


def _pvt_to_output_hash(pvt_data: pvt.PvtData) -> OutputHash:
    """Convert PvtData to OutputHash format."""
    manifest = pvt_data.get("manifest")
    if manifest is not None:
        return {"hash": pvt_data["hash"], "manifest": manifest}
    return {"hash": pvt_data["hash"]}


def _restore_path(
    path: pathlib.Path,
    output_hash: OutputHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore a file or directory from cache."""
    if path.exists() and not force:
        click.echo(f"Skipped: {path.name} (already exists)")
        return

    if force and path.exists():
        cache.remove_output(path)

    success = cache.restore_from_cache(path, output_hash, cache_dir, checkout_modes=checkout_modes)
    if not success:
        raise click.ClickException(
            f"Failed to restore '{path.name}': not found in cache. "
            + "Try 'pivot pull' to fetch from remote storage."
        )

    click.echo(f"Restored: {path.name}")


@cli_decorators.pivot_command()
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def checkout(targets: tuple[str, ...], checkout_mode: str | None, force: bool) -> None:
    """Restore tracked files and stage outputs from cache.

    If no targets specified, restores all tracked files and stage outputs.
    """
    project_root = project.get_project_root()
    cache_dir = project_root / ".pivot" / "cache" / "files"

    # Determine checkout modes - CLI flag overrides config (single mode, no fallback)
    mode_strings = [checkout_mode] if checkout_mode else config.get_checkout_mode_order()
    checkout_modes = [cache.CheckoutMode(m) for m in mode_strings]

    # Discover tracked files
    tracked_files = pvt.discover_pvt_files(project_root)

    # Get stage output info from lock files
    stage_outputs = _get_stage_output_info(project_root)

    if not targets:
        # Restore everything
        _checkout_all_tracked(tracked_files, cache_dir, checkout_modes, force)
        _checkout_all_outputs(stage_outputs, cache_dir, checkout_modes, force)
    else:
        # Restore specific targets
        for target in targets:
            _checkout_target(target, tracked_files, stage_outputs, cache_dir, checkout_modes, force)
