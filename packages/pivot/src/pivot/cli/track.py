from __future__ import annotations

import os
import pathlib
from typing import TypedDict

import click

from pivot import config, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.storage import cache, state
from pivot.storage import track as track_mod


class TrackResult(TypedDict):
    """Result of tracking a single path."""

    warning: str | None
    tracked_path: str


def _paths_overlap(path_a: pathlib.Path, path_b: pathlib.Path) -> bool:
    """Check if two paths overlap (same location or parent/child relationship)."""
    # Exact match - use samefile if both exist (handles hardlinks, case-insensitivity)
    if path_a.exists() and path_b.exists():
        try:
            if path_a.samefile(path_b):
                return True
        except OSError:
            pass

    if path_a == path_b:
        return True

    return path_a.is_relative_to(path_b) or path_b.is_relative_to(path_a)


def _get_all_stage_outputs() -> dict[str, pathlib.Path]:
    """Get stage outputs with resolved paths for overlap detection.

    Returns:
        Dict mapping normalized path -> resolved path

    Raises:
        click.ClickException: If any stage output has unresolvable issues
            (permission errors, circular symlinks, etc.)

    Note:
        Resolves all paths upfront and fails fast if any are problematic.
        This provides clear error messages showing ALL issues at once.
    """
    store = cli_helpers.get_workspace_store()
    if store is None:
        return {}

    # Workspace paths of every declared output (the same resolution checkout/verify use)
    outputs_normalized = set[str]()
    for stage_name in cli_helpers.list_stages():
        info = cli_helpers.get_stage(stage_name)
        outputs_normalized.update(str(store.resolve_display_path(out)) for out in info["outs"])

    # Pre-resolve all stage outputs with explicit error handling
    outputs_resolved = dict[str, pathlib.Path]()
    resolution_errors = list[str]()

    for norm_path in outputs_normalized:
        try:
            resolved = project.resolve_path_for_comparison(norm_path, "stage output")
            outputs_resolved[norm_path] = resolved
        except (PermissionError, RuntimeError, OSError) as e:
            # Collect errors to show all problems at once
            resolution_errors.append(f"  - {norm_path}: {e}")

    if resolution_errors:
        raise click.ClickException(
            "Cannot resolve stage outputs (fix these before tracking):\n"
            + "\n".join(resolution_errors)
        )

    return outputs_resolved


def _track_single_path(
    path_str: str,
    cache_dir: pathlib.Path,
    stage_outputs_resolved: dict[str, pathlib.Path],
    existing_tracked: dict[str, track_mod.PvtData],
    force: bool,
    *,
    warn_left_out_symlinks: bool,
    read: list[tuple[pathlib.Path, os.stat_result, str]],
) -> TrackResult:
    """Track a single file or directory.

    ``warn_left_out_symlinks`` logs the warning naming the symlinks a tracked directory leaves
    out; the caller turns it off under ``--quiet``, like this command's other warning.
    ``read`` receives the hash of each file read, for the caller to record in the StateDB.

    Returns:
        TrackResult with optional warning and the tracked path for the caller to handle output.
    """
    # Normalize path from cwd (preserving symlinks) for consistency with registry/pvt, and
    # validate it is within project root; a path typed through a symlink to the project is
    # placed under the project root
    project_root = project.get_project_root()
    relative = project.relative_within(
        project.normalize_path(path_str, base=pathlib.Path.cwd()), project_root
    )
    if relative is None:
        raise click.ClickException(f"Path '{path_str}' resolves outside project root")
    path = project_root / relative

    abs_path_str = str(path)

    # Check for broken symlinks (exist as symlinks but target doesn't exist)
    if path.is_symlink() and not path.exists():
        raise click.ClickException(f"Path '{path_str}' is a broken symlink (target does not exist)")

    # Check file exists
    if not path.exists():
        raise click.ClickException(f"Path not found: {path_str}")

    # Check if tracking file inside symlinked directory. The .pvt is written beside the data
    # (get_pvt_path), so a directory that really lies outside the project takes the .pvt with
    # it, where no command finds it again: refuse before anything is cached or written.
    warning: str | None = None
    if project.contains_symlink_in_path(path, project_root):
        outside = project.real_location_outside(path.parent, project_root.resolve())
        if outside is not None:
            raise click.ClickException(
                f"Cannot track '{path_str}': its .pvt would be written outside the project, "
                + f"in '{outside}', where pivot does not find it. Track the file where it "
                + "really is, inside the project."
            )
        warning = (
            f"Warning: '{path_str}' is inside a symlinked directory. "
            + "Tracked path may not be portable across environments."
        )

    # Check for overlap with stage outputs (resolve paths to detect symlink aliasing); an
    # OSError here reaches the command's error handler, which names the path.
    user_resolved = path.resolve()

    for out_norm, out_resolved in stage_outputs_resolved.items():
        if _paths_overlap(user_resolved, out_resolved):
            # Provide helpful error showing both normalized and resolved if different
            if str(user_resolved) != abs_path_str or str(out_resolved) != out_norm:
                # Symlink aliasing detected
                raise click.ClickException(
                    f"Cannot track '{path_str}' (resolves to '{user_resolved}'): "
                    + f"overlaps with stage output '{out_norm}' (resolves to '{out_resolved}')"
                )
            else:
                # Direct overlap
                raise click.ClickException(
                    f"Cannot track '{path_str}': overlaps with stage output '{out_norm}'"
                )

    # Check for duplicate tracking
    pvt_path = track_mod.get_pvt_path(path)
    if abs_path_str in existing_tracked and not force:
        raise click.ClickException(f"'{path_str}' is already tracked. Use --force to update.")

    # Hash and cache
    if path.is_dir():
        dir_hash = cache.hash_directory(
            path, cache_dir=cache_dir, warn_left_out_symlinks=warn_left_out_symlinks, pending=read
        )
        manifest = dir_hash["manifest"]
        total_size = sum(e["size"] for e in manifest)
        num_files = len(manifest)

        for entry in manifest:
            file_path = path / entry["relpath"]
            file_cache_path = cache.get_cache_path(cache_dir, entry["hash"])
            cache.copy_to_cache(file_path, file_cache_path, executable=entry["isexec"])

        pvt_data: track_mod.PvtData = {
            "path": path.name,
            "hash": dir_hash["hash"],
            "size": total_size,
            "num_files": num_files,
            "manifest": manifest,
        }
        if "empty_dirs" in dir_hash:
            pvt_data["empty_dirs"] = dir_hash["empty_dirs"]
    else:
        file_hash, file_stat = cache.hash_file(path, pending=read)
        file_size = file_stat.st_size
        file_cache_path = cache.get_cache_path(cache_dir, file_hash)
        cache.copy_to_cache(path, file_cache_path)

        pvt_data = {
            "path": path.name,
            "hash": file_hash,
            "size": file_size,
        }

    # Write .pvt file
    track_mod.write_pvt_file(pvt_path, pvt_data)

    # Update existing_tracked for subsequent paths
    existing_tracked[abs_path_str] = pvt_data

    return TrackResult(warning=warning, tracked_path=path_str)


@cli_decorators.pivot_command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing .pvt files")
@click.pass_context
def track(ctx: click.Context, paths: tuple[str, ...], force: bool) -> None:
    """Track files/directories for caching.

    Creates .pvt manifest files and caches content for reproducibility.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    project_root = project.get_project_root()
    cache_dir = config.get_cache_dir() / "files"

    # Get all stage outputs for overlap detection
    pipeline = cli_decorators.get_pipeline_from_context()
    stage_outputs = {} if pipeline is None else _get_all_stage_outputs()

    # Discover existing .pvt files
    existing_tracked = track_mod.discover_pvt_files(project_root)

    read = list[tuple[pathlib.Path, os.stat_result, str]]()
    for path_str in paths:
        result = _track_single_path(
            path_str,
            cache_dir,
            stage_outputs,
            existing_tracked,
            force,
            warn_left_out_symlinks=not quiet,
            read=read,
        )
        if not quiet:
            if result["warning"]:
                click.echo(result["warning"], err=True)
            click.echo(f"Tracked: {result['tracked_path']}")
    # The hashes read to track, so the next status reads none of these files.
    if read:
        with state.StateDB(config.get_state_dir()) as state_db:
            state_db.save_many(read)
