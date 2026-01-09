from __future__ import annotations

import pathlib

import click

from pivot import cache, project, pvt, registry


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
    outputs_normalized = set[str]()
    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        outputs_normalized.update(info.get("outs_paths", []))

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
    existing_tracked: dict[str, pvt.PvtData],
    force: bool,
) -> None:
    """Track a single file or directory.

    Args:
        path_str: Path to track (user input)
        cache_dir: Cache directory path
        stage_outputs_resolved: Dict mapping normalized output paths to resolved paths
        existing_tracked: Dict of already tracked files
        force: Whether to overwrite existing .pvt files
    """
    # Validate path doesn't escape project
    if pvt.has_path_traversal(path_str):
        raise click.ClickException(f"Path traversal not allowed: {path_str}")

    # Normalize path (preserve symlinks) for consistency with registry/pvt
    path = project.normalize_path(path_str)
    abs_path_str = str(path)

    # Check for broken symlinks (exist as symlinks but target doesn't exist)
    if path.is_symlink() and not path.exists():
        raise click.ClickException(f"Path '{path_str}' is a broken symlink (target does not exist)")

    # Check file exists
    if not path.exists():
        raise click.ClickException(f"Path not found: {path_str}")

    # Warn if tracking file inside symlinked directory
    project_root = project.get_project_root()
    if project.contains_symlink_in_path(path, project_root):
        click.echo(
            f"Warning: '{path_str}' is inside a symlinked directory. "
            + "Tracked path may not be portable across environments.",
            err=True,
        )

    # Check for overlap with stage outputs (resolve paths to detect symlink aliasing)
    try:
        user_resolved = project.resolve_path_for_comparison(path_str, "user path")
    except (PermissionError, RuntimeError, OSError) as e:
        raise click.ClickException(repr(e)) from e

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
    pvt_path = pvt.get_pvt_path(path)
    if abs_path_str in existing_tracked and not force:
        raise click.ClickException(f"'{path_str}' is already tracked. Use --force to update.")

    # Hash and cache
    if path.is_dir():
        tree_hash, manifest = cache.hash_directory(path)
        total_size = sum(e["size"] for e in manifest)
        num_files = len(manifest)

        # Save each file to cache
        for entry in manifest:
            file_path = path / entry["relpath"]
            file_cache_path = cache.get_cache_path(cache_dir, entry["hash"])
            if not file_cache_path.exists():
                cache.copy_to_cache(file_path, file_cache_path)

        pvt_data: pvt.PvtData = {
            "path": path.name,
            "hash": tree_hash,
            "size": total_size,
            "num_files": num_files,
            "manifest": manifest,
        }
    else:
        file_hash = cache.hash_file(path)
        file_size = path.stat().st_size
        file_cache_path = cache.get_cache_path(cache_dir, file_hash)
        if not file_cache_path.exists():
            cache.copy_to_cache(path, file_cache_path)

        pvt_data = {
            "path": path.name,
            "hash": file_hash,
            "size": file_size,
        }

    # Write .pvt file
    pvt.write_pvt_file(pvt_path, pvt_data)

    # Update existing_tracked for subsequent paths
    existing_tracked[abs_path_str] = pvt_data

    click.echo(f"Tracked: {path_str}")


@click.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing .pvt files")
def track(paths: tuple[str, ...], force: bool) -> None:
    """Track files/directories for caching.

    Creates .pvt manifest files and caches content for reproducibility.
    """
    try:
        project_root = project.get_project_root()
        cache_dir = project_root / ".pivot" / "cache" / "files"

        # Get all stage outputs for overlap detection
        stage_outputs = _get_all_stage_outputs()

        # Discover existing .pvt files
        existing_tracked = pvt.discover_pvt_files(project_root)

        for path_str in paths:
            _track_single_path(path_str, cache_dir, stage_outputs, existing_tracked, force)
    except Exception as e:
        raise click.ClickException(repr(e)) from e
