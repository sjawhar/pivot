from __future__ import annotations

import pathlib
from typing import TypedDict

import click

from pivot import config, discovery, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.storage import cache, ownership
from pivot.storage import track as track_mod


class TrackResult(TypedDict):
    """Result of tracking a single path."""

    warning: str | None
    tracked_path: str


def _track_single_path(
    path_str: str,
    cache_dir: pathlib.Path,
    stage_outputs: dict[str, ownership.DeclaredOutput],
    existing_tracked: dict[str, track_mod.PvtData],
    force: bool,
    *,
    warn_left_out_symlinks: bool,
) -> TrackResult:
    """Track a single file or directory.

    ``warn_left_out_symlinks`` logs the warning naming the symlinks a tracked directory leaves
    out; the caller turns it off under ``--quiet``, like this command's other warning.

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
    warnings = list[str]()
    if project.contains_symlink_in_path(path, project_root):
        outside = project.real_location_outside(path.parent, project_root.resolve())
        if outside is not None:
            raise click.ClickException(
                f"Cannot track '{path_str}': its .pvt would be written outside the project, "
                + f"in '{outside}', where pivot does not find it. Track the file where it "
                + "really is, inside the project."
            )
        warnings.append(
            f"Warning: '{path_str}' is inside a symlinked directory. "
            + "Tracked path may not be portable across environments."
        )

    # Check for duplicate tracking.
    pvt_path = track_mod.get_pvt_path(path)
    already_tracked = abs_path_str in existing_tracked
    if already_tracked and not force:
        raise click.ClickException(f"'{path_str}' is already tracked. Use --force to update.")

    # Check every pipeline output. OSError reaches the command error handler, which names path.
    overlap = ownership.find_declared_overlap(path, stage_outputs, project_root)
    if overlap is not None:
        if not already_tracked:
            raise click.ClickException(
                f"Cannot track '{path_str}': {overlap}, and that stage replaces its output when "
                + "it runs. Move it outside every stage output (for example under data/raw/) "
                + "and track it there."
            )
        # A file tracked there before this check existed: update it, and warn, as status does.
        warnings.append(
            f"Warning: '{path_str}' is tracked inside a stage output: {overlap}. Move it and "
            + "its .pvt sidecar outside every stage output (for example under data/raw/); until "
            + "then Pivot refuses to replace that output."
        )
    warning = "\n".join(warnings) if warnings else None

    # Hash and cache
    if path.is_dir():
        dir_hash = cache.hash_directory(
            path, cache_dir=cache_dir, warn_left_out_symlinks=warn_left_out_symlinks
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
        file_hash, _ = cache.hash_file(path)
        file_size = path.stat().st_size
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


@cli_decorators.pivot_command(auto_discover=False)
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

    # Every pipeline's outputs, not only the pipeline this directory selects: a file tracked in
    # another pipeline's output would be removed when that pipeline runs.
    stage_outputs = discovery.declared_outputs(project_root)

    # Discover existing .pvt files
    existing_tracked = track_mod.discover_pvt_files(project_root)

    for path_str in paths:
        result = _track_single_path(
            path_str,
            cache_dir,
            stage_outputs,
            existing_tracked,
            force,
            warn_left_out_symlinks=not quiet,
        )
        if not quiet:
            if result["warning"]:
                click.echo(result["warning"], err=True)
            click.echo(f"Tracked: {result['tracked_path']}")
