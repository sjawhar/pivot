from __future__ import annotations

import os
import pathlib
from typing import TypedDict

import click

from pivot import config, discovery, git, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.storage import cache, listing, ownership, state
from pivot.storage import track as track_mod


class TrackResult(TypedDict):
    """Result of tracking a single path."""

    warning: str | None
    tracked_path: str


def _tracking_files(data_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """The files tracking ``data_path`` writes, or for a file removes: its .pvt and listing."""
    pvt_path = track_mod.get_pvt_path(data_path)
    return pvt_path, listing.tracked_listing_path(pvt_path)


def _tracking_file_collision(
    path: pathlib.Path, existing_tracked: dict[str, track_mod.PvtData], project_root: pathlib.Path
) -> str | None:
    """Why tracking ``path`` would collide with another tracked path, or None.

    A collision is ``path`` being the .pvt or listing file pivot keeps for another tracked
    path, or one of ``path``'s own being another tracked path: tracking would overwrite or
    remove it. Any other name, ``.pvt`` or ``.pvt.listing`` endings included, can be tracked.
    """
    for tracked in existing_tracked:
        other = pathlib.Path(tracked)
        if other == path:
            continue
        shown = project.to_relative_path(other, project_root)
        if path in _tracking_files(other):
            return f"it is the file pivot keeps to record the tracked path '{shown}'"
        if other in _tracking_files(path):
            return f"tracking it writes the file '{shown}', which is itself a tracked path"
    return None


def _foreign_listing(
    listing_path: pathlib.Path,
    recorded: track_mod.PvtData | None,
    new_data: bytes | None,
    cache_dir: pathlib.Path,
) -> bool:
    """Whether something stands at ``listing_path`` that pivot did not write for this target:
    anything but a file holding the listing of the directory its .pvt records (``recorded``) or
    the bytes about to be written there (``new_data``).

    A listing file is pivot's when the cache (configured or legacy,
    ``cache.resolve_cache_object_path``) holds exactly its bytes at their own digest: a listing
    pivot stored, current or stale. With no cache copy at that digest it is pivot's when its
    bytes hash to the recorded directory, the rule ``listing._resolve`` reads it by. XXH64 alone
    never overrides a cache copy whose bytes differ.
    """
    if not os.path.lexists(listing_path):
        return False
    if listing_path.is_symlink() or not listing_path.is_file():
        return True
    try:
        data = listing_path.read_bytes()
    except OSError:
        return True
    if data == new_data:
        return False
    data_digest = listing.digest(data)
    cached_path = cache.resolve_cache_object_path(cache_dir, data_digest)
    if cached_path is not None:
        try:
            return cached_path.read_bytes() != data
        except OSError:
            return True
    return recorded is None or "nfiles" not in recorded or data_digest != recorded["hash"]


def _track_single_path(
    path_str: str,
    cache_dir: pathlib.Path,
    stage_outputs: dict[str, ownership.DeclaredOutput],
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

    collision = _tracking_file_collision(path, existing_tracked, project_root)
    if collision is not None:
        raise click.ClickException(f"Cannot track '{path_str}': {collision}; rename one of them")

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
    track_mod.validate_pvt_path_name(path.name)

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
    listing_path = listing.tracked_listing_path(pvt_path)
    listing_data: bytes | None = None
    if path.is_dir():
        hashed = cache.hash_directory(
            path, cache_dir=cache_dir, warn_left_out_symlinks=warn_left_out_symlinks, pending=read
        )
        for entry in hashed["listing"]["files"]:
            file_path = path / entry["relpath"]
            file_cache_path = cache.get_cache_path(cache_dir, entry["hash"])
            cache.copy_to_cache(file_path, file_cache_path, executable=entry["isexec"])
        record = hashed["record"]
        listing_data = hashed["data"]
        listing.store_object(cache_dir, record["hash"], listing_data)

        pvt_data = track_mod.PvtData(
            path=path.name,
            hash=record["hash"],
            size=record["size"],
            nfiles=record["nfiles"],
        )
    else:
        file_hash, file_stat = cache.hash_file(path, pending=read)
        file_size = file_stat.st_size
        file_cache_path = cache.get_cache_path(cache_dir, file_hash)
        cache.copy_to_cache(path, file_cache_path)

        pvt_data = track_mod.PvtData(
            path=path.name,
            hash=file_hash,
            size=file_size,
        )

    # Pivot writes or removes only a listing it wrote for this target.
    if _foreign_listing(listing_path, existing_tracked.get(abs_path_str), listing_data, cache_dir):
        shown = project.to_relative_path(listing_path, project_root)
        raise click.ClickException(
            f"Cannot track '{path_str}': '{shown}' is in the way, and it is not a listing pivot "
            + "wrote for it; move it aside (pivot never overwrites or removes a file it did not "
            + "write)"
        )
    # Write the .pvt after its listing, so no .pvt names a listing not yet written.
    if listing_data is not None:
        listing.write(listing_path, listing_data)
    else:
        listing_path.unlink(missing_ok=True)  # the listing of a directory tracked here before
    track_mod.write_pvt_file(pvt_path, pvt_data)
    written = [pvt_path] if listing_data is None else [pvt_path, listing_path]

    ignored = git.ignored_paths(project_root, written)
    if ignored:
        names = ", ".join(str(ignored_path.relative_to(project_root)) for ignored_path in ignored)
        ignored_warning = (
            f"Warning: git ignores {names}; commit {'it' if len(ignored) == 1 else 'them'} "
            + "with the data path's other tracking files (un-ignore them, or `git add -f`)"
        )
        warning = ignored_warning if warning is None else f"{warning}\n{ignored_warning}"

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
    # The hashes read to track, so the next status reads none of these files. Saved where #503's
    # resolver writes state, which the save creates only when it has a hash to save.
    state.save_file_hashes(config.locate_state_dir(), read, create=True)
