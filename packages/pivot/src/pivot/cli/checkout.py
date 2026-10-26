from __future__ import annotations

import asyncio
import contextlib
import enum
import functools
import os
import pathlib
from typing import TYPE_CHECKING, Literal, NamedTuple

import click

from pivot import config, exceptions, project, registry, types
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import errors as cli_errors
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets
from pivot.storage import cache, listing, lock, state, track
from pivot.types import HashInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

RestoreResult = Literal["restored", "skipped", "missing"]
MAX_CONCURRENT_RESTORES = 32


class RestoreOutcome(NamedTuple):
    """What restoring one tracked file or stage output did, counted in files.

    A directory counts each file of its listing; a partial in-place restore counts the files
    it wrote as restored, the ones already on disk as skipped and the ones it could not get
    from the cache as missing.
    """

    result: RestoreResult
    name: str
    restored_files: int = 0
    skipped_files: int = 0
    missing_files: int = 0


class Expected(NamedTuple):
    """What a tracked path or stage output should hold: its recorded hash and, for a directory,
    how to read its listing (the committed listing file, else the cache's object)."""

    output_hash: HashInfo
    read_members: Callable[[], listing.DirListing] | None = None


def _file_count(output_hash: HashInfo) -> int:
    return output_hash["nfiles"] if is_dir_hash(output_hash) else 1


class CheckoutBehavior(enum.StrEnum):
    """How to handle existing files during checkout."""

    ERROR = "error"  # Error if file already exists (default)
    SKIP_EXISTING = "skip_existing"  # Skip files that already exist (--only-missing)
    FORCE = "force"  # Overwrite existing files (--force)


def _get_stage_output_info(
    cache_dir: pathlib.Path,
) -> tuple[dict[str, Expected], dict[str, pathlib.Path]]:
    """Get what each cached stage output should hold, from lock files and their listings.

    Non-cached outputs (e.g. Metric with cache=False) are excluded —
    they are git-tracked and not Pivot's responsibility to restore.

    Uses per-stage state_dir from the registry for lock file lookup. Also returns each
    output's stage state_dir, whose StateDB holds the file hashes the stage's runs and
    ``pivot status`` look up.
    """
    result = dict[str, Expected]()
    state_dirs = dict[str, pathlib.Path]()
    store = cli_helpers.get_workspace_store()
    if store is None:
        return result, state_dirs

    for stage_name in cli_helpers.list_stages():
        stage_info = cli_helpers.get_stage(stage_name)
        ref_by_identity = {out.identity: out for out in stage_info["outs"]}
        cached_identities = {
            out.identity for out in stage_info["outs"] if out.tag is not types.ArtifactTag.METRIC
        }

        stage_state_dir = registry.get_stage_state_dir(stage_info, config.get_state_dir())
        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(stage_state_dir))
        lock_data = stage_lock.read()
        if lock_data:
            for identity, out_hash in lock_data["output_hashes"].items():
                if identity not in cached_identities:
                    continue
                ref = ref_by_identity.get(identity)
                if ref is None:
                    continue
                resolved = str(store.resolve_display_path(ref))
                read_members = None
                if is_dir_hash(out_hash):
                    read_members = functools.partial(
                        listing.read, stage_lock.output_listing_path(identity), out_hash, cache_dir
                    )
                result[resolved] = Expected(out_hash, read_members)
                state_dirs[resolved] = stage_state_dir

    return result, state_dirs


def _tracked_expected(data_path: str, pvt_data: track.PvtData, cache_dir: pathlib.Path) -> Expected:
    output_hash = track.pvt_to_hash_info(pvt_data)
    read_members = None
    if is_dir_hash(output_hash):
        read_members = functools.partial(
            track.tracked_members, pathlib.Path(data_path), pvt_data, cache_dir
        )
    return Expected(output_hash, read_members)


def _hash_cache(
    state_dir: pathlib.Path | None,
) -> contextlib.AbstractContextManager[state.StateDB | None]:
    """``state_dir``'s StateDB, read-only, so a check reuses the hashes runs, ``pivot status``
    and earlier checkouts recorded; None when it has none yet, so a check never creates it."""
    if state_dir is not None and (state_dir / "state.lmdb").exists():
        return state.StateDB(state_dir, readonly=True)
    return contextlib.nullcontext(None)


def _save_file_hashes(
    state_dir: pathlib.Path | None, entries: list[tuple[pathlib.Path, os.stat_result, str]]
) -> None:
    """Record the hashes of files this checkout restored or read in ``state_dir``'s StateDB,
    so the next status, explanation or run looks them up instead of reading the files."""
    if state_dir is None or not entries:
        return
    with state.StateDB(state_dir) as state_db:
        state_db.save_many(entries)


def _on_disk_matches_expected(
    path: pathlib.Path,
    expected_hash: HashInfo,
    cache_dir: pathlib.Path,
    state_db: state.StateDB | None,
    pending: list[tuple[pathlib.Path, os.stat_result, str]],
) -> bool:
    if is_dir_hash(expected_hash):
        if not path.is_dir():
            return False
        actual = cache.hash_directory(path, state_db, cache_dir=cache_dir, pending=pending)
        return actual["record"]["hash"] == expected_hash["hash"]
    if not path.is_file():
        return False
    actual_hash, _ = cache.hash_file(path, state_db, pending=pending)
    return actual_hash == expected_hash["hash"]


def _restore_path_sync(
    path: pathlib.Path,
    expected: Expected,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    state_dir: pathlib.Path | None = None,
    hash_state_dir: pathlib.Path | None = None,
) -> RestoreOutcome:
    """Restore a file or directory from cache (sync version).

    Holds the restore lock for path across the existence check and the restore, so a
    concurrent checkout never compares against, or deletes, a half-restored output.
    ``hash_state_dir`` (``state_dir`` when None) is the state directory whose StateDB the
    path's file hashes are looked up in and recorded in: a stage output's own state directory.

    Returns:
        What the restore did, for the caller to count and report.

    Raises:
        click.ClickException: For immediate failures (path traversal, unknown target,
            "already exists" without --force). Cache misses return ("missing", name)
            instead of raising.
        exceptions.CorruptListingError: A directory's listing does not hash to its record;
            nothing is written.
    """
    with cache.restore_lock(path, state_dir) as lock_token:
        return _restore_path_locked(
            path,
            expected,
            cache_dir,
            checkout_modes,
            behavior,
            state_dir,
            lock_token,
            hash_state_dir if hash_state_dir is not None else state_dir,
        )


def _restore_path_locked(
    path: pathlib.Path,
    expected: Expected,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    state_dir: pathlib.Path | None,
    lock_token: str | None,
    hash_state_dir: pathlib.Path | None,
) -> RestoreOutcome:
    """Body of _restore_path_sync; the caller holds the restore lock for path."""
    output_hash = expected.output_hash
    total = _file_count(output_hash)
    members: listing.DirListing | None = None
    if expected.read_members is not None:
        try:
            members = expected.read_members()
        except exceptions.ListingMissingError:
            return RestoreOutcome("missing", path.name, missing_files=total)
    if path.exists():
        match behavior:
            case CheckoutBehavior.ERROR:
                read = list[tuple[pathlib.Path, os.stat_result, str]]()
                with _hash_cache(hash_state_dir) as state_db:
                    matches = _on_disk_matches_expected(
                        path, output_hash, cache_dir, state_db, read
                    )
                _save_file_hashes(hash_state_dir, read)
                if matches:
                    return RestoreOutcome("skipped", path.name, skipped_files=total)
                raise click.ClickException(
                    f"'{path.name}' already exists with different content. "
                    + "Use --force to overwrite or --only-missing to skip existing files."
                )
            case CheckoutBehavior.SKIP_EXISTING:
                # Existing real directories can fill missing files in place, or swap to the
                # target when every differing or extra file is pivot's own cached output.
                # Never overwrite content the cache lacks (a local change).
                if members is None or path.is_symlink() or not path.is_dir():
                    return RestoreOutcome("skipped", path.name, skipped_files=total)
                settled = list[tuple[pathlib.Path, os.stat_result, str]]()
                with _hash_cache(hash_state_dir) as state_db:
                    outcome = cache.restore_missing_in_directory(
                        path, members, cache_dir, checkout_modes, state_db, settled
                    )
                _save_file_hashes(hash_state_dir, settled)
                if outcome["local_changes"]:
                    raise click.ClickException(
                        f"'{path.name}' has local changes that are not in the cache: "
                        + f"{', '.join(outcome['local_changes'])}. Use --force to overwrite."
                    )
                if not outcome["replace"]:
                    if outcome["missing"]:
                        return RestoreOutcome(
                            "missing",
                            path.name,
                            skipped_files=total - outcome["missing"],
                            missing_files=outcome["missing"],
                        )
                    return RestoreOutcome(
                        "restored" if outcome["restored"] else "skipped",
                        path.name,
                        restored_files=outcome["restored"],
                        skipped_files=total - outcome["restored"],
                    )
            case CheckoutBehavior.FORCE:
                missing = cache.missing_cache_objects(
                    output_hash, members, cache_dir, checkout_modes
                )
                if missing:
                    raise click.ClickException(
                        f"Cannot overwrite '{path.name}': cache object missing: {', '.join(missing)}. "
                        + "The workspace output was not changed."
                    )
            case _:  # pyright: ignore[reportUnnecessaryComparison] - defensive for future enum values
                raise ValueError(f"Unhandled checkout behavior: {behavior}")  # pyright: ignore[reportUnreachable]

    success = cache.restore_from_cache(
        path,
        output_hash,
        cache_dir,
        checkout_modes=checkout_modes,
        state_dir=state_dir,
        lock_token=lock_token,
        members=members,
    )
    if not success:
        return RestoreOutcome("missing", path.name, missing_files=total)

    _save_file_hashes(hash_state_dir, cache.known_file_hashes(path, output_hash, members))
    return RestoreOutcome("restored", path.name, restored_files=total)


async def _checkout_files_async(
    files: Mapping[str, Expected],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    callback: Callable[[int, int, str], None] | None = None,
    state_dir: pathlib.Path | None = None,
    hash_state_dirs: Mapping[str, pathlib.Path] | None = None,
) -> list[RestoreOutcome]:
    """Restore files in parallel and return what each restore did.

    ``hash_state_dirs`` maps a stage output's path to its stage's state directory, which
    holds its file hashes; every other path's are in ``state_dir``.

    Raises:
        click.ClickException: For immediate failures (aggregated if multiple).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESTORES)
    outcomes = list[RestoreOutcome]()
    immediate_errors = list[click.ClickException]()

    async def restore_one(abs_path_str: str, expected: Expected) -> None:
        path = pathlib.Path(abs_path_str)
        try:
            async with semaphore:
                outcome = await asyncio.to_thread(
                    _restore_path_sync,
                    path,
                    expected,
                    cache_dir,
                    checkout_modes,
                    behavior,
                    state_dir,
                    (hash_state_dirs or {}).get(abs_path_str),
                )
            outcomes.append(outcome)
            if callback:
                callback(len(outcomes), len(files), outcome.name)
        except click.ClickException as e:
            immediate_errors.append(e)

    try:
        async with asyncio.TaskGroup() as tg:
            for abs_path_str, output_hash in files.items():
                _ = tg.create_task(restore_one(abs_path_str, output_hash))
    except* Exception as eg:
        # Convert unexpected exceptions to a friendly error message, rendering a PivotError's
        # actionable suggestion the same way the top-level CLI error handler does (cli/errors.py).
        messages = [
            cli_errors.format_pivot_error(e) if isinstance(e, exceptions.PivotError) else str(e)
            for e in eg.exceptions
        ]
        raise click.ClickException("\n".join(messages)) from None

    # Re-raise immediate failures (all of them, aggregated)
    if immediate_errors:
        msgs = [str(e.message) for e in immediate_errors]
        raise click.ClickException("\n".join(msgs))

    return outcomes


def _all_stages_in_context() -> dict[str, registry.RegistryStageInfo]:
    pipeline = cli_decorators.get_pipeline_from_context()
    return {} if pipeline is None else cli_helpers.get_all_stages()


def _dedupe_targets(targets: tuple[str, ...]) -> list[str]:
    """Deduplicate targets: stage outputs by resolved identity, files by normalized path.

    Handles both data paths (data.txt) and .pvt paths (data.txt.pvt) resolving
    to the same file.
    """
    all_stages = _all_stages_in_context()
    seen = set[str]()
    unique = list[str]()
    for target in targets:
        # Convert .pvt to data path
        target_path = pathlib.Path(target)
        if target_path.suffix == ".pvt":
            target = str(track.get_data_path(target_path))

        resolved = cli_targets.resolve_cli_target(target, all_stages)
        if resolved is not None:
            dedupe_key = types.identity_key(resolved["identity"])
        else:
            dedupe_key = str(project.normalize_path(target))
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            unique.append(target)
    return unique


def _validate_and_build_files(
    targets: list[str],
    tracked_files: dict[str, track.PvtData],
    stage_outputs: dict[str, Expected],
    cache_dir: pathlib.Path,
) -> dict[str, Expected]:
    """Validate targets and build files dict for checkout.

    Targets should already have .pvt suffixes converted by _dedupe_targets().

    Raises:
        click.ClickException: For path traversal or unknown targets.
    """
    files = dict[str, Expected]()
    all_stages = _all_stages_in_context()

    for target in targets:
        resolved = cli_targets.resolve_cli_target(target, all_stages)
        if resolved is not None:
            store = cli_helpers.require_workspace_store()
            cached = {
                path: stage_outputs[path]
                for ref in resolved["refs"]
                if ref.tag is not types.ArtifactTag.METRIC
                and (path := str(store.resolve_display_path(ref))) in stage_outputs
            }
            if not cached:
                stage_name = resolved["stage_name"]
                raise click.ClickException(
                    f"Stage '{stage_name}' has no cached outputs. "
                    + f"Run 'pivot repro {stage_name}' first."
                )
            files.update(cached)
            continue

        # Use normalized path (preserve symlinks) to match keys in tracked_files/stage_outputs;
        # a target typed through a symlink to the project is placed under the project root
        project_root = project.get_project_root()
        relative = project.relative_within(project.normalize_path(target), project_root)
        if relative is None:
            raise click.ClickException(f"Target '{target}' resolves outside project root")
        abs_path_str = str(project_root / relative)

        # Check if it's a tracked file
        if abs_path_str in tracked_files:
            files[abs_path_str] = _tracked_expected(
                abs_path_str, tracked_files[abs_path_str], cache_dir
            )
            continue

        # Check if it's a stage output
        if abs_path_str in stage_outputs:
            files[abs_path_str] = stage_outputs[abs_path_str]
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
    stage_outputs: dict[str, Expected],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    callback: Callable[[int, int, str], None] | None = None,
    state_dir: pathlib.Path | None = None,
    output_state_dirs: dict[str, pathlib.Path] | None = None,
) -> list[RestoreOutcome]:
    """Main async checkout logic: what restoring each tracked file and stage output did.

    ``output_state_dirs`` maps each stage output's path to its stage's state directory.
    """
    if targets:
        unique_targets = _dedupe_targets(targets)
        files = _validate_and_build_files(unique_targets, tracked_files, stage_outputs, cache_dir)
        return await _checkout_files_async(
            files,
            cache_dir,
            checkout_modes,
            behavior,
            callback,
            state_dir=state_dir,
            hash_state_dirs=output_state_dirs,
        )
    else:
        # Checkout all tracked files and stage outputs
        tracked_expected = {
            path: _tracked_expected(path, pvt, cache_dir) for path, pvt in tracked_files.items()
        }
        combined_total = len(tracked_expected) + len(stage_outputs)
        shared_completed = 0

        def shared_callback(_completed: int, _total: int, filename: str) -> None:
            nonlocal shared_completed
            shared_completed += 1
            if callback:
                callback(shared_completed, combined_total, filename)

        progress_cb = shared_callback if callback else None
        # Run both in parallel with shared progress counter
        t1 = asyncio.create_task(
            _checkout_files_async(
                tracked_expected,
                cache_dir,
                checkout_modes,
                behavior,
                progress_cb,
                state_dir=state_dir,
            )
        )
        t2 = asyncio.create_task(
            _checkout_files_async(
                stage_outputs,
                cache_dir,
                checkout_modes,
                behavior,
                progress_cb,
                state_dir=state_dir,
                hash_state_dirs=output_state_dirs,
            )
        )
        tracked, outputs = await asyncio.gather(t1, t2)
        return tracked + outputs


def _print_summary(outcomes: list[RestoreOutcome], quiet: bool) -> bool:
    """Print checkout summary, counting every unit in files.

    Returns:
        True if all files were restored successfully, False if any were missing.
    """
    restored = sum(outcome.restored_files for outcome in outcomes)
    skipped = sum(outcome.skipped_files for outcome in outcomes)
    missing = sum(outcome.missing_files for outcome in outcomes)
    failures = [outcome.name for outcome in outcomes if outcome.result == "missing"]
    if not quiet and (restored or skipped):
        if restored:
            click.echo(f"Restored {restored} file(s)")
        if skipped:
            click.echo(f"Skipped {skipped} file(s) (already exist)")

    if failures:
        if restored or skipped:
            click.echo("")  # Add blank line after success summary
        click.echo(f"Missing {missing} file(s) from:")
        for name in failures[:15]:
            click.echo(f"  {name}")
        if len(failures) > 15:
            click.echo(f"  ... and {len(failures) - 15} more")
        click.echo("")
        click.echo("Run 'pivot pull' to fetch from remote storage.")
        return False

    return True


@cli_decorators.pivot_command(allow_all=True)
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

    # Determine checkout modes - CLI flag overrides config (single mode, no fallback, except
    # that a hard link refused with EMLINK is copied whatever the order is)
    checkout_modes = (
        [cache.CheckoutMode(checkout_mode)] if checkout_mode else config.get_checkout_mode_order()
    )

    # Discover tracked files
    tracked_files = track.discover_pvt_files(project_root)

    # Get stage output info from lock files (cached outputs only)
    pipeline = cli_decorators.get_pipeline_from_context()
    stage_outputs = dict[str, Expected]()
    output_state_dirs = dict[str, pathlib.Path]()
    if pipeline is not None:
        stage_outputs, output_state_dirs = _get_stage_output_info(cache_dir)

    state_dir = config.get_state_dir()

    # Run async checkout
    with cli_helpers.TransferProgress("Restoring", quiet=quiet) as progress:
        outcomes = asyncio.run(
            _checkout_main_async(
                targets,
                tracked_files,
                stage_outputs,
                cache_dir,
                checkout_modes,
                behavior,
                callback=progress.callback,
                state_dir=state_dir,
                output_state_dirs=output_state_dirs,
            )
        )

    success = _print_summary(outcomes, quiet)
    if not success:
        ctx.exit(1)
