from __future__ import annotations

import asyncio
import contextlib
import enum
import functools
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Literal, NamedTuple

import click
import lmdb

from pivot import config, project, registry, types
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets
from pivot.storage import cache, lock, ownership, state, track
from pivot.types import HashInfo, is_dir_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

RestoreResult = Literal["restored", "skipped", "missing"]
MAX_CONCURRENT_RESTORES = 32

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _open_hash_cache_for_checkout(
    state_dir: pathlib.Path | None,
) -> Iterator[state.StateDB | None]:
    """Yield ``state_dir``'s hash cache where it opens read-only, else None. Main's checkout
    never opened the database, and the cache is best-effort, so a database checkout cannot open
    (damaged, or unopenable on a writable filesystem, which ``state.open_readonly`` raises for)
    must not prevent a restore: the caller reads files and restores from objects as without a
    cache. Only the open is caught; an error inside the caller's block propagates. Other callers
    still report a damaged database (``state.open_readonly``'s policy)."""
    with contextlib.ExitStack() as stack:
        try:
            state_db = stack.enter_context(state.open_readonly(state_dir, create=False))
        except (OSError, lmdb.Error) as error:
            logger.debug("Restoring without the state database in %s: %s", state_dir, error)
            state_db = None
        yield state_db


class RestoreOutcome(NamedTuple):
    """What restoring one tracked file or stage output did, counted in files.

    A directory counts each file of its manifest; a partial in-place restore counts the files
    it wrote as restored, the ones already on disk as skipped and the ones it could not get
    from the cache as missing.
    """

    result: RestoreResult
    name: str
    restored_files: int = 0
    skipped_files: int = 0
    missing_files: int = 0


def _file_count(output_hash: HashInfo) -> int:
    return len(output_hash["manifest"]) if is_dir_hash(output_hash) else 1


class CheckoutBehavior(enum.StrEnum):
    """How to handle existing files during checkout."""

    ERROR = "error"  # Error if file already exists (default)
    SKIP_EXISTING = "skip_existing"  # Skip files that already exist (--only-missing)
    FORCE = "force"  # Overwrite existing files (--force)


def _get_stage_output_info() -> tuple[dict[str, HashInfo], dict[str, pathlib.Path]]:
    """Get output hash info from lock files for cached stage outputs only.

    Non-cached outputs (e.g. Metric with cache=False) are excluded —
    they are git-tracked and not Pivot's responsibility to restore.

    Uses per-stage state_dir from the registry for lock file lookup. Also returns each
    output's stage state_dir, whose StateDB holds the file hashes the stage's runs and
    explanations look up.
    """
    result = dict[str, HashInfo]()
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

        stage_state_dir = registry.get_stage_state_dir(stage_info, config.locate_state_dir())
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
                result[resolved] = out_hash
                state_dirs[resolved] = stage_state_dir

    return result, state_dirs


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
        # A hash pivot 0.3.3 recorded lists no empty directories, and repro still honors it.
        # When only empty directories differ, checkout would change no file, so it matches.
        return actual["hash"] == expected_hash["hash"] or (
            "empty_dirs" not in expected_hash and actual["manifest"] == expected_hash["manifest"]
        )
    if not path.is_file():
        return False
    actual_hash, _ = cache.hash_file(path, state_db, pending=pending)
    return actual_hash == expected_hash["hash"]


def _stage_claims_readers(
    stage_outputs: Mapping[str, HashInfo], cache_dir: pathlib.Path
) -> dict[str, Callable[[], ownership.StageClaims]]:
    """For each stage output path, a reader of its stage's claims (``ownership.StageClaims``):
    the hashes its lock records (``stage_outputs``), its ``nocommit:`` record, and the cache
    object root ``cache_dir``. Each reads once, on its first call, from any thread: a restore
    runs in a worker thread, where the command's click context is not available."""
    store = cli_helpers.get_workspace_store()
    if store is None:
        return {}
    default_state_dir = config.get_state_dir()
    project_root = project.get_project_root()
    readers: dict[str, Callable[[], ownership.StageClaims]] = {}
    for stage_name in cli_helpers.list_stages():
        stage_info = cli_helpers.get_stage(stage_name)
        paths = [str(store.resolve_display_path(out)) for out in stage_info["outs"]]
        reader = functools.cache(
            functools.partial(
                ownership.read_stage_claims,
                project_root,
                stage_name,
                registry.get_stage_state_dir(stage_info, default_state_dir),
                {
                    pathlib.Path(path): stage_outputs[path]
                    for path in paths
                    if path in stage_outputs
                },
                cache_dir=cache_dir,
            )
        )
        readers.update(dict.fromkeys(paths, reader))
    return readers


def _restore_targets(
    files: Mapping[str, HashInfo],
    stage_output_paths: Mapping[str, str | None],
    claims: Mapping[str, Callable[[], ownership.StageClaims]],
) -> dict[str, ownership.ForcedRestore]:
    """What restoring each file replaces, for ``ownership.forced_restore_refusal``: a stage
    output (path -> the stage, when the target named it) as that stage's output, with its claims;
    any other path as a tracked path's own restore."""
    return {
        path: ownership.ForcedRestore(
            replacement=(
                ownership.stage_output(pathlib.Path(path), stage_output_paths[path])
                if path in stage_output_paths
                else ownership.tracked_restore(pathlib.Path(path))
            ),
            restored=output_hash,
            claims=claims.get(path) if path in stage_output_paths else None,
        )
        for path, output_hash in files.items()
    }


def _refuse(targets: Iterable[ownership.ForcedRestore]) -> None:
    """Fail, naming every refused target, if restoring these would remove a user's file."""
    refusal = ownership.forced_restore_refusal(project.get_project_root(), targets)
    if refusal is not None:
        raise click.ClickException(str(refusal))


def _refuse_swap(target: ownership.ForcedRestore, entry: pathlib.Path) -> None:
    """Fail, naming the target, if the entry its swap removes, now at ``entry``, holds a user's
    file (``ownership.swap_refusal``)."""
    refusal = ownership.swap_refusal(project.get_project_root(), target, entry)
    if refusal is not None:
        raise click.ClickException(str(refusal))


def _restore_path_sync(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    state_dir: pathlib.Path | None = None,
    hash_state_dir: pathlib.Path | None = None,
    *,
    target: ownership.ForcedRestore,
    batch: state.FileHashBatch | None = None,
) -> RestoreOutcome:
    """Restore a file or directory from cache (sync version).

    Holds the restore lock for path across the existence check, the ownership checks and the
    restore, so a concurrent checkout never compares against, or deletes, a half-restored output.
    ``target`` is what the restore replaces (``_restore_targets``). ``hash_state_dir``
    (``state_dir`` when None) is the state directory whose StateDB the path's file hashes are
    looked up in and recorded in: a stage output's own state directory. The hashes of the files it
    restores or reads go into ``batch``, which the command saves once (``state.FileHashBatch``);
    without one nothing is recorded.

    Returns:
        What the restore did, for the caller to count and report.

    Raises:
        click.ClickException: For immediate failures (path traversal, unknown target,
            "already exists" without --force, a stage output that holds user-owned files).
            Cache misses return ("missing", name) instead of raising.
    """
    with cache.restore_lock(path, state_dir) as lock_token:
        return _restore_path_locked(
            path,
            output_hash,
            cache_dir,
            checkout_modes,
            behavior,
            state_dir,
            lock_token,
            hash_state_dir if hash_state_dir is not None else state_dir,
            target=target,
            batch=batch,
        )


def _restore_path_locked(
    path: pathlib.Path,
    output_hash: HashInfo,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    state_dir: pathlib.Path | None,
    lock_token: str | None,
    hash_state_dir: pathlib.Path | None,
    *,
    target: ownership.ForcedRestore,
    batch: state.FileHashBatch | None,
) -> RestoreOutcome:
    """Body of _restore_path_sync; the caller holds the restore lock for path."""

    def record(entries: list[tuple[pathlib.Path, os.stat_result, str]]) -> None:
        if batch is not None and hash_state_dir is not None:
            batch.add(hash_state_dir, entries)

    total = _file_count(output_hash)
    if path.exists():
        match behavior:
            case CheckoutBehavior.ERROR:
                read = list[tuple[pathlib.Path, os.stat_result, str]]()
                with _open_hash_cache_for_checkout(hash_state_dir) as state_db:
                    matches = _on_disk_matches_expected(
                        path, output_hash, cache_dir, state_db, read
                    )
                record(read)
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
                if not is_dir_hash(output_hash) or path.is_symlink() or not path.is_dir():
                    return RestoreOutcome("skipped", path.name, skipped_files=total)
                settled = list[tuple[pathlib.Path, os.stat_result, str]]()
                with _open_hash_cache_for_checkout(hash_state_dir) as state_db:
                    outcome = cache.restore_missing_in_directory(
                        path, output_hash, cache_dir, checkout_modes, state_db, settled
                    )
                record(settled)
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
                missing = cache.missing_cache_objects(output_hash, cache_dir, checkout_modes)
                if missing:
                    raise click.ClickException(
                        f"Cannot overwrite '{path.name}': cache object missing: {', '.join(missing)}. "
                        + "The workspace output was not changed."
                    )
            case _:  # pyright: ignore[reportUnnecessaryComparison] - defensive for future enum values
                raise ValueError(f"Unhandled checkout behavior: {behavior}")  # pyright: ignore[reportUnreachable]

    # Checked here, under the restore lock, before the replacement is built: for --force again
    # after the check of every target before the first restore, so a user file that appeared in
    # the meantime refuses this target alone; for an --only-missing swap once it is decided. The
    # swap checks once more, over the entry it removes, once the replacement is built.
    if path.exists() or path.is_symlink():
        _refuse([target])
    restored = list[tuple[pathlib.Path, os.stat_result, str]]()
    success = cache.restore_from_cache(
        path,
        output_hash,
        cache_dir,
        checkout_modes=checkout_modes,
        state_dir=state_dir,
        lock_token=lock_token,
        before_swap=functools.partial(_refuse_swap, target),
        recorded=restored,
    )
    if not success:
        return RestoreOutcome("missing", path.name, missing_files=total)

    record(restored)
    return RestoreOutcome("restored", path.name, restored_files=total)


async def _checkout_files_async(
    files: Mapping[str, HashInfo],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    callback: Callable[[int, int, str], None] | None = None,
    state_dir: pathlib.Path | None = None,
    hash_state_dirs: Mapping[str, pathlib.Path] | None = None,
    *,
    targets: Mapping[str, ownership.ForcedRestore],
    batch: state.FileHashBatch | None = None,
) -> list[RestoreOutcome]:
    """Restore files in parallel and return what each restore did.

    ``targets`` says what each file's restore replaces (``_restore_targets``).
    ``hash_state_dirs`` maps a stage output's path to its stage's state directory, which
    holds its file hashes; every other path's are in ``state_dir``.

    Raises:
        click.ClickException: For immediate failures (aggregated if multiple).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESTORES)
    outcomes = list[RestoreOutcome]()
    immediate_errors = list[click.ClickException]()

    async def restore_one(abs_path_str: str, output_hash: HashInfo) -> None:
        path = pathlib.Path(abs_path_str)
        try:
            async with semaphore:
                outcome = await asyncio.to_thread(
                    _restore_path_sync,
                    path,
                    output_hash,
                    cache_dir,
                    checkout_modes,
                    behavior,
                    state_dir,
                    (hash_state_dirs or {}).get(abs_path_str),
                    target=targets[abs_path_str],
                    batch=batch,
                )
            outcomes.append(outcome)
            if callback:
                callback(len(outcomes), len(files), outcome.name)
        except click.ClickException as e:
            immediate_errors.append(e)

    # Anything else a restore raises leaves as an exception group, which the command's error
    # handler words (cli/decorators.py).
    async with asyncio.TaskGroup() as tg:
        for abs_path_str, output_hash in files.items():
            _ = tg.create_task(restore_one(abs_path_str, output_hash))

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
    stage_outputs: dict[str, HashInfo],
) -> tuple[dict[str, HashInfo], dict[str, str | None]]:
    """Validate targets and build the files dict for checkout, with the stage output restores
    among them (path -> the stage, when the target named it).

    A restore requested through a stage is that stage's output restore, whatever else the path
    is. A path target restores the tracked path's own content when the path is tracked, and the
    stage output otherwise.

    Targets should already have .pvt suffixes converted by _dedupe_targets().

    Raises:
        click.ClickException: For path traversal or unknown targets.
    """
    files = dict[str, HashInfo]()
    stage_output_paths = dict[str, str | None]()
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
            stage_output_paths.update(dict.fromkeys(cached, resolved["stage_name"]))
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
            pvt_data = tracked_files[abs_path_str]
            files[abs_path_str] = track.pvt_to_hash_info(pvt_data)
            continue

        # Check if it's a stage output
        if abs_path_str in stage_outputs:
            files[abs_path_str] = stage_outputs[abs_path_str]
            stage_output_paths.setdefault(abs_path_str, None)
            continue

        # Unknown target
        raise click.ClickException(
            f"'{target}' is not a tracked file or stage output. "
            + "Use 'pivot list' to see stages or 'pivot track' to track files."
        )

    return files, stage_output_paths


async def _checkout_main_async(
    targets: tuple[str, ...],
    tracked_files: dict[str, track.PvtData],
    stage_outputs: dict[str, HashInfo],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    behavior: CheckoutBehavior,
    callback: Callable[[int, int, str], None] | None = None,
    state_dir: pathlib.Path | None = None,
    output_state_dirs: dict[str, pathlib.Path] | None = None,
    batch: state.FileHashBatch | None = None,
) -> list[RestoreOutcome]:
    """Main async checkout logic: what restoring each tracked file and stage output did.

    ``--force`` replaces every existing target, so every target is checked before any is
    changed: if one would remove a user's file, nothing is restored and the command fails,
    naming every refused target and the files that block it. Each target is checked again before
    its replacement is built, and at its swap (``_restore_path_locked``).

    ``output_state_dirs`` maps each stage output's path to its stage's state directory.
    """
    claims = _stage_claims_readers(stage_outputs, cache_dir)
    if targets:
        unique_targets = _dedupe_targets(targets)
        files, stage_output_paths = _validate_and_build_files(
            unique_targets, tracked_files, stage_outputs
        )
        restore_targets = _restore_targets(files, stage_output_paths, claims)
        if behavior is CheckoutBehavior.FORCE:
            _refuse(restore_targets.values())
        return await _checkout_files_async(
            files,
            cache_dir,
            checkout_modes,
            behavior,
            callback,
            state_dir=state_dir,
            hash_state_dirs=output_state_dirs,
            targets=restore_targets,
            batch=batch,
        )
    else:
        # Checkout all tracked files and stage outputs
        tracked_as_hashes: dict[str, HashInfo] = {
            path: track.pvt_to_hash_info(pvt) for path, pvt in tracked_files.items()
        }
        tracked_targets = _restore_targets(tracked_as_hashes, {}, claims)
        output_targets = _restore_targets(stage_outputs, dict.fromkeys(stage_outputs), claims)
        if behavior is CheckoutBehavior.FORCE:
            _refuse([*tracked_targets.values(), *output_targets.values()])
        combined_total = len(tracked_as_hashes) + len(stage_outputs)
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
                tracked_as_hashes,
                cache_dir,
                checkout_modes,
                behavior,
                progress_cb,
                state_dir=state_dir,
                targets=tracked_targets,
                batch=batch,
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
                targets=output_targets,
                batch=batch,
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
@cli_decorators.checkout_mode_option
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
    stage_outputs = dict[str, HashInfo]()
    output_state_dirs = dict[str, pathlib.Path]()
    if pipeline is not None:
        stage_outputs, output_state_dirs = _get_stage_output_info()

    state_dir = config.locate_state_dir()

    # Run async checkout. The file hashes every restore read or wrote are saved once per state
    # directory when it ends, also when a target failed, best-effort.
    batch = state.FileHashBatch()
    try:
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
                    batch=batch,
                )
            )
    finally:
        batch.save()

    success = _print_summary(outcomes, quiet)
    if not success:
        ctx.exit(1)
