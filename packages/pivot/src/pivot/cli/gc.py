from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import TYPE_CHECKING

import click

from pivot import config, exceptions, project
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.storage import gc, state

if TYPE_CHECKING:
    from pivot.storage.gc import GcResult, Plan, ProjectScope, Roots


def _echo(message: str, *, to_stderr: bool) -> None:
    click.echo(message, err=to_stderr)


def _prune_run_history(state_dir: pathlib.Path, *, dry_run: bool) -> tuple[int, int]:
    """Apply ``core.run_history_retention`` to this project's run history.

    Returns the number of runs pruned (or that a real run would prune) and the retention.
    Pruning also drops the run-cache rows of the runs it removes (``StateDB.prune_runs``),
    which is the policy every pipeline run already applies.
    """
    retention = config.get_run_history_retention()
    if not (state_dir / "state.lmdb").exists():
        return 0, retention
    with state.StateDB(state_dir, readonly=dry_run) as db:
        if dry_run:
            return max(0, len(db.list_runs(limit=retention + 1000)) - retention), retention
        return db.prune_runs(retention), retention


def _summary_lines(
    result: GcResult,
    roots: Roots,
    scopes: list[ProjectScope],
    object_root: pathlib.Path,
    grace_hours: float,
) -> list[str]:
    """The two lines every collection prints before saying what it removed.

    ``.pvt`` files name objects exactly as a lock does and are counted with the locks.
    """
    kept = result["kept"]
    by = kept["by"]
    counts = roots["counts"]
    return [
        f"Scanned {result['scanned']['objects']:,} objects "
        + f"({gc.format_bytes(result['scanned']['bytes'])}) in {object_root}.",
        f"Kept {kept['objects']:,} objects ({gc.format_bytes(kept['bytes'])}): "
        + f"{by['locks'] + by['run_cache']:,} named by {counts['locks']:,} locks and "
        + f"{counts['rows']:,} run-cache entries in {len(scopes)} projects,\n"
        + f"  {by['links']:,} by {counts['links']:,} workspace links, "
        + f"{by['grace']:,} published in the last {grace_hours:.1f}h, "
        + f"{by['hardlinked']:,} still hardlinked elsewhere.",
    ]


def _linked_disclosure(plan: Plan) -> str:
    """What ``--remove-linked`` removes that the default run would keep."""
    linked = plan["linked"]
    return (
        f"--remove-linked: {linked['objects']:,} of the {len(plan['doomed']):,} objects to "
        + f"remove ({gc.format_bytes(linked['bytes'])}) are still hardlinked outside this "
        + "cache; removing them frees nothing until that other entry goes."
    )


def _confirm(plan: Plan, object_root: pathlib.Path, *, assume_yes: bool) -> None:
    """Ask before removing anything.

    Raises:
        click.ClickException: stdin is not a terminal and ``--yes`` was not given.
        click.Abort: the user answered no.
    """
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise click.ClickException(
            f"Refusing to remove {len(plan['doomed']):,} objects "
            + f"({gc.format_bytes(plan['doomed_bytes'])}) from {object_root} without "
            + "confirmation: stdin is not a terminal. Re-run with --yes, or with --dry-run to "
            + "see the plan. Nothing was removed."
        )
    click.confirm(
        f"Remove {len(plan['doomed']):,} objects "
        + f"({gc.format_bytes(plan['doomed_bytes'])}) from {object_root}?",
        abort=True,
        err=True,
    )


@cli_decorators.pivot_command("gc", auto_discover=False)
@click.option("--dry-run", is_flag=True, help="Print the plan and remove nothing.")
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    help="Remove without asking. Required when stdin is not a terminal.",
)
@click.option(
    "--projects",
    "project_paths",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help=(
        "Another project that shares this cache. Repeatable. Each is recorded in the cache's "
        "register, so later runs need no flag."
    ),
)
@click.option(
    "--grace-hours",
    type=click.FloatRange(min=0),
    default=gc.DEFAULT_GRACE_HOURS,
    show_default=True,
    help="Keep every object published in the last N hours, named or not.",
)
@click.option(
    "--remove-linked",
    is_flag=True,
    help="Also remove objects a directory entry outside the cache still hardlinks.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON. Writes one document to stdout; notes go to stderr.",
)
@click.pass_context
def gc_cmd(
    ctx: click.Context,
    dry_run: bool,
    assume_yes: bool,
    project_paths: tuple[pathlib.Path, ...],
    grace_hours: float,
    remove_linked: bool,
    output_json: bool,
) -> None:
    """Remove the cache objects no record names.

    Pivot keeps every object named by a stage lock, a `.pvt` file, a run-cache row, a
    workspace link or a cache ref, in this project and in every project the cache records;
    every object published in the last --grace-hours; and every object a directory entry
    outside the cache still hardlinks. It removes the rest, and first prunes this project's
    run history to core.run_history_retention, as every pipeline run does.

    A cache outside the project root may hold another project's outputs, whose records this
    project cannot read. Pivot records each project that uses a cache under <cache>/projects/
    and refuses to collect an external cache that records none; --projects names them.

    A real run asks before it removes anything, and needs --yes when stdin is not a terminal.
    """
    started = time.monotonic()
    notes_to_stderr = output_json
    quiet = cli_helpers.get_cli_context(ctx)["quiet"]
    project_root = project.get_project_root()
    # Reads the register as earlier runs left it: this project's own entry, written by every
    # other command, would otherwise make an unrecorded cache look recorded.
    cache_dir = config.get_cache_dir(register=False)
    object_root = cache_dir / gc.OBJECTS_DIR
    state_dir = config.get_state_dir()

    scopes, gone = gc.resolve_scopes(cache_dir, project_root, state_dir, project_paths)
    gc.register_project(cache_dir, project_root, state_dir)
    if gone:
        _echo(gc.dropped_note(gone), to_stderr=notes_to_stderr)

    with gc.collection_lock(cache_dir) as locked:
        if not locked:
            if grace_hours == 0:
                raise exceptions.CacheCollectionRefusedError(
                    f"Refusing to collect {cache_dir}: this filesystem refuses flock(), so "
                    + "the cache's locks exclude nothing here and --grace-hours 0 would "
                    + "leave this run with no guard against removing an object another "
                    + "process has just published. Run with a grace period. Nothing was "
                    + "removed."
                )
            _echo(
                f"Warning: {cache_dir} is on a filesystem that refuses flock(), so pivot's "
                + "cache locks exclude nothing here; another process can publish into this "
                + f"cache while it is collected. Objects published in the last {grace_hours:.1f}h "
                + "are kept all the same.",
                to_stderr=True,
            )

        pruned, retention = _prune_run_history(state_dir, dry_run=dry_run)
        if pruned:
            _echo(
                f"{'Would prune' if dry_run else 'Pruned'} {pruned:,} runs beyond the "
                + f"retention of {retention} (core.run_history_retention).",
                to_stderr=notes_to_stderr,
            )

        try:
            roots = gc.collect_roots(scopes, cache_dir)
        except gc.ProjectUnreadableError as e:
            raise exceptions.CacheCollectionRefusedError(
                gc.refusal_unreadable(cache_dir, len(scopes), [(gc.scope_label(e.scope), e.reason)])
            ) from e

        plan = gc.build_plan(cache_dir, roots, grace_hours=grace_hours, remove_linked=remove_linked)

        # A dry run reports the plan as its removal, so its counts are the ones a real run
        # removes and `kept` is what would survive.
        removed = gc.ObjectCount(objects=len(plan["doomed"]), bytes=plan["doomed_bytes"])
        if not dry_run:
            if plan["doomed"]:
                if remove_linked and plan["linked"]["objects"]:
                    _echo(_linked_disclosure(plan), to_stderr=notes_to_stderr)
                _confirm(plan, object_root, assume_yes=assume_yes)
            removed = gc.execute_plan(plan, grace_hours=grace_hours, remove_linked=remove_linked)

    result = gc.result_document(cache_dir, plan, removed, scopes, dry_run=dry_run)
    if output_json:
        click.echo(json.dumps(result, indent=2))
        return
    if quiet:
        return

    lines = _summary_lines(result, roots, scopes, object_root, grace_hours)
    if dry_run:
        lines.append(
            f"Would remove {removed['objects']:,} objects "
            + f"({gc.format_bytes(removed['bytes'])}). Nothing was removed (--dry-run)."
        )
        if remove_linked and plan["linked"]["objects"]:
            lines.append(_linked_disclosure(plan))
    else:
        lines.append(
            f"Removed {removed['objects']:,} objects ({gc.format_bytes(removed['bytes'])}) "
            + f"in {time.monotonic() - started:.1f}s."
        )
    for line in lines:
        click.echo(line)
