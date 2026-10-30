from __future__ import annotations

import os

import click

from pivot import config, display, project
from pivot.cli import decorators as cli_decorators
from pivot.storage import naming


@cli_decorators.pivot_command("forget-cache", auto_discover=False)
@click.argument("cache")
def forget_cache(cache: str) -> None:
    """Remove CACHE from the caches this view of the project lists.

    Each view of a project (one host and mount namespace) lists, in the project's state
    directory, every cache it has named objects in. Every naming step brings the project's entry
    in each listed cache up to date, and a `pivot gc` checks each of them, so a collection can
    tell the project from a copy of it (a cloned disk). A step that cannot reach a cache this
    view lists warns, naming it, and a collection refuses while that cache cannot be read.

    Remove a cache only when this view can no longer reach it and the project no longer uses it
    (a cache that was deleted, or is on a disk that is gone), and only in the project itself,
    never in a copy of it. A cache this view reaches, whose entry for the project is current, is
    refused. CACHE is the path or the id the warning or the refusal names. Only this view's
    record goes: another view's record of the cache stays, for that view to remove. Nothing in
    any cache changes, and any pivot command that uses the removed cache lists it again.
    """
    state_dir = config.locate_state_dir()
    view_key = naming.view_key_of(*naming.this_view())
    records = naming.cache_records(state_dir)
    own = [record for record in records if record.view_key == view_key]
    matched = _matching(own, cache)
    if not matched:
        listed = sorted({_describe(record) for record in own})
        elsewhere = (
            " Another view of this project (another host or mount namespace) lists it: remove "
            + "it from there."
            if _matching([record for record in records if record.view_key != view_key], cache)
            else ""
        )
        raise click.ClickException(
            f"This view of the project does not list the cache {cache}.{elsewhere} "
            + (
                "It lists: " + "; ".join(listed)
                if listed
                else "It lists no cache: no pivot command here has named an object in one yet."
            )
        )
    try:
        project_id: str | None = naming.read_project_id(state_dir)
    except (OSError, ValueError):
        project_id = None
    if project_id is not None:
        root = project.get_project_root()
        current = [
            record
            for record in matched
            if naming.listed_cache_problem(record, root, state_dir, project_id) is None
        ]
        if current:
            raise click.ClickException(
                f"This view reaches the cache {_describe(current[0])}, and the project's entry "
                + "there is current: every naming step of the project keeps it up to date, and "
                + "a `pivot gc` that checks it finds it so. forget-cache is only for a cache this "
                + "view can no longer reach. Nothing was removed."
            )
    configured = os.path.abspath(config.locate_cache_dir())
    for record in matched:
        record.record.unlink(missing_ok=True)
        path = str(record.cache_dir) if record.cache_dir is not None else ""
        note = (
            " It is this project's configured cache, so its next pivot command lists it again."
            if path == configured
            else ""
        )
        click.echo(
            f"Removed the cache {display.escape_control(path or record.cache_id)} (id "
            + f"{record.cache_id}) from the caches this view of the project lists. Nothing in "
            + "that cache changed. This view's naming steps no longer bring its entry there up "
            + f"to date or warn about it, and a collection no longer checks that entry.{note}"
        )


def _matching(records: list[naming.CacheRecord], cache: str) -> list[naming.CacheRecord]:
    """The records naming ``cache``: by id, or by path (as given, or resolved)."""
    if naming.ENTRY_ID_RE.fullmatch(cache):
        return [record for record in records if record.cache_id == cache]
    wanted = os.path.abspath(cache)
    wanted_real = os.path.realpath(cache)
    return [
        record
        for record in records
        if record.cache_dir is not None
        and (
            os.path.abspath(record.cache_dir) == wanted
            or os.path.realpath(record.cache_dir) == wanted_real
        )
    ]


def _describe(record: naming.CacheRecord) -> str:
    where = (
        display.escape_control(str(record.cache_dir))
        if record.cache_dir is not None
        else f"an unreadable record ({display.escape_control(str(record.record))})"
    )
    since = (
        ""
        if record.unconfirmed_since is None
        else f", unreachable since {display.escape_control(record.unconfirmed_since)}"
    )
    return f"{where} (id {record.cache_id}{since})"
