from __future__ import annotations

import os

import click

from pivot import config, display
from pivot.cli import decorators as cli_decorators
from pivot.storage import naming


@cli_decorators.pivot_command("forget-cache", auto_discover=False)
@click.argument("cache")
def forget_cache(cache: str) -> None:
    """Remove CACHE from the list of caches this project uses.

    Each project lists, in its state directory, every cache it has named objects in. A
    `pivot gc` of any of those caches reads the project's entry in every other one it lists,
    and refuses when one cannot be read, so it can tell the project from a copy of it (a cloned
    disk). Remove a cache from the list when the project no longer uses it, for example a cache
    that was deleted or is on a disk that is gone. CACHE is the path or the id a refusal names.

    This changes only the project's list. It deletes nothing from any cache, and a collection
    of the removed cache still reads this project's entry there. Any pivot command that uses the
    removed cache lists it again.
    """
    state_dir = config.locate_state_dir()
    records = naming.cache_records(state_dir)
    if naming.ENTRY_ID_RE.fullmatch(cache):
        matched = [record for record in records if record.cache_id == cache]
    else:
        wanted = os.path.abspath(cache)
        wanted_real = os.path.realpath(cache)
        matched = [
            record
            for record in records
            if record.cache_dir is not None
            and (
                os.path.abspath(record.cache_dir) == wanted
                or os.path.realpath(record.cache_dir) == wanted_real
            )
        ]
    if not matched:
        listed = sorted({_describe(record) for record in records})
        raise click.ClickException(
            f"This project does not list the cache {cache}. "
            + (
                "It lists: " + "; ".join(listed)
                if listed
                else "It lists no cache: no pivot command has named an object in one yet."
            )
        )
    configured = os.path.abspath(config.locate_cache_dir())
    for cache_id in sorted({record.cache_id for record in matched}):
        removed = [record for record in matched if record.cache_id == cache_id]
        for record in removed:
            record.record.unlink(missing_ok=True)
        paths = sorted({str(record.cache_dir) for record in removed if record.cache_dir})
        note = (
            " It is this project's configured cache, so its next pivot command lists it again."
            if configured in paths
            else ""
        )
        click.echo(
            f"Removed the cache {display.escape_control(', '.join(paths) or cache_id)} (id "
            + f"{cache_id}) from the caches this project uses. Nothing in that cache changed. "
            + "A collection there still reads this project's entry, and a collection of another "
            + f"cache no longer checks it.{note}"
        )


def _describe(record: naming.CacheRecord) -> str:
    where = (
        display.escape_control(str(record.cache_dir))
        if record.cache_dir is not None
        else f"an unreadable record ({display.escape_control(str(record.record))})"
    )
    return f"{where} (id {record.cache_id})"
