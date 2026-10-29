from __future__ import annotations

import asyncio
import functools
import logging
import pathlib
from typing import TYPE_CHECKING

import click

from pivot import display
from pivot.cli import decorators as cli_decorators

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pivot import import_artifact
    from pivot.storage import track


async def _batch_check(
    pairs: list[tuple[pathlib.Path, track.PvtData]],
) -> list[import_artifact.UpdateCheck | BaseException]:
    from pivot import import_artifact as ia

    return await asyncio.gather(
        *[ia.check_for_update(pvt_path, pvt_data) for pvt_path, pvt_data in pairs],
        return_exceptions=True,
    )


async def _batch_update(
    pairs: list[tuple[pathlib.Path, track.PvtData]],
    rev: str | None,
) -> list[import_artifact.UpdateResult | BaseException]:
    from pivot import import_artifact as ia

    return await asyncio.gather(
        *[ia.update_import(pvt_path, new_rev=rev) for pvt_path, _ in pairs],
        return_exceptions=True,
    )


@cli_decorators.pivot_command(auto_discover=False)
@click.argument("targets", nargs=-1)
@click.option("--rev", default=None, help="Override git ref for update")
@click.option("--dry-run", is_flag=True, help="Show what would change without modifying")
def update(targets: tuple[str, ...], rev: str | None, dry_run: bool) -> None:
    """Update imported artifacts from their source repos.

    If no TARGETS specified, updates all imports found in the project.
    """
    from pivot import config, project
    from pivot.storage import member_skip, track

    root = project.get_project_root()

    if targets:
        pairs = list[tuple[pathlib.Path, track.PvtData]]()
        for t in targets:
            t_path = pathlib.Path(t)
            if t_path.suffix == ".pvt":
                pvt_path = project.normalize_path(t_path, base=root)
            else:
                pvt_path = track.get_pvt_path(project.normalize_path(t_path, base=root))
            pvt_data = track.read_pvt_file(pvt_path)
            if pvt_data is None:
                click.echo(f"Skipping invalid .pvt: {display.escape_control(str(pvt_path))}")
                continue
            if not track.is_import(pvt_data):
                click.echo(f"Skipping non-import: {display.escape_control(str(pvt_path))}")
                continue
            pairs.append((pvt_path, pvt_data))
    else:
        # A stage's own .pvt member of its output is no import. The command loads no pipeline,
        # so the directory outputs the stages' locks record decide; it reads no config until an
        # import .pvt is found.
        @functools.cache
        def own_member_skip() -> track.PvtSkip:
            return member_skip.own_member_skip(
                root, {}, config.locate_state_dir(), config.locate_cache_dir() / "files"
            )

        def skip(path: pathlib.Path, data: track.PvtData) -> bool:
            return own_member_skip()(path, data)

        imports = track.discover_import_pvt_files(root, skip=skip)
        pairs = list[tuple[pathlib.Path, track.PvtData]]()
        for data_path_str, pvt_data in imports.items():
            pvt_path = track.get_pvt_path(pathlib.Path(data_path_str))
            pairs.append((pvt_path, pvt_data))

    if not pairs:
        click.echo("No imports found to update.")
        return

    errors = 0
    if dry_run:
        checks = asyncio.run(_batch_check(pairs))
        for (pvt_path, _), check_or_exc in zip(pairs, checks, strict=True):
            shown = display.escape_control(str(pvt_path))
            if isinstance(check_or_exc, BaseException):
                errors += 1
                _report_error(pvt_path, check_or_exc)
            else:
                if check_or_exc["available"]:
                    current = display.escape_control(check_or_exc["current_rev"][:8])
                    latest = display.escape_control(check_or_exc["latest_rev"][:8])
                    click.echo(f"Update available: {shown} ({current} → {latest})")
                else:
                    click.echo(f"Up to date: {shown}")
    else:
        results = asyncio.run(_batch_update(pairs, rev))
        for (pvt_path, _), result_or_exc in zip(pairs, results, strict=True):
            if isinstance(result_or_exc, BaseException):
                errors += 1
                _report_error(pvt_path, result_or_exc)
            else:
                shown = display.escape_control(result_or_exc["path"])
                old_rev = display.escape_control(result_or_exc["old_rev"][:8])
                new_rev = display.escape_control(result_or_exc["new_rev"][:8])
                if result_or_exc["downloaded"]:
                    click.echo(f"Updated: {shown} ({old_rev} → {new_rev})")
                elif result_or_exc["metadata_updated"]:
                    click.echo(f"Metadata updated: {shown} ({old_rev} → {new_rev})")
                else:
                    click.echo(f"Up to date: {shown}")

    if errors:
        raise SystemExit(1)


def _report_error(pvt_path: pathlib.Path, exc: BaseException) -> None:
    from pivot import exceptions

    shown = display.escape_control(str(pvt_path))
    if not isinstance(exc, exceptions.PivotError):
        # Unexpected: --verbose shows the traceback; the message is one line naming the problem.
        logger.debug("Unexpected error updating %s", shown, exc_info=exc)
    message = f"Error updating {shown}: {exceptions.describe_error(exc)}"
    click.echo(display.safe_message(message), err=True)
