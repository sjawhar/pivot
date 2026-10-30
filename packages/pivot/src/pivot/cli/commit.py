from __future__ import annotations

import click

from pivot import exceptions, names
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.executor import commit


def _split_targets(stages: tuple[str, ...]) -> tuple[list[str] | None, list[str] | None]:
    """The stages to commit first, then the mapped stages (``Pipeline.map``) and instances to
    commit once their producers are committed (``cli_helpers.mapped_name``); None for every one
    of each when no target is given.

    Raises:
        StageNotFoundError: Naming every target that names no stage, mapped stage or instance.
    """
    if not stages:
        return None, None
    pipeline = cli_helpers.get_pipeline()
    registered = dict.fromkeys(pipeline.list_stages())
    templates = pipeline.list_templates()
    first = list[str]()
    mapped = list[str]()
    unknown = list[str]()
    for target in stages:
        name = names.resolve_stage_name(target, registered)
        if name in registered:
            first.append(name)
        elif (mapped_name := cli_helpers.mapped_name(target, templates)) is not None:
            mapped.append(mapped_name)
        else:
            unknown.append(target)
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=[*registered, *templates])
    return first, mapped


def _report(committed: list[str], failed: list[str]) -> None:
    if committed:
        click.echo(f"Committed {len(committed)} stage(s):")
        for stage_name in committed:
            click.echo(f"  {stage_name}")
    if failed:
        click.echo(f"Failed {len(failed)} stage(s):")
        for stage_name in failed:
            click.echo(f"  {stage_name}")
    if not committed and not failed:
        click.echo("Nothing to commit")


@cli_decorators.pivot_command("commit", allow_all=True)
@click.argument("stages", nargs=-1)
@click.pass_context
def commit_command(ctx: click.Context, stages: tuple[str, ...]) -> None:
    """Commit current workspace state for stages.

    Hashes current deps and outputs, writes lock files and cache.
    Without arguments, commits all stale stages. A stage mapped over a directory output
    (`pipeline.map`) is committed as its instances once its producer is current, so its
    producer is committed first; a target may name the mapped stage or one of its instances.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    pipeline = cli_helpers.get_pipeline()

    first, mapped = _split_targets(stages)
    committed, failed = (
        commit.commit_stages(pipeline, first, warn_left_out_symlinks=not quiet)
        if first is None or first
        else (list[str](), list[str]())
    )
    try:
        if (mapped is None or mapped) and pipeline.list_templates():
            # The producers are committed now: expand as a run would, then commit the instances
            # (every stale one, or those the targets name)
            result, _ = cli_helpers.expand_mapped_stages()
            instances = (
                [name for added in result["expanded"].values() for name in added]
                if mapped is None
                else list(cli_helpers.require_expanded_targets(tuple(mapped), result))
            )
            if instances:
                more_committed, more_failed = commit.commit_stages(
                    pipeline,
                    instances,
                    stale_only=mapped is None,
                    warn_left_out_symlinks=not quiet,
                )
                committed.extend(more_committed)
                failed.extend(more_failed)
    except (exceptions.PivotError, click.ClickException):
        # What was committed before the refusal stays committed: say so before the error
        if not quiet and (committed or failed):
            _report(committed, failed)
        raise

    if not quiet:
        _report(committed, failed)

    if failed:
        ctx.exit(1)
