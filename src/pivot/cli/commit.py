from __future__ import annotations

import click

from pivot import project
from pivot.cli import decorators as cli_decorators
from pivot.executor import commit
from pivot.storage import lock, project_lock


@cli_decorators.pivot_command("commit")
@click.option("--list", "list_pending", is_flag=True, help="List pending stages without committing")
@click.option("--discard", is_flag=True, help="Discard pending changes without committing")
def commit_command(list_pending: bool, discard: bool) -> None:
    """Commit pending locks from --no-commit runs to production.

    After running with --no-commit, use this command to finalize your changes.
    """
    project_root = project.get_project_root()

    # --list is read-only, doesn't need lock
    if list_pending:
        pending_stages = lock.list_pending_stages(project_root)
        if not pending_stages:
            click.echo("No pending stages")
        else:
            click.echo("Pending stages:")
            for stage_name in pending_stages:
                click.echo(f"  {stage_name}")
        return

    # Acquire lock before reading/modifying pending state to prevent races
    with project_lock.pending_state_lock():
        pending_stages = lock.list_pending_stages(project_root)

        if discard:
            if not pending_stages:
                click.echo("No pending stages to discard")
                return
            discarded = commit.discard_pending()
            click.echo(f"Discarded {len(discarded)} pending stage(s)")
            return

        if not pending_stages:
            click.echo("Nothing to commit")
            return

        committed = commit.commit_pending()
        click.echo(f"Committed {len(committed)} stage(s):")
        for stage_name in committed:
            click.echo(f"  {stage_name}")
