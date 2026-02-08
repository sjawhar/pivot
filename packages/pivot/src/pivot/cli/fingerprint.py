from __future__ import annotations

import click

from pivot.cli import decorators as cli_decorators


@click.group()
def fingerprint() -> None:
    """Manage function fingerprinting cache."""


@fingerprint.command("reset")
@cli_decorators.with_error_handling
def reset() -> None:
    """Reset cached function fingerprints.

    Use after encountering stale cache issues or when troubleshooting
    unexpected stage re-runs.
    """
    from pivot.config import io
    from pivot.storage import state

    db_path = io.get_state_db_path()
    with state.StateDB(db_path, readonly=False) as db:
        count = db.clear_ast_hashes()

    click.echo(f"Cleared {count} cached fingerprint entries.")
