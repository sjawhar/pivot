from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable

from pivot import exceptions, registry


def validate_stages_exist(stages: list[str] | None) -> None:
    """Validate that specified stages exist in the registry."""
    if not stages:
        return
    graph = registry.REGISTRY.build_dag(validate=True)
    registered = set(graph.nodes())
    unknown = [s for s in stages if s not in registered]
    if unknown:
        raise exceptions.StageNotFoundError(f"Unknown stage(s): {', '.join(unknown)}")


def make_progress_callback(action: str) -> Callable[[int], None]:
    """Create a progress callback for file transfer operations."""

    def callback(completed: int) -> None:
        click.echo(f"  {action} {completed} files...", nl=False)
        click.echo("\r", nl=False)

    return callback


def print_transfer_errors(errors: list[str], max_shown: int = 5) -> None:
    """Print transfer errors with truncation for long lists."""
    if not errors:
        return
    for err in errors[:max_shown]:
        click.echo(f"  Error: {err}", err=True)
    if len(errors) > max_shown:
        click.echo(f"  ... and {len(errors) - max_shown} more errors", err=True)
