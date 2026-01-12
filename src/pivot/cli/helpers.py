from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.cli import CliContext

from pivot import exceptions, registry


def emit_jsonl(event: object) -> None:
    """Emit a single JSONL event to stdout with flush for streaming."""
    print(json.dumps(event), flush=True)


def get_cli_context(ctx: click.Context) -> CliContext:
    """Get CLI context with defaults if not set."""
    if ctx.obj:
        return ctx.obj
    # Return dict matching CliContext structure to avoid circular import
    return {"verbose": False, "quiet": False}


def stages_to_list(stages: tuple[str, ...]) -> list[str] | None:
    """Convert Click's stage tuple to list or None if empty."""
    return list(stages) if stages else None


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
