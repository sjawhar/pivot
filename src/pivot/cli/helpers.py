from __future__ import annotations

import enum
import json
from typing import TYPE_CHECKING, Any, cast

import click

if TYPE_CHECKING:
    from collections.abc import Callable

    from pivot.cli import CliContext

from pivot import exceptions, registry


def _json_default(obj: Any) -> Any:  # noqa: ANN401 - json.dumps default requires Any
    """Handle non-standard JSON types in JSONL output."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, set):
        # Convert to sorted list for JSON serialization (deterministic output)
        # Cast needed because isinstance(obj, set) doesn't narrow the element type
        return sorted(cast("set[Any]", obj), key=str)
    # Let json.dumps raise TypeError for truly unserializable types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def emit_jsonl(event: object) -> None:
    """Emit a single JSONL event to stdout with flush for streaming.

    Handles StrEnum values automatically. Sets are converted to sorted lists.
    Other non-serializable types will raise TypeError.
    """
    print(json.dumps(event, default=_json_default), flush=True)


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
