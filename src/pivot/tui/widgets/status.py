from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pivot.types import StageStatus

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pivot.tui.types import StageInfo


def format_elapsed(elapsed: float | None) -> str:
    """Format elapsed time as (M:SS) or empty string if None."""
    if elapsed is None:
        return ""
    mins, secs = divmod(max(0, int(elapsed)), 60)
    return f"({mins}:{secs:02d})"


def get_status_symbol(status: StageStatus) -> tuple[str, str]:
    """Get compact symbol and style for a status (for row display)."""
    match status:
        case StageStatus.READY:
            return ("○", "dim")
        case StageStatus.IN_PROGRESS:
            return ("▶", "blue bold")
        case StageStatus.COMPLETED | StageStatus.RAN:
            return ("●", "green bold")
        case StageStatus.SKIPPED:
            return ("-", "yellow")
        case StageStatus.FAILED:
            return ("!", "red bold")
        case StageStatus.UNKNOWN:
            return ("?", "dim")


def get_status_label(status: StageStatus) -> tuple[str, str]:
    """Get verbose label and style for a status (for detail panel)."""
    match status:
        case StageStatus.READY:
            return ("PENDING", "dim")
        case StageStatus.IN_PROGRESS:
            return ("RUNNING", "blue bold")
        case StageStatus.COMPLETED | StageStatus.RAN:
            return ("SUCCESS", "green bold")
        case StageStatus.SKIPPED:
            return ("SKIPPED", "yellow")
        case StageStatus.FAILED:
            return ("FAILED", "red bold")
        case StageStatus.UNKNOWN:
            return ("UNKNOWN", "dim")


def get_status_icon(status: StageStatus) -> str:
    """Get status icon with Rich markup (for inline display in headers/history)."""
    match status:
        case StageStatus.RAN | StageStatus.COMPLETED:
            return "[green]✓[/]"
        case StageStatus.FAILED:
            return "[red]✗[/]"
        case StageStatus.SKIPPED:
            return "[yellow]⊘[/]"
        case _:
            return ""


class StatusCounts(TypedDict):
    """Counts of stages by status category."""

    running: int
    completed: int
    failed: int


def count_statuses(stages: Iterable[StageInfo]) -> StatusCounts:
    """Count stages by status category."""
    running = 0
    completed = 0
    failed = 0
    for s in stages:
        match s.status:
            case StageStatus.IN_PROGRESS:
                running += 1
            case StageStatus.COMPLETED | StageStatus.RAN:
                completed += 1
            case StageStatus.FAILED:
                failed += 1
            case StageStatus.READY | StageStatus.SKIPPED | StageStatus.UNKNOWN:
                pass  # Not counted
    return StatusCounts(running=running, completed=completed, failed=failed)
