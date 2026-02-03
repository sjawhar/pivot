from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

from pivot.types import DisplayCategory, StageStatus, categorize_stage_result

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pivot.tui.types import StageInfo


def format_elapsed(elapsed: float | None) -> str:
    """Format elapsed time as (M:SS) or empty string if None.

    Handles None, negative values, infinity, and NaN gracefully.
    """
    if elapsed is None or math.isnan(elapsed) or math.isinf(elapsed):
        return ""
    mins, secs = divmod(max(0, int(elapsed)), 60)
    return f"({mins}:{secs:02d})"


def get_status_symbol(status: StageStatus, reason: str = "") -> tuple[str, str]:
    """Get compact symbol and style for a status (for row display)."""
    category = categorize_stage_result(status, reason)
    match category:
        case DisplayCategory.PENDING:
            return ("○", "dim")
        case DisplayCategory.RUNNING:
            return ("▶", "blue bold")
        case DisplayCategory.SUCCESS:
            return ("●", "green bold")
        case DisplayCategory.CACHED:
            return ("↺", "yellow")
        case DisplayCategory.BLOCKED:
            return ("◇", "red")
        case DisplayCategory.CANCELLED:
            return ("!", "yellow dim")
        case DisplayCategory.FAILED:
            return ("✗", "red bold")
        case DisplayCategory.UNKNOWN:
            return ("?", "dim")


def get_status_label(status: StageStatus, reason: str = "") -> tuple[str, str]:
    """Get verbose label and style for a status (for detail panel)."""
    category = categorize_stage_result(status, reason)
    match category:
        case DisplayCategory.PENDING:
            return ("PENDING", "dim")
        case DisplayCategory.RUNNING:
            return ("RUNNING", "blue bold")
        case DisplayCategory.SUCCESS:
            return ("SUCCESS", "green bold")
        case DisplayCategory.CACHED:
            return ("CACHED", "yellow")
        case DisplayCategory.BLOCKED:
            return ("BLOCKED", "red")
        case DisplayCategory.CANCELLED:
            return ("CANCELLED", "yellow dim")
        case DisplayCategory.FAILED:
            return ("FAILED", "red bold")
        case DisplayCategory.UNKNOWN:
            return ("UNKNOWN", "dim")


def get_status_icon(status: StageStatus, reason: str = "") -> str:
    """Get status icon with Rich markup (for inline display in headers/history)."""
    category = categorize_stage_result(status, reason)
    match category:
        case DisplayCategory.SUCCESS:
            return "[green]✓[/]"
        case DisplayCategory.FAILED:
            return "[red]✗[/]"
        case DisplayCategory.CACHED:
            return "[yellow]↺[/]"
        case DisplayCategory.BLOCKED:
            return "[red]◇[/]"
        case DisplayCategory.CANCELLED:
            return "[yellow dim]![/]"
        case DisplayCategory.PENDING | DisplayCategory.RUNNING | DisplayCategory.UNKNOWN:
            return ""


# Status icon plain text (no Rich markup) for length calculations
_STATUS_ICON_PLAIN: dict[DisplayCategory, str] = {
    DisplayCategory.SUCCESS: "✓",
    DisplayCategory.FAILED: "✗",
    DisplayCategory.CACHED: "↺",
    DisplayCategory.BLOCKED: "◇",
    DisplayCategory.CANCELLED: "!",
}


def get_status_icon_plain(status: StageStatus, reason: str = "") -> str:
    """Get status icon without Rich markup (for length calculations)."""
    category = categorize_stage_result(status, reason)
    return _STATUS_ICON_PLAIN.get(category, "")


def get_status_table_cell(status: StageStatus, reason: str) -> str:
    """Get fixed-width status cell for table display (8 chars visible)."""
    category = categorize_stage_result(status, reason)
    match category:
        case DisplayCategory.SUCCESS:
            return "[green]✓ ran[/]  "
        case DisplayCategory.FAILED:
            return "[red]✗ fail[/] "
        case DisplayCategory.CACHED:
            return "[yellow]↺ cache[/]"
        case DisplayCategory.BLOCKED:
            return "[red]◇ block[/]"
        case DisplayCategory.CANCELLED:
            return "[yellow dim]! cncl[/] "
        case DisplayCategory.PENDING:
            return "[dim]PENDING[/] "
        case DisplayCategory.RUNNING:
            return "[blue bold]RUNNING[/] "
        case DisplayCategory.UNKNOWN:
            return "[dim]UNKNOWN[/] "


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
