from __future__ import annotations

from typing import TYPE_CHECKING

import textual.widgets

if TYPE_CHECKING:
    from pivot.tui.stats import DebugStats, MessageStats


def _format_message_stats(stats: MessageStats) -> str:
    """Format message statistics for display."""
    return (
        f"{stats['name']}: {stats['messages_received']} msgs ({stats['messages_per_second']:.1f}/s)"
    )


class DebugPanel(textual.widgets.Static):
    """Debug panel showing message statistics and system info."""

    _stats: DebugStats | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stats = None

    def update_stats(self, stats: DebugStats) -> None:  # pragma: no cover
        """Update displayed statistics."""
        self._stats = stats
        self._refresh_display()

    def _refresh_display(self) -> None:  # pragma: no cover
        if self._stats is None:
            self.update("[dim]No stats available[/]")
            return

        msg_stats = self._stats["tui_messages"]
        msg_str = _format_message_stats(msg_stats)

        # Format memory
        mem = self._stats["memory_mb"]
        mem_str = f"{mem:.0f}MB" if mem is not None else "N/A"

        # Format uptime
        uptime = self._stats["uptime_seconds"]
        mins, secs = divmod(int(uptime), 60)
        uptime_str = f"{mins}:{secs:02d}"

        lines = [
            f"[cyan]Messages:[/] {msg_str}",
            (
                f"[cyan]Stats:[/]    Workers: {self._stats['active_workers']}   "
                f"Mem: {mem_str}   Up: {uptime_str}"
            ),
        ]
        self.update("\n".join(lines))
