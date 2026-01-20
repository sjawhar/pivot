from __future__ import annotations

from typing import TYPE_CHECKING

import textual.widgets

if TYPE_CHECKING:
    from pivot.tui.stats import DebugStats, QueueStats


def _format_queue_stats(q: QueueStats | None, label: str) -> str:
    """Format queue statistics for display."""
    if q is None:
        return f"{label}: N/A"
    size = str(q["approximate_size"]) if q["approximate_size"] is not None else "N/A"
    return f"{label}: {size} (peak {q['high_water_mark']})"


class DebugPanel(textual.widgets.Static):
    """Debug panel showing queue statistics and system info."""

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

        tui_q = self._stats["tui_queue"]
        tui_str = _format_queue_stats(tui_q, "TUI")
        out_str = _format_queue_stats(self._stats["output_queue"], "Output")

        # Format message count with K suffix for large numbers
        total_msgs = tui_q["messages_received"]
        msgs_str = f"{total_msgs / 1000:.1f}k" if total_msgs >= 1000 else str(total_msgs)

        # Format memory
        mem = self._stats["memory_mb"]
        mem_str = f"{mem:.0f}MB" if mem is not None else "N/A"

        # Format uptime
        uptime = self._stats["uptime_seconds"]
        mins, secs = divmod(int(uptime), 60)
        uptime_str = f"{mins}:{secs:02d}"

        lines = [
            f"[cyan]Queues:[/]  {tui_str}  {out_str}",
            (
                f"[cyan]Stats:[/]   {msgs_str} msgs @ {tui_q['messages_per_second']:.1f}/s   "
                f"Workers: {self._stats['active_workers']}   Mem: {mem_str}   Up: {uptime_str}"
            ),
        ]
        self.update("\n".join(lines))
