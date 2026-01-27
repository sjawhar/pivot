from __future__ import annotations

import time
from typing import TYPE_CHECKING, override

import rich.markup
import textual.widgets

from pivot.types import StageStatus

if TYPE_CHECKING:
    from pivot.tui.types import LogEntry, StageInfo


class StageLogPanel(textual.widgets.RichLog):
    """Panel showing timestamped logs for a single stage."""

    _pending_stage: StageInfo | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(highlight=True, markup=True, id=id, classes=classes)
        self._pending_stage = None

    @override
    def on_mount(self) -> None:  # pragma: no cover
        """Write initial content on mount."""
        self.write("[dim]Initializing...[/]")

    def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
        """Display all logs for the given stage."""
        self._pending_stage = stage
        # Defer the actual write to after refresh cycle
        self.call_after_refresh(self._do_set_stage)

    def _do_set_stage(self) -> None:  # pragma: no cover
        """Actually write the stage content (called after refresh)."""
        stage = self._pending_stage

        self.clear()
        if stage is None:
            self.write("[dim]No stage selected[/]")
        elif stage.logs:
            for line, is_stderr, timestamp in stage.logs:
                self._write_line(line, is_stderr, timestamp)
        else:
            # Show status-appropriate message when no logs
            match stage.status:
                case StageStatus.SKIPPED:
                    self.write("[dim]Stage was skipped[/]")
                case StageStatus.COMPLETED | StageStatus.RAN | StageStatus.FAILED:
                    self.write("[dim]No logs recorded[/]")
                case _:
                    self.write(f"[dim]No logs yet for {rich.markup.escape(stage.name)}[/]")

    def add_log(self, line: str, is_stderr: bool, timestamp: float) -> None:  # pragma: no cover
        """Add a new log line."""
        self._write_line(line, is_stderr, timestamp)
        self.refresh()

    def set_from_history(self, logs: list[LogEntry]) -> None:  # pragma: no cover
        """Display logs from a historical execution entry."""
        self.clear()
        if logs:
            for line, is_stderr, timestamp in logs:
                self._write_line(line, is_stderr, timestamp)
        else:
            self.write("[dim]No logs recorded for this execution[/]")

    def _write_line(self, line: str, is_stderr: bool, timestamp: float) -> None:  # pragma: no cover
        time_str = time.strftime("[%H:%M:%S]", time.localtime(timestamp))
        escaped_line = rich.markup.escape(line)
        if is_stderr:
            self.write(f"[dim]{time_str}[/] [red]{escaped_line}[/]")
        else:
            self.write(f"[dim]{time_str}[/] {escaped_line}")
