from __future__ import annotations

import collections
import logging
import time
from typing import TYPE_CHECKING, override

import rich.markup
import textual.widgets

if TYPE_CHECKING:
    from pivot.tui.types import LogEntry, StageInfo

_logger = logging.getLogger(__name__)


class LogPanel(textual.widgets.RichLog):
    """Panel showing streaming logs."""

    _filter_stage: str | None
    _all_logs: collections.deque[tuple[str, str, bool]]

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(highlight=True, markup=True, id=id, classes=classes)
        self._filter_stage = None
        self._all_logs = collections.deque(maxlen=5000)

    def add_log(self, stage: str, line: str, is_stderr: bool) -> None:  # pragma: no cover
        self._all_logs.append((stage, line, is_stderr))
        if self._filter_stage is None or self._filter_stage == stage:
            self._write_log_line(stage, line, is_stderr)

    def _write_log_line(self, stage: str, line: str, is_stderr: bool) -> None:  # pragma: no cover
        prefix = f"[cyan]\\[{rich.markup.escape(stage)}][/] "
        escaped_line = rich.markup.escape(line)
        if is_stderr:
            self.write(f"{prefix}[red]{escaped_line}[/]")
        else:
            self.write(f"{prefix}{escaped_line}")

    def set_filter(self, stage: str | None) -> None:  # pragma: no cover
        """Filter logs to a specific stage or show all (None)."""
        self._filter_stage = stage
        self.clear()
        for s, line, is_stderr in self._all_logs:
            if stage is None or s == stage:
                self._write_log_line(s, line, is_stderr)


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
