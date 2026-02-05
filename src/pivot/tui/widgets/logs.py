from __future__ import annotations

import collections
import re
import time
from typing import TYPE_CHECKING, ClassVar, Self, override

import rich.markup
import textual.binding
import textual.message
import textual.widgets

from pivot.tui.types import LogEntry
from pivot.types import StageStatus

if TYPE_CHECKING:
    from pivot.tui.types import StageInfo


class LogSearchEscapePressed(textual.message.Message):
    """Posted when Escape is pressed in the log search input."""


class LogSearchInput(textual.widgets.Input):
    """Search input with Escape key handling."""

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("escape", "escape_pressed", "Cancel", show=False),
    ]

    def action_escape_pressed(self) -> None:
        """Post message when Escape is pressed."""
        self.post_message(LogSearchEscapePressed())


class StageLogPanel(textual.widgets.RichLog):
    """Panel showing timestamped logs for a single stage."""

    _pending_stage: StageInfo | None
    _raw_logs: collections.deque[LogEntry]
    _search_query: str
    _search_pattern: re.Pattern[str] | None  # Cached compiled pattern
    _current_match_idx: int  # Index into match list (not raw_logs)

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(highlight=True, markup=True, id=id, classes=classes)
        self._pending_stage = None
        self._raw_logs = collections.deque[LogEntry](maxlen=1000)
        self._search_query = ""
        self._search_pattern = None
        self._current_match_idx = 0

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

        # Clear display and raw logs (clear() handles both, plus resets search state)
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
        """Add a new log line, with search highlighting if search is active."""
        entry = LogEntry(line, is_stderr, timestamp)
        self._raw_logs.append(entry)

        # Use highlighted write if search is active, otherwise plain write
        if self._search_pattern:
            self._write_line_highlighted(entry, is_current=False)
        else:
            time_str = time.strftime("[%H:%M:%S]", time.localtime(timestamp))
            escaped_line = rich.markup.escape(line)
            if is_stderr:
                self.write(f"[dim]{time_str}[/] [red]{escaped_line}[/]")
            else:
                self.write(f"[dim]{time_str}[/] {escaped_line}")
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
        # Store in raw logs for search functionality
        self._raw_logs.append(LogEntry(line, is_stderr, timestamp))

        time_str = time.strftime("[%H:%M:%S]", time.localtime(timestamp))
        escaped_line = rich.markup.escape(line)
        if is_stderr:
            self.write(f"[dim]{time_str}[/] [red]{escaped_line}[/]")
        else:
            self.write(f"[dim]{time_str}[/] {escaped_line}")

    # =========================================================================
    # Search functionality
    # =========================================================================

    @property
    def is_search_active(self) -> bool:
        """Whether search mode is active."""
        return bool(self._search_query)

    @property
    def match_count(self) -> str:
        """Return 'current/total' format, or empty if no search."""
        if not self._search_query:
            return ""
        matches = self._get_match_indices()
        if not matches:
            return "0/0"
        # Clamp index if matches changed (e.g., deque eviction)
        idx = min(self._current_match_idx, len(matches) - 1)
        return f"{idx + 1}/{len(matches)}"

    def _get_match_indices(self) -> list[int]:
        """Get indices of matching lines in _raw_logs (computed fresh each time)."""
        if not self._search_query:
            return []
        query_lower = self._search_query.lower()
        return [i for i, entry in enumerate(self._raw_logs) if query_lower in entry.line.lower()]

    def apply_search(self, query: str) -> None:
        """Search logs for query and re-render with highlights.

        Args:
            query: Search query (case-insensitive). Empty/whitespace-only clears search.
        """
        query = query.strip()
        if query == self._search_query:
            return  # No change

        self._search_query = query
        self._current_match_idx = 0

        # Cache compiled pattern for highlighting (None if no query)
        self._search_pattern = re.compile(re.escape(query), re.IGNORECASE) if query else None

        self._rerender_logs()

    def _rerender_logs(self) -> None:  # pragma: no cover
        """Re-render all logs, applying search highlighting if active."""
        # Guard against running outside of mounted context (e.g., unit tests)
        if not self.is_attached:
            return

        super().clear()  # Clear display, preserve _raw_logs

        matches = self._get_match_indices()
        # Clamp index if needed
        if matches:
            self._current_match_idx = min(self._current_match_idx, len(matches) - 1)
            current_line_idx = matches[self._current_match_idx]
        else:
            current_line_idx = -1

        for i, entry in enumerate(self._raw_logs):
            is_current = i == current_line_idx
            self._write_line_highlighted(entry, is_current)

        # Scroll to current match
        if current_line_idx >= 0:
            self.scroll_to(y=current_line_idx, animate=False)

        self.refresh()

    def _write_line_highlighted(
        self, entry: LogEntry, is_current: bool = False
    ) -> None:  # pragma: no cover
        """Write a log line with optional search highlighting."""
        time_str = time.strftime("[%H:%M:%S]", time.localtime(entry.timestamp))

        # Apply highlighting if searching, otherwise just escape
        text = (
            self._highlight_matches(entry.line)
            if self._search_pattern
            else rich.markup.escape(entry.line)
        )

        # Build format: stderr gets [red], current match gets background highlight
        if entry.is_stderr:
            if is_current:
                self.write(f"[dim]{time_str}[/] [on yellow][red]{text}[/][/]")
            else:
                self.write(f"[dim]{time_str}[/] [red]{text}[/]")
        elif is_current:
            self.write(f"[dim]{time_str}[/] [on dark_blue]{text}[/]")
        else:
            self.write(f"[dim]{time_str}[/] {text}")

    def _highlight_matches(self, line: str) -> str:
        """Highlight search matches in a line with Rich markup.

        Security: Uses rich.markup.escape() to prevent markup injection.
        Precondition: self._search_pattern is not None.
        """
        assert self._search_pattern is not None

        parts = list[str]()
        last_end = 0

        for match in self._search_pattern.finditer(line):
            # Escape text before match
            if match.start() > last_end:
                parts.append(rich.markup.escape(line[last_end : match.start()]))
            # Highlight the match
            parts.append(f"[bold yellow]{rich.markup.escape(match.group())}[/]")
            last_end = match.end()

        # Escape remaining text after last match (or entire line if no matches)
        if last_end < len(line):
            parts.append(rich.markup.escape(line[last_end:]))

        return "".join(parts)

    def next_match(self) -> None:
        """Move to the next search match (wraps around)."""
        matches = self._get_match_indices()
        if not matches:
            return
        # Clamp first to handle deque eviction (match list may have shrunk)
        self._current_match_idx = min(self._current_match_idx, len(matches) - 1)
        self._current_match_idx = (self._current_match_idx + 1) % len(matches)
        self._rerender_logs()

    def prev_match(self) -> None:
        """Move to the previous search match (wraps around)."""
        matches = self._get_match_indices()
        if not matches:
            return
        # Clamp first to handle deque eviction (match list may have shrunk)
        self._current_match_idx = min(self._current_match_idx, len(matches) - 1)
        self._current_match_idx = (self._current_match_idx - 1) % len(matches)
        self._rerender_logs()

    @override
    def clear(self) -> Self:
        """Clear the log display, raw storage, and search state."""
        self._raw_logs.clear()
        self._search_query = ""
        self._search_pattern = None
        self._current_match_idx = 0
        return super().clear()
