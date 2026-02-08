from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar, override

import rich.markup
import textual.app
import textual.binding
import textual.containers
import textual.css.query
import textual.screen
import textual.widgets

from pivot_tui.widgets import status

if TYPE_CHECKING:
    import collections

    from pivot_tui.types import ExecutionHistoryEntry

_logger = logging.getLogger(__name__)


class HistoryListScreen(textual.screen.ModalScreen[int | None]):
    """Modal screen for selecting a history entry. Returns index or None for live."""

    CSS_PATH: ClassVar[str] = "../styles/modal.tcss"  # pyright: ignore[reportIncompatibleVariableOverride]

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("j", "select_next", "Down"),
        textual.binding.Binding("k", "select_prev", "Up"),
        textual.binding.Binding("down", "select_next", "Down", show=False),
        textual.binding.Binding("up", "select_prev", "Up", show=False),
        textual.binding.Binding("enter", "confirm", "Select"),
        textual.binding.Binding("escape", "cancel", "Cancel"),
        textual.binding.Binding("G", "go_live", "Live"),
    ]

    _history: collections.deque[ExecutionHistoryEntry]
    _selected_idx: int
    _current_view_idx: int | None
    _stage_name: str

    def __init__(
        self,
        stage_name: str,
        history: collections.deque[ExecutionHistoryEntry],
        current_idx: int | None,
    ) -> None:
        super().__init__()
        self._stage_name = stage_name
        self._history = history
        # Start selection at current view, or most recent if live view
        if current_idx is not None:
            self._selected_idx = current_idx
        else:
            self._selected_idx = max(0, len(history) - 1)
        self._current_view_idx = current_idx

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        with textual.containers.Container(id="history-dialog"):
            yield textual.widgets.Static(
                f"[bold]History: {rich.markup.escape(self._stage_name)}[/]", id="history-title"
            )
            yield textual.widgets.Static(self._render_table(), id="history-table")
            yield textual.widgets.Static("[j/k] navigate  [Enter] select  [G] live  [Esc] cancel")

    def _render_table(self) -> str:  # pragma: no cover
        """Render history entries as Rich markup table."""
        if not self._history:
            return "[dim]No history entries[/]"

        lines = list[str]()
        # Header
        lines.append("[dim]  #  │ Time     │ Duration │ Status  │ Reason[/]")
        lines.append("[dim]─────┼──────────┼──────────┼─────────┼────────────────────[/]")

        for idx, entry in enumerate(self._history):
            # Format time as HH:MM:SS
            time_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))

            # Format duration
            duration_str = f"{entry.duration:6.1f}s" if entry.duration is not None else "     - "

            # Format status using centralized helper
            status_str = status.get_status_table_cell(entry.status, entry.reason)

            # Truncate reason
            reason = entry.reason[:20] if len(entry.reason) > 20 else entry.reason
            reason_escaped = rich.markup.escape(reason)

            # Selection indicator
            is_selected = idx == self._selected_idx
            is_current = idx == self._current_view_idx
            if is_selected:
                prefix = "[reverse]▸"
                suffix = "[/]"
            elif is_current:
                prefix = " "
                suffix = " [dim]◂[/]"
            else:
                prefix = " "
                suffix = ""

            # Build row
            num_str = f"{idx + 1:3}"
            row = f"{prefix}{num_str} │ {time_str} │ {duration_str} │ {status_str} │ {reason_escaped}{suffix}"
            lines.append(row)

        return "\n".join(lines)

    def _update_table(self) -> None:  # pragma: no cover
        """Update the table display after selection change."""
        try:
            table = self.query_one("#history-table", textual.widgets.Static)
            table.update(self._render_table())
        except textual.css.query.NoMatches:
            _logger.debug("History table widget not present, skipping update")

    def action_select_next(self) -> None:  # pragma: no cover
        """Move selection to next (newer) entry."""
        if self._history and self._selected_idx < len(self._history) - 1:
            self._selected_idx += 1
            self._update_table()

    def action_select_prev(self) -> None:  # pragma: no cover
        """Move selection to previous (older) entry."""
        if self._history and self._selected_idx > 0:
            self._selected_idx -= 1
            self._update_table()

    def action_confirm(self) -> None:  # pragma: no cover
        """Confirm selection and return the selected index."""
        self.dismiss(self._selected_idx)

    def action_cancel(self) -> None:  # pragma: no cover
        """Cancel and return the original view index."""
        self.dismiss(self._current_view_idx)

    def action_go_live(self) -> None:  # pragma: no cover
        """Go to live view."""
        self.dismiss(None)
