from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, override

import rich.markup
import textual.app
import textual.containers
import textual.css.query
import textual.widgets

from pivot.tui.diff_panels import InputDiffPanel, OutputDiffPanel
from pivot.tui.widgets import status
from pivot.tui.widgets.logs import StageLogPanel

if TYPE_CHECKING:
    from pivot.tui.types import ExecutionHistoryEntry, StageInfo

_logger = logging.getLogger(__name__)


class TabbedDetailPanel(textual.containers.Vertical):
    """Tabbed panel showing stage details with Logs, Input, Output tabs."""

    _stage: StageInfo | None
    _history_index: int | None  # None = live view, else index into history deque
    _history_total: int
    _log_panel: StageLogPanel | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage = None
        self._history_index = None
        self._history_total = 0
        self._log_panel = None

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Static(id="detail-header")
        with textual.widgets.TabbedContent(id="detail-tabs"):
            with textual.widgets.TabPane("Logs", id="tab-logs"):
                self._log_panel = StageLogPanel(id="stage-logs")
                yield self._log_panel
            with textual.widgets.TabPane("Input", id="tab-input"):
                yield InputDiffPanel(id="input-panel")
            with textual.widgets.TabPane("Output", id="tab-output"):
                yield OutputDiffPanel(id="output-panel")

    def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
        """Update the displayed stage."""
        self._stage = stage
        self._history_index = None  # Reset to live view
        self._history_total = len(stage.history) if stage else 0
        stage_name = stage.name if stage else None

        self._update_header()

        # Update log panel (takes StageInfo)
        if self._log_panel:
            self._log_panel.set_stage(stage)
        else:
            _logger.debug("_log_panel not initialized during set_stage")

        # Update Input panel - use live snapshot if available
        try:
            input_panel = self.query_one("#input-panel", InputDiffPanel)
            if stage and stage.live_input_snapshot:
                input_panel.set_from_snapshot(stage.live_input_snapshot)
            else:
                input_panel.set_stage(stage_name)
        except textual.css.query.NoMatches:
            _logger.debug("input-panel not found during set_stage")

        # Update Output panel - use live snapshot if available, pass status for empty state
        try:
            output_panel = self.query_one("#output-panel", OutputDiffPanel)
            stage_status = stage.status if stage else None
            if stage and stage.live_output_snapshot:
                output_panel.set_from_snapshot(
                    stage.name, stage.live_output_snapshot, status=stage_status
                )
            else:
                output_panel.set_stage(stage_name, status=stage_status)
        except textual.css.query.NoMatches:
            _logger.debug("output-panel not found during set_stage")

    def set_history_view(
        self, index: int | None, total: int, entry: ExecutionHistoryEntry | None
    ) -> None:  # pragma: no cover
        """Set the history view state. index=None means live view."""
        self._history_index = index
        self._history_total = total
        self._update_header()

        if entry is not None:
            # Update logs panel with historical logs
            if self._log_panel:
                self._log_panel.set_from_history(entry.logs)
            else:
                _logger.debug("log_panel not initialized during set_history_view")

    def _update_header(self) -> None:  # pragma: no cover
        """Update the header with execution indicator."""
        try:
            header = self.query_one("#detail-header", textual.widgets.Static)
        except textual.css.query.NoMatches:
            return

        if self._stage is None:
            header.update("")
            return

        # Build left side: stage name + history navigation
        left_parts = list[str]()
        left_visible_len = 0

        # Stage name
        left_parts.append(f"[bold]{rich.markup.escape(self._stage.name)}[/]")
        left_visible_len += len(self._stage.name)
        left_parts.append("  ")
        left_visible_len += 2

        # History navigation indicator
        total = self._history_total
        if self._history_index is None:
            # Live view - show "Latest" instead of "1/1"
            left_arrow = "← " if total > 0 else ""
            nav_plain = f"{left_arrow}Latest ● LIVE"
            left_parts.append(f"{left_arrow}Latest [green]● LIVE[/]")
            left_visible_len += len(nav_plain)
        else:
            # Historical view - show "Run X of Y"
            current = self._history_index + 1  # 1-based display
            left_arrow = "← " if self._history_index > 0 else ""
            right_arrow = " →"  # Always show - can navigate to live view

            # Get entry for timestamp/duration
            entry = self._get_current_history_entry()
            if entry:
                ts_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
                dur_str = f"{entry.duration:.1f}s" if entry.duration is not None else "0.0s"
                status_icon_hist = status.get_status_icon(entry.status, entry.reason)
                status_icon_plain = status.get_status_icon_plain(entry.status, entry.reason)
                mode_indicator = f"[yellow]◷ {ts_str} ({dur_str})[/] {status_icon_hist}"
                nav_plain = f"{left_arrow}Run {current} of {total + 1} ◷ {ts_str} ({dur_str}) {status_icon_plain}{right_arrow}"
            else:
                mode_indicator = "[yellow]◷ (unknown)[/]"
                nav_plain = f"{left_arrow}Run {current} of {total + 1} ◷ (unknown){right_arrow}"

            left_parts.append(
                f"{left_arrow}Run {current} of {total + 1} {mode_indicator}{right_arrow}"
            )
            left_visible_len += len(nav_plain)

        # Build right side: status icon + label (right-aligned)
        status_icon = status.get_status_icon(self._stage.status, self._stage.reason)
        status_text, status_style = status.get_status_label(self._stage.status, self._stage.reason)
        if status_icon:
            right_part = f"{status_icon} [{status_style}]{status_text}[/]"
            right_visible_len = 2 + len(status_text)  # icon + space + text
        else:
            right_part = f"[{status_style}]{status_text}[/]"
            right_visible_len = len(status_text)

        # Get available width and calculate padding
        left_text = "".join(left_parts)
        available_width = header.size.width if header.size.width > 0 else 80
        padding = max(1, available_width - left_visible_len - right_visible_len - 1)

        header.update(f"{left_text}{' ' * padding}{right_part}")

    def _get_current_history_entry(self) -> ExecutionHistoryEntry | None:
        """Get the currently viewed history entry."""
        if self._stage is None or self._history_index is None:
            return None
        if 0 <= self._history_index < len(self._stage.history):
            return self._stage.history[self._history_index]
        return None

    def refresh_header(self) -> None:  # pragma: no cover
        """Refresh the header display (call when history changes)."""
        if self._stage:
            self._history_total = len(self._stage.history)
        self._update_header()
