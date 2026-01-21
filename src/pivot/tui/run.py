from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import dataclasses
import json
import logging
import os
import queue
import threading
import time
from typing import IO, TYPE_CHECKING, ClassVar, Literal, TypeVar, cast, final, override

import filelock
import rich.markup
import textual.app
import textual.binding
import textual.containers
import textual.css.query
import textual.message
import textual.screen
import textual.timer
import textual.widgets

from pivot import explain, parameters, project
from pivot.executor import ExecutionSummary
from pivot.executor import commit as commit_mod
from pivot.registry import REGISTRY
from pivot.storage import lock, project_lock
from pivot.tui import agent_server, diff_panels
from pivot.tui.diff_panels import InputDiffPanel, OutputDiffPanel
from pivot.tui.stats import DebugStats, QueueStats, QueueStatsTracker, get_memory_mb
from pivot.types import (
    DisplayMode,
    OutputChange,
    StageExplanation,
    StageStatus,
    TuiLogMessage,
    TuiMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable
    from pathlib import Path
    from typing import Protocol

    from pivot.types import OutputMessage
    from pivot.watch.engine import WatchEngine

    class WatchEngineProtocol(Protocol):
        """Protocol for WatchEngine to avoid circular imports."""

        def run(
            self,
            tui_queue: mp.Queue[TuiMessage] | None = None,
            output_queue: mp.Queue[OutputMessage] | None = None,
        ) -> None: ...
        def shutdown(self) -> None: ...
        def toggle_keep_going(self) -> bool: ...
        @property
        def keep_going(self) -> bool: ...


def _format_elapsed(elapsed: float | None) -> str:
    """Format elapsed time as (M:SS) or empty string if None."""
    if elapsed is None:
        return ""
    mins, secs = divmod(int(elapsed), 60)
    return f"({mins}:{secs:02d})"


# Status display with colors
STATUS_STYLES: dict[StageStatus, tuple[str, str]] = {
    StageStatus.READY: ("PENDING", "dim"),
    StageStatus.IN_PROGRESS: ("RUNNING", "blue bold"),
    StageStatus.COMPLETED: ("SUCCESS", "green bold"),
    StageStatus.RAN: ("SUCCESS", "green bold"),
    StageStatus.SKIPPED: ("SKIP", "yellow"),
    StageStatus.FAILED: ("FAILED", "red bold"),
    StageStatus.UNKNOWN: ("UNKNOWN", "dim"),
}


@dataclasses.dataclass
class ExecutionHistoryEntry:
    """Snapshot of a single stage execution for history navigation."""

    run_id: str
    stage_name: str
    timestamp: float
    duration: float | None
    status: StageStatus
    reason: str
    logs: list[tuple[str, bool, float]]
    input_snapshot: StageExplanation | None
    output_snapshot: list[OutputChange] | None


@dataclasses.dataclass
class _PendingHistoryState:
    """Temporary state for a stage execution in progress, before finalization."""

    run_id: str
    timestamp: float
    # Bounded deque to prevent memory growth in watch mode with verbose stages
    logs: collections.deque[tuple[str, bool, float]] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=500)
    )
    input_snapshot: StageExplanation | None = None


@dataclasses.dataclass
class StageInfo:
    """Mutable state for a single stage."""

    name: str
    index: int
    total: int
    status: StageStatus = StageStatus.READY
    reason: str = ""
    elapsed: float | None = None
    logs: collections.deque[tuple[str, bool, float]] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )
    history: collections.deque[ExecutionHistoryEntry] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=50)
    )


class TuiUpdate(textual.message.Message):
    """Custom message for executor updates."""

    msg: TuiLogMessage | TuiStatusMessage | TuiWatchMessage | TuiReloadMessage

    def __init__(
        self, msg: TuiLogMessage | TuiStatusMessage | TuiWatchMessage | TuiReloadMessage
    ) -> None:
        self.msg = msg
        super().__init__()


class ExecutorComplete(textual.message.Message):
    """Signal that executor has finished."""

    results: dict[str, ExecutionSummary]
    error: Exception | None

    def __init__(self, results: dict[str, ExecutionSummary], error: Exception | None) -> None:
        self.results = results
        self.error = error
        super().__init__()


class StageRow(textual.widgets.Static):
    """Single stage row showing index, name, status, and reason."""

    _info: StageInfo

    def __init__(self, info: StageInfo) -> None:
        super().__init__()
        self._info = info

    def update_display(self) -> None:  # pragma: no cover
        label, style = STATUS_STYLES.get(self._info.status, ("?", "dim"))
        index_str = f"[{self._info.index}/{self._info.total}]"
        elapsed_str = _format_elapsed(self._info.elapsed)
        if elapsed_str:
            elapsed_str = f" {elapsed_str}"
        reason_str = f"  ({rich.markup.escape(self._info.reason)})" if self._info.reason else ""
        name_escaped = rich.markup.escape(self._info.name)
        text = f"{index_str} {name_escaped:<20} [{style}]{label}[/]{elapsed_str}{reason_str}"
        self.update(text)

    def on_mount(self) -> None:  # pragma: no cover
        self.update_display()


class StageListPanel(textual.widgets.Static):
    """Panel showing all stages with their status."""

    _stages: list[StageInfo]
    _rows: dict[str, StageRow]

    def __init__(
        self,
        stages: list[StageInfo],
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._stages = stages
        self._rows = {}

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Static("[bold]Stages[/]", classes="section-header")
        for stage in self._stages:
            row = StageRow(stage)
            self._rows[stage.name] = row
            yield row

    def update_stage(self, name: str) -> None:  # pragma: no cover
        if name in self._rows:
            self._rows[name].update_display()

    def rebuild(self, stages: list[StageInfo]) -> None:  # pragma: no cover
        """Rebuild panel with new stage list."""
        self._stages = stages
        self._rows.clear()
        self.refresh(recompose=True)


class DetailPanel(textual.widgets.Static):
    """Panel showing details of selected stage."""

    _stage: StageInfo | None

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage = None

    def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
        self._stage = stage
        self._update_display()

    def _update_display(self) -> None:  # pragma: no cover
        if self._stage is None:
            self.update("[dim]No stage selected[/]")
            return

        label, style = STATUS_STYLES.get(self._stage.status, ("?", "dim"))
        elapsed_str = _format_elapsed(self._stage.elapsed)
        if elapsed_str:
            elapsed_str = f" {elapsed_str} elapsed"

        lines = [
            f"[bold]Stage:[/] {rich.markup.escape(self._stage.name)}",
            f"[bold]Status:[/] [{style}]{label}[/]{elapsed_str}",
        ]
        if self._stage.reason:
            lines.append(f"[bold]Reason:[/] {rich.markup.escape(self._stage.reason)}")

        self.update("\n".join(lines))


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

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(highlight=True, markup=True, id=id, classes=classes)

    def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
        """Display all logs for the given stage."""
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

    def set_from_history(self, logs: list[tuple[str, bool, float]]) -> None:  # pragma: no cover
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


class TabbedDetailPanel(textual.containers.Vertical):
    """Tabbed panel showing stage details with Logs, Input, Output tabs."""

    _stage: StageInfo | None
    _history_index: int | None  # None = live view, else index into history deque
    _history_total: int

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage = None
        self._history_index = None
        self._history_total = 0

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Static(id="detail-header")
        with textual.widgets.TabbedContent(id="detail-tabs"):
            with textual.widgets.TabPane("Logs", id="tab-logs"):
                yield StageLogPanel(id="stage-logs")
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
        try:
            self.query_one("#stage-logs", StageLogPanel).set_stage(stage)
        except textual.css.query.NoMatches:
            _logger.debug("stage-logs not found during set_stage")

        # Update diff panels (share same interface - take stage name string)
        diff_panels: list[tuple[str, type[InputDiffPanel] | type[OutputDiffPanel]]] = [
            ("#input-panel", InputDiffPanel),
            ("#output-panel", OutputDiffPanel),
        ]
        for panel_id, panel_cls in diff_panels:
            try:
                self.query_one(panel_id, panel_cls).set_stage(stage_name)
            except textual.css.query.NoMatches:
                _logger.debug(f"{panel_id} not found during set_stage")

    def set_history_view(
        self, index: int | None, total: int, entry: ExecutionHistoryEntry | None
    ) -> None:  # pragma: no cover
        """Set the history view state. index=None means live view."""
        self._history_index = index
        self._history_total = total
        self._update_header()

        if entry is not None:
            # Update logs panel with historical logs
            try:
                log_panel = self.query_one("#stage-logs", StageLogPanel)
                log_panel.set_from_history(entry.logs)
            except textual.css.query.NoMatches:
                _logger.debug("stage-logs not found during set_history_view")

    def _update_header(self) -> None:  # pragma: no cover
        """Update the header with execution indicator."""
        try:
            header = self.query_one("#detail-header", textual.widgets.Static)
        except textual.css.query.NoMatches:
            return

        if self._stage is None:
            header.update("")
            return

        # Build header components
        parts = list[str]()

        # Stage name
        parts.append(f"[bold]{rich.markup.escape(self._stage.name)}[/]")

        # Spacer
        parts.append("  ")

        # History navigation indicator
        total = self._history_total
        if self._history_index is None:
            # Live view
            current = total + 1  # Live is "after" all history entries
            # Show left arrow if history exists
            left_arrow = "← " if total > 0 else ""
            mode_indicator = "[green]● LIVE[/]"
            parts.append(f"{left_arrow}[{current}/{current}] {mode_indicator}")
        else:
            # Historical view
            current = self._history_index + 1  # 1-based display
            left_arrow = "← " if self._history_index > 0 else ""
            right_arrow = " →"  # Always show - can navigate to live view

            # Get entry for timestamp/duration
            entry = self._get_current_history_entry()
            if entry:
                ts_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
                dur_str = f"{entry.duration:.1f}s" if entry.duration is not None else "0.0s"
                status_icon = self._get_status_icon(entry.status)
                mode_indicator = f"[yellow]◷ {ts_str} ({dur_str})[/] {status_icon}"
            else:
                mode_indicator = "[yellow]◷ (unknown)[/]"

            parts.append(f"{left_arrow}[{current}/{total + 1}] {mode_indicator}{right_arrow}")

        header.update("".join(parts))

    def _get_current_history_entry(self) -> ExecutionHistoryEntry | None:
        """Get the currently viewed history entry."""
        if self._stage is None or self._history_index is None:
            return None
        if 0 <= self._history_index < len(self._stage.history):
            return self._stage.history[self._history_index]
        return None

    def _get_status_icon(self, status: StageStatus) -> str:
        """Get status icon for history entry."""
        match status:
            case StageStatus.RAN | StageStatus.COMPLETED:
                return "[green]✓[/]"
            case StageStatus.FAILED:
                return "[red]✗[/]"
            case StageStatus.SKIPPED:
                return "[yellow]⊘[/]"
            case _:
                return ""

    def refresh_header(self) -> None:  # pragma: no cover
        """Refresh the header display (call when history changes)."""
        if self._stage:
            self._history_total = len(self._stage.history)
        self._update_header()


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


_TUI_CSS: str = """
#main-split {
    height: 1fr;
}

#stage-list {
    width: 35%;
    min-width: 30;
    max-width: 50;
    height: 100%;
    border: solid $surface-lighten-1;
    padding: 1;
}

#stage-list.focused {
    border: solid $primary;
}

#detail-panel {
    width: 1fr;
    height: 100%;
    border: solid $surface-lighten-1;
}

#detail-panel.focused {
    border: solid $primary;
}

#detail-header {
    height: 1;
    padding: 0 1;
    background: $surface-lighten-1;
}

#detail-tabs {
    height: 1fr;
}

#stage-logs {
    height: 100%;
}

#input-panel {
    height: 100%;
    padding: 1;
    overflow-y: auto;
}

#output-panel {
    height: 100%;
    padding: 1;
    overflow-y: auto;
}

#log-panel {
    height: 1fr;
    border: solid yellow;
}

.section-header {
    text-style: bold;
    margin-bottom: 1;
}

#logs-view {
    height: 100%;
    display: none;
}

.view-active {
    display: block;
}

.view-hidden {
    display: none;
}

/* Split-view layout for diff panels */
.diff-panel {
    height: 100%;
}

.diff-panel #item-list {
    width: 50%;
    height: 100%;
    overflow-y: auto;
}

.diff-panel #detail-pane {
    width: 50%;
    height: 100%;
    border-left: solid $surface-lighten-1;
    padding-left: 1;
    overflow-y: auto;
}

.diff-panel.expanded #item-list {
    display: none;
}

.diff-panel.expanded #detail-pane {
    width: 100%;
    border-left: none;
}

/* Debug panel - toggleable footer showing queue stats */
#debug-panel {
    height: auto;
    max-height: 4;
    background: $surface;
    border-top: solid $primary;
    padding: 0 1;
    display: none;
}

#debug-panel.visible {
    display: block;
}
"""

_TUI_BINDINGS: list[textual.binding.BindingType] = [
    textual.binding.Binding("q", "quit", "Quit"),
    textual.binding.Binding("c", "commit", "Commit"),
    textual.binding.Binding("escape", "escape_action", "Cancel/Collapse", show=False),
    textual.binding.Binding("enter", "expand_details", "Expand", show=False),
    # Panel focus switching
    textual.binding.Binding("tab", "switch_focus", "Switch Panel"),
    # Navigation (context-aware: stages panel vs detail panel)
    textual.binding.Binding("j", "nav_down", "Down"),
    textual.binding.Binding("k", "nav_up", "Up"),
    textual.binding.Binding("down", "nav_down", "Down", show=False),
    textual.binding.Binding("up", "nav_up", "Up", show=False),
    textual.binding.Binding("h", "nav_left", "Left", show=False),
    textual.binding.Binding("l", "nav_right", "Right", show=False),
    textual.binding.Binding("left", "nav_left", "Left", show=False),
    textual.binding.Binding("right", "nav_right", "Right", show=False),
    # Changed-item navigation (in detail panel only)
    textual.binding.Binding("n", "next_changed", "Next Change", show=False),
    textual.binding.Binding("N", "prev_changed", "Prev Change", show=False),
    # History navigation (works in all tabs)
    textual.binding.Binding("[", "history_older", "Older", show=False),
    textual.binding.Binding("]", "history_newer", "Newer", show=False),
    textual.binding.Binding("G", "history_live", "Live View", show=False),
    textual.binding.Binding("H", "show_history_list", "History", show=False),
    # Tab mnemonic keys (shift+letter)
    textual.binding.Binding("L", "goto_tab_logs", "Logs Tab", show=False),
    textual.binding.Binding("I", "goto_tab_input", "Input Tab", show=False),
    textual.binding.Binding("O", "goto_tab_output", "Output Tab", show=False),
    # All logs view toggle
    textual.binding.Binding("a", "show_all_logs", "All Logs"),
    # Debug panel toggle
    textual.binding.Binding("~", "toggle_debug", "Debug"),
    # Keep-going toggle (watch mode only)
    textual.binding.Binding("g", "toggle_keep_going", "Keep-going"),
    # Keep stage filtering with number keys (4-9 for stages, 1-3 could conflict with tabs)
    *[
        textual.binding.Binding(str(i), f"filter_stage({i - 1})", f"Stage {i}", show=False)
        for i in range(1, 10)
    ],
]

_logger = logging.getLogger(__name__)

# TypeVar for App return type - RunTuiApp returns results, WatchTuiApp returns None
_AppReturnT = TypeVar("_AppReturnT")


class _BaseTuiApp(textual.app.App[_AppReturnT]):
    """Base class for TUI applications with shared stage management."""

    CSS: ClassVar[str] = _TUI_CSS
    BINDINGS: ClassVar[list[textual.binding.BindingType]] = _TUI_BINDINGS
    _TAB_IDS: ClassVar[tuple[str, str, str]] = ("tab-logs", "tab-input", "tab-output")

    def __init__(
        self,
        message_queue: mp.Queue[TuiMessage],
        stage_names: list[str] | None = None,
        tui_log: Path | None = None,
    ) -> None:
        """Initialize base TUI app state."""
        super().__init__()
        self._tui_queue: mp.Queue[TuiMessage] = message_queue
        self._stages: dict[str, StageInfo] = {}
        self._stage_order: list[str] = []
        self._selected_idx: int = 0
        self._selected_stage_name: str | None = None
        self._show_logs: bool = False
        self._focused_panel: Literal["stages", "detail"] = "stages"
        self._reader_thread: threading.Thread | None = None
        self._shutdown_event: threading.Event = threading.Event()
        self._log_file: IO[str] | None = None

        # Debug panel stats tracking
        self._tui_stats: QueueStatsTracker = QueueStatsTracker(
            "tui_queue",
            message_queue,  # pyright: ignore[reportArgumentType] - Queue is invariant
        )
        self._output_stats: QueueStatsTracker | None = None  # Set in WatchTuiApp
        self._start_time: float = 0.0  # Set in on_mount for accurate uptime
        self._debug_timer: textual.timer.Timer | None = None
        self._stats_log_timer: textual.timer.Timer | None = None

        # Open log file if configured (line-buffered for real-time tailing)
        if tui_log:
            self._log_file = open(tui_log, "w", buffering=1)  # noqa: SIM115
            # Prevent fd inheritance to child processes (avoids multiprocessing errors)
            os.set_inheritable(self._log_file.fileno(), False)
            atexit.register(self._close_log_file)

        if stage_names:
            for i, name in enumerate(stage_names, 1):
                info = StageInfo(name, i, len(stage_names))
                self._stages[name] = info
                self._stage_order.append(name)
            # Select first stage by default
            self._selected_stage_name = stage_names[0]

    @property
    def selected_stage_name(self) -> str | None:
        """Return name of currently selected stage, or None if no stages."""
        if self._stage_order and self._selected_idx < len(self._stage_order):
            return self._stage_order[self._selected_idx]
        return None

    @property
    def focused_panel(self) -> Literal["stages", "detail"]:
        """Return which panel currently has focus."""
        return self._focused_panel

    def select_stage_by_index(self, idx: int) -> None:
        """Select a stage by index (for testing)."""
        if 0 <= idx < len(self._stage_order):
            self._selected_idx = idx
            self._update_detail_panel()

    def _close_log_file(self) -> None:
        """Close the log file if open (thread-safe)."""
        # Swap-then-check pattern avoids race condition
        log_file = self._log_file
        self._log_file = None
        if log_file:
            atexit.unregister(self._close_log_file)
            log_file.close()

    def _select_stage(self, idx: int) -> None:
        """Update selection by index, keeping both index and name in sync."""
        if 0 <= idx < len(self._stage_order):
            self._selected_idx = idx
            self._selected_stage_name = self._stage_order[idx]

    def _recompute_selection_idx(self) -> None:
        """Recompute index from name after stage list changes. O(n) but infrequent."""
        if self._selected_stage_name and self._selected_stage_name in self._stage_order:
            self._selected_idx = self._stage_order.index(self._selected_stage_name)
        else:
            # Stage was removed, select first available
            self._selected_idx = 0
            self._selected_stage_name = self._stage_order[0] if self._stage_order else None

    def _write_to_log(self, data: str) -> None:  # pragma: no cover
        """Write a line to the log file, logging warning on first failure."""
        if self._log_file:
            try:
                self._log_file.write(data)
            except OSError as e:
                _logger.warning(f"TUI log write failed: {e}")
                # Disable further writes to avoid log spam
                self._log_file = None

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Header()

        with textual.containers.Horizontal(id="main-split"):
            yield StageListPanel(list(self._stages.values()), id="stage-list")
            yield TabbedDetailPanel(id="detail-panel")

        with textual.containers.Vertical(id="logs-view", classes="view-hidden"):
            yield LogPanel(id="log-panel")

        yield DebugPanel(id="debug-panel")
        yield textual.widgets.Footer()

    async def on_mount(self) -> None:  # pragma: no cover
        """Base on_mount - sets start time and starts stats log timer if configured."""
        self._start_time = time.monotonic()
        if self._log_file is not None:
            self._stats_log_timer = self.set_interval(1.0, self._write_stats_to_log)

    def _start_queue_reader(self) -> None:  # pragma: no cover
        """Start the background queue reader thread."""
        self._reader_thread = threading.Thread(target=self._read_queue, daemon=True)
        self._reader_thread.start()

    def _read_queue(self) -> None:  # pragma: no cover
        """Read from queue and post messages to Textual (runs in background thread)."""
        while not self._shutdown_event.is_set():
            try:
                msg = self._tui_queue.get(timeout=0.02)
                self._tui_stats.record_message()  # Track stats for debug panel
                if msg is None:
                    self._write_to_log('{"type": "shutdown"}\n')
                    break
                # default=str handles StrEnum serialization
                self._write_to_log(json.dumps(msg, default=str) + "\n")
                self.post_message(TuiUpdate(msg))
            except queue.Empty:
                continue
            except (EOFError, OSError, BrokenPipeError):
                # Queue was closed or broken - exit gracefully
                _logger.debug("TUI queue reader exiting: queue closed or broken")
                break

    def _handle_log(self, msg: TuiLogMessage) -> None:  # pragma: no cover
        stage = msg["stage"]
        line = msg["line"]
        is_stderr = msg["is_stderr"]
        timestamp = msg["timestamp"]

        if stage in self._stages:
            self._stages[stage].logs.append((line, is_stderr, timestamp))

        # Update all-logs panel
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log(stage, line, is_stderr)

        # Update stage-specific log panel if this stage is selected
        if self._selected_stage_name == stage:
            try:
                stage_log_panel = self.query_one("#stage-logs", StageLogPanel)
                stage_log_panel.add_log(line, is_stderr, timestamp)
            except textual.css.query.NoMatches:
                _logger.debug("stage-logs panel not found during log update")

    def _update_detail_panel(self) -> None:  # pragma: no cover
        stage = self._stages.get(self._selected_stage_name) if self._selected_stage_name else None
        detail = self.query_one("#detail-panel", TabbedDetailPanel)
        detail.set_stage(stage)

    def _update_focus_visual(self) -> None:  # pragma: no cover
        """Update visual indicators for focused panel."""
        stage_list = self.query_one("#stage-list", StageListPanel)
        detail_panel = self.query_one("#detail-panel", TabbedDetailPanel)
        is_stages_focused = self._focused_panel == "stages"
        stage_list.set_class(is_stages_focused, "focused")
        detail_panel.set_class(not is_stages_focused, "focused")

    def action_switch_focus(self) -> None:  # pragma: no cover
        """Toggle focus between stages panel and detail panel."""
        self._focused_panel = "detail" if self._focused_panel == "stages" else "stages"
        self._update_focus_visual()

    def _get_active_diff_panel(self) -> InputDiffPanel | OutputDiffPanel | None:  # pragma: no cover
        """Get the diff panel for the active tab, if any."""
        try:
            tabs = self.query_one("#detail-tabs", textual.widgets.TabbedContent)
            match tabs.active:
                case "tab-input":
                    return self.query_one("#input-panel", InputDiffPanel)
                case "tab-output":
                    return self.query_one("#output-panel", OutputDiffPanel)
                case _:
                    return None  # Logs tab has no selectable items
        except textual.css.query.NoMatches:
            return None

    def action_nav_down(self) -> None:  # pragma: no cover
        """Navigate down - stage list or item list depending on focus."""
        if self._focused_panel == "stages":
            self.action_next_stage()
        elif self._focused_panel == "detail" and (panel := self._get_active_diff_panel()):
            panel.select_next()

    def action_nav_up(self) -> None:  # pragma: no cover
        """Navigate up - stage list or item list depending on focus."""
        if self._focused_panel == "stages":
            self.action_prev_stage()
        elif self._focused_panel == "detail" and (panel := self._get_active_diff_panel()):
            panel.select_prev()

    def action_nav_left(self) -> None:  # pragma: no cover
        """Navigate left - previous tab or switch to stages panel."""
        if self._focused_panel == "detail":
            try:
                tabs = self.query_one("#detail-tabs", textual.widgets.TabbedContent)
                if tabs.active == self._TAB_IDS[0]:
                    # On leftmost tab, switch to stages panel
                    self._focused_panel = "stages"
                    self._update_focus_visual()
                elif tabs.active in self._TAB_IDS:
                    current_idx = self._TAB_IDS.index(tabs.active)
                    tabs.active = self._TAB_IDS[current_idx - 1]
            except (textual.css.query.NoMatches, ValueError):
                _logger.debug("detail-tabs not found during nav_left")

    def action_nav_right(self) -> None:  # pragma: no cover
        """Navigate right - next tab or switch to detail panel."""
        if self._focused_panel == "stages":
            self._focused_panel = "detail"
            self._update_focus_visual()
        else:
            try:
                tabs = self.query_one("#detail-tabs", textual.widgets.TabbedContent)
                if tabs.active in self._TAB_IDS:
                    current_idx = self._TAB_IDS.index(tabs.active)
                    if current_idx < len(self._TAB_IDS) - 1:
                        tabs.active = self._TAB_IDS[current_idx + 1]
            except (textual.css.query.NoMatches, ValueError):
                _logger.debug("detail-tabs not found during nav_right")

    def _goto_tab(self, tab_id: str) -> None:  # pragma: no cover
        """Jump to a specific tab and focus the detail panel."""
        try:
            tabs = self.query_one("#detail-tabs", textual.widgets.TabbedContent)
            tabs.active = tab_id
            self._focused_panel = "detail"
            self._update_focus_visual()
        except textual.css.query.NoMatches:
            _logger.debug("detail-tabs not found during goto_tab")

    def action_goto_tab_logs(self) -> None:  # pragma: no cover
        self._goto_tab("tab-logs")

    def action_goto_tab_input(self) -> None:  # pragma: no cover
        self._goto_tab("tab-input")

    def action_goto_tab_output(self) -> None:  # pragma: no cover
        self._goto_tab("tab-output")

    def action_next_stage(self) -> None:  # pragma: no cover
        if self._selected_idx < len(self._stage_order) - 1:
            self._select_stage(self._selected_idx + 1)
            self._update_detail_panel()

    def action_prev_stage(self) -> None:  # pragma: no cover
        if self._selected_idx > 0:
            self._select_stage(self._selected_idx - 1)
            self._update_detail_panel()

    def action_toggle_view(self) -> None:  # pragma: no cover
        self._show_logs = not self._show_logs
        main_split = self.query_one("#main-split")
        logs_view = self.query_one("#logs-view")
        main_split.set_class(self._show_logs, "view-hidden")
        main_split.set_class(not self._show_logs, "view-active")
        logs_view.set_class(self._show_logs, "view-active")
        logs_view.set_class(not self._show_logs, "view-hidden")

    def action_show_all_logs(self) -> None:  # pragma: no cover
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.set_filter(None)
        if not self._show_logs:
            self.action_toggle_view()

    def action_filter_stage(self, idx: int) -> None:  # pragma: no cover
        """Filter logs to stage at index idx (0-based)."""
        if idx < len(self._stage_order):
            stage_name = self._stage_order[idx]
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.set_filter(stage_name)
            if not self._show_logs:
                self.action_toggle_view()

    def action_escape_action(self) -> None:  # pragma: no cover
        """Context-aware Esc: cancel commit or collapse detail expansion."""
        # Subclasses override for commit cancellation
        # Default behavior: collapse detail panel if expanded
        panel = self._get_active_diff_panel()
        if self._focused_panel == "detail" and panel and panel.is_detail_expanded:
            panel.collapse_details()

    def action_expand_details(self) -> None:  # pragma: no cover
        """Expand details pane to full width."""
        panel = self._get_active_diff_panel()
        if self._focused_panel == "detail" and panel:
            panel.expand_details()

    def action_next_changed(self) -> None:  # pragma: no cover
        """Move selection to next changed item."""
        panel = self._get_active_diff_panel()
        if self._focused_panel == "detail" and panel:
            panel.select_next_changed()

    def action_prev_changed(self) -> None:  # pragma: no cover
        """Move selection to previous changed item."""
        panel = self._get_active_diff_panel()
        if self._focused_panel == "detail" and panel:
            panel.select_prev_changed()

    def action_toggle_debug(self) -> None:  # pragma: no cover
        """Toggle debug panel visibility."""
        debug_panel = self.query_one("#debug-panel", DebugPanel)
        if self._debug_timer is None:
            # Show panel and start update timer
            debug_panel.add_class("visible")
            self._debug_timer = self.set_interval(0.5, self._update_debug_stats)
        else:
            # Hide panel and stop update timer
            debug_panel.remove_class("visible")
            self._debug_timer.stop()
            self._debug_timer = None

    def action_toggle_keep_going(self) -> None:  # pragma: no cover
        """Toggle keep-going mode (only available in watch mode)."""
        self.notify(
            "Keep-going toggle is only available in watch mode (use --watch)", severity="warning"
        )

    def _update_debug_stats(self) -> None:  # pragma: no cover
        """Update debug panel with current stats."""
        try:
            stats = self._collect_debug_stats()
            debug_panel = self.query_one("#debug-panel", DebugPanel)
            debug_panel.update_stats(stats)
        except Exception:
            _logger.debug("Failed to update debug stats", exc_info=True)

    def _collect_debug_stats(self) -> DebugStats:  # pragma: no cover
        """Collect current debug statistics."""
        active_workers = sum(
            1 for s in self._stages.values() if s.status == StageStatus.IN_PROGRESS
        )

        return DebugStats(
            tui_queue=self._tui_stats.get_stats(),
            output_queue=self._output_stats.get_stats() if self._output_stats else None,
            active_workers=active_workers,
            memory_mb=get_memory_mb(),
            uptime_seconds=time.monotonic() - self._start_time,
        )

    def _write_stats_to_log(self) -> None:  # pragma: no cover
        """Write periodic stats snapshot to log file."""
        if self._log_file is None:
            return
        try:
            stats = self._collect_debug_stats()
            log_entry = {
                "type": "stats_snapshot",
                "timestamp": time.time(),
                **stats,
            }
            self._write_to_log(json.dumps(log_entry, default=str) + "\n")
        except Exception:
            _logger.debug("Failed to write stats to log", exc_info=True)

    @override
    async def action_quit(self) -> None:  # pragma: no cover
        # Stop debug timers before shutdown
        if self._debug_timer is not None:
            self._debug_timer.stop()
            self._debug_timer = None
        if self._stats_log_timer is not None:
            self._stats_log_timer.stop()
            self._stats_log_timer = None

        self._shutdown_event.set()
        # Wait for reader thread to finish before closing log file (avoids race)
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            if self._reader_thread.is_alive():
                _logger.debug("Reader thread did not finish within 2s timeout")
            else:
                _logger.debug("Reader thread finished cleanly")
        self._close_log_file()
        await super().action_quit()


class RunTuiApp(_BaseTuiApp[dict[str, ExecutionSummary] | None]):
    """TUI for single pipeline execution."""

    def __init__(
        self,
        stage_names: list[str],
        message_queue: mp.Queue[TuiMessage],
        executor_func: Callable[[], dict[str, ExecutionSummary]],
        tui_log: Path | None = None,
    ) -> None:
        super().__init__(message_queue, stage_names, tui_log=tui_log)
        self._executor_func: Callable[[], dict[str, ExecutionSummary]] = executor_func
        self._results: dict[str, ExecutionSummary] | None = None
        self._error: Exception | None = None
        self._executor_thread: threading.Thread | None = None

    @property
    def error(self) -> Exception | None:
        """Return any exception that occurred during execution."""
        return self._error

    @override
    async def on_mount(self) -> None:  # pragma: no cover
        await super().on_mount()  # Start stats log timer if configured
        self._update_detail_panel()
        self._start_queue_reader()
        self._executor_thread = threading.Thread(target=self._run_executor, daemon=True)
        self._executor_thread.start()

    def _run_executor(self) -> None:  # pragma: no cover
        """Run the executor (runs in background thread)."""
        results: dict[str, ExecutionSummary] = {}
        error: Exception | None = None
        try:
            results = self._executor_func()
        except Exception as e:
            error = e
        finally:
            self._tui_queue.put(None)
            self.post_message(ExecutorComplete(results, error))

    def on_tui_update(self, event: TuiUpdate) -> None:  # pragma: no cover
        """Handle executor updates in Textual's event loop."""
        msg = event.msg
        match msg["type"]:
            case TuiMessageType.LOG:
                self._handle_log(msg)
            case TuiMessageType.STATUS:
                self._handle_status(msg)
            case TuiMessageType.WATCH | TuiMessageType.RELOAD:
                pass

    def _handle_status(self, msg: TuiStatusMessage) -> None:  # pragma: no cover
        stage = msg["stage"]
        if stage not in self._stages:
            return

        info = self._stages[stage]
        info.status = msg["status"]
        info.reason = msg["reason"]
        info.elapsed = msg["elapsed"]

        stage_list = self.query_one("#stage-list", StageListPanel)
        stage_list.update_stage(stage)
        self._update_detail_panel()

        completed = sum(
            1
            for s in self._stages.values()
            if s.status
            in (StageStatus.COMPLETED, StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED)
        )
        self.title = f"pivot run ({completed}/{len(self._stages)})"  # pyright: ignore[reportUnannotatedClassAttribute]

    def on_executor_complete(self, event: ExecutorComplete) -> None:  # pragma: no cover
        """Handle executor completion."""
        self._results = event.results
        self._error = event.error
        if event.error:
            self.title = f"pivot run - FAILED: {event.error}"
        else:
            self.title = "pivot run - Complete"
        self.exit(self._results)


def run_with_tui(
    stage_names: list[str],
    message_queue: mp.Queue[TuiMessage],
    executor_func: Callable[[], dict[str, ExecutionSummary]],
    tui_log: Path | None = None,
) -> dict[str, ExecutionSummary]:  # pragma: no cover
    """Run pipeline with TUI display. Raises if executor fails."""
    app = RunTuiApp(stage_names, message_queue, executor_func, tui_log=tui_log)
    results = app.run()
    if app.error is not None:
        raise app.error
    return results or {}


def should_use_tui(display_mode: DisplayMode | None) -> bool:
    """Determine if TUI should be used based on display mode and TTY."""
    import sys

    if display_mode == DisplayMode.TUI:
        return True
    if display_mode == DisplayMode.PLAIN:
        return False
    # Auto-detect: use TUI if stdout is a TTY
    return sys.stdout.isatty()


class ConfirmCommitScreen(textual.screen.ModalScreen[bool]):
    """Modal screen for confirming commit on exit."""

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("y", "confirm(True)", "Yes"),
        textual.binding.Binding("n", "confirm(False)", "No"),
        textual.binding.Binding("escape", "confirm(False)", "Cancel"),
    ]

    DEFAULT_CSS: ClassVar[str] = """
    ConfirmCommitScreen {
        align: center middle;
    }

    ConfirmCommitScreen > #dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    ConfirmCommitScreen > #dialog > #message {
        margin-bottom: 1;
    }
    """

    @override
    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.Container(id="dialog"):
            yield textual.widgets.Static(
                "You have uncommitted changes. Commit before exit?", id="message"
            )
            yield textual.widgets.Static("[y] Yes  [n] No  [Esc] Cancel")

    def action_confirm(self, result: bool) -> None:
        self.dismiss(result)


class HistoryListScreen(textual.screen.ModalScreen[int | None]):
    """Modal screen for selecting a history entry. Returns index or None for live."""

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("j", "select_next", "Down"),
        textual.binding.Binding("k", "select_prev", "Up"),
        textual.binding.Binding("down", "select_next", "Down", show=False),
        textual.binding.Binding("up", "select_prev", "Up", show=False),
        textual.binding.Binding("enter", "confirm", "Select"),
        textual.binding.Binding("escape", "cancel", "Cancel"),
        textual.binding.Binding("G", "go_live", "Live"),
    ]

    DEFAULT_CSS: ClassVar[str] = """
    HistoryListScreen {
        align: center middle;
    }

    HistoryListScreen > #history-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    HistoryListScreen > #history-dialog > #history-title {
        text-style: bold;
        margin-bottom: 1;
    }

    HistoryListScreen > #history-dialog > #history-table {
        height: auto;
        max-height: 20;
        margin-bottom: 1;
    }
    """

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
            self._selected_idx = len(history) - 1 if history else 0
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

            # Format status
            if entry.status == StageStatus.RAN:
                status_str = "[green]✓ ran[/]  "
            elif entry.status == StageStatus.FAILED:
                status_str = "[red]✗ fail[/] "
            elif entry.status == StageStatus.SKIPPED:
                status_str = "[yellow]○ skip[/] "
            else:
                status_str = f"{entry.status.value[:7]:<7} "

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


# Constants for commit lock acquisition
_COMMIT_LOCK_POLL_INTERVAL = 5.0  # seconds between lock attempts
_COMMIT_LOCK_TIMEOUT = 60.0  # total seconds before giving up


@final
class WatchTuiApp(_BaseTuiApp[None]):
    """TUI for watch mode pipeline execution."""

    _output_queue: mp.Queue[OutputMessage] | None
    _viewing_history_index: int | None  # None = live view, else index into history deque
    _pending_history: dict[str, _PendingHistoryState]
    _current_run_id: str | None  # Track current run to detect new runs

    def __init__(
        self,
        engine: WatchEngineProtocol,
        message_queue: mp.Queue[TuiMessage],
        output_queue: mp.Queue[OutputMessage] | None = None,
        tui_log: Path | None = None,
        stage_names: list[str] | None = None,
        *,
        no_commit: bool = False,
        serve: bool = False,
    ) -> None:
        super().__init__(message_queue, stage_names, tui_log=tui_log)
        self._engine: WatchEngineProtocol = engine
        self._output_queue = output_queue
        self._engine_thread: threading.Thread | None = None
        self._no_commit: bool = no_commit
        self._commit_in_progress: bool = False
        self._cancel_commit: bool = False
        self._viewing_history_index = None
        self._pending_history = {}
        self._current_run_id = None
        self._serve: bool = serve
        self._agent_server: agent_server.AgentServer | None = None
        self._agent_server_task: asyncio.Task[None] | None = None

    @property
    def _has_running_stages(self) -> bool:
        """Check if any stages are currently in progress."""
        return any(s.status == StageStatus.IN_PROGRESS for s in self._stages.values())

    @override
    async def on_mount(self) -> None:  # pragma: no cover
        await super().on_mount()
        prefix = self._get_keep_going_prefix()
        self.title = f"{prefix}[●] Watching for changes..."
        self._start_queue_reader()
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

        # Start agent server if requested
        if self._serve:
            await self._start_agent_server()

    async def _start_agent_server(self) -> None:  # pragma: no cover
        """Start the JSON-RPC agent server."""
        socket_path = project.get_project_root() / ".pivot" / "agent.sock"
        # Cast to actual WatchEngine type - we know it's the real thing at runtime
        engine = cast("WatchEngine", self._engine)
        self._agent_server = agent_server.AgentServer(engine, socket_path)

        server = None
        try:
            server = await self._agent_server.start()
            _logger.info(f"Agent server listening on {socket_path}")
            self._agent_server_task = asyncio.create_task(server.serve_forever())
        except Exception as e:
            _logger.warning(f"Failed to start agent server: {e}")
            # Clean up - stop server if it was started (handles task creation failure)
            await self._agent_server.stop()
            self._agent_server = None

    async def _stop_agent_server(self) -> None:  # pragma: no cover
        """Stop the JSON-RPC agent server if running."""
        if self._agent_server_task is not None:
            self._agent_server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._agent_server_task
            self._agent_server_task = None

        if self._agent_server is not None:
            await self._agent_server.stop()
            self._agent_server = None

    def _run_engine(self) -> None:  # pragma: no cover
        """Run the watch engine (runs in background thread)."""
        try:
            self._engine.run(tui_queue=self._tui_queue, output_queue=self._output_queue)
        except Exception as e:
            logging.getLogger(__name__).exception(f"Watch engine failed: {e}")
            # Notify TUI about engine failure so user knows watch mode is dead
            error_msg = TuiWatchMessage(
                type=TuiMessageType.WATCH,
                status=WatchStatus.ERROR,
                message="Watch mode crashed. Please restart 'pivot watch'.",
            )
            with contextlib.suppress(Exception):
                self._tui_queue.put_nowait(error_msg)

    def on_tui_update(self, event: TuiUpdate) -> None:  # pragma: no cover
        """Handle executor updates in Textual's event loop."""
        msg = event.msg
        match msg["type"]:
            case TuiMessageType.LOG:
                self._handle_log(msg)
            case TuiMessageType.STATUS:
                self._handle_status(msg)
            case TuiMessageType.WATCH:
                self._handle_watch(msg)
            case TuiMessageType.RELOAD:
                self._handle_reload(msg)

    def _handle_status(self, msg: TuiStatusMessage) -> None:  # pragma: no cover
        stage = msg["stage"]
        status = msg["status"]
        run_id = msg["run_id"]
        is_new_stage = stage not in self._stages

        # Detect new pipeline run - clear stale pending entries from previous run
        if run_id and run_id != self._current_run_id:
            if self._pending_history:
                _logger.debug(
                    "New run %s detected, clearing %d stale pending entries from %s",
                    run_id,
                    len(self._pending_history),
                    self._current_run_id,
                )
                self._pending_history.clear()
            self._current_run_id = run_id

        if is_new_stage:
            info = StageInfo(stage, msg["index"], msg["total"])
            self._stages[stage] = info
            self._stage_order.append(stage)

        info = self._stages[stage]
        info.status = status
        info.reason = msg["reason"]
        info.elapsed = msg["elapsed"]
        info.index = msg["index"]
        info.total = msg["total"]

        # History capture: create entry on IN_PROGRESS, finalize on terminal status
        if status == StageStatus.IN_PROGRESS:
            self._create_history_entry(stage, run_id)
        elif status in (
            StageStatus.RAN,
            StageStatus.COMPLETED,
            StageStatus.SKIPPED,
            StageStatus.FAILED,
        ):
            # Pass run_id for skipped stages that never went through IN_PROGRESS
            self._finalize_history_entry(stage, status, msg["reason"], msg["elapsed"], run_id)

        if is_new_stage:
            self._rebuild_stage_list()
        else:
            stage_list = self.query_one("#stage-list", StageListPanel)
            stage_list.update_stage(stage)
        self._update_detail_panel()

    def _handle_watch(self, msg: TuiWatchMessage) -> None:  # pragma: no cover
        """Handle reactive status updates - update title bar."""
        prefix = self._get_keep_going_prefix()
        match msg["status"]:
            case WatchStatus.WAITING:
                self.title = f"{prefix}[●] Watching for changes..."
            case WatchStatus.RESTARTING:
                self.title = f"{prefix}[↻] Reloading code..."
            case WatchStatus.DETECTING:
                self.title = f"{prefix}[▶] {rich.markup.escape(msg['message'])}"
            case WatchStatus.ERROR:
                self.title = f"{prefix}[!] {rich.markup.escape(msg['message'])}"

    def _handle_reload(self, msg: TuiReloadMessage) -> None:  # pragma: no cover
        """Handle registry reload - update stage list."""
        new_stages = msg["stages"]
        old_stages = set(self._stage_order)
        new_stage_set = set(new_stages)

        removed = old_stages - new_stage_set
        added = new_stage_set - old_stages

        for name in removed:
            if name in self._stages:
                del self._stages[name]
            # Clean up pending history data for removed stages
            self._pending_history.pop(name, None)
            # Exit history view if viewing removed stage
            if self._viewing_history_index is not None and self._selected_stage_name == name:
                self._viewing_history_index = None

        for name in added:
            info = StageInfo(name, len(self._stages) + 1, len(new_stages))
            self._stages[name] = info

        self._stage_order = new_stages
        for i, name in enumerate(self._stage_order, 1):
            if name in self._stages:
                self._stages[name].index = i
                self._stages[name].total = len(self._stage_order)

        # Recompute selection index from name (handles removed stages)
        self._recompute_selection_idx()

        self._rebuild_stage_list()
        self._update_detail_panel()

    @override
    def _handle_log(self, msg: TuiLogMessage) -> None:  # pragma: no cover
        """Handle log message - also accumulate for history."""
        super()._handle_log(msg)
        # Accumulate logs for pending history entries
        stage = msg["stage"]
        if stage in self._pending_history:
            self._pending_history[stage].logs.append(
                (msg["line"], msg["is_stderr"], msg["timestamp"])
            )

    def _create_history_entry(self, stage_name: str, run_id: str) -> None:
        """Create a new history entry when stage starts executing."""
        # Capture input snapshot at stage start (why this stage is running)
        input_snapshot: StageExplanation | None = None
        try:
            registry_info = REGISTRY.get(stage_name)
            cache_dir = project.get_cache_dir()
            input_snapshot = explain.get_stage_explanation(
                stage_name=stage_name,
                fingerprint=registry_info["fingerprint"],
                deps=registry_info["deps_paths"],
                params_instance=registry_info["params"],
                overrides=parameters.load_params_yaml(),
                cache_dir=cache_dir,
            )
        except Exception:
            _logger.debug("Failed to capture input snapshot for %s", stage_name)

        self._pending_history[stage_name] = _PendingHistoryState(
            run_id=run_id,
            timestamp=time.time(),
            input_snapshot=input_snapshot,
        )

    def _finalize_history_entry(
        self,
        stage_name: str,
        status: StageStatus,
        reason: str,
        elapsed: float | None,
        run_id: str | None = None,
    ) -> None:
        """Finalize history entry when stage completes.

        For stages that went through IN_PROGRESS, uses pending state for logs/snapshots.
        For upstream-skipped stages (never ran), creates a minimal entry with the skip reason.
        """
        if stage_name not in self._stages:
            return

        info = self._stages[stage_name]
        pending = self._pending_history.pop(stage_name, None)

        if pending is None:
            # No pending state - stage was skipped without running (upstream failure)
            # Create a minimal history entry to show when/why it was skipped
            if status == StageStatus.SKIPPED and run_id:
                entry = ExecutionHistoryEntry(
                    run_id=run_id,
                    stage_name=stage_name,
                    timestamp=time.time(),
                    duration=None,  # Never ran, no duration
                    status=status,
                    reason=reason,
                    logs=[],  # No logs - never executed
                    input_snapshot=None,  # Could capture why skipped, but keep simple
                    output_snapshot=None,  # No outputs - never ran
                )
            else:
                # Non-SKIPPED status without pending is unexpected, skip
                return
        else:
            # Normal case: stage ran, finalize with captured data
            # Capture output snapshot at stage end (what the stage produced)
            output_snapshot: list[OutputChange] | None = None
            try:
                registry_info = REGISTRY.get(stage_name)
                cache_dir = project.get_cache_dir()
                stages_dir = lock.get_stages_dir(cache_dir)
                lock_data = lock.StageLock(stage_name, stages_dir).read()
                output_snapshot = diff_panels.compute_output_changes(lock_data, registry_info)
            except Exception:
                _logger.debug("Failed to capture output snapshot for %s", stage_name)

            entry = ExecutionHistoryEntry(
                run_id=pending.run_id,
                stage_name=stage_name,
                timestamp=pending.timestamp,
                duration=elapsed,
                status=status,
                reason=reason,
                logs=list(pending.logs),  # Convert bounded deque to list
                input_snapshot=pending.input_snapshot,
                output_snapshot=output_snapshot,
            )

        # Check if deque is at capacity before appending (oldest will be evicted)
        was_at_capacity = len(info.history) == info.history.maxlen
        info.history.append(entry)

        # Adjust history index if viewing this stage and an entry was evicted
        if (
            was_at_capacity
            and stage_name == self._selected_stage_name
            and self._viewing_history_index is not None
        ):
            if self._viewing_history_index == 0:
                # Was viewing oldest entry which was just evicted, go to live view
                self._viewing_history_index = None
            else:
                # Decrement to keep pointing at same logical entry
                self._viewing_history_index -= 1

        # Refresh header if this is the selected stage (history count changed)
        if stage_name == self._selected_stage_name:
            try:
                detail = self.query_one("#detail-panel", TabbedDetailPanel)
                detail.refresh_header()
            except (textual.css.query.NoMatches, textual.app.ScreenStackError):
                _logger.debug("Detail panel not present during history finalization")

    def _get_current_stage_history(self) -> collections.deque[ExecutionHistoryEntry]:
        """Get the history deque for the currently selected stage."""
        if self._selected_stage_name and self._selected_stage_name in self._stages:
            return self._stages[self._selected_stage_name].history
        return collections.deque()

    def _navigate_history_prev(self) -> bool:  # pragma: no cover
        """Navigate to previous (older) history entry. Returns True if navigation occurred."""
        history = self._get_current_stage_history()
        if not history:
            return False

        if self._viewing_history_index is None:
            # Currently in live view, go to most recent history entry
            self._viewing_history_index = len(history) - 1
        elif self._viewing_history_index > 0:
            # Go to older entry
            self._viewing_history_index -= 1
        else:
            # Already at oldest entry
            return False

        self._update_history_view()
        return True

    def _navigate_history_next(self) -> bool:  # pragma: no cover
        """Navigate to next (newer) history entry or live view. Returns True if navigation occurred."""
        history = self._get_current_stage_history()

        if self._viewing_history_index is None:
            # Already in live view
            return False

        if self._viewing_history_index < len(history) - 1:
            # Go to newer entry
            self._viewing_history_index += 1
        else:
            # Go to live view
            self._viewing_history_index = None

        self._update_history_view()
        return True

    def _update_history_view(self) -> None:  # pragma: no cover
        """Update the detail panel to show the current history view."""
        try:
            detail = self.query_one("#detail-panel", TabbedDetailPanel)
        except textual.css.query.NoMatches:
            return

        history = self._get_current_stage_history()
        total = len(history)

        # Get diff panels for snapshot updates
        try:
            input_panel = self.query_one("#input-panel", InputDiffPanel)
            output_panel = self.query_one("#output-panel", OutputDiffPanel)
        except textual.css.query.NoMatches:
            input_panel = None
            output_panel = None

        if self._viewing_history_index is None:
            # Live view - show current stage state
            stage = (
                self._stages.get(self._selected_stage_name) if self._selected_stage_name else None
            )
            detail.set_history_view(None, total, None)
            # Restore live logs
            try:
                log_panel = self.query_one("#stage-logs", StageLogPanel)
                log_panel.set_stage(stage)
            except textual.css.query.NoMatches:
                _logger.debug("Stage log panel not present, skipping live log restore")
            # Restore live diff panels
            if input_panel:
                input_panel.set_stage(self._selected_stage_name)
            if output_panel:
                output_panel.set_stage(self._selected_stage_name)
        else:
            # Historical view
            if 0 <= self._viewing_history_index < total:
                entry = history[self._viewing_history_index]
                detail.set_history_view(self._viewing_history_index, total, entry)
                # Update diff panels with snapshots
                if input_panel:
                    if entry.input_snapshot:
                        input_panel.set_from_snapshot(entry.input_snapshot)
                    else:
                        input_panel.set_stage(None)  # Show "no data" state
                if output_panel:
                    if entry.output_snapshot:
                        output_panel.set_from_snapshot(entry.stage_name, entry.output_snapshot)
                    else:
                        output_panel.set_stage(None)  # Show "no data" state

    @override
    def action_nav_up(self) -> None:  # pragma: no cover
        """Navigate up - stage list or item list (diff tabs)."""
        if self._focused_panel == "stages":
            self.action_prev_stage()
        elif self._focused_panel == "detail" and (panel := self._get_active_diff_panel()):
            panel.select_prev()

    @override
    def action_nav_down(self) -> None:  # pragma: no cover
        """Navigate down - stage list or item list (diff tabs)."""
        if self._focused_panel == "stages":
            self.action_next_stage()
        elif self._focused_panel == "detail" and (panel := self._get_active_diff_panel()):
            panel.select_next()

    def action_history_older(self) -> None:  # pragma: no cover
        """Navigate to older (previous) history entry."""
        if self._focused_panel != "detail":
            return
        self._navigate_history_prev()

    def action_history_newer(self) -> None:  # pragma: no cover
        """Navigate to newer (next) history entry or live view."""
        if self._focused_panel != "detail":
            return
        self._navigate_history_next()

    def action_history_live(self) -> None:  # pragma: no cover
        """Jump directly to live view."""
        if self._focused_panel != "detail":
            return
        if self._viewing_history_index is not None:
            self._viewing_history_index = None
            self._update_history_view()

    async def action_show_history_list(self) -> None:  # pragma: no cover
        """Show modal with history list for current stage."""
        if not self._selected_stage_name:
            return
        history = self._get_current_stage_history()
        if not history:
            self.notify("No history for this stage")
            return
        result = await self.push_screen_wait(
            HistoryListScreen(self._selected_stage_name, history, self._viewing_history_index)
        )
        self._viewing_history_index = result
        self._update_history_view()

    @override
    def _update_detail_panel(self) -> None:  # pragma: no cover
        """Update detail panel and reset history view when stage changes."""
        # Reset history view when changing stages
        self._viewing_history_index = None
        super()._update_detail_panel()

    def _rebuild_stage_list(self) -> None:  # pragma: no cover
        """Rebuild the stage list panel after stages change."""
        stage_list = self.query_one("#stage-list", StageListPanel)
        # Use _stage_order to maintain correct ordering
        ordered_stages = [self._stages[name] for name in self._stage_order if name in self._stages]
        stage_list.rebuild(ordered_stages)

    def _get_keep_going_prefix(self) -> str:  # pragma: no cover
        """Return title prefix for keep-going mode."""
        return "[-k] " if self._engine.keep_going else ""

    @override
    def action_toggle_keep_going(self) -> None:  # pragma: no cover
        """Toggle keep-going mode."""
        enabled = self._engine.toggle_keep_going()
        self.notify(f"Keep-going: {'ON' if enabled else 'OFF'}")
        # Refresh title to show mode indicator
        self.title = f"{self._get_keep_going_prefix()}[●] Watching for changes..."

    async def action_commit(self) -> None:  # pragma: no cover
        """Commit pending changes from --no-commit mode."""
        if self._commit_in_progress:
            return

        if self._has_running_stages:
            self.notify("Cannot commit while stages are running", severity="warning")
            return

        pending = await asyncio.to_thread(lock.list_pending_stages, project.get_project_root())
        if not pending:
            self.notify("Nothing to commit")
            return

        self._commit_in_progress = True
        self._cancel_commit = False
        self.notify("Acquiring commit lock... (Esc to cancel)")

        # Try to acquire lock with short timeouts, allowing cancellation between attempts
        acquired: filelock.BaseFileLock | None = None
        elapsed = 0.0

        try:
            while not self._cancel_commit and elapsed < _COMMIT_LOCK_TIMEOUT:
                try:
                    acquired = await asyncio.to_thread(
                        project_lock.acquire_pending_state_lock, _COMMIT_LOCK_POLL_INTERVAL
                    )
                    break
                except filelock.Timeout:
                    elapsed += _COMMIT_LOCK_POLL_INTERVAL
                    if not self._cancel_commit and elapsed < _COMMIT_LOCK_TIMEOUT:
                        self.notify(f"Still waiting for lock... ({int(elapsed)}s)")

            if self._cancel_commit:
                self.notify("Commit cancelled")
                return

            if acquired is None:
                self.notify(
                    f"Timed out waiting for lock ({int(_COMMIT_LOCK_TIMEOUT)}s). Try again later.",
                    severity="error",
                )
                return

            committed = await asyncio.to_thread(commit_mod.commit_pending)
            self.notify(f"Committed {len(committed)} stage(s)")
        except Exception as e:
            self.notify(f"Commit failed: {e}", severity="error")
        finally:
            # Always reset flag and release lock if acquired
            self._commit_in_progress = False
            if acquired is not None:
                acquired.release()

    @override
    def action_escape_action(self) -> None:  # pragma: no cover
        """Cancel commit if in progress, otherwise collapse detail expansion."""
        if self._commit_in_progress:
            self._cancel_commit = True
            return
        # Fall back to base behavior (collapse detail)
        super().action_escape_action()

    @override
    async def action_quit(self) -> None:  # pragma: no cover
        """Quit the app, prompting to commit if there are uncommitted changes."""
        # Stop agent server first if running
        await self._stop_agent_server()

        # Cancel any pending commit operation
        if self._commit_in_progress:
            self._cancel_commit = True

        if not self._no_commit:
            self._engine.shutdown()
            await super().action_quit()
            return

        # Don't offer commit if stages are running (could cause data inconsistency)
        if self._has_running_stages:
            self._engine.shutdown()
            await super().action_quit()
            return

        pending = await asyncio.to_thread(lock.list_pending_stages, project.get_project_root())
        if not pending:
            self._engine.shutdown()
            await super().action_quit()
            return

        should_commit = await self.push_screen_wait(ConfirmCommitScreen())
        try:
            if should_commit:
                # Acquire lock before committing to prevent race with running stages
                try:
                    commit_lock = await asyncio.to_thread(
                        project_lock.acquire_pending_state_lock, 5.0
                    )
                except filelock.Timeout:
                    self.notify(
                        "Could not acquire lock for commit. Exiting without commit.",
                        severity="warning",
                    )
                else:
                    try:
                        await asyncio.to_thread(commit_mod.commit_pending)
                    finally:
                        commit_lock.release()
        finally:
            self._engine.shutdown()
            self.exit()


def run_watch_tui(
    engine: WatchEngineProtocol,
    message_queue: mp.Queue[TuiMessage],
    output_queue: mp.Queue[OutputMessage] | None = None,
    tui_log: Path | None = None,
    stage_names: list[str] | None = None,
    *,
    no_commit: bool = False,
    serve: bool = False,
) -> None:  # pragma: no cover
    """Run watch mode with TUI display."""
    app = WatchTuiApp(
        engine,
        message_queue,
        output_queue=output_queue,
        tui_log=tui_log,
        stage_names=stage_names,
        no_commit=no_commit,
        serve=serve,
    )
    app.run()
