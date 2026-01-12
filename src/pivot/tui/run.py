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
from typing import IO, TYPE_CHECKING, ClassVar, Literal, TypeVar, final, override

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

from pivot import project
from pivot.executor import ExecutionSummary
from pivot.executor import commit as commit_mod
from pivot.storage import lock, project_lock
from pivot.tui.diff_panels import InputDiffPanel, OutputDiffPanel
from pivot.tui.stats import DebugStats, QueueStats, QueueStatsTracker, get_memory_mb
from pivot.types import (
    DisplayMode,
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

    class WatchEngineProtocol(Protocol):
        """Protocol for WatchEngine to avoid circular imports."""

        def run(
            self,
            tui_queue: mp.Queue[TuiMessage] | None = None,
            output_queue: mp.Queue[OutputMessage] | None = None,
        ) -> None: ...
        def shutdown(self) -> None: ...


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

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._stage = None

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
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
        stage_name = stage.name if stage else None

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

#detail-tabs {
    height: 100%;
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
    # Tab mnemonic keys (shift+letter)
    textual.binding.Binding("L", "goto_tab_logs", "Logs Tab", show=False),
    textual.binding.Binding("I", "goto_tab_input", "Input Tab", show=False),
    textual.binding.Binding("O", "goto_tab_output", "Output Tab", show=False),
    # All logs view toggle
    textual.binding.Binding("a", "show_all_logs", "All Logs"),
    # Debug panel toggle
    textual.binding.Binding("~", "toggle_debug", "Debug"),
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


# Constants for commit lock acquisition
_COMMIT_LOCK_POLL_INTERVAL = 5.0  # seconds between lock attempts
_COMMIT_LOCK_TIMEOUT = 60.0  # total seconds before giving up


@final
class WatchTuiApp(_BaseTuiApp[None]):
    """TUI for watch mode pipeline execution."""

    _output_queue: mp.Queue[OutputMessage] | None

    def __init__(
        self,
        engine: WatchEngineProtocol,
        message_queue: mp.Queue[TuiMessage],
        output_queue: mp.Queue[OutputMessage] | None = None,
        tui_log: Path | None = None,
        stage_names: list[str] | None = None,
        *,
        no_commit: bool = False,
    ) -> None:
        super().__init__(message_queue, stage_names, tui_log=tui_log)
        self._engine: WatchEngineProtocol = engine
        self._output_queue = output_queue
        self._engine_thread: threading.Thread | None = None
        self._no_commit: bool = no_commit
        self._commit_in_progress: bool = False
        self._cancel_commit: bool = False

    @property
    def _has_running_stages(self) -> bool:
        """Check if any stages are currently in progress."""
        return any(s.status == StageStatus.IN_PROGRESS for s in self._stages.values())

    @override
    async def on_mount(self) -> None:  # pragma: no cover
        await super().on_mount()
        self.title = "[●] Watching for changes..."
        self._start_queue_reader()
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

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
        is_new_stage = stage not in self._stages
        if is_new_stage:
            info = StageInfo(stage, msg["index"], msg["total"])
            self._stages[stage] = info
            self._stage_order.append(stage)

        info = self._stages[stage]
        info.status = msg["status"]
        info.reason = msg["reason"]
        info.elapsed = msg["elapsed"]
        info.index = msg["index"]
        info.total = msg["total"]

        if is_new_stage:
            self._rebuild_stage_list()
        else:
            stage_list = self.query_one("#stage-list", StageListPanel)
            stage_list.update_stage(stage)
        self._update_detail_panel()

    def _handle_watch(self, msg: TuiWatchMessage) -> None:  # pragma: no cover
        """Handle reactive status updates - update title bar."""
        match msg["status"]:
            case WatchStatus.WAITING:
                self.title = "[●] Watching for changes..."
            case WatchStatus.RESTARTING:
                self.title = "[↻] Reloading code..."
            case WatchStatus.DETECTING:
                self.title = f"[▶] {rich.markup.escape(msg['message'])}"
            case WatchStatus.ERROR:
                self.title = f"[!] {rich.markup.escape(msg['message'])}"

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

    def _rebuild_stage_list(self) -> None:  # pragma: no cover
        """Rebuild the stage list panel after stages change."""
        stage_list = self.query_one("#stage-list", StageListPanel)
        # Use _stage_order to maintain correct ordering
        ordered_stages = [self._stages[name] for name in self._stage_order if name in self._stages]
        stage_list.rebuild(ordered_stages)

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
) -> None:  # pragma: no cover
    """Run watch mode with TUI display."""
    app = WatchTuiApp(
        engine,
        message_queue,
        output_queue=output_queue,
        tui_log=tui_log,
        stage_names=stage_names,
        no_commit=no_commit,
    )
    app.run()
