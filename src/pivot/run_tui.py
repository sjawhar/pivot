from __future__ import annotations

import collections
import contextlib
import dataclasses
import queue
import threading
from typing import TYPE_CHECKING, Any, ClassVar, override

import rich.markup
import textual.app
import textual.binding
import textual.containers
import textual.css.query
import textual.message
import textual.widgets

from pivot import executor
from pivot.types import (
    DisplayMode,
    StageStatus,
    TuiLogMessage,
    TuiMessage,
    TuiMessageType,
    TuiReactiveMessage,
    TuiStatusMessage,
)

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable


# Type aliases for log entries
type LogEntry = tuple[str, bool]  # (line, is_stderr)
type AllLogsEntry = tuple[str, str, bool]  # (stage, line, is_stderr)

# Terminal states for stage completion
_TERMINAL_STATES: frozenset[StageStatus] = frozenset(
    {
        StageStatus.COMPLETED,
        StageStatus.RAN,
        StageStatus.SKIPPED,
        StageStatus.FAILED,
    }
)


def _format_elapsed(elapsed: float | None, prefix: str = " ") -> str:
    """Format elapsed time as ' (M:SS)' or empty string if None."""
    if elapsed is None:
        return ""
    mins, secs = divmod(int(elapsed), 60)
    return f"{prefix}({mins}:{secs:02d})"


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
    logs: collections.deque[LogEntry] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )


@dataclasses.dataclass
class PipelineStats:
    """Aggregate pipeline statistics for status bar."""

    total: int
    completed: int = 0
    failed: int = 0
    running: int = 0
    avg_stage_time: float = 0.0

    def eta_str(self) -> str:
        """Calculate ETA string based on average stage time."""
        remaining = self.total - self.completed
        if remaining <= 0 or self.avg_stage_time <= 0:
            return ""
        eta_seconds = int(remaining * self.avg_stage_time)
        if eta_seconds < 60:
            return f"~{eta_seconds}s"
        mins, secs = divmod(eta_seconds, 60)
        return f"~{mins}m {secs}s"


class StatusBar(textual.widgets.Static):
    """Persistent status bar showing pipeline progress."""

    _stats: PipelineStats
    _pipeline_name: str

    # Textual widget kwargs vary per class; Any is appropriate at this boundary
    def __init__(self, pipeline_name: str, total_stages: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline_name = pipeline_name
        self._stats = PipelineStats(total=total_stages)

    def update_stats(self, stats: PipelineStats) -> None:  # pragma: no cover
        self._stats = stats
        self._refresh_display()

    def _refresh_display(self) -> None:  # pragma: no cover
        stats = self._stats
        # Progress bar
        if stats.total > 0:
            pct = stats.completed / stats.total
            filled = int(pct * 12)
            bar = "█" * filled + "░" * (12 - filled)
        else:
            bar = "░" * 12

        # Build status line
        parts = [
            f"[bold]{self._pipeline_name}[/]",
            f"[{bar}]",
            f"{stats.completed}/{stats.total}",
        ]

        # Failure count (red if > 0)
        if stats.failed > 0:
            parts.append(f"[red bold]{stats.failed} failed[/]")

        # Running count
        if stats.running > 0:
            parts.append(f"[blue]{stats.running} running[/]")

        # ETA
        eta = stats.eta_str()
        if eta:
            parts.append(f"[dim]ETA {eta}[/]")

        self.update(" │ ".join(parts))

    def on_mount(self) -> None:  # pragma: no cover
        self._refresh_display()


class TuiUpdate(textual.message.Message):
    """Custom message for executor updates."""

    msg: TuiLogMessage | TuiStatusMessage | TuiReactiveMessage

    def __init__(self, msg: TuiLogMessage | TuiStatusMessage | TuiReactiveMessage) -> None:
        self.msg = msg
        super().__init__()


class ExecutorComplete(textual.message.Message):
    """Signal that executor has finished."""

    results: dict[str, executor.ExecutionSummary]
    error: Exception | None

    def __init__(
        self, results: dict[str, executor.ExecutionSummary], error: Exception | None
    ) -> None:
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
        reason_str = f"  ({self._info.reason})" if self._info.reason else ""
        text = f"{index_str} {self._info.name:<20} [{style}]{label}[/]{elapsed_str}{reason_str}"
        self.update(text)

    def on_mount(self) -> None:  # pragma: no cover
        self.update_display()


class StageListPanel(textual.widgets.Static):
    """Panel showing all stages with their status."""

    _stages: list[StageInfo]
    _rows: dict[str, StageRow]

    # Textual widget kwargs vary per class; Any is appropriate at this boundary
    def __init__(self, stages: list[StageInfo], **kwargs: Any) -> None:
        super().__init__(**kwargs)
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


class StageHeader(textual.widgets.Static):
    """Header showing selected stage info."""

    _stage: StageInfo | None

    # Textual widget kwargs vary per class; Any is appropriate at this boundary
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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

        # Compact single-line header
        text = f"[bold]{self._stage.name}[/] │ [{style}]{label}[/]{elapsed_str}"
        if self._stage.reason:
            text += f" │ [dim]{self._stage.reason}[/]"
        self.update(text)


class StageLogPanel(textual.widgets.RichLog):
    """Panel showing logs for selected stage only."""

    _stage_name: str | None

    # Textual widget kwargs vary per class; Any is appropriate at this boundary
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(highlight=True, markup=True, **kwargs)
        self._stage_name = None

    def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
        self._stage_name = stage.name if stage else None
        self.clear()
        if stage:
            for line, is_stderr in stage.logs:
                self._write_line(line, is_stderr)

    def add_log(self, stage: str, line: str, is_stderr: bool) -> None:  # pragma: no cover
        if self._stage_name == stage:
            self._write_line(line, is_stderr)

    def _write_line(self, line: str, is_stderr: bool) -> None:  # pragma: no cover
        escaped = rich.markup.escape(line)
        if is_stderr:
            self.write(f"[red]{escaped}[/]")
        else:
            self.write(escaped)


class DetailPanel(textual.containers.Vertical):
    """Panel showing details and logs of selected stage."""

    # Textual widget kwargs vary per class; Any is appropriate at this boundary
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield StageHeader(id="stage-header")
        yield StageLogPanel(id="stage-log")

    def set_stage(self, stage: StageInfo | None) -> None:  # pragma: no cover
        with contextlib.suppress(textual.css.query.NoMatches):
            self.query_one("#stage-header", StageHeader).set_stage(stage)
            self.query_one("#stage-log", StageLogPanel).set_stage(stage)

    def add_log(self, stage: str, line: str, is_stderr: bool) -> None:  # pragma: no cover
        with contextlib.suppress(textual.css.query.NoMatches):
            self.query_one("#stage-log", StageLogPanel).add_log(stage, line, is_stderr)


class LogPanel(textual.widgets.RichLog):
    """Panel showing streaming logs."""

    _filter_stage: str | None
    _all_logs: collections.deque[AllLogsEntry]

    # Textual widget kwargs vary per class; Any is appropriate at this boundary
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(highlight=True, markup=True, **kwargs)
        self._filter_stage = None
        self._all_logs = collections.deque(maxlen=5000)

    def add_log(self, stage: str, line: str, is_stderr: bool) -> None:  # pragma: no cover
        self._all_logs.append((stage, line, is_stderr))
        if self._filter_stage is None or self._filter_stage == stage:
            self._write_log_line(stage, line, is_stderr)

    def _write_log_line(self, stage: str, line: str, is_stderr: bool) -> None:  # pragma: no cover
        escaped = rich.markup.escape(line)
        prefix = f"[cyan][{stage}][/] "
        if is_stderr:
            self.write(f"{prefix}[red]{escaped}[/]")
        else:
            self.write(f"{prefix}{escaped}")

    def set_filter(self, stage: str | None) -> None:  # pragma: no cover
        """Filter logs to a specific stage or show all (None)."""
        self._filter_stage = stage
        self.clear()
        for s, line, is_stderr in self._all_logs:
            if stage is None or s == stage:
                self._write_log_line(s, line, is_stderr)


class RunTuiApp(textual.app.App[dict[str, executor.ExecutionSummary] | None]):
    """TUI for pipeline execution."""

    CSS: ClassVar[str] = """
    /* Status bar at top */
    #status-bar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    /* Main split panel container */
    #main-split {
        height: 1fr;
    }

    /* Stage list on left */
    #stage-list {
        width: 35%;
        min-width: 30;
        max-width: 50;
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    .section-header {
        text-style: bold;
        margin-bottom: 1;
    }

    /* Highlight selected stage */
    .stage-row-selected {
        background: $primary-darken-2;
    }

    /* Detail panel on right */
    #detail-panel {
        width: 1fr;
        height: 100%;
        border: solid $secondary;
        padding: 0 1;
    }

    #stage-header {
        height: 2;
        padding: 0 0 1 0;
        border-bottom: solid $surface-lighten-1;
    }

    #stage-log {
        height: 1fr;
    }

    /* All logs view (toggled with 'a') */
    #all-logs-panel {
        height: 1fr;
        border: solid $warning;
        padding: 1;
    }

    #all-logs-header {
        height: 1;
        margin-bottom: 1;
    }

    /* View visibility */
    #stages-view {
        height: 100%;
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
    """

    BINDINGS: ClassVar[list[textual.binding.BindingType]] = [
        textual.binding.Binding("q", "quit", "Quit"),
        textual.binding.Binding("j", "next_stage", "↓ Next"),
        textual.binding.Binding("k", "prev_stage", "↑ Prev"),
        textual.binding.Binding("down", "next_stage", "Next", show=False),
        textual.binding.Binding("up", "prev_stage", "Prev", show=False),
        textual.binding.Binding("a", "show_all_logs", "All Logs"),
        textual.binding.Binding("s", "show_stages", "Stages", show=False),
        *[
            textual.binding.Binding(str(i), f"filter_stage({i - 1})", f"Stage {i}", show=False)
            for i in range(1, 10)
        ],
    ]

    _stage_names: list[str]
    _tui_queue: mp.Queue[TuiMessage]
    _executor_func: Callable[[], dict[str, executor.ExecutionSummary]]
    _stages: dict[str, StageInfo]
    _selected_idx: int
    _show_logs: bool
    _results: dict[str, executor.ExecutionSummary] | None
    _error: Exception | None
    _reader_thread: threading.Thread | None
    _executor_thread: threading.Thread | None
    _shutdown_event: threading.Event
    _stats: PipelineStats
    _completed_times: list[float]

    def __init__(
        self,
        stage_names: list[str],
        message_queue: mp.Queue[TuiMessage],
        executor_func: Callable[[], dict[str, executor.ExecutionSummary]],
    ) -> None:
        super().__init__()
        self._stage_names = stage_names
        self._tui_queue = message_queue
        self._executor_func = executor_func

        # Stage state - use dict comprehension (ordered in Python 3.7+)
        self._stages = {
            name: StageInfo(name, i, len(stage_names)) for i, name in enumerate(stage_names, 1)
        }

        self._selected_idx = 0
        self._show_logs = False
        self._results = None
        self._error = None

        # Stats tracking
        self._stats = PipelineStats(total=len(stage_names))
        self._completed_times = list[float]()

        # Threading
        self._reader_thread = None
        self._executor_thread = None
        self._shutdown_event = threading.Event()

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        # Status bar at very top
        yield StatusBar("pivot run", len(self._stages), id="status-bar")

        # Stages view (default) - horizontal split
        with textual.containers.Horizontal(id="stages-view", classes="view-active"):
            yield StageListPanel(list(self._stages.values()), id="stage-list")
            yield DetailPanel(id="detail-panel")

        # All logs view (toggled with 'a')
        with textual.containers.Vertical(id="logs-view", classes="view-hidden"):
            yield textual.widgets.Static(
                "[bold]All Logs[/] │ [dim]Press 's' for stages, '1-9' to filter[/]",
                id="all-logs-header",
            )
            yield LogPanel(id="log-panel")

        yield textual.widgets.Footer()

    async def on_mount(self) -> None:  # pragma: no cover
        self._update_detail_panel()

        # Start reader thread immediately (it just waits on queue)
        self._reader_thread = threading.Thread(target=self._read_queue, daemon=True)
        self._reader_thread.start()

        # Delay executor start until after widgets are fully mounted
        # This prevents race condition where status updates arrive before UI is ready
        self.call_later(self._start_executor)

    def _start_executor(self) -> None:  # pragma: no cover
        """Start the executor thread after widgets are mounted."""
        self._executor_thread = threading.Thread(target=self._run_executor, daemon=True)
        self._executor_thread.start()

    def _read_queue(self) -> None:  # pragma: no cover
        """Read from queue and post messages to Textual (runs in background thread)."""
        while not self._shutdown_event.is_set():
            try:
                msg = self._tui_queue.get(timeout=0.1)
                if msg is None:
                    break
                self.post_message(TuiUpdate(msg))
            except queue.Empty:
                continue

    def _run_executor(self) -> None:  # pragma: no cover
        """Run the executor (runs in background thread)."""
        results: dict[str, executor.ExecutionSummary] = {}
        error: Exception | None = None
        try:
            results = self._executor_func()
        except Exception as e:
            error = e
        finally:
            # Guard queue.put in case manager died
            with contextlib.suppress(Exception):
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
            case TuiMessageType.REACTIVE:
                pass  # Reactive messages handled separately in reactive mode

    def _handle_log(self, msg: TuiLogMessage) -> None:  # pragma: no cover
        stage = msg["stage"]
        line = msg["line"]
        is_stderr = msg["is_stderr"]

        # Add to stage's log buffer
        if stage in self._stages:
            self._stages[stage].logs.append((line, is_stderr))

        # Add to all-logs panel
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log(stage, line, is_stderr)

        # Add to detail panel (for selected stage view)
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.add_log(stage, line, is_stderr)

    def _handle_status(self, msg: TuiStatusMessage) -> None:  # pragma: no cover
        stage = msg["stage"]
        if stage not in self._stages:
            return

        info = self._stages[stage]
        old_status = info.status
        info.status = msg["status"]
        info.reason = msg["reason"]
        info.elapsed = msg["elapsed"]

        # Update stage list UI
        stage_list = self.query_one("#stage-list", StageListPanel)
        stage_list.update_stage(stage)
        self._update_detail_panel()

        # Update stats incrementally
        self._update_stats_incremental(old_status, info)

    def _update_stats_incremental(
        self, old_status: StageStatus, info: StageInfo
    ) -> None:  # pragma: no cover
        """Update pipeline statistics incrementally (O(1) instead of O(n))."""
        new_status = info.status

        # Decrement counters for old status
        if old_status in _TERMINAL_STATES:
            self._stats.completed -= 1
        if old_status == StageStatus.FAILED:
            self._stats.failed -= 1
        if old_status == StageStatus.IN_PROGRESS:
            self._stats.running -= 1

        # Increment counters for new status
        if new_status in _TERMINAL_STATES:
            self._stats.completed += 1
        if new_status == StageStatus.FAILED:
            self._stats.failed += 1
        if new_status == StageStatus.IN_PROGRESS:
            self._stats.running += 1

        # Track elapsed times for ETA calculation
        if (
            new_status in _TERMINAL_STATES
            and old_status == StageStatus.IN_PROGRESS
            and info.elapsed is not None
        ):
            self._completed_times.append(info.elapsed)
            self._stats.avg_stage_time = sum(self._completed_times) / len(self._completed_times)

        # Update status bar
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_stats(self._stats)

    def on_executor_complete(self, event: ExecutorComplete) -> None:  # pragma: no cover
        """Handle executor completion."""
        self._results = event.results
        self._error = event.error
        # Textual's Reactive[str] title attribute
        if event.error:
            self.title = f"pivot run - FAILED: {event.error}"  # pyright: ignore[reportUnannotatedClassAttribute]
        else:
            self.title = "pivot run - Complete"

    def _update_detail_panel(self) -> None:  # pragma: no cover
        stage_names = list(self._stages.keys())
        if stage_names:
            selected_name = stage_names[self._selected_idx]
            stage = self._stages.get(selected_name)
        else:
            stage = None
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.set_stage(stage)

    def action_next_stage(self) -> None:  # pragma: no cover
        if self._selected_idx < len(self._stages) - 1:
            self._selected_idx += 1
            self._update_detail_panel()

    def action_prev_stage(self) -> None:  # pragma: no cover
        if self._selected_idx > 0:
            self._selected_idx -= 1
            self._update_detail_panel()

    def _set_view(self, show_logs: bool) -> None:  # pragma: no cover
        """Switch between stages view and all-logs view."""
        self._show_logs = show_logs
        self.query_one("#stages-view").set_classes("view-hidden" if show_logs else "view-active")
        self.query_one("#logs-view").set_classes("view-active" if show_logs else "view-hidden")

    def action_show_all_logs(self) -> None:  # pragma: no cover
        """Show all logs view (interleaved)."""
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.set_filter(None)
        self._set_view(show_logs=True)

    def action_show_stages(self) -> None:  # pragma: no cover
        """Show stages view (split panel)."""
        self._set_view(show_logs=False)

    def action_filter_stage(self, idx: int) -> None:  # pragma: no cover
        """Filter logs to stage at index idx (0-based)."""
        stage_names = list(self._stages.keys())
        if idx < len(stage_names):
            stage_name = stage_names[idx]
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.set_filter(stage_name)
            self._set_view(show_logs=True)

    @override
    async def action_quit(self) -> None:  # pragma: no cover
        self._shutdown_event.set()
        # Give reader thread time to notice shutdown and exit cleanly
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.2)
        await super().action_quit()


def run_with_tui(
    stage_names: list[str],
    message_queue: mp.Queue[TuiMessage],
    executor_func: Callable[[], dict[str, executor.ExecutionSummary]],
) -> dict[str, executor.ExecutionSummary] | None:  # pragma: no cover
    """Run pipeline with TUI display."""
    app = RunTuiApp(stage_names, message_queue, executor_func)
    return app.run()


def should_use_tui(display_mode: DisplayMode | None) -> bool:
    """Determine if TUI should be used based on display mode and TTY."""
    import sys

    match display_mode:
        case DisplayMode.TUI:
            return True
        case DisplayMode.PLAIN:
            return False
        case None:
            return sys.stdout.isatty()
