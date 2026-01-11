from __future__ import annotations

import asyncio
import atexit
import collections
import dataclasses
import json
import logging
import os
import queue
import threading
from typing import IO, TYPE_CHECKING, ClassVar, TypeVar, final, override

import filelock
import textual.app
import textual.binding
import textual.containers
import textual.message
import textual.screen
import textual.widgets

from pivot import project
from pivot.executor import ExecutionSummary
from pivot.executor import commit as commit_mod
from pivot.storage import lock, project_lock
from pivot.types import (
    DisplayMode,
    ReactiveStatus,
    StageStatus,
    TuiLogMessage,
    TuiMessage,
    TuiMessageType,
    TuiReactiveMessage,
    TuiReloadMessage,
    TuiStatusMessage,
)

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable
    from pathlib import Path
    from typing import Protocol

    from pivot.types import OutputMessage

    class ReactiveEngineProtocol(Protocol):
        """Protocol for ReactiveEngine to avoid circular imports."""

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
    logs: collections.deque[tuple[str, bool]] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )


class TuiUpdate(textual.message.Message):
    """Custom message for executor updates."""

    msg: TuiLogMessage | TuiStatusMessage | TuiReactiveMessage | TuiReloadMessage

    def __init__(
        self, msg: TuiLogMessage | TuiStatusMessage | TuiReactiveMessage | TuiReloadMessage
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
        reason_str = f"  ({self._info.reason})" if self._info.reason else ""
        text = f"{index_str} {self._info.name:<20} [{style}]{label}[/]{elapsed_str}{reason_str}"
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
            f"[bold]Stage:[/] {self._stage.name}",
            f"[bold]Status:[/] [{style}]{label}[/]{elapsed_str}",
        ]
        if self._stage.reason:
            lines.append(f"[bold]Reason:[/] {self._stage.reason}")

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
        prefix = f"[cyan][{stage}][/] "
        if is_stderr:
            self.write(f"{prefix}[red]{line}[/]")
        else:
            self.write(f"{prefix}{line}")

    def set_filter(self, stage: str | None) -> None:  # pragma: no cover
        """Filter logs to a specific stage or show all (None)."""
        self._filter_stage = stage
        self.clear()
        for s, line, is_stderr in self._all_logs:
            if stage is None or s == stage:
                self._write_log_line(s, line, is_stderr)


_TUI_CSS: str = """
#stage-list {
    height: auto;
    max-height: 50%;
    border: solid green;
    padding: 1;
}

#detail-panel {
    height: auto;
    border: solid blue;
    padding: 1;
    margin-top: 1;
}

#log-panel {
    height: 1fr;
    border: solid yellow;
    margin-top: 1;
}

.section-header {
    text-style: bold;
    margin-bottom: 1;
}

#status-view {
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

_TUI_BINDINGS: list[textual.binding.BindingType] = [
    textual.binding.Binding("q", "quit", "Quit"),
    textual.binding.Binding("c", "commit", "Commit"),
    textual.binding.Binding("escape", "cancel_commit", "Cancel", show=False),
    textual.binding.Binding("j", "next_stage", "Next"),
    textual.binding.Binding("k", "prev_stage", "Prev"),
    textual.binding.Binding("down", "next_stage", "Next", show=False),
    textual.binding.Binding("up", "prev_stage", "Prev", show=False),
    textual.binding.Binding("tab", "toggle_view", "Toggle View"),
    textual.binding.Binding("a", "show_all_logs", "All Logs"),
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
        self._show_logs: bool = False
        self._reader_thread: threading.Thread | None = None
        self._shutdown_event: threading.Event = threading.Event()
        self._log_file: IO[str] | None = None

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

    def _close_log_file(self) -> None:
        """Close the log file if open (thread-safe)."""
        # Swap-then-check pattern avoids race condition
        log_file = self._log_file
        self._log_file = None
        if log_file:
            atexit.unregister(self._close_log_file)
            log_file.close()

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

        with textual.containers.VerticalScroll(id="status-view", classes="view-active"):
            yield StageListPanel(list(self._stages.values()), id="stage-list")
            yield DetailPanel(id="detail-panel")

        with textual.containers.Vertical(id="logs-view", classes="view-hidden"):
            yield LogPanel(id="log-panel")

        yield textual.widgets.Footer()

    def _start_queue_reader(self) -> None:  # pragma: no cover
        """Start the background queue reader thread."""
        self._reader_thread = threading.Thread(target=self._read_queue, daemon=True)
        self._reader_thread.start()

    def _read_queue(self) -> None:  # pragma: no cover
        """Read from queue and post messages to Textual (runs in background thread)."""
        while not self._shutdown_event.is_set():
            try:
                msg = self._tui_queue.get(timeout=0.1)
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

        if stage in self._stages:
            self._stages[stage].logs.append((line, is_stderr))

        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.add_log(stage, line, is_stderr)

    def _update_detail_panel(self) -> None:  # pragma: no cover
        if self._stage_order:
            selected_name = self._stage_order[self._selected_idx]
            stage = self._stages.get(selected_name)
        else:
            stage = None
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.set_stage(stage)

    def action_next_stage(self) -> None:  # pragma: no cover
        if self._selected_idx < len(self._stage_order) - 1:
            self._selected_idx += 1
            self._update_detail_panel()

    def action_prev_stage(self) -> None:  # pragma: no cover
        if self._selected_idx > 0:
            self._selected_idx -= 1
            self._update_detail_panel()

    def action_toggle_view(self) -> None:  # pragma: no cover
        self._show_logs = not self._show_logs
        status_view = self.query_one("#status-view")
        logs_view = self.query_one("#logs-view")
        if self._show_logs:
            status_view.set_classes("view-hidden")
            logs_view.set_classes("view-active")
        else:
            status_view.set_classes("view-active")
            logs_view.set_classes("view-hidden")

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

    def action_cancel_commit(self) -> None:  # pragma: no cover
        """Cancel commit operation. Override in subclasses that support commit."""
        pass

    @override
    async def action_quit(self) -> None:  # pragma: no cover
        self._shutdown_event.set()
        # Wait for reader thread to finish before closing log file (avoids race)
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
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

    async def on_mount(self) -> None:  # pragma: no cover
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
            case TuiMessageType.REACTIVE | TuiMessageType.RELOAD:
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
        engine: ReactiveEngineProtocol,
        message_queue: mp.Queue[TuiMessage],
        output_queue: mp.Queue[OutputMessage] | None = None,
        tui_log: Path | None = None,
        stage_names: list[str] | None = None,
        *,
        no_commit: bool = False,
    ) -> None:
        super().__init__(message_queue, stage_names, tui_log=tui_log)
        self._engine: ReactiveEngineProtocol = engine
        self._output_queue = output_queue
        self._engine_thread: threading.Thread | None = None
        self._no_commit: bool = no_commit
        self._commit_in_progress: bool = False
        self._cancel_commit: bool = False

    @property
    def _has_running_stages(self) -> bool:
        """Check if any stages are currently in progress."""
        return any(s.status == StageStatus.IN_PROGRESS for s in self._stages.values())

    async def on_mount(self) -> None:  # pragma: no cover
        self.title = "[●] Watching for changes..."
        self._start_queue_reader()
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

    def _run_engine(self) -> None:  # pragma: no cover
        """Run the reactive engine (runs in background thread)."""
        try:
            self._engine.run(tui_queue=self._tui_queue, output_queue=self._output_queue)
        except Exception as e:
            logging.getLogger(__name__).exception(f"Reactive engine failed: {e}")

    def on_tui_update(self, event: TuiUpdate) -> None:  # pragma: no cover
        """Handle executor updates in Textual's event loop."""
        msg = event.msg
        match msg["type"]:
            case TuiMessageType.LOG:
                self._handle_log(msg)
            case TuiMessageType.STATUS:
                self._handle_status(msg)
            case TuiMessageType.REACTIVE:
                self._handle_reactive(msg)
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

    def _handle_reactive(self, msg: TuiReactiveMessage) -> None:  # pragma: no cover
        """Handle reactive status updates - update title bar."""
        match msg["status"]:
            case ReactiveStatus.WAITING:
                self.title = "[●] Watching for changes..."
            case ReactiveStatus.RESTARTING:
                self.title = "[↻] Reloading code..."
            case ReactiveStatus.DETECTING:
                self.title = f"[▶] {msg['message']}"
            case ReactiveStatus.ERROR:
                self.title = f"[!] {msg['message']}"

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

        if self._selected_idx >= len(self._stage_order):
            self._selected_idx = max(0, len(self._stage_order) - 1)

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
                if acquired is not None:
                    acquired.release()
                self._commit_in_progress = False
                self.notify("Commit cancelled")
                return

            if acquired is None:
                self._commit_in_progress = False
                self.notify(
                    f"Timed out waiting for lock ({int(_COMMIT_LOCK_TIMEOUT)}s). Try again later.",
                    severity="error",
                )
                return

            await self._do_commit(acquired)
        except Exception as e:
            self.notify(f"Commit failed: {e}", severity="error")
        finally:
            if acquired is None:
                self._commit_in_progress = False

    async def _do_commit(self, acquired_lock: filelock.BaseFileLock) -> None:  # pragma: no cover
        """Execute the commit operation with the acquired lock."""
        try:
            committed = await asyncio.to_thread(commit_mod.commit_pending)
            self.notify(f"Committed {len(committed)} stage(s)")
        except Exception as e:
            self.notify(f"Commit failed: {e}", severity="error")
        finally:
            acquired_lock.release()
            self._commit_in_progress = False

    @override
    def action_cancel_commit(self) -> None:  # pragma: no cover
        """Cancel waiting for commit lock."""
        if self._commit_in_progress:
            self._cancel_commit = True

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
    engine: ReactiveEngineProtocol,
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
