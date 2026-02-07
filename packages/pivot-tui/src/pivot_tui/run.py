from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import json
import logging
import os
import pathlib
import threading
import time
from typing import (
    IO,
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    override,
)

import filelock
import loky
import loky.process_executor
import rich.markup
import textual  # for textual.work decorator
import textual.app
import textual.binding
import textual.containers
import textual.css.query
import textual.events
import textual.message
import textual.timer
import textual.widget
import textual.widgets

from pivot import config, explain, parameters, project
from pivot.executor import ExecutionSummary
from pivot.executor import commit as commit_mod
from pivot.storage import lock, project_lock
from pivot.types import (
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)
from pivot_tui import diff_panels, rpc_client
from pivot_tui.diff_panels import InputDiffPanel, OutputDiffPanel
from pivot_tui.screens import (
    ConfirmCommitScreen,
    ConfirmKillWorkersScreen,
    HelpScreen,
    HistoryListScreen,
)
from pivot_tui.stats import DebugStats, MessageStatsTracker, get_memory_mb
from pivot_tui.types import (
    ExecutionHistoryEntry,
    LogEntry,
    PendingHistoryState,
    StageDataProvider,
    StageInfo,
)
from pivot_tui.widgets import (
    DebugPanel,
    FooterContext,
    PivotFooter,
    StageListPanel,
    StageLogPanel,
    TabbedDetailPanel,
)
from pivot_tui.widgets import status as status_utils

__all__ = [
    "MessagePoster",
    "PivotApp",
    "TuiShutdown",
    "TuiUpdate",
    "format_reload_summary",
]


def format_reload_summary(
    stages_added: list[str],
    stages_removed: list[str],
    stages_modified: list[str],
) -> str | None:
    """Format a summary message for pipeline reload changes.

    Returns None if there are no changes to report.
    """
    parts = list[str]()

    if stages_added:
        parts.append(f"{len(stages_added)} added")
    if stages_removed:
        parts.append(f"{len(stages_removed)} removed")
    if stages_modified:
        parts.append(f"{len(stages_modified)} modified")

    if not parts:
        return None

    return f"Reloaded: {', '.join(parts)}"


if TYPE_CHECKING:
    from pivot.types import OutputChange


class MessagePoster(Protocol):
    """Protocol for posting messages to a Textual app.

    Textual's post_message() is thread-safe and can be called from any thread.
    See: https://textual.textualize.io/guide/workers/#posting-messages
    """

    def post_message(self, message: textual.message.Message) -> bool:
        """Post a message to the app's message queue."""
        ...


class TuiUpdate(textual.message.Message):
    """Custom message for executor updates."""

    msg: TuiLogMessage | TuiStatusMessage | TuiWatchMessage | TuiReloadMessage

    def __init__(
        self, msg: TuiLogMessage | TuiStatusMessage | TuiWatchMessage | TuiReloadMessage
    ) -> None:
        self.msg = msg
        super().__init__()


class TuiShutdown(textual.message.Message):
    """Message sent when TuiSink is closing.

    Used by both run mode and watch mode to signal clean shutdown.
    """


_TUI_BINDINGS: list[textual.binding.BindingType] = [
    textual.binding.Binding("q", "quit", "Quit"),
    textual.binding.Binding("c", "commit", "Commit"),
    textual.binding.Binding("?", "show_help", "Help"),
    textual.binding.Binding("escape", "escape_action", "Cancel/Close", show=False),
    textual.binding.Binding("enter", "toggle_group", "Toggle Group", show=False),
    # Stage navigation (always navigates stage list)
    textual.binding.Binding("j", "nav_down", "Down"),
    textual.binding.Binding("k", "nav_up", "Up"),
    textual.binding.Binding("down", "nav_down", "Down", show=False),
    textual.binding.Binding("up", "nav_up", "Up", show=False),
    # Tab navigation
    textual.binding.Binding("tab", "next_tab", "Next Tab"),
    textual.binding.Binding("h", "prev_tab", "Prev Tab", show=False),
    textual.binding.Binding("l", "next_tab", "Next Tab", show=False),
    textual.binding.Binding("left", "prev_tab", "Prev Tab", show=False),
    textual.binding.Binding("right", "next_tab", "Next Tab", show=False),
    # Detail content scrolling
    textual.binding.Binding("ctrl+j", "scroll_detail_down", "Scroll Down", show=False),
    textual.binding.Binding("ctrl+k", "scroll_detail_up", "Scroll Up", show=False),
    # Changed-item navigation
    textual.binding.Binding("n", "next_changed", "Next Change", show=False),
    textual.binding.Binding("N", "prev_changed", "Prev Change", show=False),
    # Group collapse/expand
    textual.binding.Binding("-", "collapse_all_groups", "Collapse All", show=False),
    textual.binding.Binding("=", "expand_all_groups", "Expand All", show=False),
    # Stage filtering
    textual.binding.Binding("/", "focus_filter", "Filter", show=False),
    # Log search (Logs tab only)
    textual.binding.Binding("ctrl+f", "log_search", "Search Logs", show=False),
    # History navigation (works in all tabs, watch mode only)
    textual.binding.Binding("[", "history_older", "Older", show=False),
    textual.binding.Binding("]", "history_newer", "Newer", show=False),
    textual.binding.Binding("G", "history_live", "Live View", show=False),
    textual.binding.Binding("H", "show_history_list", "History", show=False),
    # Tab mnemonic keys (shift+letter)
    textual.binding.Binding("L", "goto_tab_logs", "Logs Tab", show=False),
    textual.binding.Binding("I", "goto_tab_input", "Input Tab", show=False),
    textual.binding.Binding("O", "goto_tab_output", "Output Tab", show=False),
    # Debug panel toggle
    textual.binding.Binding("~", "toggle_debug", "Debug"),
    # Keep-going toggle (watch mode only)
    textual.binding.Binding("g", "toggle_keep_going", "Keep-going"),
    # Force re-run (watch mode only)
    textual.binding.Binding("r", "force_rerun_stage", "Force Re-run", show=False),
    textual.binding.Binding("R", "force_rerun_all", "Force All", show=False),
]

_logger = logging.getLogger(__name__)

# Minimum terminal size for proper display
_MIN_TERMINAL_WIDTH = 80
_MIN_TERMINAL_HEIGHT = 24

# Constants for commit lock acquisition
_COMMIT_LOCK_POLL_INTERVAL = 5.0  # seconds between lock attempts
_COMMIT_LOCK_TIMEOUT = 60.0  # total seconds before giving up


class PivotApp(textual.app.App[dict[str, ExecutionSummary] | None]):
    """Unified TUI application for both run and watch modes."""

    CSS_PATH: ClassVar[str | pathlib.PurePath | list[str | pathlib.PurePath] | None] = (
        pathlib.Path(__file__).parent / "styles" / "pivot.tcss"
    )
    BINDINGS: ClassVar[list[textual.binding.BindingType]] = _TUI_BINDINGS
    _TAB_IDS: ClassVar[tuple[str, str, str]] = ("tab-logs", "tab-input", "tab-output")

    # Instance attributes (annotated for type checking since class is not @final)
    _cancel_event: threading.Event | None
    _stage_data_provider: StageDataProvider | None
    _stages: dict[str, StageInfo]
    _stage_order: list[str]
    _pending_history: dict[str, PendingHistoryState]

    def __init__(
        self,
        stage_names: list[str] | None = None,
        tui_log: pathlib.Path | None = None,
        *,
        cancel_event: threading.Event | None = None,
        watch_mode: bool = False,
        no_commit: bool = False,
        serve: bool = False,
        stage_data_provider: StageDataProvider | None = None,
    ) -> None:
        """Initialize TUI app.

        TUI is a pure display client. Engine is always managed externally by CLI.
        TUI waits for TuiShutdown message to exit (run mode) or user quit (watch mode).
        """
        super().__init__()

        self._watch_mode: bool = watch_mode

        # Core state
        self._stages = dict[str, StageInfo]()
        self._stage_order = list[str]()
        self._selected_idx: int = 0
        self._selected_stage_name: str | None = None
        self._shutdown_event: threading.Event = threading.Event()
        self._log_file: IO[str] | None = None

        # Debug panel stats tracking
        self._message_stats: MessageStatsTracker = MessageStatsTracker("tui_messages")
        self._start_time: float = 0.0
        self._debug_timer: textual.timer.Timer | None = None
        self._stats_log_timer: textual.timer.Timer | None = None

        # Run mode state
        self._cancel_event = cancel_event
        self._results: dict[str, ExecutionSummary] | None = None
        self._exit_message: str | None = None

        # Watch mode state
        self._no_commit: bool = no_commit
        self._commit_in_progress: bool = False
        self._cancel_commit: bool = False
        self._viewing_history_index: int | None = None
        self._pending_history = dict[str, PendingHistoryState]()
        self._current_run_id: str | None = None
        self._serve: bool = serve
        self._quitting: bool = False
        self._quit_lock: threading.Lock = threading.Lock()
        self._stage_data_provider = stage_data_provider
        self._log_file_lock: threading.Lock = threading.Lock()

        # Open log file if configured
        if tui_log:
            log_file = open(tui_log, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
            # set_inheritable failure is non-fatal, just means child processes
            # might inherit the fd (harmless for a log file)
            with contextlib.suppress(OSError):
                os.set_inheritable(log_file.fileno(), False)
            self._log_file = log_file
            atexit.register(self._close_log_file)

        # Initialize stages
        if stage_names:
            for i, name in enumerate(stage_names, 1):
                info = StageInfo(name, i, len(stage_names))
                self._stages[name] = info
                self._stage_order.append(name)
            self._select_stage(0)

    @property
    def selected_stage_name(self) -> str | None:
        """Return name of currently selected stage, or None if no stages."""
        return self._selected_stage_name

    @property
    def exit_message(self) -> str | None:
        """Return message to display after TUI exits (e.g., running stages warning)."""
        return self._exit_message

    @property
    def _has_running_stages(self) -> bool:
        """Check if any stages are currently in progress."""
        return any(s.status == StageStatus.IN_PROGRESS for s in self._stages.values())

    def _shutdown_loky_pool(self) -> None:
        """Force-kill loky worker pool to prevent hang on exit."""
        # Best-effort cleanup - loky can fail with fd errors in some environments
        with contextlib.suppress(Exception):
            # kill_workers=True forces immediate termination of worker processes
            loky.get_reusable_executor(max_workers=1, kill_workers=True)

    def _close_log_file(self) -> None:
        """Close the log file if open (thread-safe)."""
        with self._log_file_lock:
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
        """Recompute index from name after stage list changes."""
        if self._selected_stage_name and self._selected_stage_name in self._stage_order:
            self._selected_idx = self._stage_order.index(self._selected_stage_name)
        else:
            self._selected_idx = 0
            self._selected_stage_name = self._stage_order[0] if self._stage_order else None

    def _try_query_one[W: textual.widget.Widget](
        self, selector: str, widget_type: type[W]
    ) -> W | None:
        """Query for a widget, returning None if not found."""
        try:
            return self.query_one(selector, widget_type)
        except textual.css.query.NoMatches:
            return None

    def _write_to_log(self, data: str) -> None:  # pragma: no cover
        """Write a line to the log file (thread-safe)."""
        with self._log_file_lock:
            if self._log_file:
                try:
                    self._log_file.write(data)
                except OSError as e:
                    _logger.warning(f"TUI log write failed: {e}")
                    # Close the file to avoid fd leak before clearing reference
                    with contextlib.suppress(OSError):
                        self._log_file.close()
                    self._log_file = None

    @override
    def compose(self) -> textual.app.ComposeResult:  # pragma: no cover
        yield textual.widgets.Header()

        with textual.containers.Horizontal(id="main-split"):
            yield StageListPanel(list(self._stages.values()), id="stage-list")
            yield TabbedDetailPanel(
                id="detail-panel", stage_data_provider=self._stage_data_provider
            )

        yield DebugPanel(id="debug-panel")
        yield PivotFooter(id="pivot-footer")

    async def on_mount(self) -> None:  # pragma: no cover
        """Initialize TUI on mount."""
        self._start_time = time.monotonic()
        if self._log_file is not None:
            self._stats_log_timer = self.set_interval(1.0, self._write_stats_to_log)
        self.call_after_refresh(self._update_detail_panel)

        if self._watch_mode:
            prefix = self._get_keep_going_prefix()
            self.title = f"{prefix}[●] Watching for changes..."  # pyright: ignore[reportUnannotatedClassAttribute] - inherited from App
        # Run mode: TUI waits for TuiShutdown from external engine

    def on_resize(self, event: textual.events.Resize) -> None:  # pragma: no cover
        """Handle terminal resize - warn if too small."""
        if event.size.width < _MIN_TERMINAL_WIDTH or event.size.height < _MIN_TERMINAL_HEIGHT:
            msg = (
                f"Terminal too small ({event.size.width}x{event.size.height}). "
                f"Minimum: {_MIN_TERMINAL_WIDTH}x{_MIN_TERMINAL_HEIGHT}"
            )
            self.notify(msg, severity="warning", timeout=5)

    # =========================================================================
    # Message handling
    # =========================================================================

    def on_tui_update(self, event: TuiUpdate) -> None:  # pragma: no cover
        """Handle executor/engine updates."""
        self._message_stats.record_message()
        self._write_to_log(json.dumps(event.msg, default=str) + "\n")
        msg = event.msg
        try:
            match msg["type"]:
                case TuiMessageType.LOG:
                    self._handle_log(msg)
                case TuiMessageType.STATUS:
                    self._handle_status(msg)
                case TuiMessageType.WATCH:
                    self._handle_watch(msg)
                case TuiMessageType.RELOAD:
                    self._handle_reload(msg)
        except Exception:
            _logger.exception("Error handling TUI update for %s: %s", msg["type"], msg)

    def on_tui_shutdown(self, _event: TuiShutdown) -> None:  # pragma: no cover
        """Handle TuiSink shutdown signal (both run and watch modes).

        In run mode: signals execution is complete, triggers exit with bell notification
        In watch mode: logs shutdown, TUI continues until user quits
        """
        self._write_to_log('{"type": "shutdown"}\n')
        if not self._watch_mode:
            # Run mode: notify user and exit after shutdown signal
            self.bell()
            self._shutdown_event.set()
            self._close_log_file()
            self._shutdown_loky_pool()
            self.exit(self._results)

    def _handle_log(self, msg: TuiLogMessage) -> None:  # pragma: no cover
        """Handle log message."""
        stage = msg["stage"]
        line = msg["line"]
        is_stderr = msg["is_stderr"]
        timestamp = msg["timestamp"]

        log_entry = LogEntry(line, is_stderr, timestamp)

        # Create stage if it doesn't exist yet (logs can arrive before status)
        if stage not in self._stages:
            info = StageInfo(stage, 0, 0)  # Placeholder values, updated by status
            self._stages[stage] = info
            self._stage_order.append(stage)

        self._stages[stage].logs.append(log_entry)

        # Update stage-specific log panel if this stage is selected
        if self._selected_stage_name == stage and (
            stage_log_panel := self._try_query_one("#stage-logs", StageLogPanel)
        ):
            stage_log_panel.add_log(line, is_stderr, timestamp)

        # Watch mode: accumulate logs for history
        if self._watch_mode and stage in self._pending_history:
            self._pending_history[stage].logs.append(log_entry)

    def _handle_status(self, msg: TuiStatusMessage) -> None:  # pragma: no cover
        """Handle status update."""
        stage = msg["stage"]
        status = msg["status"]
        run_id = msg["run_id"]
        is_new_stage = stage not in self._stages

        # Watch mode: detect new pipeline run and manage history
        if self._watch_mode and run_id and run_id != self._current_run_id:
            if self._pending_history:
                _logger.debug(
                    "New run %s detected, clearing %d stale pending entries",
                    run_id,
                    len(self._pending_history),
                )
                self._pending_history.clear()
            for stage_info in self._stages.values():
                stage_info.live_input_snapshot = None
                stage_info.live_output_snapshot = None
            self._current_run_id = run_id
            self._update_detail_panel()

        if is_new_stage:
            info = StageInfo(stage, msg["index"], msg["total"])
            self._stages[stage] = info
            self._stage_order.append(stage)
        else:
            info = self._stages[stage]

        info.status = status
        info.reason = msg["reason"]
        info.elapsed = msg["elapsed"]
        # Only update index/total for dynamically discovered stages (watch mode).
        # Pre-initialized stages already have correct display indices.
        if is_new_stage:
            info.index = msg["index"]
            info.total = msg["total"]

        # Watch mode: history capture
        if self._watch_mode:
            if status == StageStatus.IN_PROGRESS:
                self._create_history_entry(stage, run_id)
            elif status in (
                StageStatus.RAN,
                StageStatus.COMPLETED,
                StageStatus.SKIPPED,
                StageStatus.FAILED,
            ):
                self._finalize_history_entry(stage, status, msg["reason"], msg["elapsed"], run_id)

        # Update UI
        if is_new_stage and self._watch_mode:
            self._rebuild_stage_list()
        else:
            stage_list = self._try_query_one("#stage-list", StageListPanel)
            if stage_list:
                # Pass info to sync row's StageInfo reference (fixes reference divergence)
                stage_list.update_stage(stage, self._selected_stage_name, info=info)
            if stage == self._selected_stage_name or not self._watch_mode:
                self._update_detail_panel()

        # Run mode: update title with progress
        if not self._watch_mode:
            completed = sum(
                1
                for s in self._stages.values()
                if s.status
                in (StageStatus.COMPLETED, StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED)
            )
            self.title = f"pivot run ({completed}/{len(self._stages)})"

    def _handle_watch(self, msg: TuiWatchMessage) -> None:  # pragma: no cover
        """Handle watch status update (watch mode only)."""
        if not self._watch_mode:
            return
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
        """Handle registry reload (watch mode only)."""
        if not self._watch_mode:
            return

        new_stages = msg["stages"]
        old_stages = set(self._stage_order)
        new_stage_set = set(new_stages)

        for name in old_stages - new_stage_set:
            if name in self._stages:
                del self._stages[name]
            self._pending_history.pop(name, None)
            if self._viewing_history_index is not None and self._selected_stage_name == name:
                self._viewing_history_index = None

        for name in new_stage_set - old_stages:
            info = StageInfo(name, len(self._stages) + 1, len(new_stages))
            self._stages[name] = info

        self._stage_order = new_stages
        for i, name in enumerate(self._stage_order, 1):
            if name in self._stages:
                self._stages[name].index = i
                self._stages[name].total = len(self._stage_order)

        self._recompute_selection_idx()
        self._rebuild_stage_list()
        self._update_detail_panel()

        # Show notification with reload summary
        summary = format_reload_summary(
            stages_added=msg["stages_added"],
            stages_removed=msg["stages_removed"],
            stages_modified=msg["stages_modified"],
        )
        if summary:
            self.notify(summary)

    # =========================================================================
    # History tracking (watch mode only)
    # =========================================================================

    def _create_history_entry(self, stage_name: str, run_id: str) -> None:
        """Create a new history entry when stage starts executing."""
        input_snapshot = None
        if self._stage_data_provider is not None:
            try:
                registry_info = self._stage_data_provider.get_stage(stage_name)
                fingerprint = self._stage_data_provider.ensure_fingerprint(stage_name)
                state_dir = config.get_state_dir()
                input_snapshot = explain.get_stage_explanation(
                    stage_name=stage_name,
                    fingerprint=fingerprint,
                    deps=registry_info["deps_paths"],
                    outs_paths=registry_info["outs_paths"],
                    params_instance=registry_info["params"],
                    overrides=parameters.load_params_yaml(),
                    state_dir=state_dir,
                )
            except Exception:
                _logger.debug("Failed to capture input snapshot for %s", stage_name)

        self._pending_history[stage_name] = PendingHistoryState(
            run_id=run_id,
            timestamp=time.time(),
            input_snapshot=input_snapshot,
        )

        if self._viewing_history_index is None and stage_name in self._stages:
            self._stages[stage_name].live_input_snapshot = input_snapshot

    def _finalize_history_entry(
        self,
        stage_name: str,
        status: StageStatus,
        reason: str,
        elapsed: float | None,
        run_id: str | None = None,
    ) -> None:
        """Finalize history entry when stage completes."""
        if stage_name not in self._stages:
            return

        info = self._stages[stage_name]
        pending = self._pending_history.pop(stage_name, None)

        if pending is None:
            if status == StageStatus.SKIPPED and run_id:
                entry = ExecutionHistoryEntry(
                    run_id=run_id,
                    stage_name=stage_name,
                    timestamp=time.time(),
                    duration=None,
                    status=status,
                    reason=reason,
                    logs=[],
                    input_snapshot=None,
                    output_snapshot=None,
                )
            else:
                return
        else:
            output_snapshot: list[OutputChange] | None = None
            if self._stage_data_provider is not None:
                try:
                    registry_info = self._stage_data_provider.get_stage(stage_name)
                    state_dir = config.get_state_dir()
                    stages_dir = lock.get_stages_dir(state_dir)
                    lock_data = lock.StageLock(stage_name, stages_dir).read()
                    output_snapshot = diff_panels.compute_output_changes(lock_data, registry_info)
                except Exception:
                    _logger.debug("Failed to capture output snapshot for %s", stage_name)

            if self._viewing_history_index is None:
                info.live_output_snapshot = output_snapshot

            entry = ExecutionHistoryEntry(
                run_id=pending.run_id,
                stage_name=stage_name,
                timestamp=pending.timestamp,
                duration=elapsed,
                status=status,
                reason=reason,
                logs=list(pending.logs),
                input_snapshot=pending.input_snapshot,
                output_snapshot=output_snapshot,
            )

        was_at_capacity = len(info.history) == info.history.maxlen
        info.history.append(entry)

        if (
            was_at_capacity
            and stage_name == self._selected_stage_name
            and self._viewing_history_index is not None
        ):
            if self._viewing_history_index == 0:
                self._viewing_history_index = None
            else:
                self._viewing_history_index -= 1

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
        return collections.deque[ExecutionHistoryEntry]()

    def _navigate_history_prev(self) -> bool:  # pragma: no cover
        """Navigate to previous (older) history entry."""
        history = self._get_current_stage_history()
        if not history:
            return False

        if self._viewing_history_index is None:
            self._viewing_history_index = len(history) - 1
        elif self._viewing_history_index > 0:
            self._viewing_history_index -= 1
        else:
            return False

        self._update_history_view()
        return True

    def _navigate_history_next(self) -> bool:  # pragma: no cover
        """Navigate to next (newer) history entry or live view."""
        history = self._get_current_stage_history()

        if self._viewing_history_index is None:
            return False

        if self._viewing_history_index < len(history) - 1:
            self._viewing_history_index += 1
        else:
            self._viewing_history_index = None

        self._update_history_view()
        return True

    def _update_history_view(self) -> None:  # pragma: no cover
        """Update the detail panel to show the current history view."""
        detail = self._try_query_one("#detail-panel", TabbedDetailPanel)
        if detail is None:
            return

        history = self._get_current_stage_history()
        total = len(history)

        input_panel = self._try_query_one("#input-panel", InputDiffPanel)
        output_panel = self._try_query_one("#output-panel", OutputDiffPanel)

        if self._viewing_history_index is None:
            stage = (
                self._stages.get(self._selected_stage_name) if self._selected_stage_name else None
            )
            detail.set_history_view(None, total, None)
            if log_panel := self._try_query_one("#stage-logs", StageLogPanel):
                log_panel.set_stage(stage)
            if input_panel:
                input_panel.set_stage(self._selected_stage_name)
            if output_panel:
                output_panel.set_stage(
                    self._selected_stage_name, status=stage.status if stage else None
                )
        else:
            if 0 <= self._viewing_history_index < total:
                entry = history[self._viewing_history_index]
                detail.set_history_view(self._viewing_history_index, total, entry)
                if input_panel:
                    if entry.input_snapshot:
                        input_panel.set_from_snapshot(entry.input_snapshot)
                    else:
                        input_panel.set_stage(None)
                if output_panel:
                    if entry.output_snapshot:
                        output_panel.set_from_snapshot(
                            entry.stage_name, entry.output_snapshot, status=entry.status
                        )
                    else:
                        output_panel.set_stage(None)

    # =========================================================================
    # UI updates
    # =========================================================================

    def _update_detail_panel(self) -> None:  # pragma: no cover
        """Update detail panel for current stage."""
        if self._watch_mode:
            self._viewing_history_index = None
        stage = self._stages.get(self._selected_stage_name) if self._selected_stage_name else None
        detail = self._try_query_one("#detail-panel", TabbedDetailPanel)
        if detail:
            detail.set_stage(stage)

    def _update_stage_list_selection(self) -> None:  # pragma: no cover
        """Update stage list panel to reflect current selection."""
        if self._selected_stage_name:
            stage_list = self._try_query_one("#stage-list", StageListPanel)
            if stage_list:
                stage_list.set_selection(self._selected_idx, self._selected_stage_name)

    def _rebuild_stage_list(self) -> None:  # pragma: no cover
        """Rebuild the stage list panel after stages change."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list:
            ordered_stages = [
                self._stages[name] for name in self._stage_order if name in self._stages
            ]
            stage_list.rebuild(ordered_stages)

    def _update_footer_context(self) -> None:  # pragma: no cover
        """Update footer context based on focus and active tab."""
        footer = self._try_query_one("#pivot-footer", PivotFooter)
        if footer is None:
            return

        # Check if stage list has focus
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list is not None and stage_list.has_focus:
            footer.set_context(FooterContext.STAGE_LIST)
            return

        # Otherwise, base context on active tab
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs is None:
            footer.set_context(FooterContext.STAGE_LIST)
            return

        match tabs.active:
            case "tab-logs":
                footer.set_context(FooterContext.LOGS)
            case "tab-input" | "tab-output":
                footer.set_context(FooterContext.DIFF)
            case _:
                footer.set_context(FooterContext.STAGE_LIST)

    def on_descendant_focus(
        self, _event: textual.events.DescendantFocus
    ) -> None:  # pragma: no cover
        """Update footer context when focus changes."""
        self._update_footer_context()

    def _get_keep_going_prefix(self) -> str:  # pragma: no cover
        """Return title prefix for keep-going mode.

        Note: keep_going toggle is not yet implemented in async Engine.
        """
        # TODO: Add keep_going support to async Engine
        return ""

    # =========================================================================
    # Actions
    # =========================================================================

    def _get_active_diff_panel(self) -> InputDiffPanel | OutputDiffPanel | None:  # pragma: no cover
        """Get the diff panel for the active tab, if any."""
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs is None:
            return None
        match tabs.active:
            case "tab-input":
                return self._try_query_one("#input-panel", InputDiffPanel)
            case "tab-output":
                return self._try_query_one("#output-panel", OutputDiffPanel)
            case _:
                return None

    def action_nav_down(self) -> None:  # pragma: no cover
        """Navigate down in stage list, skipping collapsed rows."""
        self._navigate_stage(1)

    def action_nav_up(self) -> None:  # pragma: no cover
        """Navigate up in stage list, skipping collapsed rows."""
        self._navigate_stage(-1)

    def _navigate_stage(self, direction: int) -> None:  # pragma: no cover
        """Navigate stage list by direction (+1 or -1), skipping collapsed/filtered stages."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list is None:
            return  # Can't navigate without knowing visibility state

        # Get visible stages (not collapsed and not filtered)
        visible_names = stage_list.get_visible_stage_names()
        if not visible_names:
            return  # No visible stages to navigate

        # Find current position in visible list
        current_visible_idx = -1
        if self._selected_stage_name in visible_names:
            current_visible_idx = visible_names.index(self._selected_stage_name)

        # Calculate new visible index
        if current_visible_idx == -1:
            # Current selection is hidden, go to first/last visible
            new_visible_idx = 0 if direction > 0 else len(visible_names) - 1
        else:
            new_visible_idx = current_visible_idx + direction

        # Bounds check
        if not (0 <= new_visible_idx < len(visible_names)):
            return  # At boundary, stay at current

        # Find the actual index in _stage_order
        new_name = visible_names[new_visible_idx]
        if new_name in self._stage_order:
            new_idx = self._stage_order.index(new_name)
            self._select_stage(new_idx)
            self._update_stage_list_selection()
            self._update_detail_panel()

    def _navigate_tab(self, direction: int) -> None:  # pragma: no cover
        """Navigate tabs by direction (+1 for next, -1 for prev)."""
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs is None or tabs.active not in self._TAB_IDS:
            return
        current_idx = self._TAB_IDS.index(tabs.active)
        new_idx = (current_idx + direction) % len(self._TAB_IDS)
        tabs.active = self._TAB_IDS[new_idx]
        self._update_footer_context()

    def action_prev_tab(self) -> None:  # pragma: no cover
        """Navigate to previous tab."""
        self._navigate_tab(-1)

    def action_next_tab(self) -> None:  # pragma: no cover
        """Navigate to next tab."""
        self._navigate_tab(1)

    def on_tabbed_content_tab_activated(
        self, _event: textual.widgets.TabbedContent.TabActivated
    ) -> None:  # pragma: no cover
        """Update footer context when tab is clicked."""
        self._update_footer_context()

    def _goto_tab(self, tab_id: str) -> None:  # pragma: no cover
        """Jump to a specific tab."""
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs:
            tabs.active = tab_id
            self._update_footer_context()

    def action_goto_tab_logs(self) -> None:  # pragma: no cover
        self._goto_tab("tab-logs")

    def action_goto_tab_input(self) -> None:  # pragma: no cover
        self._goto_tab("tab-input")

    def action_goto_tab_output(self) -> None:  # pragma: no cover
        self._goto_tab("tab-output")

    def action_escape_action(self) -> None:  # pragma: no cover
        """Esc: cancel commit, clear filter, or collapse detail expansion."""
        if self._watch_mode and self._commit_in_progress:
            self._cancel_commit = True
            return
        # Clear filter if active
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list and stage_list.has_active_filter:
            stage_list.clear_filter()
            return
        # Collapse detail expansion if expanded
        panel = self._get_active_diff_panel()
        if panel and panel.is_detail_expanded:
            panel.collapse_details()

    def action_focus_filter(self) -> None:  # pragma: no cover
        """Focus the stage filter input."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list:
            stage_list.focus_filter()

    def action_toggle_group(self) -> None:  # pragma: no cover
        """Toggle collapse for group containing selected stage."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list:
            group_base = stage_list.get_group_at_selection()
            if group_base:
                stage_list.toggle_group(group_base)

    def action_collapse_all_groups(self) -> None:  # pragma: no cover
        """Collapse all stage groups."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list:
            stage_list.collapse_all_groups()

    def action_expand_all_groups(self) -> None:  # pragma: no cover
        """Expand all stage groups."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        if stage_list:
            stage_list.expand_all_groups()

    def action_scroll_detail_down(self) -> None:  # pragma: no cover
        """Scroll detail panel content down."""
        panel = self._get_active_diff_panel()
        if panel:
            panel.select_next()

    def action_scroll_detail_up(self) -> None:  # pragma: no cover
        """Scroll detail panel content up."""
        panel = self._get_active_diff_panel()
        if panel:
            panel.select_prev()

    def action_next_changed(self) -> None:  # pragma: no cover
        """Move selection to next changed item in detail panel."""
        # Don't intercept if an Input widget has focus (let it type the character)
        if isinstance(self.focused, textual.widgets.Input):
            return
        # On Logs tab with active search, navigate to next match
        if self._is_log_search_active():
            log_panel = self._try_query_one("#stage-logs", StageLogPanel)
            if log_panel:
                log_panel.next_match()
                self._update_search_count()
            return

        panel = self._get_active_diff_panel()
        if panel:
            panel.select_next_changed()

    def action_prev_changed(self) -> None:  # pragma: no cover
        """Move selection to previous changed item in detail panel."""
        # Don't intercept if an Input widget has focus (let it type the character)
        if isinstance(self.focused, textual.widgets.Input):
            return
        # On Logs tab with active search, navigate to previous match
        if self._is_log_search_active():
            log_panel = self._try_query_one("#stage-logs", StageLogPanel)
            if log_panel:
                log_panel.prev_match()
                self._update_search_count()
            return

        panel = self._get_active_diff_panel()
        if panel:
            panel.select_prev_changed()

    def action_log_search(self) -> None:  # pragma: no cover
        """Activate log search (Logs tab only)."""
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs is None or tabs.active != "tab-logs":
            return
        detail_panel = self._try_query_one("#detail-panel", TabbedDetailPanel)
        if detail_panel:
            detail_panel.show_log_search()

    def _is_log_search_active(self) -> bool:  # pragma: no cover
        """Check if we're on Logs tab with active search."""
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs is None or tabs.active != "tab-logs":
            return False
        log_panel = self._try_query_one("#stage-logs", StageLogPanel)
        return log_panel is not None and log_panel.is_search_active

    def _update_search_count(self) -> None:  # pragma: no cover
        """Update the search match count display."""
        detail_panel = self._try_query_one("#detail-panel", TabbedDetailPanel)
        if detail_panel:
            detail_panel.update_search_count()

    def action_toggle_debug(self) -> None:  # pragma: no cover
        """Toggle debug panel visibility."""
        debug_panel = self._try_query_one("#debug-panel", DebugPanel)
        if debug_panel is None:
            return
        if self._debug_timer is None:
            debug_panel.add_class("visible")
            self._debug_timer = self.set_interval(0.5, self._update_debug_stats)
        else:
            debug_panel.remove_class("visible")
            self._debug_timer.stop()
            self._debug_timer = None

    def action_toggle_keep_going(self) -> None:  # pragma: no cover
        """Toggle keep-going mode (watch mode only).

        Note: This feature is temporarily unavailable while migrating to async Engine.
        """
        # TODO: Add keep_going support to async Engine
        self.notify(
            "Keep-going toggle is temporarily unavailable during async migration",
            severity="warning",
        )

    def action_show_help(self) -> None:  # pragma: no cover
        """Show help screen with all keybindings."""
        self.push_screen(HelpScreen())

    def action_history_older(self) -> None:  # pragma: no cover
        """Navigate to older history entry (watch mode only)."""
        if not self._watch_mode:
            return
        self._navigate_history_prev()

    def action_history_newer(self) -> None:  # pragma: no cover
        """Navigate to newer history entry (watch mode only)."""
        if not self._watch_mode:
            return
        self._navigate_history_next()

    def action_history_live(self) -> None:  # pragma: no cover
        """Jump directly to live view (watch mode only)."""
        if not self._watch_mode:
            return
        if self._viewing_history_index is not None:
            self._viewing_history_index = None
            self._update_history_view()

    @textual.work
    async def action_show_history_list(self) -> None:  # pragma: no cover
        """Show modal with history list (watch mode only)."""
        if not self._watch_mode or not self._selected_stage_name:
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

    async def action_commit(self) -> None:  # pragma: no cover
        """Commit pending changes (watch mode only)."""
        if not self._watch_mode:
            return
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
        self.notify("Acquiring commit lock... (Esc to cancel)", timeout=0)

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
                        self.notify(f"Still waiting for lock... ({int(elapsed)}s)", timeout=0)

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
            self._commit_in_progress = False
            if acquired is not None:
                acquired.release()

    async def action_force_rerun_stage(self) -> None:  # pragma: no cover
        """Force re-run the currently selected stage (watch mode only)."""
        if not self._watch_mode:
            return
        if self._selected_stage_name is None:
            self.notify("No stage selected", severity="warning")
            return
        if self._has_running_stages:
            self.notify("Cannot re-run while stages are running", severity="warning")
            return

        stage_name = self._selected_stage_name
        socket_path = project.get_project_root() / ".pivot" / "agent.sock"

        self.notify(f"Forcing re-run of {stage_name}...")
        success = await rpc_client.send_run_command(socket_path, stages=[stage_name], force=True)
        if not success:
            self.notify("Failed to send re-run command", severity="error")

    async def action_force_rerun_all(self) -> None:  # pragma: no cover
        """Force re-run all stages (watch mode only)."""
        if not self._watch_mode:
            return
        if self._has_running_stages:
            self.notify("Cannot re-run while stages are running", severity="warning")
            return

        socket_path = project.get_project_root() / ".pivot" / "agent.sock"

        self.notify("Forcing re-run of all stages...")
        success = await rpc_client.send_run_command(socket_path, stages=None, force=True)
        if not success:
            self.notify("Failed to send re-run command", severity="error")

    # =========================================================================
    # Debug stats
    # =========================================================================

    def _update_debug_stats(self) -> None:  # pragma: no cover
        """Update debug panel with current stats."""
        try:
            stats = self._collect_debug_stats()
            debug_panel = self._try_query_one("#debug-panel", DebugPanel)
            if debug_panel:
                debug_panel.update_stats(stats)
        except Exception:
            _logger.debug("Failed to update debug stats", exc_info=True)

    def _collect_debug_stats(self) -> DebugStats:  # pragma: no cover
        """Collect current debug statistics."""
        counts = status_utils.count_statuses(self._stages.values())
        return DebugStats(
            tui_messages=self._message_stats.get_stats(),
            active_workers=counts["running"],
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

    # =========================================================================
    # Quit
    # =========================================================================

    @override
    async def action_quit(self) -> None:  # pragma: no cover
        """Quit the app."""
        if self._debug_timer is not None:
            self._debug_timer.stop()
            self._debug_timer = None
        if self._stats_log_timer is not None:
            self._stats_log_timer.stop()
            self._stats_log_timer = None

        # Prevent re-entry from rapid quit presses
        with self._quit_lock:
            if self._quitting:
                return
            self._quitting = True

        if self._watch_mode:
            self._quit_with_commit_prompt()
        else:
            self._quit_run_mode()

    @textual.work
    async def _quit_run_mode(self) -> None:  # pragma: no cover
        """Worker to handle quit in run mode (may show confirmation dialog)."""
        if self._has_running_stages:
            should_quit = await self.push_screen_wait(ConfirmKillWorkersScreen())
            if not should_quit:
                self._quitting = False  # Allow retry after cancel
                return
            # User confirmed quit - unregister loky's atexit handler to prevent hang
            # The handler waits for worker threads which blocks exit
            if self._cancel_event is not None:
                self._cancel_event.set()
            atexit.unregister(
                loky.process_executor._python_exit  # pyright: ignore[reportPrivateUsage]
            )
            # Store message to print after TUI exits
            running = sum(1 for s in self._stages.values() if s.status == StageStatus.IN_PROGRESS)
            self._exit_message = (
                f"Note: {running} stage(s) are still running. Press Ctrl+C to forcefully kill them."
            )
        # Exit cleanly - Textual restores terminal, loky cleanup skipped
        self._shutdown_event.set()
        self._close_log_file()
        self.exit()

    @textual.work
    async def _quit_with_commit_prompt(self) -> None:  # pragma: no cover
        """Worker to handle quit with commit prompt (watch mode)."""
        try:
            if self._commit_in_progress:
                self._cancel_commit = True

            if self._no_commit and not self._has_running_stages:
                pending = await asyncio.to_thread(
                    lock.list_pending_stages, project.get_project_root()
                )
                if pending:
                    should_commit = await self.push_screen_wait(ConfirmCommitScreen())
                    if should_commit:
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
            # Engine is managed externally (by CLI) - just exit TUI
            self.exit()
