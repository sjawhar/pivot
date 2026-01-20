from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import json
import logging
import os
import pathlib
import queue
import threading
import time
from typing import (
    IO,
    TYPE_CHECKING,
    ClassVar,
    Literal,
    override,
)

import filelock
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
from pivot.registry import REGISTRY
from pivot.storage import lock, project_lock
from pivot.tui import agent_server, diff_panels
from pivot.tui.diff_panels import InputDiffPanel, OutputDiffPanel
from pivot.tui.screens import ConfirmCommitScreen, HelpScreen, HistoryListScreen
from pivot.tui.stats import DebugStats, QueueStatsTracker, get_memory_mb
from pivot.tui.types import ExecutionHistoryEntry, LogEntry, PendingHistoryState, StageInfo
from pivot.tui.widgets import (
    DebugPanel,
    LogPanel,
    StageListPanel,
    StageLogPanel,
    TabbedDetailPanel,
)
from pivot.tui.widgets import status as status_utils
from pivot.types import (
    DisplayMode,
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiQueue,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)

__all__ = [
    "PivotApp",
    "TuiUpdate",
    "ExecutorComplete",
    "run_with_tui",
    "run_watch_tui",
    "should_use_tui",
]

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable

    from pivot.types import OutputChange, OutputMessage
    from pivot.watch.engine import WatchEngine


class TuiUpdate(textual.message.Message):
    """Custom message for executor updates."""

    msg: TuiLogMessage | TuiStatusMessage | TuiWatchMessage | TuiReloadMessage

    def __init__(
        self, msg: TuiLogMessage | TuiStatusMessage | TuiWatchMessage | TuiReloadMessage
    ) -> None:
        self.msg = msg
        super().__init__()


class ExecutorComplete(textual.message.Message):
    """Signal that executor has finished (run mode only)."""

    results: dict[str, ExecutionSummary]
    error: Exception | None

    def __init__(self, results: dict[str, ExecutionSummary], error: Exception | None) -> None:
        self.results = results
        self.error = error
        super().__init__()


_TUI_BINDINGS: list[textual.binding.BindingType] = [
    textual.binding.Binding("q", "quit", "Quit"),
    textual.binding.Binding("c", "commit", "Commit"),
    textual.binding.Binding("?", "show_help", "Help"),
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
    # History navigation (works in all tabs, watch mode only)
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
    # Stage filtering with number keys
    *[
        textual.binding.Binding(str(i), f"filter_stage({i - 1})", f"Stage {i}", show=False)
        for i in range(1, 10)
    ],
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
    _executor_func: Callable[[], dict[str, ExecutionSummary]] | None
    _engine: WatchEngine | None
    _output_queue: mp.Queue[OutputMessage] | None

    def __init__(
        self,
        message_queue: TuiQueue,
        stage_names: list[str] | None = None,
        tui_log: pathlib.Path | None = None,
        *,
        # Run mode parameters
        executor_func: Callable[[], dict[str, ExecutionSummary]] | None = None,
        # Watch mode parameters
        engine: WatchEngine | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
        no_commit: bool = False,
        serve: bool = False,
    ) -> None:
        """Initialize TUI app.

        For run mode: provide executor_func
        For watch mode: provide engine
        """
        super().__init__()

        # Determine mode from parameters
        self._watch_mode: bool = engine is not None
        if not self._watch_mode and executor_func is None:
            msg = "Either executor_func (run mode) or engine (watch mode) must be provided"
            raise ValueError(msg)

        # Core state
        self._tui_queue: TuiQueue = message_queue
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
            message_queue,
        )
        self._output_stats: QueueStatsTracker | None = None
        self._start_time: float = 0.0
        self._debug_timer: textual.timer.Timer | None = None
        self._stats_log_timer: textual.timer.Timer | None = None

        # Run mode state
        self._executor_func = executor_func
        self._results: dict[str, ExecutionSummary] | None = None
        self._error: Exception | None = None
        self._executor_thread: threading.Thread | None = None

        # Watch mode state
        self._engine = engine
        self._output_queue = output_queue
        if output_queue is not None:
            self._output_stats = QueueStatsTracker(
                "output_queue",
                output_queue,
            )
        self._engine_thread: threading.Thread | None = None
        self._no_commit: bool = no_commit
        self._commit_in_progress: bool = False
        self._cancel_commit: bool = False
        self._viewing_history_index: int | None = None
        self._pending_history: dict[str, PendingHistoryState] = {}
        self._current_run_id: str | None = None
        self._serve: bool = serve
        self._agent_server: agent_server.AgentServer | None = None
        self._agent_server_task: asyncio.Task[None] | None = None
        self._quitting: bool = False
        self._quit_lock: threading.Lock = threading.Lock()
        self._log_file_lock: threading.Lock = threading.Lock()

        # Open log file if configured
        if tui_log:
            self._log_file = open(tui_log, "w", buffering=1)  # noqa: SIM115
            os.set_inheritable(self._log_file.fileno(), False)
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
        if self._stage_order and self._selected_idx < len(self._stage_order):
            return self._stage_order[self._selected_idx]
        return None

    @property
    def focused_panel(self) -> Literal["stages", "detail"]:
        """Return which panel currently has focus."""
        return self._focused_panel

    @property
    def error(self) -> Exception | None:
        """Return any exception that occurred during execution (run mode only)."""
        return self._error

    @property
    def _has_running_stages(self) -> bool:
        """Check if any stages are currently in progress."""
        return any(s.status == StageStatus.IN_PROGRESS for s in self._stages.values())

    def select_stage_by_index(self, idx: int) -> None:
        """Select a stage by index (for testing)."""
        self._select_stage(idx)
        self._update_detail_panel()

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
        """Initialize TUI on mount."""
        self._start_time = time.monotonic()
        if self._log_file is not None:
            self._stats_log_timer = self.set_interval(1.0, self._write_stats_to_log)

        # Defer initial panel update to after widgets are fully mounted
        self.call_after_refresh(self._update_detail_panel)
        self._start_queue_reader()

        if self._watch_mode:
            # Watch mode: start engine and optionally agent server
            prefix = self._get_keep_going_prefix()
            self.title = f"{prefix}[●] Watching for changes..."  # pyright: ignore[reportUnannotatedClassAttribute] - inherited from App
            self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
            self._engine_thread.start()
            if self._serve:
                await self._start_agent_server()
        else:
            # Run mode: start executor
            self._executor_thread = threading.Thread(target=self._run_executor, daemon=True)
            self._executor_thread.start()

    def on_resize(self, event: textual.events.Resize) -> None:  # pragma: no cover
        """Handle terminal resize - warn if too small."""
        if event.size.width < _MIN_TERMINAL_WIDTH or event.size.height < _MIN_TERMINAL_HEIGHT:
            msg = (
                f"Terminal too small ({event.size.width}x{event.size.height}). "
                f"Minimum: {_MIN_TERMINAL_WIDTH}x{_MIN_TERMINAL_HEIGHT}"
            )
            self.notify(msg, severity="warning", timeout=5)

    # =========================================================================
    # Background threads
    # =========================================================================

    def _start_queue_reader(self) -> None:  # pragma: no cover
        """Start the background queue reader thread."""
        self._reader_thread = threading.Thread(target=self._read_queue, daemon=True)
        self._reader_thread.start()

    def _read_queue(self) -> None:  # pragma: no cover
        """Read from queue and post messages to Textual."""
        while not self._shutdown_event.is_set():
            try:
                msg = self._tui_queue.get(timeout=0.02)
                self._tui_stats.record_message()
                if msg is None:
                    self._write_to_log('{"type": "shutdown"}\n')
                    break
                self._write_to_log(json.dumps(msg, default=str) + "\n")
                _logger.debug(f"TUI recv: {msg['type']} stage={msg.get('stage', '?')}")  # noqa: G004
                self.post_message(TuiUpdate(msg))
            except queue.Empty:
                continue
            except Exception:
                _logger.exception("Error in TUI queue reader")

    def _run_executor(self) -> None:  # pragma: no cover
        """Run the executor (run mode, background thread)."""
        results: dict[str, ExecutionSummary] = {}
        error: Exception | None = None
        try:
            if self._executor_func:
                results = self._executor_func()
        except Exception as e:
            error = e
        finally:
            self._tui_queue.put(None)
            self.post_message(ExecutorComplete(results, error))

    def _run_engine(self) -> None:  # pragma: no cover
        """Run the watch engine (watch mode, background thread)."""
        try:
            if self._engine:
                self._engine.run(tui_queue=self._tui_queue, output_queue=self._output_queue)
        except Exception as e:
            _logger.exception(f"Watch engine failed: {e}")
            error_msg = TuiWatchMessage(
                type=TuiMessageType.WATCH,
                status=WatchStatus.ERROR,
                message="Watch mode crashed. Please restart 'pivot watch'.",
            )
            with contextlib.suppress(Exception):
                self._tui_queue.put_nowait(error_msg)

    # =========================================================================
    # Agent server (watch mode only)
    # =========================================================================

    async def _start_agent_server(self) -> None:  # pragma: no cover
        """Start the JSON-RPC agent server."""
        if self._engine is None:
            return
        socket_path = project.get_project_root() / ".pivot" / "agent.sock"
        # Clean up stale socket from previous crash
        socket_path.unlink(missing_ok=True)
        self._agent_server = agent_server.AgentServer(self._engine, socket_path)

        try:
            server = await self._agent_server.start()
            _logger.info(f"Agent server listening on {socket_path}")
            self._agent_server_task = asyncio.create_task(server.serve_forever())
        except Exception as e:
            _logger.warning(f"Failed to start agent server: {e}")
            await self._agent_server.stop()
            self._agent_server = None

    async def _stop_agent_server(self) -> None:  # pragma: no cover
        """Stop the JSON-RPC agent server if running."""
        if self._agent_server_task is not None:
            self._agent_server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._agent_server_task, timeout=2.0)
            self._agent_server_task = None

        if self._agent_server is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._agent_server.stop(), timeout=2.0)
            self._agent_server = None

    # =========================================================================
    # Message handling
    # =========================================================================

    def on_tui_update(self, event: TuiUpdate) -> None:  # pragma: no cover
        """Handle executor/engine updates."""
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

        # Update all-logs panel
        log_panel = self._try_query_one("#log-panel", LogPanel)
        if log_panel:
            log_panel.add_log(stage, line, is_stderr)

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
                stage_list.update_stage(stage, self._selected_stage_name)
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

    def on_executor_complete(self, event: ExecutorComplete) -> None:  # pragma: no cover
        """Handle executor completion (run mode only)."""
        self._results = event.results
        self._error = event.error
        if event.error:
            self.title = f"pivot run - FAILED: {event.error}"
        else:
            self.title = "pivot run - Complete"
        # Clean up threads before exiting (same as action_quit for run mode)
        self._shutdown_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
        self._close_log_file()
        self.exit(self._results)

    # =========================================================================
    # History tracking (watch mode only)
    # =========================================================================

    def _create_history_entry(self, stage_name: str, run_id: str) -> None:
        """Create a new history entry when stage starts executing."""
        input_snapshot = None
        try:
            registry_info = REGISTRY.get(stage_name)
            state_dir = config.get_state_dir()
            input_snapshot = explain.get_stage_explanation(
                stage_name=stage_name,
                fingerprint=registry_info["fingerprint"],
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
            try:
                registry_info = REGISTRY.get(stage_name)
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
        return collections.deque(maxlen=50)

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

    def _update_focus_visual(self) -> None:  # pragma: no cover
        """Update visual indicators for focused panel."""
        stage_list = self._try_query_one("#stage-list", StageListPanel)
        detail_panel = self._try_query_one("#detail-panel", TabbedDetailPanel)
        is_stages_focused = self._focused_panel == "stages"
        if stage_list:
            stage_list.set_class(is_stages_focused, "focused")
        if detail_panel:
            detail_panel.set_class(not is_stages_focused, "focused")

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

    def _get_keep_going_prefix(self) -> str:  # pragma: no cover
        """Return title prefix for keep-going mode."""
        if self._watch_mode and self._engine:
            return "[-k] " if self._engine.keep_going else ""
        return ""

    # =========================================================================
    # Actions
    # =========================================================================

    def action_switch_focus(self) -> None:  # pragma: no cover
        """Toggle focus between stages panel and detail panel."""
        self._focused_panel = "detail" if self._focused_panel == "stages" else "stages"
        self._update_focus_visual()

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
        """Navigate down."""
        if self._focused_panel == "stages":
            self.action_next_stage()
        elif self._focused_panel == "detail" and (panel := self._get_active_diff_panel()):
            panel.select_next()

    def action_nav_up(self) -> None:  # pragma: no cover
        """Navigate up."""
        if self._focused_panel == "stages":
            self.action_prev_stage()
        elif self._focused_panel == "detail" and (panel := self._get_active_diff_panel()):
            panel.select_prev()

    def action_nav_left(self) -> None:  # pragma: no cover
        """Navigate left - previous tab or switch to stages panel."""
        if self._focused_panel == "detail":
            tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
            if tabs is None:
                return
            try:
                if tabs.active == self._TAB_IDS[0]:
                    self._focused_panel = "stages"
                    self._update_focus_visual()
                elif tabs.active in self._TAB_IDS:
                    current_idx = self._TAB_IDS.index(tabs.active)
                    tabs.active = self._TAB_IDS[current_idx - 1]
            except ValueError:
                _logger.debug("Tab index error during nav_left")

    def action_nav_right(self) -> None:  # pragma: no cover
        """Navigate right - next tab or switch to detail panel."""
        if self._focused_panel == "stages":
            self._focused_panel = "detail"
            self._update_focus_visual()
        else:
            tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
            if tabs is None:
                return
            try:
                if tabs.active in self._TAB_IDS:
                    current_idx = self._TAB_IDS.index(tabs.active)
                    if current_idx < len(self._TAB_IDS) - 1:
                        tabs.active = self._TAB_IDS[current_idx + 1]
            except ValueError:
                _logger.debug("Tab index error during nav_right")

    def _goto_tab(self, tab_id: str) -> None:  # pragma: no cover
        """Jump to a specific tab and focus the detail panel."""
        tabs = self._try_query_one("#detail-tabs", textual.widgets.TabbedContent)
        if tabs:
            tabs.active = tab_id
            self._focused_panel = "detail"
            self._update_focus_visual()

    def action_goto_tab_logs(self) -> None:  # pragma: no cover
        self._goto_tab("tab-logs")

    def action_goto_tab_input(self) -> None:  # pragma: no cover
        self._goto_tab("tab-input")

    def action_goto_tab_output(self) -> None:  # pragma: no cover
        self._goto_tab("tab-output")

    def action_next_stage(self) -> None:  # pragma: no cover
        if self._selected_idx < len(self._stage_order) - 1:
            self._select_stage(self._selected_idx + 1)
            self._update_stage_list_selection()
            self._update_detail_panel()

    def action_prev_stage(self) -> None:  # pragma: no cover
        if self._selected_idx > 0:
            self._select_stage(self._selected_idx - 1)
            self._update_stage_list_selection()
            self._update_detail_panel()

    def action_toggle_view(self) -> None:  # pragma: no cover
        self._show_logs = not self._show_logs
        main_split = self._try_query_one("#main-split", textual.containers.Horizontal)
        logs_view = self._try_query_one("#logs-view", textual.containers.Vertical)
        if main_split:
            main_split.set_class(self._show_logs, "view-hidden")
            main_split.set_class(not self._show_logs, "view-active")
        if logs_view:
            logs_view.set_class(self._show_logs, "view-active")
            logs_view.set_class(not self._show_logs, "view-hidden")

    def action_show_all_logs(self) -> None:  # pragma: no cover
        log_panel = self._try_query_one("#log-panel", LogPanel)
        if log_panel:
            log_panel.set_filter(None)
        if not self._show_logs:
            self.action_toggle_view()

    def action_filter_stage(self, idx: int) -> None:  # pragma: no cover
        """Filter logs to stage at index idx (0-based)."""
        if idx < len(self._stage_order):
            stage_name = self._stage_order[idx]
            log_panel = self._try_query_one("#log-panel", LogPanel)
            if log_panel:
                log_panel.set_filter(stage_name)
            if not self._show_logs:
                self.action_toggle_view()

    def action_escape_action(self) -> None:  # pragma: no cover
        """Context-aware Esc: cancel commit or collapse detail expansion."""
        if self._watch_mode and self._commit_in_progress:
            self._cancel_commit = True
            return
        panel = self._get_active_diff_panel()
        if self._focused_panel == "detail" and panel and panel.is_detail_expanded:
            panel.collapse_details()

    def action_expand_details(self) -> None:  # pragma: no cover
        """Expand details pane or toggle group collapse."""
        if self._focused_panel == "stages":
            stage_list = self._try_query_one("#stage-list", StageListPanel)
            if stage_list:
                group_base = stage_list.get_group_at_selection()
                if group_base:
                    stage_list.toggle_group(group_base)
        elif self._focused_panel == "detail":
            panel = self._get_active_diff_panel()
            if panel:
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
        """Toggle keep-going mode (watch mode only)."""
        if not self._watch_mode or not self._engine:
            self.notify(
                "Keep-going toggle is only available in watch mode (use --watch)",
                severity="warning",
            )
            return
        enabled = self._engine.toggle_keep_going()
        self.notify(f"Keep-going: {'ON' if enabled else 'OFF'}")
        self.title = f"{self._get_keep_going_prefix()}[●] Watching for changes..."

    def action_show_help(self) -> None:  # pragma: no cover
        """Show help screen with all keybindings."""
        self.push_screen(HelpScreen())

    def action_history_older(self) -> None:  # pragma: no cover
        """Navigate to older history entry (watch mode only)."""
        if not self._watch_mode or self._focused_panel != "detail":
            return
        self._navigate_history_prev()

    def action_history_newer(self) -> None:  # pragma: no cover
        """Navigate to newer history entry (watch mode only)."""
        if not self._watch_mode or self._focused_panel != "detail":
            return
        self._navigate_history_next()

    def action_history_live(self) -> None:  # pragma: no cover
        """Jump directly to live view (watch mode only)."""
        if not self._watch_mode or self._focused_panel != "detail":
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
            tui_queue=self._tui_stats.get_stats(),
            output_queue=self._output_stats.get_stats() if self._output_stats else None,
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

        if self._watch_mode:
            with self._quit_lock:
                if self._quitting:
                    return
                self._quitting = True
            self._quit_with_commit_prompt()
        else:
            self._shutdown_event.set()
            if self._reader_thread:
                self._reader_thread.join(timeout=2.0)
            self._close_log_file()
            await super().action_quit()

    @textual.work
    async def _quit_with_commit_prompt(self) -> None:  # pragma: no cover
        """Worker to handle quit with commit prompt (watch mode)."""
        try:
            await self._stop_agent_server()

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
            if self._engine:
                self._engine.shutdown()
            self.exit()


# =============================================================================
# Entry points
# =============================================================================


def run_with_tui(
    stage_names: list[str],
    message_queue: TuiQueue,
    executor_func: Callable[[], dict[str, ExecutionSummary]],
    tui_log: pathlib.Path | None = None,
) -> dict[str, ExecutionSummary]:  # pragma: no cover
    """Run pipeline with TUI display. Raises if executor fails."""
    app = PivotApp(
        message_queue,
        stage_names=stage_names,
        tui_log=tui_log,
        executor_func=executor_func,
    )
    results = app.run()
    if app.error is not None:
        raise app.error
    return results or {}


def run_watch_tui(
    engine: WatchEngine,
    message_queue: TuiQueue,
    output_queue: mp.Queue[OutputMessage] | None = None,
    tui_log: pathlib.Path | None = None,
    stage_names: list[str] | None = None,
    *,
    no_commit: bool = False,
    serve: bool = False,
) -> None:  # pragma: no cover
    """Run watch mode with TUI display."""
    app = PivotApp(
        message_queue,
        stage_names=stage_names,
        tui_log=tui_log,
        engine=engine,
        output_queue=output_queue,
        no_commit=no_commit,
        serve=serve,
    )
    app.run()


def should_use_tui(display_mode: DisplayMode | None) -> bool:
    """Determine if TUI should be used based on display mode and TTY."""
    import sys

    if display_mode == DisplayMode.TUI:
        return True
    if display_mode == DisplayMode.PLAIN:
        return False
    return sys.stdout.isatty()
