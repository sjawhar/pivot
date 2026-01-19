from __future__ import annotations

import collections
import multiprocessing as mp
import pathlib
import queue
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest
import textual.binding
import textual.widgets

from helpers import register_test_stage
from pivot import executor, loaders, outputs
from pivot.tui import run as run_tui
from pivot.types import (
    DisplayMode,
    OutputMessage,
    StageStatus,
    TuiLogMessage,
    TuiMessage,
    TuiMessageType,
    TuiStatusMessage,
    is_tui_log_message,
    is_tui_status_message,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.managers import SyncManager


# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _Step1Outputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step1.txt", loaders.PathOnly())]


class _Step2Outputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step2.txt", loaders.PathOnly())]


class _Step3Outputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("step3.txt", loaders.PathOnly())]


# =============================================================================
# Module-level helper functions for stages
# =============================================================================


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _helper_process_print(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    print("Processing data")
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _helper_failing_stage(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    raise RuntimeError("Stage failed!")


def _helper_step1(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _Step1Outputs:
    _ = input_file
    pathlib.Path("step1.txt").write_text("step1")
    return {"output": pathlib.Path("step1.txt")}


def _helper_step2(
    step1_file: Annotated[pathlib.Path, outputs.Dep("step1.txt", loaders.PathOnly())],
) -> _Step2Outputs:
    _ = step1_file
    pathlib.Path("step2.txt").write_text("step2")
    return {"output": pathlib.Path("step2.txt")}


def _helper_step3(
    step2_file: Annotated[pathlib.Path, outputs.Dep("step2.txt", loaders.PathOnly())],
) -> _Step3Outputs:
    _ = step2_file
    pathlib.Path("step3.txt").write_text("step3")
    return {"output": pathlib.Path("step3.txt")}


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def _drain_queue(tui_queue: mp.Queue[TuiMessage]) -> list[TuiMessage]:
    """Drain all messages from a TUI queue until None sentinel or timeout."""
    messages = list[TuiMessage]()
    while True:
        try:
            msg = tui_queue.get(timeout=0.1)
            if msg is None:
                break
            messages.append(msg)
        except queue.Empty:
            break
    return messages


@pytest.fixture
def tui_queue_with_manager() -> Generator[tuple[mp.Queue[TuiMessage], SyncManager]]:
    """Create a TUI queue with proper manager cleanup."""
    manager = mp.Manager()
    tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]
    yield tui_queue, manager
    manager.shutdown()


# =============================================================================
# _format_elapsed Tests
# =============================================================================


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        # None returns empty string
        (None, ""),
        # Seconds
        (5.0, "(0:05)"),
        (59.9, "(0:59)"),
        # Minutes
        (60.0, "(1:00)"),
        (125.5, "(2:05)"),
        # Large values
        (3661.0, "(61:01)"),
    ],
)
def test_format_elapsed(elapsed: float | None, expected: str) -> None:
    """_format_elapsed formats elapsed time correctly."""
    assert run_tui._format_elapsed(elapsed) == expected


# =============================================================================
# should_use_tui Tests
# =============================================================================


@pytest.mark.parametrize(
    ("display_mode", "expected"),
    [
        (DisplayMode.TUI, True),
        (DisplayMode.PLAIN, False),
    ],
)
def test_should_use_tui_explicit_mode(display_mode: DisplayMode, expected: bool) -> None:
    """should_use_tui returns correct value for explicit display modes."""
    assert run_tui.should_use_tui(display_mode) is expected


# =============================================================================
# StageInfo Tests
# =============================================================================


def test_stage_info_initialization() -> None:
    """StageInfo initializes with correct defaults."""
    info = run_tui.StageInfo("test_stage", 1, 5)

    assert info.name == "test_stage"
    assert info.index == 1
    assert info.total == 5
    assert info.status == StageStatus.READY
    assert info.reason == ""
    assert info.elapsed is None
    assert isinstance(info.logs, collections.deque)
    assert len(info.logs) == 0


def test_stage_info_logs_bounded() -> None:
    """StageInfo logs deque has maxlen of 1000."""
    info = run_tui.StageInfo("test", 1, 1)

    for i in range(1500):
        info.logs.append((f"line {i}", False, 1234567890.0 + i))

    assert len(info.logs) == 1000, "Logs should be bounded to 1000 entries"
    assert info.logs[0] == ("line 500", False, 1234567890.0 + 500), (
        "Oldest entries should be dropped"
    )


# =============================================================================
# TuiUpdate Message Tests
# =============================================================================


@pytest.mark.parametrize(
    ("msg", "expected_type"),
    [
        (
            TuiLogMessage(
                type=TuiMessageType.LOG,
                stage="test",
                line="output",
                is_stderr=False,
                timestamp=1234567890.0,
            ),
            "log",
        ),
        (
            TuiStatusMessage(
                type=TuiMessageType.STATUS,
                stage="test",
                index=1,
                total=5,
                status=StageStatus.IN_PROGRESS,
                reason="",
                elapsed=None,
                run_id="20240101_120000_abcd1234",
            ),
            "status",
        ),
    ],
    ids=["log_message", "status_message"],
)
def test_tui_update_wraps_messages(
    msg: TuiLogMessage | TuiStatusMessage, expected_type: str
) -> None:
    """TuiUpdate correctly wraps different message types."""
    update = run_tui.TuiUpdate(msg)
    assert update.msg == msg
    assert update.msg is not None
    assert update.msg["type"] == expected_type


# =============================================================================
# ExecutorComplete Message Tests
# =============================================================================


@pytest.mark.parametrize(
    ("results", "error", "expected_results", "has_error"),
    [
        (
            {"stage1": executor.ExecutionSummary(status=StageStatus.RAN, reason="code changed")},
            None,
            {"stage1": executor.ExecutionSummary(status=StageStatus.RAN, reason="code changed")},
            False,
        ),
        ({}, ValueError("something went wrong"), {}, True),
    ],
    ids=["success", "with_error"],
)
def test_executor_complete(
    results: dict[str, executor.ExecutionSummary],
    error: Exception | None,
    expected_results: dict[str, executor.ExecutionSummary],
    has_error: bool,
) -> None:
    """ExecutorComplete stores results and error appropriately."""
    complete = run_tui.ExecutorComplete(results, error=error)
    assert complete.results == expected_results
    if has_error:
        assert complete.error is not None
    else:
        assert complete.error is None


# =============================================================================
# RunTuiApp Initialization Tests
# =============================================================================


def test_run_tui_app_init(
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """RunTuiApp initializes with stage names and queue."""
    tui_queue, _manager = tui_queue_with_manager
    stage_names = ["stage1", "stage2", "stage3"]

    def executor_func() -> dict[str, executor.ExecutionSummary]:
        return {}

    app = run_tui.RunTuiApp(stage_names, tui_queue, executor_func)

    assert len(app._stages) == 3
    assert list(app._stage_order) == stage_names
    assert app._selected_idx == 0
    assert app._show_logs is False
    assert app._results is None
    assert app.error is None


def test_run_tui_app_stage_info_indexes(
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """RunTuiApp assigns correct 1-based indexes to stages."""
    tui_queue, _manager = tui_queue_with_manager
    stage_names = ["first", "second", "third"]

    def executor_func() -> dict[str, executor.ExecutionSummary]:
        return {}

    app = run_tui.RunTuiApp(stage_names, tui_queue, executor_func)

    assert app._stages["first"].index == 1
    assert app._stages["second"].index == 2
    assert app._stages["third"].index == 3

    for _name, info in app._stages.items():
        assert info.total == 3


# =============================================================================
# TUI Queue Integration Tests
# =============================================================================


def test_executor_emits_status_messages_to_queue(
    pipeline_dir: pathlib.Path,
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """Executor emits TuiStatusMessage for stage start and completion."""
    tui_queue, _manager = tui_queue_with_manager
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_helper_process, name="process")

    executor.run(show_output=False, tui_queue=tui_queue)

    messages = _drain_queue(tui_queue)
    status_messages = [m for m in messages if is_tui_status_message(m)]

    assert len(status_messages) >= 2, "Should have at least start and complete status"

    start_msg = status_messages[0]
    assert start_msg["stage"] == "process"
    assert start_msg["status"] == StageStatus.IN_PROGRESS
    assert start_msg["index"] == 1
    assert start_msg["total"] == 1
    assert "run_id" in start_msg, "Status message must include run_id"
    assert start_msg["run_id"], "run_id must be non-empty"

    end_msg = status_messages[-1]
    assert end_msg["stage"] == "process"
    assert end_msg["status"] in (StageStatus.RAN, StageStatus.SKIPPED, StageStatus.COMPLETED)
    assert "elapsed" in end_msg
    assert "run_id" in end_msg, "Status message must include run_id"
    assert end_msg["run_id"] == start_msg["run_id"], (
        "run_id must be consistent across stage lifecycle"
    )


def test_executor_emits_log_messages_to_queue(
    pipeline_dir: pathlib.Path,
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """Executor emits TuiLogMessage for stage output."""
    tui_queue, _manager = tui_queue_with_manager
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_helper_process_print, name="process")

    executor.run(show_output=False, tui_queue=tui_queue)

    messages = _drain_queue(tui_queue)
    log_messages = [m for m in messages if is_tui_log_message(m)]

    assert len(log_messages) >= 1, "Should have at least one log message"
    assert any("Processing data" in m["line"] for m in log_messages), "Log should contain stdout"


def test_executor_emits_failed_status_on_stage_failure(
    pipeline_dir: pathlib.Path,
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """Executor emits FAILED status when stage raises an exception."""
    tui_queue, _manager = tui_queue_with_manager
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_helper_failing_stage, name="failing_stage")

    executor.run(show_output=False, tui_queue=tui_queue)

    messages = _drain_queue(tui_queue)
    status_messages = [m for m in messages if is_tui_status_message(m)]

    failed_msgs = [m for m in status_messages if m["status"] == StageStatus.FAILED]
    assert len(failed_msgs) >= 1, "Should have at least one FAILED status message"
    assert failed_msgs[0]["stage"] == "failing_stage"
    assert "run_id" in failed_msgs[0], "Failed status must include run_id"
    assert failed_msgs[0]["run_id"], "run_id must be non-empty"


def test_executor_emits_status_for_multiple_stages(
    pipeline_dir: pathlib.Path,
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """Executor emits status messages for all stages in multi-stage pipeline."""
    tui_queue, _manager = tui_queue_with_manager
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_helper_step1, name="step1")
    register_test_stage(_helper_step2, name="step2")
    register_test_stage(_helper_step3, name="step3")

    executor.run(show_output=False, tui_queue=tui_queue)

    messages = _drain_queue(tui_queue)
    status_messages = [m for m in messages if is_tui_status_message(m)]
    stages_with_status = {m["stage"] for m in status_messages}

    assert "step1" in stages_with_status
    assert "step2" in stages_with_status
    assert "step3" in stages_with_status

    # All status messages must include run_id
    for msg in status_messages:
        assert "run_id" in msg, f"Status for {msg['stage']} must include run_id"
        assert msg["run_id"], f"run_id for {msg['stage']} must be non-empty"


def test_executor_status_includes_correct_index_and_total(
    pipeline_dir: pathlib.Path,
    tui_queue_with_manager: tuple[mp.Queue[TuiMessage], SyncManager],
) -> None:
    """Executor status messages include correct index and total counts."""
    tui_queue, _manager = tui_queue_with_manager
    (pipeline_dir / "input.txt").write_text("hello")

    register_test_stage(_helper_step1, name="step1")
    register_test_stage(_helper_step2, name="step2")

    executor.run(show_output=False, tui_queue=tui_queue)

    messages = _drain_queue(tui_queue)
    status_messages = [m for m in messages if is_tui_status_message(m)]

    for msg in status_messages:
        assert msg["total"] == 2, "Total should be 2 stages"
        assert msg["index"] in (1, 2), "Index should be 1 or 2"


# =============================================================================
# STATUS_STYLES Tests
# =============================================================================


def test_status_styles_covers_all_statuses() -> None:
    """STATUS_STYLES dict has entries for all relevant StageStatus values."""
    assert StageStatus.READY in run_tui.STATUS_STYLES
    assert StageStatus.IN_PROGRESS in run_tui.STATUS_STYLES
    assert StageStatus.COMPLETED in run_tui.STATUS_STYLES
    assert StageStatus.RAN in run_tui.STATUS_STYLES
    assert StageStatus.SKIPPED in run_tui.STATUS_STYLES
    assert StageStatus.FAILED in run_tui.STATUS_STYLES
    assert StageStatus.UNKNOWN in run_tui.STATUS_STYLES


def test_status_styles_returns_tuple() -> None:
    """STATUS_STYLES values are (label, style) tuples."""
    for status, (label, style) in run_tui.STATUS_STYLES.items():
        assert isinstance(label, str), f"Label for {status} should be string"
        assert isinstance(style, str), f"Style for {status} should be string"
        assert len(label) > 0, f"Label for {status} should not be empty"


# =============================================================================
# WatchTuiApp Tests
# =============================================================================


class _MockEngine:
    """Mock engine for WatchTuiApp tests."""

    def __init__(self) -> None:
        self._keep_going: bool = False

    def run(
        self,
        tui_queue: mp.Queue[TuiMessage] | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
    ) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def toggle_keep_going(self) -> bool:
        self._keep_going = not self._keep_going
        return self._keep_going

    @property
    def keep_going(self) -> bool:
        return self._keep_going


@pytest.mark.parametrize(
    ("no_commit", "expected"),
    [
        (False, False),
        (True, True),
    ],
)
def test_watch_tui_app_init_no_commit(no_commit: bool, expected: bool) -> None:
    """WatchTuiApp initializes no_commit correctly."""
    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]
        app = run_tui.WatchTuiApp(_MockEngine(), tui_queue, no_commit=no_commit)
        assert app._no_commit is expected
    finally:
        manager.shutdown()


# =============================================================================
# ConfirmCommitScreen Tests
# =============================================================================


def test_confirm_commit_screen_has_bindings() -> None:
    """ConfirmCommitScreen has y, n, and escape bindings."""
    bindings = run_tui.ConfirmCommitScreen.BINDINGS
    assert len(bindings) == 3
    # Extract keys from Binding objects - bindings are Binding instances
    binding_keys = set[str]()
    for b in bindings:
        if isinstance(b, textual.binding.Binding):
            binding_keys.add(b.key)
        else:
            binding_keys.add(b[0])  # Tuple format: (key, action, description)
    assert "y" in binding_keys
    assert "n" in binding_keys
    assert "escape" in binding_keys


def test_confirm_commit_screen_has_css() -> None:
    """ConfirmCommitScreen has default CSS defined."""
    css = run_tui.ConfirmCommitScreen.DEFAULT_CSS
    assert "ConfirmCommitScreen" in css
    assert "dialog" in css


def test_confirm_commit_screen_instantiation() -> None:
    """ConfirmCommitScreen can be instantiated."""
    screen = run_tui.ConfirmCommitScreen()
    assert isinstance(screen, run_tui.ConfirmCommitScreen)


# =============================================================================
# Pilot-Based Interactive Tests
# =============================================================================


@pytest.fixture
def mock_tui_queue() -> Generator[mp.Queue[TuiMessage]]:
    """Create a mock TUI queue for testing."""
    manager = mp.Manager()
    tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]
    yield tui_queue
    manager.shutdown()


@pytest.fixture
def simple_run_app(mock_tui_queue: mp.Queue[TuiMessage]) -> run_tui.RunTuiApp:
    """Create a simple RunTuiApp for testing."""

    def executor_func() -> dict[str, executor.ExecutionSummary]:
        return {}

    return run_tui.RunTuiApp(["stage1", "stage2", "stage3"], mock_tui_queue, executor_func)


@pytest.mark.asyncio
async def test_run_app_mounts_with_correct_structure(
    simple_run_app: run_tui.RunTuiApp,
) -> None:
    """RunTuiApp mounts with stage list and detail panels."""
    async with simple_run_app.run_test():
        # Check stage list exists
        stage_list = simple_run_app.query_one("#stage-list", run_tui.StageListPanel)
        assert stage_list is not None

        # Check detail panel exists
        detail_panel = simple_run_app.query_one("#detail-panel", run_tui.TabbedDetailPanel)
        assert detail_panel is not None

        # Check tabs exist
        tabbed_content = simple_run_app.query_one("#detail-tabs", textual.widgets.TabbedContent)
        assert tabbed_content is not None


@pytest.mark.asyncio
async def test_run_app_action_nav_down_changes_selection(
    simple_run_app: run_tui.RunTuiApp,
) -> None:
    """action_nav_down navigates between stages."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()

        # Initial selection is first stage
        assert simple_run_app.selected_stage_name == "stage1"

        # Call action directly
        simple_run_app.action_nav_down()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage2"

        # Call again
        simple_run_app.action_nav_down()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage3"


@pytest.mark.asyncio
async def test_run_app_action_nav_up_changes_selection(
    simple_run_app: run_tui.RunTuiApp,
) -> None:
    """action_nav_up navigates between stages."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()

        # Start at last stage
        simple_run_app.select_stage_by_index(2)

        # Call action directly
        simple_run_app.action_nav_up()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage2"

        # Call again
        simple_run_app.action_nav_up()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage1"


@pytest.mark.asyncio
async def test_run_app_navigation_stays_at_bounds(
    simple_run_app: run_tui.RunTuiApp,
) -> None:
    """Navigation stays at list bounds (no wrap)."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()

        # At first stage, up should stay at first stage
        simple_run_app.select_stage_by_index(0)
        simple_run_app.action_nav_up()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage1", "Should stay at first stage"

        # At last stage, down should stay at last stage
        simple_run_app.select_stage_by_index(2)
        simple_run_app.action_nav_down()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage3", "Should stay at last stage"


@pytest.mark.asyncio
async def test_run_app_action_switch_focus(simple_run_app: run_tui.RunTuiApp) -> None:
    """action_switch_focus toggles between panels."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()

        # Initial focus should be on stages panel
        assert simple_run_app.focused_panel == "stages"

        # Call action to switch
        simple_run_app.action_switch_focus()
        await pilot.pause()
        assert simple_run_app.focused_panel == "detail"

        # Call again to switch back
        simple_run_app.action_switch_focus()
        await pilot.pause()
        assert simple_run_app.focused_panel == "stages"


@pytest.mark.asyncio
async def test_run_app_quit_action(simple_run_app: run_tui.RunTuiApp) -> None:
    """action_quit exits the app."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()
        # Call quit action - should not raise
        await simple_run_app.action_quit()


@pytest.mark.asyncio
async def test_run_app_stages_shown(mock_tui_queue: mp.Queue[TuiMessage]) -> None:
    """Stage names appear in the app."""
    stage_names = ["alpha", "beta", "gamma"]

    def executor_func() -> dict[str, executor.ExecutionSummary]:
        return {}

    app = run_tui.RunTuiApp(stage_names, mock_tui_queue, executor_func)

    async with app.run_test() as pilot:
        await pilot.pause()
        # Verify all stages are in the app's stage dict
        assert "alpha" in app._stages
        assert "beta" in app._stages
        assert "gamma" in app._stages


# =============================================================================
# TabbedDetailPanel Tests
# =============================================================================


def test_tabbed_detail_panel_init() -> None:
    """TabbedDetailPanel initializes with None stage."""
    panel = run_tui.TabbedDetailPanel(id="test-detail")
    assert panel._stage is None


# =============================================================================
# StageRow Tests
# =============================================================================


def test_stage_row_init() -> None:
    """StageRow initializes with StageInfo."""
    info = run_tui.StageInfo("test", 1, 3)
    row = run_tui.StageRow(info)
    assert row._info is info


# =============================================================================
# StageListPanel Tests
# =============================================================================


def test_stage_list_panel_init() -> None:
    """StageListPanel initializes with stages list."""
    stages = [
        run_tui.StageInfo("s1", 1, 2),
        run_tui.StageInfo("s2", 2, 2),
    ]
    panel = run_tui.StageListPanel(stages, id="test-list")
    assert panel._stages == stages
    assert panel._rows == {}  # Empty until mounted


# =============================================================================
# DetailPanel Tests
# =============================================================================


def test_detail_panel_init() -> None:
    """DetailPanel initializes with None stage."""
    panel = run_tui.DetailPanel(id="test-detail")
    assert panel._stage is None


def test_detail_panel_set_stage() -> None:
    """DetailPanel.set_stage updates internal stage."""
    panel = run_tui.DetailPanel()
    info = run_tui.StageInfo("test", 1, 1)
    panel.set_stage(info)
    assert panel._stage is info


# =============================================================================
# LogPanel Tests
# =============================================================================


def test_log_panel_init() -> None:
    """LogPanel initializes with empty logs and no filter."""
    panel = run_tui.LogPanel()
    assert panel._filter_stage is None
    assert len(panel._all_logs) == 0


# =============================================================================
# StageLogPanel Tests
# =============================================================================


def test_stage_log_panel_init() -> None:
    """StageLogPanel can be instantiated."""
    panel = run_tui.StageLogPanel(id="test-logs")
    assert isinstance(panel, run_tui.StageLogPanel)
