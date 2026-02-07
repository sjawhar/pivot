from __future__ import annotations

import collections
import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict
from unittest.mock import AsyncMock

import pytest
import textual.binding
import textual.widgets

from pivot import loaders, outputs
from pivot_tui import run as run_tui
from pivot_tui.screens import ConfirmCommitScreen
from pivot_tui.types import LogEntry, StageInfo
from pivot_tui.widgets import (
    StageListPanel,
    StageLogPanel,
    StageRow,
    TabbedDetailPanel,
)
from pivot_tui.widgets import status as tui_status
from pivot.types import (
    StageStatus,
    TuiLogMessage,
    TuiMessageType,
    TuiReloadMessage,
    TuiStatusMessage,
    TuiWatchMessage,
    WatchStatus,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline

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
# StageInfo Tests
# =============================================================================


def test_stage_info_initialization() -> None:
    """StageInfo initializes with correct defaults."""
    info = StageInfo("test_stage", 1, 5)

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
    info = StageInfo("test", 1, 1)

    for i in range(1500):
        info.logs.append(LogEntry(f"line {i}", False, 1234567890.0 + i))

    assert len(info.logs) == 1000, "Logs should be bounded to 1000 entries"
    assert info.logs[0] == LogEntry("line 500", False, 1234567890.0 + 500), (
        "Oldest entries should be dropped"
    )


# =============================================================================
# TuiUpdate Message Tests
# =============================================================================


@pytest.mark.parametrize(
    ("msg", "expected_type"),
    [
        pytest.param(
            TuiLogMessage(
                type=TuiMessageType.LOG,
                stage="test",
                line="output",
                is_stderr=False,
                timestamp=1234567890.0,
            ),
            TuiMessageType.LOG,
            id="log_message",
        ),
        pytest.param(
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
            TuiMessageType.STATUS,
            id="status_message",
        ),
    ],
)
def test_tui_update_wraps_messages(
    msg: TuiLogMessage | TuiStatusMessage, expected_type: TuiMessageType
) -> None:
    """TuiUpdate correctly wraps different message types."""
    update = run_tui.TuiUpdate(msg)
    assert update.msg == msg
    assert update.msg is not None
    assert update.msg["type"] == expected_type


# =============================================================================
# PivotApp Initialization Tests (Run Mode)
# =============================================================================


def test_run_tui_app_init() -> None:
    """PivotApp initializes with stage names."""
    stage_names = ["stage1", "stage2", "stage3"]

    app = run_tui.PivotApp(stage_names=stage_names)

    assert len(app._stages) == 3
    assert list(app._stage_order) == stage_names
    assert app._selected_idx == 0
    assert app._results is None


def test_run_tui_app_stage_info_indexes() -> None:
    """PivotApp assigns correct 1-based indexes to stages."""
    stage_names = ["first", "second", "third"]

    app = run_tui.PivotApp(stage_names=stage_names)

    assert app._stages["first"].index == 1
    assert app._stages["second"].index == 2
    assert app._stages["third"].index == 3

    for _name, info in app._stages.items():
        assert info.total == 3


# =============================================================================
# STATUS_STYLES Tests
# =============================================================================


def test_status_functions_cover_all_statuses() -> None:
    """get_status_symbol and get_status_label handle all StageStatus values."""
    for status in StageStatus:
        symbol, style = tui_status.get_status_symbol(status)
        assert isinstance(symbol, str), f"Symbol for {status} should be string"
        assert isinstance(style, str), f"Style for {status} should be string"

        label, label_style = tui_status.get_status_label(status)
        assert isinstance(label, str), f"Label for {status} should be string"
        assert isinstance(label_style, str), f"Label style for {status} should be string"


def test_status_functions_return_non_empty() -> None:
    """Status functions return non-empty values."""
    for status in StageStatus:
        symbol, _ = tui_status.get_status_symbol(status)
        assert len(symbol) > 0, f"Symbol for {status} should not be empty"

        label, _ = tui_status.get_status_label(status)
        assert len(label) > 0, f"Label for {status} should not be empty"


# =============================================================================
# PivotApp Tests (Watch Mode)
# =============================================================================


@pytest.mark.parametrize(
    ("no_commit", "expected"),
    [
        (False, False),
        (True, True),
    ],
)
def test_watch_tui_app_init_no_commit(no_commit: bool, expected: bool) -> None:
    """PivotApp (watch mode) initializes no_commit correctly."""
    app = run_tui.PivotApp(
        stage_names=["stage1"],
        watch_mode=True,
        no_commit=no_commit,
    )
    assert app._no_commit is expected


# =============================================================================
# ConfirmCommitScreen Tests
# =============================================================================


def test_confirm_commit_screen_has_bindings() -> None:
    """ConfirmCommitScreen has y, n, and escape bindings."""
    bindings = ConfirmCommitScreen.BINDINGS
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
    """ConfirmCommitScreen has CSS path defined."""
    assert ConfirmCommitScreen.CSS_PATH is not None
    assert "modal.tcss" in ConfirmCommitScreen.CSS_PATH


def test_confirm_commit_screen_instantiation() -> None:
    """ConfirmCommitScreen can be instantiated."""
    screen = ConfirmCommitScreen()
    assert isinstance(screen, ConfirmCommitScreen)


# =============================================================================
# Pilot-Based Interactive Tests
# =============================================================================


@pytest.fixture
def simple_run_app(
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
) -> run_tui.PivotApp:
    """Create a simple PivotApp for testing."""
    return run_tui.PivotApp(
        stage_names=["stage1", "stage2", "stage3"],
    )


@pytest.mark.asyncio
async def test_run_app_mounts_with_correct_structure(
    simple_run_app: run_tui.PivotApp,
) -> None:
    """PivotApp mounts with stage list and detail panels."""
    async with simple_run_app.run_test():
        # Check stage list exists
        stage_list = simple_run_app.query_one("#stage-list", StageListPanel)
        assert stage_list is not None

        # Check detail panel exists
        detail_panel = simple_run_app.query_one("#detail-panel", TabbedDetailPanel)
        assert detail_panel is not None

        # Check tabs exist
        tabbed_content = simple_run_app.query_one("#detail-tabs", textual.widgets.TabbedContent)
        assert tabbed_content is not None


@pytest.mark.asyncio
async def test_run_app_action_nav_down_changes_selection(
    simple_run_app: run_tui.PivotApp,
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
    simple_run_app: run_tui.PivotApp,
) -> None:
    """action_nav_up navigates between stages."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()

        # Start at last stage
        simple_run_app._select_stage(2)
        simple_run_app._update_detail_panel()

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
    simple_run_app: run_tui.PivotApp,
) -> None:
    """Navigation stays at list bounds (no wrap)."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()

        # At first stage, up should stay at first stage
        simple_run_app._select_stage(0)
        simple_run_app._update_detail_panel()
        simple_run_app.action_nav_up()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage1", "Should stay at first stage"

        # At last stage, down should stay at last stage
        simple_run_app._select_stage(2)
        simple_run_app._update_detail_panel()
        simple_run_app.action_nav_down()
        await pilot.pause()
        assert simple_run_app.selected_stage_name == "stage3", "Should stay at last stage"


@pytest.mark.asyncio
async def test_run_app_quit_action(simple_run_app: run_tui.PivotApp) -> None:
    """action_quit exits the app."""
    async with simple_run_app.run_test() as pilot:
        await pilot.pause()
        # Call quit action - should not raise
        await simple_run_app.action_quit()


@pytest.mark.asyncio
async def test_run_app_stages_shown(
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
) -> None:
    """Stage names appear in the app."""
    stage_names = ["alpha", "beta", "gamma"]

    app = run_tui.PivotApp(stage_names=stage_names)

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
    panel = TabbedDetailPanel(id="test-detail")
    assert panel._stage is None


# =============================================================================
# StageRow Tests
# =============================================================================


def test_stage_row_init() -> None:
    """StageRow initializes with StageInfo."""
    info = StageInfo("test", 1, 3)
    row = StageRow(info)
    assert row._info is info


# =============================================================================
# StageListPanel Tests
# =============================================================================


def test_stage_list_panel_init() -> None:
    """StageListPanel initializes with stages list."""
    stages = [
        StageInfo("s1", 1, 2),
        StageInfo("s2", 2, 2),
    ]
    panel = StageListPanel(stages, id="test-list")
    assert panel._stages == stages
    assert panel._rows == {}  # Empty until mounted


# =============================================================================
# StageLogPanel Tests
# =============================================================================


def test_stage_log_panel_init() -> None:
    """StageLogPanel can be instantiated."""
    panel = StageLogPanel(id="test-logs")
    assert isinstance(panel, StageLogPanel)


# =============================================================================
# TUI Log File Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tui_app_with_tui_log_writes_to_file(
    tmp_path: pathlib.Path,
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
) -> None:
    """PivotApp writes messages to tui_log file when configured."""
    import json

    log_path = tmp_path / "tui.jsonl"

    app = run_tui.PivotApp(
        stage_names=["stage1"],
        tui_log=log_path,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        # Send a status message via post_message (new direct approach)
        msg = TuiStatusMessage(
            type=TuiMessageType.STATUS,
            stage="stage1",
            index=1,
            total=1,
            status=StageStatus.RAN,
            reason="code changed",
            elapsed=1.5,
            run_id="test123",
        )
        app.post_message(run_tui.TuiUpdate(msg))

        # Give app time to process message
        await pilot.pause()
        await pilot.pause()

    # Close the app to flush log file
    app._close_log_file()

    # Verify log file was written
    assert log_path.exists()
    content = log_path.read_text()

    # Should contain JSON lines
    if content.strip():
        lines = content.strip().split("\n")
        # At least one valid JSON line
        assert len(lines) >= 1
        for line in lines:
            if line.strip():
                data = json.loads(line)
                assert "type" in data


@pytest.mark.asyncio
async def test_tui_app_without_tui_log_no_file_created(
    tmp_path: pathlib.Path,
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
) -> None:
    """PivotApp does not create log file when tui_log is None."""
    log_path = tmp_path / "tui.jsonl"

    app = run_tui.PivotApp(
        stage_names=["stage1"],
        tui_log=None,  # No log file
    )

    async with app.run_test() as pilot:
        await pilot.pause()

    # Log file should not exist
    assert not log_path.exists()


# =============================================================================
# Watch Mode TUI Tests
# =============================================================================


@pytest.mark.asyncio
async def test_watch_tui_app_with_serve_flag(
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
) -> None:
    """PivotApp (watch mode) initializes serve mode correctly."""
    app = run_tui.PivotApp(
        stage_names=["stage1"],
        watch_mode=True,
        serve=True,
    )

    # App should be configured for serve mode
    assert app._serve is True

    async with app.run_test() as pilot:
        await pilot.pause()
        # App should mount and be ready in watch mode
        assert app._watch_mode is True


@pytest.mark.asyncio
async def test_watch_tui_app_with_tui_log(
    tmp_path: pathlib.Path,
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
) -> None:
    """PivotApp (watch mode) writes to tui_log when configured."""
    log_path = tmp_path / "watch_tui.jsonl"

    app = run_tui.PivotApp(
        stage_names=["stage1"],
        watch_mode=True,
        tui_log=log_path,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

    # Close log file
    app._close_log_file()

    # Log file should have been created (even if empty, touch happened in CLI)
    # The app itself creates and opens the file
    assert log_path.exists()


# =============================================================================
# Reload Summary Message Formatting (Issue #289)
# =============================================================================


@pytest.mark.parametrize(
    ("added", "removed", "modified", "expected"),
    [
        pytest.param(
            ["new"],
            ["old"],
            ["changed"],
            "Reloaded: 1 added, 1 removed, 1 modified",
            id="all_change_types",
        ),
        pytest.param([], [], ["a", "b"], "Reloaded: 2 modified", id="only_modified"),
        pytest.param(["a", "b", "c"], [], [], "Reloaded: 3 added", id="only_added"),
        pytest.param([], ["old_stage"], [], "Reloaded: 1 removed", id="only_removed"),
        pytest.param(
            ["new_a", "new_b"],
            ["old_x"],
            [],
            "Reloaded: 2 added, 1 removed",
            id="added_and_removed",
        ),
        pytest.param([], [], [], None, id="no_changes"),
    ],
)
def test_format_reload_summary(
    added: list[str],
    removed: list[str],
    modified: list[str],
    expected: str | None,
) -> None:
    """format_reload_summary builds summary from non-empty change lists."""
    summary = run_tui.format_reload_summary(
        stages_added=added,
        stages_removed=removed,
        stages_modified=modified,
    )
    assert summary == expected


def test_handle_reload_calls_notify_with_summary(mocker: MockerFixture) -> None:
    """_handle_reload calls notify with formatted summary when stages change."""
    app = run_tui.PivotApp(
        stage_names=["stage_a", "stage_b"],
        watch_mode=True,
    )

    # Mock internal methods that require mounted app
    mocker.patch.object(app, "_recompute_selection_idx")
    mocker.patch.object(app, "_rebuild_stage_list")
    mocker.patch.object(app, "_update_detail_panel")
    mock_notify = mocker.patch.object(app, "notify")

    # Create reload message with changes
    reload_msg = TuiReloadMessage(
        type=TuiMessageType.RELOAD,
        stages=["stage_a", "stage_c"],
        stages_added=["stage_c"],
        stages_removed=["stage_b"],
        stages_modified=["stage_a"],
    )

    # Call _handle_reload directly
    app._handle_reload(reload_msg)

    # Verify notify was called with the expected summary
    mock_notify.assert_called_once_with("Reloaded: 1 added, 1 removed, 1 modified")


def test_handle_reload_no_notify_when_no_changes(mocker: MockerFixture) -> None:
    """_handle_reload does not call notify when no stages changed."""
    app = run_tui.PivotApp(
        stage_names=["stage_a"],
        watch_mode=True,
    )

    # Mock internal methods that require mounted app
    mocker.patch.object(app, "_recompute_selection_idx")
    mocker.patch.object(app, "_rebuild_stage_list")
    mocker.patch.object(app, "_update_detail_panel")
    mock_notify = mocker.patch.object(app, "notify")

    # Reload message with no changes
    reload_msg = TuiReloadMessage(
        type=TuiMessageType.RELOAD,
        stages=["stage_a"],
        stages_added=[],
        stages_removed=[],
        stages_modified=[],
    )

    app._handle_reload(reload_msg)

    # notify should not be called when there are no changes
    mock_notify.assert_not_called()


# =============================================================================
# TuiShutdown Bell Notification Tests (Run Mode)
# =============================================================================


def test_on_tui_shutdown_calls_bell_in_run_mode(mocker: MockerFixture) -> None:
    """on_tui_shutdown calls bell() in run mode to notify user of completion."""
    app = run_tui.PivotApp(stage_names=["stage1"])

    # Mock methods that require mounted app or have side effects
    mocker.patch.object(app, "_shutdown_event")
    mocker.patch.object(app, "_close_log_file")
    mocker.patch.object(app, "_shutdown_loky_pool")
    mocker.patch.object(app, "exit")
    mocker.patch.object(app, "_write_to_log")
    mock_bell = mocker.patch.object(app, "bell")

    # Create TuiShutdown event
    event = run_tui.TuiShutdown()

    # Call the handler
    app.on_tui_shutdown(event)

    # Bell should be called in run mode
    mock_bell.assert_called_once()


def test_on_tui_shutdown_no_bell_in_watch_mode(mocker: MockerFixture) -> None:
    """on_tui_shutdown does not call bell() in watch mode."""
    app = run_tui.PivotApp(stage_names=["stage1"], watch_mode=True)

    # Mock methods that require mounted app
    mocker.patch.object(app, "_write_to_log")
    mock_bell = mocker.patch.object(app, "bell")

    # Create TuiShutdown event
    event = run_tui.TuiShutdown()

    # Call the handler
    app.on_tui_shutdown(event)

    # Bell should NOT be called in watch mode
    mock_bell.assert_not_called()


# =============================================================================
# format_reload_summary Edge Cases
# =============================================================================


def test_format_reload_summary_with_empty_strings_vs_empty_lists() -> None:
    """format_reload_summary treats empty lists correctly."""
    # Empty lists should return None
    summary = run_tui.format_reload_summary(
        stages_added=[],
        stages_removed=[],
        stages_modified=[],
    )
    assert summary is None


def test_format_reload_summary_single_item_pluralization() -> None:
    """format_reload_summary handles singular vs plural correctly."""
    summary = run_tui.format_reload_summary(
        stages_added=["one"],
        stages_removed=[],
        stages_modified=[],
    )
    assert summary == "Reloaded: 1 added"
    assert "1 added" in summary, "Should use singular '1 added' not '1 addeds'"


def test_format_reload_summary_order_is_consistent() -> None:
    """format_reload_summary always uses added, removed, modified order."""
    summary = run_tui.format_reload_summary(
        stages_added=["a"],
        stages_removed=["b"],
        stages_modified=["c"],
    )
    # Check that order is always: added, removed, modified
    assert summary is not None
    assert summary.index("added") < summary.index("removed")
    assert summary.index("removed") < summary.index("modified")


# =============================================================================
# TuiUpdate Message Tests - Additional Edge Cases
# =============================================================================


def test_tui_update_preserves_message_identity() -> None:
    """TuiUpdate preserves the original message object."""
    msg = TuiStatusMessage(
        type=TuiMessageType.STATUS,
        stage="test",
        index=1,
        total=1,
        status=StageStatus.RAN,
        reason="done",
        elapsed=1.0,
        run_id="test_id",
    )
    update = run_tui.TuiUpdate(msg)
    assert update.msg is msg, "Should preserve original message object"


def test_tui_update_with_reload_message() -> None:
    """TuiUpdate works with TuiReloadMessage."""
    msg = TuiReloadMessage(
        type=TuiMessageType.RELOAD,
        stages=["s1", "s2"],
        stages_added=["s2"],
        stages_removed=[],
        stages_modified=["s1"],
    )
    update = run_tui.TuiUpdate(msg)
    assert update.msg["type"] == "reload"
    assert update.msg["stages"] == ["s1", "s2"]


def test_tui_update_with_watch_message() -> None:
    """TuiUpdate works with TuiWatchMessage."""
    msg = TuiWatchMessage(
        type=TuiMessageType.WATCH,
        status=WatchStatus.WAITING,
        message="Watching",
    )
    update = run_tui.TuiUpdate(msg)
    assert update.msg["type"] == "watch"
    assert update.msg["status"] == WatchStatus.WAITING


# =============================================================================
# StageInfo Edge Cases
# =============================================================================


def test_stage_info_logs_maxlen_is_1000() -> None:
    """StageInfo logs deque maxlen should be exactly 1000."""
    info = StageInfo("test", 1, 1)
    assert info.logs.maxlen == 1000, "Logs maxlen should be 1000 to limit memory"


def test_stage_info_initial_state_is_ready() -> None:
    """StageInfo initializes with READY status."""
    info = StageInfo("test", 1, 1)
    assert info.status == StageStatus.READY


def test_stage_info_index_and_total_are_set() -> None:
    """StageInfo stores index and total for display."""
    info = StageInfo("my_stage", 3, 10)
    assert info.index == 3
    assert info.total == 10
    assert info.name == "my_stage"


# =============================================================================
# PivotApp Initialization Edge Cases
# =============================================================================


def test_run_tui_app_watch_mode() -> None:
    """PivotApp can be created with watch_mode=True."""
    app = run_tui.PivotApp(
        stage_names=["s1"],
        watch_mode=True,
    )
    assert app._watch_mode is True


def test_run_tui_app_with_empty_stage_list() -> None:
    """PivotApp handles empty stage list."""
    app = run_tui.PivotApp(stage_names=[])
    assert len(app._stages) == 0
    assert app._stage_order == []
    assert app.selected_stage_name is None


# =============================================================================
# Status Functions Consistency Tests
# =============================================================================


def test_status_functions_return_tuple_structures() -> None:
    """get_status_symbol and get_status_label return (str, str) tuples."""
    for status in StageStatus:
        symbol_result = tui_status.get_status_symbol(status)
        assert isinstance(symbol_result, tuple)
        assert len(symbol_result) == 2
        assert isinstance(symbol_result[0], str)
        assert isinstance(symbol_result[1], str)

        label_result = tui_status.get_status_label(status)
        assert isinstance(label_result, tuple)
        assert len(label_result) == 2
        assert isinstance(label_result[0], str)
        assert isinstance(label_result[1], str)


# =============================================================================
# PivotApp Property Tests
# =============================================================================


def test_pivot_app_selected_stage_name_when_no_stages() -> None:
    """selected_stage_name returns None when no stages exist."""
    app = run_tui.PivotApp(stage_names=[])
    assert app.selected_stage_name is None


def test_pivot_app_selected_stage_name_with_stages() -> None:
    """selected_stage_name returns first stage initially."""
    app = run_tui.PivotApp(stage_names=["s1", "s2", "s3"])
    assert app.selected_stage_name == "s1"


def test_pivot_app_exit_message_property_initial_state() -> None:
    """exit_message property is None initially."""
    app = run_tui.PivotApp(stage_names=["s1"])
    assert app.exit_message is None


# =============================================================================
# Force Re-Run Action Tests
# =============================================================================


@pytest.mark.anyio
async def test_action_force_rerun_stage_calls_rpc(mocker: MockerFixture) -> None:
    """action_force_rerun_stage should send RPC command for selected stage."""
    app = run_tui.PivotApp(stage_names=["stage_a", "stage_b"], watch_mode=True)
    app._selected_stage_name = "stage_a"

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)
    mock_send.return_value = True
    mocker.patch.object(app, "notify")
    mocker.patch("pivot_tui.run.project.get_project_root", return_value=pathlib.Path("/fake/root"))

    await app.action_force_rerun_stage()

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["stages"] == ["stage_a"]
    assert call_kwargs["force"] is True


@pytest.mark.anyio
async def test_action_force_rerun_all_calls_rpc(mocker: MockerFixture) -> None:
    """action_force_rerun_all should send RPC command for all stages."""
    app = run_tui.PivotApp(stage_names=["stage_a", "stage_b"], watch_mode=True)

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)
    mock_send.return_value = True
    mocker.patch.object(app, "notify")
    mocker.patch("pivot_tui.run.project.get_project_root", return_value=pathlib.Path("/fake/root"))

    await app.action_force_rerun_all()

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["stages"] is None
    assert call_kwargs["force"] is True


@pytest.mark.anyio
async def test_action_force_rerun_not_in_watch_mode(mocker: MockerFixture) -> None:
    """action_force_rerun should do nothing in run mode."""
    # Create run mode app (not watch mode - default)
    app = run_tui.PivotApp(stage_names=["stage_a"])
    app._selected_stage_name = "stage_a"

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)

    await app.action_force_rerun_stage()

    mock_send.assert_not_called()


@pytest.mark.anyio
async def test_action_force_rerun_no_stage_selected(mocker: MockerFixture) -> None:
    """action_force_rerun_stage should notify when no stage selected."""
    app = run_tui.PivotApp(stage_names=[], watch_mode=True)
    app._selected_stage_name = None

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)
    mock_notify = mocker.patch.object(app, "notify")

    await app.action_force_rerun_stage()

    mock_send.assert_not_called()
    mock_notify.assert_called_once()
    assert "No stage selected" in str(mock_notify.call_args)


@pytest.mark.anyio
async def test_action_force_rerun_while_running(mocker: MockerFixture) -> None:
    """action_force_rerun should warn when stages are running."""
    app = run_tui.PivotApp(stage_names=["stage_a"], watch_mode=True)
    app._selected_stage_name = "stage_a"
    # Simulate running stage
    app._stages["stage_a"].status = StageStatus.IN_PROGRESS

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)
    mock_notify = mocker.patch.object(app, "notify")

    await app.action_force_rerun_stage()

    mock_send.assert_not_called()
    mock_notify.assert_called_once()
    assert "running" in str(mock_notify.call_args).lower()


@pytest.mark.anyio
async def test_action_force_rerun_all_while_running(mocker: MockerFixture) -> None:
    """action_force_rerun_all should warn when stages are running."""
    app = run_tui.PivotApp(stage_names=["stage_a", "stage_b"], watch_mode=True)
    # Simulate running stage
    app._stages["stage_a"].status = StageStatus.IN_PROGRESS

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)
    mock_notify = mocker.patch.object(app, "notify")

    await app.action_force_rerun_all()

    mock_send.assert_not_called()
    mock_notify.assert_called_once()
    assert "running" in str(mock_notify.call_args).lower()


@pytest.mark.anyio
async def test_action_force_rerun_stage_failure_notifies(mocker: MockerFixture) -> None:
    """action_force_rerun_stage should notify on RPC failure."""
    app = run_tui.PivotApp(stage_names=["stage_a"], watch_mode=True)
    app._selected_stage_name = "stage_a"

    mock_send = mocker.patch("pivot_tui.run.rpc_client.send_run_command", new_callable=AsyncMock)
    mock_send.return_value = False  # RPC failure
    mock_notify = mocker.patch.object(app, "notify")
    mocker.patch("pivot_tui.run.project.get_project_root", return_value=pathlib.Path("/fake/root"))

    await app.action_force_rerun_stage()

    # Should have been called twice: once for "Forcing re-run...", once for failure
    assert mock_notify.call_count == 2
    # Last call should be the error
    last_call_args = str(mock_notify.call_args_list[-1])
    assert "Failed" in last_call_args or "error" in last_call_args


# =============================================================================
# StageDataProvider Protocol Tests
# =============================================================================


def test_stage_data_provider_protocol_is_importable() -> None:
    """StageDataProvider protocol can be imported from tui.types."""
    from pivot_tui.types import StageDataProvider

    assert hasattr(StageDataProvider, "get_stage")
    assert hasattr(StageDataProvider, "ensure_fingerprint")


def test_pivot_app_accepts_stage_data_provider(mocker: MockerFixture) -> None:
    """PivotApp stores stage_data_provider when passed."""
    from pivot_tui.types import StageDataProvider

    provider = mocker.MagicMock(spec=StageDataProvider)
    app = run_tui.PivotApp(stage_names=["s1"], stage_data_provider=provider)
    assert app._stage_data_provider is provider


def test_create_history_entry_uses_provider(mocker: MockerFixture) -> None:
    """_create_history_entry uses stage_data_provider instead of cli_helpers."""
    from pivot_tui.types import StageDataProvider

    mock_provider = mocker.MagicMock(spec=StageDataProvider)
    mock_provider.get_stage.return_value = {
        "deps_paths": [],
        "outs_paths": [],
        "params": None,
    }
    mock_provider.ensure_fingerprint.return_value = {"func": "abc123"}

    app = run_tui.PivotApp(
        stage_names=["stage_a"],
        watch_mode=True,
        stage_data_provider=mock_provider,
    )

    mocker.patch("pivot_tui.run.explain.get_stage_explanation", return_value=None)
    mocker.patch("pivot_tui.run.parameters.load_params_yaml", return_value={})
    mocker.patch("pivot_tui.run.config.get_state_dir", return_value=pathlib.Path("/fake"))

    app._create_history_entry("stage_a", "run-1")

    mock_provider.get_stage.assert_called_with("stage_a")
    mock_provider.ensure_fingerprint.assert_called_with("stage_a")


def test_create_history_entry_without_provider() -> None:
    """_create_history_entry works without provider (no snapshot)."""
    app = run_tui.PivotApp(stage_names=["stage_a"], watch_mode=True)
    # No provider, no cli_helpers mock — should not raise
    app._create_history_entry("stage_a", "run-1")
    assert "stage_a" in app._pending_history
    assert app._pending_history["stage_a"].input_snapshot is None
