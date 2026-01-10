from __future__ import annotations

import collections
import multiprocessing as mp
import pathlib
import queue

import pytest

import pivot
from pivot import executor, project, registry, run_tui
from pivot.types import (
    DisplayMode,
    StageStatus,
    TuiLogMessage,
    TuiMessage,
    TuiMessageType,
    TuiStatusMessage,
)


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Reset registry before each test."""
    registry.REGISTRY.clear()


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set up a temporary pipeline directory."""
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    project._project_root_cache = None
    return tmp_path


# =============================================================================
# _format_elapsed Tests
# =============================================================================


def test_format_elapsed_none() -> None:
    """_format_elapsed returns empty string for None."""
    assert run_tui._format_elapsed(None) == ""


def test_format_elapsed_seconds() -> None:
    """_format_elapsed formats seconds correctly."""
    assert run_tui._format_elapsed(5.0) == " (0:05)"
    assert run_tui._format_elapsed(59.9) == " (0:59)"


def test_format_elapsed_seconds_no_prefix() -> None:
    """_format_elapsed with empty prefix omits leading space."""
    assert run_tui._format_elapsed(5.0, prefix="") == "(0:05)"


def test_format_elapsed_minutes() -> None:
    """_format_elapsed formats minutes correctly."""
    assert run_tui._format_elapsed(60.0) == " (1:00)"
    assert run_tui._format_elapsed(125.5) == " (2:05)"


def test_format_elapsed_large_values() -> None:
    """_format_elapsed handles large values."""
    assert run_tui._format_elapsed(3661.0) == " (61:01)"


# =============================================================================
# should_use_tui Tests
# =============================================================================


def test_should_use_tui_explicit_tui() -> None:
    """should_use_tui returns True when display mode is TUI."""
    assert run_tui.should_use_tui(DisplayMode.TUI) is True


def test_should_use_tui_explicit_plain() -> None:
    """should_use_tui returns False when display mode is PLAIN."""
    assert run_tui.should_use_tui(DisplayMode.PLAIN) is False


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
        info.logs.append((f"line {i}", False))

    assert len(info.logs) == 1000, "Logs should be bounded to 1000 entries"
    assert info.logs[0] == ("line 500", False), "Oldest entries should be dropped"


# =============================================================================
# PipelineStats Tests
# =============================================================================


def test_pipeline_stats_eta_str_no_remaining() -> None:
    """ETA is empty when all stages are completed."""
    stats = run_tui.PipelineStats(total=5, completed=5, failed=0, avg_stage_time=10.0)
    assert stats.eta_str() == ""


def test_pipeline_stats_eta_str_no_avg_time() -> None:
    """ETA is empty when no average time is available."""
    stats = run_tui.PipelineStats(total=5, completed=2, failed=0, avg_stage_time=0.0)
    assert stats.eta_str() == ""


def test_pipeline_stats_eta_str_seconds() -> None:
    """ETA formats as seconds when < 60s."""
    stats = run_tui.PipelineStats(total=5, completed=2, failed=0, avg_stage_time=10.0)
    assert stats.eta_str() == "~30s"


def test_pipeline_stats_eta_str_minutes() -> None:
    """ETA formats as minutes when >= 60s."""
    stats = run_tui.PipelineStats(total=10, completed=2, failed=0, avg_stage_time=30.0)
    assert stats.eta_str() == "~4m 0s"


# =============================================================================
# TuiUpdate Message Tests
# =============================================================================


def test_tui_update_with_log_message() -> None:
    """TuiUpdate can wrap log messages."""
    log_msg = TuiLogMessage(type=TuiMessageType.LOG, stage="test", line="output", is_stderr=False)
    update = run_tui.TuiUpdate(log_msg)

    assert update.msg == log_msg
    assert update.msg["type"] == TuiMessageType.LOG


def test_tui_update_with_status_message() -> None:
    """TuiUpdate can wrap status messages."""
    status_msg = TuiStatusMessage(
        type=TuiMessageType.STATUS,
        stage="test",
        index=1,
        total=5,
        status=StageStatus.IN_PROGRESS,
        reason="",
        elapsed=None,
    )
    update = run_tui.TuiUpdate(status_msg)

    assert update.msg == status_msg
    assert update.msg["type"] == TuiMessageType.STATUS


# =============================================================================
# ExecutorComplete Message Tests
# =============================================================================


def test_executor_complete_success() -> None:
    """ExecutorComplete stores results on success."""
    results = {"stage1": executor.ExecutionSummary(status=StageStatus.RAN, reason="code changed")}
    complete = run_tui.ExecutorComplete(results, error=None)

    assert complete.results == results
    assert complete.error is None


def test_executor_complete_with_error() -> None:
    """ExecutorComplete stores error on failure."""
    error = ValueError("something went wrong")
    complete = run_tui.ExecutorComplete({}, error=error)

    assert complete.results == {}
    assert complete.error is error


# =============================================================================
# RunTuiApp Initialization Tests
# =============================================================================


def test_run_tui_app_init() -> None:
    """RunTuiApp initializes with stage names and queue."""
    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]
        stage_names = ["stage1", "stage2", "stage3"]

        def executor_func() -> dict[str, executor.ExecutionSummary]:
            return {}

        app = run_tui.RunTuiApp(stage_names, tui_queue, executor_func)

        assert app._stage_names == stage_names
        assert len(app._stages) == 3
        assert list(app._stages.keys()) == stage_names
        assert app._selected_idx == 0
        assert app._show_logs is False
        assert app._results is None
        assert app._error is None
    finally:
        manager.shutdown()


def test_run_tui_app_stage_info_indexes() -> None:
    """RunTuiApp assigns correct 1-based indexes to stages."""
    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]
        stage_names = ["first", "second", "third"]

        def executor_func() -> dict[str, executor.ExecutionSummary]:
            return {}

        app = run_tui.RunTuiApp(stage_names, tui_queue, executor_func)

        assert app._stages["first"].index == 1
        assert app._stages["second"].index == 2
        assert app._stages["third"].index == 3

        for _name, info in app._stages.items():
            assert info.total == 3
    finally:
        manager.shutdown()


# =============================================================================
# TUI Queue Integration Tests
# =============================================================================


def test_executor_emits_status_messages_to_queue(pipeline_dir: pathlib.Path) -> None:
    """Executor emits TuiStatusMessage for stage start and completion."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        pathlib.Path("output.txt").write_text("done")

    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        executor.run(show_output=False, tui_queue=tui_queue)

        messages = list[TuiMessage]()
        while True:
            try:
                msg = tui_queue.get(timeout=0.1)
                if msg is None:
                    break
                messages.append(msg)
            except queue.Empty:
                break

        status_messages = [m for m in messages if m is not None and m["type"] == "status"]

        assert len(status_messages) >= 2, "Should have at least start and complete status"

        start_msg = status_messages[0]
        assert start_msg["stage"] == "process"
        assert start_msg["status"] == StageStatus.IN_PROGRESS
        assert start_msg["index"] == 1
        assert start_msg["total"] == 1

        end_msg = status_messages[-1]
        assert end_msg["stage"] == "process"
        assert end_msg["status"] in (StageStatus.RAN, StageStatus.SKIPPED, StageStatus.COMPLETED)
        assert isinstance(end_msg["elapsed"], float | None)
    finally:
        manager.shutdown()


def test_executor_emits_log_messages_to_queue(pipeline_dir: pathlib.Path) -> None:
    """Executor emits TuiLogMessage for stage output."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def process() -> None:
        print("Processing data")
        pathlib.Path("output.txt").write_text("done")

    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        executor.run(show_output=False, tui_queue=tui_queue)

        messages = list[TuiMessage]()
        while True:
            try:
                msg = tui_queue.get(timeout=0.1)
                if msg is None:
                    break
                messages.append(msg)
            except queue.Empty:
                break

        log_messages = [m for m in messages if m is not None and m["type"] == "log"]

        assert len(log_messages) >= 1, "Should have at least one log message"
        assert any("Processing data" in m["line"] for m in log_messages if m is not None), (
            "Log should contain stdout"
        )
    finally:
        manager.shutdown()


def test_executor_emits_failed_status_on_stage_failure(pipeline_dir: pathlib.Path) -> None:
    """Executor emits FAILED status when stage raises an exception."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["output.txt"])
    def failing_stage() -> None:
        raise RuntimeError("Stage failed!")

    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        executor.run(show_output=False, tui_queue=tui_queue)

        messages = list[TuiMessage]()
        while True:
            try:
                msg = tui_queue.get(timeout=0.1)
                if msg is None:
                    break
                messages.append(msg)
            except queue.Empty:
                break

        status_messages = [m for m in messages if m is not None and m["type"] == "status"]

        failed_msgs = [m for m in status_messages if m["status"] == StageStatus.FAILED]
        assert len(failed_msgs) >= 1, "Should have at least one FAILED status message"
        assert failed_msgs[0]["stage"] == "failing_stage"
    finally:
        manager.shutdown()


def test_executor_emits_status_for_multiple_stages(pipeline_dir: pathlib.Path) -> None:
    """Executor emits status messages for all stages in multi-stage pipeline."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["step1.txt"])
    def step1() -> None:
        pathlib.Path("step1.txt").write_text("step1")

    @pivot.stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        pathlib.Path("step2.txt").write_text("step2")

    @pivot.stage(deps=["step2.txt"], outs=["step3.txt"])
    def step3() -> None:
        pathlib.Path("step3.txt").write_text("step3")

    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        executor.run(show_output=False, tui_queue=tui_queue)

        messages = list[TuiMessage]()
        while True:
            try:
                msg = tui_queue.get(timeout=0.1)
                if msg is None:
                    break
                messages.append(msg)
            except queue.Empty:
                break

        status_messages = [m for m in messages if m is not None and m["type"] == "status"]
        stages_with_status = {m["stage"] for m in status_messages}

        assert "step1" in stages_with_status
        assert "step2" in stages_with_status
        assert "step3" in stages_with_status
    finally:
        manager.shutdown()


def test_executor_status_includes_correct_index_and_total(pipeline_dir: pathlib.Path) -> None:
    """Executor status messages include correct index and total counts."""
    (pipeline_dir / "input.txt").write_text("hello")

    @pivot.stage(deps=["input.txt"], outs=["step1.txt"])
    def step1() -> None:
        pathlib.Path("step1.txt").write_text("step1")

    @pivot.stage(deps=["step1.txt"], outs=["step2.txt"])
    def step2() -> None:
        pathlib.Path("step2.txt").write_text("step2")

    manager = mp.Manager()
    try:
        tui_queue: mp.Queue[TuiMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

        executor.run(show_output=False, tui_queue=tui_queue)

        messages = list[TuiMessage]()
        while True:
            try:
                msg = tui_queue.get(timeout=0.1)
                if msg is None:
                    break
                messages.append(msg)
            except queue.Empty:
                break

        status_messages = [m for m in messages if m is not None and m["type"] == "status"]

        for msg in status_messages:
            assert msg["total"] == 2, "Total should be 2 stages"
            assert msg["index"] in (1, 2), "Index should be 1 or 2"
    finally:
        manager.shutdown()


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
