"""Tests for event sinks."""

from __future__ import annotations

import time
from io import StringIO

import anyio
import pytest
import rich.live
from rich.console import Console

from pivot.engine import engine as engine_mod
from pivot.engine import sinks
from pivot.engine.types import (
    EngineDiagnostic,
    LogLine,
    OutputEvent,
    SinkState,
    StageCompleted,
    StageExecutionState,
    StageStarted,
    StageStateChanged,
)
from pivot.types import DisplayCategory, StageStatus


def _helper_render_markup(text: str) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    console.print(text)
    return output.getvalue()


def _helper_render_renderable(renderable: object) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    console.print(renderable)
    return output.getvalue()


# =============================================================================
# Formatting Helper Tests
# =============================================================================


def test_format_stage_line_ran_includes_duration() -> None:
    line = sinks._format_stage_line(
        index=1,
        total=12,
        stage="train",
        category=DisplayCategory.SUCCESS,
        duration_ms=3400.0,
        name_width=10,
    )
    rendered = _helper_render_markup(line)
    assert "[ 1/12]" in rendered, "Should include right-aligned progress counter"
    assert "✓" in rendered, "Should include checkmark for RAN"
    assert "train" in rendered, "Should include stage name"
    assert "done" in rendered, "Should include done status word"
    assert "3.4s" in rendered, "Should include formatted duration"


def test_format_stage_line_skipped_omits_duration() -> None:
    line = sinks._format_stage_line(
        index=2,
        total=12,
        stage="skip_stage",
        category=DisplayCategory.CACHED,
        duration_ms=1200.0,
        name_width=12,
    )
    rendered = _helper_render_markup(line)
    assert "○" in rendered, "Should include skip symbol"
    assert "cached" in rendered, "Should include cached status word"
    assert "1.2s" not in rendered, "Cached stages should not include duration"


def test_format_stage_line_failed_uses_failed_status() -> None:
    line = sinks._format_stage_line(
        index=3,
        total=12,
        stage="fail_stage",
        category=DisplayCategory.FAILED,
        duration_ms=1000.0,
        name_width=12,
    )
    rendered = _helper_render_markup(line)
    assert "✗" in rendered, "Should include failure symbol"
    assert "FAILED" in rendered, "Should include FAILED status word"
    assert "1.0s" not in rendered, "Failed stages should not include duration"


def test_format_stage_line_truncates_long_names() -> None:
    line = sinks._format_stage_line(
        index=1,
        total=1,
        stage="very_long_stage_name",
        category=DisplayCategory.SUCCESS,
        duration_ms=100.0,
        name_width=8,
    )
    rendered = _helper_render_markup(line)
    assert "very_lo…" in rendered, "Should truncate long names with ellipsis"
    assert "very_long_stage_name" not in rendered, "Should not include full long name"


def test_format_skip_group_line_includes_range_and_count() -> None:
    line = sinks._format_skip_group_line(start_index=1, end_index=3, total=9, count=3)
    rendered = _helper_render_markup(line)
    assert "1–3/9" in rendered, "Should include collapsed range"
    assert "3 stages not run" in rendered, "Should include skipped count text"
    assert "○" in rendered, "Should include skip symbol"


def test_format_error_detail_indents_each_line() -> None:
    details = sinks._format_error_detail("Line 1\nLine 2", total=12)
    assert len(details) == 2, "Should return one line per input line"
    rendered = [_helper_render_markup(line) for line in details]
    leading_spaces = [len(line) - len(line.lstrip(" ")) for line in rendered]
    assert leading_spaces[0] >= 2, "Error detail lines should be indented"
    assert leading_spaces[0] == leading_spaces[1], "Indentation should be consistent"
    assert "Line 1" in rendered[0], "First error line should be present"
    assert "Line 2" in rendered[1], "Second error line should be present"


# =============================================================================
# StaticConsoleSink Tests
# =============================================================================


async def test_static_sink_prints_ran_stage_line() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        seq=0,
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1200.0,
        index=1,
        total=1,
        run_id="test-run",
        input_hash=None,
    )
    await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "1/1" in result, "Should include progress counter"
    assert "train" in result, "Should include stage name"
    assert "done" in result, "Should include done status"
    assert "1.2s" in result, "Should include duration"


async def test_static_sink_prints_skipped_stages_individually_when_low_count() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    events = [
        StageCompleted(
            type="stage_completed",
            seq=idx,
            stage=f"skip_{idx}",
            status=StageStatus.CACHED,
            reason="up-to-date",
            duration_ms=10.0,
            index=idx + 1,
            total=3,
            run_id="test-run",
            input_hash=None,
        )
        for idx in range(2)
    ]
    for event in events:
        await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "skip_0" in result, "Should include first skipped stage"
    assert "skip_1" in result, "Should include second skipped stage"
    assert "cached" in result, "Should show cached status for up-to-date stages"
    assert "stages not run" not in result, "Should not collapse low skip counts"


async def test_static_sink_collapses_skips_over_threshold() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    for idx in range(21):
        event = StageCompleted(
            type="stage_completed",
            seq=idx,
            stage=f"skip_{idx}",
            status=StageStatus.CACHED,
            reason="up-to-date",
            duration_ms=10.0,
            index=idx + 1,
            total=21,
            run_id="test-run",
            input_hash=None,
        )
        await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "21 stages not run" in result, "Should collapse skipped stages over threshold"


async def test_static_sink_does_not_collapse_skips_at_threshold() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    for idx in range(20):
        event = StageCompleted(
            type="stage_completed",
            seq=idx,
            stage=f"skip_{idx}",
            status=StageStatus.CACHED,
            reason="up-to-date",
            duration_ms=10.0,
            index=idx + 1,
            total=20,
            run_id="test-run",
            input_hash=None,
        )
        await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "stages not run" not in result, "Should not collapse skips at threshold"


async def test_static_sink_ignores_waiting_on_lock_state_changes() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    event = StageStateChanged(
        type="stage_state_changed",
        seq=0,
        stage="train",
        state=StageExecutionState.WAITING_ON_LOCK,
        previous_state=StageExecutionState.PREPARING,
        run_id="test-run",
    )
    await sink.handle(event)
    await sink.close()

    assert output.getvalue() == "", "Should not print state changes"


async def test_static_sink_prints_failed_with_error_details() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        seq=0,
        stage="train",
        status=StageStatus.FAILED,
        reason="first line\nsecond line",
        duration_ms=100.0,
        index=1,
        total=1,
        run_id="test-run",
        input_hash=None,
    )
    await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "FAILED" in result, "Should include FAILED status"
    assert "first line" in result, "Should include first error detail"
    assert "second line" in result, "Should include second error detail"


async def test_static_sink_ignores_stage_started() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    event = StageStarted(
        type="stage_started",
        seq=0,
        stage="train",
        index=1,
        total=2,
        run_id="test-run",
    )
    await sink.handle(event)
    await sink.close()

    assert output.getvalue() == "", "Should not print stage_started events"


async def test_static_sink_sorts_completions_by_index() -> None:
    """Completions are printed sorted by index on close(), not arrival order."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    # Send in reverse index order
    for idx, name in [(3, "third"), (1, "first"), (2, "second")]:
        await sink.handle(
            StageCompleted(
                type="stage_completed",
                seq=idx - 1,
                stage=name,
                status=StageStatus.RAN,
                reason="",
                duration_ms=100.0,
                index=idx,
                total=3,
                run_id="test-run",
                input_hash=None,
            )
        )
    await sink.close()

    result = output.getvalue()
    assert result.index("first") < result.index("second") < result.index("third"), (
        "Completions should be sorted by index"
    )


async def test_static_sink_prints_log_lines_when_show_output_enabled() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result, "Should include stage name"
    assert "Processing batch" in result, "Should include log line text"


async def test_static_sink_stderr_log_line() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Warning: GPU not available",
        is_stderr=True,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result, "Should include stage name"
    assert "Warning: GPU not available" in result, "Should include stderr text"


async def test_static_sink_escapes_rich_markup_in_log_lines() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="[bold red]FAKE ERROR[/bold red] - this should display literally",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result, "Should include stage name"
    assert "FAKE ERROR" in result, "Should include escaped log content"
    assert "this should display literally" in result, "Should include full log line"
    assert "[bold red]" in result or "\\[bold red]" in result, (
        "Should render markup tags as literal text"
    )


async def test_static_sink_ignores_log_lines_when_show_output_disabled() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console, show_output=False)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)

    assert output.getvalue() == "", "Should ignore log lines when show_output=False"


async def test_static_sink_log_lines_stream_before_completions() -> None:
    """Log lines print in real-time, completions print on close."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console, show_output=True)

    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=0,
            stage="skip_stage",
            status=StageStatus.CACHED,
            reason="cached",
            duration_ms=10.0,
            index=1,
            total=2,
            run_id="test-run",
            input_hash=None,
        )
    )
    await sink.handle(
        LogLine(
            type="log_line",
            seq=1,
            stage="log_stage",
            line="hello",
            is_stderr=False,
        )
    )

    # Before close: only log lines should be printed (completions are buffered)
    before_close = output.getvalue()
    assert "log_stage" in before_close, "Log lines should stream immediately"
    assert "skip_stage" not in before_close, "Completions should be buffered until close"

    await sink.close()

    # After close: completions appear
    result = output.getvalue()
    assert "skip_stage" in result, "Completions should appear after close"


async def test_static_sink_prints_engine_diagnostics() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.StaticConsoleSink(console=console)

    event = EngineDiagnostic(
        type="engine_diagnostic",
        seq=0,
        message="Scheduler delay",
        detail="Loop blocked",
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "Engine diagnostic" in result, "Should include diagnostic prefix"
    assert "Scheduler delay" in result, "Should include diagnostic message"
    assert "Loop blocked" in result, "Should include diagnostic detail"


# =============================================================================
# LiveConsoleSink Tests
# =============================================================================


async def test_live_sink_prints_completion_to_scrollback() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    await sink.handle(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=1,
            total=1,
            run_id="test-run",
        )
    )
    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=1,
            stage="train",
            status=StageStatus.RAN,
            reason="",
            duration_ms=1200.0,
            index=1,
            total=1,
            run_id="test-run",
            input_hash=None,
        )
    )
    await sink.close()

    result = output.getvalue()
    assert "train" in result, "Should include stage name"
    assert "done" in result, "Should include completion status"


async def test_live_sink_tracks_running_stages() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    start = time.monotonic()
    await sink.handle(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=1,
            total=1,
            run_id="test-run",
        )
    )

    assert "train" in sink._running, "Should track running stage"  # type: ignore[reportPrivateUsage] - testing internal state
    assert sink._running["train"] >= start, "Start time should be recorded"  # type: ignore[reportPrivateUsage] - testing internal state
    assert isinstance(sink._live, rich.live.Live), "Should start Rich Live renderer"  # type: ignore[reportPrivateUsage] - testing internal state

    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=1,
            stage="train",
            status=StageStatus.RAN,
            reason="",
            duration_ms=50.0,
            index=1,
            total=1,
            run_id="test-run",
            input_hash=None,
        )
    )
    await sink.close()

    assert "train" not in sink._running, "Should remove stage after completion"  # type: ignore[reportPrivateUsage] - testing internal state


async def test_live_sink_ignores_waiting_on_lock() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    event = StageStateChanged(
        type="stage_state_changed",
        seq=0,
        stage="train",
        state=StageExecutionState.WAITING_ON_LOCK,
        previous_state=StageExecutionState.PREPARING,
        run_id="test-run",
    )
    await sink.handle(event)
    await sink.close()

    assert output.getvalue() == "", "Should ignore waiting-on-lock events"


async def test_live_sink_collapses_many_skips_in_scrollback() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    for idx in range(21):
        await sink.handle(
            StageCompleted(
                type="stage_completed",
                seq=idx,
                stage=f"skip_{idx}",
                status=StageStatus.CACHED,
                reason="cached",
                duration_ms=10.0,
                index=idx,
                total=21,
                run_id="test-run",
                input_hash=None,
            )
        )
    await sink.close()

    result = output.getvalue()
    assert "21 stages not run" in result, "Should collapse skipped stages over threshold"


async def test_live_sink_prints_all_statuses_on_close() -> None:
    """LiveConsoleSink prints ran, skipped, failed completions on close."""
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=0,
            stage="ran_stage",
            status=StageStatus.RAN,
            reason="",
            duration_ms=1000.0,
            index=1,
            total=3,
            run_id="test-run",
            input_hash=None,
        )
    )
    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=1,
            stage="skipped_stage",
            status=StageStatus.CACHED,
            reason="cached",
            duration_ms=2000.0,
            index=2,
            total=3,
            run_id="test-run",
            input_hash=None,
        )
    )
    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=2,
            stage="failed_stage",
            status=StageStatus.FAILED,
            reason="boom",
            duration_ms=500.0,
            index=3,
            total=3,
            run_id="test-run",
            input_hash=None,
        )
    )
    await sink.close()

    result = output.getvalue()
    assert "ran_stage" in result and "done" in result, "Should include ran stage"
    assert "skipped_stage" in result and "cached" in result, (
        "Should include skipped stage with cached status"
    )
    assert "failed_stage" in result and "FAILED" in result, "Should include failed stage"


async def test_live_sink_show_output_prints_log_lines() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)
    await sink.close()

    result = output.getvalue()
    assert "train" in result, "Should include stage name"
    assert "Processing batch" in result, "Should include log line text"


async def test_live_sink_hides_log_lines_by_default() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console, show_output=False)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)
    await sink.close()

    assert output.getvalue() == "", "Should ignore log lines when show_output=False"


async def test_live_sink_live_renderable_shows_running_stages() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    await sink.handle(
        StageStarted(
            type="stage_started",
            seq=0,
            stage="train",
            index=1,
            total=1,
            run_id="test-run",
        )
    )

    rendered = _helper_render_renderable(
        sink._build_live_group()  # type: ignore[reportPrivateUsage] - testing live renderable
    )
    await sink.close()

    assert "train" in rendered, "Live renderable should include running stage"


async def test_live_sink_live_renderable_shows_progress() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = sinks.LiveConsoleSink(console=console)

    await sink.handle(
        StageCompleted(
            type="stage_completed",
            seq=0,
            stage="train",
            status=StageStatus.RAN,
            reason="",
            duration_ms=1000.0,
            index=1,
            total=3,
            run_id="test-run",
            input_hash=None,
        )
    )

    rendered = _helper_render_renderable(
        sink._build_live_group()  # type: ignore[reportPrivateUsage] - testing live renderable
    )
    await sink.close()

    assert "1/3" in rendered, "Live renderable should include progress counts"


# =============================================================================
# ResultCollectorSink Tests
# =============================================================================


async def test_result_collector_sink_collects_completed() -> None:
    """ResultCollectorSink collects stage_completed events."""
    sink = sinks.ResultCollectorSink()

    event = StageCompleted(
        type="stage_completed",
        seq=0,
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1000,
        index=1,
        total=1,
        run_id="test-run",
        input_hash=None,
    )
    await sink.handle(event)

    results = await sink.get_results()
    assert "train" in results
    assert results["train"]["status"] == StageStatus.RAN

    await sink.close()


async def test_result_collector_sink_ignores_other_events() -> None:
    """ResultCollectorSink ignores non-completed events."""
    sink = sinks.ResultCollectorSink()

    event = StageStarted(
        type="stage_started",
        seq=0,
        stage="train",
        index=1,
        total=2,
        run_id="test-run",
    )
    await sink.handle(event)

    results = await sink.get_results()
    assert len(results) == 0


@pytest.mark.anyio
async def test_result_collector_sink_concurrent_access() -> None:
    """ResultCollectorSink protects shared state with lock under concurrent access."""
    sink = sinks.ResultCollectorSink()

    async def worker(stage_name: str) -> None:
        event = StageCompleted(
            type="stage_completed",
            seq=0,
            stage=stage_name,
            status=StageStatus.RAN,
            reason="test",
            duration_ms=100.0,
            index=1,
            total=1,
            run_id="test-run",
            input_hash=None,
        )
        for _ in range(50):
            await sink.handle(event)
            _ = await sink.get_results()

    # Run multiple concurrent tasks (like Engine dispatching to sinks)
    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(worker, f"stage_{i}")

    results = await sink.get_results()
    assert len(results) == 5, "Should have results from all stages without data races"


@pytest.mark.anyio
async def test_result_collector_sink_prevents_lost_updates() -> None:
    """ResultCollectorSink doesn't lose updates during concurrent writes.

    This test verifies that the lock actually prevents race conditions by
    checking that the final result for each stage matches the last iteration.
    """
    sink = sinks.ResultCollectorSink()

    async def worker(stage_name: str) -> None:
        for i in range(100):
            event = StageCompleted(
                type="stage_completed",
                seq=i,
                stage=stage_name,
                status=StageStatus.RAN,
                reason=f"iteration_{i}",
                duration_ms=float(i),
                index=1,
                total=1,
                run_id="test-run",
                input_hash=None,
            )
            await sink.handle(event)

    # Run multiple concurrent tasks
    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(worker, f"stage_{i}")

    results = await sink.get_results()
    assert len(results) == 5, "Should have results from all stages"

    # Verify final iteration was recorded for each stage (not lost to race)
    for i in range(5):
        stage_result = results[f"stage_{i}"]
        assert stage_result["reason"] == "iteration_99", (
            f"stage_{i} should have final iteration result, not intermediate"
        )
        assert stage_result["duration_ms"] == 99.0, (
            f"stage_{i} should have final duration, verifying no corruption"
        )


# =============================================================================
# Per-Sink Queue Dispatch Tests
# =============================================================================

_EVENTS = [
    StageStarted(
        type="stage_started",
        seq=i,
        stage=f"s{i}",
        index=i + 1,
        total=5,
        run_id="test-run",
    )
    for i in range(5)
]


class _SlowSink:
    """Sink that sleeps on each event to simulate a slow consumer."""

    def __init__(self) -> None:
        self.received: list[OutputEvent] = []

    async def handle(self, event: OutputEvent) -> None:
        await anyio.sleep(0.01)
        self.received.append(event)

    async def close(self) -> None:
        pass


class _FastCollectorSink:
    """Sink that records events immediately."""

    def __init__(self) -> None:
        self.received: list[OutputEvent] = []

    async def handle(self, event: OutputEvent) -> None:
        self.received.append(event)

    async def close(self) -> None:
        pass


async def test_slow_sink_does_not_block_fast_sink() -> None:
    """Fast sink receives all events while slow sink is still processing.

    With per-sink queues, each sink processes independently. The fast sink
    should finish well before the slow sink.
    """
    slow = _SlowSink()
    fast = _FastCollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(fast)
        eng.add_sink(slow)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)

            # Send events through the engine's output channel
            for event in _EVENTS:
                await eng.emit(event)

            # Close output channel to signal end of stream
            assert eng._output_send is not None
            await eng._output_send.aclose()

    # Both sinks received all events
    assert len(fast.received) == 5, "Fast sink should receive all 5 events"
    assert len(slow.received) == 5, "Slow sink should receive all 5 events"


async def test_per_sink_ordering_preserved() -> None:
    """Events arrive at each sink in the order they were emitted."""
    slow = _SlowSink()
    fast = _FastCollectorSink()

    events = [
        StageCompleted(
            type="stage_completed",
            seq=i,
            stage=f"stage_{i}",
            status=StageStatus.RAN,
            reason="",
            duration_ms=float(i),
            index=i + 1,
            total=10,
            input_hash=None,
        )
        for i in range(10)
    ]

    async with engine_mod.Engine() as eng:
        eng.add_sink(fast)
        eng.add_sink(slow)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)

            for event in events:
                await eng.emit(event)

            assert eng._output_send is not None
            await eng._output_send.aclose()

    # Verify ordering for fast sink (all events are StageCompleted with "stage" key)
    fast_stages: list[str] = []
    for e in fast.received:
        if e["type"] == "stage_completed":
            assert isinstance(e, dict)
            fast_stages.append(e["stage"])  # type: ignore[typeddict-item] - narrowed by type check
    assert fast_stages == [f"stage_{i}" for i in range(10)], "Fast sink events out of order"

    # Verify ordering for slow sink
    slow_stages: list[str] = []
    for e in slow.received:
        if e["type"] == "stage_completed":
            assert isinstance(e, dict)
            slow_stages.append(e["stage"])  # type: ignore[typeddict-item] - narrowed by type check
    assert slow_stages == [f"stage_{i}" for i in range(10)], "Slow sink events out of order"


@pytest.mark.anyio
async def test_queue_full_disables_stalled_sink() -> None:
    """When a sink stops consuming and its queue fills, it is disabled."""

    class _StallingSink:
        """Sink that stops consuming after a few events."""

        def __init__(self, consume_count: int) -> None:
            self._consume_count: int = consume_count
            self.received: int = 0
            self._stall: anyio.Event = anyio.Event()

        async def handle(self, event: OutputEvent) -> None:
            self.received += 1
            if self.received >= self._consume_count:
                # Stop consuming — simulate a stuck sink
                await self._stall.wait()

        async def unstall(self) -> None:
            self._stall.set()

        async def close(self) -> None:
            self._stall.set()  # Ensure cleanup doesn't hang

    stalling = _StallingSink(consume_count=1)
    fast = _FastCollectorSink()

    sent_count = 0

    async with engine_mod.Engine() as eng:
        eng.add_sink(fast)
        eng.add_sink(stalling)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)

            # Send enough events to fill the stalling sink's 1024-item queue.
            # The stalling sink blocks on event 1, so the queue fills after 1025 more.
            for i in range(2000):
                await eng.emit(
                    StageStarted(
                        type="stage_started", seq=i, stage=f"s{i}", index=i + 1, total=2000
                    )
                )
                sent_count += 1

            # Unstall the sink so everything can drain
            await stalling.unstall()

            # Close output to let dispatch finish
            assert eng._output_send is not None
            await eng._output_send.aclose()

    # Fast sink receives events and sees the stalled sink disabled
    assert sent_count == 2000, "Expected to send all events without stalling"
    disabled_events = [
        event
        for event in fast.received
        if event["type"] == "sink_state_changed"
        and event["state"] == SinkState.DISABLED
        and event["sink_id"] == "_StallingSink"
    ]
    assert disabled_events, "Expected stalled sink to be disabled"


async def test_sink_error_does_not_stop_other_events() -> None:
    """A sink that raises on one event continues receiving subsequent events."""
    collector = _FastCollectorSink()

    class _ErrorOnceSink:
        """Sink that raises on the first event, then works normally."""

        def __init__(self) -> None:
            self.received: list[OutputEvent] = []
            self._first: bool = True

        async def handle(self, event: OutputEvent) -> None:
            if self._first:
                self._first = False
                raise ValueError("boom")
            self.received.append(event)

        async def close(self) -> None:
            pass

    error_sink = _ErrorOnceSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(collector)
        eng.add_sink(error_sink)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)

            for event in _EVENTS:
                await eng.emit(event)

            assert eng._output_send is not None
            await eng._output_send.aclose()

    # Collector got all events (unaffected by error sink)
    assert len(collector.received) == 5, "Collector should receive all events"
    # Error sink got events after the first (which raised)
    assert len(error_sink.received) == 4, "Error sink should receive events after the error"
