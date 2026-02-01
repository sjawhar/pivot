"""Tests for event sinks."""

from __future__ import annotations

import anyio
import pytest

from pivot.engine import types
from pivot.types import StageStatus

# =============================================================================
# ConsoleSink Tests
# =============================================================================


async def test_console_sink_handles_stage_started() -> None:
    """ConsoleSink prints stage_started events."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageStarted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=2,
    )
    await sink.handle(event)
    await sink.close()

    assert "train" in output.getvalue()


async def test_console_sink_handles_stage_completed_ran() -> None:
    """ConsoleSink prints done message for RAN status."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageCompleted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1500,
        index=0,
        total=1,
    )
    await sink.handle(event)

    assert "train" in output.getvalue()
    assert "done" in output.getvalue()


async def test_console_sink_handles_stage_completed_skipped() -> None:
    """ConsoleSink prints skipped message for SKIPPED status."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageCompleted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.SKIPPED,
        reason="up-to-date",
        duration_ms=10,
        index=0,
        total=1,
    )
    await sink.handle(event)

    assert "train" in output.getvalue()
    assert "skipped" in output.getvalue()


async def test_console_sink_handles_stage_completed_failed() -> None:
    """ConsoleSink prints FAILED message for FAILED status."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageCompleted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.FAILED,
        reason="exception",
        duration_ms=100,
        index=0,
        total=1,
    )
    await sink.handle(event)

    assert "train" in output.getvalue()
    assert "FAILED" in output.getvalue()


async def test_console_sink_ignores_other_events() -> None:
    """ConsoleSink ignores events it doesn't handle."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event: types.EngineStateChanged = {
        "type": "engine_state_changed",
        "state": types.EngineState.ACTIVE,
    }
    await sink.handle(event)

    # Should not print anything for unhandled events
    assert output.getvalue() == ""


# =============================================================================
# ResultCollectorSink Tests
# =============================================================================


async def test_result_collector_sink_collects_completed() -> None:
    """ResultCollectorSink collects stage_completed events."""
    from pivot.engine.sinks import ResultCollectorSink
    from pivot.engine.types import StageCompleted

    sink = ResultCollectorSink()

    event = StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1000,
        index=0,
        total=1,
    )
    await sink.handle(event)

    results = await sink.get_results()
    assert "train" in results
    assert results["train"]["status"] == StageStatus.RAN

    await sink.close()


async def test_result_collector_sink_ignores_other_events() -> None:
    """ResultCollectorSink ignores non-completed events."""
    from pivot.engine.sinks import ResultCollectorSink
    from pivot.engine.types import StageStarted

    sink = ResultCollectorSink()

    event = StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=1,
    )
    await sink.handle(event)

    results = await sink.get_results()
    assert len(results) == 0


@pytest.mark.anyio
async def test_result_collector_sink_concurrent_access() -> None:
    """ResultCollectorSink protects shared state with lock under concurrent access."""
    from pivot.engine.sinks import ResultCollectorSink
    from pivot.engine.types import StageCompleted

    sink = ResultCollectorSink()

    async def worker(stage_name: str) -> None:
        event = StageCompleted(
            type="stage_completed",
            stage=stage_name,
            status=StageStatus.RAN,
            reason="test",
            duration_ms=100.0,
            index=0,
            total=1,
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
    from pivot.engine.sinks import ResultCollectorSink
    from pivot.engine.types import StageCompleted

    sink = ResultCollectorSink()

    async def worker(stage_name: str) -> None:
        for i in range(100):
            event = StageCompleted(
                type="stage_completed",
                stage=stage_name,
                status=StageStatus.RAN,
                reason=f"iteration_{i}",
                duration_ms=float(i),
                index=0,
                total=1,
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


async def test_console_sink_formats_duration_correctly() -> None:
    """ConsoleSink formats duration with correct precision in output."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageCompleted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1500.0,
        index=0,
        total=1,
    )
    await sink.handle(event)

    text = output.getvalue()
    assert "train" in text, "Stage name should appear"
    assert "done" in text, "Status should appear"
    # Rich adds ANSI escape codes that can split numbers, verify components
    assert "1." in text and "5s" in text, "Duration components should appear"


async def test_console_sink_running_message_format() -> None:
    """ConsoleSink prints 'Running <stage>...' for stage_started events."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageStarted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageStarted(
        type="stage_started",
        stage="train",
        index=0,
        total=2,
    )
    await sink.handle(event)

    text = output.getvalue()
    assert "Running train" in text, "Should include 'Running' prefix"
    assert "..." in text, "Should include trailing ellipsis"
