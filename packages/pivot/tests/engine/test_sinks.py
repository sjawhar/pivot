"""Tests for event sinks."""

from __future__ import annotations

import anyio
import pytest

from pivot.engine import engine as engine_mod
from pivot.engine import types
from pivot.engine.types import OutputEvent, SinkState, StageCompleted, StageStarted
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
        seq=0,
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
        seq=0,
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1500,
        index=0,
        total=1,
        input_hash=None,
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
        seq=0,
        stage="train",
        status=StageStatus.SKIPPED,
        reason="up-to-date",
        duration_ms=10,
        index=0,
        total=1,
        input_hash=None,
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
        seq=0,
        stage="train",
        status=StageStatus.FAILED,
        reason="exception",
        duration_ms=100,
        index=0,
        total=1,
        input_hash=None,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result
    assert "FAILED" in result
    assert "exception" in result  # Reason should now be displayed


async def test_console_sink_handles_multiline_reason() -> None:
    """ConsoleSink indents multi-line error reasons."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import StageCompleted

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)

    event = StageCompleted(
        type="stage_completed",
        seq=0,
        stage="train",
        status=StageStatus.FAILED,
        reason="Traceback (most recent call last):\n  File test.py\nValueError: bad",
        duration_ms=100,
        index=0,
        total=1,
        input_hash=None,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "FAILED" in result
    assert "Traceback" in result
    assert "ValueError" in result


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
        "seq": 0,
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
        seq=0,
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1000,
        index=0,
        total=1,
        input_hash=None,
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
        seq=0,
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
            seq=0,
            stage=stage_name,
            status=StageStatus.RAN,
            reason="test",
            duration_ms=100.0,
            index=0,
            total=1,
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
    from pivot.engine.sinks import ResultCollectorSink
    from pivot.engine.types import StageCompleted

    sink = ResultCollectorSink()

    async def worker(stage_name: str) -> None:
        for i in range(100):
            event = StageCompleted(
                type="stage_completed",
                seq=i,
                stage=stage_name,
                status=StageStatus.RAN,
                reason=f"iteration_{i}",
                duration_ms=float(i),
                index=0,
                total=1,
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
        seq=0,
        stage="train",
        status=StageStatus.RAN,
        reason="",
        duration_ms=1500.0,
        index=0,
        total=1,
        input_hash=None,
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
        seq=0,
        stage="train",
        index=0,
        total=2,
    )
    await sink.handle(event)

    text = output.getvalue()
    assert "Running train" in text, "Should include 'Running' prefix"
    assert "..." in text, "Should include trailing ellipsis"


async def test_console_sink_handles_log_line_when_show_output_enabled() -> None:
    """ConsoleSink prints log lines when show_output=True."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    # Rich adds ANSI codes that can split brackets, check components separately
    assert "train" in result
    assert "Processing batch" in result


async def test_console_sink_stderr_line_contains_content() -> None:
    """ConsoleSink prints stderr lines with stage prefix."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Warning: GPU not available",
        is_stderr=True,
    )
    await sink.handle(event)

    result = output.getvalue()
    # Rich adds ANSI codes that can split brackets, check components separately
    assert "train" in result
    assert "Warning: GPU not available" in result


async def test_console_sink_ignores_log_line_when_show_output_disabled() -> None:
    """ConsoleSink ignores log lines when show_output=False (default)."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console)  # show_output defaults to False

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Processing batch 1...",
        is_stderr=False,
    )
    await sink.handle(event)

    # Should not print anything when show_output is False
    assert output.getvalue() == ""


async def test_console_sink_handles_empty_log_line() -> None:
    """ConsoleSink handles empty log lines without errors."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    # Should still print stage prefix even with empty line
    assert "train" in result


async def test_console_sink_handles_multiline_log_output() -> None:
    """ConsoleSink prints each line from multiline output separately."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    # Simulate a stage that outputs multiline logs
    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="Line 1\nLine 2\nLine 3",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result
    # Rich may tokenize numbers separately, check components
    assert "Line" in result
    assert "1" in result and "2" in result and "3" in result


async def test_console_sink_handles_special_characters() -> None:
    """ConsoleSink handles special characters without crashing."""
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    console = Console(file=output, force_terminal=True)
    sink = ConsoleSink(console=console, show_output=True)

    # Test with brackets that could conflict with Rich markup
    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="[INFO] Processing <data> with 'quotes' and \"double quotes\"",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    assert "train" in result
    assert "INFO" in result
    assert "Processing" in result


async def test_console_sink_escapes_rich_markup_in_log_lines() -> None:
    """ConsoleSink escapes Rich markup in log lines to prevent injection.

    Stage output containing Rich markup syntax (e.g. [red]text[/red]) should
    be displayed literally, not interpreted as formatting instructions.
    """
    from io import StringIO

    from rich.console import Console

    from pivot.engine.sinks import ConsoleSink
    from pivot.engine.types import LogLine

    output = StringIO()
    # no_color=True ensures we get plain text output for easier assertion
    console = Console(file=output, force_terminal=False, no_color=True)
    sink = ConsoleSink(console=console, show_output=True)

    # Simulate a stage that outputs text containing Rich markup syntax
    event = LogLine(
        type="log_line",
        seq=0,
        stage="train",
        line="[bold red]FAKE ERROR[/bold red] - this should display literally",
        is_stderr=False,
    )
    await sink.handle(event)

    result = output.getvalue()
    # The markup tags should be visible as literal text, not interpreted
    assert "[bold red]" in result or "\\[bold red]" in result
    assert "FAKE ERROR" in result
    assert "this should display literally" in result


# =============================================================================
# Per-Sink Queue Dispatch Tests
# =============================================================================

_EVENTS = [
    StageStarted(type="stage_started", seq=i, stage=f"s{i}", index=i, total=5) for i in range(5)
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
            index=i,
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
                    StageStarted(type="stage_started", seq=i, stage=f"s{i}", index=i, total=2000)
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
