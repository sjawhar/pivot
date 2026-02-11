# pyright: reportMissingImports=false
from __future__ import annotations

import anyio
import pytest

from pivot.engine import engine as engine_mod
from pivot.engine.types import OutputEvent, StageStarted


def _helper_require_seq(event: OutputEvent) -> int:
    assert "seq" in event, "Expected seq on output event"
    return event["seq"]


def _helper_require_run_id(event: OutputEvent) -> str:
    assert "run_id" in event, "Expected run_id on output event"
    return event["run_id"]


class _SeqCollectorSink:
    def __init__(self) -> None:
        self.received: list[OutputEvent] = []

    async def handle(self, event: OutputEvent) -> None:
        self.received.append(event)

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_seq_monotonic_and_shared_across_sinks() -> None:
    sink_a = _SeqCollectorSink()
    sink_b = _SeqCollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(sink_a)
        eng.add_sink(sink_b)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(5):
                await eng.emit(StageStarted(type="stage_started", stage=f"s{i}", index=i, total=5))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    seq_a = [_helper_require_seq(event) for event in sink_a.received]
    seq_b = [_helper_require_seq(event) for event in sink_b.received]
    assert seq_a == seq_b, "All sinks must observe the same seq stream"
    assert seq_a == sorted(seq_a), "Seq must be monotonic"


@pytest.mark.anyio
async def test_seq_continues_across_engine_instances() -> None:
    sink_a = _SeqCollectorSink()
    async with engine_mod.Engine() as eng:
        eng.add_sink(sink_a)
        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(3):
                await eng.emit(StageStarted(type="stage_started", stage=f"a{i}", index=i, total=3))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    max_seq = max(_helper_require_seq(event) for event in sink_a.received)

    sink_b = _SeqCollectorSink()
    async with engine_mod.Engine() as eng:
        eng.add_sink(sink_b)
        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(2):
                await eng.emit(StageStarted(type="stage_started", stage=f"b{i}", index=i, total=2))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    min_seq = min(_helper_require_seq(event) for event in sink_b.received)
    assert min_seq > max_seq, "Seq should continue across Engine instances"


@pytest.mark.anyio
async def test_run_id_stamped_on_output_events() -> None:
    sink = _SeqCollectorSink()
    async with engine_mod.Engine() as eng:
        eng.add_sink(sink)
        eng._current_run_id = "run-123"
        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            await eng.emit(StageStarted(type="stage_started", stage="s0", index=0, total=1))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    assert sink.received, "Expected at least one output event"
    assert _helper_require_run_id(sink.received[0]) == "run-123"
