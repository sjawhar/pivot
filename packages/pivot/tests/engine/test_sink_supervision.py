# pyright: reportMissingImports=false
from __future__ import annotations

from typing import cast

import anyio
import pytest

from pivot.engine import engine as engine_mod
from pivot.engine.types import OutputEvent, SinkState, SinkStateChanged, StageStarted


class _ExplodingSink:
    def __init__(self) -> None:
        self.calls: int = 0

    async def handle(self, event: OutputEvent) -> None:
        self.calls += 1
        raise ValueError("boom")

    async def close(self) -> None:
        pass


class _CollectorSink:
    def __init__(self) -> None:
        self.received: list[OutputEvent] = []

    async def handle(self, event: OutputEvent) -> None:
        self.received.append(event)

    async def close(self) -> None:
        pass


class _SignalSink:
    def __init__(self) -> None:
        self.received: list[OutputEvent] = []
        self.enabled_event: anyio.Event = anyio.Event()

    async def handle(self, event: OutputEvent) -> None:
        self.received.append(event)
        if event["type"] == "sink_state_changed" and event["state"] == SinkState.ENABLED:
            self.enabled_event.set()

    async def close(self) -> None:
        pass


def _helper_make_runtime(
    sink: _CollectorSink | _SignalSink,
) -> engine_mod._SinkRuntime:
    send, recv = anyio.create_memory_object_stream[OutputEvent](1)
    return engine_mod._SinkRuntime(
        sink=sink,
        sink_id=type(sink).__name__,
        send=send,
        recv=recv,
    )


def _helper_require_run_id(event: OutputEvent) -> str:
    if "run_id" not in event:
        raise AssertionError("Expected run_id on sink event")
    return event["run_id"]


@pytest.mark.anyio
async def test_sink_disabled_after_consecutive_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "_SINK_FAILURE_THRESHOLD", 2)
    exploding = _ExplodingSink()
    collector = _CollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(collector)
        eng.add_sink(exploding)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(4):
                await eng.emit(StageStarted(type="stage_started", stage=f"s{i}", index=i, total=4))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    assert exploding.calls == 2, "Sink should stop receiving after disable"
    disabled_events = [
        event
        for event in collector.received
        if event["type"] == "sink_state_changed" and event["state"] == SinkState.DISABLED
    ]
    assert len(disabled_events) == 1, "SinkStateChanged(DISABLED) should be emitted"


class _SlowSink:
    def __init__(self) -> None:
        self.calls: int = 0

    async def handle(self, event: OutputEvent) -> None:
        self.calls += 1
        await anyio.sleep(0.05)

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_sink_timeout_disables_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "_SINK_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(engine_mod, "_SINK_HANDLE_TIMEOUT_S", 0.01)

    slow = _SlowSink()
    collector = _CollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(collector)
        eng.add_sink(slow)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(4):
                await eng.emit(StageStarted(type="stage_started", stage=f"t{i}", index=i, total=4))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    disabled_events = [
        event
        for event in collector.received
        if event["type"] == "sink_state_changed" and event["state"] == SinkState.DISABLED
    ]
    assert disabled_events, "Timeout should disable sink"


@pytest.mark.anyio
async def test_sink_not_disabled_before_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "_SINK_FAILURE_THRESHOLD", 3)
    exploding = _ExplodingSink()
    collector = _CollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(collector)
        eng.add_sink(exploding)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(2):
                await eng.emit(StageStarted(type="stage_started", stage=f"s{i}", index=i, total=2))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    disabled_events = [
        event
        for event in collector.received
        if event["type"] == "sink_state_changed" and event["state"] == SinkState.DISABLED
    ]
    assert not disabled_events, "Sink should not disable before reaching threshold"


class _FlakySink:
    def __init__(self, fail_times: int) -> None:
        self._fail_times: int = fail_times
        self.calls: int = 0

    async def handle(self, event: OutputEvent) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("flaky")

    async def close(self) -> None:
        pass


@pytest.mark.anyio
async def test_sink_reenabled_after_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_mod, "_SINK_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(engine_mod, "_SINK_BACKOFF_BASE_S", 0.01)
    monkeypatch.setattr(engine_mod, "_SINK_BACKOFF_MAX_S", 0.05)

    flaky = _FlakySink(fail_times=2)
    collector = _SignalSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(collector)
        eng.add_sink(flaky)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(3):
                await eng.emit(StageStarted(type="stage_started", stage=f"b{i}", index=i, total=3))
            with anyio.fail_after(1.0):
                await collector.enabled_event.wait()
            await eng.emit(StageStarted(type="stage_started", stage="after", index=3, total=4))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    events = [event for event in collector.received if event["type"] == "sink_state_changed"]
    assert any(event["state"] == SinkState.DISABLED for event in events)
    assert any(event["state"] == SinkState.ENABLED for event in events)
    assert flaky.calls >= 3, "Sink should receive events after re-enable"


@pytest.mark.anyio
async def test_emit_sink_state_falls_back_when_output_closed() -> None:
    collector = _CollectorSink()
    runtime = _helper_make_runtime(collector)

    async with engine_mod.Engine() as eng:
        eng._current_run_id = "run-123"
        eng._sink_runtimes = [runtime]
        assert eng._output_send is not None
        await eng._output_send.aclose()

        await eng._emit_sink_state(
            SinkStateChanged(
                type="sink_state_changed",
                sink_id="Collector",
                state=SinkState.DISABLED,
                reason="test",
                failure_count=1,
                backoff_s=0.1,
            )
        )

        received = await runtime.recv.receive()

    sink_event = cast("SinkStateChanged", received)
    assert sink_event["state"] == SinkState.DISABLED
    assert _helper_require_run_id(sink_event) == "run-123"


@pytest.mark.anyio
async def test_emit_sink_state_calls_sink_when_queue_closed() -> None:
    collector = _CollectorSink()
    runtime = _helper_make_runtime(collector)

    async with engine_mod.Engine() as eng:
        eng._current_run_id = "run-456"
        eng._sink_runtimes = [runtime]
        assert eng._output_send is not None
        await eng._output_send.aclose()
        await runtime.send.aclose()

        await eng._emit_sink_state(
            SinkStateChanged(
                type="sink_state_changed",
                sink_id="Collector",
                state=SinkState.DISABLED,
                reason="test",
                failure_count=1,
                backoff_s=0.1,
            )
        )

    assert collector.received, "Expected direct sink handling when queue is closed"
    assert _helper_require_run_id(collector.received[0]) == "run-456"


@pytest.mark.anyio
async def test_reenable_sink_aborts_when_stop_event_set() -> None:
    collector = _CollectorSink()
    runtime = _helper_make_runtime(collector)
    runtime.enabled = False
    runtime.disabled_until = anyio.current_time() + 0.2

    async with engine_mod.Engine() as eng:
        stop_event = anyio.Event()

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._reenable_sink, runtime, stop_event)
            await anyio.sleep(0.01)
            stop_event.set()

        with anyio.move_on_after(0.05) as scope:
            assert eng._output_recv is not None
            await eng._output_recv.receive()
        assert scope.cancel_called, "No sink re-enable event expected"

    assert runtime.enabled is False
    assert runtime.disabled_until is not None
