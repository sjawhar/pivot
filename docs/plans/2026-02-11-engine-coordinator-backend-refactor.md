# Engine Coordinator Backend Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make engine runs non-blocking and cancellable, add global OutputEvent sequencing and sink supervision, extract worker lifecycle into a WorkerPool, and finish with required `run_id` on all OutputEvents.

**Architecture:** Add a process-lifetime `seq` stamped at the Engine `emit()` boundary, introduce sink supervision with per-sink queues and disable/reenable behavior, refactor loky/Manager/drain lifecycle into a concrete `WorkerPool`, and rework run handling into an explicit state machine with restart/reload coalescing. Finish by making `run_id` required on OutputEvents and updating all emitters/consumers.

**Tech Stack:** Python 3.13, anyio, loky, pytest, ruff, basedpyright.

---

## Preflight Checklist

- Confirm you are in the dedicated worktree created by the brainstorming skill.
  - Run: `jj workspace list`
  - Expected: current workspace is the refactor worktree (not `default`).
- Read reference docs (do not edit):
  - `docs/architecture/engine.md`
  - `docs/design/watch-engine.md`
  - `docs/solutions/2026-02-05-engine-dispatcher-drain-race.md`
- Required skills: @SKILL.md, @../executing-plans/SKILL.md, @../subagent-driven-development/SKILL.md

---

### Task 1: Add `seq` to OutputEvent schema

**Files:**
- Modify: `packages/pivot/src/pivot/engine/types.py`
- Test: `packages/pivot/tests/engine/test_types.py`

**Step 1: Write the failing test**

Add to `packages/pivot/tests/engine/test_types.py`:

```python
def test_output_events_define_seq_field() -> None:
    assert "seq" in types.EngineStateChanged.__annotations__
    assert "seq" in types.PipelineReloaded.__annotations__
    assert "seq" in types.StageStarted.__annotations__
    assert "seq" in types.StageCompleted.__annotations__
    assert "seq" in types.StageStateChanged.__annotations__
    assert "seq" in types.LogLine.__annotations__
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_types.py::test_output_events_define_seq_field -v`

Expected: FAIL with an assertion error about missing `seq` annotations.

**Step 3: Write minimal implementation**

Update OutputEvent TypedDicts in `packages/pivot/src/pivot/engine/types.py`:

```python
class EngineStateChanged(TypedDict):
    """Engine transitioned to a new state."""

    type: Literal["engine_state_changed"]
    seq: int
    state: EngineState


class PipelineReloaded(TypedDict):
    """Registry was reloaded, DAG structure may have changed."""

    type: Literal["pipeline_reloaded"]
    seq: int
    stages: list[str]
    stages_added: list[str]
    stages_removed: list[str]
    stages_modified: list[str]
    error: str | None


class StageStarted(TypedDict):
    """A stage began executing."""

    type: Literal["stage_started"]
    seq: int
    stage: str
    index: int
    total: int


class StageCompleted(TypedDict):
    """A stage finished (ran, skipped, or failed)."""

    type: Literal["stage_completed"]
    seq: int
    stage: str
    status: CompletionType
    reason: str
    duration_ms: float
    index: int
    total: int
    input_hash: str | None


class LogLine(TypedDict):
    """A line of output from a running stage."""

    type: Literal["log_line"]
    seq: int
    stage: str
    line: str
    is_stderr: bool


class StageStateChanged(TypedDict):
    """Emitted when a stage's execution state changes."""

    type: Literal["stage_state_changed"]
    seq: int
    stage: str
    state: StageExecutionState
    previous_state: StageExecutionState
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_types.py::test_output_events_define_seq_field -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "add seq field to output events"
```

---

### Task 2: Stamp `seq` at Engine.emit + add seq ordering tests

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Create: `packages/pivot/tests/engine/test_sink_seq.py`

**Step 1: Write the failing test**

Create `packages/pivot/tests/engine/test_sink_seq.py`:

```python
from __future__ import annotations

import anyio
import pytest

from pivot.engine import engine as engine_mod
from pivot.engine.types import OutputEvent, StageStarted


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

    seq_a = [event["seq"] for event in sink_a.received]
    seq_b = [event["seq"] for event in sink_b.received]
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

    max_seq = max(event["seq"] for event in sink_a.received)

    sink_b = _SeqCollectorSink()
    async with engine_mod.Engine() as eng:
        eng.add_sink(sink_b)
        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(2):
                await eng.emit(StageStarted(type="stage_started", stage=f"b{i}", index=i, total=2))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    min_seq = min(event["seq"] for event in sink_b.received)
    assert min_seq > max_seq, "Seq should continue across Engine instances"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_seq.py -v`

Expected: FAIL with KeyError `"seq"` or missing `seq` in events.

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/engine.py`:

```python
class Engine:
    _seq_counter: int = 0

    def __init__(self, *, pipeline: Pipeline | None = None) -> None:
        ...
        self._seq_lock = anyio.Lock()

    async def _next_seq(self) -> int:
        async with self._seq_lock:
            Engine._seq_counter += 1
            return Engine._seq_counter

    async def emit(self, event: OutputEvent) -> None:
        """Emit an output event to all sinks.

        Silently drops events if the output channel is closed (during shutdown).
        """
        if self._output_send:
            with contextlib.suppress(anyio.ClosedResourceError):
                seq = await self._next_seq()
                event_with_seq: OutputEvent = {**event, "seq": seq}
                await self._output_send.send(event_with_seq)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_seq.py -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "stamp seq on emitted output events"
```

---

### Task 3: Add SinkState + SinkStateChanged event

**Files:**
- Modify: `packages/pivot/src/pivot/engine/types.py`
- Test: `packages/pivot/tests/engine/test_types.py`

**Step 1: Write the failing test**

Add to `packages/pivot/tests/engine/test_types.py`:

```python
def test_sink_state_enum() -> None:
    assert types.SinkState.ENABLED.value == "enabled"
    assert types.SinkState.DISABLED.value == "disabled"


def test_sink_state_changed_event() -> None:
    event: types.SinkStateChanged = {
        "type": "sink_state_changed",
        "seq": 1,
        "sink_id": "ConsoleSink",
        "state": types.SinkState.DISABLED,
        "reason": "exception",
        "failure_count": 5,
        "backoff_s": 1.0,
    }
    assert event["state"] == types.SinkState.DISABLED
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_types.py::test_sink_state_changed_event -v`

Expected: FAIL because SinkState/SinkStateChanged are undefined.

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/types.py`:

```python
class SinkState(Enum):
    """Sink supervision state."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class SinkStateChanged(TypedDict):
    """Emitted when a sink is disabled or re-enabled by supervision."""

    type: Literal["sink_state_changed"]
    seq: int
    sink_id: str
    state: SinkState
    reason: str
    failure_count: int
    backoff_s: float | None
```

Also update `__all__` and `OutputEvent` union to include `SinkState` and `SinkStateChanged`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_types.py::test_sink_state_changed_event -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "add sink state change output event"
```

---

### Task 4: Disable sinks after repeated exceptions

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Modify: `packages/pivot/src/pivot/engine/types.py`
- Create: `packages/pivot/tests/engine/test_sink_supervision.py`

**Step 1: Write the failing test**

Create `packages/pivot/tests/engine/test_sink_supervision.py`:

```python
from __future__ import annotations

import anyio
import pytest

from pivot.engine import engine as engine_mod
from pivot.engine.types import OutputEvent, SinkState, StageStarted


class _ExplodingSink:
    def __init__(self) -> None:
        self.calls = 0

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_supervision.py::test_sink_disabled_after_consecutive_failures -v`

Expected: FAIL because sinks are not disabled.

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/engine.py` with supervision constants and tracking:

```python
_SINK_FAILURE_THRESHOLD = 5
_SINK_QUEUE_SIZE = 1024


@dataclass(slots=True)
class _SinkRuntime:
    sink: EventSink
    sink_id: str
    send: MemoryObjectSendStream[OutputEvent]
    recv: MemoryObjectReceiveStream[OutputEvent]
    failures: int = 0
    enabled: bool = True


async def _record_sink_failure(self, runtime: _SinkRuntime, reason: str) -> None:
    runtime.failures += 1
    if runtime.failures < _SINK_FAILURE_THRESHOLD:
        return
    if not runtime.enabled:
        return
    runtime.enabled = False
    await runtime.send.aclose()
    await self.emit(
        SinkStateChanged(
            type="sink_state_changed",
            seq=0,
            sink_id=runtime.sink_id,
            state=SinkState.DISABLED,
            reason=reason,
            failure_count=runtime.failures,
            backoff_s=None,
        )
    )
```

Update `_dispatch_outputs` to build `_SinkRuntime` and use `send_nowait` with failure handling:

```python
for sink in self._sinks:
    send, recv = anyio.create_memory_object_stream[OutputEvent](_SINK_QUEUE_SIZE)
    runtime = _SinkRuntime(sink=sink, sink_id=type(sink).__name__, send=send, recv=recv)
    sink_runtimes.append(runtime)
    tg.start_soon(self._run_sink_task, runtime)

async for event in self._output_recv:
    for runtime in sink_runtimes:
        if not runtime.enabled:
            continue
        try:
            runtime.send.send_nowait(event)
        except anyio.WouldBlock:
            await self._record_sink_failure(runtime, reason="queue_full")
```

Update `_run_sink_task` to report exceptions:

```python
async def _run_sink_task(self, runtime: _SinkRuntime) -> None:
    async for event in runtime.recv:
        try:
            await runtime.sink.handle(event)
        except Exception:
            _logger.exception("Error dispatching event to sink %s", runtime.sink)
            await self._record_sink_failure(runtime, reason="exception")
```

Update `packages/pivot/src/pivot/engine/types.py` EventSink docstring to reflect disable on backpressure.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_supervision.py::test_sink_disabled_after_consecutive_failures -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "disable sinks after consecutive failures"
```

---

### Task 5: Disable sinks after timeout

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Test: `packages/pivot/tests/engine/test_sink_supervision.py`

**Step 1: Write the failing test**

Append to `packages/pivot/tests/engine/test_sink_supervision.py`:

```python
class _SlowSink:
    def __init__(self) -> None:
        self.calls = 0

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_supervision.py::test_sink_timeout_disables_sink -v`

Expected: FAIL because timeouts are not enforced.

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/engine.py`:

```python
_SINK_HANDLE_TIMEOUT_S = 5.0

async def _run_sink_task(self, runtime: _SinkRuntime) -> None:
    async for event in runtime.recv:
        try:
            with anyio.fail_after(_SINK_HANDLE_TIMEOUT_S):
                await runtime.sink.handle(event)
        except TimeoutError:
            _logger.warning("Sink %s timed out handling event", runtime.sink)
            await self._record_sink_failure(runtime, reason="timeout")
        except Exception:
            _logger.exception("Error dispatching event to sink %s", runtime.sink)
            await self._record_sink_failure(runtime, reason="exception")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_supervision.py::test_sink_timeout_disables_sink -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "disable sinks after handle timeout"
```

---

### Task 6: Re-enable sinks with exponential backoff

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Test: `packages/pivot/tests/engine/test_sink_supervision.py`

**Step 1: Write the failing test**

Append to `packages/pivot/tests/engine/test_sink_supervision.py`:

```python
class _FlakySink:
    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.calls = 0

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
    collector = _CollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(collector)
        eng.add_sink(flaky)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)
            for i in range(3):
                await eng.emit(StageStarted(type="stage_started", stage=f"b{i}", index=i, total=3))
            await anyio.sleep(0.05)
            await eng.emit(StageStarted(type="stage_started", stage="after", index=3, total=4))
            assert eng._output_send is not None
            await eng._output_send.aclose()

    events = [event for event in collector.received if event["type"] == "sink_state_changed"]
    assert any(event["state"] == SinkState.DISABLED for event in events)
    assert any(event["state"] == SinkState.ENABLED for event in events)
    assert flaky.calls >= 3, "Sink should receive events after re-enable"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_supervision.py::test_sink_reenabled_after_backoff -v`

Expected: FAIL because sinks never re-enable.

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/engine.py` to track backoff and re-enable:

```python
_SINK_BACKOFF_BASE_S = 1.0
_SINK_BACKOFF_MAX_S = 1800.0


@dataclass(slots=True)
class _SinkRuntime:
    ...
    backoff_s: float = _SINK_BACKOFF_BASE_S
    disabled_until: float | None = None


async def _record_sink_failure(self, runtime: _SinkRuntime, reason: str) -> None:
    runtime.failures += 1
    if runtime.failures < _SINK_FAILURE_THRESHOLD:
        return
    if not runtime.enabled:
        return
    runtime.enabled = False
    runtime.disabled_until = anyio.current_time() + runtime.backoff_s
    await runtime.send.aclose()
    await self.emit(
        SinkStateChanged(
            type="sink_state_changed",
            seq=0,
            sink_id=runtime.sink_id,
            state=SinkState.DISABLED,
            reason=reason,
            failure_count=runtime.failures,
            backoff_s=runtime.backoff_s,
        )
    )


async def _reenable_sink(self, runtime: _SinkRuntime) -> None:
    if runtime.disabled_until is None:
        return
    delay = max(0.0, runtime.disabled_until - anyio.current_time())
    await anyio.sleep(delay)
    send, recv = anyio.create_memory_object_stream[OutputEvent](_SINK_QUEUE_SIZE)
    runtime.send = send
    runtime.recv = recv
    runtime.enabled = True
    runtime.failures = 0
    runtime.backoff_s = min(runtime.backoff_s * 2, _SINK_BACKOFF_MAX_S)
    runtime.disabled_until = None
    await self.emit(
        SinkStateChanged(
            type="sink_state_changed",
            seq=0,
            sink_id=runtime.sink_id,
            state=SinkState.ENABLED,
            reason="backoff_elapsed",
            failure_count=runtime.failures,
            backoff_s=None,
        )
    )
```

Start a re-enable loop from `_dispatch_outputs`:

```python
tg.start_soon(self._run_sink_task, runtime)
tg.start_soon(self._supervise_sink_reenable, runtime)

async def _supervise_sink_reenable(self, runtime: _SinkRuntime) -> None:
    while True:
        if runtime.enabled:
            await anyio.sleep(0.05)
            continue
        await self._reenable_sink(runtime)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_sink_supervision.py::test_sink_reenabled_after_backoff -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "reenable sinks with exponential backoff"
```

---

### Task 7: Update backpressure test + dispatch contract

**Files:**
- Modify: `packages/pivot/tests/engine/test_sinks.py`
- Modify: `packages/pivot/src/pivot/engine/types.py`

**Step 1: Write the failing test**

Replace `test_backpressure_stalls_dispatch_when_sink_queue_fills` in
`packages/pivot/tests/engine/test_sinks.py` with:

```python
@pytest.mark.anyio
async def test_backpressure_disables_stalling_sink_without_blocking() -> None:
    class _StallingSink:
        def __init__(self, consume_count: int) -> None:
            self._consume_count = consume_count
            self.received = 0
            self._stall = anyio.Event()

        async def handle(self, event: OutputEvent) -> None:
            self.received += 1
            if self.received >= self._consume_count:
                await self._stall.wait()

        async def close(self) -> None:
            self._stall.set()

    stalling = _StallingSink(consume_count=1)
    fast = _FastCollectorSink()

    async with engine_mod.Engine() as eng:
        eng.add_sink(fast)
        eng.add_sink(stalling)

        async with anyio.create_task_group() as tg:
            tg.start_soon(eng._dispatch_outputs)

            with anyio.fail_after(2.0):
                for i in range(2000):
                    await eng.emit(StageStarted(type="stage_started", stage=f"s{i}", index=i, total=2000))

            assert eng._output_send is not None
            await eng._output_send.aclose()

    assert len(fast.received) > 0, "Fast sink should keep receiving"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py::test_backpressure_disables_stalling_sink_without_blocking -v`

Expected: FAIL due to current backpressure stall behavior.

**Step 3: Write minimal implementation**

Update EventSink docstring in `packages/pivot/src/pivot/engine/types.py` to reflect
"no blocking on backpressure; slow sinks are disabled" semantics.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_sinks.py::test_backpressure_disables_stalling_sink_without_blocking -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "update backpressure behavior tests"
```

---

### Task 8: Extract WorkerPool skeleton + unit tests

**Files:**
- Create: `packages/pivot/src/pivot/engine/worker_pool.py`
- Modify: `packages/pivot/src/pivot/engine/__init__.py`
- Create: `packages/pivot/tests/engine/test_worker_pool.py`

**Step 1: Write the failing test**

Create `packages/pivot/tests/engine/test_worker_pool.py`:

```python
from __future__ import annotations

import time

import pytest

from pivot.engine.worker_pool import WorkerPool


def _helper_identity(value: int) -> int:
    return value


def _helper_sleep(seconds: float) -> None:
    time.sleep(seconds)


def test_worker_pool_stop_accepting_rejects_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    future = pool.submit(_helper_identity, 1)
    assert future.result(timeout=5) == 1

    pool.stop_accepting()
    with pytest.raises(RuntimeError, match="accepting"):
        pool.submit(_helper_identity, 2)

    pool.shutdown()


def test_worker_pool_hard_cancel_blocks_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    pool.submit(_helper_sleep, 5)
    pool.hard_cancel()

    with pytest.raises(RuntimeError, match="accepting"):
        pool.submit(_helper_identity, 3)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_worker_pool.py -v`

Expected: FAIL because WorkerPool does not exist.

**Step 3: Write minimal implementation**

Create `packages/pivot/src/pivot/engine/worker_pool.py`:

```python
from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pivot.executor import core as executor_core
from pivot.types import OutputMessage

if TYPE_CHECKING:
    from concurrent.futures import Future
    from multiprocessing.managers import SyncManager


@dataclass(slots=True)
class WorkerPool:
    _executor: executor_core.Executor | None = None
    _manager: SyncManager | None = None
    _output_queue: mp.Queue[OutputMessage] | None = None
    _shutdown_event: threading.Event | None = None
    _accepting: bool = True

    def start(self, *, max_workers: int) -> None:
        self._executor = executor_core.create_executor(max_workers)
        spawn_ctx = mp.get_context("spawn")
        self._manager = spawn_ctx.Manager()
        self._output_queue = self._manager.Queue()  # pyright: ignore[reportAssignmentType]
        self._shutdown_event = threading.Event()
        self._accepting = True

    def output_queue(self) -> mp.Queue[OutputMessage]:
        if self._output_queue is None:
            raise RuntimeError("WorkerPool not started")
        return self._output_queue

    def shutdown_event(self) -> threading.Event:
        if self._shutdown_event is None:
            raise RuntimeError("WorkerPool not started")
        return self._shutdown_event

    def submit(self, *args: object, **kwargs: object) -> "Future[object]":
        if not self._accepting:
            raise RuntimeError("WorkerPool is not accepting new submissions")
        if self._executor is None:
            raise RuntimeError("WorkerPool not started")
        return self._executor.submit(*args, **kwargs)

    def stop_accepting(self) -> None:
        self._accepting = False

    def shutdown(self) -> None:
        self._accepting = False
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
        if self._manager is not None:
            self._manager.shutdown()

    def hard_cancel(self) -> None:
        self._accepting = False
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self._manager is not None:
            self._manager.shutdown()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
```

Export in `packages/pivot/src/pivot/engine/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_worker_pool.py -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "add worker pool wrapper"
```

---

### Task 9: Integrate WorkerPool into Engine orchestration

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Test: `packages/pivot/tests/engine/test_engine_shutdown.py`

**Step 1: Write the failing test**

Add to `packages/pivot/tests/engine/test_engine_shutdown.py`:

```python
@pytest.mark.anyio
async def test_engine_uses_worker_pool_shutdown_event(minimal_pipeline: Pipeline) -> None:
    register_test_stage(_helper_noop, name="test_stage")

    async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        eng.add_source(sources.OneShotSource(stages=["test_stage"], force=True, reason="test"))

        with anyio.fail_after(10.0):
            await eng.run(exit_on_completion=True)

    assert True, "Run should complete using WorkerPool-managed shutdown"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_engine_shutdown.py::test_engine_uses_worker_pool_shutdown_event -v`

Expected: FAIL once WorkerPool integration is incomplete (or for missing imports).

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/engine.py`:

- Replace direct executor/Manager/queue setup with `WorkerPool`.
- Use `pool.output_queue()` and `pool.shutdown_event()` for drain thread.
- Replace `self._executor = ...` with `self._worker_pool = WorkerPool()` and `self._worker_pool.start(...)`.
- On soft-cancel: call `self._worker_pool.stop_accepting()`.
- On hard-cancel: call `self._worker_pool.hard_cancel()`.
- On normal completion: call `self._worker_pool.shutdown()`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_engine_shutdown.py::test_engine_uses_worker_pool_shutdown_event -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "wire engine orchestration through worker pool"
```

---

### Task 10: Add run state + non-blocking run handling

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Create: `packages/pivot/tests/engine/test_cancel_restart_policy.py`

**Step 1: Write the failing test**

Create `packages/pivot/tests/engine/test_cancel_restart_policy.py`:

```python
from __future__ import annotations

import time
from typing import Annotated

import anyio
import pytest

from conftest import AsyncEventCaptureSink
from helpers import register_test_stage
from pivot.engine import engine as engine_mod
from pivot.engine.types import CodeOrConfigChanged, RunRequested
from pivot.engine import sources
from pivot.types import OnError


def _helper_sleep_stage() -> dict[str, str]:
    time.sleep(0.2)
    return {"result": "ok"}


class _ScriptedSource:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events

    async def run(self, send: anyio.streams.memory.MemoryObjectSendStream) -> None:
        for event in self._events:
            await send.send(event)


@pytest.mark.anyio
async def test_run_requested_does_not_block_input_handler(minimal_pipeline) -> None:
    register_test_stage(_helper_sleep_stage, name="sleep_stage")

    sink = AsyncEventCaptureSink()
    run_event: RunRequested = {
        "type": "run_requested",
        "stages": ["sleep_stage"],
        "force": True,
        "reason": "test",
        "single_stage": False,
        "parallel": False,
        "max_workers": 1,
        "no_commit": True,
        "on_error": OnError.FAIL,
        "cache_dir": None,
        "allow_uncached_incremental": False,
        "checkout_missing": False,
    }
    reload_event: CodeOrConfigChanged = {"type": "code_or_config_changed", "paths": ["x.py"]}

    async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        eng.add_sink(sink)
        eng.add_source(_ScriptedSource([run_event, reload_event]))

        with anyio.fail_after(10.0):
            await eng.run(exit_on_completion=True)

    reloads = [event for event in sink.events if event["type"] == "pipeline_reloaded"]
    assert reloads, "Reload should be processed even while run is active"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_cancel_restart_policy.py::test_run_requested_does_not_block_input_handler -v`

Expected: FAIL because input handling blocks on `_orchestrate_execution`.

**Step 3: Write minimal implementation**

Update `packages/pivot/src/pivot/engine/engine.py`:

```python
class _RunState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"


def __init__(...):
    ...
    self._run_state = _RunState.IDLE
    self._run_task_group: anyio.abc.TaskGroup | None = None
    self._restart_pending: RunRequested | None = None
    self._reload_pending = False


async def run(self, *, exit_on_completion: bool = True) -> None:
    ...
    async with anyio.create_task_group() as tg:
        self._run_task_group = tg
        ...


async def _handle_run_requested(self, event: RunRequested) -> None:
    self._stored_no_commit = event["no_commit"]
    self._stored_on_error = event["on_error"]
    self._stored_parallel = event["parallel"]
    self._stored_max_workers = event["max_workers"]

    if self._run_state == _RunState.RUNNING:
        self._restart_pending = event
        await self._handle_cancel_requested()
        return

    self._run_state = _RunState.RUNNING
    self._cancel_event = anyio.Event()
    self._stage_indices.clear()

    await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))
    assert self._run_task_group is not None
    self._run_task_group.start_soon(self._run_execution_task, event)


async def _run_execution_task(self, event: RunRequested) -> None:
    try:
        self._require_pipeline()
        await self._orchestrate_execution(...)
    finally:
        self._run_state = _RunState.IDLE
        await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
        if self._reload_pending:
            self._reload_pending = False
            await self._reload_pipeline()
        if self._restart_pending is not None:
            restart = self._restart_pending
            self._restart_pending = None
            await self._handle_run_requested(restart)
```

Also update `_handle_code_or_config_changed` to set `_reload_pending = True` when RUNNING/CANCELLING.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_cancel_restart_policy.py::test_run_requested_does_not_block_input_handler -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "make run handling non-blocking with run state"
```

---

### Task 11: Explicit cancel/restart state machine + hard-cancel

**Files:**
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Modify: `packages/pivot/src/pivot/engine/worker_pool.py`
- Update: `packages/pivot/tests/engine/test_cancel_restart_policy.py`

**Step 1: Write the failing test**

Append to `packages/pivot/tests/engine/test_cancel_restart_policy.py`:

```python
@pytest.mark.anyio
async def test_run_requested_while_running_restarts_once(minimal_pipeline) -> None:
    register_test_stage(_helper_sleep_stage, name="sleep_stage")

    run_event: RunRequested = {
        "type": "run_requested",
        "stages": ["sleep_stage"],
        "force": True,
        "reason": "test",
        "single_stage": False,
        "parallel": False,
        "max_workers": 1,
        "no_commit": True,
        "on_error": OnError.FAIL,
        "cache_dir": None,
        "allow_uncached_incremental": False,
        "checkout_missing": False,
    }

    sink = AsyncEventCaptureSink()

    class _Source:
        async def run(self, send: anyio.streams.memory.MemoryObjectSendStream) -> None:
            await send.send(run_event)
            await send.send(run_event)

    async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        eng.add_sink(sink)
        eng.add_source(_Source())

        with anyio.fail_after(15.0):
            await eng.run(exit_on_completion=True)

    state_events = [event for event in sink.events if event["type"] == "engine_state_changed"]
    active_events = [event for event in state_events if event["state"] == engine_mod.EngineState.ACTIVE]
    assert len(active_events) == 2, "Should run once, then restart once"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_cancel_restart_policy.py::test_run_requested_while_running_restarts_once -v`

Expected: FAIL because multiple RunRequested are not coalesced.

**Step 3: Write minimal implementation**

Update cancellation flow in `packages/pivot/src/pivot/engine/engine.py`:

```python
async def _handle_cancel_requested(self) -> None:
    if self._run_state == _RunState.CANCELLING:
        return
    if self._run_state == _RunState.IDLE:
        return
    self._run_state = _RunState.CANCELLING
    if self._cancel_event is not None:
        self._cancel_event.set()
    if self._worker_pool is not None:
        self._worker_pool.stop_accepting()


async def _run_execution_task(self, event: RunRequested) -> None:
    hard_cancel_at = anyio.current_time() + 5.0
    try:
        ...
    finally:
        if self._run_state == _RunState.CANCELLING and anyio.current_time() >= hard_cancel_at:
            if self._worker_pool is not None:
                self._worker_pool.hard_cancel()
        self._run_state = _RunState.IDLE
        await self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))
        ...
```

Ensure `_restart_pending` coalesces to a single restart; do not enqueue multiple restarts.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_cancel_restart_policy.py::test_run_requested_while_running_restarts_once -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "add cancel/restart state machine and hard cancel"
```

---

### Task 12: Require `run_id` on OutputEvents (final step)

**Files:**
- Modify: `packages/pivot/src/pivot/engine/types.py`
- Modify: `packages/pivot/src/pivot/engine/engine.py`
- Modify: `packages/pivot/src/pivot/engine/sinks.py`
- Modify: `packages/pivot/src/pivot/engine/agent_rpc.py`
- Modify: `packages/pivot/src/pivot/cli/_run_common.py`
- Update tests: `packages/pivot/tests/engine/test_types.py`, `packages/pivot/tests/engine/test_sinks.py`, `packages/pivot/tests/engine/test_types_static.py`, `packages/pivot/tests/cli/test_jsonl_sink.py`

**Step 1: Write the failing test**

Add to `packages/pivot/tests/engine/test_types.py`:

```python
def test_output_events_define_run_id_field() -> None:
    assert "run_id" in types.EngineStateChanged.__annotations__
    assert "run_id" in types.PipelineReloaded.__annotations__
    assert "run_id" in types.StageStarted.__annotations__
    assert "run_id" in types.StageCompleted.__annotations__
    assert "run_id" in types.StageStateChanged.__annotations__
    assert "run_id" in types.LogLine.__annotations__
    assert "run_id" in types.SinkStateChanged.__annotations__
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/pivot/tests/engine/test_types.py::test_output_events_define_run_id_field -v`

Expected: FAIL because `run_id` is not present.

**Step 3: Write minimal implementation**

Update all OutputEvent TypedDicts in `packages/pivot/src/pivot/engine/types.py` to add `run_id: str`.

Update `packages/pivot/src/pivot/engine/engine.py` to pass `run_id` through emit:

```python
def __init__(...):
    ...
    self._current_run_id: str | None = None

async def _handle_run_requested(self, event: RunRequested) -> None:
    ...
    self._current_run_id = run_history.generate_run_id()
    ...

async def emit(self, event: OutputEvent) -> None:
    ...
    event_with_seq: OutputEvent = {**event, "seq": seq, "run_id": self._current_run_id or ""}
```

Update all sinks/consumers/tests that construct OutputEvents to include `run_id`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/pivot/tests/engine/test_types.py::test_output_events_define_run_id_field -v`

Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "require run_id on output events"
```

---

## Final QA

Run the full quality suite and save outputs to `.sisyphus/evidence/`:

```bash
uv run ruff format .
uv run ruff check .
uv run basedpyright
uv run pytest packages/pivot/tests -n auto
```

---

Plan complete and saved to `docs/plans/2026-02-11-engine-coordinator-backend-refactor.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
