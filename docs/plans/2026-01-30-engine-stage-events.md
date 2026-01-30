# Engine Stage Events Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Have Engine emit StageStarted/StageCompleted events to sinks, replacing duplicate display code in CLI.

**Architecture:** Engine wraps the executor's progress_callback to intercept stage events, translates them to engine event types, and emits to registered sinks. CLI creates appropriate sinks (ConsoleSink/JsonlSink/TuiSink) and lets them handle display instead of the executor's direct Console/tui_queue/progress_callback paths.

**Tech Stack:** Python 3.13+, TypedDict, Protocol

---

## Task 1: Add Progress Callback Adapter to Engine.run_once()

**Files:**
- Modify: `src/pivot/engine/engine.py`
- Modify: `tests/engine/test_engine.py`

**Step 1: Write failing test for progress callback interception**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_run_once_emits_stage_started_events() -> None:
    """run_once() emits StageStarted events to sinks."""
    from pivot.types import RunEventType

    class RecordingSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    with patch("pivot.engine.engine.executor") as mock_executor:
        # Simulate executor calling progress_callback with stage events
        def fake_run(**kwargs: object) -> dict[str, object]:
            callback = kwargs.get("progress_callback")
            if callback:
                callback({
                    "type": RunEventType.STAGE_START,
                    "stage": "train",
                    "index": 1,
                    "total": 2,
                    "timestamp": "2026-01-30T00:00:00Z",
                })
            return {"train": {"status": "ran", "reason": ""}}

        mock_executor.run.side_effect = fake_run

        eng = engine.Engine()
        sink = RecordingSink()
        eng.add_sink(sink)
        eng.run_once()

        # Should have StageStarted event (plus state change events)
        stage_events = [e for e in sink.events if e["type"] == "stage_started"]
        assert len(stage_events) == 1
        assert stage_events[0]["stage"] == "train"
        assert stage_events[0]["index"] == 1
        assert stage_events[0]["total"] == 2


def test_engine_run_once_emits_stage_completed_events() -> None:
    """run_once() emits StageCompleted events to sinks."""
    from pivot.types import RunEventType, StageStatus

    class RecordingSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    with patch("pivot.engine.engine.executor") as mock_executor:
        def fake_run(**kwargs: object) -> dict[str, object]:
            callback = kwargs.get("progress_callback")
            if callback:
                callback({
                    "type": RunEventType.STAGE_COMPLETE,
                    "stage": "train",
                    "status": StageStatus.RAN,
                    "reason": "inputs changed",
                    "duration_ms": 1234.5,
                    "timestamp": "2026-01-30T00:00:00Z",
                })
            return {"train": {"status": "ran", "reason": ""}}

        mock_executor.run.side_effect = fake_run

        eng = engine.Engine()
        sink = RecordingSink()
        eng.add_sink(sink)
        eng.run_once()

        stage_events = [e for e in sink.events if e["type"] == "stage_completed"]
        assert len(stage_events) == 1
        assert stage_events[0]["stage"] == "train"
        assert stage_events[0]["status"] == StageStatus.RAN
        assert stage_events[0]["duration_ms"] == 1234.5


def test_engine_run_once_forwards_to_user_progress_callback() -> None:
    """run_once() still forwards events to user's progress_callback."""
    from pivot.types import RunEventType

    user_events = list[object]()

    def user_callback(event: object) -> None:
        user_events.append(event)

    with patch("pivot.engine.engine.executor") as mock_executor:
        def fake_run(**kwargs: object) -> dict[str, object]:
            callback = kwargs.get("progress_callback")
            if callback:
                callback({
                    "type": RunEventType.STAGE_START,
                    "stage": "train",
                    "index": 1,
                    "total": 1,
                    "timestamp": "2026-01-30T00:00:00Z",
                })
            return {}

        mock_executor.run.side_effect = fake_run

        eng = engine.Engine()
        eng.run_once(progress_callback=user_callback)

        # User callback should still receive the event
        assert len(user_events) == 1
        assert user_events[0]["type"] == RunEventType.STAGE_START
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_run_once_emits_stage_started_events -v
```

Expected: FAIL (no stage events emitted)

**Step 3: Write minimal implementation**

Update `src/pivot/engine/engine.py`:

Add imports at top:
```python
from pivot.engine.types import (
    EngineState,
    EngineStateChanged,
    EventSink,
    EventSource,
    InputEvent,
    OutputEvent,
    RunRequested,
    StageCompleted,
    StageStarted,
)
from pivot.types import OnError, RunEventType
```

Update run_once() to wrap progress_callback:
```python
    def run_once(
        self,
        stages: list[str] | None = None,
        force: bool = False,
        single_stage: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        no_commit: bool = False,
        no_cache: bool = False,
        show_output: bool = True,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
        on_error: OnError = OnError.FAIL,
        cache_dir: pathlib.Path | None = None,
        tui_queue: TuiQueue | None = None,
        output_queue: mp.Queue[OutputMessage] | None = None,
        progress_callback: Callable[[RunJsonEvent], None] | None = None,
        explain_mode: bool = False,
    ) -> dict[str, ExecutionSummary]:
        """Execute stages once and return."""
        # Emit state transition: IDLE -> ACTIVE
        self._state = EngineState.ACTIVE
        self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.ACTIVE))

        # Create progress callback adapter that emits to sinks
        def progress_adapter(event: RunJsonEvent) -> None:
            self._handle_progress_event(event)
            # Forward to user callback if provided
            if progress_callback is not None:
                progress_callback(event)

        try:
            # Delegate to existing executor
            result = executor.run(
                stages=stages,
                force=force,
                single_stage=single_stage,
                parallel=parallel,
                max_workers=max_workers,
                no_commit=no_commit,
                no_cache=no_cache,
                show_output=show_output,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
                on_error=on_error,
                cache_dir=cache_dir,
                tui_queue=tui_queue,
                output_queue=output_queue,
                progress_callback=progress_adapter,
                explain_mode=explain_mode,
                cancel_event=self._cancel_event,
            )
            return result
        finally:
            # Emit state transition: ACTIVE -> IDLE
            self._state = EngineState.IDLE
            self.emit(EngineStateChanged(type="engine_state_changed", state=EngineState.IDLE))

    def _handle_progress_event(self, event: RunJsonEvent) -> None:
        """Translate executor progress event to engine event and emit to sinks."""
        match event["type"]:
            case RunEventType.STAGE_START:
                self.emit(
                    StageStarted(
                        type="stage_started",
                        stage=event["stage"],
                        index=event["index"],
                        total=event["total"],
                    )
                )
            case RunEventType.STAGE_COMPLETE:
                self.emit(
                    StageCompleted(
                        type="stage_completed",
                        stage=event["stage"],
                        status=event["status"],
                        reason=event["reason"],
                        duration_ms=event["duration_ms"],
                        index=event["index"],
                        total=event["total"],
                    )
                )
            case _:
                pass  # Ignore schema_version and execution_result events
```

Add the `index` and `total` fields to StageCompleteEvent handling. Note: executor's StageCompleteEvent doesn't have index/total, so we need to track them.

Actually, looking at the executor code, StageCompleteEvent doesn't include index/total. We need to track the last StageStartEvent to get these values. Update the implementation:

```python
    def __init__(self) -> None:
        """Initialize the engine in IDLE state."""
        self._state: EngineState = EngineState.IDLE
        self._sinks: list[EventSink] = list[EventSink]()
        self._sources: list[EventSource] = list[EventSource]()
        self._event_queue: queue.Queue[InputEvent] = queue.Queue()
        self._shutdown_event: threading.Event = threading.Event()
        self._cancel_event: threading.Event = threading.Event()
        self._graph: nx.DiGraph[str] | None = None
        # Track stage index/total from start events for completion events
        self._stage_indices: dict[str, tuple[int, int]] = {}

    def _handle_progress_event(self, event: RunJsonEvent) -> None:
        """Translate executor progress event to engine event and emit to sinks."""
        match event["type"]:
            case RunEventType.STAGE_START:
                # Store index/total for this stage
                self._stage_indices[event["stage"]] = (event["index"], event["total"])
                self.emit(
                    StageStarted(
                        type="stage_started",
                        stage=event["stage"],
                        index=event["index"],
                        total=event["total"],
                    )
                )
            case RunEventType.STAGE_COMPLETE:
                # Get stored index/total from start event
                index, total = self._stage_indices.get(event["stage"], (0, 0))
                self.emit(
                    StageCompleted(
                        type="stage_completed",
                        stage=event["stage"],
                        status=event["status"],
                        reason=event["reason"],
                        duration_ms=event["duration_ms"],
                        index=index,
                        total=total,
                    )
                )
            case _:
                pass  # Ignore schema_version and execution_result events
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/engine/test_engine.py -v
```

Expected: PASS (all tests)

**Step 5: Run quality checks**

```bash
uv run ruff format src/pivot/engine && uv run ruff check src/pivot/engine && uv run basedpyright src/pivot/engine
```

**Step 6: Commit**

```bash
jj describe -m "feat(engine): emit stage events via progress callback adapter

Engine intercepts executor progress events, translates to engine events,
and emits to registered sinks. User callbacks still receive events."
```

---

## Task 2: Update CLI Plain Mode to Use ConsoleSink

**Files:**
- Modify: `src/pivot/cli/run.py`
- Modify: `tests/cli/test_run.py` (if needed)

**Step 1: Update plain mode to create ConsoleSink**

In `src/pivot/cli/run.py`, update the plain mode execution block (around line 555-574):

Current code:
```python
    else:
        from pivot.engine import engine as engine_mod

        eng = engine_mod.Engine()
        try:
            results = eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                explain_mode=explain,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                show_output=not quiet,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
```

Replace with:
```python
    else:
        from pivot.engine import engine as engine_mod
        from pivot.engine import sinks as engine_sinks
        from pivot.tui import console as tui_console

        eng = engine_mod.Engine()

        # Add ConsoleSink for stage progress display (unless quiet)
        console: tui_console.Console | None = None
        if not quiet:
            console = tui_console.Console()
            eng.add_sink(engine_sinks.ConsoleSink(console))

        try:
            results = eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                explain_mode=explain,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                show_output=not quiet,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
```

**Step 2: Run CLI tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "not tui and not json and not watch"
```

Expected: PASS

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/cli && uv run ruff check src/pivot/cli && uv run basedpyright src/pivot/cli
```

**Step 4: Commit**

```bash
jj describe -m "feat(cli): use ConsoleSink for plain mode stage display

Stage progress flows through Engine -> ConsoleSink -> Console."
```

---

## Task 3: Update CLI JSON Mode to Use JsonlSink

**Files:**
- Modify: `src/pivot/cli/run.py`

**Step 1: Update JSON mode to use JsonlSink**

In `src/pivot/cli/run.py`, update the JSON mode block (around line 511-554):

Current code uses `progress_callback=cli_helpers.emit_jsonl`. Update to:

```python
    elif as_json:
        from pivot.engine import engine as engine_mod
        from pivot.engine import sinks as engine_sinks

        # JSONL streaming mode
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

        start_time = time.perf_counter()
        eng = engine_mod.Engine()

        # Add JsonlSink for stage event streaming
        eng.add_sink(engine_sinks.JsonlSink(callback=cli_helpers.emit_jsonl))

        try:
            results = eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=cache_dir,
                explain_mode=False,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                show_output=False,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
        # ... rest of execution result handling ...
```

Note: Remove `progress_callback=cli_helpers.emit_jsonl` from run_once() since JsonlSink now handles it.

**Step 2: Run JSON output tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "json"
```

Expected: PASS

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/cli && uv run ruff check src/pivot/cli && uv run basedpyright src/pivot/cli
```

**Step 4: Commit**

```bash
jj describe -m "feat(cli): use JsonlSink for JSON mode stage events

Stage events flow through Engine -> JsonlSink -> emit_jsonl."
```

---

## Task 4: Update CLI TUI Mode to Use TuiSink

**Files:**
- Modify: `src/pivot/cli/run.py`

**Step 1: Update TUI mode to use TuiSink**

In `src/pivot/cli/run.py`, update `_run_with_tui()` function (around line 144-215):

Update the executor_func inside to use TuiSink:

```python
    from pivot.engine import engine as engine_mod
    from pivot.engine import sinks as engine_sinks

    # Generate run_id for TUI tracking
    import uuid
    run_id = str(uuid.uuid4())[:8]

    # Create executor function using Engine with TuiSink
    def executor_func() -> dict[str, ExecutionSummary]:
        eng = engine_mod.Engine()
        # Share TUI's cancel_event so TUI can signal cancellation to Engine
        eng._cancel_event = cancel_event  # pyright: ignore[reportPrivateUsage]

        # Add TuiSink for stage events
        eng.add_sink(engine_sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))

        try:
            return eng.run_once(
                stages=stages_list,
                single_stage=single_stage,
                cache_dir=resolved_cache_dir,
                show_output=False,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
        finally:
            eng.close()
```

Note: Remove `tui_queue=tui_queue` from run_once() since TuiSink now handles it.

**Step 2: Run TUI tests**

```bash
uv run pytest tests/cli/test_run.py -v -k "tui"
uv run pytest tests/tui/ -v
```

Expected: PASS

**Step 3: Run quality checks**

```bash
uv run ruff format src/pivot/cli && uv run ruff check src/pivot/cli && uv run basedpyright src/pivot/cli
```

**Step 4: Commit**

```bash
jj describe -m "feat(cli): use TuiSink for TUI mode stage events

Stage events flow through Engine -> TuiSink -> tui_queue -> TUI."
```

---

## Task 5: Remove Duplicate Display Code from Executor

**Files:**
- Modify: `src/pivot/executor/core.py`
- Modify: `tests/executor/test_executor.py` (if needed)

**Step 1: Analyze executor display paths**

The executor currently has three parallel notification paths in `StageLifecycle`:
1. `self.console.stage_start/stage_result` - direct console output
2. `self.tui_queue.put()` - TUI messages
3. `self.progress_callback()` - JSONL events

Since Engine now handles stage events via sinks:
- Plain mode: ConsoleSink receives events, calls Console
- TUI mode: TuiSink receives events, puts to tui_queue
- JSON mode: JsonlSink receives events, calls emit_jsonl

We can remove the direct console and tui_queue paths from executor, keeping only progress_callback (which Engine wraps).

**Step 2: Update StageLifecycle to remove duplicate paths**

In `src/pivot/executor/core.py`, update `StageLifecycle`:

Remove `console` and `tui_queue` parameters from `__init__` and the class:
```python
class StageLifecycle:
    """Manages stage execution state and notifications."""

    total_stages: int
    progress_callback: Callable[[RunJsonEvent], None] | None
    run_id: str

    def __init__(
        self,
        total_stages: int,
        progress_callback: Callable[[RunJsonEvent], None] | None = None,
        run_id: str = "",
    ) -> None:
        self.total_stages = total_stages
        self.progress_callback = progress_callback
        self.run_id = run_id
```

Update `mark_started` to only call progress_callback:
```python
    def mark_started(self, state: StageState) -> None:
        """Mark stage as started and send progress notification."""
        state.status = StageStatus.IN_PROGRESS
        state.start_time = time.perf_counter()
        logger.debug(f"Stage started: {state.name}")  # noqa: G004

        if self.progress_callback:
            self.progress_callback(
                StageStartEvent(
                    type=RunEventType.STAGE_START,
                    stage=state.name,
                    index=state.index,
                    total=self.total_stages,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )
```

Update `_notify_complete` to only call progress_callback:
```python
    def _notify_complete(self, state: StageState, result: StageResult) -> None:
        """Send completion notification."""
        result_status = result["status"]
        result_reason = result["reason"]
        duration = state.get_duration()
        logger.debug(f"Stage completed: {state.name} -> {result_status}")  # noqa: G004

        if self.progress_callback:
            self.progress_callback(
                StageCompleteEvent(
                    type=RunEventType.STAGE_COMPLETE,
                    stage=state.name,
                    status=result_status,
                    reason=result_reason,
                    duration_ms=(duration or 0) * 1000,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )
```

**Step 3: Update run() function to remove console/tui_queue parameters**

In `src/pivot/executor/core.py`, update the `run()` function signature and `StageLifecycle` instantiation:

Remove `console`, `tui_queue` parameters from run().

Update StageLifecycle creation:
```python
    lifecycle = StageLifecycle(
        total_stages=len(stages_to_run),
        progress_callback=progress_callback,
        run_id=run_id,
    )
```

**Step 4: Update Engine.run_once() to stop passing tui_queue**

In `src/pivot/engine/engine.py`, remove `tui_queue` from the executor.run() call since it's no longer used.

**Step 5: Run all tests**

```bash
uv run pytest tests/ -n auto
```

Expected: PASS (all tests)

**Step 6: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 7: Commit**

```bash
jj describe -m "refactor(executor): remove duplicate display code

Stage events now flow only through progress_callback. Console and
tui_queue display handled by Engine sinks (ConsoleSink, TuiSink)."
```

---

## Task 6: Final Integration Test

**Files:**
- Modify: `tests/engine/test_engine.py`

**Step 1: Write end-to-end integration test**

Add to `tests/engine/test_engine.py`:
```python
def test_engine_full_event_flow_integration() -> None:
    """Engine emits complete event flow: state changes + stage events."""
    from pivot.types import RunEventType, StageStatus

    class RecordingSink:
        def __init__(self) -> None:
            self.events = list[types.OutputEvent]()

        def handle(self, event: types.OutputEvent) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    with patch("pivot.engine.engine.executor") as mock_executor:
        def fake_run(**kwargs: object) -> dict[str, object]:
            callback = kwargs.get("progress_callback")
            if callback:
                # Simulate full execution flow
                callback({
                    "type": RunEventType.STAGE_START,
                    "stage": "preprocess",
                    "index": 1,
                    "total": 2,
                    "timestamp": "2026-01-30T00:00:00Z",
                })
                callback({
                    "type": RunEventType.STAGE_COMPLETE,
                    "stage": "preprocess",
                    "status": StageStatus.RAN,
                    "reason": "inputs changed",
                    "duration_ms": 100.0,
                    "timestamp": "2026-01-30T00:00:01Z",
                })
                callback({
                    "type": RunEventType.STAGE_START,
                    "stage": "train",
                    "index": 2,
                    "total": 2,
                    "timestamp": "2026-01-30T00:00:02Z",
                })
                callback({
                    "type": RunEventType.STAGE_COMPLETE,
                    "stage": "train",
                    "status": StageStatus.SKIPPED,
                    "reason": "unchanged",
                    "duration_ms": 0.0,
                    "timestamp": "2026-01-30T00:00:02Z",
                })
            return {
                "preprocess": {"status": "ran", "reason": ""},
                "train": {"status": "skipped", "reason": "unchanged"},
            }

        mock_executor.run.side_effect = fake_run

        eng = engine.Engine()
        sink = RecordingSink()
        eng.add_sink(sink)
        eng.run_once()

        # Verify event sequence
        event_types = [e["type"] for e in sink.events]
        assert event_types == [
            "engine_state_changed",  # ACTIVE
            "stage_started",         # preprocess
            "stage_completed",       # preprocess
            "stage_started",         # train
            "stage_completed",       # train
            "engine_state_changed",  # IDLE
        ]

        # Verify stage details
        stage_started = [e for e in sink.events if e["type"] == "stage_started"]
        assert stage_started[0]["stage"] == "preprocess"
        assert stage_started[0]["index"] == 1
        assert stage_started[1]["stage"] == "train"
        assert stage_started[1]["index"] == 2

        stage_completed = [e for e in sink.events if e["type"] == "stage_completed"]
        assert stage_completed[0]["status"] == StageStatus.RAN
        assert stage_completed[1]["status"] == StageStatus.SKIPPED
```

**Step 2: Run test**

```bash
uv run pytest tests/engine/test_engine.py::test_engine_full_event_flow_integration -v
```

Expected: PASS

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

Expected: PASS (all tests)

**Step 4: Run all quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 5: Final commit**

```bash
jj describe -m "test(engine): add full event flow integration test

Verifies Engine emits state changes + stage events in correct sequence."
```

---

## Summary

After completing this plan, you will have:

1. **Engine emits stage events** - `run_once()` intercepts executor progress events and emits `StageStarted`/`StageCompleted` to sinks

2. **CLI uses sinks for display**:
   - Plain mode: `ConsoleSink` → `Console`
   - JSON mode: `JsonlSink` → `emit_jsonl`
   - TUI mode: `TuiSink` → `tui_queue`

3. **Executor simplified** - `StageLifecycle` only calls `progress_callback`, no direct console/tui_queue

4. **Single event flow** - All stage display goes through Engine sinks, eliminating duplicate paths

**Event flow after implementation:**
```
executor.run()
    └─ progress_callback()
        └─ Engine._handle_progress_event()
            └─ Engine.emit()
                ├─ ConsoleSink.handle() → Console.stage_start/result()
                ├─ TuiSink.handle() → tui_queue.put()
                └─ JsonlSink.handle() → emit_jsonl()
```
