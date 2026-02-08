# Logging Transport: Replace Manager Queue + Polling Drain

**Goal:** Replace `multiprocessing.Manager().Queue()` with a plain spawn-context `multiprocessing.Queue` and replace the polling drain loop with a dedicated blocking drain thread that forwards messages into anyio.

**Architecture:** The engine currently uses `Manager().Queue()` (which spawns a separate manager process) and polls it with 20ms timeouts. We replace the Manager queue with `mp.get_context("spawn").Queue()` (no manager process needed), replace the polling drain task with a blocking drain thread that calls `queue.get()` with no timeout, and use a sentinel value to signal clean shutdown. The drain thread forwards messages into the async event loop via `anyio.from_thread.run`. Note: Manager is still used elsewhere for queue pickling across spawn boundaries in ProcessPoolExecutor.

**Tech Stack:** Python multiprocessing, anyio, threading (via `anyio.to_thread`)

---

## Summary of Changes

**Engine side** (`src/pivot/engine/engine.py`):
1. Replace `Manager().Queue()` with `spawn_ctx.Queue()`
2. Replace `_drain_output_queue` (polling loop) with a blocking drain thread
3. Use a sentinel (`None`) to signal the drain thread to stop
4. Remove manager shutdown cleanup
5. Remove `_get_from_queue` helper and `_OUTPUT_QUEUE_DRAIN_TIMEOUT` constant

**Worker side** (`src/pivot/executor/worker.py`):
- No changes needed. Workers already use `queue.put()` with `block=False`. The `Queue` type annotation is the same `multiprocessing.Queue[OutputMessage]`.

**Test fixtures** (`tests/conftest.py`):
- Replace `Manager().Queue()` with `spawn_ctx.Queue()` to match production

**Test files** (`tests/test_dep_injection.py`):
- Uses plain `Queue()` for `_run_stage_function_with_injection` calls — no change needed (it's same-process, not cross-process)

---

### Task 1: Replace Manager Queue with spawn-context Queue in engine

**Files:**
- Modify: `src/pivot/engine/engine.py:547-549` (queue creation)
- Modify: `src/pivot/engine/engine.py:723-728` (manager shutdown cleanup)
- Modify: `src/pivot/engine/engine.py:9` (remove `import multiprocessing as mp` if possible, or keep for `mp.Queue` type)

**Step 1: Modify queue creation in `_orchestrate_execution`**

Replace lines 547-549:
```python
        spawn_ctx = mp.get_context("spawn")
        local_manager = spawn_ctx.Manager()
        output_queue: mp.Queue[OutputMessage] = local_manager.Queue()  # pyright: ignore[reportAssignmentType]
```
With:
```python
        spawn_ctx = mp.get_context("spawn")
        output_queue: mp.Queue[OutputMessage] = spawn_ctx.Queue()  # pyright: ignore[reportAssignmentType]
```

**Step 2: Remove manager shutdown in finally block**

Replace lines 723-728:
```python
        finally:
            self._executor = None
            # Manager shutdown can fail if the manager process died unexpectedly
            with contextlib.suppress(OSError, BrokenPipeError):
                local_manager.shutdown()
```
With:
```python
        finally:
            self._executor = None
```

**Step 3: Remove unused `queue` import**

Remove from imports at line 11:
```python
import queue
```

(The `queue` module was only used in `_get_from_queue` for `queue.Empty`. After Task 2 removes that method, this import is unused.)

**Step 4: Run tests to verify queue creation still works**

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py -x -q`
Expected: PASS (workers don't care what Queue implementation they get)

---

### Task 2: Replace polling drain with blocking drain thread

**Files:**
- Modify: `src/pivot/engine/engine.py:60-62` (remove `_OUTPUT_QUEUE_DRAIN_TIMEOUT` constant)
- Modify: `src/pivot/engine/engine.py:764-805` (replace `_drain_output_queue` and `_get_from_queue`)
- Modify: `src/pivot/engine/engine.py:560-564` (drain task startup)
- Modify: `src/pivot/engine/engine.py:720-722` (drain shutdown signal)
- Add import: `from anyio import from_thread` at top or use `anyio.from_thread.run`

**Step 1: Remove `_OUTPUT_QUEUE_DRAIN_TIMEOUT` constant**

Delete line 62:
```python
_OUTPUT_QUEUE_DRAIN_TIMEOUT = 0.02
```

**Step 2: Replace `_drain_output_queue` method**

Replace the current polling implementation (lines 764-805):
```python
    async def _drain_output_queue(
        self,
        output_queue: mp.Queue[OutputMessage],
        stop_event: anyio.Event,
    ) -> None:
        """Drain output messages from worker processes and emit LogLine events."""
        while not stop_event.is_set():
            try:
                # Poll the queue in a thread to not block the event loop
                msg = await anyio.to_thread.run_sync(
                    lambda: self._get_from_queue(output_queue, timeout=_OUTPUT_QUEUE_DRAIN_TIMEOUT)
                )
                if msg is None:
                    continue
                ...
            except Exception:
                ...

    def _get_from_queue(self, q: mp.Queue[OutputMessage], timeout: float) -> OutputMessage | None:
        ...
```

With a blocking drain thread approach:
```python
    async def _drain_output_queue(
        self,
        output_queue: mp.Queue[OutputMessage],
    ) -> None:
        """Drain output messages from worker processes and emit LogLine events.

        Runs a blocking thread that calls queue.get() without polling.
        The thread exits when it receives a sentinel (None).
        Messages are forwarded into the async event loop via anyio.from_thread.run.
        """
        await anyio.to_thread.run_sync(
            lambda: self._blocking_drain(output_queue),
            abandon_on_cancel=True,
        )

    def _blocking_drain(self, output_queue: mp.Queue[OutputMessage]) -> None:
        """Block on queue.get() and forward messages to the async event loop.

        Runs in a dedicated thread. Exits when sentinel (None) is received.
        """
        while True:
            try:
                msg = output_queue.get()
            except (EOFError, OSError):
                break

            # Sentinel signals shutdown
            if msg is None:
                break

            try:
                stage_name, line, is_stderr = msg
            except (TypeError, ValueError):
                continue

            try:
                anyio.from_thread.run(
                    self._emit_log_line, stage_name, line, is_stderr
                )
            except Exception:
                # Event loop closed or cancelled — stop draining
                break

    async def _emit_log_line(self, stage_name: str, line: str, is_stderr: bool) -> None:
        """Emit a single LogLine event. Called from drain thread via from_thread.run."""
        await self.emit(
            LogLine(
                type="log_line",
                stage=stage_name,
                line=line,
                is_stderr=is_stderr,
            )
        )
```

**Step 3: Update drain task startup (remove stop_event)**

Replace lines 560-564:
```python
            # Start output drain task
            output_stop_event = anyio.Event()

            async with anyio.create_task_group() as tg:
                tg.start_soon(self._drain_output_queue, output_queue, output_stop_event)
```
With:
```python
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._drain_output_queue, output_queue)
```

**Step 4: Replace stop event with sentinel on shutdown**

Replace lines 720-722:
```python
                # Signal output drain task to stop
                output_stop_event.set()
```
With:
```python
                # Send sentinel to stop blocking drain thread
                output_queue.put(None)
```

**Step 5: Run full test suite**

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/ -n auto -q`
Expected: PASS

---

### Task 3: Update test fixture to use spawn-context Queue directly

**Files:**
- Modify: `tests/conftest.py:380-392` (output_queue fixture)

**Step 1: Replace Manager-based fixture**

Replace:
```python
@pytest.fixture
def output_queue() -> Generator[mp.Queue[OutputMessage]]:
    """Create a multiprocessing queue for worker output using spawn context.

    Uses spawn context to match production behavior and avoid Python 3.13+
    deprecation warnings about fork() in multi-threaded contexts.
    """
    spawn_ctx = mp.get_context("spawn")
    manager = spawn_ctx.Manager()
    # Manager().Queue() returns Queue[Any] - cast through object for type safety
    queue = cast("mp.Queue[OutputMessage]", cast("object", manager.Queue()))
    yield queue
    manager.shutdown()
```
With:
```python
@pytest.fixture
def output_queue() -> Generator[mp.Queue[OutputMessage]]:
    """Create a multiprocessing queue for worker output using spawn context.

    Uses spawn context to match production behavior and avoid Python 3.13+
    deprecation warnings about fork() in multi-threaded contexts.
    """
    spawn_ctx = mp.get_context("spawn")
    q: mp.Queue[OutputMessage] = spawn_ctx.Queue()  # pyright: ignore[reportAssignmentType]
    yield q
```

Also check if `cast` is still used elsewhere in conftest.py — if not, remove from imports.

**Step 2: Run the worker tests to verify fixture works**

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py tests/execution/test_execution_modes.py tests/test_run_cache_lock_update.py -x -q`
Expected: PASS

---

### Task 4: Run quality checks and final verification

**Step 1: Run linting and type checking**

Run: `cd /home/sami/pivot/roadmap-383 && uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: PASS (no new errors)

**Step 2: Run full test suite**

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/ -n auto`
Expected: All tests pass

---

## What Changed (Reviewer Checklist)

- [ ] `multiprocessing.Manager()` removed from engine — no more manager process for log queue
- [ ] Drain loop replaced: polling with 20ms timeout → blocking `queue.get()` + sentinel
- [ ] Deterministic shutdown: sentinel `None` replaces `anyio.Event` stop signal
- [ ] `_get_from_queue` helper removed (no longer needed)
- [ ] `_OUTPUT_QUEUE_DRAIN_TIMEOUT` constant removed
- [ ] Manager shutdown cleanup (`contextlib.suppress(OSError, BrokenPipeError)`) removed
- [ ] Test fixture uses `spawn_ctx.Queue()` directly (matches production)
- [ ] Worker code unchanged — `Queue[OutputMessage]` interface is identical
