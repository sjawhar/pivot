# Worker Output Capture — Bounded Memory + FD-Compatible stdout/stderr

**Goal:** Replace the unbounded `output_lines` list with a bounded ring buffer and give `_QueueWriter` a real file descriptor via `os.pipe()` so libraries that call `fileno()` work.

**Architecture:** Two changes to `_QueueWriter` in `src/pivot/executor/worker.py`:
1. Replace the `list[tuple[str, bool]]` accumulator with an `_OutputRingBuffer` that evicts oldest lines when capacity is reached, adding a truncation indicator.
2. Back each `_QueueWriter` with an `os.pipe()` — the write-end FD is returned by `fileno()`, a reader thread drains the read-end into the existing line-splitting + queue-sending logic.

**Tech Stack:** Python 3.13+, `os.pipe()`, `threading`, `collections.deque`

---

## Key Observations

1. **`output_lines` in `StageResult` is dead data.** No consumer reads `result["output_lines"]` — the engine only uses `status`, `reason`, `metrics`, and `deferred_writes`. Real-time output flows through the multiprocessing `Queue`. The `output_lines` field exists purely as a "just in case" backup stored in the result dict.

2. **The ring buffer replaces the list passed to `_QueueWriter`.** Instead of `list.append()`, we call `ring_buffer.append()`. The ring buffer is a thin wrapper around `collections.deque(maxlen=N)`.

3. **The pipe-backed FD is transparent.** The stage function writes to what it thinks is stdout/stderr. `contextlib.redirect_stdout/stderr` points to `_QueueWriter`. When something calls `fileno()`, it gets the write-end of a pipe. A background thread reads the pipe read-end and feeds bytes into the same `_QueueWriter.write()` path.

4. **Thread safety is already handled.** `_QueueWriter` has a `threading.Lock` protecting `_buffer`. The pipe reader thread just calls `write()` like any other thread.

## Design Decisions

**Ring buffer max lines default:** 1000 lines. Configurable per `_OutputRingBuffer(max_lines=N)`. This bounds memory to ~1000 * avg_line_len bytes (typically <1MB).

**Truncation indicator:** When the buffer overflows, the oldest line is evicted. After all output is collected, if `dropped_count > 0`, we prepend a single indicator line: `"[{dropped_count} earlier lines truncated]"`. This goes into the ring buffer's snapshot, not into the real-time queue (which already sent those lines).

**Pipe lifecycle:** The pipe is lazily initialized via `_ensure_pipe()` on first `fileno()` call, the reader thread starts immediately, and both FDs are closed in `__exit__` (write-end first to signal EOF, then join reader thread, then close read-end).

**`StageResult.output_lines` type change:** Change from `list[tuple[str, bool]]` to `list[tuple[str, bool]]` — same type, but now populated from ring buffer snapshot. No type change needed. We keep the field for backward compatibility of the TypedDict shape (tests and serialization).

---

## Task 1: Add `_OutputRingBuffer` class

**Files:**
- Modify: `src/pivot/executor/worker.py` (add class after `_QueueWriter`)
- Test: `tests/execution/test_executor_worker.py` (new tests)

### Step 1: Write failing tests for ring buffer

Add to `tests/execution/test_executor_worker.py`, after the existing `_QueueWriter` tests section:

```python
# =============================================================================
# _OutputRingBuffer Tests
# =============================================================================


def test_ring_buffer_stores_lines_within_capacity() -> None:
    """Ring buffer stores lines when under max_lines."""
    buf = worker._OutputRingBuffer(max_lines=5)
    buf.append("line1", False)
    buf.append("line2", True)
    assert buf.snapshot() == [("line1", False), ("line2", True)]
    assert buf.dropped_count == 0


def test_ring_buffer_evicts_oldest_on_overflow() -> None:
    """Ring buffer evicts oldest lines when exceeding max_lines."""
    buf = worker._OutputRingBuffer(max_lines=3)
    for i in range(5):
        buf.append(f"line{i}", False)
    snap = buf.snapshot()
    assert len(snap) == 3
    assert snap[0] == ("line2", False)
    assert snap[2] == ("line4", False)
    assert buf.dropped_count == 2


def test_ring_buffer_truncation_indicator() -> None:
    """Ring buffer includes truncation indicator when lines were dropped."""
    buf = worker._OutputRingBuffer(max_lines=2)
    for i in range(5):
        buf.append(f"line{i}", False)
    snap = buf.snapshot_with_truncation()
    assert len(snap) == 3  # indicator + 2 kept lines
    assert snap[0] == ("[3 earlier lines truncated]", False)
    assert snap[1] == ("line3", False)
    assert snap[2] == ("line4", False)


def test_ring_buffer_no_truncation_indicator_when_no_overflow() -> None:
    """Ring buffer snapshot_with_truncation returns plain snapshot when nothing dropped."""
    buf = worker._OutputRingBuffer(max_lines=10)
    buf.append("only line", False)
    snap = buf.snapshot_with_truncation()
    assert snap == [("only line", False)]


def test_ring_buffer_thread_safe() -> None:
    """Ring buffer is thread-safe under concurrent appends."""
    import concurrent.futures
    import threading

    buf = worker._OutputRingBuffer(max_lines=500)
    barrier = threading.Barrier(5)

    def writer(tid: int) -> None:
        barrier.wait()
        for i in range(100):
            buf.append(f"t{tid}-{i}", False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(writer, t) for t in range(5)]
        for f in futs:
            f.result()

    snap = buf.snapshot()
    assert len(snap) == 500
    assert buf.dropped_count == 0
```

### Step 2: Run tests to verify they fail

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py -k "ring_buffer" -v`
Expected: FAIL — `worker._OutputRingBuffer` does not exist

### Step 3: Implement `_OutputRingBuffer`

Add to `src/pivot/executor/worker.py`, just before the `_QueueWriter` class:

```python
class _OutputRingBuffer:
    """Bounded ring buffer for captured output lines.

    Uses collections.deque(maxlen=N) for O(1) append with automatic eviction.
    Thread-safe: all mutations protected by lock.
    """

    _lines: collections.deque[tuple[str, bool]]
    _dropped_count: int
    _lock: threading.Lock

    def __init__(self, max_lines: int = 1000) -> None:
        self._lines = collections.deque(maxlen=max_lines)
        self._dropped_count = 0
        self._lock = threading.Lock()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def append(self, line: str, is_stderr: bool) -> None:
        with self._lock:
            if len(self._lines) == self._lines.maxlen:
                self._dropped_count += 1
            self._lines.append((line, is_stderr))

    def snapshot(self) -> list[tuple[str, bool]]:
        with self._lock:
            return list(self._lines)

    def snapshot_with_truncation(self) -> list[tuple[str, bool]]:
        with self._lock:
            lines = list(self._lines)
            if self._dropped_count > 0:
                indicator = f"[{self._dropped_count} earlier lines truncated]"
                lines.insert(0, (indicator, False))
            return lines
```

Also add `import collections` to the imports at the top of `worker.py`.

### Step 4: Run tests to verify they pass

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py -k "ring_buffer" -v`
Expected: PASS

---

## Task 2: Wire ring buffer into `_QueueWriter` and `execute_stage`

**Files:**
- Modify: `src/pivot/executor/worker.py` — change `_QueueWriter`, `_run_stage_function_with_injection`, `execute_stage`, `_make_result`
- Modify: `src/pivot/engine/engine.py:619-622` — remove `output_lines=[]` from error StageResult
- Modify: `tests/execution/test_executor_worker.py` — update existing tests
- Modify: `tests/test_dep_injection.py` — update callers

### Step 1: Change `_QueueWriter` to use `_OutputRingBuffer`

Replace the `_output_lines: list[tuple[str, bool]]` field and `output_lines` constructor parameter with `_ring_buffer: _OutputRingBuffer`:

In `_QueueWriter.__init__`:
- Remove `output_lines` parameter
- Add `ring_buffer: _OutputRingBuffer` parameter
- Store as `self._ring_buffer = ring_buffer`

In `_QueueWriter._send_line`:
- Change `self._output_lines.append(...)` to `self._ring_buffer.append(line, self._is_stderr)`

### Step 2: Change `_run_stage_function_with_injection` signature

Change parameter from `output_lines: list[tuple[str, bool]]` to `ring_buffer: _OutputRingBuffer`:

```python
def _run_stage_function_with_injection(
    func: Callable[..., Any],
    stage_name: str,
    output_queue: Queue[OutputMessage],
    ring_buffer: _OutputRingBuffer,
    ...
```

Pass `ring_buffer` to both `_QueueWriter` constructors.

### Step 3: Change `execute_stage` to use ring buffer

Replace `output_lines: list[tuple[str, bool]] = []` with `ring_buffer = _OutputRingBuffer()` (uses default 1000 max_lines).

In `_make_result`, change parameter from `output_lines: list[tuple[str, bool]]` to `ring_buffer: _OutputRingBuffer`, and use `ring_buffer.snapshot_with_truncation()`:

```python
def _make_result(
    status: Literal[StageStatus.RAN, StageStatus.SKIPPED, StageStatus.FAILED],
    reason: str,
    ring_buffer: _OutputRingBuffer,
) -> StageResult:
    return StageResult(
        status=status,
        reason=reason,
        output_lines=ring_buffer.snapshot_with_truncation(),
        metrics=metrics.get_entries(),
    )
```

Update all call sites in `execute_stage`:
- Where `_make_result(status, reason, [])` is used for early returns (no output captured yet), pass a fresh `_OutputRingBuffer(max_lines=0)` or just use an empty ring buffer. **Simpler:** create the ring buffer at the top of `execute_stage` and pass it everywhere.
- Where `_make_result(status, reason, output_lines)` is used, change to `ring_buffer`.
- The two early-return `StageResult(...)` constructions that hardcode `output_lines=[]` should also use `ring_buffer.snapshot_with_truncation()`.

### Step 4: Update `engine.py` error path

In `src/pivot/engine/engine.py:619-622`, the error StageResult:
```python
failed_result = StageResult(
    status=StageStatus.FAILED,
    reason=str(e),
    output_lines=[],
)
```
This stays as `output_lines=[]` — it's an engine-side error, not worker output.

### Step 5: Update tests

**`tests/execution/test_executor_worker.py`:**

All tests that create `output_lines: list[tuple[str, bool]] = []` and pass to `_QueueWriter` or `_run_stage_function_with_injection` need to:
1. Create `ring_buffer = worker._OutputRingBuffer(max_lines=1000)` instead
2. Pass `ring_buffer=ring_buffer` instead of `output_lines=output_lines`
3. Assert against `ring_buffer.snapshot()` instead of `output_lines`

This affects approximately 20+ test functions. The changes are mechanical:
- `output_lines: list[tuple[str, bool]] = []` → `ring_buffer = worker._OutputRingBuffer()`
- `output_lines=output_lines` → `ring_buffer=ring_buffer`
- `assert output_lines == [...]` → `assert ring_buffer.snapshot() == [...]`
- `assert len(output_lines) == N` → `assert len(ring_buffer.snapshot()) == N`
- `output_lines[i]` → `ring_buffer.snapshot()[i]`

**`tests/test_dep_injection.py`:** Same mechanical change at lines 480 and 531.

### Step 6: Run full test suite

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py tests/test_dep_injection.py -v`
Expected: PASS

---

## Task 3: Add pipe-backed `fileno()` support to `_QueueWriter`

**Files:**
- Modify: `src/pivot/executor/worker.py` — add pipe to `_QueueWriter`
- Test: `tests/execution/test_executor_worker.py` — new + updated tests

### Step 1: Write failing tests

```python
def test_queue_writer_fileno_returns_valid_fd(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter.fileno() returns a writable file descriptor."""
    ring_buffer = worker._OutputRingBuffer()
    with worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, ring_buffer=ring_buffer
    ) as writer:
        fd = writer.fileno()
        assert isinstance(fd, int)
        # Write through the FD directly
        os.write(fd, b"hello from fd\n")
    # Reader thread drains pipe into ring buffer
    assert ("hello from fd", False) in ring_buffer.snapshot()


def test_queue_writer_fd_captures_subprocess_output(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """_QueueWriter captures output from subprocess using fileno()."""
    import subprocess

    ring_buffer = worker._OutputRingBuffer()
    with worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, ring_buffer=ring_buffer
    ) as writer:
        subprocess.run(
            [sys.executable, "-c", "print('subprocess hello')"],
            stdout=writer.fileno(),
            check=True,
        )
    assert ("subprocess hello", False) in ring_buffer.snapshot()


def test_queue_writer_pipe_and_write_interleave(
    output_queue: mp.Queue[OutputMessage],
) -> None:
    """Output via write() and via FD both appear in ring buffer."""
    ring_buffer = worker._OutputRingBuffer()
    with worker._QueueWriter(
        "test_stage", output_queue, is_stderr=False, ring_buffer=ring_buffer
    ) as writer:
        writer.write("from write\n")
        os.write(writer.fileno(), b"from fd\n")
    snap = ring_buffer.snapshot()
    lines = [line for line, _ in snap]
    assert "from write" in lines
    assert "from fd" in lines
```

### Step 2: Run tests to verify they fail

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py -k "fileno" -v`
Expected: FAIL

### Step 3: Implement pipe-backed FD in `_QueueWriter`

The design:
- In `__init__`, call `os.pipe()` → `(read_fd, write_fd)`. Store both.
- In `__enter__`, start a daemon thread that reads from `read_fd` in a loop, feeding bytes into `self.write()`.
- `fileno()` returns `write_fd`.
- In `__exit__`, close `write_fd` first (signals EOF to reader), join the reader thread, then close `read_fd`.

```python
class _QueueWriter:
    _stage_name: str
    _queue: Queue[OutputMessage]
    _is_stderr: bool
    _ring_buffer: _OutputRingBuffer
    _buffer: str
    _redirect: contextlib.AbstractContextManager[object]
    _lock: threading.Lock
    _read_fd: int
    _write_fd: int
    _reader_thread: threading.Thread | None

    def __init__(
        self,
        stage_name: str,
        output_queue: Queue[OutputMessage],
        *,
        is_stderr: bool,
        ring_buffer: _OutputRingBuffer,
    ) -> None:
        self._stage_name = stage_name
        self._queue = output_queue
        self._is_stderr = is_stderr
        self._ring_buffer = ring_buffer
        self._buffer = ""
        self._lock = threading.Lock()
        self._read_fd, self._write_fd = os.pipe()
        self._reader_thread = None
        if is_stderr:
            self._redirect = contextlib.redirect_stderr(self)
        else:
            self._redirect = contextlib.redirect_stdout(self)

    def _pipe_reader(self) -> None:
        """Read from pipe read-end and feed into write() for line splitting."""
        try:
            while True:
                data = os.read(self._read_fd, 8192)
                if not data:
                    break
                self.write(data.decode("utf-8", errors="replace"))
        except OSError:
            pass  # Pipe closed

    def __enter__(self) -> _QueueWriter:
        self._reader_thread = threading.Thread(
            target=self._pipe_reader, daemon=True, name=f"pipe-reader-{self._stage_name}"
        )
        self._reader_thread.start()
        self._redirect.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._redirect.__exit__(exc_type, exc_val, exc_tb)
        # Close write-end to signal EOF to reader thread
        with contextlib.suppress(OSError):
            os.close(self._write_fd)
        # Wait for reader thread to finish draining
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
        # Close read-end
        with contextlib.suppress(OSError):
            os.close(self._read_fd)
        self.flush()

    def fileno(self) -> int:
        """Return write-end of pipe for FD-compatible operations."""
        return self._write_fd

    # write(), flush(), _send_line(), isatty() remain the same
    # except _send_line uses ring_buffer instead of output_lines
```

### Step 4: Update the existing `fileno` test

The old test `test_queue_writer_fileno_raises_unsupported_operation` should be **deleted** and replaced by the new `test_queue_writer_fileno_returns_valid_fd` test.

### Step 5: Run tests

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/execution/test_executor_worker.py -k "fileno or pipe or subprocess" -v`
Expected: PASS

### Step 6: Run full test suite

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/ -n auto`
Expected: PASS

---

## Task 4: Run quality checks and final validation

**Files:** All modified files

### Step 1: Format and lint

Run: `cd /home/sami/pivot/roadmap-383 && uv run ruff format . && uv run ruff check .`
Expected: Clean

### Step 2: Type check

Run: `cd /home/sami/pivot/roadmap-383 && uv run basedpyright`
Expected: Clean

### Step 3: Full test suite

Run: `cd /home/sami/pivot/roadmap-383 && uv run pytest tests/ -n auto`
Expected: All pass

---

## Files Changed Summary

| File | Change |
|------|--------|
| `src/pivot/executor/worker.py` | Add `_OutputRingBuffer`, refactor `_QueueWriter` (ring buffer + pipe FD), update `_make_result`, `execute_stage`, `_run_stage_function_with_injection` signatures |
| `src/pivot/engine/engine.py` | No change needed (already uses `output_lines=[]` in error path) |
| `src/pivot/types.py` | No change needed (`output_lines: list[tuple[str, bool]]` type stays same) |
| `tests/execution/test_executor_worker.py` | Update ~20 tests from `output_lines` list to `ring_buffer`, add ring buffer tests, add pipe FD tests, delete old `fileno` raises test |
| `tests/test_dep_injection.py` | Update 2 call sites from `output_lines` to `ring_buffer` |

## Uncertainty / Open Questions

1. **Pipe reader thread join timeout:** 5 seconds should be generous. If a stage writes massive data to the pipe, the reader should drain quickly since it's reading into memory. If this proves flaky, we can increase or add a warning.

2. **Two `_QueueWriter` instances per stage (stdout + stderr):** Each creates its own pipe. That's 4 FDs per stage execution. Should be fine — Unix default limit is 1024 FDs and workers run one stage at a time.

3. **`collections.deque` maxlen behavior:** When `len(deque) == maxlen`, the next `append()` silently drops the leftmost item. We manually track `_dropped_count` before the append to count truncated lines accurately.
