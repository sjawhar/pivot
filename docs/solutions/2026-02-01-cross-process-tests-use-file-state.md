---
tags: [python, testing, multiprocessing]
category: gotcha
module: tests
symptoms: ["test assertions fail despite code running", "shared state not updated", "list appears empty in assertions"]
---

# Cross-Process Tests: Use File-Based State, Not Shared Lists

## Problem

When testing code that runs in worker processes (via `ProcessPoolExecutor`, loky, or similar), you cannot use Python lists or other mutable objects to collect results from workers. Each process gets its own copy of the data structure, so modifications in the worker process are invisible to the test.

```python
# BROKEN: execution_count will always be 0 in test assertions
execution_count = [0]

def stage_that_runs_in_worker():
    execution_count[0] += 1  # Increments a COPY, not the original
    Path("output.txt").write_text("done")

def test_stage_runs_exactly_once():
    # Submit to process pool...
    executor.submit(stage_that_runs_in_worker)
    future.result()

    assert execution_count[0] == 1  # FAILS: still 0!
```

This happens because:

1. **Spawn context** (default on macOS, loky's default): Child process starts fresh, imports the module, and gets a new `execution_count = [0]`
2. **Fork context** (traditional Linux): Child process copies parent's memory, including `execution_count`, but subsequent modifications are copy-on-write

Either way, the parent's list never sees the child's modifications.

## Solution

Use file-based state that persists across process boundaries:

```python
def stage_that_runs_in_worker(counter_file: Path):
    count = int(counter_file.read_text()) if counter_file.exists() else 0
    counter_file.write_text(str(count + 1))
    Path("output.txt").write_text("done")

def test_stage_runs_exactly_once(tmp_path: Path):
    counter_file = tmp_path / "counter.txt"

    # Submit to process pool...
    executor.submit(stage_that_runs_in_worker, counter_file)
    future.result()

    assert counter_file.read_text() == "1"  # Works!
```

For more complex state, use appropriate IPC mechanisms:

| State Type | Mechanism |
|------------|-----------|
| Simple counter | File with integer |
| Execution log | File with one line per call |
| Structured data | JSON file |
| Real-time communication | `multiprocessing.Manager().Queue()` |
| Shared primitive values | `multiprocessing.Value()` |

### Pattern: Counter File for Execution Tracking

From `tests/execution/test_executor_worker.py`:

```python
def test_execute_stage_reruns_when_fingerprint_changes(
    worker_env: Path, output_queue: mp.Queue[OutputMessage], tmp_path: Path
) -> None:
    """Worker reruns stage when code fingerprint changes."""
    (tmp_path / "input.txt").write_text("data")
    counter = tmp_path / "counter.txt"

    def stage_func_v1() -> None:
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))

    # First run
    result1 = executor.execute_stage("test_stage", stage_info_v1, worker_env, output_queue)
    assert result1["status"] == "ran"
    assert counter.read_text() == "1"

    # Second run with different fingerprint
    result2 = executor.execute_stage("test_stage", stage_info_v2, worker_env, output_queue)
    assert result2["status"] == "ran"
    assert counter.read_text() == "2"
```

### When Lists DO Work (Same-Process Calls)

If you're testing worker logic directly without going through the process pool, lists work fine:

```python
def test_worker_function_directly(tmp_path: Path):
    """Direct call - no subprocess, so list mutation works."""
    execution_count = [0]

    def stage_func() -> None:
        execution_count[0] += 1
        (tmp_path / "output.txt").write_text(f"output {execution_count[0]}")

    # Direct call - NOT submitted to process pool
    worker.execute_stage("test_stage", stage_info, worker_env, output_queue)

    assert execution_count[0] == 1  # Works because same process
```

This pattern appears in `tests/execution/test_execution_modes.py` for tests that call `worker.execute_stage()` directly. These tests verify worker logic without the overhead of inter-process communication. However, if you need to test the full pipeline including process pool submission, you must use file-based state.

## Key Insight

Python multiprocessing creates isolated memory spaces. Any state verification across process boundaries must use:

1. **Filesystem** - files, directories (simplest, most reliable)
2. **Manager objects** - `Manager().Queue()`, `Manager().dict()` (for real-time IPC)
3. **Shared memory** - `Value()`, `Array()` (for primitives only)

Never rely on in-memory Python objects to share state between parent and child processes. The test process and worker processes live in separate memory spaces with no automatic synchronization.
