---
tags: [python, multiprocessing, loky, performance]
category: implementation
module: executor
symptoms: ["slow stage execution", "worker startup overhead", "process pool recreation"]
---

# loky Reusable Executor for Warm Worker Pools

## Problem

Creating a new `ProcessPoolExecutor` for each pipeline run incurs significant startup overhead:

1. **Process spawning** - Each new worker requires `fork()` or `spawn()`, which takes 50-200ms per process
2. **Import time** - Workers must re-import heavy libraries (numpy, pandas, scikit-learn) on every spawn
3. **Connection setup** - IPC channels between coordinator and workers require handshaking

For iterative development workflows (edit-run-edit-run), this overhead compounds: a 5-worker pool might add 500ms+ to every execution, even for stages that complete in milliseconds.

```python
# Naive approach - creates new pool every time
def run_stages():
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Pool created here, workers spawn...
        executor.submit(stage_func)
        # Pool destroyed here, workers die
```

## Solution

Use `loky.get_reusable_executor()` which maintains a singleton worker pool across calls:

```python
import loky

def create_executor(max_workers: int) -> Executor:
    """Get reusable loky executor - workers persist across calls."""
    return loky.get_reusable_executor(max_workers=max_workers)
```

The key behaviors:

1. **First call** - Creates worker pool, spawns processes
2. **Subsequent calls** - Returns the existing pool if compatible (same `max_workers`)
3. **Workers stay alive** - Between stage executions, workers idle but remain ready
4. **Imports cached** - Heavy libraries imported once per worker lifetime, not per stage

### Pre-warming Workers

To ensure workers are ready before the first real task, submit no-op functions:

```python
def _noop() -> None:
    """Module-level function for pickling."""
    pass

def _warm_workers(pool: loky.ProcessPoolExecutor, count: int) -> None:
    """Submit no-ops to establish worker channels."""
    futures = [pool.submit(_noop) for _ in range(count)]
    for f in futures:
        f.result()

def prepare_workers(stage_count: int, max_workers: int | None = None) -> int:
    """Pre-warm loky worker pool before TUI startup."""
    workers = compute_max_workers(stage_count, max_workers)
    pool = loky.get_reusable_executor(max_workers=workers)
    _warm_workers(pool, workers)
    return workers
```

Pre-warming is important when using a TUI: Textual inherits file descriptors at startup, and loky workers created after TUI initialization can inherit TUI's PTY descriptors, causing display corruption.

### Killing Workers for Code Reload

In watch mode, code changes require fresh workers with updated imports:

```python
def restart_workers(stage_count: int, max_workers: int | None = None) -> int:
    """Kill existing workers and spawn fresh ones."""
    workers = compute_max_workers(stage_count, max_workers)
    pool = loky.get_reusable_executor(max_workers=workers, kill_workers=True)
    _warm_workers(pool, workers)
    return workers
```

The `kill_workers=True` flag terminates existing workers before creating new ones, ensuring updated code is loaded.

### Cleanup on Exit

Register an atexit handler to prevent orphaned workers:

```python
import atexit
import functools

def _cleanup_worker_pool() -> None:
    """Kill loky worker pool on process exit."""
    with contextlib.suppress(Exception):
        loky.get_reusable_executor(max_workers=1, kill_workers=True)

@functools.cache  # Single registration across threads
def _ensure_cleanup_registered() -> None:
    atexit.register(_cleanup_worker_pool)
```

## Key Insight

The standard library's `ProcessPoolExecutor` is designed for batch workloads where pool creation is amortized over many tasks. For iterative workflows with frequent re-runs, the per-run overhead dominates.

`loky.get_reusable_executor()` inverts this by treating the worker pool as long-lived infrastructure rather than per-invocation ephemera:

| Aspect | Standard ProcessPoolExecutor | loky Reusable Executor |
|--------|------------------------------|------------------------|
| Pool lifetime | Per `with` block | Process lifetime |
| Worker spawn | Every run | Once (or on resize/kill) |
| Import overhead | Every worker spawn | Once per worker lifetime |
| Idle workers | Terminated | Wait for next task |

The tradeoff is memory: idle workers consume RAM. For pipelines where iteration speed matters more than memory footprint, warm workers provide significant latency reduction.

### Why Not threading?

Python's GIL serializes CPU-bound work in threads. Stages that do NumPy/pandas computation release the GIL and benefit from threads, but pure Python stages (file I/O, string processing) would run sequentially. Process pools provide true parallelism regardless of workload type.
