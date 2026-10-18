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

Use Pivot's `_get_executor()` policy, which sets the shared timeout, child initializer and
reuse mode:

```python
def create_executor(max_workers: int) -> Executor:
    """Get Pivot's reusable executor."""
    return _get_executor(max_workers)
```

The key behaviors:

1. **First call** - Creates the reusable executor, then spawns processes on warm-up or submit
2. **Later compatible calls** - Reuse the executor; loky may resize if requested capacity grows
3. **Workers stay alive** - Between runs in a long-lived Engine, workers idle and remain ready
4. **Imports cached** - Libraries imported by stages remain loaded for each worker lifetime

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
    """Pre-warm Pivot's configured reusable worker pool before TUI startup."""
    workers = compute_max_workers(stage_count, max_workers)
    _warm_workers(_get_executor(workers), workers)
    return workers
```

Pre-warming is important when using a TUI: Textual inherits file descriptors at startup, and loky workers created after TUI initialization can inherit TUI's PTY descriptors, causing display corruption.

### Killing Workers for Code Reload

In watch mode, code changes require fresh workers with updated imports:

```python
def restart_workers(stage_count: int, max_workers: int | None = None) -> int:
    """Kill existing workers and spawn fresh ones."""
    workers = compute_max_workers(stage_count, max_workers)
    _warm_workers(_get_executor(workers, kill_workers=True), workers)
    return workers
```

`reuse=False` makes loky replace a compatible live executor instead of returning it unchanged,
ensuring updated imports are loaded.

### Cleanup on Exit

The Engine shuts its workers and manager down when it exits. An atexit fallback remains useful
when code creates a reusable executor outside an Engine context.

Neither path runs when the coordinator is killed (SIGKILL, or a SIGTERM with no handler), and
workers use `timeout=None`, so nothing else would end them. The output-queue manager therefore
starts with `_exit_when_parent_dies`, a thread that exits it once its parent pid changes. Every
worker starts with `core._init_worker`, which runs `worker.unwind_when_stopped` (and, while a
discovery's import session lasts, installs discovery's checked loader):

- SIGTERM raises `worker.WorkerTerminated` (a `KeyboardInterrupt`) in the running stage, so its
  cleanup runs, and the worker exits once the stage has unwound, or after
  `worker.TERMINATION_GRACE_S` (10 s) whatever the stage does. An idle worker exits at once.
- SIGINT (Ctrl-C at a terminal reaches the coordinator and its workers together) only raises
  `WorkerTerminated` in the running stage; the worker stays up until the coordinator stops it.
  loky's manager thread takes a worker that exits unasked for a broken pool and SIGKILLs every
  worker it still lists, so a worker exiting on SIGINT would cut the others' cleanup off. A
  SIGINT inherited as ignored (a background job of a non-interactive shell) stays ignored.
- When the parent pid changes, a thread sends the main thread SIGTERM and exits the worker
  after the same grace itself, in case the stage replaced the handler or is blocked.
- `WorkerTerminated` is raised into a stage once; later stop signals do not interrupt its
  cleanup. A stage that catches it and returns has its result discarded before any output is
  written.
- Every exit first SIGKILLs the processes still descended from the worker (psutil). A process
  that detached from that tree is not reached.

`WorkerPool.hard_cancel` (an Engine leaving on an exception, such as Ctrl-C) takes the workers
out of the executor's `_processes` first, for the reason above, then sends each SIGTERM, waits up
to the same grace, and SIGKILLs any worker left with its descendants (by pid: loky's process
object has no `kill()`). Once `_processes` is cleared, only `executor.shutdown` wakes loky's
manager thread, so `hard_cancel` calls it in a `finally`: a coordinator whose manager thread is
left waiting never exits. Start any new long-lived child process with one of these initializers.

## Key Insight

The standard library's `ProcessPoolExecutor` is designed for batch workloads where pool creation is amortized over many tasks. For iterative workflows with frequent re-runs, the per-run overhead dominates.

`loky.get_reusable_executor()` inverts this by treating the worker pool as long-lived
infrastructure rather than per-invocation ephemera:

| Aspect | Standard ProcessPoolExecutor | Pivot reusable executor |
|---|---|---|
| Pool lifetime | Per `with` block | Engine lifetime |
| Worker spawn | Every run | Once, after a code restart, or when capacity grows |
| Import overhead | Every worker spawn | Once per worker lifetime |
| Idle workers | Terminated | Wait for next task until Engine exit |

The tradeoff is memory: idle workers consume RAM. For pipelines where iteration speed matters more than memory footprint, warm workers provide significant latency reduction.

### Why Not threading?

Python's GIL serializes CPU-bound work in threads. Stages that do NumPy/pandas computation release the GIL and benefit from threads, but pure Python stages (file I/O, string processing) would run sequentially. Process pools provide true parallelism regardless of workload type.
