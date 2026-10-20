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
def create_executor(
    max_workers: int, *, idle_timeout_s: float | None = None
) -> Executor:
    """Get Pivot's reusable executor without growing it before work is submitted."""
    return _get_executor(max_workers, idle_timeout_s=idle_timeout_s)
```

The key behaviors:

1. **First dispatch** - `core.submit()` sets loky's private `_max_workers` to the number of
   submitted, unfinished stages, capped by the Engine's limit. It also accepts a
   resource-admission preview of the dispatched stage's direct consumers, so a wide next wave
   can finish worker startup while its parent runs.
2. **Later dispatches** - An existing idle worker takes the stage. A new worker starts only when
   every live worker already has work.
3. **Idle cleanup** - One-shot Engines give idle workers a 10-second timeout. The timeout is
   126 times the 78.898 ms maximum coordinator handoff gap measured on a 100-stage pandas/NumPy
   chain, while remaining short relative to long remote-stage waits.
4. **Long-lived Engines** - Watch, serve and persistent TUI Engines use no idle timeout, so a
   later run reuses their warm workers.

This relies on loky 3.5.6's `_max_workers`, `_pending_work_items` and `_submit_resize_lock`.
Those internals must be rechecked when the loky pin changes: an upstream change to their
submission or worker-exit accounting can reintroduce refill churn.

### Starting Process Trackers for the TUI

Textual captures `sys.stderr` with an object whose `fileno()` is -1. Both loky's and
`multiprocessing`'s resource trackers pass `sys.stderr.fileno()` while launching, so the
trackers must start before Textual begins. `core.start_resource_trackers()` does that in
`run_tui_with_engine()`.

Workers themselves may start after the TUI begins. They use the already-running trackers, and
Pivot's normal worker output queue continues to capture stage logs.

### Killing Workers for Code Reload

In watch mode, code changes require fresh workers with updated imports. `restart_workers()`
kills the reusable executor and starts one fresh worker; subsequent workers start on demand:

```python
def restart_workers(max_workers: int, *, idle_timeout_s: float | None) -> int:
    """Replace stale workers and leave one fresh worker ready for the next dispatch."""
    pool = _get_executor(max_workers, idle_timeout_s=idle_timeout_s, kill_workers=True)
    submit(pool, max_workers, _noop).result()
    return 1
```

`reuse=False` makes loky replace a compatible live executor instead of returning it unchanged,
ensuring updated imports are loaded.

### Cleanup on Exit

The Engine shuts its workers and manager down when it exits. An atexit fallback remains useful
when code creates a reusable executor outside an Engine context.

Neither path runs when the coordinator is killed (SIGKILL, or a SIGTERM with no handler).
The output-queue manager therefore starts with `_exit_when_parent_dies`, a thread that exits it
once its parent pid changes. Every worker starts with `core._init_worker`, which runs
`worker.unwind_when_stopped` (and, while a discovery's import session lasts, installs
discovery's checked loader):

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

The standard library's `ProcessPoolExecutor` is designed for batch workloads where pool creation
is amortized over many tasks. For iterative workflows with frequent re-runs, the per-run
overhead dominates.

`loky.get_reusable_executor()` makes the worker pool long-lived infrastructure while Pivot's
submission helper controls how many workers are live:

| Aspect | Standard ProcessPoolExecutor | Pivot reusable executor |
|---|---|---|
| Pool lifetime | Per `with` block | Engine lifetime |
| Worker spawn | Every run | When an admitted stage has no idle worker |
| Import overhead | Every worker spawn | Once per worker lifetime |
| One-shot idle worker | N/A | Exits after the idle timeout |
| Long-lived idle worker | N/A | Waits for the next run |

The tradeoff is deliberate: one-shot runs reclaim workers while they wait, while long-lived
interactive Engines retain their warm pool for the next run.

### Why Not threading?

Python's GIL serializes CPU-bound work in threads. Stages that do NumPy/pandas computation release the GIL and benefit from threads, but pure Python stages (file I/O, string processing) would run sequentially. Process pools provide true parallelism regardless of workload type.
