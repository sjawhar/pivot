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
) -> loky.ProcessPoolExecutor:
    """Get Pivot's reusable executor without growing it before work is submitted."""
    return _get_executor(max_workers, idle_timeout_s=idle_timeout_s)
```

The key behaviors:

1. **First dispatch** - `core.submit()` sets loky's private `_max_workers` to the number of
   submitted, unfinished stages, capped by the Engine's limit. The Engine records each direct
   consumer wave's half-timeout deadline, and its existing 100 ms execution loop services those
   deadlines, so a parent costs no thread. Once a deadline passes, the loop sends a
   barrier-coordinated ping to each idle pre-started worker. Pings stop and queued ping futures
   are cancelled when the parent resolves, fails, or the run ends; those workers then follow
   the normal timeout.
2. **Later dispatches** - An existing idle worker takes the stage. A new worker starts only when
   every live worker already has work.
3. **Idle cleanup** - One-shot Engines give idle workers a 10-second timeout. The timeout is
   126 times the 78.898 ms maximum coordinator handoff gap measured on a 100-stage pandas/NumPy
   chain, while remaining short relative to long remote-stage waits.
4. **Long-lived Engines** - Watch, serve and persistent TUI Engines use no idle timeout, so a
   later run reuses their warm workers.

Pivot constrains Loky to `>=3.5.6,<3.8`: 3.5.0–3.5.5 print `resource_tracker` tracebacks on
Python 3.13, and the code reads private executor state that a later release may move. It
finds the reusable executor in the legacy `_executor` singleton (3.5.6, 3.6) or 3.7's
`_executor_storage`; an unknown layout raises an error naming the installed Loky version. The
fresh-wheel test installs 3.5.6, 3.6.0 and 3.7.0 and checks the worker count and idle exit for
each. An upgrade must recheck reusable-executor lookup plus `_max_workers` and
`_pending_work_items` submission accounting; an upstream change can reintroduce refill churn.

### Starting Process Trackers for the TUI

Textual captures `sys.stderr` with an object whose `fileno()` is -1. Both loky's and
`multiprocessing`'s resource trackers pass `sys.stderr.fileno()` while launching, so the
trackers must start before Textual begins. `core.start_resource_trackers()` does that in
`run_tui_with_engine()`.

Workers themselves may start after the TUI begins. They use the already-running trackers, and
Pivot's normal worker output queue continues to capture stage logs.

### Killing Workers for Code Reload

In watch mode, code changes require fresh workers with updated imports. `WorkerPool.restart()`
first shuts down the executor the pool holds with `kill_workers=True`, then calls
`restart_workers()` to start one fresh worker; subsequent workers start on demand:

```python
def _replace_executor(self, idle_timeout_s: float | None) -> loky.ProcessPoolExecutor:
    if self._executor is not None:
        executor = self._executor
        self._executor = None
        executor.shutdown(wait=True, kill_workers=True)
    executor_core.restart_workers(self._max_workers, idle_timeout_s=idle_timeout_s)
    return executor_core.create_executor(self._max_workers, idle_timeout_s=idle_timeout_s)
```

The pool shuts down the executor it holds rather than asking loky for "the" reusable executor.
Loky 3.7 keeps one reusable executor per thread, and the reload restarts from an anyio worker
thread that never created the pool's executor, so `get_reusable_executor(kill_workers=True)`
there finds no executor and leaves the previous workers alive. `reuse=False` still makes loky
replace a compatible live executor instead of returning it unchanged. Every path that replaces
the pool's executor goes through `_replace_executor`: a reload's `restart()`, and a `start()`
that finds stale user code or a different idle timeout. After a restart from another thread,
loky's lookup on the calling thread no longer finds the executor the pool holds, so a `start()`
that let loky swap it for a new timeout would leave the restart's worker alive.

The pool forgets the old executor as soon as it is shut down. If the fresh worker then fails to
start, the Engine logs the failed restart and the pool holds no executor, so the next run's
`start()` creates one as on first use. A shut-down executor rejects every submission, so
keeping it would make that run fail.

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
