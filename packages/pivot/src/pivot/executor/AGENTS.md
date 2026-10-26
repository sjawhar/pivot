# Pivot Executor - Development Guidelines

Workers execute in separate processes via `loky.get_reusable_executor()`. An Engine keeps its
workers and manager alive across its runs; each run receives a fresh manager queue for output,
and a manager that cannot be reached when a run starts is replaced then.
The loky initializer (`core._init_worker`, which runs `worker.unwind_when_stopped`) and the manager's initializer
(`core._init_manager`, which runs `core._exit_when_parent_dies`) start the parent-death watcher, so a worker or manager exits
when its Pivot process dies; a worker first unwinds its running stage so its `finally` blocks
run, within `worker.TERMINATION_GRACE_S`. The manager also ignores SIGTERM (multiprocessing
already makes it ignore SIGINT), so a stop signalled to the whole process group leaves it to
its owner's shutdown, which removes its `pymp-*` directory.
Stop a worker with `worker.terminate`, which sends SIGTERM at its main thread (tgkill), not with
`os.kill`: Linux can give a signal sent to the process to another of its threads, such as one of
NumPy's OpenBLAS threads, and the stage on the main thread then never stops (see
`docs/solutions/2026-02-01-loky-reusable-executor-warm-workers.md`).

One-shot Engines give idle workers `core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S`. Long-lived watch,
serve and TUI Engines use no idle timeout so they stay warm between runs. A run's first
submission starts the run's resource width (`Scheduler.resource_width`). After it,
`core.submit()` drives loky's private `_max_workers` to the submitted work count, so idle workers
are not respawned until a stage needs one. Parents running past half the one-shot idle timeout
periodically ping idle workers pre-started for their direct consumers. The pings stop when the
parent resolves, so the workers then follow the normal idle timeout. Keep
`reuse=False if kill_workers else "auto"` in the shared executor policy. A pool retains an
executor only if `core.is_executor_usable` finds one that exists, is neither shut down nor broken,
and has a call queue at least as wide as the pool. A restart, stale user code, a new idle timeout
or an unusable held executor replaces through `WorkerPool._replace_executor`, which stops the
executor the pool holds before asking loky for a new one. Loky 3.7 keeps its reusable executor per
thread, and the watch reload restarts from an anyio worker thread.

loky fixes an executor's call queue at creation (`queue_size`, 2 × max(cpu, width) + 1) and
will not reuse one whose queue is smaller than the requested width. `submit()` lowers loky's
`_max_workers` to the pending work, so `core._reusable_worker_target` passes that narrowed width
back only when loky will reuse the executor. When loky will replace it (missing, shut down,
broken, arguments changed or a queue too small for the request), it passes the requested width,
so the executor that replaces it gets a queue that fits.

Pivot supports `loky>=3.5.6,<3.8`: 3.5.6 and 3.6 keep the reusable executor in the legacy
`_executor` singleton; 3.7 keeps it in `_executor_storage`. Reject an unknown layout with an
error that names the installed Loky version. While a discovery's import session lasts, the
worker initializer (`core._init_worker`) also has each worker import user code through
discovery's checked loader (`discovery.install_user_code_loader`); loky replaces a live pool
whose initializer arguments differ.

## WorkerStageInfo Contract

`WorkerStageInfo` (TypedDict at `worker.py`) is the coordinator-to-worker contract:

| Field | Purpose |
|-------|---------|
| `func` | Stage function (must be picklable) |
| `fingerprint` | Code manifest for change detection |
| `deps` | Input dependency paths |
| `outs` | Output artifact references |
| `incremental_inputs` | Parameter-to-own-output identities for prior incremental values |
| `unsafe_fingerprint` | The fingerprint downgraded a refusal to a warning; the run cache is neither read nor written |
| `no_fingerprint` | The stage is `@no_fingerprint`: its declared files are its code, so the worker does not check the user modules it imports |
| `project_root` | Absolute path to project root |
| `state_dir` | Absolute path to `.pivot/` directory |

## Path Derivation

Workers derive all paths from `project_root` and `state_dir` in `WorkerStageInfo`:
- `state_dir = stage_info["state_dir"]` (passed directly to `StateDB`)
- Workers `chdir(project_root)` before execution

Do not assume paths from `cache_dir` location—it's passed separately to `execute_stage()`.

## Nested Parallelism

Stages using joblib/scikit-learn with `n_jobs > 1` create nested multiprocessing, causing `resource_tracker` race conditions between Pivot's loky pool and joblib's nested pool.

**Default:** Threading backend (`parallel_config(backend="threading")`). Safe for NumPy/pandas workloads that release the GIL.

**Override:** Set `PIVOT_NESTED_PARALLELISM=processes` to use loky with memmapping disabled.

## Pickling Requirements

- Stage functions must be **module-level** (not lambdas, closures, or `__main__` definitions)
- Output TypedDicts and custom Reader/Writer/Loader instances must also be module-level
- See `docs/solutions/` for loky pickling gotchas
