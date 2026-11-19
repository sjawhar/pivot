# Pivot Executor

Runs stages in worker processes from loky's reusable executor; `WorkerPool`
(`engine/worker_pool.py`) owns the pool for an Engine's lifetime. Pool sizing, idle timeouts,
warm workers and stopping mid-stage: `docs/architecture/execution.md`.

Tests: `uv run pytest packages/pivot/tests/execution -n auto`.

## Contracts

- `WorkerStageInfo` (`worker.py`) is everything the coordinator sends a worker, and all of it is
  pickled: stage functions, output TypedDicts and loader instances are module-level, and queues
  are manager queues, not `multiprocessing.Queue()` (`docs/contributing/gotchas.md`,
  Multiprocessing).
- A worker takes its paths from `WorkerStageInfo`: it changes into `project_root` and opens
  `state_dir`. `cache_dir` is a separate `execute_stage()` argument; derive no path from another.
- Workers open StateDB read-only and return `DeferredWrites` for the coordinator to apply; the one direct worker
  write is the `*.pvt` member record (`storage/AGENTS.md`, StateDB).

## Worker lifecycle

- Stop a worker with `worker.terminate`, which sends SIGTERM to its main thread (tgkill), not
  with `os.kill`: the kernel can give a process signal to another thread, such as a NumPy
  OpenBLAS thread, and the stage then never stops.
- Ask loky for an executor only through `core.py` (`create_executor`, `create_fresh_executor`),
  and replace a pool's executor only through `WorkerPool._replace_executor`, which stops the held
  executor first. loky 3.7 keeps one reusable executor per thread, and a watch reload restarts
  from an anyio worker thread.
- Keep `reuse=False if kill_workers else "auto"`. loky replaces a live pool whose creation
  arguments differ; `core._init_worker`'s arguments carry the digests of the loaded user code,
  so a worker never runs an edited module's old code.
- `core.py` reads loky internals (`_executor` on 3.5.6 and 3.6, `_executor_storage` on 3.7,
  `_max_workers`, `_flags`, `queue_size`). Check them against a new loky release before
  widening `loky>=3.5.6,<3.8` in `packages/pivot/pyproject.toml`.

## Nested parallelism

Stages run under joblib's threading backend, so a stage's `n_jobs > 1` does not start a second
process pool that races Pivot's over `resource_tracker`. `PIVOT_NESTED_PARALLELISM=processes`
selects loky with memmapping off (`worker._execute_with_joblib_protection`).
