# Pivot Executor - Development Guidelines

Workers execute in separate processes via `loky.get_reusable_executor()`.

## WorkerStageInfo Contract

`WorkerStageInfo` (TypedDict at `worker.py`) is the coordinator-to-worker contract:

| Field | Purpose |
|-------|---------|
| `func` | Stage function (must be picklable) |
| `fingerprint` | Code manifest for change detection |
| `deps` | Input dependency paths |
| `outs` | Output specs (`BaseOut` instances) |
| `project_root` | Absolute path to project root |
| `state_dir` | Absolute path to `.pivot/` directory |

## Path Derivation

Workers derive all paths from `project_root` and `state_dir` in `WorkerStageInfo`:
- `state_db_path = stage_info["state_dir"] / "state.db"`
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
