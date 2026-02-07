# Pivot - Project Rules

**Python 3.13+ | Unix only | 90%+ coverage | Pre-alpha (breaking changes OK)**

## Project Status

**Pre-alpha** - breaking changes acceptable, no migration code or compatibility shims needed.

## Project Structure

```
src/pivot/
├── cli/           # Click commands, decorators, helpers
├── config/        # YAML parsing, validation
├── dag/           # Dependency graph construction
├── engine/        # Execution coordinator, watch mode
├── executor/      # Worker process execution (see executor/CLAUDE.md)
├── pipeline/      # Pipeline state, lock files
├── remote/        # S3/remote storage
├── storage/       # StateDB, cache management (see storage/CLAUDE.md)
├── tui/           # Textual-based terminal UI
├── fingerprint.py # Code hashing, change detection
├── registry.py    # Stage registration, discovery
├── stage_def.py   # Stage definition extraction
├── loaders.py     # File I/O (CSV, JSON, Pickle, etc.)
├── outputs.py     # Out, DirectoryOut, IncrementalOut
└── types.py       # TypedDicts, enums, type aliases
```

## Key Entry Points

| File | Purpose |
|------|---------|
| `src/pivot/cli/__init__.py` | CLI entry point (`pivot` command) |
| `src/pivot/engine/engine.py` | Pipeline execution coordinator |
| `src/pivot/executor/worker.py` | Worker process execution |
| `src/pivot/storage/state_db.py` | LMDB state database |
| `src/pivot/fingerprint.py` | Code change detection |

## Institutional Knowledge

**`docs/solutions/`** contains documented learnings. Check before implementing features touching:
- Multiprocessing/loky (pickling, worker lifecycle)
- Type system (TypedDict, generics, Protocol)
- Fingerprinting (what triggers re-runs)
- LMDB/StateDB (prefixes, transactions)

---

## Core Design

- **Artifact-first**: The DAG emerges from artifact dependencies, not explicit wiring
- **Pipeline is canonical**: The registry (from `pivot.yaml`/`pipeline.py`) is the single source of truth for stage metadata — deps, outs, cache flags, params. Lock files record execution state (hashes, generations), not stage definitions. Never infer stage properties from lock files.
- Per-stage lock files, automatic code fingerprinting, warm worker pools
- `ProcessPoolExecutor` for true parallelism (GIL would serialize threads)
- Invalidation is content-addressed: same inputs + same code = same outputs

## Skip Detection

Three-tier algorithm: (1) O(1) generation tracking in `worker.can_skip_via_generation()`, (2) O(n) lock file comparison (fingerprint + params + dep hashes), (3) run cache lookup via input hash match.

StateDB prefixes: `hash:` (file hashes), `gen:` (output generations), `dep:` (stage dep generations), `runcache:` (run cache).

## Worker Execution

Workers execute in separate processes via `loky.get_reusable_executor()`.

### WorkerStageInfo Contract

`WorkerStageInfo` (TypedDict at `src/pivot/executor/worker.py`) is the coordinator-to-worker contract. Key fields (non-exhaustive):

| Field | Purpose |
|-------|---------|
| `func` | The stage function (must be picklable) |
| `fingerprint` | Code manifest for change detection |
| `deps` | Input dependency paths |
| `outs` | Output specs (`BaseOut` instances) |
| `project_root` | Absolute path to project root |
| `state_dir` | Absolute path to `.pivot/` directory |

### Path Derivation

Workers derive all paths from `project_root` and `state_dir` explicitly passed in `WorkerStageInfo`:
- `state_db_path = stage_info["state_dir"] / "state.db"`
- Workers `chdir(project_root)` before execution

Do not assume paths from `cache_dir` location—it's passed separately to `execute_stage()`.

### Nested Parallelism

Stages using joblib/scikit-learn with `n_jobs > 1` create nested multiprocessing, causing `resource_tracker` race conditions between Pivot's loky pool and joblib's nested pool.

**Default behavior:** Threading backend (`parallel_config(backend="threading")`). Safe for NumPy/pandas workloads that release the GIL.

**Override:** Set `PIVOT_NESTED_PARALLELISM=processes` to use loky with memmapping disabled.

## Caching Principle

**Files are cached individually, not directories.** Each output file is hashed and stored in cache by its content hash. The stage lockfile contains a manifest listing all output files with their hashes. This enables fine-grained cache hits even when only some files change.

## Artifact-Centric Mental Model (Critical)

- Think **artifact-first**, not **stage-first**. The DAG emerges from artifact dependencies.
- **Right:** "This file changed. What needs to happen because of that?"

## Running Stages

```bash
pivot repro                  # Run entire pipeline (DAG-aware)
pivot repro my_stage         # Run my_stage AND all its dependencies
pivot repro --watch          # Watch mode - re-run on file changes
pivot repro --dry-run        # Validate DAG without executing

pivot run my_stage           # Run ONLY my_stage (no dependency resolution)
pivot run stage1 stage2      # Run specific stages in order (no deps)
```

**Key difference:** `repro` resolves dependencies automatically; `run` executes only the named stages.

## Stage Registration

Two methods: `pivot.yaml` (config pointing to functions) or `pipeline.py` (direct `pipeline.register()`).

**Discovery order:** `pivot.yaml` → `pivot.yml` → `pipeline.py`

```python
def train(
    params: TrainParams,
    data: Annotated[DataFrame, Dep("input.csv", CSV())],
) -> Annotated[DataFrame, Out("output.csv", CSV())]:
    return data.dropna()
```

- **Dependencies**: `Annotated[T, Dep(path, reader)]` on parameters (reader is `Reader[R]`)
- **Outputs**: `Annotated[T, Out(path, writer)]` in TypedDict return type (writer is `Writer[W]`)
- **Incremental Outputs**: `Annotated[T, IncrementalOut(path, loader)]` for bidirectional (loader is `Loader[W, R]`)
- **Parameters**: `params: MyParams` where `MyParams` extends `StageParams`
- Stages must be **pure, serializable, module-level functions** (lambdas/closures fail pickling)
- Config belongs in Pydantic classes, not YAML files

---

## Code Quality

- Type hints everywhere; `ruff format` (100 chars); `ruff check`
- `_prefix` for private functions; import modules not functions
- Zero tolerance for basedpyright warnings—use targeted ignores: `# type: ignore[code] - reason`
- Prefer type stubs (`pandas-stubs`, `types-PyYAML`) over ignores
- Check `typings/` first for untyped packages; use `scripts/generate_stubs.py` if needed
- **aioboto3/S3 types:** Use `types_aiobotocore_s3.S3Client` (not `Any`) for S3 client parameters. Import under `TYPE_CHECKING`. The `types-aioboto3[all]` stubs are in dev deps.
- **aioboto3 sessions:** Create one `aioboto3.Session()` per `S3Remote` instance (in `__init__`), not per method call. Sessions are credential-management objects — recreating them re-reads credential chains and env vars. Each method creates its own client via `async with self._session.client("s3")`, which is lightweight but gets a fresh connection pool. Batch methods share one client across concurrent tasks via `asyncio.gather`.

## Python 3.13+ Types

- Empty collections: `list[int]()` not `: list[int] = []`
- Simplified Generator: `Generator[int]` not `Generator[int, None, None]`
- **Enums over Literals** for programmatic values (catches typos at type-check time)

## TypedDict

- Use over dataclasses/namedtuples (zero overhead, native JSON)
- Direct access only, never `.get()`. For optional: `if "key" in d: d["key"]`
- Constructor syntax: `Result(status="ok")` not `{"status": "ok"}`

## Import Style

Import modules, not functions: `from pivot import fingerprint` then `fingerprint.func()`.

**Exceptions:** `TYPE_CHECKING` blocks, `pivot.types`, `typing` module, CLI lazy imports.

## Error Handling

- Validate at boundaries (CLI, file I/O, config), trust internals
- Let errors propagate; catch only where you can handle meaningfully
- Failed operations should be atomic—return to last known good state

## Simplicity Over Abstraction

- Don't create thin wrappers—inline single-use library calls
- Don't over-modularize—inline modules with one public function
- Three similar lines > premature abstraction

---

## Path Handling

All paths in lockfiles must be **relative** (to stage cwd), never absolute.

## Development

```bash
uv sync --active                                                    # Install deps
uv run pytest tests/ -n auto                                        # Test
uv run ruff format . && uv run ruff check . && uv run basedpyright  # Quality
```

**Run all quality checks before returning to user or pushing.**
