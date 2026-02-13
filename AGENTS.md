# Pivot - Project Rules

**Python 3.13+ | Unix only | 90%+ coverage | Pre-alpha (breaking changes OK)**

## Project Status

**Pre-alpha** - breaking changes acceptable, no migration code or compatibility shims needed.

## Project Structure

```
packages/pivot/src/pivot/
├── cli/           # Click commands, decorators, helpers (see cli/AGENTS.md)
├── config/        # YAML parsing, validation
├── dag/           # Dependency graph construction
├── engine/        # Execution coordinator, watch mode, RPC server
├── executor/      # Worker process execution (see executor/AGENTS.md)
├── pipeline/      # Pipeline state, lock files
├── remote/        # S3/remote storage
├── storage/       # StateDB, cache management (see storage/AGENTS.md)
├── fingerprint.py # Code hashing, change detection
├── registry.py    # Stage registration, discovery
├── stage_def.py   # Stage definition extraction
├── loaders.py     # File I/O (CSV, JSON, Pickle, etc.)
├── outputs.py     # Out, DirectoryOut, IncrementalOut
└── types.py       # TypedDicts, enums, type aliases

packages/pivot-tui/src/pivot_tui/  # (see pivot_tui/AGENTS.md)
├── client.py          # PivotRpc/PivotClient protocols + TypedDicts
├── rpc_client_impl.py # RpcPivotClient: JSON-RPC 2.0 over Unix socket
├── event_poller.py    # EventPoller: polls events, converts to TUI messages
├── run.py             # PivotApp (Textual app), main TUI logic
├── diff.py            # Diff formatting helpers
├── diff_panels.py     # Input/Output diff panel renderers
├── types.py           # TUI message types, enums
├── widgets/           # Textual widgets (stage list, panels, logs, debug)
├── screens/           # Modal screens (help, history, confirm dialogs)
├── console.py         # Plain-text console output (non-TUI mode)
└── testing/
    └── fake_server.py # FakeRpcServer test double
```

## Key Entry Points

| File | Purpose |
|------|---------|
| `packages/pivot/src/pivot/cli/__init__.py` | CLI entry point (`pivot` command) |
| `packages/pivot/src/pivot/engine/engine.py` | Pipeline execution coordinator |
| `packages/pivot/src/pivot/engine/agent_rpc.py` | JSON-RPC 2.0 server (Unix socket) |
| `packages/pivot/src/pivot/executor/worker.py` | Worker process execution |
| `packages/pivot/src/pivot/storage/state.py` | LMDB state database |
| `packages/pivot/src/pivot/fingerprint.py` | Code change detection |
| `packages/pivot-tui/src/pivot_tui/run.py` | TUI application (Textual) |
| `packages/pivot-tui/src/pivot_tui/rpc_client_impl.py` | TUI's RPC client |
| `packages/pivot/src/pivot/cli/_run_common.py` | 3-thread TUI launch coordinator |

## Institutional Knowledge

**`docs/solutions/`** contains documented learnings. Check before implementing features touching:
- Multiprocessing/loky (pickling, worker lifecycle)
- Type system (TypedDict, generics, Protocol)
- Fingerprinting (what triggers re-runs)
- LMDB/StateDB (prefixes, transactions)
- Path resolution / symlinks (normpath vs resolve, registration vs execution validation)

---

## Core Design

- **Artifact-first**: The DAG emerges from artifact dependencies, not explicit wiring
- **Pipeline is canonical**: The registry (from `pivot.yaml`/`pipeline.py`) is the single source of truth for stage metadata — deps, outs, cache flags, params. Lock files record execution state (hashes, generations), not stage definitions. Never infer stage properties from lock files.
- Per-stage lock files, automatic code fingerprinting, warm worker pools
- `ProcessPoolExecutor` for true parallelism (GIL would serialize threads)
- Invalidation is content-addressed: same inputs + same code = same outputs

## Skip Detection

Three-tier algorithm: generation tracking → lock file comparison → run cache lookup. See `storage/AGENTS.md` for StateDB prefixes and implementation details.

## Fingerprinting Guidelines

Fingerprint checks must be cheap — O(1) shallow/static checks only. No deep recursive field traversal for immutability detection.

**Error policy:**
- **Soundness risk** (changes could go undetected) → error
- **Noise risk** (spurious reruns only) → warn

Aggregate all fingerprinting errors before reporting. Don't fail on the first one. Error messages must name the captured variables/types and suggest using `StageParams` or `Dep`.

Unsafe fingerprinting is opt-in via a single config flag. Default is strict (error).

## Caching Guidelines

The manifest cache (`sm:`) is best-effort for performance. Correctness must never depend on it.

## Architecture Guidelines

Avoid module-level mutable state. Use `contextvars.ContextVar` or explicit parameters for contextual state like "current stage name."

Keep LMDB/DB read transactions short-lived. Materialize generator outputs inside a context manager — don't leave transactions pinned via lazy generators.

Centralize decision logic. The `explain` and `run` paths must use the same skip-decision code — no divergent implementations that can drift.

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

- **Dependencies**: `Annotated[T, Dep(path, loader)]` on parameters (`loader` is `Reader[R]`)
- **Outputs**: `Annotated[T, Out(path, loader)]` in TypedDict return type (`loader` is `Writer[W]`)
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
- **Network I/O:** See `remote/AGENTS.md` for async patterns, aioboto3 session management, and aiohttp usage.

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

**No duplicate imports:** If a module is imported at runtime (`import pathlib`), do NOT also import from it under `TYPE_CHECKING` (`from pathlib import Path`). Use `pathlib.Path` in annotations instead. Each module should appear in exactly one place — either a runtime module import or a `TYPE_CHECKING` import, never both.

## Error Handling

- Validate at boundaries (CLI, file I/O, config), trust internals
- Let errors propagate; catch only where you can handle meaningfully
- Catch specific expected exceptions — avoid blanket `except Exception` unless re-raising or logging at a boundary
- Failed operations should be atomic—return to last known good state
- Before adding defensive changes (locks, guards, extra validation), verify whether existing mechanisms (DAG ordering, lock files, transaction isolation) already guarantee safety. Don't add redundant protection.
- Demote expensive diagnostic checks (`exists()`, stat calls) to debug level. Guard with `logger.isEnabledFor(logging.DEBUG)` to avoid overhead when debug logging is off.

## Simplicity Over Abstraction

- Don't create thin wrappers—inline single-use library calls
- Don't over-modularize—inline modules with one public function
- Three similar lines > premature abstraction

---

## Path Handling

All paths in lockfiles must be **relative** (to stage cwd), never absolute.

## Development

```bash
uv sync --active                                                                      # Install deps
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto                  # Test
uv run ruff format . && uv run ruff check . && uv run basedpyright                    # Quality
```

**Run all quality checks before returning to user or pushing.**
