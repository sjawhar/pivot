# Pivot - Project Rules

**Python 3.13+ | Unix only | 90%+ coverage | Pre-alpha (breaking changes OK)**

## Project Status

**Pre-alpha** - breaking changes acceptable, no migration code or compatibility shims needed.

## Project Structure

```
packages/pivot/src/pivot/
├── cli/           # Click commands, decorators, helpers (see cli/AGENTS.md)
├── config/        # YAML parsing, validation
├── dag/           # DAG rendering for `pivot dag` (ASCII, DOT, Mermaid)
├── engine/        # Execution coordinator, watch mode, RPC server
├── executor/      # Worker process execution (see executor/AGENTS.md)
├── remote/        # S3 and GCS remote cache storage (see remote/AGENTS.md)
├── storage/       # StateDB, lock files, cache management (see storage/AGENTS.md)
├── compose.py     # @stage, Pipeline, ArtifactHandle, metric/plot/incremental/directory tags, format inference
├── discovery.py   # Finds and loads pipeline.py
├── fingerprint.py # Code hashing, change detection
├── registry.py    # StageRegistry, RegistryStageInfo (built by Pipeline)
├── stage_def.py   # StageParams, output writing helpers
├── loaders.py     # File I/O (CSV, JSON, Pickle, etc.)
└── types.py       # TypedDicts, enums, type aliases, ArtifactIdentity

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
- **Pipeline is canonical**: The registry (built from `pipeline.py`) is the single source of truth for stage metadata — deps, outs, tags, params. Lock files record execution state (hashes, generations), not stage definitions. Never infer stage properties from lock files.
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

Aggregate all fingerprinting errors before reporting. Don't fail on the first one. Error messages must name the captured variables/types and suggest using `StageParams`.

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
pivot repro demo/train       # Run demo/train AND all its dependencies
pivot repro --watch          # Watch mode - re-run on file changes
pivot repro --dry-run        # Validate DAG without executing

pivot run demo/train         # Run ONLY demo/train (no dependency resolution)
pivot run demo/a demo/b      # Run specific stages in order (no deps)
```

**Key difference:** `repro` resolves dependencies automatically; `run` executes only the named stages.

## Defining Stages and Pipelines

`pipeline.py` defines stages with `@stage` and wires them in a `with Pipeline("name") as pipeline:` block; discovery loads it and reads the module variable `pipeline`. Stage names are `<pipeline>/<function>`.

```python
class TrainParams(StageParams):
    min_score: float = 0.5


@stage
def train(data: pd.DataFrame, params: TrainParams) -> pd.DataFrame:
    return data[data["score"] >= params.min_score]


with Pipeline("demo") as pipeline:
    train(pipeline.input("input.csv"), params=TrainParams())
```

- **Inputs**: `pipeline.input(name)` reads `data/raw/<name>`; format from the extension (`compose.py` `_infer_format_from_extension`)
- **Dependencies**: stage parameters that receive handles (or lists/tuples of handles) at the call site; no annotations on parameters. `handle["train.csv"]` on a `directory` output depends on that one file
- **Outputs**: the return type; format inferred (`DataFrame`→JSONL, `dict`/`list`→YAML, `str`→text) or set with `Annotated[T, loader]`; `TypedDict` for several outputs; `metric`/`plot`/`incremental` tags; `Annotated[dict[str, T], directory]` for a variable set of files. Pivot chooses paths (`data/<pipeline>/<stage>.<ext>`, a directory `data/<pipeline>/<stage>/`)
- **Parameters**: one `StageParams` subclass per stage, defined beside it; `params.yaml` overrides by stage name
- **Variants**: Python loops, `pipeline.variant("name")`; composition: imported handles, `Pipeline.include()`
- Stages must be **pure, module-level functions** (workers receive them by pickling); config belongs in `StageParams`, not module-level collections

User-facing guide: `skills/writing-pivot-stages/SKILL.md`.

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

**CI runs the test suite without uv on PATH**, so a test that starts pivot runs `pathlib.Path(sys.executable).parent / "pivot"`.

**A regression found on main gets a line in `.github/known-regressions`, by PR.** Main then goes red and nothing publishes until the fix's PR deletes the line.
