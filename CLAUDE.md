# Pivot - Project Rules

**Python 3.13+ | Unix only | 90%+ coverage | Pre-alpha (breaking changes OK)**

---

## Project Status

This project is **pre-alpha**. No backwards compatibility is needed.

- **Breaking changes are acceptable** — don't add migration code or compatibility shims
- **No legacy format support** — only support the current version
- **Validate design empirically** — test decisions on real workloads, not just theory

---

## Core Design

- Per-stage lock files, automatic code fingerprinting, warm worker pools
- `ProcessPoolExecutor` for true parallelism (not threads—GIL would serialize)

## Skip Detection

Two-tier algorithm: (1) O(1) generation tracking in `worker.can_skip_via_generation()`, (2) O(n) hash comparison fallback. Run cache allows skipping via input hash match.

StateDB prefixes: `hash:` (file hashes), `gen:` (output generations), `dep:` (stage dep generations), `runcache:` (run cache).

## Artifact-Centric Mental Model (Critical)

- Think **artifact-first**, not **stage-first**. The DAG emerges from artifact dependencies.
- **Right:** "This file changed. What needs to happen because of that?"
- Invalidation is content-addressed: same inputs + same code = same outputs
- Stage execution order is derived from the artifact graph, not explicit wiring
- Watch mode must distinguish external changes (trigger re-run) from stage outputs (don't trigger)

## Stage Registration (Critical)

Two pipeline definition methods:

1. **`pivot.yaml`** - Config file with `stages:` section pointing to Python functions (most common)
2. **`pipeline.py`** - Python module that calls `REGISTRY.register()` directly

**Discovery order:** `pivot.yaml` → `pivot.yml` → `pipeline.py`

- Stages must be **pure, serializable, module-level functions** for multiprocessing:
- Workers receive pickled functions—lambdas, closures, and `__main__` definitions fail.
- Stage functions, output TypedDicts, and custom loaders must be **module-level

```python
def train(
    params: TrainParams,
    data: Annotated[DataFrame, Dep("input.csv", CSV())],
) -> Annotated[DataFrame, Out("output.csv", CSV())]:
    return data.dropna()
```

** (required for type hint resolution and pickling). Loader code is fingerprinted—changes trigger re-runs.

- **Dependencies**: `param: Annotated[T, Dep(path, loader)]` on function parameters
- **Outputs**: `field: Annotated[T, Out(path, loader)]` in TypedDict return type
- **Parameters**: `params: MyParams` where `MyParams` extends `StageParams`
- config belongs in code, not YAML.** Use Pydantic classes for configuration, not `params.yaml`.
- This enables type checking, IDE support, and change detection through fingerprinting.

## Code Quality

- Type hints everywhere; `ruff format` (100 chars); `ruff check`
- `_prefix` for private functions; import modules not functions
- NEVER modify rules in `pyproject.toml` without permission
- Zero tolerance for basedpyright warnings—resolve all errors AND warnings
- No blanket `# pyright: reportFoo=false`—use targeted ignores with specific codes:
- Prefer type stubs (`pandas-stubs`, `types-PyYAML`) over ignores
- Empty collections: `list[int]()` not `: list[int] = []`
- Simplified Generator: `Generator[int]` not `Generator[int, None, None]`
- `Callable` over `Any` for functions; document why when using `Any`
- Prefer better code over comments. Add comments only for non-obvious WHY, timing constraints, or known limitations. Never comment obvious WHAT.
- No module-level docstrings. Simple functions get one-line docstrings—skip Args/Returns if type hints make it obvious.
- Write evergreen docs—avoid "recently added" or "as of version X".
- **Early returns:** Keep main logic at top indentation; avoid pyramid of doom
- **Match statements:** Prefer over if/elif for enum dispatch and type discrimination
- **Enums over Literals:** For programmatic values (catches typos at type-check time)

## TypedDict

- Zero runtime overhead, native JSON serialization. Use over dataclasses (need `asdict()`) or namedtuples (serialize as arrays).
- Never use `.get()`—direct access only. For optional fields: `if "key" in d: d["key"]`
- Always use constructor syntax: `return Result(status="ok")` not `{"status": "ok"}`

## Pydantic

- Use for data needing validation with clear errors (config files, user input, API boundaries).
- Avoid in hot paths—use TypedDict there.
- **Config belongs in code, not YAML.** Use Pydantic classes for configuration, not `params.yaml`. This enables type checking, IDE support, and change detection through fingerprinting.

## Path Handling

All paths in lockfiles must be **relative** (to stage cwd), never absolute. This ensures portability and correct cache behavior.

## Import Style

- Import modules, not functions: `from pivot import fingerprint` then `fingerprint.func()`.
- **No lazy imports**—all imports at module level. This ensures fingerprinting captures dependencies and makes imports explicit.

**Exceptions:**
- `TYPE_CHECKING` blocks: Import types directly (`from pathlib import Path`)
- `pivot.types`: Import directly (`from pivot.types import StageStatus`)
- `typing` module: Always direct (`from typing import Any`)
- Optional/platform-specific modules: Lazy import with try/except when module may not exist (e.g., `resource` on Windows)
- CLI modules: Lazy imports acceptable in `pivot.cli` to reduce startup time

## Error Handling Philosophy

- **Validate boundaries, trust internals.** Validate aggressively at entry points (CLI, file I/O, config parsing). Once validated, trust data downstream — no redundant internal validation.
- Let errors propagate—catch at boundaries where you can handle meaningfully. Silent failures are worse than crashes.

**When to suppress vs propagate:**
| Condition | Action |
|-----------|--------|
| Unknown/invalid state | Propagate — fail fast |
| Invariant violation | Propagate — this is a bug |
| Cache miss, optional feature | Log and continue with fallback |
| Resource exhaustion (queue full) | Propagate — architectural issue |

**Failed operations should be atomic** — return to last known good state. If a stage fails, don't update the lockfile.

- **Validate upfront, validate early.** Check all preconditions before performing any side effects. This ensures operations are atomic—either all succeed or none do.
- Fail fast with clear errors—don't silently fix or skip invalid inputs.
- Never silently ignore validation failures (typos in field names, type mismatches, missing required data).

## Simplicity Over Abstraction

- **Don't create thin wrapper functions** — if it just calls one library function, inline it
- **Don't over-modularize** — a module with one public function used by one other module should be inlined
- **Don't add options without justification** — if you can't articulate when each option would be used, you don't need options
- **Three similar lines > premature abstraction** — wait until the pattern is clear before extracting

## Development

```bash
uv sync --active       # Install deps
uv run pytest tests/ -n auto  # Test
uv run ruff format . && uv run ruff check . && uv run basedpyright .  # Quality
```

- Before Returning to User or pushing (Critical), Run all quality checks

## Critical Discoveries

1. **Single underscore functions ARE tracked**—only dunders (`__name__`) filtered
2. **Circular imports:** Extract shared types to separate module
3. **AST manipulation:** Function bodies need at least one statement—add `ast.Pass()` if empty
4. **Path overlap detection:** Use pygtrie, not string matching (`data/` vs `data/file.csv`)
5. **loky can't pickle `mp.Queue()`**—use `mp.Manager().Queue()`
6. **Reusable executor:** `loky.get_reusable_executor()` keeps workers warm
7. **Cross-process tests:** Use file-based state, not shared lists (each process copies)
8. **Atomic writes:** Track fd closure when using `mkstemp()` + rename
9. **IncrementalOut uses COPY mode**—hardlinks/symlinks would corrupt cache
10. **StateDB path strategies:** `resolve()` for hash keys (dedup), `normpath()` for generation keys (logical paths)
11. **LMDB for all state:** Extend StateDB with prefixes, don't add new databases
12. **ruamel.yaml for editable config** (preserves comments), **PyYAML for read-only**
13. **Stage functions and TypedDicts must be module-level**—`get_type_hints()` needs importable `__module__`
14. **Lambda fingerprinting is non-deterministic**—lambdas without source fall back to `id(func)`, causing unnecessary re-runs across interpreter sessions. Always use named functions in stage definitions.