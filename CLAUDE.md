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

- Per-stage lock files (32x faster than DVC), automatic code fingerprinting, warm worker pools
- `ProcessPoolExecutor` for true parallelism (not threads—GIL would serialize)
- TDD: write tests BEFORE implementation

## Artifact-Centric Mental Model (Critical)

Think **artifact-first**, not **stage-first**. The DAG emerges from artifact dependencies.

- **Wrong:** "Stage A triggers Stage B"
- **Right:** "This file changed. What needs to happen because of that?"

**Implications:**
- Invalidation is content-addressed: same inputs + same code = same outputs
- Stage execution order is derived from the artifact graph, not explicit wiring
- Watch mode must distinguish external changes (trigger re-run) from stage outputs (don't trigger)
- Outputs can also be inputs to downstream stages — the full dependency graph matters

## Stage Registration (Critical)

Three pipeline definition methods—all must be handled in registry/discovery code:

1. **`pivot.yaml`** - Config file with `stages:` section (most common)
2. **`pipeline.py`** - Python script using `@stage` decorators, executed via `runpy.run_path()`
3. **`@stage` decorators** in importable modules - Registered on import

**Discovery order:** `pivot.yaml` → `pivot.yml` → `pipeline.py` → decorator modules

## Stage Functions (Critical)

Must be **pure, serializable, module-level functions** for multiprocessing:

```python
# Good - module-level, no captured variables
@stage(deps=['data.csv'], outs=['output.csv'])
def process_data(): ...

# Bad - closure captures variable (not picklable)
def make_stage(threshold):
    @stage(deps=['x'], outs=['y'])
    def process():
        if value > threshold: ...  # Captures threshold!
    return process
```

Workers receive pickled functions—lambdas, closures, and `__main__` definitions fail.

## StageDef Conventions

StageDef and custom loaders must be **module-level** (required for type hint resolution and pickling). Loader code is fingerprinted—changes trigger re-runs. YAML `deps`/`outs` completely replace StageDef defaults.

## Code Quality

- Type hints everywhere; `ruff format` (100 chars); `ruff check`
- One-line docstrings; comments explain WHY not WHAT
- `_prefix` for private functions; import modules not functions
- No circular dependencies

## Linting/Types (Critical)

- NEVER modify rules in `pyproject.toml` without permission
- Zero tolerance for basedpyright warnings—resolve all errors AND warnings
- No blanket `# pyright: reportFoo=false`—use targeted ignores with specific codes:
  ```python
  return json.load(f)  # type: ignore[return-value] - json returns Any
  ```
- Prefer type stubs (`pandas-stubs`, `types-PyYAML`) over ignores

## Python 3.13+ Types

- Empty collections: `list[int]()` not `: list[int] = []`
- Simplified Generator: `Generator[int]` not `Generator[int, None, None]`
- `Callable` over `Any` for functions; document why when using `Any`

## TypedDict

Zero runtime overhead, native JSON serialization. Use over dataclasses (need `asdict()`) or namedtuples (serialize as arrays).

- Never use `.get()`—direct access only. For optional fields: `if "key" in d: d["key"]`
- Always use constructor syntax: `return Result(status="ok")` not `{"status": "ok"}`

## Pydantic

Use for data needing validation with clear errors (config files, user input, API boundaries). Avoid in hot paths—use TypedDict there.

**Config belongs in code, not YAML.** Use Pydantic classes for configuration, not `params.yaml`. This enables type checking, IDE support, and change detection through fingerprinting.

## Path Handling

All paths in lockfiles must be **relative** (to stage cwd), never absolute. This ensures portability and correct cache behavior.

## Import Style

Import modules, not functions: `from pivot import fingerprint` then `fingerprint.func()`.

**No lazy imports**—all imports at module level. This ensures fingerprinting captures dependencies and makes imports explicit.

**Exceptions:**
- `TYPE_CHECKING` blocks: Import types directly (`from pathlib import Path`)
- `pivot.types`: Import directly (`from pivot.types import StageStatus`)
- `typing` module: Always direct (`from typing import Any`)
- Optional/platform-specific modules: Lazy import with try/except when module may not exist (e.g., `resource` on Windows)
- CLI modules: Lazy imports acceptable in `pivot.cli` to reduce startup time

## Docstrings

No module-level docstrings. Simple functions get one-line docstrings—skip Args/Returns if type hints make it obvious.

```python
# Good
def resolve_path(path: str) -> pathlib.Path:
    """Resolve relative path from project root; absolute paths unchanged."""

# Bad - repeats type hints
def resolve_path(path: str) -> pathlib.Path:
    """Resolve path relative to project root.

    Args:
        path: File path (relative or absolute)
    Returns:
        Resolved absolute path
    """
```

## Comments

Prefer better code over comments. Add comments only for non-obvious WHY, timing constraints, or known limitations. Never comment obvious WHAT (`# Add node` before `graph.add_node()`).

Write evergreen docs—avoid "recently added" or "as of version X".

## Code Patterns

- **Early returns:** Keep main logic at top indentation; avoid pyramid of doom
- **Match statements:** Prefer over if/elif for enum dispatch and type discrimination
- **Private functions:** `_prefix` for module-internal helpers
- **Enums over Literals:** For programmatic values (catches typos at type-check time)

## Error Handling Philosophy

**Validate boundaries, trust internals.** Validate aggressively at entry points (CLI, file I/O, config parsing). Once validated, trust data downstream — no redundant internal validation.

Let errors propagate—catch at boundaries where you can handle meaningfully. Silent failures are worse than crashes.

```python
# Good - propagate, catch at CLI
def run_pipeline(stages):
    return execute(build_dag(stages))  # May raise

# CLI catches
except StageNotFoundError as e:
    click.echo(f"Error: {e}", err=True)
```

**When to suppress vs propagate:**
| Condition | Action |
|-----------|--------|
| Unknown/invalid state | Propagate — fail fast |
| Invariant violation | Propagate — this is a bug |
| Cache miss, optional feature | Log and continue with fallback |
| Resource exhaustion (queue full) | Propagate — architectural issue |

**Failed operations should be atomic** — return to last known good state. If a stage fails, don't update the lockfile.

## Input Validation

Validate at boundaries, then trust downstream. Fail fast with clear errors—don't silently fix or skip invalid inputs.

## Simplicity Over Abstraction

- **Don't create thin wrapper functions** — if it just calls one library function, inline it
- **Don't over-modularize** — a module with one public function used by one other module should be inlined
- **Don't add options without justification** — if you can't articulate when each option would be used, you don't need options
- **Three similar lines > premature abstraction** — wait until the pattern is clear before extracting
- **No nested functions** — use module-level for testability and fingerprinting

## CLI

**Use `@cli_decorators.pivot_command()`** instead of `@click.command()`. Provides auto-discovery and error handling. Use `auto_discover=False` for commands not using registry (`init`, `schema`, `push`, `pull`).

**Shell completion required:** Use `shell_complete=completion.complete_stages` for stage args, `complete_targets` for file/stage args.

**Explicit output:** Always show messages for empty states. JSON output must include all requested keys (empty arrays, not omitted).

See `src/pivot/cli/CLAUDE.md` for detailed CLI guidelines.

## Development

```bash
uv sync --active       # Install deps
pytest tests/ -n auto  # Test
ruff format . && ruff check . && basedpyright .  # Quality
```

## Before Returning to User (Critical)

Run all four: `ruff format .`, `ruff check .`, `basedpyright .`, `pytest tests/ -n auto`

## Before Pushing (Critical)

`uv run ruff format . && uv run ruff check . && uv run basedpyright . && uv run pytest tests/ -n auto`

## Testing Philosophy

**Real assurance over passing tests.** Tests should prove correctness, not just exercise code paths. Prefer real execution over mocks — mock external boundaries (network, filesystem in unit tests), not internal logic.

See `tests/CLAUDE.md` for detailed test guidelines.

## Mental Model Mistakes to Avoid

These are recurring patterns that lead to corrections:

1. **Over-engineering** — Adding validation modes, config options, or abstractions for hypothetical future needs. If you can't articulate when each option would be used, you don't need options.

2. **Defensive over-programming** — Try/catch that returns a default on both paths; "safe" wrappers that swallow errors and hide bugs. Let errors surface.

3. **Import hacks over structure** — Using lazy imports or sys.path to fix circular dependencies instead of restructuring. Extract shared types to a separate module.

4. **Type safety regression** — Defaulting to `Any` when typing gets hard. Find the correct type or use TypeVar. `Any` is a last resort.

5. **Stage-centric thinking** — Thinking "Stage A triggers Stage B" instead of "this artifact changed, what depends on it?" See Artifact-Centric Mental Model above.

## Critical Discoveries

1. **Test helpers must be module-level**—`getclosurevars()` doesn't see imports in inline closures
2. **Single underscore functions ARE tracked**—only dunders (`__name__`) filtered
3. **Circular imports:** Extract shared types to separate module
4. **AST manipulation:** Function bodies need at least one statement—add `ast.Pass()` if empty
5. **Path overlap detection:** Use pygtrie, not string matching (`data/` vs `data/file.csv`)
6. **loky can't pickle `mp.Queue()`**—use `mp.Manager().Queue()`
7. **Reusable executor:** `loky.get_reusable_executor()` keeps workers warm
8. **Cross-process tests:** Use file-based state, not shared lists (each process copies)
9. **Atomic writes:** Track fd closure when using `mkstemp()` + rename
10. **IncrementalOut uses COPY mode**—hardlinks/symlinks would corrupt cache
11. **StateDB path strategies:** `resolve()` for hash keys (dedup), `normpath()` for generation keys (logical paths)
12. **LMDB for all state:** Extend StateDB with prefixes, don't add new databases
13. **ruamel.yaml for editable config** (preserves comments), **PyYAML for read-only**
14. **StageDef must be module-level**—`get_type_hints()` needs importable `__module__`
