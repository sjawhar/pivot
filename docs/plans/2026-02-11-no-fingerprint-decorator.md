# `@pivot.no_fingerprint()` Decorator + Public API Exports

**Date:** 2026-02-11
**Status:** Ready for implementation

## Problem

Pivot's AST fingerprinting is the most complex part of the system. It walks function closures, hashes transitive dependencies, detects mutable captures, and caches results across multiple tiers. When it works, it's powerful — code changes trigger re-runs automatically. When it has bugs, it's the hardest thing to debug.

A colleague testing the Pivot migration found a "change dependencies, doesn't repro" bug on their first attempt. With the primary author leaving, the team needs a safe fallback that preserves Pivot's benefits (auto-parallelization, warm workers, DAG resolution) while using simpler, more predictable change detection.

## Solution

A `@pivot.no_fingerprint()` decorator that replaces AST-based closure walking with simple `.py` file hashing. Per-stage opt-in, co-located with the stage code.

## API

```python
import pivot

# Disable AST fingerprinting, use file-level hashing only
@pivot.no_fingerprint()
def train(
    data: Annotated[DataFrame, Dep("input.csv", CSV())],
) -> Annotated[DataFrame, Out("output.csv", CSV())]:
    return data.dropna()

# Declare additional code dependency files
@pivot.no_fingerprint(code_deps=["src/utils.py", "src/model_helpers.py"])
def train(
    data: Annotated[DataFrame, Dep("input.csv", CSV())],
) -> Annotated[DataFrame, Out("output.csv", CSV())]:
    return data.dropna()
```

## Behavior

- Hashes the `.py` file containing the stage function (via `inspect.getfile(func)`)
- Hashes any additional files listed in `code_deps` (resolved relative to project root)
- Uses existing `cache.hash_file()` with mtime caching via StateDB — no new hashing code
- Produces a `code_manifest` like `{"file:stages/train.py": "abc123", "file:src/utils.py": "def456"}`
- All three tiers of skip detection work identically — they compare dict contents, not key formats
- Decorator always wins regardless of any global config

### What triggers a re-run

| Change | With AST fingerprint | With `@no_fingerprint` |
|--------|---------------------|----------------------|
| Input data file changed | Re-run | Re-run |
| Parameters changed | Re-run | Re-run |
| Stage source file changed | Re-run | Re-run |
| Helper function in same file changed | Re-run | Re-run |
| Helper function in different file changed | Re-run | Re-run only if listed in `code_deps` |
| Comment/whitespace in stage file changed | Skip (AST ignores) | Re-run (file hash changes) |
| Unrelated code in stage file changed | Re-run | Re-run |

### Trade-offs vs AST fingerprinting

**Advantages:**
- No closure walking, no AST parsing — eliminates the entire class of fingerprinting bugs
- Predictable: "if I edit this file, the stage re-runs"
- Faster: file hash with mtime cache is O(1), vs AST analysis
- Simple to reason about and debug

**Disadvantages:**
- Doesn't detect changes in imported helper files (unless listed in `code_deps`)
- Comment/whitespace changes trigger unnecessary re-runs
- Manual maintenance of `code_deps` list

## Implementation

### Files changed

| File | Change | Lines |
|------|--------|-------|
| `pivot/__init__.py` | Export `no_fingerprint`, `Pipeline`, `StageParams`, `DirectoryOut` | ~8 |
| `pivot/decorators.py` (new) | Decorator implementation | ~15 |
| `pivot/registry.py` | Branch in `_compute_fingerprint()`, new `_compute_file_fingerprint()` | ~20 |
| `tests/test_no_fingerprint.py` (new) | Integration tests | ~150 |

### No changes to

- `worker.py` — skip detection compares dicts, doesn't care about key format
- `lock.py` — stores whatever dict it gets
- `engine.py` — calls `ensure_fingerprint()` the same way
- `types.py` — `LockData.code_manifest` is already `dict[str, str]`
- StateDB — generation tracking works the same

### Decorator (`pivot/decorators.py`)

```python
def no_fingerprint(
    code_deps: list[str] | None = None,
) -> Callable[[F], F]:
    """Disable AST fingerprinting for a stage. Use file-level hashing instead."""
    def decorator(func: F) -> F:
        func.__pivot_no_fingerprint__ = True
        func.__pivot_code_deps__ = code_deps or []
        return func
    return decorator
```

### Fingerprint computation (`pivot/registry.py`)

In `_compute_fingerprint()`, add a branch:

```python
def _compute_fingerprint(stage_name: str, info: RegistryStageInfo) -> dict[str, str]:
    func = info["func"]
    if getattr(func, "__pivot_no_fingerprint__", False):
        return _compute_file_fingerprint(func)
    # ... existing AST fingerprinting unchanged ...
```

New helper:

```python
def _compute_file_fingerprint(func: Callable[..., Any]) -> dict[str, str]:
    """Compute file-hash fingerprint (no AST analysis)."""
    result: dict[str, str] = {}

    # Hash the source file containing the stage function
    source_file = pathlib.Path(inspect.getfile(func))
    file_hash, _ = cache.hash_file(source_file)
    rel_path = str(project.relative_path(source_file))
    result[f"file:{rel_path}"] = file_hash

    # Hash additional code_deps
    code_deps: list[str] = getattr(func, "__pivot_code_deps__", [])
    root = project.get_project_root()
    for dep_path in code_deps:
        abs_path = root / dep_path
        dep_hash, _ = cache.hash_file(abs_path)
        result[f"file:{dep_path}"] = dep_hash

    return result
```

## Public API Exports

Alongside `no_fingerprint`, expose the key pipeline-authoring classes from the top-level `pivot` package so users don't need deep imports.

### New exports

| Export | Source | Purpose |
|--------|--------|---------|
| `Pipeline` | `pivot.pipeline.pipeline` | Main pipeline registration class |
| `StageParams` | `pivot.stage_def` | Base class for stage parameters |
| `DirectoryOut` | `pivot.outputs` | Directory output spec |
| `no_fingerprint` | `pivot.decorators` | Disable AST fingerprinting decorator |

### Already exported

`Out`, `Dep`, `Metric`, `Plot`, `IncrementalOut`, `PlaceholderDep`, `loaders`, `stage_def`

### After

```python
# Users can write:
from pivot import Pipeline, StageParams, Out, Dep, DirectoryOut, no_fingerprint

# Instead of:
from pivot.pipeline.pipeline import Pipeline
from pivot.stage_def import StageParams
from pivot.outputs import DirectoryOut
```

### Implementation

Add entries to both `TYPE_CHECKING` block and `_LAZY_IMPORTS` dict in `pivot/__init__.py`, following the existing lazy-import pattern (~8 lines).

## Tests (integration, `execute_stage` level)

Following the pattern in `test_skip_detection_integration.py`:

1. **Skip when unchanged**: Stage with `@no_fingerprint()` runs, second run skips
2. **Re-run on source file change**: Compute file-hash manifest, modify `.py` file, recompute — hashes differ, stage re-runs
3. **Re-run on code_deps change**: Same pattern with a `code_deps` file
4. **No re-run on unrelated change**: Modify an unrelated file — skip still works
5. **Mixed pipeline**: One fingerprinted stage + one `@no_fingerprint` stage — both behave correctly
6. **Missing code_deps file**: Clear error message

## Design decisions

1. **Per-stage decorator, not global config** — YAGNI. If you want all stages to use file hashing, decorate each one. Avoids a second code path.
2. **Decorator always wins** — No interaction with global config. Simple mental model.
3. **File hash, not "no tracking"** — We still detect code changes (file-level), just without AST analysis. An empty manifest would mean code changes are invisible, which is dangerous.
4. **`code_deps` as explicit list** — Users declare what they depend on. No import parsing, no magic. Predictable.
5. **Same lockfile format** — `code_manifest` is still `dict[str, str]`. Just different keys (`file:` prefix instead of `self:`/`func:`/`class:`).
