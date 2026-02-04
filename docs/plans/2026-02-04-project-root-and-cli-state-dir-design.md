# Project Root Detection and CLI State Directory Fix

**Status:** Ready for review

**Goal:** Fix two related issues preventing cross-pipeline workflows:
1. Project root detection incorrectly treats subdirectory `.pivot` folders as project roots
2. CLI commands use a single global `state_dir` instead of per-pipeline state directories

---

## Background

### Problem 1: Project Root Detection

When running `pivot repro` from a subdirectory pipeline (e.g., `model_reports/time_horizon_1_0/`):

1. First run works — `find_project_root()` finds `.git` at the actual project root
2. Execution creates `.pivot` in the subdirectory for per-pipeline state
3. Second run fails — `find_project_root()` finds the newly created `.pivot` in the subdirectory and treats it as project root
4. Paths like `../data/external/release_dates.yaml` now "escape" this incorrect project root

**Root cause:** `find_project_root()` returns the **first** `.pivot` or `.git` found walking up from cwd. It should return the **topmost** `.pivot`.

### Problem 2: CLI State Directory Usage

CLI commands (`checkout`, `verify`, `remote push/pull`, `data get`) call `config.get_state_dir()` once and use it for all stages. But:

- Stages from different pipelines have different `state_dir` values (stored in `RegistryStageInfo["state_dir"]`)
- After `build_dag()` resolves external dependencies, the pipeline contains stages from multiple source pipelines
- Each stage's lock files live in its own `state_dir`, not the project root's `.pivot`

**Additional issue:** These CLI commands don't call `build_dag()`, so they miss transitively included stages from other pipelines entirely.

---

## Architecture

### Three Distinct Concepts

| Concept | Purpose | Location |
|---------|---------|----------|
| **Project root** | Path validation boundary | Topmost `.pivot` in ancestry |
| **Cache directory** | Shared artifact cache | `project_root / ".pivot" / "cache"` |
| **State directory** | Per-pipeline locks, state.db | `pipeline._root / ".pivot"` |

- Cache is shared across all pipelines (stored at project root)
- State (locks, state.db) is per-pipeline (stored with each pipeline)
- Project root defines the boundary for path validation

### Scope of CLI Operations

CLI commands should operate on:
- The current pipeline's stages
- **Plus** all transitively resolved dependencies (which may come from other pipelines)

This matches what `build_dag()` already does via `resolve_external_dependencies()` — it pulls in stages from parent/sibling pipelines that produce needed outputs. After resolution, each included stage retains its original `state_dir`.

---

## Design

### Part 1: Project Root Detection

**File:** `src/pivot/project.py`

Change `find_project_root()` algorithm:

```python
def find_project_root() -> pathlib.Path:
    """Find the topmost .pivot directory in ancestry.

    Walks from cwd to filesystem root, returning the highest .pivot found.
    This ensures subdirectory .pivot folders (used for per-pipeline state)
    don't get mistaken for project roots.

    Raises:
        ProjectRootNotFoundError: If no .pivot directory found in ancestry.
    """
    current = pathlib.Path.cwd().resolve()
    topmost_pivot: pathlib.Path | None = None

    for parent in [current, *current.parents]:
        if (parent / ".pivot").exists():
            topmost_pivot = parent  # Keep walking, update to higher one

    if topmost_pivot is None:
        raise ProjectRootNotFoundError(
            "No .pivot directory found. Run 'pivot init' to initialize a project."
        )

    return topmost_pivot
```

**Key changes:**
- Only looks for `.pivot`, not `.git`
- Walks all the way up, uses the **topmost** `.pivot` found
- Raises error instead of falling back to cwd

**File:** `src/pivot/exceptions.py`

Add new exception:

```python
class ProjectRootNotFoundError(PivotError):
    """Raised when no .pivot directory found in directory ancestry."""

    def get_suggestion(self) -> str | None:
        return "Run 'pivot init' in your project root directory."
```

### Part 2: CLI Commands Must Call `build_dag()`

**Affected commands:** `checkout`, `verify`, `remote push/pull`, `data get`

These commands need to call `cli_helpers.build_dag()` before operating on stages. This triggers `resolve_external_dependencies()` and includes transitive dependencies from other pipelines.

**Example pattern:**

```python
@cli_decorators.pivot_command()
def checkout(...):
    # Build DAG to resolve external dependencies
    cli_helpers.build_dag(validate=True)

    # Now cli_helpers.list_stages() includes all transitively needed stages
    for stage_name in cli_helpers.list_stages():
        ...
```

### Part 3: Use Per-Stage `state_dir`

**File:** `src/pivot/cli/helpers.py`

Add helper function:

```python
def get_stage_state_dir(stage_name: str) -> pathlib.Path:
    """Get state directory for a stage, falling back to default.

    Each stage may have its own state_dir (from its source pipeline).
    Falls back to config.get_state_dir() if stage has no explicit state_dir.
    """
    stage_info = get_stage(stage_name)
    return stage_info["state_dir"] or config.get_state_dir()
```

**Update stage-iterating functions** to use `get_stage_state_dir(stage_name)`:

```python
# Before (wrong - single state_dir for all stages)
state_dir = config.get_state_dir()
for stage_name in cli_helpers.list_stages():
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))

# After (correct - per-stage state_dir)
for stage_name in cli_helpers.list_stages():
    state_dir = cli_helpers.get_stage_state_dir(stage_name)
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
```

**For batching efficiency** (S3 operations): Group stages by `state_dir` before batch operations:

```python
from collections import defaultdict

def _group_stages_by_state_dir() -> dict[pathlib.Path, list[str]]:
    """Group stage names by their state directory for batch operations."""
    groups = defaultdict[pathlib.Path, list[str]](list)
    for stage_name in cli_helpers.list_stages():
        state_dir = cli_helpers.get_stage_state_dir(stage_name)
        groups[state_dir].append(stage_name)
    return dict(groups)

# Then batch per state_dir group
for state_dir, stage_names in _group_stages_by_state_dir().items():
    transfer.push(cache_dir, state_dir, ..., targets=stage_names, ...)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/pivot/project.py` | New `find_project_root()` algorithm — topmost `.pivot`, no `.git` fallback |
| `src/pivot/exceptions.py` | Add `ProjectRootNotFoundError` |
| `src/pivot/cli/helpers.py` | Add `get_stage_state_dir()` helper |
| `src/pivot/cli/checkout.py` | Call `build_dag()`, use per-stage state_dir in `_get_stage_output_info()` |
| `src/pivot/cli/verify.py` | Call `build_dag()`, use per-stage state_dir |
| `src/pivot/cli/remote.py` | Call `build_dag()`, use per-stage state_dir, group by state_dir for batching |
| `src/pivot/cli/data.py` | Call `build_dag()`, use per-stage state_dir |

---

## Migration / Breaking Changes

1. **`pivot init` required before first run** — Projects can no longer implicitly use `.git` as project root marker. Users must run `pivot init` first. This is acceptable since most workflows already create `.pivot` via `pivot init` or first `pivot repro`.

2. **No backwards compatibility shim** — Per AGENTS.md, this is pre-alpha and breaking changes are acceptable.

---

## Testing Considerations

1. **Project root detection:**
   - Test topmost `.pivot` selection when multiple exist in ancestry
   - Test error when no `.pivot` found
   - Test that subdirectory `.pivot` doesn't become project root

2. **CLI state_dir usage:**
   - Test `checkout` with stages from multiple pipelines (different state_dirs)
   - Test `verify` with transitively resolved dependencies
   - Test `remote push` batching groups by state_dir correctly

3. **Integration:**
   - End-to-end test: run from subdirectory pipeline that depends on sibling pipeline outputs
