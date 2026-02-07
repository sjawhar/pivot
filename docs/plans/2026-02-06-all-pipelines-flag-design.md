# `--all` Flag: Run Commands Across All Pipelines

**Goal:** Add an `--all` flag to CLI commands (`repro`, `status`, `verify`, `commit`, `push`, `pull`) that discovers every pipeline in the project, merges them into one unified DAG, and executes as normal. No change to runtime behavior — parallelism, dependency resolution, and error handling work identically to today, just with a bigger graph.

**Problem:** A project can contain multiple pipelines (e.g., `eval-pipeline/pivot` has six). Today, each command operates on the single pipeline discovered from the current directory. To run the full project, users must `cd` into each pipeline directory and run commands separately.

**Constraints:**
- No new concepts or commands — just a flag on existing commands
- One big DAG, not sequential per-pipeline execution
- Each pipeline's state directory must be respected (lock files, StateDB)
- Watch mode must reload correctly with `--all`
- Default behavior (without `--all`) is unchanged

---

## 1. Discovery: `discover_pipeline(all_pipelines=True)`

Extend the existing `discover_pipeline()` function with an `all_pipelines` parameter:

```python
def discover_pipeline(
    project_root: pathlib.Path | None = None,
    *,
    all_pipelines: bool = False,
) -> Pipeline | None:
```

When `all_pipelines=True`:

1. Call `_glob_all_pipelines(project_root)` to find all `pipeline.py` / `pivot.yaml` / `pivot.yml` files (already implemented, skips `.venv`, `.git`, `.pivot`, etc.)
2. Load each via `load_pipeline_from_path()` (already implemented)
3. Create a synthetic root: `Pipeline("all", root=project_root)`
4. Call `combined.include(other)` for each discovered pipeline

`include()` provides:
- **Auto-prefix on name collision** — when a stage name already exists, incoming stages are prefixed with `{pipeline_name}/` to disambiguate. The first pipeline's stages keep their bare names. This avoids errors when pipelines share common stage names like `train`.
- **Deep copies** preserving each stage's original `state_dir`

When `all_pipelines=False` (default): today's behavior, unchanged.

The combined pipeline goes into the Click context like today's single pipeline. All downstream code (`build_dag`, `get_all_stages`, etc.) works unchanged. `resolve_external_dependencies()` becomes a no-op since all stages are already present.

### External dependency behavior

When `--all` is active, all stages from all discovered pipelines are present in the combined registry. After merging, `_discover_all_pipelines` checks for unresolved dependencies (deps not produced by any discovered pipeline) and logs a warning listing them. This surfaces misconfiguration early without blocking execution.

### Existing code reused

| Function | Location | Status |
|----------|----------|--------|
| `_glob_all_pipelines()` | `pipeline/pipeline.py` | Exists (private, used by tier-3 scan) |
| `load_pipeline_from_path()` | `discovery.py` | Exists |
| `Pipeline.include()` | `pipeline/pipeline.py` | Exists |

---

## 2. CLI Integration

Add `--all` to the `@pivot_command()` decorator, since that's where auto-discovery already happens. The decorator:

1. Reads the `--all` flag from the Click context
2. Calls `discover_pipeline(root, all_pipelines=True)` instead of `discover_pipeline(root)`
3. Stores the pipeline in context as usual — downstream code is unchanged

Gate exposure with `pivot_command(allow_all=True)` so only opted-in commands accept the flag. Phase 1 enables it on `verify` only; Phase 2 enables the rest.

### Commands receiving `--all`

| Command | Phase | Notes |
|---------|-------|-------|
| `verify` | 1 | First milestone — harden plumbing before broad exposure |
| `repro` | 2 | Run all pipelines as one DAG |
| `status` | 2 | Show status across all pipelines |
| `commit` | 2 | Commit pending locks from all pipelines |
| `push` | 2 | Push cached outputs from all pipelines (switch to `auto_discover=True`) |
| `pull` | 2 | Pull outputs for all pipelines (switch to `auto_discover=True`) |

`push` and `pull` currently use `auto_discover=False` because they read from lock files directly. With `--all`, they switch to `auto_discover=True` so the registry is loaded and per-stage `state_dir` is available for lock file resolution. If registry load fails, surface a clear error rather than breaking the existing locks-only flow.

---

## 3. State-Dir Awareness (Core Refactor)

### Problem

The coordinator (engine, CLI commands) uses `config.get_state_dir()` as a single global path for StateDB, lock files, and pending locks. Workers already use per-stage `state_dir` correctly. When `--all` combines stages from pipelines with different `state_dir` values, the coordinator writes to the wrong location.

Example: `eval_pipeline/base/` uses `root=project.get_project_root()` (shared `.pivot/`), but `model_reports/time_horizon_1_0/` uses its own root (separate `.pivot/`). Combining them into one DAG means lock files and StateDB writes must go to the correct location per stage.

### Fix

Centralize per-stage `state_dir` lookup in a helper function to avoid missed call sites:

```python
def get_stage_state_dir(stage_info: RegistryStageInfo) -> pathlib.Path:
    """Get the state directory for a stage, falling back to project config."""
    return stage_info["state_dir"] or config.get_state_dir()
```

Replace `config.get_state_dir()` with this helper at every affected call site. The data is already available — every call site has access to `RegistryStageInfo` (which carries `state_dir`) via the stage map or registry.

### Affected call sites

**Engine (`engine.py`) — has `stage_info` via `self._get_stage(name)`:**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 533 | `state_dir = config.get_state_dir()` | Use per-stage `state_dir` from `stage_info` | 2 |
| 557 | `state_db_path = config.get_state_db_path()` | Cache StateDB connections per unique `state_dir` | 2 |
| 1081 | `state_dir = config.get_state_dir()` | Look up `stage_info["state_dir"]` per stage in loop | 2 |
| 1111 | `StateDB(config.get_state_db_path())` | Write run history to appropriate DB | 2 |

**Executor (`executor/core.py`) — has `all_stages` dict:**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 410 | `cache_dir = config.get_cache_dir()` | Cache dir is shared — no change needed | — |
| 412 | `StateDB(config.get_state_db_path())` | Use per-stage `state_dir` | 2 |
| 457 | `state_dir = config.get_state_dir()` | Use `all_stages[stage_name]["state_dir"]` | 2 |

**Executor (`executor/commit.py`):**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 27 | `state_dir = config.get_state_dir()` | Accept `state_dir` as parameter or look up per stage | 2 |
| 30 | `StateDB(config.get_state_db_path())` | Use per-stage `state_dir` | 2 |

**CLI verify (`cli/verify.py`) — has `all_stages` dict:**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 308 | `state_dir = config.get_state_dir()` | Use per-stage `state_dir` via helper | 1 |

**CLI remote (`cli/remote.py`) — needs registry access with `--all`:**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 66, 141, 229 | `state_dir = config.get_state_dir()` | Look up per-stage `state_dir` from registry | 2 |
| 86, 158, 248 | `StateDB(config.get_state_db_path())` | Use per-stage `state_dir` | 2 |

**CLI checkout (`cli/checkout.py`) — has stage info available:**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 336 | `state_dir = config.get_state_dir()` | Use per-stage `state_dir` | 2 |

**Status (`status.py`) — has `all_stages` dict:**

| Line | Current | Fix | Phase |
|------|---------|-----|-------|
| 197 | `state_dir = config.get_state_dir()` | Use `all_stages[stage_name]["state_dir"]` | 1 (verify uses this) |
| 295 | `state_dir = config.get_state_dir()` | Same | 1 |
| 346, 405 | `StateDB(config.get_state_db_path())` | Use per-stage `state_dir` | 1 |

### Not affected

- `config.get_cache_dir()` calls — cache is shared across all pipelines
- `cli/history.py` — run history is project-level
- `cli/doctor.py` — health check on shared cache
- `show/` commands — read-only display
- `cli/data.py`, `cli/track.py` — shared cache/tracking
- TUI code — can be a follow-up

### StateDB connection cache

The engine holds a `dict[Path, StateDB]` mapping `state_dir → connection`, lazily opened. In the common case (all pipelines share one `state_dir`), this dict has one entry — zero overhead vs today.

---

## 4. Watch Mode (Phase 2)

When the engine is initialized with `--all`, it stores that flag (e.g., `_stored_all_pipelines: bool`). On code/config change:

- **`_reload_registry`** calls `discovery.discover_pipeline(root, all_pipelines=self._stored_all_pipelines)` instead of `discovery.discover_pipeline(root)`. Since both paths go through `discover_pipeline()`, the logic is centralized.

- **Watch paths**: Add pipeline config files (`pipeline.py`, `pivot.yaml`/`yml`) to watch paths so that adding/removing a pipeline triggers a reload. Artifact watch paths already work for free — `get_watch_paths()` extracts them from the bipartite graph, which in `--all` mode includes artifacts from all pipelines.

- **`_clear_project_modules`** works for free — it removes all modules under the project root, covering all sub-pipeline directories.

---

## 5. Testing Strategy

### Existing fixtures to reuse

| Fixture | Location | Use |
|---------|----------|-----|
| `test_pipeline_include_preserves_state_dir*` | `tests/pipeline/test_pipeline.py:338-376` | Template for state-dir isolation tests |
| `test_lazy_resolution_*` | `tests/integration/test_lazy_resolution.py` | Multi-pipeline integration patterns |
| `create_pipeline_py()` | `tests/helpers.py` | Generate `pipeline.py` files in subdirectories |
| `isolated_pivot_dir` | `tests/conftest.py:354` | Full filesystem isolation for CLI tests |
| `make_valid_lock_content` | `tests/conftest.py:305` | Factory for lock file data |
| `pipeline_dir` | `tests/conftest.py:326` | Creates `.pivot` directory per pipeline |

### Phase 1 tests

**Unit: `discover_pipeline(all_pipelines=True)`**
- Temp project with 2-3 sub-pipelines (some sharing root, some with separate roots)
- Combined pipeline contains all stages from all discovered pipelines
- Name collisions raise `PipelineConfigError` with pipeline file paths
- Empty project returns None
- Collision error message includes both pipeline paths

**Integration: `verify --all` across mixed state_dir roots**
- Two pipelines: one sharing project root, one with separate root
- Lock files in each pipeline's own `.pivot/stages/`
- `verify --all` reads lock files from correct state dirs
- StateDB reads from correct databases per stage

### Phase 2 tests

- Watch mode: reload preserves `--all`, pipeline config file changes trigger reload
- Push/pull across multiple state dirs: targeted and untargeted
- Collision diagnostics: clear errors with pipeline file paths
- Mixed shared vs isolated roots: stages execute with correct state isolation
- `repro --all`: lock files written to correct per-stage state dirs, deferred writes routed correctly

Existing tests continue passing since default behavior (`all_pipelines=False`) is unchanged.

---

## Implementation Phases

### Phase 1: `verify --all` (first milestone)

Harden the discovery and state-dir plumbing before broad exposure.

1. **Extend `discover_pipeline(all_pipelines=True)`** — centralize all-pipeline discovery, include collision errors with pipeline file paths
2. **Gate `--all` in `pivot_command`** — add `allow_all=True` parameter, enable on `verify` only
3. **Centralize per-stage state_dir helper** — `get_stage_state_dir()` to avoid missed call sites
4. **Update `cli/verify.py` and `status.py`** — use helper for lock/StateDB access
5. **Define external-dep behavior** — warn or error if deps are outside discovered pipelines
6. **Tests** — unit discovery + name collision; integration `verify --all` across mixed state_dir roots

### Phase 2: Remaining commands

Enable `--all` across all commands using the proven plumbing.

1. **Enable `--all` on `repro`, `status`, `commit`, `push`, `pull`** — same discovery path
2. **Apply per-stage state_dir helper at all remaining call sites** — engine, executor, remote, checkout
3. **Watch mode** — persist `all_pipelines` flag, add pipeline config files to watch paths
4. **Push/pull resilience** — if registry load fails with `--all`, surface clear error rather than breaking locks-only flows
5. **Tests** — watch reload, push/pull across multiple state_dirs, collision diagnostics, mixed roots
