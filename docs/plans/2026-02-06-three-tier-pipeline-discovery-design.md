# Three-Tier Pipeline Discovery

**Goal:** Automatically discover which pipeline produces a given output, without loading every pipeline.py in the project, and without assuming code and data are co-located.

**Problem:** The current discovery algorithm (`find_pipeline_paths_for_dependency`) traverses parent directories from a dependency's file path to find a `pipeline.py`. This breaks when the producing pipeline's code lives in a different directory tree from its outputs (e.g., code in `eval_pipeline/difficulty/`, outputs in `data/difficulty/processed/`). The traversal never reaches the producer.

**Constraints:**
- Discovery must be automatic (no explicit wiring by users)
- Must not require loading/importing all pipeline.py files (startup latency)
- Must not require fingerprinting the entire DAG when only part is needed
- Must work correctly in watch mode (fresh state per rebuild)
- Stale cache must never cause incorrect behavior, only wasted work

---

## Architecture: Three-Tier Resolution

When `resolve_external_dependencies()` encounters an unresolved dependency, it tries three strategies in order, stopping at the first success:

### Tier 1 — Traverse-up (existing behavior)

Walk up from the dependency's file path looking for `pipeline.py` / `pivot.yaml`. Works when code and data are co-located. Zero cost, no cached state required. This is the current `find_pipeline_paths_for_dependency()`.

### Tier 2 — Output index cache (new)

Read a cached hint file at `.pivot/cache/outputs/{dep_path}`. If it exists, its content is the project-relative directory of the pipeline that produces this output. Load that pipeline, **verify** it still produces the dep. If stale (doesn't produce it), fall through to tier 3.

### Tier 3 — Full scan (new, fallback)

Glob all `**/pipeline.py` and `**/pivot.yaml` / `**/pivot.yml` files in the project. Load each (using the shared per-call `loaded_pipelines` dict) until a producer is found. Handles cold start (no index yet) and stale index entries. Only triggers when tiers 1 and 2 both fail.

### Cost profile

| Tier | When it fires | Cost |
|------|--------------|------|
| 1 | Always (first attempt) | ~microseconds (stat calls) |
| 2 | Tier 1 miss | ~50-100us (one `read_text()`) |
| 3 | Tiers 1+2 miss (cold start, stale cache, restructured project) | ~10-400ms (glob + Python imports) |

---

## Output Index: Filesystem-as-Hashmap

**Location:** `<project_root>/.pivot/cache/outputs/<dep_path>`

Each file's content is a single string: the project-relative directory of the producing pipeline.

**Example:**
```
.pivot/cache/outputs/data/difficulty/processed/task_difficulty.yaml
```
Contains:
```
eval_pipeline/difficulty
```

### Why filesystem, not JSON?

- **O(1) lookup** — read exactly the file you need, not an entire index
- **No serialization** — content is a plain string
- **No read-merge-write** — each pipeline writes its own entries independently
- **Atomic per-entry** — updating one entry doesn't risk corrupting others
- **Debuggable** — `cat .pivot/cache/outputs/data/foo.csv` shows the producer

### Reading (EAFP pattern)

```python
try:
    pipeline_dir = (project_root / ".pivot/cache/outputs" / dep_path).read_text().strip()
except FileNotFoundError:
    pipeline_dir = None
```

One syscall. No `exists()` check.

### Writing

Written as a side effect after `build_dag()` completes successfully. For every stage in the pipeline (including externally-included stages), write each output path:

```python
def _write_output_index(self) -> None:
    cache_dir = project.get_project_root() / ".pivot" / "cache" / "outputs"
    for stage_name in self.list_stages():
        info = self.get(stage_name)
        pipeline_dir = str(info["state_dir"].parent.relative_to(project.get_project_root()))
        for out_path in info["outs_paths"]:
            target = cache_dir / out_path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(pipeline_dir)
            except OSError:
                logger.debug(f"Failed to write output index for {out_path}")
```

Cache writes are never fatal — `OSError` is caught and logged.

### Invalidation

No explicit invalidation needed. Stale entries are harmless:
1. Tier 2 reads stale entry → loads the pipeline it points to
2. Pipeline no longer produces the dep → tier 2 returns None
3. Falls through to tier 3 (full scan) → finds actual producer
4. Next `build_dag()` rewrites the index with correct data

The output index is a **read hint only**. Correctness never depends on it.

---

## Modified `resolve_external_dependencies()` Flow

```python
def resolve_external_dependencies(self) -> None:
    project_root = project.get_project_root()

    # Build local_outputs and work queue (unchanged)
    local_outputs = set[str]()
    all_deps = set[str]()
    for stage_name in self.list_stages():
        info = self.get(stage_name)
        local_outputs.update(info["outs_paths"])
        all_deps.update(info["deps_paths"])

    work = all_deps - local_outputs
    if not work:
        return

    # Per-call caches (fresh every call — safe for watch mode)
    loaded_pipelines: dict[Path, Pipeline | None] = {}
    all_pipeline_paths: list[Path] | None = None  # lazy, for tier 3

    while work:
        dep_path = work.pop()
        if dep_path in local_outputs:
            continue

        # Tier 1: traverse-up (existing behavior)
        producer = _find_producer_via_traversal(dep_path, project_root, loaded_pipelines)

        # Tier 2: output index cache
        if producer is None:
            producer = _find_producer_via_index(dep_path, project_root, loaded_pipelines)

        # Tier 3: full scan
        if producer is None:
            if all_pipeline_paths is None:
                all_pipeline_paths = _glob_all_pipelines(project_root)
            producer = _find_producer_via_scan(
                dep_path, all_pipeline_paths, loaded_pipelines
            )

        if producer is not None:
            stage_info = copy.deepcopy(producer)
            self._registry.add_existing(stage_info)
            local_outputs.update(stage_info["outs_paths"])
            work.update(d for d in stage_info["deps_paths"] if d not in local_outputs)
```

All three `_find_producer_via_*` methods return `RegistryStageInfo | None` and share the `loaded_pipelines` dict to avoid re-importing the same pipeline.py.

### `build_dag()` changes

```python
def build_dag(self, validate: bool = True) -> DiGraph[str]:
    self.resolve_external_dependencies()
    dag = self._registry.build_dag(validate=validate)
    self._write_output_index()  # new: populate cache for future runs
    return dag
```

---

## Watch Mode Interaction

No special handling needed. The design is naturally safe:

1. **Per-call freshness** — `loaded_pipelines` and `all_pipeline_paths` are local variables, fresh every call.
2. **New Pipeline per reload** — `_handle_code_or_config_changed()` creates a brand new Pipeline instance via `discover_pipeline()`. `_external_deps_resolved` starts as `False`.
3. **Stale index is harmless** — Verify step catches it, falls through to tier 3. Index is rewritten after successful `build_dag()`.
4. **New pipeline files** — Tiers 1 and 2 might miss them (no index entry), but tier 3's fresh glob finds them.

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Automatic discovery (no explicit wiring) | Users shouldn't have to maintain dependency declarations |
| Filesystem-as-hashmap for index | O(1) lookup, no parsing, no serialization format, atomic per-entry |
| Write index after `build_dag()`, not mid-resolution | Simpler (single write point), `local_outputs` set already prevents redundant work within a call |
| No index invalidation on watch changes | Verify-then-fallthrough handles staleness; invalidation would force expensive tier 3 |
| Per-call state only (no module-level caches) | Watch mode safety — fresh state every rebuild |
| Catch `OSError` on index writes | Cache is purely an optimization, never fatal |
| EAFP for index reads (try/except, not exists()) | Faster (one syscall), avoids TOCTOU races |

## Known Limitations / Future Work

- **Stage name collisions across pipelines** — If two external pipelines have stages with the same name, only the first found is included. Pre-existing issue, not introduced by this design.
- **Performance for genuine external files** — Deps that exist on disk but aren't produced by any pipeline will trigger traversal on every resolve. Tier 2 cache miss is cheap (one `FileNotFoundError`), but worth monitoring.
- **Syntax errors in external pipelines** — Currently logged at debug level, causing confusing `DependencyNotFoundError`. Should improve error messaging to surface load failures.
- **Cross-call pipeline cache** — Promoting `loaded_pipelines` to process-level scope would avoid re-importing the same pipeline.py across multiple `build_dag()` calls in one `pivot repro`. Requires care with watch mode invalidation. Deferred.
