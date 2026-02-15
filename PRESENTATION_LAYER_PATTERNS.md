# Presentation Layer Patterns - Existing Code Analysis

**Context:** Path-free engine changes complete. This document catalogs existing path generation, symlink tree creation, and presentation utilities to inform Task 7 (presentation layer design) and test updates.

## Executive Summary

**Key Finding:** Pivot has TWO path generation systems:
1. **WorkspaceStore** (`storage/store.py`) - Identity-to-path resolution for compose API
2. **Legacy path utilities** (`compose.py`, `path_utils.py`) - Used by registry-based stages

**Presentation Gap:** No unified "presentation tree" builder exists. Checkout/restore operations work file-by-file, not as tree construction.

---

## 1. Path Generation Patterns

### 1.1 WorkspaceStore (Store Protocol Implementation)

**Location:** `packages/pivot/src/pivot/storage/store.py:123-246`

**Purpose:** Resolve `ArtifactRef` (identity-based) to filesystem paths for compose API stages.

**Key Methods:**
```python
def _resolve_output_path(self, ref: types.ArtifactRef) -> pathlib.Path:
    """Generate workspace path from artifact identity + tag + format."""
    stage_name = ref.identity.producer
    prefix = "data"  # or "metrics"/"plots" based on tag
    
    # Pattern: {prefix}/{pipeline_name}/{stage_name}[/{key}].{ext}
    if key is None:
        return project_root / prefix / pipeline_name / f"{stage_name}.{ext}"
    else:
        return project_root / prefix / pipeline_name / stage_name / f"{key}.{ext}"
```

**Path Structure:**
- Single output: `data/{pipeline}/{stage}.{ext}`
- Multi-output: `data/{pipeline}/{stage}/{key}.{ext}`
- Directories: `data/{pipeline}/{stage}/` (no extension)
- Metrics: `metrics/{pipeline}/{stage}.yaml`
- Plots: `plots/{pipeline}/{stage}.png`

**Collision Detection:** Tracks `_path_map: dict[str, ArtifactIdentity]` to detect when two artifacts resolve to same path.

**Input Resolution:**
```python
def _resolve_input_path(self, ref: types.ArtifactRef) -> pathlib.Path:
    """Resolve pipeline inputs from bindings or default location."""
    if name in self._input_bindings:
        return project_root / binding
    return project_root / "data" / "raw" / name
```

---

### 1.2 Legacy Path Generation (Registry-Based)

**Location:** `packages/pivot/src/pivot/compose.py:69-79`

```python
def _generate_artifact_path(
    pipeline_name: str,
    stage_name: str,
    output_spec: _OutputSpec,
    is_single_output: bool,
) -> str:
    """Generate relative path for registry-based stages."""
    prefix = _artifact_dir_prefix(output_spec.tag)  # "data"/"metrics"/"plots"
    ext = _format_extension(output_spec.format)
    if is_single_output:
        return f"{prefix}/{pipeline_name}/{stage_name}.{ext}"
    return f"{prefix}/{pipeline_name}/{stage_name}/{output_spec.key}.{ext}"
```

**Usage:** Called during `Pipeline.build()` to populate registry with paths. **Not used at runtime** — workers use paths from registry.

---

### 1.3 Path Canonicalization

**Location:** `packages/pivot/src/pivot/path_utils.py:9-38`

```python
def canonicalize_artifact_path(path: str, base: pathlib.Path) -> str:
    """Produce single canonical form for artifact paths.
    
    Canonical form:
    - Absolute (resolved from base if relative)
    - Normalized (no .., no //, no trailing dots)
    - POSIX separators (backslashes → forward slashes)
    - Trailing slash preserved for DirectoryOut
    """
```

**Critical:** This is the ONE function for producing artifact paths for in-memory use (registry, DAG, engine). Lockfiles convert to/from project-relative at their own boundary.

---

## 2. Symlink/Checkout Patterns

### 2.1 Cache Checkout Modes

**Location:** `packages/pivot/src/pivot/storage/cache.py:376-440`

**Three modes with fallback:**
```python
CheckoutMode.SYMLINK   # Symlink to cache (read-only)
CheckoutMode.HARDLINK  # Hardlink to cache (copy-on-write)
CheckoutMode.COPY      # Full copy (writable)
```

**Default order:** `[HARDLINK, SYMLINK, COPY]` (from `config.get_checkout_mode_order()`)

**Key Functions:**
- `_checkout_from_cache(path, cache_path, mode)` - Create single link
- `_checkout_with_fallback(path, cache_path, modes)` - Try modes in order
- `restore_from_cache(path, cache_dir, hash_info, checkout_modes)` - Restore file/dir from cache

**Directory Handling:**
- **SYMLINK mode:** Cache entire directory, symlink to it (fast path)
- **HARDLINK/COPY modes:** Restore each file individually from manifest

---

### 2.2 CLI Checkout Command

**Location:** `packages/pivot/src/pivot/cli/checkout.py`

**Purpose:** Restore tracked files from cache (user-facing command).

**Key Features:**
- `CheckoutBehavior.ERROR` - Fail if file exists (default)
- `CheckoutBehavior.SKIP_EXISTING` - Skip existing files (`--only-missing`)
- `CheckoutBehavior.FORCE` - Overwrite existing (`--force`)

**Async restoration:** Uses `asyncio.Semaphore(MAX_CONCURRENT_RESTORES=32)` for parallel file restoration.

**Data Sources:**
- `_get_stage_output_info()` - Read lock files for cached stage outputs
- Filters out non-cached outputs (e.g., `Metric(cache=False)`)

**No tree building:** Operates file-by-file, not as unified tree construction.

---

### 2.3 Worker Dependency Restoration

**Location:** `packages/pivot/src/pivot/executor/worker.py:679-767`

**Two restoration paths:**

1. **File dependencies** (`_restore_file_dep`):
   ```python
   restored = cache.restore_from_cache(
       path, files_cache_dir, hash_info,
       checkout_modes=checkout_modes,
       state_dir=state_dir,
   )
   ```

2. **Directory dependencies** (`_restore_directory_dep`):
   - Restores entire directory from manifest
   - Uses same `restore_from_cache()` with `DirHash`

**Idempotency:** Checks if file already matches expected hash before restoring.

---

## 3. Store Protocol (Identity → Path Resolution)

**Location:** `packages/pivot/src/pivot/storage/store.py:15-26`

```python
class Store(Protocol):
    def checkout(self, ref: ArtifactRef) -> pathlib.Path:
        """Resolve artifact identity to filesystem path."""
    
    def prepare_output(self, ref: ArtifactRef) -> pathlib.Path:
        """Prepare output location (create dirs, temp files)."""
    
    def commit(self, ref: ArtifactRef, path: pathlib.Path) -> str:
        """Finalize output, return hash."""
    
    def hash_artifact(self, ref: ArtifactRef) -> HashInfo:
        """Compute hash of artifact."""
    
    def exists(self, ref: ArtifactRef) -> bool:
        """Check if artifact exists."""
```

**Two Implementations:**

1. **CacheStore** - Uses `cache/refs/{producer}/{key}` symlinks pointing to content-addressed cache
2. **WorkspaceStore** - Generates paths from identity using conventions (see 1.1)

**Usage:** Workers call `store.checkout(ref)` to get dependency paths, `store.prepare_output(ref)` for outputs.

---

## 4. Presentation-Related Utilities

### 4.1 Show/Plots Module

**Location:** `packages/pivot/src/pivot/show/plots.py`

**Key Functions:**
- `_plot_display_path(out, project_root)` - Convert `ArtifactRef` to relative display path
- `collect_plots_from_stages()` - Discover Plot outputs from registry
- `render_plots_html(plots, output_path)` - Generate HTML gallery

**Path Resolution:**
```python
def _plot_display_path(out, project_root) -> str | None:
    if isinstance(out, types.ArtifactRef):
        return _identity_str(out)  # "stage_name" or "stage_name:key"
    if isinstance(out, outputs.Plot):
        abs_path = str(project.normalize_path(expanded.path))
        return project.to_relative_path(abs_path, project_root)
```

**Note:** Uses identity strings for `ArtifactRef`, filesystem paths for legacy `Plot` outputs.

---

### 4.2 Show/Data Module

**Location:** `packages/pivot/src/pivot/show/data.py`

**Similar pattern to plots:**
- `_data_rel_path(out, project_root)` - Convert to relative path
- Filters out Metric/Plot outputs (only data artifacts)

---

## 5. TODOs Related to Presentation

### 5.1 Watch Path Resolution (Task 4)

**Location:** `packages/pivot/src/pivot/engine/graph.py:311-317`

```python
def get_watch_paths(g: nx.DiGraph[str]) -> list[str]:
    """Return watch paths (identity-based artifacts require Store resolution).
    
    TODO: Resolve ArtifactIdentity to filesystem paths via Store (Task 4).
    """
    _ = g
    return []
```

**Blocker:** Watch mode needs to resolve artifact identities to filesystem paths for file watching. Currently returns empty list.

---

### 5.2 Store-Based Watch Resolution

**Location:** `packages/pivot/src/pivot/engine/engine.py:2040`

```python
# TODO: Store-based watch resolution will re-enable this handler.
```

**Context:** Watch mode handler disabled pending Store-based path resolution.

---

## 6. Symlink Safety & Path Validation

### 6.1 Path Policy Module

**Location:** `packages/pivot/src/pivot/path_policy.py`

**Key Checks:**
- `check_exists=True` - Resolve symlinks to detect escapes outside base directory
- `symlink_escape_action` - "warn" or "error" when symlink resolves outside base

**Used by:** `stage_def.py` for defense-in-depth validation of output paths.

---

### 6.2 Project Utilities

**Location:** `packages/pivot/src/pivot/project.py:66-83`

```python
def contains_symlink_in_path(path: pathlib.Path, base: pathlib.Path) -> bool:
    """Check if any component from base to path is a symlink.
    
    Example: If /project/data is a symlink, and path is /project/data/file.csv,
    returns True because 'data' component is a symlink.
    """
```

**Usage:** CLI commands warn when tracking files inside symlinked directories.

---

## 7. Cache Path Structure

**Location:** `packages/pivot/src/pivot/storage/cache.py:229`

```python
def get_cache_path(cache_dir: pathlib.Path, hash: str) -> pathlib.Path:
    """Get cache path for a hash (XX/XXXX... structure)."""
    return cache_dir / hash[:2] / hash
```

**Structure:** Content-addressed storage with 2-char prefix sharding (e.g., `cache/ab/abcdef123456...`).

---

## 8. Key Insights for Task 7 (Presentation Layer)

### 8.1 No Unified Tree Builder

**Current state:** Checkout/restore operations work file-by-file:
- `cli/checkout.py` - Iterates over files, calls `restore_from_cache()` per file
- `executor/worker.py` - Restores dependencies one at a time

**Missing:** No function that takes a set of `ArtifactRef` and builds a complete workspace tree in one operation.

---

### 8.2 Two Path Generation Systems

**Problem:** Duplication between:
1. `WorkspaceStore._resolve_output_path()` - Runtime path generation
2. `compose._generate_artifact_path()` - Build-time path generation

**Risk:** Divergence if conventions change (e.g., adding subdirectories, changing extensions).

**Recommendation:** Unify path generation logic. Consider making `WorkspaceStore` the single source of truth.

---

### 8.3 Identity vs. Path Duality

**Current approach:**
- Engine/DAG use `ArtifactIdentity` (producer + key)
- Workers receive paths in `WorkerStageInfo`
- Store protocol bridges the gap

**For presentation layer:**
- Need to resolve identities → paths for display/watch
- `WorkspaceStore.checkout()` already does this
- Could extract path generation logic into standalone utility

---

### 8.4 Symlink Tree Creation

**Current capability:**
- `CacheStore` creates `cache/refs/{producer}/{key}` symlinks
- Individual file checkout via `_checkout_from_cache()`

**Missing for presentation:**
- Bulk tree creation (e.g., "materialize all outputs for these stages")
- Atomic tree updates (create new tree, swap symlink)
- Tree diffing (what changed between two presentations)

---

## 9. Recommendations for Task 7

### 9.1 Extract Path Generation

Create `pivot/presentation/paths.py`:
```python
def resolve_artifact_path(
    ref: ArtifactRef,
    project_root: pathlib.Path,
    pipeline_name: str,
) -> pathlib.Path:
    """Single source of truth for artifact path generation."""
```

Refactor `WorkspaceStore._resolve_output_path()` to use this.

---

### 9.2 Create Tree Builder

Create `pivot/presentation/tree.py`:
```python
def build_presentation_tree(
    artifacts: list[ArtifactRef],
    target_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    checkout_modes: list[CheckoutMode],
) -> dict[ArtifactRef, pathlib.Path]:
    """Build complete workspace tree from artifact identities."""
```

Use existing `restore_from_cache()` internally, but coordinate across all artifacts.

---

### 9.3 Unify with Store Protocol

Consider extending `Store` protocol:
```python
class Store(Protocol):
    def materialize_tree(
        self,
        refs: list[ArtifactRef],
        target_dir: pathlib.Path,
    ) -> dict[ArtifactRef, pathlib.Path]:
        """Materialize multiple artifacts into target directory."""
```

This keeps presentation logic within the Store abstraction.

---

### 9.4 Watch Path Resolution

Implement `get_watch_paths()` using Store:
```python
def get_watch_paths(g: nx.DiGraph[str], store: Store) -> list[str]:
    """Resolve artifact identities to filesystem paths for watching."""
    paths = []
    for node in g.nodes:
        if g.nodes[node]["type"] == NodeType.ARTIFACT:
            identity = parse_artifact_node(node)
            ref = identity_to_ref(identity)  # Need registry lookup
            path = store.checkout(ref)
            paths.append(str(path))
    return paths
```

**Blocker:** Need registry context to convert `ArtifactIdentity` → `ArtifactRef` (requires format/tag).

---

## 10. Tests to Update

### 10.1 Path Generation Tests

**Files to check:**
- `tests/test_compose.py` - Test `_generate_artifact_path()`
- `tests/test_store.py` - Test `WorkspaceStore._resolve_output_path()`

**Update needed:** Ensure both systems generate identical paths for same inputs.

---

### 10.2 Checkout Tests

**Files to check:**
- `tests/cli/test_checkout.py` - CLI checkout command
- `tests/storage/test_cache.py` - `restore_from_cache()` tests

**Update needed:** Add tests for bulk tree restoration (if implementing tree builder).

---

### 10.3 Store Protocol Tests

**Files to check:**
- `tests/storage/test_store.py` - `CacheStore` and `WorkspaceStore` tests

**Update needed:** Test collision detection, path resolution edge cases.

---

## 11. Related Documentation

**Existing docs:**
- `packages/pivot/src/pivot/storage/AGENTS.md` - StateDB patterns, path storage rules
- `packages/pivot/src/pivot/executor/AGENTS.md` - Worker path derivation
- `packages/pivot/src/pivot/cli/AGENTS.md` - CLI command patterns

**Missing docs:**
- Presentation layer architecture (Task 7 deliverable)
- Path generation conventions (should document WorkspaceStore patterns)
- Store protocol usage guide

---

## Appendix: Code Locations Quick Reference

| Pattern | File | Lines |
|---------|------|-------|
| WorkspaceStore path generation | `storage/store.py` | 177-198 |
| Legacy path generation | `compose.py` | 69-79 |
| Path canonicalization | `path_utils.py` | 9-38 |
| Cache checkout modes | `storage/cache.py` | 376-440 |
| CLI checkout command | `cli/checkout.py` | 1-420 |
| Worker dependency restoration | `executor/worker.py` | 679-767 |
| Store protocol definition | `storage/store.py` | 15-26 |
| Plot display paths | `show/plots.py` | 48-60 |
| Watch path TODO | `engine/graph.py` | 311-317 |
| Symlink safety checks | `path_policy.py` | 30-154 |
| Cache path structure | `storage/cache.py` | 229 |

---

**Document Status:** Complete analysis of existing patterns. Ready for Task 7 design phase.
