# Design: Use .pvt File Hashes for Verification When Files Missing

**Issue:** [#265](https://github.com/sjawhar/pivot/issues/265)
**Date:** 2026-01-29

## Problem

When running `pivot verify --allow-missing` in CI, stages fail with "Missing deps" even when the dependency has a `.pvt` file with a valid hash. This prevents using pivot's verification in CI environments where only `.pvt` pointers exist (not the actual data files).

## Solution

Extend `--allow-missing` to use hashes from `.pvt` files when actual files are missing.

### Verification Flow

For each dependency when `--allow-missing` is set:

```
File exists on disk?
├─ YES → hash actual file (current behavior)
└─ NO → Is path in tracked_trie (exact match or inside tracked dir)?
        ├─ YES → use hash from .pvt data
        └─ NO → add to missing_deps (error)
```

### Key Design Decisions

1. **`.pvt` is fallback, not primary** - If file exists, always hash it directly
2. **Reuse existing tracked file discovery** - DAG building already discovers `.pvt` files and builds a trie; pass this data downstream rather than re-discovering
3. **Support nested paths** - A dep like `data/file.csv` inside tracked directory `data/` looks up the file's hash from the directory manifest
4. **Applies to both `verify` and `run --dry-run`** - Anywhere `--allow-missing` is used

## Data Flow

```
verify.py (has allow_missing flag)
  → status.get_pipeline_status(allow_missing=True)
    → registry.build_dag()
        returns: dag, tracked_files, tracked_trie
    → explain.get_stage_explanation(tracked_files, tracked_trie, allow_missing)
        uses: .pvt hashes for missing files when allow_missing=True
```

### New Parameters to Thread

| Parameter | Type | From | To |
|-----------|------|------|-----|
| `allow_missing` | `bool` | CLI | `get_stage_explanation()` |
| `tracked_files` | `dict[Path, PvtData]` | `build_dag()` | `get_stage_explanation()` |
| `tracked_trie` | `pygtrie.StringTrie` | `build_dag()` | `get_stage_explanation()` |

## Implementation Details

### Logic in `explain.py`

```python
def get_stage_explanation(
    ...,
    allow_missing: bool = False,
    tracked_files: dict[Path, PvtData] | None = None,
    tracked_trie: StringTrie | None = None,
):
    deps = [...]  # list of dependency paths

    if allow_missing and tracked_files and tracked_trie:
        deps_to_hash = []
        pvt_hashes: dict[str, HashInfo] = {}
        missing_deps = []

        for dep in deps:
            dep_path = Path(dep)
            if dep_path.exists():
                deps_to_hash.append(dep)
            else:
                hash_info = _find_tracked_hash(dep_path, tracked_files, tracked_trie)
                if hash_info:
                    pvt_hashes[normalize_path(dep)] = hash_info
                else:
                    missing_deps.append(dep)

        file_hashes, more_missing, unreadable = worker.hash_dependencies(deps_to_hash)
        dep_hashes = {**file_hashes, **pvt_hashes}
        missing_deps.extend(more_missing)
    else:
        dep_hashes, missing_deps, unreadable = worker.hash_dependencies(deps)
```

### Finding Hash for Nested Paths

```python
def _find_tracked_hash(
    dep: Path,
    tracked_files: dict[Path, PvtData],
    tracked_trie: StringTrie,
) -> HashInfo | None:
    # Check if path or ancestor is tracked
    tracked_path = _find_tracked_ancestor(dep, tracked_trie)
    if not tracked_path:
        return None

    pvt_data = tracked_files[tracked_path]

    # Exact match - use top-level hash
    if dep == tracked_path:
        hash_info: HashInfo = {"hash": pvt_data["hash"]}
        if "manifest" in pvt_data:
            hash_info["manifest"] = pvt_data["manifest"]
        return hash_info

    # Nested path - find in manifest
    if "manifest" not in pvt_data:
        return None  # Single file .pvt can't contain nested paths

    relpath = str(dep.relative_to(tracked_path))
    for entry in pvt_data["manifest"]:
        if entry["relpath"] == relpath:
            return {"hash": entry["hash"]}

    return None  # Path not found in manifest
```

## Files to Modify

1. **`src/pivot/cli/verify.py`** - Pass `allow_missing` to status call
2. **`src/pivot/cli/run.py`** - Same for `--dry-run` path
3. **`src/pivot/status.py`** - Thread parameters to explain, return tracked data from DAG
4. **`src/pivot/explain.py`** - Add `.pvt` fallback logic
5. **`src/pivot/registry.py`** - Expose `tracked_files` and `tracked_trie` from `build_dag()`

## Edge Cases

| Case | Behavior |
|------|----------|
| File exists locally | Hash actual file (current behavior) |
| File missing, exact `.pvt` match | Use `.pvt` hash |
| File missing, inside tracked directory | Look up hash from directory manifest |
| File missing, no `.pvt` | Error - can't verify |
| File exists but differs from `.pvt` | Hash actual file; `.pvt` mismatch is separate concern |
| Directory dep with manifest | Include manifest in hash info |

## Testing

1. Verify with missing file but valid `.pvt` → should pass
2. Verify with missing file, no `.pvt` → should fail
3. Verify with file inside tracked directory → should use manifest hash
4. Verify with file present → should hash actual file (ignore `.pvt`)
5. `run --dry-run` with same scenarios
