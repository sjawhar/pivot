---
tags: [python, caching, filesystem, incremental]
category: design
module: executor.worker
symptoms: ["cache corruption", "unexpected file modifications", "stale data in cache"]
---

# Incremental Outputs Use COPY Mode for Cache Restoration

## Problem

An output tagged `incremental` accumulates state across runs. Before its stage
runs, Pivot restores the prior artifact from cache and passes the loaded value
to the parameter bound with `pipeline.previous()`. A hardlink or symlink would
either mutate cached bytes or make the restored value unwritable.

- **Hardlink corruption**: Modifying the restored file also modifies the cached version (they share the same inode), destroying the historical snapshot
- **Symlink failure**: Symlinks to read-only cache files can't be written to at all

```python
# Scenario with hardlink restoration (WRONG):
# 1. Cache has "database.txt" with hash abc123
# 2. Restore creates hardlink: database.txt -> cache/ab/c123
# 3. Stage appends to database.txt
# 4. PROBLEM: cache/ab/c123 is now corrupted with new data
# 5. Any stage that skipped and needs to restore abc123 gets wrong content
```

## Solution

`_prepare_incremental_inputs` restores the prior artifact through
`WorkspaceStore.restore_from_cache`, which uses `CheckoutMode.COPY`:

```python
output_path = store.resolve_display_path(ref)
cache.remove_output(output_path)
if not store.restore_from_cache(ref, output_hash, state_dir=state_dir):
    raise exceptions.CacheRestoreError(...)
values[param_name] = _load_dep_from_path(ref, output_path)
```

The restored artifact is independent of its cache object before the stage
receives it.

The `CheckoutMode.COPY` ensures:
1. The restored file is a true copy with its own inode
2. Modifications don't affect the cached version
3. The file is writable (mode 0o644 vs cache's 0o444)

## Why Not Hardlink or Symlink?

Regular outputs are replaced after a stage returns, so sharing immutable cached
content is safe. Incremental stages may modify the restored file or directory
before returning their next result. They require a writable copy.

## Key Insight

Cache integrity depends on immutability: once a file is in cache, its content must never change. Hardlinks and symlinks provide efficient space sharing precisely because they point to the same content. For incremental outputs that need modification, copying is the only safe option. The performance cost of copying is acceptable because:

1. Incremental outputs are typically small (caches, indices, accumulators)
2. The alternative (cache corruption) has catastrophic consequences
3. The copy happens once per stage execution, not per file access
