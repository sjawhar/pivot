---
tags: [python, caching, filesystem, incremental]
category: design
module: outputs
symptoms: ["cache corruption", "unexpected file modifications", "stale data in cache"]
---

# IncrementalOut Uses COPY Mode for Cache Restoration

## Problem

`IncrementalOut` is designed for outputs that accumulate data across runs. Before each execution, the previous output is restored from cache so the stage can read, modify, and write back. However, if restored via hardlink or symlink:

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

`IncrementalOut` always uses `CheckoutMode.COPY` when restoring from cache:

```python
def _prepare_outputs_for_execution(
    stage_outs: Sequence[outputs.BaseOut],
    lock_data: LockData | None,
    files_cache_dir: pathlib.Path,
) -> None:
    for out in stage_outs:
        path = pathlib.Path(cast("str", out.path))

        if isinstance(out, outputs.IncrementalOut):
            cache.remove_output(path)  # Clear stale state
            out_hash = output_hashes.get(str(out.path))
            if out_hash:
                # COPY mode: creates independent writable file
                restored = cache.restore_from_cache(
                    path, out_hash, files_cache_dir, cache.CheckoutMode.COPY
                )
                if not restored:
                    raise exceptions.CacheRestoreError(...)
        else:
            # Regular Out: just delete before run
            cache.remove_output(path)
```

The `CheckoutMode.COPY` ensures:
1. The restored file is a true copy with its own inode
2. Modifications don't affect the cached version
3. The file is writable (mode 0o644 vs cache's 0o444)

## Why Not Hardlink/Symlink for Regular Outputs?

Regular `Out` uses hardlink/symlink because outputs are typically replaced atomically, not modified in place. The save-to-cache flow is:

1. Stage writes new file
2. Pivot hashes and copies to cache
3. Pivot replaces original with hardlink/symlink to cache

This is safe because step 1 creates a fresh file (not modifying the linked one).

## Key Insight

Cache integrity depends on immutability: once a file is in cache, its content must never change. Hardlinks and symlinks provide efficient space sharing precisely because they point to the same content. For incremental outputs that need modification, copying is the only safe option. The performance cost of copying is acceptable because:

1. Incremental outputs are typically small (caches, indices, accumulators)
2. The alternative (cache corruption) has catastrophic consequences
3. The copy happens once per stage execution, not per file access
