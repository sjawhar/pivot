---
tags: [python, paths, database, caching]
category: design
module: storage
symptoms: ["duplicate cache entries", "generation lookup misses", "symlink path confusion"]
---

# StateDB Path Strategies: `resolve()` for Hashes, `normpath()` for Generations

## Problem

Pivot's StateDB stores two types of path-keyed data with fundamentally different requirements:

1. **File hash cache** - Maps paths to content hashes for skip detection
2. **Generation counters** - Tracks how many times each output has been produced

Using the same path resolution strategy for both causes subtle bugs:

```python
# Project structure after stage execution:
# output.csv -> .pivot/cache/abc123/output.csv  (symlink to cache)

# If we use resolve() for both:
path = Path("output.csv")
path.resolve()  # Returns .pivot/cache/abc123/output.csv

# Hash lookup: CORRECT - deduplicates symlinks pointing to same file
# Generation lookup: WRONG - key changes every time hash changes!
```

The generation counter key would change from `gen:output.csv` to `gen:.pivot/cache/abc123/output.csv` after the first run, then to `gen:.pivot/cache/def456/output.csv` after the content changes. Each run creates a "new" output path as far as generation tracking is concerned.

Conversely, using `normpath()` for hash lookups:

```python
# Two symlinks to same physical file:
# data/input.csv -> /shared/datasets/sales.csv
# archive/sales.csv -> /shared/datasets/sales.csv

# If we use normpath() for hash lookups:
path1 = Path("data/input.csv")      # normpath: data/input.csv
path2 = Path("archive/sales.csv")   # normpath: archive/sales.csv

# We compute and store the same hash TWICE - wasted work
```

## Solution

Use different path resolution strategies based on the semantic meaning of the lookup:

```python
def _make_key_file_hash(path: pathlib.Path) -> bytes:
    """Create LMDB key for file hash entry (follows symlinks for physical deduplication).

    Uses resolve() to follow symlinks because hash caching is about physical file identity.
    Multiple symlinks pointing to the same file should share one cached hash.
    """
    return _HASH_PREFIX + str(path.resolve()).encode()


def _make_key_output_generation(path: pathlib.Path) -> bytes:
    """Create LMDB key for output generation entry (preserves symlinks for logical path tracking).

    Uses normpath(absolute()), NOT resolve(), because Pivot outputs become symlinks
    to cache after execution. resolve() would follow these symlinks to cache paths
    that change per-run. We track the LOGICAL path the user declared.
    """
    return _GEN_PREFIX + os.path.normpath(path.absolute()).encode()
```

The key difference:

| Strategy | Function | Purpose | Example |
|----------|----------|---------|---------|
| `resolve()` | File hash lookup | Physical identity dedup | Symlinks to same file share one hash |
| `normpath(absolute())` | Generation tracking | Logical path identity | `output.csv` stays `output.csv` even when symlinked to cache |

In practice for Pivot:

```python
# After stage execution, output.csv is a symlink:
# output.csv -> .pivot/cache/abc123/output.csv

# Hash lookup (resolve):
hash_key = "hash:/home/user/project/.pivot/cache/abc123/output.csv"
# Correct! If another file links here, they share the cached hash.

# Generation lookup (normpath):
gen_key = "gen:/home/user/project/output.csv"
# Correct! Tracks the declared output path, not where it points.
```

## Key Insight

Path resolution strategy must match the **semantic meaning** of the operation:

- **Content-addressed lookups** (hashes) care about **physical identity** - use `resolve()` to follow symlinks and deduplicate
- **Logical path lookups** (generations, dependencies) care about **declared identity** - use `normpath()` to preserve the path as the user/system specified it

This is why Pivot outputs become symlinks to the cache: it allows the same physical content to be reused across runs while maintaining stable logical paths for tracking. The two path strategies work together to make this transparent.
