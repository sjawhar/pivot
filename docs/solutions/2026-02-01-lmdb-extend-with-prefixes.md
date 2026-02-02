---
tags: [python, lmdb, database, architecture]
category: design
module: storage
symptoms: ["state scattered across files", "inconsistent state updates", "complex state management"]
---

# LMDB for All State: Extend StateDB with Prefixes

## Problem

A build system needs to track multiple types of state: file hashes for change detection, generation counters for outputs, dependency relationships between stages, run history, and run cache entries. The naive approach creates separate storage for each:

```python
# Scattered state - each concern in a separate file/database
hash_cache = HashCache("hashes.db")
generation_db = GenerationDB("generations.db")
dep_tracker = DependencyTracker("deps.json")
run_history = RunHistoryDB("runs.db")
```

This causes problems:

1. **Non-atomic updates** - After a stage runs, you need to update generations, record dependencies, and write run cache. If the process crashes between writes, state becomes inconsistent.

2. **Complex cleanup** - Garbage collection must coordinate across multiple stores to avoid dangling references.

3. **Multiple file handles** - Each database needs its own connection management, increasing resource usage and complexity.

4. **Testing burden** - Tests must mock or manage multiple storage backends.

## Solution

Use a single LMDB database with key prefixes to namespace different data types:

```python
# Key prefixes for different entry types
_HASH_PREFIX = b"hash:"      # File hash entries
_GEN_PREFIX = b"gen:"        # Output generation counters
_DEP_PREFIX = b"dep:"        # Stage dependency generations
_REMOTE_PREFIX = b"remote:"  # Remote index entries
_RUN_PREFIX = b"run:"        # Run history entries
_RUNCACHE_PREFIX = b"runcache:"  # Run cache for skip detection
_FP_PREFIX = b"fp:"          # AST fingerprint cache
```

Each data type gets methods that use the appropriate prefix:

```python
def _make_key_file_hash(path: pathlib.Path) -> bytes:
    return _HASH_PREFIX + str(path.resolve()).encode()

def _make_key_output_generation(path: pathlib.Path) -> bytes:
    return _GEN_PREFIX + os.path.normpath(path.absolute()).encode()

def _make_key_dep_generation(stage_name: str, dep_path: str) -> bytes:
    return _DEP_PREFIX + f"{stage_name}:{dep_path}".encode()
```

Atomic multi-type updates happen in a single transaction:

```python
def apply_deferred_writes(
    self,
    stage_name: str,
    output_paths: list[str],
    deferred: DeferredWrites,
) -> None:
    """Apply all deferred writes in a single atomic transaction."""
    with self._env.begin(write=True) as txn:
        # Dependency generations
        if "dep_generations" in deferred:
            for dep_path, gen in deferred["dep_generations"].items():
                key = _make_key_dep_generation(stage_name, dep_path)
                txn.put(key, struct.pack(">Q", gen))

        # Output generations (increment)
        for path_str in output_paths:
            key = _make_key_output_generation(pathlib.Path(path_str))
            value = txn.get(key)
            current = struct.unpack(">Q", value)[0] if value else 0
            txn.put(key, struct.pack(">Q", current + 1))

        # Run cache
        if "run_cache_input_hash" in deferred:
            key = _RUNCACHE_PREFIX + f"{stage_name}:{...}".encode()
            txn.put(key, run_history.serialize_to_bytes(...))
```

Prefix-scoped iteration enables efficient cleanup and queries:

```python
def get_dep_generations(self, stage_name: str) -> dict[str, int] | None:
    """Get all dependency generations for a stage using cursor iteration."""
    prefix = _DEP_PREFIX + stage_name.encode() + b":"
    results = dict[str, int]()
    with self._env.begin() as txn:
        cursor = txn.cursor()
        if cursor.set_range(prefix):  # Position at first matching key
            for key, value in cursor:
                if not key.startswith(prefix):
                    break  # Past our prefix range
                dep_path = key[len(prefix):].decode()
                results[dep_path] = struct.unpack(">Q", value)[0]
    return results if results else None
```

## Key Insight

LMDB's sorted key-value model makes prefix namespacing efficient: `cursor.set_range(prefix)` jumps directly to the first matching key, and iteration stops when keys no longer match the prefix. This gives you the benefits of separate logical databases (isolation, scoped queries) with the benefits of a single physical database (atomic cross-namespace transactions, single file handle, unified backup).

When adding new state types to a system, prefer extending an existing key-value store with a new prefix over creating a new database. The consistency guarantees of atomic transactions across all state types outweigh the apparent cleanliness of separate storage.
