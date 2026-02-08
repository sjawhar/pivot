---
tags: [python, caching, skip-detection, statedb, fingerprinting]
category: architecture
module: executor, storage
symptoms: ["stage re-runs unexpectedly", "stage skips when it shouldn't", "generation check falls through to hashing"]
---

# Skip Detection Invariants

## Overview

Skip detection decides whether a stage needs to re-execute. It uses a three-tier algorithm with increasing cost, short-circuiting at the first conclusive tier.

## Three-Tier Algorithm

### Tier 1: O(1) Generation Check

`worker.can_skip_via_generation()` runs **before** any dependency hashing. It compares monotonic generation counters stored in StateDB.

Every time a stage's outputs are committed, the coordinator increments a generation counter for each output path. When a downstream stage checks whether to skip, it compares the current generation of each dependency against the generations recorded when the stage last ran.

**Preconditions for generation skip:**
- Lock file exists with matching code manifest (`fingerprint`) and params
- Output paths in lock file match current output paths
- All dependency paths have generation counters in StateDB
- All dependency generations match the recorded values from last run
- File metadata in StateDB matches on-disk stat (catches external modifications)

If **any** of these fail, the generation check returns `False` and execution falls through to Tier 2.

### Tier 2: O(n) Lock File Comparison

`StageLock.is_changed_with_lock_data()` compares the full lock file state:
- Code manifest (fingerprint of stage function + helper functions)
- Parameters (effective params after overrides)
- Dependency hashes (content hashes of all input files/directories)
- Output path list (detect added/removed outputs)

This requires hashing all dependency files, hence O(n) in the number/size of dependencies.

### Tier 3: Run Cache Lookup

If Tier 2 determines the stage must run (e.g., a dependency changed), the run cache is checked before actual execution. The run cache maps `(fingerprint + params + dep_hashes + out_specs) → output_hashes`, so if the same configuration was previously executed (even in a different pipeline run), outputs can be restored from cache without re-executing.

The run cache key (`input_hash`) includes output cache flags, so toggling `cache=True/False` on an output produces a different key.

## Pivot-Produced Artifact Boundary

Generation skip only works when **all** dependencies have generation counters in StateDB. Counters are created by the coordinator when a stage's outputs are committed — meaning only **Pivot-produced artifacts** have them.

External files (user-created data, config files, files not produced by any Pivot stage) lack generation counters. When a stage depends on external files:

1. `can_skip_via_generation()` finds `current_gen is None` for the external dep
2. Returns `False`, falling through to hash-based comparison (Tier 2)
3. Tier 2 hashes the file and compares against the lock file

This is by design: external files can change at any time without Pivot's knowledge, so the only reliable check is content hashing.

## dep_generations: StateDB Only

Dependency generations (`{dep_path: generation}`) are stored **only in StateDB**, not in lock files. The `LockData` TypedDict contains `code_manifest`, `params`, `dep_hashes`, and `output_hashes` — no `dep_generations` field.

Workers compute dependency generations via `compute_dep_generation_map()` and return them in `DeferredWrites`. The coordinator applies these to StateDB via `apply_deferred_writes()`. Old lock files that may have contained `dep_generations` are handled gracefully — the field is simply ignored during lock file reading since `LockData` doesn't declare it.

## File-Hash Write-Back

Workers open StateDB in **readonly mode** during execution to avoid write contention. However, dependency hashing produces file hash entries (path, mtime_ns, size, inode, hash) that should be cached in StateDB for future O(1) lookups.

The solution: workers collect these entries in `file_hash_entries` (returned from `hash_dependencies()`) and include them in `DeferredWrites`. The coordinator writes them back to StateDB after stage completion via `apply_deferred_writes()`. This gives the performance benefit of hash caching without requiring workers to hold write locks on StateDB.

`DeferredWrites.file_hash_entries` is a list of `(path, mtime_ns, size, inode, hash)` tuples. The coordinator writes these entries back to StateDB as provided. If a file changed between the worker's read and the coordinator's write, the entry may contain stale metadata — the next run will detect the mismatch and re-hash (self-healing).

## Logical Path vs Physical Identity

Lock files and StateDB use different path semantics:

- **Lock files** store project-relative paths. In memory, paths are first normalized to absolute form via `project.normalize_path()`, and `storage/lock.py` converts between those absolute paths and the project-relative paths written to (and read from) the lock file. Symlinks are preserved. Using project-relative paths ensures portability across machines and stable lock file content.

- **StateDB file hash cache** uses `path.resolve()` (follows symlinks) for the key. This means multiple symlinks to the same physical file share one cached hash entry, avoiding redundant hashing.

- **StateDB generation counters** use `os.path.normpath(path.absolute())` (preserves symlinks). This tracks the *logical* path the user declared, not where symlinks point. This is important because Pivot outputs become symlinks to cache after execution — `resolve()` would follow them to cache paths that change per-run.

## Manifest Cache

Fingerprint manifests (the code_manifest dict mapping `"self:func_name"` → hash) are cached in StateDB under the `sm:` prefix, along with source maps that record which source files contributed to the manifest.

**Flush boundaries** (where pending manifest cache writes are committed to StateDB):
- After discovery (`discovery.py` — when stages are found via `pivot.yaml`/`pipeline.py`)
- After YAML config load
- After module load
- In watch mode, before reload

**Selective invalidation in watch mode:** When a source file changes, the engine uses a reverse index to find which manifest cache entries depend on that file, and invalidates only those entries. This avoids re-fingerprinting all stages when only one source file changed. The reverse index is built by scanning all `sm:` entries' source maps.

## Safe Fingerprinting

Pivot fingerprints stage functions by hashing their AST and the ASTs of any helper functions they call. Closures that capture **mutable** variables (lists, dicts, instances) are problematic because their runtime state isn't tracked by the AST hash — a change to a captured list won't trigger re-execution.

By default, mutable closure captures raise `StageDefinitionError` with a message explaining the issue and suggesting alternatives (StageParams or Dep inputs).

**Escape hatches:**
- Config: `core.unsafe_fingerprinting: true` in `pivot.yaml`
- Environment: `PIVOT_UNSAFE_FINGERPRINTING=1`

When enabled, mutable captures produce a warning instead of an error. This is unsafe because it can lead to silent wrong outputs — the captured value may change without triggering re-execution.

Immutable captures (strings, numbers, tuples, frozen dataclasses, frozensets) are always allowed since they can't change after creation.
