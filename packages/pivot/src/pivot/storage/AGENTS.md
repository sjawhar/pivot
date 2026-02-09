# Pivot Storage - Development Guidelines

LMDB-backed state database for caching and skip detection.

## Skip Detection Algorithm

Three-tier algorithm for deciding whether to skip stage execution:

1. **O(1) generation tracking** - `worker.can_skip_via_generation()` checks if generation counter matches
2. **O(n) lock file comparison** - Compare fingerprint + params + dependency hashes
3. **Run cache lookup** - Check if input hash matches a cached run

## dep_generations: StateDB Only

Dependency generations (`{dep_path: generation}`) live **only in StateDB**, not in lock files. Workers compute them via `compute_dep_generation_map()` and return them in `DeferredWrites`. The coordinator applies them via `apply_deferred_writes()`. Old lock files that may have contained `dep_generations` are handled gracefully (field ignored on read).

## File-Hash Write-Back (DeferredWrites)

Workers open StateDB in readonly mode. Dependency hashing produces file hash entries that should be cached for future O(1) lookups. These are collected in `DeferredWrites.file_hash_entries` (a list of `(path, mtime_ns, size, inode, hash)` tuples) and written back to StateDB by the coordinator after stage completion.

## StateDB Key Prefixes

StateDB uses key prefixes for namespacing. Current prefixes in `pivot/storage/state.py`:

| Prefix | Purpose |
|--------|---------|
| `hash:` | File content hashes (keyed by resolved/physical path) |
| `gen:` | Output generation counters (keyed by logical/normpath) |
| `dep:` | Stage dependency generations (`stage:dep_path` → generation) |
| `runcache:` | Run cache entries (`stage:input_hash` → output hashes) |
| `sm:` | Fingerprint manifest cache (stage manifests + source maps) |
| `run:` | Run history entries |
| `remote:` | Remote index entries |
| `remote_url:` | Remote URL tracking for change detection |
| `fp:` | AST fingerprint/hash cache entries |

**When adding new state types**, define a new prefix constant and document it here.

## LMDB Specifics

- Single writer, multiple readers
- Transactions are mandatory—use context managers
- Map size must be set upfront (we use 10GB virtual default)
- Keys and values are bytes—use consistent encoding

## Concurrent Write Safety

LMDB allows only one writer at a time (process-wide via flock on `data.mdb`).
When multiple `pivot` processes share the same StateDB:
- Reads are always non-blocking (MVCC snapshots)
- Writes serialize — second writer blocks until first commits/aborts
- `StateDB._write_transaction()` wraps LMDB writes with an outer flock on `pivot-write.lock`
  - The outer flock provides **timeout detection only** (LMDB's `env.begin(write=True)` blocks indefinitely)
  - LMDB's internal write serialization (via `lock.mdb`) handles actual mutual exclusion
- On timeout: raises `PivotDBWriteTimeoutError` with diagnostic message

## Path Storage

All paths stored in the database must be **relative** to ensure portability across machines and correct cache behavior.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.
