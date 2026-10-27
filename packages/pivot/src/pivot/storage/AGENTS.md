# Pivot Storage - Development Guidelines

LMDB-backed state database for caching and skip detection.

## Skip Detection Algorithm

Three-tier algorithm for deciding whether to skip stage execution:

1. **O(1) generation tracking** - `worker.can_skip_via_generation()` checks if generation counter matches
2. **O(n) lock file comparison** - Compare fingerprint + params + dependency hashes
3. **Run cache lookup** - Check if input hash matches a cached run

For status and explain, a stage is stale when repro will rewrite its lock by running it or restoring it from the run cache; it is cached when its lock already matches the pipeline and repro can at most restore outputs.

## dep_generations: StateDB Only

Dependency generations (`{dep_path: generation}`) live **only in StateDB**, not in lock files. Workers compute them via `compute_dep_generation_map()` and return them in `DeferredWrites`. The coordinator applies them via `apply_deferred_writes()`. Old lock files that may have contained `dep_generations` are handled gracefully (field ignored on read).

## File-Hash Write-Back (DeferredWrites)

Workers open StateDB in readonly mode. Dependency hashing produces file hash entries that should be cached for future O(1) lookups. These are collected in `DeferredWrites.file_hash_entries` (a list of `(path, mtime_ns, size, inode, hash)` tuples) and written back to StateDB by the coordinator after stage completion.

## StateDB Key Prefixes

StateDB uses key prefixes for namespacing. Current prefixes in `pivot/storage/state.py`:

| Prefix | Purpose |
|--------|---------|
| `hash:` | File content hashes (keyed by resolved/physical path) |
| `gen:` | Output generation counters (keyed by artifact identity, `stage` or `stage:key`) |
| `dep:` | Stage dependency generations (`stage:dep_identity` → generation; one file of a directory output records its directory's identity) |
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

## Restore Locking

- Restores with a `state_dir` take an exclusive per-target `flock` in `state_dir/locks/`.
- Restore lock files persist. Do not unlink or age-delete them: a waiter can hold the
  old inode while a new opener locks a replacement inode, defeating mutual exclusion.
- File restores stage their link or copy beside the target, under a name tagged with the
  target's token, and atomically replace it. Directory restores build in a private
  temporary directory and swap it. Both run under the target's restore lock, and each
  first removes the token-tagged entries a killed restore of the same target left.

## Publishing Cache Objects

- Every writer of a file object publishes it with `cache.publish_cache_object`, which links it
  into place and keeps an object already there. A published object never changes, mode
  included: a hardlink checkout looks the name up, then links the inode it found, and once a
  rename takes that inode's last name Linux fails the link with ENOENT, though the name never
  went missing; a symlink checkout reads its executable bit through the name. A directory
  entry whose executable bit the object does not carry is checked out as a copy.
- On a filesystem without hard links, `publish_cache_object` renames, and only when no object is
  there yet. Nothing there can hardlink an object, but the check and the rename are two steps:
  two writers that both find the name free both rename, the later object wins, and a symlink
  checkout reads the later object's mode.
- Every temporary entry in a prefix directory (a file staged for an object, a download, a tree
  being copied) is made through `cache.staged_cache_file` or `_staging_in`, named
  `.pivot_tmp_*`, while the process holds a shared `flock` on the directory's
  `.pivot_cache.lock`. A process that gets the exclusive lock first (it never waits for it), at
  most once per `_CACHE_SWEEP_INTERVAL` (the lock's mtime records the last sweep), removes every
  `.pivot_tmp_*` entry: none has a live owner. An entry it cannot remove is logged and left.
  Never judge a temporary entry abandoned by host and pid: a process in another PID namespace,
  or on another host sharing the cache, looks dead.
- Sharing a cache across hosts needs `flock` that is coherent across them (NFSv4, or NFSv3 with
  NLM). With host-local locks (NFS `nolock` or `local_lock=flock`/`all`, Lustre `localflock`)
  the locks here exclude nothing across hosts: one host's sweep can remove another host's live
  temporary entry, and two hosts can repair one tree at once. A publisher checks its tree before
  marking it, which narrows the window for marking a copy that lost files without closing it: a
  sweep already deleting through the copy's directory descriptors keeps deleting after the
  rename. Where `flock()` is refused (Lustre without `flock`: ENOSYS; NFS when its lock manager
  does not answer: ENOLCK) staging goes on without the shared lock, and this process neither
  sweeps nor repairs a tree. Only where every host sharing the cache is refused does nothing
  sweep: an NFSv3 client whose own `rpc.statd` is down gets ENOLCK while other clients' locks
  work, and their sweeps can remove its live temporary entries, with the effects above.
- A lock file in the cache is created writable by whoever can write its directory, since NFS
  takes an exclusive lock only on a file opened for writing; one this user cannot write is
  opened read-only. Lock files are never deleted.
- A whole-directory SYMLINK tree is built under a temporary name beside its cache name, renamed
  into place, checked, then marked (`<tree>.content-checked`), because a restore symlinks to a
  tree as soon as its name exists. A restore or commit checks every unmarked tree member's path,
  content hash, size and executable bit, plus its directories, under the tree's exclusive
  `<tree>.lock`; it repairs mismatches in place from the files' own cache objects, then marks the
  tree. A tree it cannot check or repair (a file's object is absent, or this user cannot read or
  write the tree) is not used: a restore goes file by file, a commit fails. Only
  `<tree>.content-checked` is trusted: `<tree>.layout-checked` (pivot 0.3.4, partial trees
  included) and `<tree>.verified` (pivot 0.3.5 to 0.5.11, sizes only) are older marks that never
  checked content, so such a tree is checked again and the old mark removed. The mark is only
  written where this user can write the tree's prefix directory: for a user who cannot, every
  symlink restore or commit checks an old-marked tree again, and a symlink commit through a
  stale one fails, until a user who can write the cache runs one
  `pivot checkout --checkout-mode symlink` through it. That cost is the price of not trusting a
  size-only mark.

## Path Storage

All paths stored in the database must be **relative** to ensure portability across machines and correct cache behavior.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.
