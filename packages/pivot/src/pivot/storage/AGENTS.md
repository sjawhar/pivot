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

## File-Hash Write-Back

Every command that writes the workspace records the hash of each file it reads, commits or restores, so `pivot status --explain` finds a matching entry and reads no file. The one exception is a run with `--no-commit`, which by design writes no lock, cache or StateDB entry. A new command that writes or reads workspace files must record them the same way:

- `hash_file(..., pending=...)` and `hash_directory(..., pending=...)` append `(path, stat, hash)` for each file they had to read instead of saving it; a caller with a writable StateDB saves them with one `save_many`.
- A restore records each file with the stat of what it staged, taken before the rename puts it in place (the same inode): `cache.restore_from_cache(..., recorded=...)`, `cache.atomic_write_file`'s return value. A write that lands after the rename then changes the stat, so the entry does not vouch for it.
- Workers open StateDB read-only. They hand back, in `DeferredWrites.file_hash_entries` (`(path, mtime_ns, size, dev, inode, hash)` tuples, `types.FileHashEntry`, from `cache.deferred_file_hash_entries`), the dependencies and outputs they read to decide (also when the stage is cached by lock comparison), the outputs they commit and the outputs they restore; the coordinator writes them after the stage. The coordinator's own skip check writes what it reads and restores.
- `pivot checkout` (every mode, also the files `--only-missing` keeps or restores and the existing files it compares), `pivot pull`, `pivot get` (from the cache, from git or from the remote), `pivot import`, `pivot update`, `pivot track` and repro's restore of missing tracked files (`--checkout-missing`) record what they write or read. Stage outputs go in the StateDB of the output's stage state directory, which explanations read; everything else in the configured state directory.
- `pivot status` and every explanation (`explain.get_stage_explanation`, whose callers are status, the engine and the RPC server) read the StateDB read-only and save the hashes they had to read afterwards, one write per state directory, with `state.write_back_file_hashes`. That write is best-effort: a write lock another process holds for longer than 2 s (the engine makes one attempt, and none in a `--no-commit` run), a database this user cannot write, a read-only filesystem or a full map skips it with a debug log, never a failure. It never creates a database. A database whose data file this user cannot write opens read-only with LMDB's lock file, and a writable `StateDB` on it raises `StateDBError`; when the lock file cannot be written (another user's, a read-only filesystem), `state.open_readonly` yields None and the files are read instead.
- Every write of a `hash:` entry also writes the identity index (`inode:`), after all the batch's `hash:` entries so their pages lie together, and drops the path's old inode's index entry. A lookup that misses by path follows it: a replica made outside pivot that hard-links pivot's read-only outputs finds them without reading them. BET's replica copies `.pivot/state.lmdb`, so its entries name the original's paths; `cp -al` hard-links it, so the copy and the original share one database. `_make_key_file_identity` states why that lookup is never weaker than the path lookup.

## StateDB Key Prefixes

StateDB uses key prefixes for namespacing. Current prefixes in `pivot/storage/state.py`:

| Prefix | Purpose |
|--------|---------|
| `hash:` | File content hashes (keyed by the absolute path as given, `os.fsencode(os.path.abspath(path))`; the value's `(mtime_ns, size, dev, inode)` comes from `stat()`, which follows symlinks. No ctime: a new hard link changes it, so every hardlink checkout or hard-linked replica would invalidate its entries) |
| `inode:` | Identity index: `(st_dev, st_ino)` → the `hash:` key last written for that inode. A hint; a lookup serves that entry only if it matches the file's stat and its path still names the same device and inode |
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
  into place, checked, then marked (`<tree>.verified`), because a restore symlinks to a tree as
  soon as its name exists. A restore or commit checks an unmarked tree (exact files, sizes,
  executable bits and directories) under the tree's exclusive `<tree>.lock`, repairs it in place
  from the files' own cache objects, then marks it. A tree it cannot repair is not used: a
  restore goes file by file, a commit fails. `<tree>.layout-checked`, the mark pivot 0.3.4 also
  gave partial trees, is not trusted.

## Path Storage

Generation, dependency and run-cache keys hold artifact identities and project-relative paths, so they mean the same thing on every machine. File-hash keys and the identity index hold absolute paths on purpose: an entry describes one file on this machine, checked by its `stat`.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.
