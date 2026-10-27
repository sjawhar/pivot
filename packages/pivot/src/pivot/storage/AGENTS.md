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
| `nocommit:` | What Pivot wrote in a stage's outputs that no lock records yet (`stage` → its `*.pvt` directory members, path → content hash, each added before the file is written); a written lock removes it |

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

## Output Ownership

- Pivot replaces only what it owns. A tracked file, an imported file and their `.pvt` sidecars
  are the user's, even when one sits inside a stage output.
- Every path that removes or replaces a stage output in the workspace calls
  `ownership.refuse_output_replacements` (or `find_output_overlaps`) before it removes anything
  and raises `OutputOwnershipError` naming each file. A stage's output write checks all of the
  stage's outputs first (`worker._write_outputs_with_store` →
  `WorkspaceStore.refuse_output_replacements`), then each one again in
  `WorkspaceStore.prepare_output` just before clearing it. Also checked:
  `WorkspaceStore.restore_from_cache` (generation, lock and run-cache restores),
  `WorkspaceStore.commit` (the stage's own commit and `pivot commit`), an incremental output's
  restore, `pivot checkout` and `pivot pull`, `pivot get --force` and the presentation layer. Add
  the call to any new path that removes an output.
- A forced restore of a tracked or git path to its own place (`checkout --force`, `pull --force`,
  `get --force`) may replace that path's own content and nothing else: it is checked as an
  `ownership.tracked_restore`, which looks only beneath the path (not at its own sidecar or the
  sidecars above it) and refuses for every other tracked file, import or sidecar there. A stage
  output restored to its own path is checked as a stage output; an `--output` path (relative to
  the working directory) as a destination the user named. A restore requested through a stage
  (a stage-name target of `checkout` or `pull`, a stage output under `--all`) is that stage's
  output restore whatever else the path is (`checkout._validate_and_build_files`), so a tracked
  file, sidecar or import at or beneath it refuses it.
- A command checks every target before it changes any, and if it refuses one it changes nothing:
  `get --force` (`restore._refuse_forced_replacements`) and `checkout --force` and `pull --force`
  (`checkout._ownership_refusal`, which checks each target on its own so the refusal names every
  refused target and the files that block it). An `--only-missing` swap is checked in
  `_restore_path_locked`, under the restore lock, once the swap is decided; default checkout and
  `--only-missing` without a swap replace nothing.
- The check looks only where a replacement reaches: for a stage output or destination, the
  sidecar beside it and beside each directory above it (one that does not parse counts; one that
  parses counts when it names that path), and every `*.pvt` under a real directory. It never walks
  the project, so it does not see a `.pvt` outside an output whose `path` field names a file inside
  it. Pivot never writes one (`track`, `import` and `update` write the data file's own name), and
  catching a hand-edited one would need a project-wide walk on every output write.
- A directory under a replacement that Pivot cannot read is a blocking overlap
  (`ownership.UnreadableOverlap`): `os.walk` reports it through `onerror`, and Pivot refuses,
  naming it, rather than skip it, because a tracked file may hide inside where it cannot look.
- A stage may return a directory member named `*.pvt`. One is the stage's own when the stage
  records it (`ownership.StageMembers`): the lock's output hash, the hash a restore writes, this
  run's return value, or the stage's `nocommit:<stage>` record (`ownership.no_commit_members`, a
  file counted only while it holds the recorded content, since the record outlives a write the
  lock's outputs later replace). At a recorded path, a `.pvt` that does not parse is the stage's,
  and one that parses is the stage's only if its content is a recorded one; a `.pvt` at any other
  path is the user's. `pivot commit` passes the members the lock and the record list and nothing
  else, so it refuses, naming it, a `.pvt` that does not parse at any other path: were it to
  claim that file, the lock would list it and the next forced run would delete it. `status` and
  the engine read the lock and the record for this only when an output holds a `*.pvt`.
- Pivot never writes a stage's `*.pvt` member into the workspace before it is recorded as the
  stage's (`ownership.record_members` adds it to the `nocommit:` record), so a crash leaves at
  worst a record without a file. A run (`--no-commit` included) writes each member under a name
  that is not a `*.pvt`, records its content, then renames it into place
  (`worker._write_output_with_store`); a run-cache restore records before
  `WorkspaceStore.restore_from_cache` writes (`record_members`); `pivot get` of a stage records in
  `restore.restore_file`. Generation and lock restores, incremental restores, `checkout` and
  `pull` write only what the lock records. These writes only add to the record
  (`StateDB.add_no_commit_pvt_members` merges and keeps every other field); only a written lock
  removes it (`clear_no_commit_record`, from a run without `--no-commit` or `pivot commit`). An
  entry whose file a later write replaced claims nothing, by the content check above.
- A worker writes the StateDB itself in exactly two cases, both only for a stage whose directory
  output holds a `*.pvt` member: before it renames a member it wrote into place, and before a
  run-cache restore writes one. Every other worker write goes through `DeferredWrites`. The
  reason: a deferred write is applied only after the worker returns, so a crash in between would
  leave the stage's member in the workspace with no record, and every later check would call it
  a user's sidecar. The write is not best-effort: when it fails (the write lock times out, the
  database is full or read-only, it cannot be opened), `record_members` raises
  `MemberRecordError` naming the stage and the reason, nothing is moved into place, the staged
  copy is removed, and the stage fails.
- The check runs when a command starts replacing an output, not again just before each removal,
  for a restore, a symlink-mode commit, checkout, `get --force` and presentation; a sidecar created
  inside an output during that window is not seen. That is acceptable because `track` and
  `import` refuse to create a sidecar inside an output, so only a hand-made file can appear there.
- `track` and `import` load every pipeline (`discovery.declared_outputs`, as `--all` discovers
  them) and refuse a new path overlapping any declared output, naming the stage. A `pipeline.py`
  that fails to load fails them; one that loads but defines no Pivot pipeline declares nothing. A
  `pipeline.py` in a directory discovery excludes (`venv/`, `node_modules/`, …) is not counted;
  the store's check still protects its outputs. Re-tracking (`track --force`) or re-importing
  (`import --force`) a path already saved there warns and updates it in place. `status` and the
  engine warn, once per run, about a sidecar already inside an output they load.

## Path Storage

All paths stored in the database must be **relative** to ensure portability across machines and correct cache behavior.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.
