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
| `nocommit:` | What a stage's outputs hold that no lock records yet (`stage` → a JSON object, `types.NoCommitRecord`): a `--no-commit` run's code manifest, marks included, and output hashes, for `pivot commit`, and the `*.pvt` directory members Pivot wrote (`pvt_members`, path → content hash, each added before the file is written); a written lock removes the whole record. Each writer replaces only its own keys and keeps every other key, in one write transaction. The reader validates each of the three keys on its own, dropping only a malformed one, and returns every other key as stored, so a record holding only another feature's key reads as present; `pivot commit` uses the run's manifest only when both of the run's keys are present |

**When adding new state types**, define a new prefix constant and document it here.

## The `.pvt` Walk Memo

`track.discover_pvt_files` must find every `.pvt` that `os.walk` over the project finds. Never prune it by directory name or by stage output: `track` and `import` accept paths under `.git`, `.pivot` and other pipelines' outputs. The memo makes an unchanged directory cost one `lstat` instead of a listing:

- A listing is reused only when the directory's stamp (`st_dev`, `st_ino`, `st_mtime_ns`, `st_ctime_ns`) matches, its mtime and ctime are at least 2 s before the walk start (a later change in the same timestamp tick would leave the stamp unchanged), its mtime and ctime are not both whole seconds (on a one-second-timestamp filesystem a clock stepped back into a recorded second would keep the stamp; such filesystems are always listed), both fit the stamp's signed 64-bit nanoseconds (1677 to 2262; any other time is listed every walk), and its mount type is on the allowlist read from `/proc/self/mountinfo`. A device mountinfo does not list (a btrfs subvolume reports its own) counts as btrfs only when the deepest mount holding the directory is btrfs.
- It holds names only. Every `.pvt` found is read on every walk, since rewriting a file does not change its directory.
- Only the repro/run pre-flight (`executor/core.py` `verify_tracked_files`) writes it, and only when the walk's listings differ from it. It writes through the descriptor (`os.write`) under an `flock` (when another pre-flight holds it, this one skips its write) and empties the file first, so a walk reading it meanwhile, or after a pre-flight killed mid-write, sees a prefix of the new memo. A prefix never parses (it ends inside a record, or holds fewer listings than the root record counts), so that walk lists every directory. A write that fails partway (a full disk) empties the file again, so no cut memo keeps the space. Every other caller only reads it, and reading never creates anything. A memo that cannot be read means every directory is listed.
- It is not in the state database, which other accounts can read. It is a file of the running account's own, `state.lmdb/pvt-memo-<uid>` in the state directory (`.pivot/.gitignore` keeps `state.lmdb/` out of git), created with mode 0600. The pre-flight creates it only in a `state.lmdb` directory that already exists and is on an allowlisted filesystem, and rewrites it in place, so that directory's stamp does not change at each write. A walk opens it without following a symlink (`O_NOFOLLOW`, and `O_NONBLOCK` so a FIFO there cannot block it) and uses it, for reading or writing, only when an `fstat` of the open file shows a regular file, on a filesystem of the walk's allowlist (`_MEMO_FS_TYPES` through `_device_allowed`, with mountinfo read once per walk; there owner and mode are the file's own and the kernel enforces them, while bindfs, CIFS and NFS can report the mount's), owned by the running account, with no group or other permission bits (a POSIX ACL's mask shows in the group bits) and one link. Otherwise the walk lists every directory, writes nothing, and logs the reason at debug level. A file of another account, of another kind or on another filesystem is never written, replaced or removed. When the account's own memo fails only the mode or link check (its owner loosened it, `chmod -R go+rX .pivot` say), the read-only walk only refuses it, and the pre-flight empties it, unlinks it and creates a new 0600 file, so its names leave that path, every descriptor opened on it and any other link to it. So no other account reads the memo's names, whatever the modes of the project's directories and wherever the state directory or `state.lmdb` points, except while the owner leaves the file loose, until the next pre-flight replaces it. The cost is that `open` and one `fstat` per walk (one more when a loosened memo is replaced). Each account keeps its own memo in a shared state directory.
- A listing in which an entry's type could not be read (the `lstat` a filesystem without `d_type` needs failed) is never recorded, so a walk that can read that type never reuses it.

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

## Output Ownership

- Pivot replaces only what it owns. A tracked file, an imported file and their `.pvt` sidecars
  are the user's, even when one sits inside a stage output.
- Every path that removes or replaces a stage output in the workspace calls
  `ownership.refuse_output_replacements` (or `find_output_overlaps`) before it removes anything
  and raises `OutputOwnershipError` naming each file. A stage that writes more than one output
  checks all of them first (`worker._write_outputs_with_store` →
  `WorkspaceStore.refuse_output_replacements`), so a refusal changes none of them, and then each
  one again in `WorkspaceStore.prepare_output` just before clearing it. A stage with one output
  gets the `prepare_output` check alone: nothing happens between the two. Also checked:
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
- Every forced restore makes one decision, `ownership.forced_restore_refusal`, over targets built
  once per command (`checkout._restore_targets` for `checkout` and `pull`,
  `restore._restore_targets` for `get`; a `get` target whose `.pvt` is not valid is only reported,
  never replaced, so it is not a target). The command checks every target before it changes any,
  and if it refuses one it changes nothing; each target is checked on its own, so the refusal
  names every refused target and the files that block it. Then each target is checked again,
  under its restore lock, just before its replacement is built (`checkout._restore_path_locked`,
  `restore._replace_checked`): a user file that appeared since the first check refuses that
  target alone, naming it, and the command exits 1, before any work is spent on it. Last, its
  swap checks the entry it removes (`ownership.swap_refusal`, passed to
  `cache.restore_from_cache` as `before_swap`): a directory restore renames the old entry aside
  first, as does a file restore over a directory, and the check walks it there, so a file that
  landed in the target while its replacement was built refuses it too; the entry goes back to
  its path and nothing is removed. A file that replaces a file or symlink is checked in place,
  just before the rename. Only a path that exists is checked. An `--only-missing` swap is checked
  the same way, once the swap is decided; default checkout and `--only-missing` without a swap
  replace nothing.
- The check looks only where a replacement reaches: for a stage output or destination, the
  sidecar beside it and beside each directory above it (one that does not parse counts; one that
  parses counts when it names that path), and every `*.pvt` under a real directory. It never walks
  the project, so it does not see a `.pvt` outside an output whose `path` field names a file inside
  it. Pivot never writes one (`track`, `import` and `update` write the data file's own name), and
  catching a hand-edited one would need a project-wide walk on every output write.
- The common case, a project where no output holds or sits under a `.pvt`, costs one `lstat` per
  path the check looks at: `ownership._lstat_mode` answers "nothing there" for a sidecar and
  "a regular file" for an output, and a `.pvt` that does exist, or an output that is not a plain
  file, goes on to the `pathlib` predicates, which decide (and raise) as before. Keep it that way:
  a `.pvt` the walk would read, or a claim it would build, is work every run of every project
  pays. `WorkspaceStore` resolves each output's path once per ref, since a stage asks for it at
  its write, at each ownership check and at its commit.
- A directory under a replacement that Pivot cannot read is a blocking overlap
  (`ownership.UnreadableOverlap`): `os.walk` reports it through `onerror`, and Pivot refuses,
  naming it, rather than skip it, because a tracked file may hide inside where it cannot look.
- A stage may write a `*.pvt` file in a directory output: a `directory` output's key, or a file
  inside a directory a `PathOnly` output returns. One is the stage's own when the stage records it
  at that path, by the one rule `ownership.stage_members` (every write, restore, commit, warning
  and forced restore uses it): the hash the stage's lock records for the output, the hash a
  restore writes, what this run writes, or the stage's `nocommit:<stage>` record (a file counted
  only while it holds the recorded content, since the record outlives a write the lock's outputs
  later replace). The claims are read once per run or command (`ownership.StageClaims`). Whether
  an output is a directory is read from those hashes and from what the run writes, never from its
  tag. At a recorded path, a `.pvt` that does not parse is the stage's, and one that parses is
  the stage's only if its content is a recorded one, or, at a path the lock or the restored hash
  lists, a content Pivot holds in the cache (a lock that changed through git lists another run's
  content there; replacing a cached file loses nothing the cache cannot restore). A `.pvt` at any
  other path is the user's. `pivot commit` passes the claims and nothing else, so it refuses,
  naming it, a `.pvt` that does not parse at any other path: were it to claim that file, the lock
  would list it and the next forced run would delete it. A refusal fails that stage alone; the
  other stages are still committed. `status` and the engine read the claims only when an output
  holds a `*.pvt`.
- Pivot never writes a stage's `*.pvt` member into the workspace before it is recorded as the
  stage's (`ownership.record_members` adds it to the `nocommit:` record), so a crash leaves at
  worst a record without a file in the output. A run (`--no-commit` included) writes each output
  entry that brings one (a `*.pvt` key, a `PathOnly` directory holding one) under a name beside
  the output, never inside it, that is not a `*.pvt` (`worker._staging_path`), records the
  content of each `*.pvt` in it, then renames it into place (`worker._save_recorded`). A run
  killed before that rename leaves the staged copy (`.<output>.pivot-staged…`) beside the output,
  where no check of the output looks, so the next run or restore of the output is not refused;
  Pivot never removes that copy. A restore records the members of the hash it writes unless the
  stage's lock records that hash (`ownership.record_restored`): a run-cache restore in
  `WorkspaceStore.restore_from_cache`, `pivot get` of a stage in `restore.restore_file`.
  Generation and lock restores, incremental restores, `checkout` and `pull` write only what the
  lock records. These writes only add to the record (`StateDB.add_no_commit_pvt_members` merges
  and keeps every other field); only a written lock removes it (`clear_no_commit_record`, from a
  run without `--no-commit` or `pivot commit`). An entry whose file a later write replaced claims
  nothing, by the content check above.
- A worker writes the StateDB itself in exactly two cases, both only for a stage whose directory
  output holds a `*.pvt` member: before it renames a member it wrote into place, and before a
  run-cache restore writes one. Every other worker write goes through `DeferredWrites`. The
  reason: a deferred write is applied only after the worker returns, so a crash in between would
  leave the stage's member in the workspace with no record, and every later check would call it
  a user's sidecar. The write is not best-effort: when it fails (the write lock times out, the
  database is full or read-only, it cannot be opened), `record_members` raises
  `MemberRecordError` naming the stage and the reason, nothing is moved into place, the staged
  copy is removed, and the stage fails.
- A forced restore checks each target three times: with every target before any is changed,
  under its restore lock before its replacement is built, and at its swap over the entry the swap
  removes, renamed aside where a directory is involved. Once renamed, no writer that names its
  files by path can add one to that entry, so a file copied into a target while the command runs
  refuses that target. What it can still miss: a writer holding an open handle on a directory
  inside the target (its working directory, or a directory descriptor), which can add a file to
  the renamed entry after the walk; a sidecar created beside the target, or beside a directory
  above it, in the instant between the swap's check and its rename; and a restore killed between
  renaming the old entry aside and checking it, whose backup the next restore of that path (or a
  directory restore in the same directory an hour later) removes unchecked
  (`cache._cleanup_fresh_restore_orphans`, `cache._cleanup_stale_restore_temps`). An entry the
  check refuses goes back to its path; when something took the path meanwhile, it is renamed to
  `<name>.pivot-kept-<hex>` beside it and a warning names it, never left under the backup name a
  later restore removes. Every other replacement (a stage's write, a restore in a run, a
  symlink-mode commit, presentation) is checked when it starts replacing that output, so a file
  that lands in the output after that check is removed with it. `track` and `import` refuse to
  create a sidecar inside an output, so only a file copied or made by hand can appear there.
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
