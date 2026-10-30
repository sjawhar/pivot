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

Commands that read or write workspace files may save their hashes best-effort. A hash-cache
failure does not fail the command. Read-only commands do not write locks, run history, or outputs.
`--no-commit` writes no lock or `run:` history entry. It records the uncommitted run in
`nocommit:` for a later `pivot commit`. Every command that writes the workspace saves the hash
of each file it reads, commits or restores, so `pivot status --explain` and `pivot verify` find a
matching entry and read no file. A new command that reads or writes workspace files must record
them the same way:

- `hash_file(..., pending=...)` and `hash_directory(..., pending=...)` append `(path, stat,
  hash)` for each file they read. The caller passes these entries to `state.save_file_hashes` or
  `state.FileHashBatch`, which writes once per state directory.
- Every cache save goes through `state.save_best_effort` (`state.save_file_hashes`) or
  `StateDB.save_many_best_effort` (save-on-read in `hash_file`/`hash_directory`). They wait at
  most `_WRITE_BACK_TIMEOUT` (2 s) for the write lock, log a failed write at debug level, and keep
  the command's result. A fingerprint-cache flush that is skipped keeps its entries pending for
  the next flush; a skipped hash write costs one later read.
- `state.open_readonly(..., create=False)` looks hashes up without creating a database. A
  database this user cannot open safely (`PermissionError`, `lmdb.ReadonlyError`) yields no
  cache first. Another `OSError` or `lmdb.Error` yields no cache only when
  `os.statvfs(state.lmdb)` reports a read-only filesystem; otherwise it raises. `pivot checkout`
  catches a failure to open the cache and restores without it; status and verify raise.
- A restore records each file with the stat of what it staged before the rename puts it in place
  (`cache.restore_from_cache(..., recorded=...)`, `cache.atomic_write_file`'s return value). A
  write after the rename changes the stat. The entry then does not vouch for the new bytes.
- Workers open StateDB read-only. They return `DeferredWrites.file_hash_entries`
  (`types.FileHashEntry`, from `cache.deferred_file_hash_entries`) for the coordinator to write
  after the stage. For a normal run these are not a cache save: they share one transaction with
  the stage's generations, which is the run's own commit. Under `--no-commit` only the
  `nocommit:` record is authoritative and the coordinator saves the hashes best-effort. The
  coordinator's skip check writes what it reads and restores, best-effort.
- `pivot checkout`, `pivot pull`, `pivot get`, `pivot import`, `pivot update`, `pivot track`,
  repro's `--checkout-missing` path, `pivot status`, its explanations, `pivot diff` and
  `pivot plots diff` record what they write or read. Stage outputs use the StateDB in the output
  stage's state directory. Other paths use the configured state directory.
- Every `hash:` write also writes the `inode:` index after the hash entries. An identity lookup
  is allowlisted by mount type (`storage/mounts.py`; tmpfs only with `inode64`) and confirms the
  recorded path's device, inode and change time, so a hard-linked replica reuses its original
  hash without weakening a path lookup.

## Directory Input Hashes and `.pivotignore`

- Only `WorkspaceStore.hash_artifact` passes `ignored` to `cache.hash_directory`, and only for a `pipeline.input` directory: the entries the project's `.pivotignore` names (`ignore.IgnoreFilter.member_filter`, matched by their path inside the project) are left out of that dependency hash. Outputs, tracked data, `save_to_cache`, restores and checkout comparisons hash every file; a file left out there is lost on restore.
- A dependency on a stage's directory output is the producer's whole output hash: `explain.producer_output_hashes` reuses producer lock hashes and `sync.get_target_hashes` fetches dependency blobs from the producer's lock and listing.
- A directory input's committed listing (`listing.input_listing_path`, `.pivot/stages/=inputs/<path>.listing`) holds the members its dependency hash counts, so it leaves out what `.pivotignore` names: the lock's record is the hash of those bytes, and `listing.parse` refuses any other listing for it. A tracked directory's `.pvt.listing` is the whole set, because a restore needs every file.
- No restore reads an `=inputs` listing: it is read only by `cli/verify.py` (the cache objects a plain directory input needs), the engine's git-ignored warning and `status.py`'s orphaned-listing report. A restore takes its members from an output's listing beside the producer's lock, from a `.pvt.listing`, or from the cache object at the recorded hash (`storage/restore.py`, `WorkspaceStore.restore_from_cache`, `show/selection.py`). Keep it that way: an `=inputs` listing names only the files the dependency hash counts.
- The listing object of a filtered input hash is written only to the cache of the machine that ran the stage (`StageLock.write`); only an untargeted `pivot push`, which sends the whole cache, sends it. So for a directory input held by tracked data, `verify --allow-missing` requires what restores that data (`verify._tracked_input_objects`: the `.pvt` listing object and files), as pull fetches it, never that object.
- An input hash taken from a record (a tracked directory's `.pvt`, in explain's `--allow-missing` fallback) goes through `WorkspaceStore.recorded_input_hash`, which applies `cache.filter_listing` to the whole listing: the directory the filtered walk gives the recorded tree. When that listing is missing or corrupt, the `.pvt`'s whole hash stands (at worst a false "stale"); the lock's own hash never does, since it would compare with itself.
- `pull`/`fetch <stage>` takes a tracked input's blobs from the `.pvt` files at, under or above the input's path (`sync._tracked_blobs_for_input`), not from the lock's dependency hash, so every member of the tracked data is fetched.
- `ignore.py` reads and matches `.pivotignore` as git does a root `.gitignore`: a port of git 2.52's `add_patterns_from_buffer`, `match_basename`, `match_pathname` and `dowild` (bytes; the literal prefix, of which wildmatch still sees the last byte, git commit 1940a02dc1; `WM_ABORT_*`; git's parent rule). Keep it a port: a regular expression backtracks where `dowild` aborts. `tests/storage/test_input_ignore_matches_git.py` checks the filtered walk, `filter_listing` and `check_ignore` against `git ls-files` on generated trees and patterns (fixed seed; `PIVOT_IGNORE_GIT_CASES` for more); with a git older than 2.52 it leaves out the patterns whose first wildcard is a `**` right after a byte that is not `/`, where older git differs.
- The `hash:` cache holds one content hash per file, so filtered and whole walks share it; no directory hash is cached by path.

## StateDB Key Prefixes

StateDB uses key prefixes for namespacing. Current prefixes in `pivot/storage/state.py`:

| Prefix | Purpose |
|--------|---------|
| `hash:` | File content hashes (keyed by the absolute path as given, `os.fsencode(os.path.abspath(path))`; the value's `(mtime_ns, size, dev, inode)` comes from `stat()`, which follows symlinks. No ctime: a new hard link changes it, so every hardlink checkout or hard-linked replica would invalidate its entries) |
| `inode:` | `(st_dev, st_ino)` → a `hash:` key hint; served only after the hash stat and recorded-path device, inode and change-time checks |
| `gen:` | Output generation counters (keyed by artifact identity, `stage` or `stage:key`) |
| `dep:` | Stage dependency generations (`stage:dep_identity` → generation; one file of a directory output records its directory's identity) |
| `runcache:` | Run cache entries (`stage:input_hash` → output hashes) |
| `sm:` | Fingerprint manifest cache (stage manifests + source maps) |
| `run:` | Run history entries |
| `remote:` | Remote index entries |
| `remote_url:` | Remote URL tracking for change detection |
| `fp:` | AST fingerprint/hash cache entries |
| `nocommit:` | What a stage's outputs hold that no lock records yet (`stage` → a JSON object, `types.NoCommitRecord`): a `--no-commit` run's code manifest, marks included, output hashes and, when it read any, the objects it read by hash (`reads`), for `pivot commit`, and the `*.pvt` directory members Pivot wrote (`pvt_members`, path → content hash, each added before the file is written); a written lock removes the whole record. Each writer replaces only its own keys and keeps every other key, in one write transaction; a run's record replaces all three of its keys, so an earlier run's `reads` never stays beside a later run's outputs. The reader validates each of the four keys on its own, dropping only a malformed one, and returns every other key as stored, so a record holding only another feature's key reads as present; `pivot commit` uses the run's manifest and reads only when the run's manifest and outputs are both present |

**When adding new state types**, define a new prefix constant and document it here.

## The `.pvt` Walk Memo

`track.discover_pvt_files` must find every `.pvt` that `os.walk` over the project finds. Never prune it by directory name or by stage output: `track` and `import` accept paths under `.git`, `.pivot` and other pipelines' outputs. The memo makes an unchanged directory cost one `lstat` instead of a listing:

- A listing is reused only when the directory's stamp (`st_dev`, `st_ino`, `st_mtime_ns`, `st_ctime_ns`) matches, its mtime and ctime are at least 2 s before the walk start (a later change in the same timestamp tick would leave the stamp unchanged), its mtime and ctime are not both whole seconds (on a one-second-timestamp filesystem a clock stepped back into a recorded second would keep the stamp; such filesystems are always listed), both fit the stamp's signed 64-bit nanoseconds (1677 to 2262; any other time is listed every walk), and `storage.mounts` says its device keeps live identities unique. A device mountinfo does not list (a btrfs subvolume reports its own) counts as btrfs only when the deepest mount holding the directory is btrfs; tmpfs is allowed only with `inode64`.
- It holds names only. Every `.pvt` found is read on every walk, since rewriting a file does not change its directory.
- Only the repro/run pre-flight (`executor/core.py` `verify_tracked_files`) writes it, and only when the walk's listings differ from it. It writes through the descriptor (`os.write`) under an `flock` (when another pre-flight holds it, this one skips its write) and empties the file first, so a walk reading it meanwhile, or after a pre-flight killed mid-write, sees a prefix of the new memo. A prefix never parses (it ends inside a record, or holds fewer listings than the root record counts), so that walk lists every directory. A write that fails partway (a full disk) empties the file again, so no cut memo keeps the space. Every other caller only reads it, and reading never creates anything. A memo that cannot be read means every directory is listed.
- It is not in the state database, which other accounts can read. It is a file of the running account's own, `state.lmdb/pvt-memo-<uid>` in the state directory (the state directory's `.gitignore` keeps `state.lmdb/` out of git), created with mode 0600. The pre-flight creates it only in a `state.lmdb` directory that already exists and is on an allowlisted filesystem, and rewrites it in place, so that directory's stamp does not change at each write. A walk opens it without following a symlink (`O_NOFOLLOW`, and `O_NONBLOCK` so a FIFO there cannot block it) and uses it, for reading or writing, only when an `fstat` of the open file shows a regular file, on a filesystem of the walk's allowlist (`storage.mounts` `device_allowed`, with mountinfo read once per walk; there owner and mode are the file's own and the kernel enforces them, while bindfs, CIFS and NFS can report the mount's), owned by the running account, with no group or other permission bits (a POSIX ACL's mask shows in the group bits) and one link. Otherwise the walk lists every directory, writes nothing, and logs the reason at debug level. A file of another account, of another kind or on another filesystem is never written, replaced or removed. When the account's own memo fails only the mode or link check (its owner loosened it, `chmod -R go+rX .pivot` say), the read-only walk only refuses it, and the pre-flight empties it, unlinks it and creates a new 0600 file, so its names leave that path, every descriptor opened on it and any other link to it. So no other account reads the memo's names, whatever the modes of the project's directories and wherever the state directory or `state.lmdb` points, except while the owner leaves the file loose, until the next pre-flight replaces it. The cost is one `open` and one `fstat` per walk (one more when a loosened memo is replaced). Each account keeps its own memo in a shared state directory.
- A listing in which an entry's type could not be read (the `lstat` a filesystem without `d_type` needs failed) is never recorded, so a walk that can read that type never reuses it.

The walk never skips a `.pvt` file. Its `skip` argument leaves a `.pvt` out of the result only after the file is read. Every command that discovers the project's `.pvt` files passes `member_skip.own_member_skip`, which leaves out a stage's own member of its output (the `ownership.stage_members` rule) and keeps every other `.pvt` there, which a restore of that output then refuses. The command's own stages decide their outputs from the registry. Any other `.pvt` is decided from recorded state alone, never by importing another `pipeline.py`: it is a member when a directory output another stage's lock records at a path holds it, and that lock's committed listing (an older pivot's lock: its inline listing) or the stage's `nocommit:` record lists it with its current content. `lock.recorded_directories` reads those locks, in both directories of the 0.5.2 layout, at the first `.pvt` outside the command's outputs. It says nothing and never fails: it lists the stages directories as `Path.is_file` would (an entry it cannot `stat`, such as a looping symlink, is no lock), chooses each stage's current lock with `StageLock`'s chooser without its notices (`_choose_stage_lock`), decodes it with `parse_committed` (no format warning, nothing written to the cache), reads the listing beside it with `listing.OnStale.QUIET` (a stale one is passed over without the warning, and a cache object found only in the legacy state directory draws no notice), and takes a lock it cannot read or decode (a YAML scalar the safe loader cannot build included) as recording nothing, so its member is read as a tracked file. Within one state-directory record it decodes a lock file again only when the file changed (`state_dir.recorded_locks`), so a `--watch` run sees each lock written since. When every pipeline is loaded (`--all`, `track`) only the locks of stages no pipeline declares are read. Whether a stage run wrote a file is execution state, not a stage definition; no output placement is re-derived. A pipeline that fails to load, and a stage that no longer exists, are decided the same way: their members are skipped, not restored. A sidecar `track` wrote at such a member's path could carry the bytes the lock records and then be skipped, so `track` refuses a path whose sidecar the lock of a stage no pipeline declares lists as a member (`member_skip.orphaned_member`, spelling the sidecar as the walk reaches it: its directory where it really lies, the sidecar itself unresolved, since a member may be a symlink to its cache object), naming the lock (or both) to remove before `pivot track --force`. A `nocommit:` record that cannot be read is taken as absent (`ownership.read_stage_claims`), which can only refuse. `remote/sync.py` applies the same rule to the one `.pvt` it reads by path (`_get_file_hash_from_pvt`), so no target or dependency resolves to what a member records.

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
- A stage write with a `state_dir` (the worker passes the project's to
  `WorkspaceStore.prepare_output`) and a symlink-mode commit hold the same lock while they
  replace an old output (a directory renamed aside, checked and removed; a file checked in place
  and unlinked), so no restore's cleanup removes a backup while they check it.

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
  temporary entry. Where `flock()` is refused (Lustre without `flock`: ENOSYS; NFS when its lock
  manager does not answer: ENOLCK) staging still goes on without the shared lock, and this
  process neither sweeps nor repairs a tree. It never trusts a tree merely because its XXH64
  name exists. Only where every host sharing the cache is refused does nothing sweep: an NFSv3
  client whose own `rpc.statd` is down gets ENOLCK while other clients' locks work, and their
  sweeps can remove its live temporary entries.
- A lock file in the cache is created writable by whoever can write its directory, since NFS
  takes an exclusive lock only on a file opened for writing; one this user cannot write is
  opened read-only. Lock files are never deleted.
- A whole-directory SYMLINK tree lives at `<cache root>/trees/XX/YYYY...` (`cache.tree_path`),
  named by the directory hash. Its `<tree>.content-checked` marker holds the canonical listing
  bytes after Pivot verifies the tree. A tree is used without a check only when its non-empty
  marker equals the supplied listing's bytes. A different non-empty marker means another
  listing was verified for this tree, so the tree is neither used nor changed. An unmarked tree,
  or one with an empty, `.layout-checked` (pivot 0.3.4) or `.verified` (pivot 0.3.5–0.5.11)
  marker, is checked under the tree's exclusive `<tree>.lock`. A restore or
  `missing_cache_objects` passes `repair=False`. It can mark a tree that already matches, but it
  never changes that tree. Only the commit path passes `repair=True`, because its listing came
  from actual output bytes. It can repair from individual objects, hashes the tree again, and
  marks it only when the result equals its listing bytes. `_publish_tree` never marks. A tree
  that cannot be checked or repaired is not used. A restore goes file by file. A commit fails.
  The marker is staged under the prefix directory's shared lock (`_staging_in`). A user who
  cannot write the prefix directory cannot mark a tree, so each of their symlink restores and
  commits checks it again.
- The object at a directory's hash under `files/` is its listing (`listing.store_object`),
  read-only like every object. Every listing read follows one rule (`listing._resolve`): a
  committed listing (workspace file, or a file at a git revision) that hashes to the recorded
  directory hash is taken only when, if the cache also holds the listing, its bytes equal that
  canonical cache copy. Bytes that differ despite the same XXH64 are refused as a collision; a
  listing that has the wrong hash or cannot be read falls back to the cache object with one
  warning, and is refused only when that object is absent too, or at once with
  `OnStale.REFUSE` (`pivot verify`). A reader of a stage the command does not run passes
  `OnStale.QUIET`, which falls back without a word.

## Lock and Listing Files

- A lock records a directory as `{listing, size, nfiles}` on disk (`StorageLockData` holds the
  listing's hash under `hash`, as for a file; `lock._from_document`/`_to_document` convert). The
  key is not `hash` because every released pivot reads an entry with a `hash` and no inline
  `manifest` as one file. A lock that records a directory keeps every entry under `records` and
  leaves `deps` and `outs` empty (`lock._shaped`): released pivots read that as a lock that
  records nothing, rerun an ordinary stage, and stop at an incremental one instead of restarting
  it from empty. Run-cache keys hold `run_history._DIRECTORY_RECORD` for the same reason; an
  older pivot's entry is looked up under `worker.older_pivot_input_hash`.
- A stage with an incremental output commits a restart proof beside its lock
  (`lock.restart_proof_path`, a `.incremental` file this pivot owns). No released pivot writes
  one, so this file — or, for a directory output, this pivot's committed output listing — beside
  a lock an older pivot wrote is proof this pivot wrote the lock first and the older pivot
  replaced it, which may have restarted the output from empty. Such a stage is refused
  (`lock.restart_proofs`): `status`/`verify` through `explain.get_stage_explanation`, `repro`
  in `engine` before it schedules anything.
- `StageLock.write` writes each directory output's listing (`listing.output_listing_path`),
  each directory input's (`listing.input_listing_path`) and, for an incremental stage, the
  restart proof before the lock, leaving a file that already holds the bytes untouched. It is
  the one writer of lock, listing and restart-proof files (Task 2.2 of the BET-36 plan moves
  that call to the coordinator).
- A lock whose `schema_version` is below `lock.SCHEMA_VERSION` (an older pivot's, listing a
  directory's files inline) is read as it is, with one warning per process: `StageLock.read`
  converts each inline listing to its record and stores the listing in the cache it is given,
  where every listing read finds it. Pass the cache to every `read`. A newer pivot's lock reads
  as missing.
- `pivot init` writes `*.listing -merge` into `.gitattributes`, so a merge conflicts on a
  listing instead of producing bytes that hash to neither side.
- A lock lists the objects its run read by hash (`pivot.read_by_hash`, `pivot/reads.py`) under a
  top-level `reads` key: content hashes, sorted, each once, written only when the run read one,
  so a stage that reads nothing by hash writes the lock it wrote before. Released pivots allow an
  unknown top-level key and read such a lock as any other. The run-cache entry the run files
  (`RunCacheEntry.reads`) and its `nocommit:` record carry the same list, so a run-cache restore
  and `pivot commit` write it too. The input that names a hash reruns the stage; `reads` is what
  that run read. Every skip decision passes `reads.lock_unavailable` to `skip.check_stage`
  (status, `explain`, the engine's pre-check and the worker's tier 2; the worker's tier 1
  declines on it, and a run-cache entry whose objects are gone is not used), so a lock that
  records an object neither in the cache nor known to be on the remote is stale everywhere at
  once. The check is local, as for outputs: an object the cache lacks counts as on the remote
  only when the remote index (`remote:`) records it for the remote's URL; it never contacts the
  remote, so plain `status` and `repro` never wait on the network. `pivot push`, `pull` and
  `fetch` fill that index, and so does a completed run whose read fetched an object: the worker
  returns what its reads fetched in `DeferredWrites.remote_hashes` (`reads.Recording.fetched`),
  under `--no-commit` too, and `apply_deferred_writes` adds it in the run's transaction, storing
  the fetch's URL when none is stored for the remote and adding nothing when another one is. A
  failed run applies no deferred writes, so its fetches stay unrecorded.
  `reads.unavailable` returns the reason and whether a usable remote may hold the objects
  (`types.UnavailableReads.pullable`); when those objects alone change the stage, `check_stage`
  sets `recovery` to `Recovery.PULL_READS`. Explain keeps it only when pull also restores every
  output the stage lacks, and status only when every stale upstream stage has a recovery too;
  `pivot status` suggests `pivot pull` from that field, never from the reason's text. A trust-me
  `pivot commit <stage>` keeps the previous lock's `reads` only when that lock records the same
  code, params, dependency hashes and outputs.

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
- A forced restore checks each target three times: with every target before any is changed, under
  its restore lock before its replacement is built, and at its swap over the entry it removes,
  renamed aside where a directory is involved. A run-cache, generation or incremental output
  restore and repro's restore of a missing tracked file make the same swap check; an incremental
  output's restore removes nothing before it. A stage write clears its old output the same way:
  `WorkspaceStore.prepare_output` calls `cache.remove_output_checked`, which renames an old
  directory at the output's path aside, beside it (to `cache._locked_backup_path` under the
  path's restore lock, else to a fresh `cache._backup_path`), checks it there
  (`ownership.swap_refusal`), and removes it only when the check finds no user file; when the
  check refuses, the directory goes back to its path (`cache._put_back_checked`) and the refusal
  names the file. When removing the backup fails, what is left of it goes back and the removal's
  own error propagates. An old file or symlink is checked in place and unlinked, as a restore
  checks a file it replaces: renaming it aside would close only the instant between the check and
  the unlink, and on overlayfs renaming a file on the lower layer copies all of it up into the
  writable layer first. The unlink fails, removing nothing, on a directory that took the path
  after the check. A symlink-mode commit replaces a directory output with its link to the cache
  tree the same way (`WorkspaceStore.commit` passes the check to `cache.store_hashed_output`). A
  stage writer then calls `WorkspaceStore.refuse_output_replacement_at_write` after
  `prepare_output` and before its loader writes, so a user file that lands in the new, empty
  output is refused with the same target and sidecar message rather than leaking a loader's
  "Directory not empty" error. Once a swap or a stage write renames a directory aside, no writer
  that names its files by path can add one to it, so a file copied into a target while the
  command runs refuses that target. What the checks can still miss: a writer holding an open
  handle on a directory inside the target (its working directory, or a directory descriptor),
  which can add a file to the renamed entry after the walk; a sidecar created beside the target,
  or beside a directory above it, in the instant between the swap's check and its rename; a file
  moved over a file output in the instant between the last check and its unlink (a stage write)
  or the rename that replaces it (a restore); and a file moved over a file output while the
  stage's own writer writes it (after `refuse_output_replacement_at_write`, before the loader's
  or `worker._save_recorded`'s rename over the path), which the stage's output replaces. For an
  output Pivot caches, the commit's check then refuses, naming the sidecar; a metric output (whose
  commit runs no check) and a `--no-commit` run (which commits nothing) finish with exit 0, and
  the next `pivot status` or `pivot repro` warns, naming it. A directory output loses in that
  window only a user file that has the name of a file the stage writes; the writer adds its other
  files beside the user's, and the commit refuses.
- Where the filesystem cannot rename a directory aside (`cache._rename_aside`: EXDEV from
  overlayfs with `redirect_dir` off, the default on many kernels, for a directory on a lower
  layer; ENOTSUP or EOPNOTSUPP), a swap, a stage write and a symlink-mode commit check it in place
  and remove it there, as before the rename aside existed: on such a filesystem a file copied into
  it after that check is removed with it. A stage write or symlink-mode commit whose rename is
  refused for want of write permission on the parent retries once the parent is made writable
  (`cache._remove_checked`); a restore's swap (`cache._swap_in`) does not. Any other rename error
  (EBUSY for an output that is a mount point) fails before anything is removed.
- A directory restore killed between renaming the old entry aside and checking it leaves a backup.
  The next directory restore of that path holds the same restore lock and checks that backup
  before removing it. If the check finds a user file, Pivot keeps the backup and warns, naming the
  file; it restores the new output without deleting the backup. The cleanup catches only
  `OutputOwnershipError` (`cache._remove_backup_left`), so every swap check raises the refusal
  itself, `pivot checkout`'s and `pivot pull`'s included (`checkout._refuse_swap`; the command
  reports a refusal from the swap itself as a failure). Every backup is named with its path's
  restore token. The unlocked sweep a directory restore runs in its parent
  (`cache._cleanup_stale_restore_temps`) removes only staged replacements, since a backup keeps
  the old entry's mtime and may be under another restore's check. A file restore never runs that
  cleanup (`cache._restore_file_from_cache`), so when one is killed after renaming a directory at
  its path aside, that directory stays beside the path as `.pivot_backup_<token>.<hex>`, with
  everything in it: no file restore removes it (only a later directory restore at that path
  considers it), and it is the user's to delete. Every directory a forced or run restore meets
  at its path is renamed aside and checked, including one made after a file target's check. An
  entry the check refuses goes back to its path; when something took the path meanwhile, it is
  renamed to `<name>.pivot-kept-<hex>` beside it and a warning names it, never left under the
  backup name a later restore considers. A stage write killed between renaming its old directory
  output aside and removing it leaves the backup at `.pivot_backup_<token>.old`
  (`cache._locked_backup_path`); only a stage write or a symlink-mode commit of a directory makes
  one, since a file output is never renamed aside. The next stage write of that path, under the
  same restore lock, checks that backup first and removes it, or keeps it and warns, naming the
  file, as a directory restore does with every backup; a stage write that keeps it renames its own
  old output to a fresh `cache._backup_path`, which only a directory restore considers. A
  directory restore of the path considers it like any other backup. Commits other than a
  symlink-mode commit of a directory output, and presentation, still check only when they start
  replacing an output: a commit in hardlink mode (or symlink mode, for a file output) replaces
  each file the stage wrote with its link to the cache after that one check. `track` and `import`
  refuse to create a sidecar inside an output, so only a file copied or made by hand can appear
  there.
- `track` and `import` load every pipeline (`discovery.declared_outputs`, as `--all` discovers
  them) and refuse a new path overlapping any declared output, naming the stage. A `pipeline.py`
  that fails to load fails them; one that loads but defines no Pivot pipeline declares nothing. A
  `pipeline.py` in a directory discovery excludes (`venv/`, `node_modules/`, …) is not counted;
  the store's check still protects its outputs. Re-tracking (`track --force`) or re-importing
  (`import --force`) a path already saved there warns and updates it in place. `status` and the
  engine warn, once per run, about a sidecar already inside an output they load.

## Path Storage

Generation, dependency and run-cache keys hold artifact identities and project-relative paths, so they mean the same thing on every machine. File-hash keys and the identity index hold absolute paths on purpose: an entry describes one file on this machine, checked by its `stat`.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.
