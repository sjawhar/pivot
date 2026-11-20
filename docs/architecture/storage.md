# Storage

`pivot.storage` keeps Pivot's state: StateDB, the per-stage lock and listing files, the
content-addressed cache, restores, the output-ownership checks and `pivot gc`. This page maps each
rule that spans modules to the code that keeps it. The short list of rules a change must keep is
`packages/pivot/src/pivot/storage/AGENTS.md`. What users see is in
[Caching & Skip Detection](../concepts/caching.md), [Outputs](../concepts/outputs.md) and
[Dependencies](../concepts/dependencies.md); a rule inside one function is in its docstring.

## StateDB key prefixes

StateDB (`storage/state.py`) is one LMDB environment per state directory,
`<state dir>/state.lmdb`, with a 10 GiB virtual map (`_MAP_SIZE`). Keys and values are bytes.
Each kind of entry has a prefix constant at the top of `state.py`:

| Prefix | Key → value |
|--------|-------------|
| `hash:` | the absolute path as given (`os.fsencode(os.path.abspath(path))`) → `(mtime_ns, size, dev, inode)` from a `stat()` that follows symlinks, and the content hash |
| `inode:` | `(st_dev, st_ino)` → the `hash:` key of one path with that identity; served only after the stat and the recorded path's device, inode and change time match |
| `gen:` | artifact identity (`stage` or `stage:key`) → output generation counter |
| `dep:` | `stage:dep_identity` → the dependency's generation when the stage ran; a file of a directory output records its directory's identity |
| `runcache:` | `stage:input_hash` → the run's output hashes (`run_history.RunCacheEntry`) |
| `run:` | run id → run history manifest |
| `sm:` | fingerprint manifest cache (stage manifests and source maps); best-effort, never needed for correctness |
| `fp:` | AST fingerprint and hash cache |
| `remote:` | `remote_name:hash` → present when the remote is known to hold the object |
| `remote_url:` | remote name → the URL its `remote:` entries were recorded for |
| `nocommit:` | stage → `types.NoCommitRecord`: what the stage's outputs hold that no lock records yet |

A file-hash key costs no system call to build; the stat in its value decides whether the cached
hash still applies, so a link re-pointed at another file no longer matches. Generation keys use
the artifact identity, so they stay the same wherever an output's path points.

A `nocommit:` record holds a `--no-commit` run's code manifest, output hashes and `reads`, for
`pivot commit`, and the `*.pvt` members Pivot wrote (`pvt_members`). Each writer replaces only its
own keys, in one write transaction; a run replaces all three of its keys, so an earlier run's
`reads` never stays beside a later run's outputs. The reader validates each key on its own and
drops only a malformed one. Only a written lock removes the record
(`DeferredWrites.clear_no_commit_record`).

A write waits for `StateDB._write_transaction` up to the StateDB's `write_timeout` (30 s by
default); a best-effort hash-cache save waits `_WRITE_BACK_TIMEOUT` (2 s). The worker/coordinator
split and the generation counters are in [Execution Model](execution.md#statedb-architecture).

## File-hash write-back

Every command that hashes, commits or restores a workspace file saves its hash, so the next
`pivot status --explain` or `pivot verify` reads no file (the commands are listed in
[File Change Detection](../concepts/caching.md#file-change-detection)).

- A caller collects entries with `pending=` (`cache.hash_file`, `cache.hash_directory`) or
  `recorded=` (`cache.restore_from_cache`) and saves them with `state.save_file_hashes` or
  `state.FileHashBatch`, which writes once per state directory. A stage output's entry goes to
  the StateDB of the output stage's state directory; any other path's to the configured one.
- A worker returns its entries in `DeferredWrites.file_hash_entries`
  (`cache.deferred_file_hash_entries`). In a normal run they share the run's own transaction with
  the stage's generations; under `--no-commit` the coordinator saves them best-effort.
- Every `hash:` write also writes the `inode:` entries, after the hash entries. Which filesystems
  the identity index trusts is decided in `storage/mounts.py`.
- The `hash:` cache holds one content hash per file, shared by filtered and whole directory
  walks; no directory hash is cached by path.

## Directory inputs and `.pivotignore`

- `WorkspaceStore.hash_artifact` passes `ignored` to `cache.hash_directory` for a
  `pipeline.input` directory, and nothing else does.
- A directory input's committed listing (`listing.input_listing_path`) holds only the members its
  hash counts; `listing.parse` refuses any other listing for that hash. Only `cli/verify.py`, the
  engine's git-ignored warning and `status.py`'s orphaned-listing report read it. A restore takes
  members from an output's listing, a `.pvt.listing` or the cache object at the recorded hash.
- The listing object of a filtered hash is written only to the cache of the machine that ran the
  stage (`StageLock.write`), and only an untargeted `pivot push` sends it. So for a directory
  input held by tracked data, `verify --allow-missing` asks for what restores that data
  (`verify._tracked_input_objects`), never that object.
- An input hash taken from a record (a tracked directory's `.pvt`, in explain's `--allow-missing`
  fallback) goes through `WorkspaceStore.recorded_input_hash`, which filters the whole listing
  with `cache.filter_listing`. When that listing is missing, the `.pvt`'s whole hash stands.
- A dependency on a stage's directory output is the producer's whole output hash
  (`explain.producer_output_hashes`).
- `push`, `fetch` and `pull` of a target transfer one set, `sync.get_target_hashes`: what
  `pivot checkout` of the target restores, the objects its lock lists under `reads`, and a mapped
  stage instance's producer listing.
- `ignore.py` ports git 2.52's `add_patterns_from_buffer`, `match_basename`, `match_pathname` and
  `dowild`. `packages/pivot/tests/storage/test_input_ignore_matches_git.py` compares the filtered
  walk, `filter_listing` and `check_ignore` with `git ls-files` on generated trees (set
  `PIVOT_IGNORE_GIT_CASES` for more cases).

## `.pvt` discovery

`track.discover_pvt_files` walks the whole project as `os.walk` does and reads every `.pvt` it
finds. The repro/run pre-flight (`executor/core.py` `verify_tracked_files`) records a memo: for
each directory that changed at least 2 s before the walk, its subdirectory and `.pvt` names under
its stamp (`st_dev`, `st_ino`, `st_mtime_ns`, `st_ctime_ns`). A later walk still stats each
directory and lists only those whose stamp changed, or changed in the last 2 s. A stamp whose
mtime and ctime are both whole seconds, or whose time does not fit signed 64-bit nanoseconds, is
never trusted. The memo is used only where `storage/mounts.py` allows it: ext2/3/4, XFS, Btrfs,
F2FS, ZFS, and tmpfs mounted with `inode64`.

- Only the pre-flight writes the memo, through its descriptor under an `flock`, emptying it
  first: a walk that reads it meanwhile sees a prefix, which never parses. Every other caller only
  reads it; a memo that cannot be read means every directory is listed. It holds no `.pvt`
  content, and one recorded under another project root is not used.
- The memo is a file of the running account's own, `state.lmdb/pvt-memo-<uid>` in the state
  directory, mode 0600, because other accounts can read StateDB. A walk opens it without following
  a symlink and uses it only when an `fstat` shows a regular file on an allowed filesystem, owned
  by the running account, with no group or other bits and one link; otherwise it lists every
  directory and writes nothing. The next pre-flight replaces a memo its owner loosened; another
  account's file is never touched.
- A listing in which an entry's type could not be read is never recorded.

A `.pvt` inside a stage's directory output can be the stage's own data rather than a tracked
file. Every command that discovers `.pvt` files passes `member_skip.own_member_skip`. The
command's own stages decide from the registry; any other `.pvt` is decided from the locks other
stages left (`lock.recorded_directories`, cached per lock file by `state_dir.recorded_locks`) and
their `nocommit:` records (`ownership.read_stage_claims`). `pivot track` refuses a path whose
sidecar a stage no pipeline declares records as its member (`member_skip.orphaned_member`).
`remote/sync.py` applies the same rule to the one `.pvt` it reads by path
(`_get_file_hash_from_pvt`).

## Lock and listing files

- A lock records a directory as `{listing, size, nfiles}` on disk, not under `hash`, which a
  released pivot reads as one file; `types.StorageLockData` holds the listing's hash under `hash`
  (`lock._from_document`, `lock._to_document`). A lock that records a directory keeps every entry
  under `records` and leaves `deps` and `outs` empty (`lock._shaped`), so a released pivot reads
  it as recording nothing. Run-cache keys hold `run_history._DIRECTORY_RECORD` for the same
  reason; an older entry is found under `worker.older_pivot_input_hash`.
- A lock whose `schema_version` is below `lock.SCHEMA_VERSION` is read as it is, with one warning
  per process; a newer pivot's lock reads as missing.
- A restart proof (`lock.restart_proof_path`) beside a lock an older pivot wrote means that pivot
  may have restarted an incremental output; `lock.restart_proofs` refuses the stage in
  `explain.get_stage_explanation` (status, verify) and in the engine before it schedules.
- Every listing read follows `listing._resolve`: a committed listing is taken only when it hashes
  to the recorded hash and equals the cache's copy, if there is one. `listing.OnStale` chooses
  what a stale one does: `REFUSE` for `pivot verify`, `QUIET` for a stage the command does not run.
  Where a cache was read and could not give the listing, the refusal is
  `ListingUnavailableError` (its docstring says where), which a restore under
  `repro --all --keep-going` tolerates (`executor/tracked_restore.py`); every other refusal is a
  plain `CorruptListingError`.

### Objects read by hash

A lock lists the objects its run read with `pivot.read_by_hash` (`pivot/reads.py`) under a
top-level `reads` key, as do the run-cache entry (`RunCacheEntry.reads`) and the `nocommit:`
record.

- Every skip decision passes `reads.lock_unavailable` to `skip.check_stage` (status, explain, the
  engine's pre-check, the worker's tier 2; tier 1 declines on it). The check is local: an object
  the cache lacks counts as on the remote only when `remote:` records it for the remote's URL.
- A run that fetched an object returns it in `DeferredWrites.remote_hashes`
  (`reads.Recording.fetched`), applied in the run's transaction.
- When only unavailable reads make a stage stale and a remote may hold them
  (`types.UnavailableReads.pullable`), `check_stage` sets `Recovery.PULL_READS`; `pivot status`
  suggests `pivot pull` from that field.
- A read names nothing until a lock does, so each commit fetches again any read object the cache
  lost (`reads.fetch_missing`) and, inside its naming unit, refuses to write a lock naming one that
  is still missing (`reads.require_cached`).

## Output ownership

Pivot never removes a tracked file, an import or a `.pvt` sidecar to replace a stage output. The
user-facing rules and their limits are under [`pivot track`](../cli/index.md#pivot-track).

- Checked before anything is removed: a stage write (a stage with several outputs checks them all
  first, `worker._write_outputs_with_store` → `WorkspaceStore.refuse_output_replacements`; then
  `WorkspaceStore.prepare_output` checks each one just before clearing it), generation, lock and
  run-cache restores (`WorkspaceStore.restore_from_cache`), commits (`WorkspaceStore.commit`), an
  incremental output's restore, `checkout`, `pull`, `get --force` and the presentation layer.
- A stage write clears its old output with `cache.remove_output_checked`, which renames a
  directory aside, checks it there (`ownership.swap_refusal`) and puts it back when the check
  refuses. Then `WorkspaceStore.refuse_output_replacement_at_write` runs before the loader writes.
  A symlink-mode commit does the same through `cache.store_hashed_output`.
- A forced restore makes one decision, `ownership.forced_restore_refusal`, over targets built once
  (`checkout._restore_targets`, `restore._restore_targets`), and checks each target three times:
  with every target before any changes, under its restore lock before its replacement is built
  (`checkout._restore_path_locked`, `restore._replace_checked`), and at its swap
  (`ownership.swap_refusal`, passed to `cache.restore_from_cache` as `before_swap`).
- A swap's backup cleanup (`cache._remove_backup_left`) catches only `OutputOwnershipError`, so
  every swap check raises that error itself (`checkout._refuse_swap`).
- A stage's own `*.pvt` members: `ownership.stage_members` decides from the lock, the hash a
  restore writes, what this run writes and the `nocommit:` record, read once per command
  (`ownership.StageClaims`). A run stages each entry that brings one beside the output
  (`worker._staging_path`), records it (`ownership.record_members`, which writes
  `StateDB.add_no_commit_pvt_members` from the worker and raises `MemberRecordError` when it
  cannot), then renames it into place (`worker._save_recorded`). A restore records the members of
  a hash the lock does not record (`ownership.record_restored`).

## Cache objects

- Every file object is published by `cache.publish_cache_object` and every temporary entry made
  by `cache.staged_cache_file` or `cache._staging_in`; the sweep of `.pivot_tmp_*` entries runs at
  most once per `_CACHE_SWEEP_INTERVAL` per prefix directory.
- A whole-directory tree (`cache.tree_path`) is used unchecked only when its `.content-checked`
  marker holds the listing's bytes. Restores and `cache.missing_cache_objects` check it with
  `repair=False`; only the commit path repairs (`cache._ensure_tree_complete`).
- A directory restore starts from that tree only where it links the tree whole
  (`cache.restore_tree`: SYMLINK first, and links into the cache allowed or no COPY among the
  modes); `cache.missing_cache_objects`, which `pivot get` and `checkout --force` ask, reads the
  same rule, so where links are refused and COPY is among the modes it looks up each member's
  object, as the restore that copies it does.
- A listing is also a cache object, at the directory's hash (`listing.store_object`).

## The naming lock and the register

[What Keeps a Cache Object Alive](../concepts/caching.md#what-keeps-a-cache-object-alive)
describes the protocol and the register as users see them. The writer side is
`storage/naming.py`; the collector, `storage/gc.py`, imports it. Writers import only `naming`, so
a worker never pays to import the collector.

- Naming-unit sites (`naming.naming_unit`): `executor/worker.py` (commit and run-cache skip),
  `executor/commit.py`, `cli/track.py`, `storage/restore.py` (restore from a revision) and
  `naming.register_use`. `WorkspaceStore.prepare_commit` hashes and publishes before the unit
  (`cache.publish_hashed_output`); `WorkspaceStore.finish_commit` re-checks and links inside it.
- The register and the lock files belong to the object store: `naming.store_layout` and
  `naming.register_root` find them, and writers refuse an `unshared` layout
  (`naming.unshared_store_problem`) as `gc.resolve_scopes` does.
- `Engine._orchestrate_execution` registers the project (`naming.register_use`) before any stage
  runs, so a project is in the register while its stages run. A restore that links objects
  outside a unit registers first, in the cache it restores from: `config.register_cache_use` for
  `checkout` and `pull`, `restore_outputs_to_hashes(register=…)` for a skipped stage, and
  `verify_tracked_files` for `--checkout-missing`. A restore inside a unit registers nothing more:
  a nested unit could wait on a collection that waits on the outer one.
- Where a registration fails with `EROFS`, the process makes no link into that cache
  (`cache.refuse_links_from`); a restore copies instead, or refuses when copy is not a checkout
  mode (`cache.refuse_unregistered_link`).
- A collection reads the register with `gc.read_register` and checks every other cache a project
  lists with `gc._other_caches_reason`. It removes nothing through a path: `gc.store_snapshot`
  holds each store directory open, and `gc.execute_plan` removes through descriptors that
  `gc._open_directory_at` checked.
- A record gc cannot fully interpret refuses the collection (`gc._lock_keys`, `gc._pvt_keys` and
  `gc._run_cache_keys` raise `_UninterpretableRecordError`; `gc.collect_roots` turns it into
  `ProjectUnreadableError`). The one exception is a `.pvt` a stage wrote into its own output.
