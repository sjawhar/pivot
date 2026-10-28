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

The rule: commands that read a file may save its hash to the cache, best-effort; no read-only command writes a lock, a run record or an output. `--no-commit` writes no lock and no `run:` history entry. It writes a `nocommit:` record holding what the run did, which a later `pivot commit` uses.

Every command that writes the workspace saves the hash of each file it reads, commits or restores, so an explanation (`pivot status --explain`, `pivot verify`) finds a matching entry and reads no file. Saving file hashes never fails a command. `state.open_readonly(..., create=False)` looks file hashes up without creating a database: on an existing database, an `OSError` or `lmdb.Error` yields no cache only when `os.statvfs(state.lmdb)` says the database itself is on a read-only filesystem (the parent may be a different writable mount); otherwise it raises (a damaged database, or one unopenable on a writable mount). `pivot checkout` catches any failure to open the hash cache and restores without it, since main's checkout never opened the database; it catches only the open, so an error inside its own block still propagates. Status and verify may raise, as on main. Every save of file-hash entries goes through `state.save_best_effort` (`state.save_file_hashes` for `(path, stat, hash)` entries) or `StateDB.save_many_best_effort` (`hash_file`'s and `hash_directory`'s save-on-read), which wait at most 2 s for the write lock, log a failed write at debug level and keep the command's result. A run's deferred writes are not a cache save: they put a stage's file hashes in the same transaction as its generations, and that transaction is the run's own commit. A new command that writes or reads workspace files must record them the same way:

- `hash_file(..., pending=...)` and `hash_directory(..., pending=...)` append `(path, stat, hash)` for each file they had to read instead of saving it; the caller saves them in one `state.save_file_hashes` call. A command over many targets collects them in a `state.FileHashBatch` and saves once per state directory when it ends (`checkout`, `get`), not once per target.
- A restore records each file with the stat of what it staged, taken before the rename puts it in place (the same inode): `cache.restore_from_cache(..., recorded=...)`, `cache.atomic_write_file`'s return value. A write that lands after the rename then changes the stat, so the entry does not vouch for it.
- Workers open StateDB read-only. They hand back, in `DeferredWrites.file_hash_entries` (`(path, mtime_ns, size, dev, inode, hash)` tuples, `types.FileHashEntry`, from `cache.deferred_file_hash_entries`), the dependencies and outputs they read to decide (also when the stage is cached by lock comparison), the outputs they commit and the outputs they restore; the coordinator writes them after the stage. The coordinator's own skip check writes what it reads and restores.
- `pivot checkout` (every mode, also the files `--only-missing` keeps or restores and the existing files it compares), `pivot pull`, `pivot get` (from the cache, from git or from the remote), `pivot import`, `pivot update`, `pivot track` and repro's restore of missing tracked files (`--checkout-missing`) record what they write or read. Stage outputs go in the StateDB of the output's stage state directory, which explanations read; everything else in the configured state directory.

## StateDB Key Prefixes

StateDB uses key prefixes for namespacing. Current prefixes in `pivot/storage/state.py`:

| Prefix | Purpose |
|--------|---------|
| `hash:` | File content hashes (keyed by the absolute path as given, `os.fsencode(os.path.abspath(path))`; the value's `(mtime_ns, size, dev, inode)` comes from `stat()`, which follows symlinks. No ctime: a new hard link changes it, so every hardlink checkout or hard-linked replica would invalidate its entries) |
| `gen:` | Output generation counters (keyed by artifact identity, `stage` or `stage:key`) |
| `dep:` | Stage dependency generations (`stage:dep_identity` → generation; one file of a directory output records its directory's identity) |
| `runcache:` | Run cache entries (`stage:input_hash` → output hashes) |
| `sm:` | Fingerprint manifest cache (stage manifests + source maps) |
| `run:` | Run history entries |
| `remote:` | Remote index entries |
| `remote_url:` | Remote URL tracking for change detection |
| `fp:` | AST fingerprint/hash cache entries |
| `nocommit:` | What a stage's outputs hold that no lock records yet (`stage` → a JSON object, `types.NoCommitRecord`): a `--no-commit` run's code manifest, marks included, and output hashes, for `pivot commit`, and the `*.pvt` directory members another feature writes (`pvt_members`); a written lock removes the whole record. Each writer replaces only its own keys and keeps every other key, in one write transaction. The reader validates each of the three keys on its own, dropping only a malformed one, and returns every other key as stored, so a record holding only another feature's key reads as present; `pivot commit` uses the run's manifest only when both of the run's keys are present |

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
  temporary entry, and two hosts can repair one tree at once. A commit checks the tree it
  published before marking it, which narrows the window for marking a copy that lost files
  without closing it: a sweep already deleting through the copy's directory descriptors keeps
  deleting after the rename. Where `flock()` is refused (Lustre without `flock`: ENOSYS; NFS when
  its lock manager does not answer: ENOLCK) staging goes on without the shared lock, and this
  process neither sweeps nor repairs a tree. Only where every host sharing the cache is refused
  does nothing sweep: an NFSv3 client whose own `rpc.statd` is down gets ENOLCK while other
  clients' locks work, and their sweeps can remove its live temporary entries, with the effects
  above.
- A lock file in the cache is created writable by whoever can write its directory, since NFS
  takes an exclusive lock only on a file opened for writing; one this user cannot write is
  opened read-only. Lock files are never deleted.
- A whole-directory SYMLINK tree is built under a temporary name beside its cache name, renamed
  into place, checked, then marked (`<tree>.content-checked`), because a restore symlinks to a
  tree as soon as its name exists. The marker is staged under the prefix directory's shared lock
  (`_staging_in`), so a sweep never removes it between a check and its rename. A restore or commit
  checks every unmarked tree member's path, content hash, size and executable bit, plus its
  directories, under the tree's exclusive `<tree>.lock`, and marks a tree that matches. Only a
  commit repairs a tree that does not match, in place from the files' own cache objects
  (`_ensure_tree_complete(..., repair=True)`): its manifest is `hash_directory` of the bytes it
  commits. A restore never writes into a tree. A tree the restore or commit cannot use (it
  differs and, for a commit, cannot be repaired: a file's object is absent, or this user cannot
  read or write the tree) is not used: a restore goes file by file, a commit fails. Only
  `<tree>.content-checked` is trusted: `<tree>.layout-checked` (pivot 0.3.4, partial trees
  included) and `<tree>.verified` (pivot 0.3.5 to 0.5.11, sizes only) are older marks that never
  checked content, so such a tree is checked again, and the old mark is removed when the tree is
  marked. The mark is only written where this user can write the tree's prefix directory: for a
  user who cannot, every symlink restore or commit checks an old-marked tree again, and a symlink
  commit through a stale one fails, until a user who can write the cache commits the output in
  symlink mode. That cost is the price of not trusting a size-only mark.
- A tree's name is the XXH64 of its manifest. Content addressing by XXH64 guards against
  accident, not against a chosen-digest forgery: XXH64's finalization is invertible, so 16 free
  bytes near the end of a manifest someone commits in a `.pvt` or lock steer it to any tree's
  name in under a second. No **restore** trust decision may rest on an attacker failing to hit an
  XXH64 digest: a restore's manifest is a claim and never drives a write into the cache. A commit
  is different: its manifest is `hash_directory` of the stage-written bytes, and it may repair an
  unmarked tree those bytes' XXH64 names. This is identical at main: a user's own stage can write
  colliding bytes into a tree an output is linked to, where they still match the lock; closing it
  would give up #522's automatic repair of stale legacy trees. The marker and canonical-manifest
  rules below protect against committed `.pvt` and lock manifests, not stage-written collisions.
  For those committed manifests, a tree is trusted only for the manifest pivot checked it to hold:
  the marker records the SHA-256 of that manifest's canonical JSON
  (`cache._canonical_manifest`: the four fields pivot writes per entry, entries sorted by relpath
  and empty directories sorted, as `hash_directory` writes them, a relpath or empty directory
  listed twice refused; `cache._mark_tree`), and a restore trusts the tree only when the caller's
  canonical manifest has that digest. A tree marked for another manifest is neither trusted nor
  changed; a tree with an empty marker (an older pivot's) is checked but never repaired; an
  unmarked tree is checked, and marked only if it holds the caller's manifest exactly, its
  recorded empty directories included (exactly the tree's, as `hash_directory` finds them). So
  one tree has one canonical manifest, and no other form of its entries (another order, a
  duplicate, a full directory listed as empty) can mark it for another digest. A restore's
  manifest comes from a `.pvt` or a lock, so it is a claim, not a source of truth: it never drives
  a write into the cache. Repairing toward it and then comparing the result with that same
  manifest proves nothing (a forged manifest would rewrite the bytes of every output already
  linked to the tree). A commit repairs toward `hash_directory` of the bytes it commits, and after
  a repair the tree's manifest is recomputed and must equal that manifest entry for entry before
  the tree is marked.
  A manifest whose XXH64 is not the tree's name, or that lacks a field pivot writes or holds one
  JSON cannot encode, is refused before anything is read; that only spares work. Any refused
  manifest restores file by file, where each restored file is the cache object its own hash
  names. `.pvt` manifest relpaths that are absolute or hold `..` are refused
  (`track.pvt_field_error`, `track.read_pvt_file`).
  Known limits: pivot does not re-read a marked tree, so a tree whose files someone with write
  access to the cache edits, or whose marker they rewrite, after it was marked stays trusted; an
  empty marker an older pivot wrote, which this user cannot replace, costs a content check on
  every symlink restore through that tree until a user who can write the cache re-marks it; an
  unmarked tree that does not match is restored file by file until a commit of that output
  repairs it; and a symlink commit can replace an unmarked tree with stage-written bytes whose
  `hash_directory` XXH64 collides with the tree's name. That is identical on main: an output
  already linked to the tree then reads the stage-written bytes and still matches its lock. A
  marked tree refuses this commit, but closing the unmarked case would remove #522's automatic
  repair of stale legacy trees.

## Path Storage

All paths stored in the database must be **relative** to ensure portability across machines and correct cache behavior.

See `docs/solutions/2026-02-01-statedb-path-strategies.md` for path handling patterns.

## Cache Lock Order (the naming-lock protocol)

`pivot gc` is the only pivot command that deletes a user's cached data. Its safety rests on
mutual exclusion with writers, not on timing. The protocol uses two reader/writer locks plus a
turnstile to prevent starvation. **The lock order must never be inverted; any path that takes
them in another order is a deadlock bug.**

```
gc:      gc.lock (EX)  →  prefix .pivot_cache.lock (EX, one at a time, to list)  →  naming.turnstile (EX)  →  naming.lock (EX)
writer:  naming.turnstile (SH)  →  naming.lock (SH)  →  (turnstile UN)  →  prefix .pivot_cache.lock (SH, only to re-publish)
```

- **`gc.lock`** (EX, non-blocking): excludes two gc runs on one cache. Taken first, held for the
  whole run including the prompt. Writers must NOT take it.
- **`naming.turnstile`** (EX for gc, SH for writers): the turnstile prevents writer starvation.
  gc takes it exclusive (new writers queue), then takes naming exclusive (waits for in-flight
  units). Writers take it shared, then naming shared, then release turnstile immediately.
- **`naming.lock`** (EX for gc, SH for writers): the mutual-exclusion lock. While gc holds it
  exclusive, no writer can name a cache object. While a writer holds it shared, gc waits.
- **Prefix `.pivot_cache.lock`** (EX for gc's listing, SH for a staging publisher): gc lists a
  prefix only under it, so it waits for a publish in progress there before it takes any naming
  lock. gc **never** takes a prefix lock while it holds `naming.lock` exclusive: it must not
  wait on another process there, or every writer waits behind it. Its deletions need no prefix
  lock, because a publisher outside a naming unit is re-checked inside one before anything
  names its object.

### Writer commit unit

A naming unit covers only the **naming step**: from the **last existence check on each cache
object** to the **durable record that names it** (lock file, `.pvt`, run-cache row or workspace
link). It must NEVER span stage execution, hashing, copying or a download. Hash and publish
before the unit; download outside it; inside it, re-check every object (publishing again any a
collection removed since) and then link and write the record. See `gc.naming_unit`,
`WorkspaceStore.prepare_commit` / `finish_commit`, and `cache.publish_hashed_output`.

Sites: `executor/worker.py` (normal commit and run-cache skip), `executor/commit.py` (`pivot
commit`), `cli/track.py` (`pivot track`), `storage/restore.py` (restore from revision), and
`config.register_cache_use` (a register write). Every site and `pivot gc` take the lock files
from `gc.naming_lock_paths(<cache root as configured>)`, never from a path derived from the
resolved object root (a `files/` symlink would split the lock).

**Every unit registers its project** (`naming_unit(cache_root, project=(root, state_dir))`),
under the shared lock and before anything is named, so every process that names an object is
in the register, whichever command it runs and whether or not the cache existed when it began.
Only a test standing in for a writer passes `project=None`. Registration tries to write the
project id, with the inode of the state directory it was made in, to
`<state dir>/state.lmdb/pivot-project-id` (a copy gets a new id), records mount points at or
below the project root and state directory, and records cache-root submounts in
`<cache>/mounts/`. Entries and cache mount records are keyed per view (hostname and mount
namespace). An id write failure leaves an id-less entry, so a collection refuses it but the
writer preserves #503's unusable-state behavior. A register write failure refuses the writer.
The only exception is `EROFS` on the cache: a read-only cache cannot be collected either. A
restore that links a cache object into a workspace outside a naming unit (`checkout`, a skipped
stage's restore, `--checkout-missing`) runs a registering unit first.

**A registered project is live only after a real read.** `read_register` resolves the state
directory from that project's own `core.state_dir` and requires the entry's project id there,
made in that state directory. For this view's entries every recorded mount must be mounted;
another view's entry may record only its root and state directory. Any missing evidence
refuses collection and names the entry. `resolve_scopes` then opens every scope's root,
`.pivot` and state directory. When `core.state_dir` differs from `.pivot`, collection reads
both state layouts so a 0.5.2 lock in `.pivot` still protects its object. The writer-side
registration memo (`_VERIFIED_REGISTRATIONS`) only lets a naming unit skip re-registration; no
collection read uses it.

### Where flock is refused

**gc refuses, writers proceed.** On a filesystem that refuses `flock` (NFS `nolock`, Lustre
without `flock`), `collection_lock` refuses the gc run outright — no objects are deleted. Writers
proceed unlocked because there is no collector to race. If a future change makes gc collect on
such a mount, the writers must also take the lock. This pair is by design; do not change one side
without the other.

### Where the lock files cannot be opened

**Writers refuse.** If a writer cannot open `naming.turnstile` or `naming.lock` (they do not
exist yet and this user cannot create them in the cache root), `naming_unit` raises
`CacheError` naming the file and the remedy: another user's `pivot gc` could create the files
and delete while this writer named objects unlocked. The only exception is `EROFS`: on a
read-only filesystem nothing, a collection included, can delete. Lock files that exist but are
not writable are opened read-only and locked (`cache.open_cache_lock`).
