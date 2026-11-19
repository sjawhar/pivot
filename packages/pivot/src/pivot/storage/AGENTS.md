# Pivot storage

StateDB (LMDB), lock and listing files, the content-addressed cache, restores, output ownership
and `pivot gc`. Which module keeps each rule below, and the StateDB prefix table:
`docs/architecture/storage.md`. User-facing behaviour: `docs/concepts/caching.md`.

Test: `uv run pytest packages/pivot/tests/storage -n auto`; one file:
`uv run pytest packages/pivot/tests/storage/test_state.py`.

## StateDB
- One LMDB environment per state directory (`state.py`). New state gets a key prefix constant in
  `state.py` and a row in the docs page's table, not a new database.
- Workers open StateDB read-only and return `DeferredWrites`; the coordinator applies them in one
  transaction (`StateDB.apply_deferred_writes`). One exception: a worker records a `*.pvt` member
  in `nocommit:` before it writes the file (`ownership.record_members`); if that fails, the stage fails.
- Dependency generations live only in StateDB (`dep:`), never in lock files.
- Writes go through `StateDB._write_transaction`: its flock on `pivot-write.lock` only adds a
  timeout (`PivotDBWriteTimeoutError`); LMDB serializes the writers.
- `hash:` and `inode:` keys are absolute paths (one file on this machine); the other keys hold
  artifact identities or project-relative paths. The `hash:` stat has no ctime: a hard link changes it.

## File-hash write-back
- A command that reads or writes workspace files saves each file's hash: pass `pending=` to
  `cache.hash_file`/`hash_directory`, then `state.save_file_hashes` or `state.FileHashBatch`.
- Hash-cache saves are best-effort (`state.save_best_effort`, `StateDB.save_many_best_effort`):
  a failed save never fails the command.

## Locks, listings and `.pvt` files
- `StageLock.write` is the one writer of lock, listing and restart-proof files. Pass the cache to
  every `StageLock.read`: it stores an older lock's inline listings there.
- A new lock key that names cache objects joins `lock._LOCK_KEYS` and the roots of
  `lock.recorded_hashes` in the same change: gc refuses a record it cannot fully read.
- Only `WorkspaceStore.hash_artifact` passes `ignored` to `cache.hash_directory`, for a
  `pipeline.input` directory. Outputs, tracked data and restores hash every file (a file left
  out is lost on restore); no restore reads an `=inputs` listing.
- `ignore.py` is a port of git's matcher; keep it a port (a regex backtracks where `dowild`
  aborts). `packages/pivot/tests/storage/test_input_ignore_matches_git.py` compares it with git.
- `track.discover_pvt_files` finds every `.pvt` that `os.walk` finds: never prune it by directory
  name or stage output. Another stage's `*.pvt` members are decided from recorded locks
  (`member_skip.own_member_skip`), never by importing its `pipeline.py`.

## Output ownership
- Pivot replaces only what it owns: a tracked or imported file and its `.pvt` are the user's,
  also inside a stage output.
- Every path that removes or replaces an output calls `ownership.refuse_output_replacements` (or
  `find_output_overlaps`) before it removes anything. Add the call to each new such path. Stage
  writes use `WorkspaceStore.refuse_output_replacements` and `WorkspaceStore.prepare_output`.
- `ownership.stage_members` is the one rule for whether a `*.pvt` in an output is the stage's.
- Keep the common case (no `.pvt` near an output) at one `lstat` per path the check looks at.

## Cache objects and restores
- Publish file objects with `cache.publish_cache_object`. A published object never changes, mode
  included.
- Make each temporary entry in a prefix directory with `cache.staged_cache_file` or `_staging_in`
  (`.pivot_tmp_*`, under a shared flock on `.pivot_cache.lock`). Never judge one abandoned by
  host and pid: a process in another PID namespace or on another host looks dead.
- Never delete a lock file in the cache or in `<state dir>/locks/`: a waiter can hold the old
  inode while a new opener locks a new one.
- A cache shared across hosts needs flock coherent across them. With host-local locks, one host's
  sweep can remove another host's live temporary entry, or files of a tree being built. Where
  flock is refused, staging goes on without the shared lock (a host whose locks work can still
  sweep those entries), and that process neither sweeps nor repairs a tree. So a tree is marked
  only once it is checked, and only a commit repairs one.

## Cache lock order (the naming-lock protocol)
```
gc:     gc.lock EX → prefix .pivot_cache.lock EX (one at a time, to list) → naming.turnstile EX → naming.lock EX
writer: naming.turnstile SH → naming.lock SH → turnstile UN → prefix .pivot_cache.lock SH (only to re-publish)
```
- Take them in no other order. Writers never take `gc.lock`; gc takes no prefix lock while it
  holds `naming.lock`.
- A naming unit (`naming.naming_unit`) covers only the naming step: from the last existence check
  of each object to the record that names it. Hash, publish and download before it; never run a
  stage inside one. Get the lock paths from `naming.naming_lock_paths(<cache root as configured>)`.
- Every unit registers its project; a restore that links objects outside a unit registers first.
- Between publish and record, gc keeps every object whose `st_ctime` is inside the grace period
  (`--grace-hours`) and every object another directory entry hardlinks.
- gc refuses, writers proceed. Where flock is refused, `gc.collection_lock` refuses the collection,
  so writers run unlocked. Change both sides together.
- Writers refuse when they cannot open `naming.turnstile` or `naming.lock`, except on `EROFS`.
