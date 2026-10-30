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
- Parameters compare by canonical JSON encoding, so values with different serialized types invalidate
  a stage even when Python considers them equal.
- Dependency hashes (content hashes of all input files/directories)
- Output path list (detect added/removed outputs)

This requires hashing all dependency files, hence O(n) in the number/size of dependencies.
The coordinator reads the stage lock only after holding the stage's artifact locks, so a waiting
run evaluates the state committed by the writer ahead of it. It takes READ on the outputs to
decide, and WRITE (released READ first, then decided again) only to restore one from the cache;
see `docs/architecture/execution.md` ("Artifact Locks").

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

Workers open StateDB in **readonly mode** during execution to avoid write contention. (One write
cannot wait for the coordinator: a worker adds a `*.pvt` member to the stage's `nocommit:` record
before it writes that member, so a crash never leaves it unrecorded; see `storage/AGENTS.md`.)
However, dependency hashing produces file hash entries (path, mtime_ns, size, dev, inode, hash)
that should be cached in StateDB for future O(1) lookups.

The solution: workers collect these entries in `file_hash_entries` (returned from `hash_dependencies()`) and include them in `DeferredWrites`. The coordinator writes them back to StateDB after stage completion via `apply_deferred_writes()`. This gives the performance benefit of hash caching without requiring workers to hold write locks on StateDB.

`DeferredWrites.file_hash_entries` is a list of `(path, mtime_ns, size, dev, inode, hash)` tuples (`types.FileHashEntry`). The coordinator writes these entries back to StateDB as provided. If a file changed between the worker's read and the coordinator's write, the entry may contain stale metadata — the next run will detect the mismatch and re-hash (self-healing).

## Logical Path vs Physical Identity

Lock files and StateDB use different path semantics:

- **Lock files** store project-relative paths. In memory, paths are first normalized to absolute form via `project.normalize_path()`, and `storage/lock.py` converts between those absolute paths and the project-relative paths written to (and read from) the lock file. Symlinks are preserved. Using project-relative paths ensures portability across machines and stable lock file content.

- **StateDB file hash cache** keys an entry by the absolute path as given (`os.fsencode(os.path.abspath(path))`), with no symlink followed. The entry holds the `(mtime_ns, size, device, inode)` of a `stat()` that follows symlinks, so two links to one file hold two entries, and a link re-pointed at another file, on this filesystem or another, no longer matches.

- **StateDB generation counters** are keyed by the artifact identity (`stage` or `stage:key`), not a path, so a key never moves when an output becomes a symlink into the cache.

## Manifest Cache

Fingerprint manifests (the code_manifest dict mapping `"self:func_name"` → hash) are cached in StateDB under the `sm:` prefix, along with source maps that record which source files contributed to the manifest. The key names the stage function by its identity (`fingerprint.function_identity`: module, qualname, project-relative source file, code hash, closure cell values and default values) as well as the stage name, because two `pipeline.py` files can declare the same pipeline name and so register the same stage name with different functions, and one factory (`make_stage(2)`, `make_stage(5)`) builds functions that differ only in a closure cell. The same identity decides whether `Pipeline.include()` sees one stage reached twice or two stages under one name. A source map holds the files of the functions walked plus the file of every project module loaded when the manifest was computed: a manifest records values bound in other modules' namespaces (`settings.FACTOR`, `from settings import FACTOR`, a module rebinding an imported helper), and an edit to any of those modules must invalidate the entry. An entry also records every live value the walk read: each global, closure cell, default argument, `module.NAME` attribute, partial argument, class in type hints and pydantic schema (field defaults included), as a path from the stage function and a token for what was found there. A value computed at import from the environment, a settings file or the date changes without touching any source file, so the cache saves only the walk: a hit requires every recorded file's content digest to be unchanged and every recorded read to give its token again (one dict, cell or attribute read per entry; a schema hash is computed once per class per process). The source map and the function-hash cache (`fp:`) record a file by its content, not its mtime, size and inode, which an edit can keep (two writes inside one kernel timestamp tick, or a tool that restores the mtime); each file is read once per `fingerprint.fingerprint_pass()`. Manifests computed with unsafe-capture warnings are never cached. `pivot fingerprint reset` clears both `fp:` and `sm:` entries.

**Flush boundaries** (where pending manifest cache writes are committed to StateDB):
- After discovery loads `pipeline.py` (`discovery.py`)
- In watch mode, before reload
- As each CLI invocation ends (`fingerprint.flush_and_close`, from the `cli` callback's close callbacks). Process exit runs it again for a library caller and for a TUI session whose engine thread fingerprints after that callback.

**Where a pending entry is written:** each entry is filed under the state directory read where it was produced (`fingerprint._producing_state_dir`), and every flush writes it there and resolves no state directory. A CLI invocation holds that directory for the whole command. A library caller has no invocation, so every library entry point that produces entries holds one state-directory record for its whole call (`state_dir.entry_invocation`): an engine run (`executor.run`), discovery, a pipeline's load, each `Pipeline.ensure_fingerprint`, each `fingerprint.get_stage_fingerprint_cached` and `executor.commit.commit_stages`. A pipeline resolves `core.state_dir` once, when it loads, and records that directory on its stages; its fingerprints, runs and commits start their record from it (`Pipeline.resolved_state_dirs`). The record keeps each root's outcome, failures included: a value that cannot be used (`''` falls back to `.pivot`; `5` fails), and a path that cannot become a directory (a regular file, a read-only parent), kept as `StateDirTargetError`. For that path the stages keep `.pivot`, and the record decides once, with the value, what a command that needs the configured directory gets (`StateDirTargetError.fallback`): `.pivot` when the path still holds a state database the process can write, otherwise the path, where the command fails as 0.5.2 did. A pipeline keeps the `.pivot` fallback for such a path (`state_dir.kept_by_pipeline`), and a worker's record starts with its Pivot process's resolution, not a directory derived from it. One entry point reads the configuration at most once. So a pipeline's fingerprints and cache entries stay in one directory, also when a stage or the caller edits `core.state_dir`, re-points a symbolic link it names, or makes the path it names usable after the load. A run of such a pipeline (`executor.run(pipeline=...)`) refuses instead (`discovery.require_current_state_dirs`, which reads the configuration now through `state_dir.current_resolution`): its locks would name objects in a directory `pivot gc` no longer reads. A caller who wants the new directory loads the pipeline again, and `include()` refuses to combine pipelines whose loads resolved two directories. Nothing derived from a state directory is remembered outside a record: the default cache (`<state dir>/cache`) is derived where it is used, the readonly lookup database follows the producing directory, and a low-level call such as `fingerprint.get_stage_fingerprint` by hand reads and resolves the configuration for each lookup. A flush runs where its producer no longer does, so an edit between production and the flush takes no entry either.

**Selective invalidation in watch mode:** When a source file changes, the engine uses a reverse index to find which manifest cache entries depend on that file, and invalidates only those entries. This avoids re-fingerprinting all stages when only one source file changed. The reverse index is built by scanning all `sm:` entries' source maps.

## Safe Fingerprinting

Pivot fingerprints stage functions by hashing their AST and the ASTs of any helper functions they call. A module-level `bool`, `int`, `float`, `str`, `bytes` or `None` that a stage reads is recorded by value (`const:` keys), so editing it re-runs the stage. So is a `tuple` or `frozenset` whose members are all such scalars (frozenset members sort by canonical JSON text, never by `hash()`, which is seeded per process) and a compiled regex (its full pattern and flags; its `repr` truncates the pattern at 200 characters). Any other module-level value a stage reads (list, dict, set, a tuple holding a non-scalar, enum member, class instance, or a method bound to an instance such as `np.random.default_rng(SEED).normal`) is not safely trackable: its contents can change without the AST changing. A library function or class it reads is recorded by name (`ext:` keys).

By default, such captures raise `StageDefinitionError` with a message that names the variable and suggests moving it into `StageParams`.

**Escape hatches:**
- Config: `core.unsafe_fingerprinting: true` in `.pivot/config.yaml` (`pivot config set core.unsafe_fingerprinting true`)
- Environment: `PIVOT_UNSAFE_FINGERPRINTING=1`

When enabled, these captures produce a warning instead of an error. This is unsafe because it can lead to silent wrong outputs: the captured value may change without triggering re-execution. A stage whose fingerprint needed the downgrade (`RegistryStageInfo.unsafe_fingerprint`, copied to `WorkerStageInfo`) is kept out of the run cache: the worker neither looks it up nor files the result, and `pivot commit` does not file it either.
