# Rearchitect Commit: Drop Pending Locks, Drop --no-cache

**Goal:** Eliminate the "pending lock" concept, simplify `--no-commit` mode, drop `--no-cache`, and remove `None` from `OutputHash` throughout the codebase.

**Architecture:** (1) `--no-commit` hashes outputs but skips caching/locks/generations, writing only run cache entries, (2) `pivot repro` still commits by default, (3) `pivot commit` snapshots current workspace state using the registry and outputs on disk (the "trust me" path for code changes with no behavioral change), (4) `OutputHash` becomes non-nullable (`FileHash | DirHash`).

**Tech Stack:** Python 3.13+, pytest, pivot internals

---

## Context

After #371 (non-cached outputs get real hashes), `compute_input_hash_from_lock()` produces wrong input hashes because it infers the `cache` flag from `output_hash is None` — which no longer holds. This is the immediate bug (#372).

But the deeper issue is architectural: `commit_pending()` reads pending lock files and recomputes derived state (input hashes) that the worker already computed at execution time. The pending lock concept duplicates data that exists (or should exist) in the run cache, and `--no-cache` adds complexity by introducing `None` output hashes throughout the system.

### What changes

| Concept | Before | After |
|---------|--------|-------|
| `--no-commit` | Writes pending locks, skips caching | Runs stage, no durable writes (no cache, locks, generations, or run cache) |
| `pivot commit` | Reads pending locks, recomputes input hash | Snapshots workspace state (registry + filesystem) for "trust me" commits |
| Pending locks | Full lock data in `.pivot/pending/stages/` | Removed entirely |
| `--no-cache` | Skips hashing/caching, stores `None` hashes | Removed entirely |
| `OutputHash` | `FileHash \| DirHash \| None` | `FileHash \| DirHash` (same as `HashInfo`) |
| `input_hash` availability | Recomputed from lock data | Propagated through `StageResult` from worker |

---

## Invariants and edge-case decisions

- **Single writer for durable state:** worker-run commits and `pivot commit` both flow through the same commit writer; no duplicate derivation logic.
- **`pivot repro` still commits by default:** `--no-commit` is explicitly non-durable (no locks/generations/cache/run cache).
- **Filesystem is source of truth for `pivot commit`:** it recomputes `input_hash` from registry + deps + params and hashes current outputs on disk. It does not consult run cache, and it updates/overwrites the run cache entry for the computed `input_hash`.
- **Missing outputs:** `pivot commit` errors for that stage and writes no lock/StateDB entries for it. The command exits non-zero if any stage fails; other stages may still commit.
- **Output existence rules:** file outputs must exist; directory outputs must exist and their hash includes a full manifest (adds/removals change the hash). Missing directories are treated as missing outputs.
- **`StageResult.input_hash` rule:** required once dependency hashing begins; `None` is only for early failures before dep hashing.
- **Skip behavior:** `--no-commit` does not create run cache entries, so subsequent `pivot repro` always recomputes and re-commits.

## Design

### 1. Drop `--no-cache`

`--no-cache` skips output hashing and stores `None` in lock files. With the new `--no-commit` (hashes but doesn't copy to cache), `--no-cache` provides marginal benefit (skipping hash computation) at significant complexity cost (`None` flowing through the entire system).

**Removals:**
- `no_cache` field from `WorkerStageInfo`
- `--no-cache` CLI flag from `pivot repro` / `pivot run`
- `_verify_outputs_exist()` in `worker.py` (only existed for `--no-cache`)
- IncrementalOut / `--no-cache` incompatibility check in `execute_stage()`
- All `output_hash is None` checks throughout the codebase

**Type simplification:**
- `OutputHash = FileHash | DirHash | None` becomes just `HashInfo` (`FileHash | DirHash`)
- `LockData["output_hashes"]` becomes `dict[str, HashInfo]`
- `OutputHash` type alias removed (redundant with `HashInfo`)
- `output_hash_to_entry()` always returns an entry (never `None`)
- `OutEntry["hash"]` becomes `str` (not `str | None`)

**Files affected:**
- `src/pivot/types.py` — remove `OutputHash`, update `LockData`, `OutEntry`
- `src/pivot/executor/worker.py` — remove `no_cache` paths, remove `_verify_outputs_exist`
- `src/pivot/storage/lock.py` — update `_convert_to_storage_format`, `_convert_from_storage_format`
- `src/pivot/run_history.py` — simplify `output_hash_to_entry`
- `src/pivot/remote/sync.py` — remove `is not None` checks (replaced by registry filtering)
- `src/pivot/cli/checkout.py` — remove `None` checks
- `src/pivot/cli/verify.py` — remove `None` checks

### 2. Drop pending locks

The "pending" concept is a staging area that duplicates data belonging in the run cache. With the run cache written at execution time (even in `--no-commit` mode), pending locks are redundant.

**Removals:**
- `commit_pending()` and `discard_pending()` in `src/pivot/executor/commit.py`
- `COMMITTED_RUN_ID` sentinel
- `_PENDING_DIR`, `get_pending_stages_dir()`, `get_pending_lock()`, `list_pending_stages()` in `src/pivot/storage/lock.py`
- `pending_state_lock()` in `src/pivot/storage/project_lock.py`
- Pending lock reads/writes in worker (`execute_stage` lines 208, 228-230, 298, 1014-1018)
- `--list` and `--discard` flags on `pivot commit`
- `.pivot/pending/` directory concept

### 3. Rearchitect `--no-commit`

`--no-commit` becomes: execute the stage and hash outputs, but write nothing durable (no cache copy, no locks, no generations, no run cache).

**Worker changes in `execute_stage()`:**

After stage function runs:
```
if no_commit:
    output_hashes = {str(out.path): _hash_output(path) for out in stage_outs}
else:
    output_hashes = _save_outputs_to_cache(stage_outs, files_cache_dir, checkout_modes)
```

In `_commit_lock_and_build_deferred()`:
```
if no_commit:
    # No durable writes in --no-commit mode
    return _build_noop_deferred()
production_lock.write(lock_data)
return _build_deferred_writes(stage_info, input_hash, output_hashes, state_db)
```

`--no-commit` does not write run cache entries, so subsequent `pivot repro` always recomputes and re-commits.

**What the coordinator applies for `--no-commit` deferred writes:**
- Nothing (noop)

### 4. New `pivot commit [stage_names...]`

Replaces `commit_pending()`. Computes from current workspace state using the registry.

**Behavior:**
- `pivot commit` (no args): commits all stages where production lock doesn't match current input state (same stages `pivot status` would show as stale)
- `pivot commit train preprocess`: unconditionally commits the named stages

**Intended use case:**
- You changed code but outputs are still correct (expensive stages). `pivot commit` hashes current deps and outputs, then writes locks/state without re-running the stage.

**Algorithm (per stage):**
1. Compute current state: fingerprint code, hash deps, get `out_specs` from registry
2. Compute `input_hash` via `compute_input_hash(fingerprint, params, deps, out_specs)`
3. Compare to production lock — skip if unchanged (unless explicitly targeted)
4. Hash outputs on disk, save to cache (`save_to_cache`)
5. Write production lock
6. Update StateDB: dep_generations, output generations, run cache entry

**Notes:**
- `pivot commit` treats the filesystem as the source of truth; it should error for a stage if required outputs are missing.
- Commit within a stage is atomic: either all outputs for the stage are committed and StateDB updated, or none are.
- Use a single commit writer for both worker-run commits and `pivot commit` to avoid duplicated logic.

**Parallelism:**
- Per-stage work (fingerprint, dep hashing, output hashing, cache saving) runs in parallel (thread or process pool)
- StateDB writes are serial (LMDB single-writer constraint)

**Location:** `src/pivot/executor/commit.py` keeps the core `commit_stages()` function. `src/pivot/cli/commit.py` is the thin Click wrapper.

### 5. Remote sync filtering

`sync.py` currently skips `None` output hashes to avoid syncing non-cached outputs. With `None` removed, it needs registry-based filtering instead.

**Pattern (same as checkout/verify from #371):**
```python
stage_info = cli_helpers.get_stage(stage_name)
non_cached_paths = {str(out.path) for out in stage_info["outs"] if not out.cache}
# Skip non_cached_paths when collecting hashes for push/pull
```

**Functions affected:**
- `get_stage_output_hashes()` — filter non-cached outputs
- `_get_file_hash_from_stages()` — filter non-cached outputs
- `get_target_hashes()` — filter non-cached outputs when resolving stage name targets

### 6. Propagate `input_hash` through `StageResult`

The worker computes `input_hash` in `_check_skip_or_run()`. Instead of discarding it and recomputing later, propagate it back.

**Changes:**
- Add `input_hash: str | None` to `StageResult` (required field, `None` for early failures before dep hashing)
- Set it in all `_make_result()` calls and direct `StageResult` constructions
- `_write_run_history()` in `engine.py` reads `input_hash` from the result
- Remove `compute_input_hash_from_lock()` from `run_history.py`
- Update `StageRunRecord["input_hash"]` type to `str | None`

---

## What's NOT changing

| Area | Reason |
|------|--------|
| Run cache storage (stays in StateDB) | Moving to file-based is orthogonal; separate concern |
| Three-tier skip detection | Unchanged in the worker |
| `--no-commit` on `pivot run` | Works the same way (skip caching/locks/generations) |
| Lock file format (StorageLockData) | Only change: `OutEntry["hash"]` becomes non-nullable |
| `_restore_outputs` core logic | Unchanged, just remove `None` branch |

## Verification

1. `uv run pytest tests/ -x -n auto`
2. `uv run ruff format . && uv run ruff check . && uv run basedpyright`
3. Smoke test: `pivot repro --no-commit` → outputs on disk, no lock files → `pivot commit` → lock files written, outputs cached
4. Smoke test: `pivot repro --no-commit` → `pivot repro` (without --no-commit) → re-runs and commits normally
5. Edge case: `pivot commit` with a missing output → stage fails, no lock or StateDB writes for that stage; other stages still commit
6. Edge case: `pivot commit` after outputs changed on disk → hashes and locks reflect current filesystem state
