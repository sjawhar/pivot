# P0: Cache/State Correctness + Fingerprint Caching Completion

## TL;DR

> **Quick Summary**: Fix the core cache/state soundness gaps and complete the fingerprint manifest caching system that's already partially implemented. This makes generation-based skip detection truly O(1), eliminates the dual-source-of-truth for dep_generations, completes #363 (manifest cache flush), enables #358 (selective re-fingerprinting), and adds safe-by-default fingerprinting that errors on unsound closure captures.
>
> **Deliverables**:
> - Generation skip reordered to run before dep hashing (C0)
> - `dep_generations` consolidated to StateDB-only (C1)
> - Manifest cache flush boundaries added for watch mode (C2/#363)
> - Selective re-fingerprinting via reverse index (#358)
> - Safe fingerprinting defaults with mutable-closure error + unsafe mode escape hatch
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 5 → Task 6

---

## Context

### Original Request
Convert P0 items from deep architecture review into an executable work plan. P1/P2 work (Engine refactor, logging transport, concurrency design) is explicitly deferred.

### Interview Summary
**Key Decisions**:
- Single-run correctness first; concurrency deferred
- Aggressive remediation posture (breaking changes OK, pre-alpha)
- Generation skip soundness boundary: Pivot-produced artifacts only
- `dep_generations` single source of truth: StateDB-only (preferred)
- Manifest cache: best-effort perf optimization, never correctness-critical
- Mutable closure captures: ERROR by default, single boolean unsafe mode escape hatch
- Frozen instances: allow via cheap static detection only (no recursion)
- Error messages: name offending variables + suggest `StageParams`/`Dep(...)` fix

### Metis Review
**Identified Gaps** (addressed):
- Mixed deps edge case (Pivot-produced + external) → explicitly gate generation skip
- Legacy lockfiles missing `dep_generations` → handle gracefully in schema migration
- Watch-mode rapid flush perf concern → limit flush to well-defined checkpoints
- Nested mutable structures in closures (tuple containing list) → treat tuple-of-mutable as mutable

---

## Work Objectives

### Core Objective
Make Pivot's skip detection, fingerprinting, and caching systems correct-by-default and performant in all execution modes (one-shot, watch, TUI).

### Concrete Deliverables
- Modified `src/pivot/executor/worker.py`: generation skip before hashing
- Modified `src/pivot/types.py`: `DeferredWrites` extended with file-hash cache entries
- Modified `src/pivot/storage/state.py`: apply file-hash entries in deferred writes
- Modified `src/pivot/storage/lock.py`: `dep_generations` removed from schema
- Modified `src/pivot/fingerprint.py`: flush boundaries, safe-fingerprinting validation
- Modified `src/pivot/discovery.py` and `src/pivot/pipeline/yaml.py`: manifest flush calls
- Modified `src/pivot/engine/engine.py`: manifest flush in watch reload path
- New/modified config: `core.unsafe_fingerprinting` in `src/pivot/config/models.py`
- New/modified tests across `tests/`

### Definition of Done
- [x] `uv run pytest tests/ -n auto` passes
- [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` clean
- [x] Generation skip does not hash deps when generation matches
- [x] `dep_generations` only lives in StateDB; lockfiles no longer store or read it
- [x] Watch-mode manifest cache persists across reloads
- [x] Mutable closure capture raises error by default; `PIVOT_UNSAFE_FINGERPRINTING=1` downgrades to warning

### Must Have
- All existing tests continue to pass
- Each behavioral change covered by new tests
- Backward compatibility for lockfiles missing `dep_generations` field

### Must NOT Have (Guardrails)
- No Engine refactoring (P1)
- No logging transport changes (P1)
- No concurrency/locking design work (P2)
- No new abstractions or module splits unless required for the fix
- No recursive immutability proofs for closure captures
- No parallel fingerprinting (#364) — explicitly deferred
- Follow existing code patterns in each module

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: YES (tests-after — add tests for each behavioral change)
- **Framework**: pytest (`uv run pytest tests/ -n auto`)

### Quality Gates
```bash
uv run pytest tests/ -n auto                                        # All tests pass
uv run ruff format . && uv run ruff check . && uv run basedpyright  # Clean quality
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — independent fixes):
├── Task 1: Reorder generation skip before dep hashing (C0)
├── Task 4: Complete manifest cache flush boundaries (C2/#363)
└── Task 7: Safe fingerprinting defaults + unsafe mode

Wave 2 (After Wave 1):
├── Task 2: Extend DeferredWrites with file-hash cache entries (C0 cont.)
├── Task 3: Consolidate dep_generations to StateDB-only (C1)
├── Task 5: Selective re-fingerprinting via reverse index (#358)
└── Task 6: Add skip-invariant documentation + tests

Critical Path: Task 1 → Task 2 → Task 3 → Task 5 → Task 6
Parallel Speedup: ~35% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 6 | 4, 7 |
| 2 | 1 | 3, 6 | 4, 5, 7 |
| 3 | 2 | 6 | 5 |
| 4 | None | 5 | 1, 7 |
| 5 | 4 | 6 | 2, 3 |
| 6 | 2, 3, 5 | None | None (final integration) |
| 7 | None | None | 1, 4 |

---

## TODOs

- [x] 1. Reorder worker skip detection: generation check before dep hashing (C0)

  **What to do**:
  - In `_check_skip_or_run()` (`src/pivot/executor/worker.py:417`), move the generation-based skip check (`can_skip_via_generation()`) to run BEFORE `hash_dependencies()` is called.
  - Currently the worker calls `hash_dependencies()` unconditionally at the top of `execute_stage()`, then passes `dep_hashes` into `_check_skip_or_run()`. Restructure so that:
    1. Read lock data first (cheap)
    2. Attempt `can_skip_via_generation()` with lock data + fingerprint + params (no hashing)
    3. Only if generation check fails → hash dependencies → proceed to lock comparison + run cache
  - Gate generation skip to only attempt when deps are Pivot-produced artifacts (have generation counters in StateDB). If any dep lacks a generation counter, fall through to hashing immediately.
  - Add tests proving that when generation skip succeeds, `hash_dependencies()` is never called (use mock/spy).

  **Must NOT do**:
  - Do not change the `can_skip_via_generation()` algorithm itself (just when it's called)
  - Do not modify the lock file format
  - Do not change the `WorkerStageInfo` contract

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful understanding of the worker execution flow and skip detection tiers; must preserve correctness invariants while reordering.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 4, 7)
  - **Blocks**: Task 2, Task 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/pivot/executor/worker.py:417-462` — `_check_skip_or_run()`: current skip detection flow that takes `dep_hashes` as input. This must be restructured to optionally skip hashing.
  - `src/pivot/executor/worker.py:972-1041` — `can_skip_via_generation()`: the generation check logic. Note lines 1002-1005 where it falls back to `lock_data["dep_generations"]` (this fallback will be removed in Task 3, but for now keep it).
  - `src/pivot/executor/worker.py:200-300` — `execute_stage()`: the top-level worker entry point where `hash_dependencies()` is called. This is where the reordering must happen.

  **API/Type References**:
  - `src/pivot/types.py:121-131` — `DeferredWrites` TypedDict (will be extended in Task 2)
  - `src/pivot/types.py:134-142` — `StageResult` TypedDict
  - `src/pivot/executor/worker.py` — `WorkerStageInfo` TypedDict

  **Test References**:
  - `tests/storage/test_state.py` — StateDB test patterns
  - `tests/fingerprint/test_fingerprint.py` — fingerprint test patterns

  **Acceptance Criteria**:

  - [x] New test: when all deps have matching generations in StateDB, `hash_dependencies()` is NOT called
  - [x] New test: when any dep lacks a generation counter, `hash_dependencies()` IS called and skip falls through to lock comparison
  - [x] New test: mixed deps (Pivot-produced + external) correctly fall through to hashing
  - [x] `uv run pytest tests/ -n auto` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Generation skip avoids dep hashing
    Tool: Bash
    Preconditions: Test environment set up with pytest
    Steps:
      1. uv run pytest tests/ -k "generation_skip" -v
      2. Assert: exit code 0
      3. Assert: output contains "PASSED" for all generation skip tests
    Expected Result: All generation skip tests pass
    Evidence: pytest output captured

  Scenario: Full test suite still passes after reordering
    Tool: Bash
    Preconditions: All changes applied
    Steps:
      1. uv run pytest tests/ -n auto
      2. Assert: exit code 0
    Expected Result: No regressions
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `fix(executor): reorder skip detection to try generation check before dep hashing`
  - Files: `src/pivot/executor/worker.py`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 2. Extend DeferredWrites with file-hash cache write-back (C0 cont.)

  **What to do**:
  - Add a new optional field `file_hash_entries` to `DeferredWrites` in `src/pivot/types.py`:
    ```python
    file_hash_entries: list[tuple[str, int, int, int, str]]  # (path, mtime_ns, size, inode, hash)
    ```
  - In `hash_dependencies()` (`src/pivot/executor/worker.py`), collect file hash results as the worker computes them. Return these in `DeferredWrites` alongside existing fields.
  - In `StateDB.apply_deferred_writes()` (`src/pivot/storage/state.py:674`), add handling for `file_hash_entries`: batch-write them in the same atomic transaction.
  - This allows the worker (readonly StateDB) to contribute hash cache entries that the coordinator persists, making future runs' hash lookups hit cache.

  **Must NOT do**:
  - Do not make workers open StateDB in write mode
  - Do not change the LMDB transaction model (single-writer via coordinator)
  - Do not modify existing `DeferredWrites` fields

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Touches the coordinator-worker contract boundary; must preserve atomicity guarantees.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Task 1)
  - **Blocks**: Task 3, Task 6
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/pivot/storage/state.py:674-743` — `apply_deferred_writes()`: the single-txn atomic write. Add file-hash entries here following the same pattern as dep_generations.
  - `src/pivot/storage/state.py:213-227` — `save()` and `save_many()`: the file-hash write pattern to replicate inside deferred writes.
  - `src/pivot/storage/cache.py:84-113` — `hash_file()`: where file hashes are computed and (when writable) saved. The worker path skips the save; instead collect entries for deferred write-back.

  **API/Type References**:
  - `src/pivot/types.py:121-131` — `DeferredWrites` TypedDict to extend
  - `src/pivot/storage/state.py:56-63` — `_make_key_file_hash()` key format
  - `src/pivot/storage/state.py:125-134` — `_pack_value()` / `_unpack_value()` binary format

  **Acceptance Criteria**:

  - [x] `DeferredWrites` has new optional `file_hash_entries` field
  - [x] `apply_deferred_writes()` persists file-hash entries in same atomic txn
  - [x] Worker collects file-hash entries during dep hashing and includes them in result
  - [x] New test: after worker run, coordinator applies deferred writes, subsequent `state_db.get()` returns cached hash
  - [x] `uv run pytest tests/ -n auto` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Deferred file-hash write-back works
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -k "deferred_write" -v
      2. Assert: exit code 0
    Expected Result: File-hash entries are persisted via deferred writes
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `feat(storage): add file-hash cache write-back via DeferredWrites`
  - Files: `src/pivot/types.py`, `src/pivot/executor/worker.py`, `src/pivot/storage/state.py`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 3. Consolidate dep_generations to StateDB-only (C1)

  **What to do**:
  - Remove `dep_generations` from `_REQUIRED_LOCK_KEYS` in `src/pivot/storage/lock.py:47`.
  - Remove `dep_generations` from `StorageLockData` and `LockData` TypedDicts in `src/pivot/types.py`.
  - Remove `dep_generations` from `_convert_to_storage_format()` and `_convert_from_storage_format()` in `src/pivot/storage/lock.py`.
  - In `is_lock_data()` validator (`src/pivot/storage/lock.py:61`), stop requiring `dep_generations` — but handle old lockfiles gracefully (ignore the field if present).
  - In `can_skip_via_generation()` (`src/pivot/executor/worker.py:972`), remove the fallback to `lock_data["dep_generations"]` (lines 1002-1005). Only use `state_db.get_dep_generations()`.
  - Update all call sites that construct `LockData(... dep_generations={})` to remove the field.
  - Update `src/pivot/explain.py` if it references `dep_generations` from lock data.

  **Must NOT do**:
  - Do not break reading of old lockfiles that have `dep_generations` (just ignore the field)
  - Do not change the StateDB dep-generation write path (it already works correctly)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Touches a cross-cutting schema change across lock/types/worker/explain; must handle backward compatibility.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 2)
  - **Blocks**: Task 6
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `src/pivot/storage/lock.py:47` — `_REQUIRED_LOCK_KEYS` frozenset to modify
  - `src/pivot/storage/lock.py:96-162` — Storage format converters to update
  - `src/pivot/executor/worker.py:1002-1005` — Lock data fallback to remove
  - `src/pivot/executor/worker.py` — All `LockData(... dep_generations={})` constructors (search for `dep_generations`)

  **API/Type References**:
  - `src/pivot/types.py` — `LockData` and `StorageLockData` TypedDicts to modify
  - `src/pivot/storage/state.py:412-454` — StateDB `get_dep_generations()` / `record_dep_generations()` (these stay unchanged)

  **Test References**:
  - `tests/storage/test_lock.py` — Lock file tests to update

  **Acceptance Criteria**:

  - [x] `dep_generations` removed from `LockData`, `StorageLockData`, `_REQUIRED_LOCK_KEYS`
  - [x] Old lockfiles with `dep_generations` field still parse correctly (field ignored)
  - [x] `can_skip_via_generation()` no longer references `lock_data["dep_generations"]`
  - [x] New lockfiles written without `dep_generations` field
  - [x] New test: lockfile without `dep_generations` is valid
  - [x] New test: old lockfile with `dep_generations` still loads without error
  - [x] `uv run pytest tests/ -n auto` → PASS
  - [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` → clean

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Lock schema migration is backward-compatible
    Tool: Bash
    Steps:
      1. uv run pytest tests/storage/test_lock.py -v
      2. Assert: exit code 0
    Expected Result: Old and new lockfile formats both work
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `fix(storage): consolidate dep_generations to StateDB-only, remove from lockfiles`
  - Files: `src/pivot/types.py`, `src/pivot/storage/lock.py`, `src/pivot/executor/worker.py`, `src/pivot/explain.py`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 4. Complete manifest cache flush boundaries (C2 / #363)

  **What to do**:
  - Add `fingerprint.flush_manifest_cache()` calls at these lifecycle points:
    1. `src/pivot/discovery.py` — after pipeline discovery completes (alongside existing `flush_ast_hash_cache()` calls at lines 130, 388)
    2. `src/pivot/pipeline/yaml.py` — after YAML pipeline loading (alongside existing `flush_ast_hash_cache()` call at line 215)
    3. `src/pivot/engine/engine.py:_handle_code_or_config_changed()` (line 1175) — after registry reload in watch mode, before next run cycle
  - Verify that `flush_manifest_cache()` is safe to call multiple times (it is — it's a drain-and-write pattern).
  - Add a test that verifies manifest cache entries are persisted to StateDB after discovery (not just at atexit).

  **Must NOT do**:
  - Do not change the manifest cache data format
  - Do not change the atexit flush (keep it as a safety net)
  - Do not add flush calls in hot loops (only at lifecycle checkpoints)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small, targeted change — adding function calls at 3 known locations.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 7)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/pivot/discovery.py:130` and `src/pivot/discovery.py:388` — Existing `flush_ast_hash_cache()` calls. Add `flush_manifest_cache()` alongside each.
  - `src/pivot/pipeline/yaml.py:215` — Existing `flush_ast_hash_cache()` call. Add `flush_manifest_cache()` alongside.
  - `src/pivot/engine/engine.py:1175-1200` — `_handle_code_or_config_changed()` watch reload handler. Add manifest flush after registry reload.

  **API/Type References**:
  - `src/pivot/fingerprint.py:274-292` — `flush_manifest_cache()` function

  **Acceptance Criteria**:

  - [x] `flush_manifest_cache()` called in discovery.py (2 locations), yaml.py (1 location), engine.py watch reload (1 location)
  - [x] New test: after discovery, `sm:` entries exist in StateDB (not just pending in memory)
  - [x] `uv run pytest tests/ -n auto` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Manifest cache persists after discovery
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -k "manifest_cache" -v
      2. Assert: exit code 0
    Expected Result: Manifest cache flush tests pass
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `perf(fingerprint): add manifest cache flush boundaries for watch mode (#363)`
  - Files: `src/pivot/discovery.py`, `src/pivot/pipeline/yaml.py`, `src/pivot/engine/engine.py`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 5. Selective re-fingerprinting via reverse index (#358)

  **What to do**:
  - In the watch-mode reload path (`src/pivot/engine/engine.py:_handle_code_or_config_changed()`), after receiving changed file paths from the watcher:
    1. Load cached manifests from StateDB for all registered stages
    2. For each cached manifest, check if any of its recorded source files (`"s"` key) match the changed paths
    3. Only invalidate + re-fingerprint stages whose source files overlap with changes
    4. Stages with no overlap keep their cached manifest (skip fingerprinting entirely)
  - Add a helper function in `src/pivot/fingerprint.py` like `invalidate_manifests_for_paths(changed_paths: list[str])` that scans `sm:` entries and deletes those referencing changed files.
  - If no cached manifests exist (cold start), fall back to fingerprinting all stages (existing behavior).

  **Must NOT do**:
  - Do not change the manifest cache format
  - Do not attempt parallel fingerprinting (#364)
  - Do not change one-shot (non-watch) behavior

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding of the manifest cache data structure and the watch-mode reload flow.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `src/pivot/fingerprint.py:319-352` — `get_stage_fingerprint_cached()`: shows how manifest is stored with source map `"s": {rel_path: [mtime_ns, size, inode]}`. This is the reverse index.
  - `src/pivot/fingerprint.py:146-148` — `_make_manifest_cache_key()`: key format for `sm:` entries
  - `src/pivot/engine/engine.py:1175-1200` — Watch reload handler that currently re-fingerprints all stages

  **API/Type References**:
  - `src/pivot/engine/types.py:79-81` — `CodeOrConfigChanged` event with `paths` field (the changed file paths from watcher)
  - `src/pivot/storage/state.py:329-347` — `get_raw()` / `put_raw()` for reading/writing `sm:` entries

  **Acceptance Criteria**:

  - [x] New function `invalidate_manifests_for_paths()` or similar in `src/pivot/fingerprint.py`
  - [x] Watch reload only re-fingerprints stages whose source files changed
  - [x] Stages unaffected by file changes retain cached manifests
  - [x] Cold start (no cached manifests) falls back to fingerprinting all stages
  - [x] New test: with 5 stages, changing 1 source file only invalidates affected stage(s)
  - [x] `uv run pytest tests/ -n auto` → PASS

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Selective re-fingerprinting works
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -k "selective_fingerprint" -v
      2. Assert: exit code 0
    Expected Result: Only affected stages re-fingerprinted
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `perf(fingerprint): selective re-fingerprinting in watch mode via reverse index (#358)`
  - Files: `src/pivot/fingerprint.py`, `src/pivot/engine/engine.py`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 6. Add skip-invariant documentation + integration tests

  **What to do**:
  - Add a `docs/solutions/` document explaining the skip detection invariants:
    - What is keyed by logical path vs physical identity (resolve vs normpath)
    - What data must exist for generation skipping to be sound
    - The tiering: generation → lock compare → run cache (now actually tiered)
    - The Pivot-produced-artifact soundness boundary
  - Add integration tests that exercise the full skip detection pipeline end-to-end:
    - Stage with all Pivot deps → generation skip works (no hashing)
    - Stage with external dep → falls through to hash-based check
    - StateDB cleared → graceful degradation to lock comparison
    - Lock file missing → full run
  - Update `src/pivot/storage/AGENTS.md` to reflect the changes (dep_generations removed from lockfiles, file-hash write-back added).

  **Must NOT do**:
  - Do not change any runtime behavior (documentation + tests only)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Mix of documentation writing and integration test authoring; needs thorough understanding of all prior tasks.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (final, after Tasks 2, 3, 5)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 2, 3, 5

  **References**:

  **Documentation References**:
  - `src/pivot/storage/AGENTS.md` — Storage guidelines to update
  - `docs/solutions/` — Existing solution docs for pattern reference
  - This plan file (`p0-cache-fingerprint-correctness.md`) — Context and design decisions sections

  **Test References**:
  - `tests/storage/test_state.py` — StateDB test patterns
  - `tests/storage/test_lock.py` — Lock test patterns

  **Acceptance Criteria**:

  - [x] New doc in `docs/solutions/` covering skip detection invariants
  - [x] `src/pivot/storage/AGENTS.md` updated (dep_generations removed, file-hash write-back documented)
  - [x] Integration tests covering all 4 skip scenarios (generation, external dep, cleared StateDB, missing lock)
  - [x] `uv run pytest tests/ -n auto` → PASS

  **Commit**: YES
  - Message: `docs(storage): document skip detection invariants and add integration tests`
  - Files: `docs/solutions/`, `src/pivot/storage/AGENTS.md`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

- [x] 7. Safe fingerprinting defaults + unsafe mode escape hatch

  **What to do**:
  - In `_process_closure_values()` (`src/pivot/fingerprint.py:451`), add validation when processing closure variables:
    - For each value that is a user-class instance (currently handled by `_process_instance_dependency()`):
      - Check if it's a frozen dataclass (`dataclasses.is_dataclass(value) and value.__dataclass_params__.frozen`)
      - Check if it's a frozen Pydantic model (`hasattr(type(value), 'model_config') and type(value).model_config.get('frozen', False)`)
      - If frozen → allow (existing behavior: fingerprint by class)
      - If NOT frozen → raise `StageDefinitionError` with message naming the variable, its type, and suggesting `StageParams` or `Dep(...)`
    - For mutable collections (dict, list, set) captured as closure variables that contain non-callable, non-primitive content:
      - Raise `StageDefinitionError` with same guidance
  - Add project config `core.unsafe_fingerprinting` (boolean, default false) in `src/pivot/config/models.py`:
    - When true, downgrade the above errors to warnings (log.warning)
  - Add env var `PIVOT_UNSAFE_FINGERPRINTING` that overrides config (either one enables unsafe mode)
  - Read the config/env in `get_stage_fingerprint()` or `_process_closure_values()` and branch accordingly.
  - Error message format:
    ```
    Stage '{stage_name}': closure captures mutable variable '{var_name}' (type: {type_name}).
    Pivot cannot track changes to mutable runtime state, which may cause silent wrong outputs.
    Fix: pass this data via StageParams or declare it as a Dep(...) input.
    To suppress: set core.unsafe_fingerprinting=true or PIVOT_UNSAFE_FINGERPRINTING=1
    ```

  **Must NOT do**:
  - Do not recurse into objects to prove immutability (static checks only)
  - Do not change fingerprinting behavior for primitives, tuples, frozensets, or callables
  - Do not add per-stage or per-variable granularity (single boolean only)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Needs careful logic for detecting mutability + integration with config system + error message design.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 4)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/pivot/fingerprint.py:451-489` — `_process_closure_values()`: where closure variables are categorized and processed. Add validation here.
  - `src/pivot/fingerprint.py:519-533` — `_process_instance_dependency()`: current instance handling (class-only fingerprint)
  - `src/pivot/fingerprint.py:487-488` — `_is_user_class_instance()`: detection function
  - `src/pivot/registry.py:309-316` — Existing lambda warning pattern (follow this style for error messages)

  **API/Type References**:
  - `src/pivot/config/models.py:59-73` — `CoreConfig` model (add `unsafe_fingerprinting: bool = False`)
  - `src/pivot/config/io.py` — Config reader functions (add `get_unsafe_fingerprinting()`)
  - `src/pivot/exceptions.py` — `StageDefinitionError` for the error to raise

  **Test References**:
  - `tests/fingerprint/test_fingerprint.py` — Existing fingerprint tests

  **Acceptance Criteria**:

  - [x] Stage closing over mutable dict/list/set raises `StageDefinitionError` by default
  - [x] Stage closing over non-frozen user-class instance raises `StageDefinitionError` by default
  - [x] Stage closing over frozen dataclass → allowed (no error)
  - [x] Stage closing over frozen Pydantic model → allowed (no error)
  - [x] Error message names the variable, its type, and suggests `StageParams`/`Dep(...)`
  - [x] `PIVOT_UNSAFE_FINGERPRINTING=1` downgrades error to warning
  - [x] `core.unsafe_fingerprinting=true` in config downgrades error to warning
  - [x] Primitives, tuples, frozensets, callables → no change in behavior
  - [x] `uv run pytest tests/ -n auto` → PASS
  - [x] `uv run ruff format . && uv run ruff check . && uv run basedpyright` → clean

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Mutable closure capture raises error
    Tool: Bash
    Steps:
      1. uv run pytest tests/ -k "safe_fingerprint" -v
      2. Assert: exit code 0
      3. Assert: output contains "PASSED" for mutable capture test
    Expected Result: Error raised for mutable captures, allowed for frozen instances
    Evidence: pytest output captured

  Scenario: Unsafe mode suppresses error
    Tool: Bash
    Steps:
      1. PIVOT_UNSAFE_FINGERPRINTING=1 uv run pytest tests/ -k "unsafe_fingerprint" -v
      2. Assert: exit code 0
    Expected Result: Warning instead of error when env var set
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `feat(fingerprint): safe-by-default closure validation with unsafe mode escape hatch`
  - Files: `src/pivot/fingerprint.py`, `src/pivot/config/models.py`, `src/pivot/config/io.py`, `tests/`
  - Pre-commit: `uv run pytest tests/ -n auto`

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `fix(executor): reorder skip detection to try generation check before dep hashing` | worker.py | `uv run pytest tests/ -n auto` |
| 2 | `feat(storage): add file-hash cache write-back via DeferredWrites` | types.py, worker.py, state.py | `uv run pytest tests/ -n auto` |
| 3 | `fix(storage): consolidate dep_generations to StateDB-only, remove from lockfiles` | types.py, lock.py, worker.py | `uv run pytest tests/ -n auto` |
| 4 | `perf(fingerprint): add manifest cache flush boundaries for watch mode (#363)` | discovery.py, yaml.py, engine.py | `uv run pytest tests/ -n auto` |
| 5 | `perf(fingerprint): selective re-fingerprinting in watch mode (#358)` | fingerprint.py, engine.py | `uv run pytest tests/ -n auto` |
| 6 | `docs(storage): document skip detection invariants and add integration tests` | docs/, AGENTS.md, tests/ | `uv run pytest tests/ -n auto` |
| 7 | `feat(fingerprint): safe-by-default closure validation with unsafe mode escape hatch` | fingerprint.py, config/ | `uv run pytest tests/ -n auto` |

---

## Success Criteria

### Verification Commands
```bash
uv run pytest tests/ -n auto                                        # All tests pass
uv run ruff format . && uv run ruff check . && uv run basedpyright  # Quality clean
```

### Final Checklist
- [x] Generation skip runs before dep hashing (C0 fixed)
- [x] File-hash cache entries written back via DeferredWrites (C0 perf)
- [x] `dep_generations` only in StateDB; lockfiles no longer store it (C1 fixed)
- [x] Old lockfiles with `dep_generations` still load correctly (backward compat)
- [x] Manifest cache flushed at discovery/registration/watch reload (C2/#363 complete)
- [x] Watch mode only re-fingerprints affected stages (#358 complete)
- [x] Mutable closure captures error by default (safe fingerprinting)
- [x] Unsafe mode escape hatch works (config + env var)
- [x] Skip detection invariants documented in `docs/solutions/`
- [x] All tests pass, quality checks clean
