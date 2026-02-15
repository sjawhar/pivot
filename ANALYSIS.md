# Skip Detection & Run Cache Lock Update - Failure Analysis

## Test Failures Summary

4 tests failing across 2 test files:
1. `test_generation_skip_with_pivot_produced_deps` - Wrong skip reason
2. `test_deferred_file_hash_writeback` - Missing file_hash_entries in deferred writes
3. `test_explain_shows_cached_after_run_cache_skip` - Explain shows wrong state after run cache skip
4. `test_run_cache_skip_does_not_increment_output_generations` - Output generations not being tracked

## Architecture Overview

### Three-Tier Skip Detection Algorithm

```
Tier 1: O(1) Generation Check (can_skip_via_generation)
  ↓ (if fails)
Tier 2: O(n) Lock File Comparison (_check_skip_or_run)
  ↓ (if fails)
Tier 3: Run Cache Lookup (_try_skip_via_run_cache)
```

### Data Flow

```
Worker Process (execute_stage)
  ├─ Skip Detection
  │   ├─ Tier 1: can_skip_via_generation() [lines 287-323]
  │   ├─ Tier 2: _check_skip_or_run() [lines 343-350]
  │   └─ Tier 3: _try_skip_via_run_cache() [lines 383-434]
  │
  ├─ Stage Execution (if no skip)
  │
  └─ Lock & State Updates
      ├─ Build lock_data [lines 501-506]
      ├─ _commit_lock_and_build_deferred() [lines 510-518]
      │   ├─ production_lock.write(lock_data)
      │   └─ _build_deferred_writes() → DeferredWrites
      │       ├─ increment_outputs flag
      │       ├─ dep_generations
      │       ├─ run_cache_entry
      │       └─ file_hash_entries
      │
      └─ Return StageResult with deferred_writes

Coordinator (engine.py)
  └─ state_db.apply_deferred_writes(stage_name, output_paths, deferred)
      ├─ Increment output generations (if increment_outputs=True)
      ├─ Record dep_generations
      ├─ Write run_cache_entry
      └─ Write file_hash_entries
```

## Failure Point Analysis

### Failure 1: test_generation_skip_with_pivot_produced_deps

**Expected:** `reason == "unchanged (generation)"`
**Actual:** `reason == "unchanged"`

**Location:** `worker.py:343-380` (_check_skip_or_run path)

**Root Cause:** The test expects Tier 1 (generation) skip but gets Tier 2 (lock comparison) skip.

**Code Path:**
```python
# Line 287-323: Tier 1 generation check
if can_skip_via_generation(...):
    return _make_result(StageStatus.CACHED, "unchanged (generation)", ...)

# Line 343-380: Tier 2 lock comparison
skip_reason, run_reason, input_hash = _check_skip_or_run(...)
if skip_reason is not None:
    return _make_result(StageStatus.CACHED, skip_reason, ...)  # skip_reason = "unchanged"
```

**Issue:** `can_skip_via_generation()` is returning False when it should return True for Pivot-produced deps.

**Investigation Points:**
- `can_skip_via_generation()` at line 1385-1434
- `state_db.get_dep_generations(stage_name)` - are dep_generations being recorded?
- `compute_dep_generation_map()` at line 1437-1455 - is it computing correct generations?
- `apply_deferred_writes()` at state.py:784-869 - is it applying dep_generations?

### Failure 2: test_deferred_file_hash_writeback

**Expected:** `"file_hash_entries" in deferred`
**Actual:** `deferred = {'increment_outputs': True, 'run_cache_input_hash': ..., 'run_cache_entry': ...}`

**Location:** `worker.py:1485-1529` (_build_deferred_writes)

**Root Cause:** `file_hash_entries` not being included in deferred writes.

**Code Path:**
```python
# Line 325-327: hash_dependencies collects file_hash_entries
dep_hashes, missing, unreadable, file_hash_entries = hash_dependencies(...)

# Line 510-518: Pass file_hash_entries to _commit_lock_and_build_deferred
deferred = _commit_lock_and_build_deferred(
    ...,
    file_hash_entries=file_hash_entries,  # ← Should be passed here
)

# Line 1526-1527: Conditionally add to result
if file_hash_entries:
    result["file_hash_entries"] = file_hash_entries
```

**Issue:** `file_hash_entries` is None or empty when it should contain hash entries from dependency hashing.

**Investigation Points:**
- `hash_dependencies()` at line 1325+ - is it returning file_hash_entries?
- Are file_hash_entries being collected during the first run?
- Is the test checking the right result (first run vs subsequent runs)?

### Failure 3: test_explain_shows_cached_after_run_cache_skip

**Expected:** `will_run=False` after run cache skip
**Actual:** `will_run=True, reason='Input dependencies changed'`

**Location:** `explain.py:85-240` (get_stage_explanation)

**Root Cause:** Lock file not updated after run cache skip, causing explain to see stale dep_hashes.

**Code Path:**
```python
# worker.py:403-434: Run cache skip path
if run_cache_skip is not None:
    if no_commit:
        return _make_result(StageStatus.CACHED, "unchanged (run cache)", ...)
    
    # Line 411-416: Build NEW lock_data with CURRENT dep_hashes
    new_lock_data = LockData(
        code_manifest=current_fingerprint,
        params=current_params,
        dep_hashes=dict(sorted(dep_hashes.items())),  # ← Current state
        output_hashes=dict(sorted(run_cache_skip["output_hashes"].items())),
    )
    
    # Line 417-426: Commit lock and build deferred (with increment_outputs=False)
    deferred = _commit_lock_and_build_deferred(
        ...,
        increment_outputs=False,  # ← Critical: don't bump generations
    )
```

**Issue:** The lock file IS being updated (line 411-426), but explain is still seeing dep_changes.

**Detailed Error:**
```python
'dep_changes': [
    {'identity': '/tmp/.../input.txt', 'old_hash': None, 'new_hash': 'bfec...', 'change_type': 'added'},
    {'identity': 'input.txt', 'old_hash': 'bfec...', 'new_hash': None, 'change_type': 'removed'}
]
```

This suggests a **path normalization issue** - the same file appears twice with different paths (absolute vs relative).

**Investigation Points:**
- Path normalization in `hash_dependencies()` vs lock file storage
- `_canonicalize_out()` at line 555-557
- Lock file dep_hashes key format vs explain's dep_hashes key format
- `explain.py:186-190` - how are dep_hashes being computed?

### Failure 4: test_run_cache_skip_does_not_increment_output_generations

**Expected:** `gen_after_run1 == 1`
**Actual:** `gen_after_run1 == None`

**Location:** `state.py:846-851` (apply_deferred_writes output generation increment)

**Root Cause:** Output generations not being incremented after first run.

**Code Path:**
```python
# worker.py:510-518: After stage execution
deferred = _commit_lock_and_build_deferred(
    ...,
    file_hash_entries=file_hash_entries,
    # increment_outputs defaults to True (line 1467)
)

# worker.py:1485-1529: _build_deferred_writes
if increment_outputs:
    result["increment_outputs"] = True  # ← Flag set in deferred writes

# state.py:809-851: apply_deferred_writes
increment_outputs = "increment_outputs" in deferred and deferred["increment_outputs"]
if increment_outputs:
    for identity in output_paths:
        key = _make_key_output_generation(identity)
        value = txn.get(key)
        current = struct.unpack(">Q", value)[0] if value else 0
        txn.put(key, struct.pack(">Q", current + 1))
```

**Issue:** Either:
1. `apply_deferred_writes()` is not being called by the test
2. `output_paths` parameter doesn't match the identity format used in `_make_key_output_generation()`
3. The test is querying with wrong path format

**Test Code:**
```python
# Line 130: Output paths computed as data/test/{stage_name}.txt
out_paths = [f"data/test/{stage_name}.txt"]

# Line 132-133: Apply deferred writes
with state.StateDB(state_dir / "state.db") as state_db:
    state_db.apply_deferred_writes(stage_name, out_paths, deferred)

# Line 371: Query generation
with state.StateDB(state_db_path) as db:
    gen_after_run1 = db.get_generation(str(tmp_path / "output.txt"))
```

**Path Mismatch:**
- `apply_deferred_writes` receives: `["data/test/test_stage.txt"]`
- Test queries: `str(tmp_path / "output.txt")` → absolute path

**Investigation Points:**
- `_make_key_output_generation()` at state.py:85-89 - what format does it expect?
- `get_generation()` - how does it construct the key?
- Output path format in lock_data vs StateDB keys
- `_get_normalized_out_paths()` at worker.py:592-598

## Common Themes

### 1. Path Normalization Issues
Multiple failures involve path format mismatches:
- Absolute vs relative paths
- Normalized vs raw paths
- Identity keys vs file paths

### 2. Deferred Writes Flow
The coordinator pattern (worker returns deferred writes, coordinator applies them) has gaps:
- File hash entries not being collected/passed
- Output generation keys not matching query keys
- Dep generation recording unclear

### 3. Skip Detection State Management
The three-tier algorithm depends on StateDB state being correctly maintained:
- Dep generations must be recorded for Tier 1
- Lock files must be updated for Tier 2
- Run cache must be populated for Tier 3

## Next Steps for Debugging

### Priority 1: Output Generation Tracking
1. Trace `apply_deferred_writes()` call in test
2. Verify `output_paths` parameter format
3. Check `_make_key_output_generation()` key construction
4. Verify `get_generation()` query format

### Priority 2: Path Normalization
1. Map path formats through the system:
   - Worker dep_hashes keys
   - Lock file dep_hashes keys
   - Explain dep_hashes keys
   - StateDB generation keys
2. Identify where normalization breaks down

### Priority 3: Deferred Writes Collection
1. Trace `hash_dependencies()` return value
2. Verify `file_hash_entries` is populated
3. Check if it's being passed through to `_build_deferred_writes()`

### Priority 4: Generation Skip Logic
1. Verify `dep_generations` are being recorded in StateDB
2. Check `can_skip_via_generation()` logic for Pivot-produced deps
3. Trace why Tier 1 fails and falls through to Tier 2
