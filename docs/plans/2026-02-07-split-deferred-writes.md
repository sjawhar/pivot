# Split Deferred StateDB Writes for Run-Cache Skip Safety


**Goal:** Ensure run-cache SKIPPED results record dep_generations and run_cache metadata in StateDB without incorrectly incrementing output generations.

**Architecture:** Add an `increment_outputs` flag to `DeferredWrites`. Worker sets it `True` for RAN, omits it for SKIPPED. `StateDB.apply_deferred_writes()` only bumps output generations when the flag is present and True. Engine applies deferred writes for both RAN and SKIPPED results.

**Tech Stack:** Python 3.13+, TypedDict, LMDB (via `state.py`)

---

## Background

`DeferredWrites` is a TypedDict returned by workers for the coordinator to apply atomically to StateDB. It currently contains `dep_generations`, `run_cache_input_hash`, and `run_cache_entry`. Output generation increments are always applied for every output path passed to `apply_deferred_writes()`.

**Bug:** The engine only applies deferred writes for `StageStatus.RAN` (engine.py:608). Run-cache skips return `StageStatus.SKIPPED` with `deferred_writes`, so those writes are silently dropped. This means dep_generations are never recorded after a run-cache skip, causing generation-based O(1) skip detection to fail on the next run.

If we simply apply deferred writes for SKIPPED too, output generations would be incorrectly incremented (the outputs were restored from cache, not re-produced). We need a way to apply dep_generations and run_cache data WITHOUT bumping output generations.

---

## Task 1: Add `increment_outputs` flag to `DeferredWrites`

**Files:**
- Modify: `src/pivot/types.py:113-122`

**Step 1: Add the field**

In `src/pivot/types.py`, add `increment_outputs` to the `DeferredWrites` TypedDict:

```python
class DeferredWrites(TypedDict, total=False):
    """Deferred StateDB writes from worker for coordinator to apply.

    Uses total=False so keys are only present when there's data to write.
    Stage name and output paths are passed separately by coordinator.
    """

    dep_generations: dict[str, int]  # {dep_path: generation}
    run_cache_input_hash: str
    run_cache_entry: RunCacheEntry
    increment_outputs: bool  # True → increment output generations; absent → skip
```

**Step 2: Verify no type errors introduced**

Run: `cd /home/sami/pivot/roadmap-379 && uv run basedpyright src/pivot/types.py`
Expected: PASS (field is optional via `total=False`)

---

## Task 2: Guard output generation increments in `StateDB.apply_deferred_writes`

**Files:**
- Modify: `src/pivot/storage/state.py:674-740`

**Step 1: Write failing tests**

In `tests/storage/test_state.py`, add two tests after the existing `test_apply_deferred_writes_output_generations`:

```python
def test_apply_deferred_writes_skips_output_increment_when_flag_absent(
    tmp_path: pathlib.Path,
) -> None:
    """Output generations should NOT be incremented when increment_outputs is absent."""
    db_path = tmp_path / "state.db"
    output1 = tmp_path / "output1.csv"
    deferred: DeferredWrites = {"dep_generations": {"/dep.csv": 5}}

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [str(output1)], deferred)
        assert db.get_generation(output1) is None
        assert db.get_dep_generations("stage") == {"/dep.csv": 5}


def test_apply_deferred_writes_increments_output_when_flag_true(
    tmp_path: pathlib.Path,
) -> None:
    """Output generations should be incremented when increment_outputs is True."""
    db_path = tmp_path / "state.db"
    output1 = tmp_path / "output1.csv"
    deferred: DeferredWrites = {"increment_outputs": True}

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [str(output1)], deferred)
        assert db.get_generation(output1) == 1

        # Second apply increments to 2
        db.apply_deferred_writes("stage", [str(output1)], deferred)
        assert db.get_generation(output1) == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sami/pivot/roadmap-379 && uv run pytest tests/storage/test_state.py::test_apply_deferred_writes_skips_output_increment_when_flag_absent tests/storage/test_state.py::test_apply_deferred_writes_increments_output_when_flag_true -v`
Expected: First test FAILS (output generation is 1, not None), second test PASSES

**Step 3: Update `apply_deferred_writes` to guard on flag**

In `src/pivot/storage/state.py`, modify the `apply_deferred_writes` method. Change the output generations block (lines 723-729) from unconditional to conditional:

Before:
```python
                # Output generations (increment)
                for path_str in output_paths:
                    path = pathlib.Path(path_str)
                    key = _make_key_output_generation(path)
                    value = txn.get(key)
                    current = struct.unpack(">Q", value)[0] if value else 0
                    txn.put(key, struct.pack(">Q", current + 1))
```

After:
```python
                # Output generations (only increment when explicitly requested)
                if deferred.get("increment_outputs", False):
                    for path_str in output_paths:
                        path = pathlib.Path(path_str)
                        key = _make_key_output_generation(path)
                        value = txn.get(key)
                        current = struct.unpack(">Q", value)[0] if value else 0
                        txn.put(key, struct.pack(">Q", current + 1))
```

Also move the output_paths key length validation inside the same guard:

Before (lines 698-703):
```python
        for path_str in output_paths:
            key = _make_key_output_generation(pathlib.Path(path_str))
            if len(key) > _MAX_KEY_SIZE:
                raise PathTooLongError(
                    f"Output path too long ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path_str}"
                )
```

After:
```python
        if deferred.get("increment_outputs", False):
            for path_str in output_paths:
                key = _make_key_output_generation(pathlib.Path(path_str))
                if len(key) > _MAX_KEY_SIZE:
                    raise PathTooLongError(
                        f"Output path too long ({len(key)} bytes, max {_MAX_KEY_SIZE}): {path_str}"
                    )
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sami/pivot/roadmap-379 && uv run pytest tests/storage/test_state.py -k "deferred_writes" -v`
Expected: ALL deferred_writes tests PASS

**Step 5: Update existing test that relied on unconditional increment**

The existing `test_apply_deferred_writes_output_generations` passes an empty `DeferredWrites` dict and expects output generations to be incremented. This needs to set `increment_outputs=True`:

```python
def test_apply_deferred_writes_output_generations(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes increments output generations."""
    db_path = tmp_path / "state.db"
    output1 = tmp_path / "output1.csv"
    output2 = tmp_path / "output2.csv"
    deferred: DeferredWrites = {"increment_outputs": True}

    with state.StateDB(db_path) as db:
        # First apply - outputs should be at generation 1
        db.apply_deferred_writes("stage", [str(output1), str(output2)], deferred)
        assert db.get_generation(output1) == 1
        assert db.get_generation(output2) == 1

        # Second apply - outputs should increment to 2
        db.apply_deferred_writes("stage", [str(output1), str(output2)], deferred)
        assert db.get_generation(output1) == 2
        assert db.get_generation(output2) == 2
```

Also update `test_apply_deferred_writes_all_fields` to include the flag:

```python
def test_apply_deferred_writes_all_fields(tmp_path: pathlib.Path) -> None:
    """apply_deferred_writes handles all fields atomically."""
    db_path = tmp_path / "state.db"
    output_path = tmp_path / "output.csv"
    run_cache_entry = run_history.RunCacheEntry(
        run_id="run_456",
        output_hashes=[run_history.OutputHashEntry(path=str(output_path), hash="def456")],
    )
    deferred: DeferredWrites = {
        "dep_generations": {"/dep.csv": 10},
        "run_cache_input_hash": "input_abc",
        "run_cache_entry": run_cache_entry,
        "increment_outputs": True,
    }

    with state.StateDB(db_path) as db:
        db.apply_deferred_writes("stage", [str(output_path)], deferred)

        # Verify all writes applied
        assert db.get_dep_generations("stage") == {"/dep.csv": 10}
        assert db.get_generation(output_path) == 1
        result = db.lookup_run_cache("stage", "input_abc")
        assert result is not None
        assert result["run_id"] == "run_456"
```

**Step 6: Run full deferred_writes test suite**

Run: `cd /home/sami/pivot/roadmap-379 && uv run pytest tests/storage/test_state.py -k "deferred_writes" -v`
Expected: ALL PASS

---

## Task 3: Set `increment_outputs` flag in worker

**Files:**
- Modify: `src/pivot/executor/worker.py:999-1052`

**Step 1: Add `increment_outputs` parameter to `_commit_lock_and_build_deferred`**

```python
def _commit_lock_and_build_deferred(
    stage_info: WorkerStageInfo,
    lock_data: LockData,
    input_hash: str,
    output_hashes: dict[str, OutputHash],
    pending_lock: lock.StageLock,
    production_lock: lock.StageLock,
    state_db: state.StateDB,
    no_commit: bool,
    *,
    increment_outputs: bool = True,
) -> DeferredWrites | None:
    """Commit lock file and build deferred writes.

    For no_commit: computes dep_generations, writes to pending_lock, returns None.
    For commit: writes to production_lock, returns DeferredWrites for StateDB.
    """
    if no_commit:
        dep_gens = compute_dep_generation_map(stage_info["deps"], state_db)
        lock_data["dep_generations"] = dep_gens
        pending_lock.write(lock_data)
        return None
    production_lock.write(lock_data)
    return _build_deferred_writes(
        stage_info, input_hash, output_hashes, state_db, increment_outputs=increment_outputs
    )
```

**Step 2: Add `increment_outputs` parameter to `_build_deferred_writes`**

```python
def _build_deferred_writes(
    stage_info: WorkerStageInfo,
    input_hash: str,
    output_hashes: dict[str, OutputHash],
    state_db: state.StateDB,
    *,
    increment_outputs: bool = True,
) -> DeferredWrites:
    """Build deferred writes for coordinator to apply."""
    result: DeferredWrites = {}

    if increment_outputs:
        result["increment_outputs"] = True

    # Dependency generations (read current values)
    gen_record = compute_dep_generation_map(stage_info["deps"], state_db)
    if gen_record:
        result["dep_generations"] = gen_record

    # Run cache entry — only cached outputs belong in run cache
    cached_paths = {cast("str", out.path) for out in stage_info["outs"] if out.cache}
    output_entries = [
        entry
        for path, oh in output_hashes.items()
        if path in cached_paths
        and (entry := run_history.output_hash_to_entry(path, oh)) is not None
    ]
    if output_entries:
        result["run_cache_input_hash"] = input_hash
        result["run_cache_entry"] = run_history.RunCacheEntry(
            run_id=stage_info["run_id"],
            output_hashes=output_entries,
        )

    return result
```

**Step 3: Pass `increment_outputs=False` from run-cache skip call site**

In `execute_stage()`, the run-cache skip path (line ~293) calls `_commit_lock_and_build_deferred`. Add the keyword arg:

```python
                            deferred = _commit_lock_and_build_deferred(
                                stage_info,
                                new_lock_data,
                                input_hash,
                                run_cache_skip["output_hashes"],
                                pending_lock,
                                production_lock,
                                state_db,
                                no_commit,
                                increment_outputs=False,
                            )
```

The RAN path (line ~356) keeps the default `increment_outputs=True` (no change needed).

**Step 4: Type-check**

Run: `cd /home/sami/pivot/roadmap-379 && uv run basedpyright src/pivot/executor/worker.py`
Expected: PASS

---

## Task 4: Apply deferred writes for SKIPPED results in engine

**Files:**
- Modify: `src/pivot/engine/engine.py:607-613`

**Step 1: Change the condition to include SKIPPED**

Before:
```python
                                # Apply deferred writes for successful stages
                                if result["status"] == StageStatus.RAN and not no_commit:
                                    stage_info = self._get_stage(stage_name)
                                    output_paths = [str(out.path) for out in stage_info["outs"]]
                                    executor_core.apply_deferred_writes(
                                        stage_name, output_paths, result, state_db
                                    )
```

After:
```python
                                # Apply deferred writes for RAN and SKIPPED stages
                                if result["status"] in (StageStatus.RAN, StageStatus.SKIPPED) and not no_commit:
                                    stage_info = self._get_stage(stage_name)
                                    output_paths = [str(out.path) for out in stage_info["outs"]]
                                    executor_core.apply_deferred_writes(
                                        stage_name, output_paths, result, state_db
                                    )
```

**Step 2: Type-check**

Run: `cd /home/sami/pivot/roadmap-379 && uv run basedpyright src/pivot/engine/engine.py`
Expected: PASS

---

## Task 5: Integration test — run-cache skip records dep_generations

**Files:**
- Modify: `tests/test_run_cache_lock_update.py`

**Step 1: Add integration test**

Add this test to `tests/test_run_cache_lock_update.py`:

```python
def test_run_cache_skip_records_dep_generations(
    worker_env: pathlib.Path, output_queue: mp.Queue[OutputMessage], tmp_path: pathlib.Path
) -> None:
    """Run cache skip should record dep_generations in StateDB via deferred writes.

    After a run-cache skip, generation-based skip detection should work on the
    next run (verifies deferred writes are applied for SKIPPED results).
    """
    state_db_path = tmp_path / ".pivot" / "state.db"
    input_file = tmp_path / "input.txt"
    input_file.write_text("state_A")

    stage_info = _make_stage_info(tmp_path, func=lambda data: f"processed: {data}")

    # Step 1: First run — produces outputs and builds initial state
    result1 = executor.execute_stage("test_stage", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info, result1, state_db_path)

    # Step 2: Run with different input (state B)
    input_file.write_text("state_B")
    stage_info_b = _make_stage_info(tmp_path, func=lambda data: f"processed: {data}")
    result2 = executor.execute_stage("test_stage", stage_info_b, worker_env, output_queue)
    assert result2["status"] == "ran"
    _apply_deferred_writes("test_stage", stage_info_b, result2, state_db_path)

    # Step 3: Revert to state A — should trigger run-cache skip
    input_file.write_text("state_A")
    stage_info_a = _make_stage_info(tmp_path, func=lambda data: f"processed: {data}")
    result3 = executor.execute_stage("test_stage", stage_info_a, worker_env, output_queue)
    assert result3["status"] == "skipped"
    assert "run cache" in result3["reason"]

    # CRITICAL: Deferred writes should contain dep_generations but NOT increment_outputs
    assert "deferred_writes" in result3
    deferred = result3["deferred_writes"]
    assert "dep_generations" in deferred
    assert "increment_outputs" not in deferred or not deferred.get("increment_outputs", False)
```

**Step 2: Run test to verify**

Run: `cd /home/sami/pivot/roadmap-379 && uv run pytest tests/test_run_cache_lock_update.py::test_run_cache_skip_records_dep_generations -v`
Expected: PASS

---

## Task 6: Run full test suite and quality checks

**Step 1: Run all tests**

Run: `cd /home/sami/pivot/roadmap-379 && uv run pytest tests/ -n auto`
Expected: ALL PASS

**Step 2: Run quality checks**

Run: `cd /home/sami/pivot/roadmap-379 && uv run ruff format . && uv run ruff check . && uv run basedpyright`
Expected: ALL PASS

---

## Summary of changes

| File | Change | Lines |
|------|--------|-------|
| `src/pivot/types.py` | Add `increment_outputs: bool` field to `DeferredWrites` | ~1 line |
| `src/pivot/storage/state.py` | Guard output generation increment on `increment_outputs` flag | ~6 lines changed |
| `src/pivot/executor/worker.py` | Add `increment_outputs` kwarg to `_commit_lock_and_build_deferred` and `_build_deferred_writes`; pass `False` from run-cache skip path | ~8 lines changed |
| `src/pivot/engine/engine.py` | Change condition from `== StageStatus.RAN` to `in (StageStatus.RAN, StageStatus.SKIPPED)` | 1 line changed |
| `tests/storage/test_state.py` | Add 2 new tests, update 2 existing tests for flag behavior | ~40 lines |
| `tests/test_run_cache_lock_update.py` | Add integration test for dep_generations after run-cache skip | ~35 lines |
