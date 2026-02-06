# Restore Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Always compute output hashes (even for `cache=False` outputs like `Metric()`), and stop using `hash is None` as the sentinel for "not cached" throughout the codebase.

**Architecture:** (1) Worker computes hashes for all outputs but only saves cached ones to the cache directory, (2) CLI commands (checkout, verify) filter non-cached outputs using the stage registry.

**Tech Stack:** Python 3.13+, pytest, pivot internals

---

## Context

`Metric()` outputs have `cache=False` by default, meaning they're git-tracked and not Pivot's responsibility to restore. Two bugs exist:

1. **Lockfile incompleteness:** `_save_outputs_to_cache()` stores `None` as the hash for non-cached outputs. The hash should always be computed — it's useful for diffs/status.

2. **`None` sentinel is overloaded:** The codebase uses `output_hash is None` to mean "not cached" in many places. This is fragile and needs to be replaced with explicit `cache` flag checks.

### Impact of removing the `None` sentinel

| Location | What it does | Fix strategy |
|----------|-------------|--------------|
| `worker.py:645` `_save_outputs_to_cache` | Sets `None` for non-cached | Compute real hash (Task 1) |
| `worker.py:1112` `_try_skip_via_run_cache` | Sets `None` for non-cached in lock data | Compute real hash (Task 1) |
| `worker.py:533` `_restore_outputs_from_cache` | Passes all outputs including non-cached to `_restore_outputs` | Filter at call site (Task 1) |
| `worker.py:501` `_restore_outputs` | `None` = just verify exists | Keep for backward compat with old lockfiles |
| `worker.py:1026` `_build_deferred_writes` | Filters `None` via `output_hash_to_entry` | Filter by `cache` flag explicitly (Task 1) |
| `run_history.py:111` `compute_input_hash_from_lock` | Derives `cache` flag from `oh is not None` | Accept one-time run cache miss; follow-up issue filed |
| `commit.py:52` `commit_pending` | Uses `compute_input_hash_from_lock` | No change; follow-up issue to rearchitect |
| `checkout.py:137` `_checkout_files_async` | Skips `None` hashes | Non-cached filtered upstream; keep `None` check for backward compat |
| `verify.py:54` `_extract_file_hashes` | Skips `None` hashes | Non-cached filtered upstream; keep `None` check for backward compat |
| `sync.py:65` `get_stage_output_hashes` | Skips `None` for remote sync | Leave as-is for now (minor inefficiency, not correctness) |

---

### Task 1: Worker — compute real hashes, fix lock-file skip path

**Files:**
- Modify: `src/pivot/executor/worker.py` — `_save_outputs_to_cache` (line 640), `_restore_outputs_from_cache` (line 533), `_try_skip_via_run_cache` (line 1106), `_build_deferred_writes` (line 1026)

#### New helper: `_hash_output(path, state_db)`

Extract the duplicated hash-without-caching logic into a small helper (used by both `_save_outputs_to_cache` and `_try_skip_via_run_cache`):

```python
def _hash_output(
    path: pathlib.Path, state_db: state.StateDB | None = None
) -> FileHash | DirHash:
    """Compute output hash without saving to cache."""
    if path.is_dir():
        tree_hash, manifest = cache.hash_directory(path, state_db)
        return DirHash(hash=tree_hash, manifest=manifest)
    file_hash = cache.hash_file(path, state_db)
    return FileHash(hash=file_hash)
```

Note: `FileHash` and `DirHash` are already imported from `pivot.types` in this file. Pass `state_db` when available to use the StateDB hash cache (avoids re-hashing from disk).

#### `_save_outputs_to_cache` (line 640-645)

Compute real hashes for non-cached outputs using the new helper. No `state_db` available here (consistent with `save_to_cache` which also doesn't use it).

```python
if out.cache:
    output_hashes[str(out.path)] = cache.save_to_cache(
        path, files_cache_dir, checkout_modes=checkout_modes
    )
else:
    output_hashes[str(out.path)] = _hash_output(path)
```

#### `_restore_outputs_from_cache` (line 533-553)

Filter non-cached outputs at the call site — same pattern `_try_skip_via_run_cache` already uses at lines 1082-1090. Don't modify `_restore_outputs` itself (the `None` check at line 501 remains for backward compat with old lockfiles).

```python
def _restore_outputs_from_cache(
    stage_outs: list[outputs.BaseOut],
    lock_data: LockData,
    files_cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    *,
    state_db: state.StateDB | None = None,
) -> bool:
    """Restore missing outputs from cache for lock file skip detection."""
    # Non-cached outputs (Metric) are git-tracked — just verify they exist
    for out in stage_outs:
        if not out.cache:
            if not pathlib.Path(cast("str", out.path)).exists():
                return False

    # Only restore cached outputs from cache
    cached_path_strings = [cast("str", out.path) for out in stage_outs if out.cache]
    return _restore_outputs(
        cached_path_strings,
        lock_data["output_hashes"],
        files_cache_dir,
        checkout_modes,
        use_normalized_paths=True,
        state_db=state_db,
    )
```

#### `_try_skip_via_run_cache` (line 1106-1112)

Compute real hashes for non-cached outputs instead of setting `None`. The non-cached files are known to exist (checked at line 1083-1087). Pass `state_db` (available as a parameter at line 1059) to leverage the StateDB hash cache.

```python
output_hashes: dict[str, OutputHash] = {}
for out in stage_outs:
    out_path = str(out.path)
    if out.cache:
        output_hashes[out_path] = output_hash_map[out_path]
    else:
        output_hashes[out_path] = _hash_output(pathlib.Path(out_path), state_db)
```

#### `_build_deferred_writes` (line 1026-1030)

Run cache entries should only contain cached outputs. Currently `output_hash_to_entry` filters by `oh is None`, which won't work after this change. Filter explicitly using `stage_info["outs"]`:

```python
cached_paths = {str(out.path) for out in stage_info["outs"] if out.cache}
output_entries = [
    entry
    for path, oh in output_hashes.items()
    if path in cached_paths and (entry := run_history.output_hash_to_entry(path, oh)) is not None
]
```

---

### Task 2: CLI commands — filter non-cached outputs

**Files:**
- Modify: `src/pivot/cli/checkout.py` — `_get_stage_output_info` (line 35), remove `None` check in `_validate_and_build_files` (line 207)
- Modify: `src/pivot/cli/verify.py` — `_get_stage_lock_hashes` (line 65)

#### Checkout: `_get_stage_output_info` (line 35-49)

Filter non-cached outputs using the stage registry:

```python
def _get_stage_output_info(state_dir: pathlib.Path) -> dict[str, OutputHash]:
    """Get output hash info from lock files for cached stage outputs only.

    Non-cached outputs (e.g. Metric with cache=False) are excluded —
    they are git-tracked and not Pivot's responsibility to restore.
    """
    result = dict[str, OutputHash]()

    for stage_name in cli_helpers.list_stages():
        stage_info = cli_helpers.get_stage(stage_name)
        non_cached_paths = {str(out.path) for out in stage_info["outs"] if not out.cache}

        stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
        lock_data = stage_lock.read()
        if lock_data and "output_hashes" in lock_data:
            for out_path, out_hash in lock_data["output_hashes"].items():
                norm_path = str(project.normalize_path(out_path))
                if norm_path not in non_cached_paths:
                    result[norm_path] = out_hash

    return result
```

#### Checkout: `_validate_and_build_files` (line 205-211)

Remove the `output_hash is None` check — non-cached outputs won't be in `stage_outputs`. If someone targets a non-cached output, it falls through to "Unknown target" error.

#### Verify: `_get_stage_lock_hashes` (line 65-82)

Same pattern — filter non-cached outputs before extracting hashes:

```python
def _get_stage_lock_hashes(
    stage_name: str, state_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(state_dir))
    lock_data = stage_lock.read()
    if lock_data is None:
        return {}, {}

    # Filter non-cached outputs — they're git-tracked, not in cache
    stage_info = cli_helpers.get_stage(stage_name)
    non_cached_paths = {str(out.path) for out in stage_info["outs"] if not out.cache}
    cached_output_hashes: dict[str, OutputHash] = {
        path: h for path, h in lock_data["output_hashes"].items() if path not in non_cached_paths
    }

    return (
        _extract_file_hashes(cached_output_hashes),
        _extract_file_hashes(lock_data["dep_hashes"]),
    )
```

Keep `if hash_info is None: continue` in `_extract_file_hashes` for backward compat with old lockfiles.

---

### Task 3: Tests and quality checks

**Files:**
- New test in `tests/execution/test_executor.py` or nearby
- New test in `tests/cli/test_cli_checkout.py`
- Any tests that assert `output_hash is None` for non-cached outputs (note: `test_execution_modes.py:449` asserts `None` for `--no-cache` mode, which is unaffected — don't change it)

#### Test 1: Non-cached outputs get real hashes

Verify `_save_outputs_to_cache` computes real hashes for `Metric()` outputs and doesn't save them to the cache directory.

#### Test 2: Checkout skips non-cached outputs

Integration test: create a stage with both `Out()` and `Metric()` outputs, run it, delete both outputs, `pivot checkout` restores only the `Out()` output.

Define a module-level helper function for the stage (required for fingerprinting — see test conventions in AGENTS.md).

#### Test 3: Lock-file skip with mixed outputs

Verify that `_restore_outputs_from_cache` correctly skips non-cached outputs (verifies they exist on disk) and restores cached outputs from cache.

#### Run full suite and quality checks

```bash
uv run pytest tests/ -x -n auto
uv run ruff format . && uv run ruff check . && uv run basedpyright
```

---

## Files Modified Summary

| File | Change |
|------|--------|
| `src/pivot/executor/worker.py` | New `_hash_output` helper; use it in `_save_outputs_to_cache` and `_try_skip_via_run_cache`; filter non-cached in `_restore_outputs_from_cache` and `_build_deferred_writes` |
| `src/pivot/cli/checkout.py` | Filter non-cached outputs in `_get_stage_output_info`; remove `None` check in `_validate_and_build_files` |
| `src/pivot/cli/verify.py` | Filter non-cached outputs in `_get_stage_lock_hashes` |

## Not Changed (Intentionally)

| File | Reason |
|------|--------|
| `src/pivot/remote/sync.py` | Leave `is not None` checks — minor inefficiency for new lockfiles (non-cached files synced to remote), not a correctness issue. Tighten later. |
| `src/pivot/executor/worker.py` `_restore_outputs` | Keep `None` check at line 501 for backward compat with old lockfiles |
| `src/pivot/cli/checkout.py` `_checkout_files_async` | Keep `None` check at line 137 for backward compat with old lockfiles |
| `src/pivot/cli/verify.py` `_extract_file_hashes` | Keep `None` check for backward compat with old lockfiles |
| `src/pivot/executor/commit.py` | `compute_input_hash_from_lock` will produce wrong input hash for new lockfiles — `commit_pending` writes an orphaned run cache entry (keyed by wrong hash, never looked up). Harmless but wasteful. Follow-up: rearchitect `commit_pending` to compute from current graph — see #372. |

## Verification

1. `uv run pytest tests/ -x -n auto`
2. `uv run ruff format . && uv run ruff check . && uv run basedpyright`
3. Smoke test: stage with `Metric()` output → run → check lockfile has real hash → `pivot checkout` doesn't restore it
