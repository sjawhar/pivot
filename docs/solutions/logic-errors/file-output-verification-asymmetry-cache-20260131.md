---
module: Cache/Executor
date: 2026-01-31
problem_type: logic_error
component: service_object
symptoms:
  - "Single file Out outputs only checked existence, not content hash"
  - "Corrupted output files silently accepted as valid cache hits"
  - "Asymmetry between file and directory verification logic"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags: [cache, output-verification, skip-detection, file-hash, pivot]
---

# Troubleshooting: File Output Verification Only Checked Existence, Not Content

## Problem

Single file `Out` outputs only checked existence during skip detection, while `DirectoryOut` verified content hashes. This meant corrupted/modified output files were silently accepted as valid cache hits, causing stale data to persist.

## Environment

- Module: Cache/Executor (`src/pivot/executor/worker.py`, `src/pivot/storage/cache.py`)
- Python Version: 3.13+
- Affected Component: Output verification during skip detection
- Date: 2026-01-31

## Symptoms

- Single file `Out` outputs only checked `path.exists()` during skip detection
- `DirectoryOut` correctly verified content hashes via `_directory_needs_restore()`
- Corrupted output files were silently accepted as valid cache hits
- Users could have stale/corrupted outputs that didn't match the lockfile

## What Didn't Work

**Direct solution:** The problem was identified through integration tests and fixed on the first attempt after analysis.

Tests F2 (`test_corrupted_file_restored_on_cache_hit`) and F3 (`test_corrupted_file_triggers_rerun_when_cache_empty`) were written first and confirmed the bug existed.

## Solution

Added `_file_needs_restore()` function to verify file content matches cached hash, symmetric with the existing `_directory_needs_restore()` function.

**Code changes:**

```python
# Before (broken) - src/pivot/executor/worker.py:495-502
needs_restore = not path.exists()  # Only checks existence for files

# For directories, verify contents match even if path exists
if not needs_restore and is_dir_hash(output_hash):
    needs_restore = _directory_needs_restore(path, output_hash)
```

```python
# After (fixed) - src/pivot/executor/worker.py:496-500
# Verify content matches cached hash (directories and files)
if is_dir_hash(output_hash):
    needs_restore = _directory_needs_restore(path, output_hash, state_db)
else:
    needs_restore = _file_needs_restore(path, output_hash, state_db)
```

**New function added:**

```python
def _file_needs_restore(
    path: pathlib.Path, cached_hash: FileHash, state_db: state.StateDB | None = None
) -> bool:
    """Check if file content differs from cached hash.

    Returns True if restoration is needed (file missing or content mismatch).
    """
    if not path.exists():
        return True

    try:
        current_hash = cache.hash_file(path, state_db)
        return current_hash != cached_hash["hash"]
    except OSError:
        return True
```

## Why This Works

1. **ROOT CAUSE:** The original code had asymmetric verification logic. Directories used `_directory_needs_restore()` to verify content hashes, but files only checked `path.exists()`. The hash data WAS available in `output_hash["hash"]` but wasn't being used for files.

2. **Why the solution works:** The new `_file_needs_restore()` function mirrors `_directory_needs_restore()` behavior - it computes the current file hash and compares it to the cached hash. If they differ (or if any error occurs), it returns `True` to trigger cache restoration.

3. **Underlying issue:** This was a logic error where the developer added directory verification but didn't extend the same logic to single files. The code structure suggested files and directories should be handled the same way, but the implementation diverged.

## Prevention

- When adding new output types, ensure verification logic matches caching logic
- Add integration tests that corrupt outputs and verify correct restoration behavior
- Use symmetric patterns - if directories verify content, files should too
- The test suite now includes comprehensive file output verification tests:
  - `test_missing_file_restored_on_cache_hit`
  - `test_corrupted_file_restored_on_cache_hit`
  - `test_corrupted_file_triggers_rerun_when_cache_empty`
  - `test_perfect_match_skips_without_modification`

## Related Issues

No related issues documented yet.

## Additional Fixes in This PR

1. **Fixed `ignore_filter` to use absolute paths** - Copilot review caught that `is_ignored()` was called with paths relative to the hashed directory, but `.pivotignore` patterns are relative to project root. Changed to pass absolute paths.

2. **Fixed tests using `build/` in `_IGNORE_DIRS`** - Tests were using `build/` directory which is hardcoded in `_IGNORE_DIRS`, so they passed even if `ignore_filter` was broken. Renamed to `artifacts/` and `output/`.

3. **Fixed lock file assertion format** - Test checked for JSON `{"status": "failed"}` but lock files are YAML. Updated to assert lock file doesn't exist after failure.
