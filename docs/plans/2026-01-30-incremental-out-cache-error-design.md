# IncrementalOut Cache Error Message Improvement

**Issue:** #232
**Status:** Ready for implementation

## Problem

When a lockfile references a cached IncrementalOut file that doesn't exist locally, the error message is unhelpful:

```
Failed to restore IncrementalOut 'path/to/file.yaml' from cache
```

Users don't know how to fix this.

## Solution

Improve the error message to provide actionable guidance:

```
Cache missing for IncrementalOut 'path/to/file.yaml'. Run `pivot pull` to fetch from remote, or delete `/path/to/.pivot/stages/<stage>.lock` to start fresh.
```

The lock file path is dynamically determined based on whether the lock data came from a pending lock (`--no-commit` mode) or the production lock.

## Implementation

### File: `src/pivot/executor/worker.py`

**Change 1:** Update error message in `_prepare_outputs_for_execution` (~line 492)

The function raises a clean base message:

```python
if not restored:
    raise exceptions.CacheRestoreError(
        f"Cache missing for IncrementalOut '{out.path}'"
    )
```

**Change 2:** Wrap call site (~line 294) to add stage-specific context

```python
try:
    _prepare_outputs_for_execution(stage_outs, lock_data, files_cache_dir)
except exceptions.CacheRestoreError as e:
    lock_path = pending_lock.path if pending_lock_data else production_lock.path
    raise exceptions.CacheRestoreError(
        f"{e}. Run `pivot pull` to fetch from remote, or delete "
        + f"`{lock_path}` to start fresh."
    ) from e
```

Note: The lock path is dynamically determined based on whether the lock data came from a pending lock (used in `--no-commit` mode) or the production lock.

### Tests

Add test to `tests/storage/test_incremental_out.py`:

- `test_prepare_outputs_incremental_missing_cache_error` — verify the base error message from `_prepare_outputs_for_execution`

Integration test coverage for the full error message (with stage name) would require an end-to-end test through the worker, which may be overkill for an error message change.

## Design Decisions

1. **Error with instructions** (not auto-recovery) — respects "fail fast with clear errors" principle
2. **Both options in message** — `pivot pull` or delete lockfile, without prioritizing one
3. **Caller adds context** — keeps `_prepare_outputs_for_execution` signature clean; stage name added at call site where it's naturally available
