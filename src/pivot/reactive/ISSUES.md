# Reactive Engine - Known Issues and Improvements

This file tracks non-critical issues and improvement opportunities identified during development of the reactive execution engine.

## Low Severity

### 1. TOCTOU Race in File Index Building
**Location:** `engine.py` in `_build_file_to_stages_index()`

If stages are registered/unregistered while `_build_file_to_stages_index()` iterates (rare in hot-reload scenarios), the result could be inconsistent.

**Suggested fix:** Capture stage list once at the start and handle KeyError if a stage is removed mid-iteration.

### 2. Debounce Returns Empty on Shutdown with Pending Changes
**Location:** `engine.py` in `_collect_and_debounce()`

If shutdown is set after changes are collected but before the quiet period completes, the function returns an empty set, discarding collected changes. This is minor since we're shutting down anyway.

## Test Coverage Gaps

~~1. No test for watcher thread exception handling recovery~~ - Added `test_watcher_thread_exception_triggers_shutdown`
~~2. No test for symlink resolution in watch filter~~ - Added `test_watch_filter_resolves_symlinks`, `test_watch_filter_handles_broken_symlink`
~~3. No test for queue overflow behavior (100+ changes)~~ - Added `test_queue_overflow_triggers_full_rebuild_sentinel`, `test_watch_loop_handles_queue_overflow`
~~4. No test for concurrent shutdown while run is in progress~~ - Added `test_concurrent_shutdown_during_debounce`, `test_concurrent_shutdown_during_run`
~~5. No integration test with actual `watchfiles` (all are mocked)~~ - Added `test_integration_watchfiles_detects_real_file_change`, `test_integration_watchfiles_detects_python_code_change`

Additional tests added:
- `test_get_stages_affected_handles_deleted_file` - Tests graceful handling of deleted files
- `test_dag_cache_invalidation_on_code_change` - Tests DAG cache invalidation
- `test_file_index_caching_returns_same_instance` - Tests file index caching

## Resolved Issues

The following issues were addressed:

- **Unbounded DAG Rebuilds**: DAG is now cached and invalidated only on code changes
- **Queue Overflow with Unbounded Memory Growth**: Added max pending threshold with sentinel for full rebuild
- **File Index Rebuilt on Every Change Event**: File index is now cached
- **Worker Restart Doesn't Verify Success**: Now logs worker count after restart
- **Redundant .py Suffix Check**: `code_changed` is now passed as a parameter
- **Trivial Wrapper Methods**: Removed `_send_status` and `_send_error`
- **Resource Leak on Exception Path**: Added try/finally in `run()` to ensure watcher thread is joined
- **`_stages` List Passed by Reference**: Now copies the list in `__init__`
- **Mutability Leak in Return Value**: `_get_affected_stages` now returns a copy
- **Sentinel Bug**: Changed sentinel from `Path(".")` to named constant `_FULL_REBUILD_SENTINEL` and check for it explicitly
- **Config File Detection**: Now detects changes to `pivot.yaml`, `pivot.yml`, and `pipeline.py` as code changes
- **Cache Invalidation**: Moved to `_invalidate_caches()` method with atomic replacement pattern
- **Deleted File Handling**: `_resolve_path_for_matching()` handles deleted files by using normalized absolute path
- **Duplicate REGISTRY.get() Pattern**: Extracted to `_iter_stage_infos()` helper
- **Split Affected Stages Logic**: `_get_stages_affected_by_files()` split into `_get_stages_with_direct_dep_match()` and `_get_stages_with_containment_match()` for clearer responsibilities
- **Path Resolution Consistency**: Created `_resolve_path_for_matching()` for consistent path handling
- **params.yaml Detection**: Added `params.yaml` and `params.yml` to `_CONFIG_FILE_NAMES` so parameter changes trigger reruns
- **Registry Reload**: Added `_reload_registry()` method that re-imports stage modules when code changes, allowing new/removed stages to be detected
- **Partial Registry Reload**: `_reload_registry()` now preserves the previous valid registry when reload fails (syntax errors, import errors). The engine tracks `_pipeline_errors` and skips execution while invalid, showing an error banner. When the user fixes the error and saves, the next reload clears the error state and resumes normal operation.

## Known Limitations
- **Incremental outputs during worker kill**: If workers are killed (via `kill_workers=True`) while writing to an incremental output, the file may be partially written. However, the cache contains the last good version, so the next run will restore from cache before re-executing.
- **Environment variables**: Changes to environment variables are not detected (not file-based).
- **External packages**: Updates to pip packages are not detected.
- **External modifications to intermediate files**: The watcher filters ALL stage outputs to prevent infinite loops (stage produces output → detected → stage runs again). This means external modifications to intermediate files (outputs that are also deps of downstream stages) are not detected. **Workaround**: Modify an input file to trigger a full re-run.

## Future Enhancements

### Intermediate File Detection
**Priority:** Medium

Currently, external modifications to intermediate files (e.g., `clean.csv` which is output of `clean` stage and dep of `features` stage) are not detected because all outputs are filtered to prevent infinite loops.

**Potential solutions:**
1. **Timestamp-based filtering**: Record execution start time, ignore file changes with mtime after that timestamp
2. **State-based filtering**: Use permissive filter during "waiting" state, restrictive filter during "running" state
3. **Content-based**: Track file hashes before/after execution, only trigger on external changes

**Test case:** `test_get_affected_stages_includes_downstream_when_intermediate_file_changes` in `tests/reactive/test_engine.py` verifies the core logic works correctly - the issue is at the watcher filter level.
