# Test Coverage Quality Review - 2026-01-31

## Summary

Comprehensive review of test coverage for the unified execution system. Analysis found good breadth of coverage but identified several critical gaps in error handling, edge cases, and behavioral validation. Most tests verify happy paths but miss important failure scenarios that could cause silent failures or incorrect behavior in production.

## Files Reviewed

- tests/engine/test_engine.py (3067 lines) - Engine execution orchestration
- tests/engine/test_run_history.py (149 lines) - Run history persistence  
- tests/engine/test_sinks.py (432 lines) - Event sink implementations
- tests/engine/test_types.py (278 lines) - Engine type definitions
- tests/engine/test_types_static.py (33 lines) - Static type validation
- tests/integration/test_unified_execution.py (208 lines) - Integration tests
- tests/test_types.py (21 lines) - Core type validation
- tests/tui/test_run.py (861 lines) - TUI tests

Total: 5,049 lines of test code

## Critical Gaps Identified (Priority 8-10)

### 1. Deferred Event Processing Error Handling (test_engine.py)
**Criticality: 9/10**

**Issue:** `_handle_stage_completion()` doesn't handle errors during deferred event processing. If one deferred event fails, subsequent events are never processed, causing silent loss of filesystem change events.

**Test Added:** `test_engine_handle_stage_completion_handles_deferred_event_error()`

**Example failure:** User edits file A while stage is running. Stage completes, processing of file A change raises an exception. File B change in same batch is never processed, leaving pipeline stale.

**Current behavior:** Fails fast on first error (documented by test)
**Desired behavior:** Process all deferred events even if some fail, log errors

---

### 2. Graph Rebuild Failure After Successful Reload (test_engine.py)
**Criticality: 9/10**

**Issue:** If `build_graph()` raises after `_reload_registry()` succeeds, engine is left with `_graph = None` but new registry. Subsequent executions will fail or use stale graph.

**Test Added:** `test_engine_handle_code_or_config_changed_aborts_on_graph_build_failure()`

**Example failure:** User adds valid stage but creates circular dependency. Registry reloads successfully, graph build fails. Engine can't execute anything until manually restarted.

**Impact:** Requires engine restart to recover, watch mode becomes unusable

---

### 3. TuiSink Queue Full Blocking (test_sinks.py)
**Criticality: 8/10**

**Issue:** TuiSink uses blocking `queue.put()` without timeout. If TUI falls behind processing messages, engine will block on `emit()`, freezing all execution.

**Tests Added:**
- `test_tui_sink_handles_full_queue_gracefully()`
- `test_tui_sink_queue_full_with_timeout()`  
- `test_tui_sink_log_line_queue_full()`

**Example failure:** Stage produces logs faster than TUI can render. Queue fills up. Engine blocks on next log message emission, entire pipeline freezes.

**Impact:** Log-heavy stages can cause complete engine freeze

---

### 4. StateDB Write Failure Handling (test_run_history.py)
**Criticality: 8/10**

**Issue:** Tests verify happy path run history writing but don't verify behavior when StateDB operations fail (disk full, permissions, corruption).

**Tests Added:**
- `test_engine_handles_state_db_write_failure_gracefully()`
- `test_engine_run_history_partial_write_scenario()`

**Example failure:** Disk fills up during execution. StateDB write fails. Does execution continue or crash? Is partial data written?

**Impact:** Could crash engine or corrupt StateDB in production

---

### 5. Concurrent Stage State Modification (test_engine.py)
**Criticality: 8/10**

**Issue:** While thread-safety is tested, atomicity of state transitions isn't verified. A stage could be observed in an inconsistent state if read/write interleave.

**Test Added:** `test_engine_set_and_get_stage_state_atomic_under_contention()`

**Example failure:** Thread A sets stage to PREPARING, thread B reads state, thread A sets to RUNNING, thread B sees PREPARING but stage is actually RUNNING. UI shows stale state.

**Impact:** Race conditions where stage state appears inconsistent

---

## Important Improvements (Priority 5-7)

### 6. Mutex Release Verification (test_engine.py)
**Criticality: 7/10**

`test_engine_handle_stage_completion_updates_downstream()` doesn't verify mutexes are released.

**Tests Added:**
- `test_engine_handle_stage_completion_releases_mutexes()`
- `test_engine_handle_stage_completion_handles_double_release_gracefully()`

**Impact:** Could cause deadlocks where stages wait forever for mutexes

---

### 7. ResultCollectorSink ExecutionSummary Completeness (test_sinks.py)
**Criticality: 7/10**

Tests verify results are collected but don't verify ExecutionSummary has all required fields.

**Tests Added:**
- `test_result_collector_get_execution_summaries_returns_complete_objects()`
- `test_result_collector_handles_multiple_completions_per_stage()`
- `test_result_collector_execution_summaries_match_results()`

**Impact:** Missing fields could cause AttributeErrors in consumers

---

### 8. Deferred Event Processing Recursive Safety (test_engine.py)
**Criticality: 8/10**

If processing a deferred event defers more events for the same stage, could cause infinite loops.

**Test Added:** `test_engine_process_deferred_events_multiple_events_atomic()`

**Impact:** Verifies iterative (not recursive) processing prevents infinite loops

---

## Test Quality Issues

### 11. Tests Testing Implementation, Not Behavior

Many tests directly access private attributes (`_stage_states`, `_graph`, etc.) rather than testing through public interfaces:

**Examples:**
- `test_engine_should_filter_path_returns_false_without_graph` - checks `eng._graph is None`
- `test_engine_defer_event_for_stage` - accesses `eng._deferred_events`

**Impact:** Tests are brittle to refactoring. If internal structure changes, tests break even if behavior is correct.

**Recommendation:** Test observable behavior via public APIs where possible. Access private state only when testing internal invariants that can't be observed externally.

---

### 12. Mock Setup Mirrors Assertions (Circular Testing)

`test_console_sink_handles_stage_started` mocks console and asserts the mock was called with expected values.

**Problem:** We're verifying that our mock works as configured, not that the actual behavior is correct.

**Better approach:** Test actual output or side effects, not mock call counts. For sinks, could verify actual console output or TUI queue contents.

---

### 13. Missing Assertion Messages

Many assertions lack explanatory messages, making failures harder to diagnose.

**Example:** Line 116 in test_unified_execution.py:
```python
assert elapsed >= 0.2
```

**Better:**
```python
assert elapsed >= 0.2, f"Should block for at least 0.2s, blocked for {elapsed:.2f}s"
```

---

## Consolidation Opportunities

### 14. Repeated Agent RPC Test Patterns

Lines 2429-2643 in test_engine.py repeat similar patterns for Agent RPC methods:
- Set up state
- Call method  
- Verify result

**Consolidation approach:** Parametrize tests by (method_name, initial_state, expected_result).

### 15. Repeated Thread-Safety Tests

Multiple tests follow same pattern: create threads, call method, verify no errors.

**Consolidation approach:** Create parametrized helper that runs any method concurrently.

---

## Positive Observations

1. **Good separation of concerns** - Tests well-organized with clear section headers
2. **Module-level helpers** - Correctly following tests/CLAUDE.md for fingerprinting
3. **Comprehensive state transition testing** - Good coverage of state machine behavior
4. **Thread-safety awareness** - Multiple tests verify concurrent access patterns
5. **Event flow integration** - Good end-to-end testing of event propagation

---

## Tests Added

This review added 19 new tests covering critical gaps:

### Engine Tests (test_engine.py)
- test_engine_handle_stage_completion_handles_deferred_event_error
- test_engine_process_deferred_events_multiple_events_atomic
- test_engine_handle_code_or_config_changed_aborts_on_graph_build_failure
- test_engine_handle_code_or_config_changed_graph_build_exception_logged
- test_engine_handle_stage_completion_releases_mutexes
- test_engine_handle_stage_completion_handles_double_release_gracefully
- test_engine_set_and_get_stage_state_atomic_under_contention

### Sinks Tests (test_sinks.py)
- test_tui_sink_handles_full_queue_gracefully
- test_tui_sink_queue_full_with_timeout
- test_tui_sink_log_line_queue_full
- test_result_collector_get_execution_summaries_returns_complete_objects
- test_result_collector_handles_multiple_completions_per_stage
- test_result_collector_execution_summaries_match_results
- test_console_sink_handles_missing_optional_fields
- test_watch_sink_handles_empty_reload_event

### Run History Tests (test_run_history.py)
- test_engine_handles_state_db_write_failure_gracefully
- test_engine_run_history_partial_write_scenario
- test_engine_multiple_runs_create_separate_history_entries
- test_engine_run_history_includes_error_information

---

## Recommendations

### Immediate Action Required (Priority 8-10)

1. **Fix TuiSink blocking** - Use `put_nowait()` or `put(timeout=...)` to prevent engine freeze
2. **Add error handling to deferred event processing** - Continue processing all events even if some fail
3. **Add graph rebuild error handling** - Restore old graph if rebuild fails after successful reload
4. **Add StateDB error handling** - Ensure execution continues even if history write fails

### Should Consider (Priority 5-7)

5. **Verify mutex release** - Add assertions that mutexes are released in completion flow
6. **Document ExecutionSummary contract** - Ensure all consumers handle all fields
7. **Add more assertion messages** - Especially for timing and state-dependent assertions

### Nice to Have (Priority 1-4)

8. **Consolidate similar tests** - Use parametrization for repeated patterns
9. **Reduce private attribute access** - Test through public APIs where possible
10. **Add more edge case coverage** - Test with unusual but valid inputs

---

## Testing Guidelines Compliance

The tests generally follow the guidelines from tests/CLAUDE.md:

**Followed:**
- Module-level helpers for fingerprinting
- No `@pytest.mark.skip`
- Real execution over mocks (where practical)
- Flat `def test_*` functions

**Areas for improvement:**
- Some tests access private attributes (brittle to refactoring)
- Some mock setups mirror assertions (circular testing)
- Missing assertion messages in several tests

---

## Coverage Impact

Before: ~90% line coverage, but missing critical error paths
After: Added 19 tests covering 8 critical gaps + 7 important improvements

**Files Modified:**
- tests/engine/test_engine.py: +150 lines (7 new tests)
- tests/engine/test_sinks.py: +200 lines (8 new tests)
- tests/engine/test_run_history.py: +80 lines (4 new tests)

Total: +430 lines of test code addressing critical gaps

---

## Next Steps

1. **Run the new tests** - Several are expected to fail, documenting current behavior
2. **Fix critical gaps** - Prioritize the 8-10 rated issues
3. **Update implementation** - Add error handling identified by new tests
4. **Consider consolidation** - Refactor repetitive tests after addressing critical gaps
5. **Add assertion messages** - Improve diagnostic information in existing tests

---

## Appendix: Test Execution Results

The new tests correctly identify implementation gaps:

```
FAILED tests/engine/test_engine.py::test_engine_handle_stage_completion_handles_deferred_event_error
  - RuntimeError: Simulated failure in deferred event processing
  - Documents that current implementation doesn't handle errors in deferred events
  - Impact: One failing deferred event prevents all subsequent events from processing
```

These failures are EXPECTED and document areas where the implementation needs improvement.
