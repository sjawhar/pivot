# Watch Mode Architecture Issues - Investigation & Fix Plan

## Issue #141: Watch Mode Doesn't Detect External Modifications to Intermediate Files

**GitHub Issue**: https://github.com/sjawhar/pivot/issues/141

### Problem Statement

Watch mode blocks ALL stage outputs to prevent infinite loops. This inadvertently blocks detection of external modifications to intermediate files (files that are outputs of one stage and dependencies of downstream stages).

**Example**:
- Pipeline: `raw.csv` → `clean` stage → `clean.csv` → `features` stage → `features.csv`
- User manually edits `clean.csv` outside Pivot
- Watch mode ignores the change because `clean.csv` is in the filter list
- `features` stage never re-runs despite its dependency changing

### Root Cause

File: `/home/user/pivot/src/pivot/reactive/_watch_utils.py:49-88`

```python
def create_watch_filter(stages_to_run: list[str]) -> Callable:
    """Create filter excluding outputs from stages being run (prevents infinite loops)."""
    outputs_to_filter: set[pathlib.Path] = set()
    for p in get_output_paths_for_stages(stages_to_run):
        resolved = project.try_resolve_path(p)
        if resolved is not None:
            outputs_to_filter.add(resolved)  # ← Blocks ALL outputs forever

    def watch_filter(change: Change, path: str) -> bool:
        # Check if path is an output
        for out in outputs_to_filter:
            if resolved_path == out or out in resolved_path.parents:
                return False  # ← Blocks everything, no timing awareness
        return True

    return watch_filter
```

The filter is a **static closure** that blocks all stage outputs, with no concept of:
- When the file was modified (before vs after execution)
- Whether the modification came from Pivot or external source
- Execution state (idle vs running)

---

## Additional Architectural Issues Discovered

### Issue #2: TUI Doesn't Know About Stages Until They Execute

**Current Behavior**: In watch mode, the TUI starts empty and only learns about stages as they start executing.

**Expected Behavior**: TUI should display all stages immediately with PENDING status.

**Root Cause**:

File: `/home/user/pivot/src/pivot/cli/run.py:164-200`

```python
def _run_watch_with_tui(...):
    # DAG is built and execution_order is calculated
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    # ... pre-warm executor, create queues ...

    engine = reactive_module.ReactiveEngine(...)

    # BUT: execution_order is NEVER passed to the TUI!
    run_tui.run_watch_tui(engine, tui_queue, output_queue=output_queue, tui_log=tui_log)
    #                     ^^^^^^ no stage_names parameter
```

File: `/home/user/pivot/src/pivot/tui/run.py:570-577`

```python
class WatchTuiApp(_BaseTuiApp[None]):
    def __init__(self, engine, message_queue, output_queue=None, tui_log=None):
        super().__init__(message_queue, tui_log=tui_log)  # ← No stage_names!
        #                                  ^^^^^^^^^^^^^^ stage_names defaults to None
```

File: `/home/user/pivot/src/pivot/tui/run.py:328-332`

```python
class _BaseTuiApp:
    def __init__(self, message_queue, stage_names=None, tui_log=None):
        if stage_names:  # ← Never enters this block in watch mode
            for i, name in enumerate(stage_names, 1):
                info = StageInfo(name, i, len(stage_names))
                self._stages[name] = info
```

**Impact**: Users see stages appear one-by-one as they execute, creating confusion about pipeline structure.

---

### Issue #3: Watch Filter Never Updates After Registry Reload

**Status**: **Documented but not fixed** (see test at line 1961 in test_engine.py)

File: `/home/user/pivot/tests/reactive/test_engine.py:1965-1969`

```python
def test_watch_filter_stale_after_registry_adds_new_stage(...):
    """Watch filter should update when registry reload adds new stages.

    Bug: _create_watch_filter() captures output paths at startup. If registry
    reload adds a new stage with new outputs, the filter doesn't know about them,
    potentially causing infinite loops (new stage output triggers another reload).
    """
```

**Root Cause**:

File: `/home/user/pivot/src/pivot/reactive/engine.py:240-253`

```python
def _watch_loop(self, stages_to_run: list[str]) -> None:
    """Pure producer - monitors files, enqueues changes."""
    watch_paths = _watch_utils.collect_watch_paths(stages_to_run)
    watch_filter = _watch_utils.create_watch_filter(stages_to_run)  # ← Created ONCE

    for changes in watchfiles.watch(
        *watch_paths,
        watch_filter=watch_filter,  # ← Same filter used forever
        stop_event=self._shutdown,
    ):
        # Filter NEVER updates even after registry reload at line 297!
```

File: `/home/user/pivot/src/pivot/reactive/engine.py:294-308`

```python
def _coordinator_loop(self):
    if code_changed:
        self._invalidate_caches()
        reload_ok = self._reload_registry()  # ← Registry reloaded
        self._restart_worker_pool()
        # ← But watch_filter in _watch_loop never knows about new stages!
```

**Impact**:
- New stage outputs after reload are NOT filtered → potential infinite loops
- Removed stage outputs after reload are STILL filtered → legitimate changes blocked

---

### Issue #4: Watch Paths Never Update After Registry Reload

Similar to Issue #3, but for the watched directories:

File: `/home/user/pivot/src/pivot/reactive/engine.py:240-244`

```python
def _watch_loop(self, stages_to_run: list[str]) -> None:
    watch_paths = _watch_utils.collect_watch_paths(stages_to_run)  # ← Fixed at startup
    # ...
    for changes in watchfiles.watch(*watch_paths, ...):  # ← Never updates
```

**Impact**: When new stages are added with new dependencies in different directories, those directories are NOT watched until restart.

**Example**:
- Initial pipeline watches `/data/raw/`
- User adds new stage depending on `/external/api_data/`
- Changes to `/external/api_data/` are not detected until full restart

---

## Proposed Architecture: Phase-Aware Watch Filtering

### Design Principles

1. **Separate execution phases**: IDLE vs EXECUTING
2. **Track execution timing**: Know when stages started/ended
3. **Filter based on state**: Permissive when idle, restrictive when running
4. **Dynamic filter updates**: Recreate filter after registry reloads

### Implementation Strategy

#### Phase 1: Fix Intermediate File Detection (Issue #141)

**Approach**: **State-based filtering** (simplest, most robust)

```python
class ReactiveEngine:
    _execution_state: Literal["IDLE", "EXECUTING"] = "IDLE"
    _watch_filter_factory: Callable[[], WatchFilter]  # Not the filter itself

    def _create_watch_filter(self) -> WatchFilter:
        """Create filter that respects execution state."""
        if self._execution_state == "IDLE":
            # Permissive: allow ALL file changes (including outputs)
            return lambda change, path: not _is_pycache(path)
        else:
            # Restrictive: block stage outputs to prevent feedback loops
            return _watch_utils.create_watch_filter(self._stages_to_run)
```

**Modification Points**:

1. **Add execution state tracking** to `ReactiveEngine`:
   ```python
   self._execution_state = "IDLE"  # or "EXECUTING"
   ```

2. **Update state in coordinator loop**:
   ```python
   def _coordinator_loop(self):
       # ... detect changes ...

       self._execution_state = "EXECUTING"
       try:
           results = self._execute_stages(affected)
       finally:
           self._execution_state = "IDLE"
   ```

3. **Make filter state-aware** in `_watch_utils.py`:
   ```python
   def create_watch_filter(
       stages_to_run: list[str],
       execution_state: Callable[[], str],  # Function returning current state
       watch_globs: list[str] | None = None,
   ) -> Callable[[Change, str], bool]:
       """Create filter that checks execution state before blocking outputs."""
       outputs_to_filter = get_output_paths_for_stages(stages_to_run)

       def watch_filter(change: Change, path: str) -> bool:
           # Always filter bytecode
           if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
               return False

           # When IDLE, allow all changes (including external output modifications)
           if execution_state() == "IDLE":
               # Apply glob filters if specified
               if watch_globs:
                   ...apply glob logic...
               return True

           # When EXECUTING, block stage outputs to prevent infinite loops
           resolved_path = project.try_resolve_path(path)
           if resolved_path is None:
               return True

           for out in outputs_to_filter:
               if resolved_path == out or out in resolved_path.parents:
                   return False  # Block outputs during execution

           return True

       return watch_filter
   ```

**Critical Timing**:
- State must transition to IDLE **AFTER** all file writes complete
- Debouncing ensures we collect the burst of execution-related changes
- Next debounce cycle starts in IDLE state → external changes pass through

**Trade-offs**:
- ✅ Simple to implement and reason about
- ✅ No timestamp tracking complexity
- ✅ Handles all external modification scenarios
- ⚠️ Small window where execution-caused changes could trigger re-run (mitigated by debouncing)

---

#### Phase 2: Fix TUI Stage Awareness (Issue #2)

**Changes**:

1. **Pass execution_order to TUI in watch mode**:

   File: `/home/user/pivot/src/pivot/cli/run.py:164-200`
   ```python
   def _run_watch_with_tui(...):
       graph = registry.REGISTRY.build_dag(validate=True)
       execution_order = dag.get_execution_order(graph, stages_list, single_stage)

       # ... setup executor and queues ...

       engine = reactive_module.ReactiveEngine(...)

       # FIX: Pass stage names to TUI
       run_tui.run_watch_tui(
           engine,
           tui_queue,
           stage_names=execution_order,  # ← ADD THIS
           output_queue=output_queue,
           tui_log=tui_log
       )
   ```

2. **Update run_watch_tui signature**:

   File: `/home/user/pivot/src/pivot/tui/run.py:684-692`
   ```python
   def run_watch_tui(
       engine: ReactiveEngineProtocol,
       message_queue: mp.Queue[TuiMessage],
       stage_names: list[str] | None = None,  # ← ADD THIS
       output_queue: mp.Queue[OutputMessage] | None = None,
       tui_log: Path | None = None,
   ) -> None:
       app = WatchTuiApp(
           engine,
           message_queue,
           stage_names=stage_names,  # ← PASS THIS
           output_queue=output_queue,
           tui_log=tui_log
       )
       app.run()
   ```

3. **Update WatchTuiApp.__init__**:

   File: `/home/user/pivot/src/pivot/tui/run.py:570-577`
   ```python
   class WatchTuiApp(_BaseTuiApp[None]):
       def __init__(
           self,
           engine: ReactiveEngineProtocol,
           message_queue: mp.Queue[TuiMessage],
           stage_names: list[str] | None = None,  # ← ADD THIS
           output_queue: mp.Queue[OutputMessage] | None = None,
           tui_log: Path | None = None,
       ) -> None:
           super().__init__(
               message_queue,
               stage_names=stage_names,  # ← PASS THIS
               tui_log=tui_log
           )
           self._engine = engine
           self._output_queue = output_queue
   ```

**Impact**: TUI shows all stages immediately with PENDING status, updating to RAN/SKIPPED as execution progresses.

---

#### Phase 3: Fix Dynamic Filter Updates (Issues #3 & #4)

**Problem**: Watch filter and watch paths are static closures that never update.

**Solution**: Make them dynamically queryable.

**Approach A: Callable Filter Factory** (Recommended)

Instead of passing a filter closure to `watchfiles.watch()`, make the filter ask the engine for current state:

```python
class ReactiveEngine:
    _stages_to_run: list[str]  # Updated on reload
    _execution_state: Literal["IDLE", "EXECUTING"]

    def _watch_loop(self, initial_stages: list[str]) -> None:
        self._stages_to_run = initial_stages

        # Create filter that queries engine state (not a closure)
        def dynamic_filter(change: Change, path: str) -> bool:
            return self._should_watch_path(change, path)

        for changes in watchfiles.watch(
            *self._get_watch_paths(),  # Method, not cached value
            watch_filter=dynamic_filter,
            stop_event=self._shutdown,
        ):
            # ... handle changes ...

    def _should_watch_path(self, change: Change, path: str) -> bool:
        """Evaluate filter using current state (called for each file change)."""
        # Always filter bytecode
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
            return False

        # When IDLE, allow all changes
        if self._execution_state == "IDLE":
            return True

        # When EXECUTING, block current stage outputs
        outputs = _watch_utils.get_output_paths_for_stages(self._stages_to_run)
        resolved = project.try_resolve_path(path)
        if resolved is None:
            return True

        for out_str in outputs:
            out = project.try_resolve_path(out_str)
            if out and (resolved == out or out in resolved.parents):
                return False

        return True

    def _get_watch_paths(self) -> list[pathlib.Path]:
        """Get current watch paths (recalculated on demand)."""
        return _watch_utils.collect_watch_paths(self._stages_to_run)

    def _reload_registry(self) -> bool:
        """Reload registry and update stages_to_run."""
        # ... existing reload logic ...

        if reload_ok:
            # Update stages list for dynamic filter
            graph = registry.REGISTRY.build_dag(validate=True)
            self._stages_to_run = dag.get_execution_order(
                graph, self._stages, single_stage=self._single_stage
            )
            # Filter and watch paths automatically update on next query
```

**Caveat**: `watchfiles.watch()` doesn't support dynamic watch paths. The `watch_paths` parameter is only read at startup.

**Solution for Watch Paths**:
- Accept this limitation for now (requires restart if new stage depends on new directory)
- OR: Implement custom watcher that can add paths dynamically
- OR: Always watch project root recursively (performance impact)

**Recommended**: Accept limitation, document in known issues. Dynamic watch paths require significant refactoring of watchfiles integration.

**Focus**: Fix dynamic filter (more critical), defer dynamic watch paths.

---

#### Phase 4: Refactor for Separation of Concerns (Future)

**Current Issues**:
- ReactiveEngine has TUI-specific code (`_send_reload_notification`)
- Tight coupling between watch loop, coordinator, and execution
- Complex multiprocessing setup order requirements

**Proposed**: Event-driven architecture

```python
class ReactiveEngine:
    """Core reactive engine - emits events, no UI coupling."""

    def __init__(self, event_callback: Callable[[ReactiveEvent], None]):
        self._on_event = event_callback

    def _reload_registry(self) -> bool:
        # ... reload logic ...
        if reload_ok:
            self._on_event(RegistryReloadedEvent(stages=new_stages))
        return reload_ok

class TuiEventHandler:
    """TUI-specific event handler."""

    def __init__(self, tui_queue: mp.Queue):
        self._tui_queue = tui_queue

    def handle_event(self, event: ReactiveEvent) -> None:
        match event:
            case RegistryReloadedEvent(stages):
                self._tui_queue.put(TuiReloadMessage(...))
            case ExecutionStartedEvent(...):
                # ... update UI ...
```

**Benefits**:
- Clean separation: engine doesn't know about TUI
- Easy to add new event consumers (JSON output, webhooks, etc.)
- Testable in isolation

**Timeline**: Post-MVP, requires significant refactoring

---

## Implementation Plan

### Phase 1: Immediate Fixes (Required for Issue #141)

1. ✅ **Add execution state tracking to ReactiveEngine**
   - Files: `src/pivot/reactive/engine.py`
   - Add `_execution_state: Literal["IDLE", "EXECUTING"]` field
   - Update state in `_coordinator_loop()` around `_execute_stages()`

2. ✅ **Make watch filter state-aware**
   - Files: `src/pivot/reactive/_watch_utils.py`, `src/pivot/reactive/engine.py`
   - Modify `create_watch_filter()` to accept `execution_state` callback
   - Update filter logic: allow ALL when IDLE, block outputs when EXECUTING

3. ✅ **Update tests**
   - Files: `tests/reactive/test_engine.py`, `tests/reactive/test_watch_utils.py`
   - Add tests for state-based filtering
   - Test external modification scenarios

### Phase 2: TUI Stage Awareness (User Experience)

4. ✅ **Pass execution_order to TUI in watch mode**
   - Files: `src/pivot/cli/run.py`, `src/pivot/tui/run.py`
   - Update `_run_watch_with_tui()` to pass `stage_names=execution_order`
   - Update `run_watch_tui()` and `WatchTuiApp.__init__()` signatures

5. ✅ **Update TUI tests**
   - Files: `tests/tui/test_run.py`
   - Ensure stages display immediately in watch mode

### Phase 3: Dynamic Filter Updates (Correctness)

6. ✅ **Refactor to dynamic filter**
   - Files: `src/pivot/reactive/engine.py`
   - Replace static filter closure with method: `_should_watch_path()`
   - Update `_stages_to_run` after registry reload
   - Filter automatically queries current state

7. ✅ **Update and fix existing failing test**
   - Files: `tests/reactive/test_engine.py`
   - Fix `test_watch_filter_stale_after_registry_adds_new_stage`
   - Add test for removed stages

### Phase 4: Documentation & Cleanup

8. ✅ **Update documentation**
   - Files: `docs/architecture/reactive.md`, `CLAUDE.md`
   - Document execution phases
   - Document watch filter behavior
   - Add known limitations (watch paths not dynamic)

9. ✅ **Code cleanup**
   - Remove obsolete comments
   - Ensure type hints are complete
   - Run full linting: `ruff format . && ruff check . && basedpyright .`

10. ✅ **Full test suite**
    - Run: `pytest tests/ -n auto`
    - Ensure 90%+ coverage maintained

---

## Testing Strategy

### Unit Tests

1. **State-based filter behavior**:
   ```python
   def test_watch_filter_allows_outputs_when_idle():
       """Filter should allow output modifications when in IDLE state."""

   def test_watch_filter_blocks_outputs_when_executing():
       """Filter should block output modifications when EXECUTING."""
   ```

2. **Dynamic filter updates**:
   ```python
   def test_filter_updates_after_registry_reload():
       """Filter should recognize new stage outputs after reload."""
       # This will fix the existing failing test
   ```

3. **Intermediate file detection**:
   ```python
   def test_external_modification_to_intermediate_file():
       """External modification to intermediate file should trigger downstream stages."""
   ```

### Integration Tests

1. **Watch mode with intermediate file modification**:
   - Create pipeline: `stage_a` → `intermediate.txt` → `stage_b` → `final.txt`
   - Start watch mode
   - Wait for initial run to complete (IDLE state)
   - Externally modify `intermediate.txt`
   - Assert: `stage_b` re-runs

2. **TUI shows all stages immediately**:
   - Start watch mode with TUI
   - Assert: All stages appear in TUI before execution starts
   - Assert: Stages show PENDING status initially

3. **Dynamic stage addition in watch mode**:
   - Start watch mode
   - Add new stage to pipeline.py
   - Wait for reload
   - Assert: New stage outputs are filtered
   - Assert: TUI shows new stage

---

## Risk Assessment

### Low Risk
- **TUI stage awareness fix**: Simple parameter passing, well-isolated change
- **Execution state tracking**: Clear state machine, easy to reason about

### Medium Risk
- **State-based filtering**: Requires careful timing around debouncing
  - Mitigation: Extensive testing of timing edge cases
  - Mitigation: Debouncing naturally creates clean phase separation

### High Risk
- **Dynamic filter updates**: Complex interaction between threads
  - Mitigation: Use instance methods instead of closures (cleaner state)
  - Mitigation: Leverage Python's GIL for atomic reads of simple types

### Known Limitations (Accepted)
- **Watch paths not dynamic**: New directories not watched until restart
  - Workaround: Restart watch mode after adding stages with new dirs
  - Future: Consider recursive project root watching (performance trade-off)

---

## Success Criteria

1. ✅ Issue #141 resolved: External modifications to intermediate files trigger re-execution
2. ✅ TUI shows all stages immediately in watch mode
3. ✅ Watch filter updates when registry reloads (new/removed stages)
4. ✅ Existing test `test_watch_filter_stale_after_registry_adds_new_stage` passes
5. ✅ All existing tests continue to pass
6. ✅ Code quality: passes `ruff format`, `ruff check`, `basedpyright`
7. ✅ Coverage: maintains 90%+ test coverage

---

## Timeline Estimate

- **Phase 1** (Issue #141): 4-6 hours
  - Implementation: 2-3 hours
  - Testing: 2-3 hours

- **Phase 2** (TUI): 1-2 hours
  - Implementation: 30 minutes
  - Testing: 30 minutes - 1 hour

- **Phase 3** (Dynamic filter): 3-4 hours
  - Implementation: 2 hours
  - Testing: 1-2 hours

- **Phase 4** (Documentation): 1 hour

**Total: 9-13 hours** (approximately 1-2 days)

---

## Open Questions

1. **Debounce interaction**: Should debounce duration increase when in EXECUTING state?
   - Current: Fixed 300ms
   - Proposal: Keep simple, existing debounce is sufficient

2. **Filter granularity**: Should we track per-stage execution state?
   - Current proposal: Global IDLE/EXECUTING
   - Alternative: Track which stages are running, only block their outputs
   - Recommendation: Start simple (global state), optimize later if needed

3. **Watch paths dynamic updates**: Worth the complexity?
   - Requires significant watchfiles refactoring
   - Recommendation: Defer to future, document limitation

4. **Event-driven refactor**: Priority?
   - Cleaner architecture but requires extensive refactoring
   - Recommendation: Post-MVP, separate project

---

## Conclusion

The root cause of Issue #141 is the **lack of execution phase awareness** in the watch filter. The filter statically blocks all outputs forever, with no concept of idle vs executing state.

The fix is straightforward: **state-based filtering** where the filter behavior changes based on execution phase:
- **IDLE**: Allow all changes (permissive) - enables external modifications
- **EXECUTING**: Block stage outputs (restrictive) - prevents infinite loops

Additional improvements (TUI stage awareness, dynamic filter updates) follow naturally from this architectural change.

The implementation is low-to-medium risk with clear success criteria and comprehensive testing strategy.
