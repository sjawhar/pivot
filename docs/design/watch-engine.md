# Watch Engine Design

> **Note:** This design document describes the original WatchEngine implementation, which has been superseded by the unified Engine architecture. See [Watch Mode Architecture](../architecture/watch.md) for the current implementation.

---

*This document captures the design decisions and trade-offs for the watch execution engine.*

## Summary

Replace the current separate watch mode with a unified watch execution engine that continuously monitors all pipeline dependencies and automatically triggers stage re-execution when changes are detected.

## Motivation

The current watch mode (`pivot run --watch`) works but has limitations:
- Separate code path from normal execution
- Full pipeline re-run on any change
- No TUI integration for live status

A watch engine provides:
- Seamless development experience
- Only affected stages re-run
- Live TUI showing pipeline state
- Automatic code change handling

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          WATCH ENGINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │  Watcher Thread  │  ← Pure producer, never blocks                │
│  │  (watchfiles)    │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼ Queue (bounded, thread-safe)                            │
│                                                                      │
│  ┌──────────────────┐                                               │
│  │ Coordinator Loop │                                               │
│  │                  │                                               │
│  │  1. Collect & debounce changes                                   │
│  │  2. If code change: restart workers                              │
│  │  3. Determine affected stages                                    │
│  │  4. executor.run() [BLOCKING]                                    │
│  │  5. Report to TUI                                                │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Watcher Thread** | Monitors filesystem via watchfiles (Rust-backed), enqueues change events |
| **Change Queue** | Thread-safe bounded queue connecting watcher to coordinator |
| **Coordinator Loop** | Debounces changes, triggers worker restart on code changes, determines affected stages, runs executor |
| **Executor** | Runs stages in worker pool (unchanged from batch mode) |

## Key Design Decisions

### 1. Worker Restart Instead of Hot Reload

When Python code changes, **restart the worker pool** rather than using `importlib.reload()`:

```python
self._pool = loky.get_reusable_executor(
    max_workers=self._max_workers,
    kill_workers=True  # Terminates existing workers
)
```

**Why?** Hot reload via `importlib.reload()` has known reliability issues:

- Import staleness (modules importing the reloaded module keep old references)
- cloudpickle caching (may serve pickles with old code)
- Module-level side effects re-executing

Worker restart avoids these issues by starting fresh Python interpreters.

### 2. Blocking Executor for Natural Serialization

The coordinator blocks on `executor.run()`, which means:
- Changes accumulate in queue during execution
- Code changes processed between execution waves
- No complex coordination logic needed

This is the key architectural insight: **the blocking call naturally serializes execution waves**, eliminating entire classes of race conditions.

### 3. Bounded Queue with Coalescing

Prevent memory exhaustion during long executions:

```python
def _watch_loop(self) -> None:
    pending: set[Path] = set()

    for changes in watchfiles.watch(
        *self._watch_paths,
        stop_event=self._shutdown
    ):
        pending.update(Path(c[1]) for c in changes)

        try:
            self._change_queue.put_nowait(pending)
            pending = set()
        except queue.Full:
            pass  # Keep accumulating
```

### 4. Debounce with Maximum Wait

```python
def _collect_and_debounce(self, max_wait_s: float = 5.0) -> set[Path]:
    changes: set[Path] = set()
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        try:
            batch = self._change_queue.get(timeout=0.1)
            changes.update(batch)
        except queue.Empty:
            if changes:
                return changes  # Quiet period done

    return changes  # Hit deadline
```

### 5. Output Filtering

Filter all registered stage outputs from watcher to prevent infinite loops.

## Implementation Plan

### Phase 1: Core Engine
- [x] Create `src/pivot/watch/` module structure
- [x] Implement `WatchEngine` class with watcher thread
- [x] Implement bounded queue with coalescing
- [x] Implement coordinator loop with debouncing
- [x] Add worker pool restart on code changes

### Phase 2: Change Detection
- [x] Determine affected stages from changed paths
- [x] Recompute fingerprints after worker restart
- [x] Output filtering (prevent loops)

### Phase 3: Error Handling
- [x] Display execution errors without stopping watch loop
- [x] Handle invalid pipeline (syntax errors, circular dependencies)
- [x] Show error in TUI, keep last valid state, wait for fix

### Phase 4: TUI Integration
- [x] Live status display (stages, states)
- [x] Keyboard shortcuts (`f` force run, `q` quit)
- [x] Error overlay for pipeline errors

### Phase 5: Polish
- [x] Graceful shutdown (stop_event for watcher)
- [x] Structured logging
- [x] Documentation

## Edge Cases

| Case | Handling |
|------|----------|
| File changed during hash | Debounce handles most cases |
| Long execution + many changes | Bounded queue with coalescing |
| Watcher shutdown during quiet period | Use `stop_event` parameter |
| Continuous file changes (logs) | Max debounce timeout (5s) |
| Invalid pipeline after code change | Show error, keep last valid state |

## CLI Usage

```bash
# Basic watch mode
pivot run --watch

# With TUI (default when TTY)
pivot run --watch --display tui

# Plain text output
pivot run --watch --display plain
```

## Performance Targets

| Operation | Target Latency |
|-----------|----------------|
| File change detection | <50ms |
| Debounce quiet period | 100-300ms |
| Worker restart | ~300ms |
| Total code change → execution start | ~500ms |

## Related Issues

- #102 - `--force` flag (works with watch mode via keyboard shortcut)
- #110 - Hot Reload Exploration

## References

- [Architecture doc](/docs/architecture/watch.md)
- [Buck2 incremental computation](https://engineering.fb.com/2023/04/06/open-source/buck2-open-source-large-scale-build-system/)
- [watchfiles library](https://github.com/samuelcolvin/watchfiles)
- [loky reusable executor](https://loky.readthedocs.io/)
