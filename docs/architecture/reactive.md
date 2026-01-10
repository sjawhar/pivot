# Reactive Execution Engine

The reactive execution engine provides continuous pipeline monitoring and automatic re-execution when dependencies change.

## Overview

Unlike batch execution (`pivot run`), reactive mode keeps the pipeline running and automatically responds to file changes:

```bash
pivot run --reactive
```

The engine monitors:

- **Stage function code** - Python files defining stages
- **Input data files** - Files declared as `deps`
- **Configuration** - `params.yaml` and `pivot.yaml`

When changes are detected, only affected stages and their downstream dependencies re-run.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REACTIVE ENGINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │  Watcher Thread  │  ← Pure producer, never blocks                │
│  │                  │                                               │
│  │  watchfiles.watch()                                              │
│  │       │                                                          │
│  │       ▼                                                          │
│  │  change_queue.put(paths)                                         │
│  └──────────────────┘                                               │
│           │                                                          │
│           │ Queue (bounded, thread-safe)                            │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Coordinator Loop │  ← Orchestrates execution                     │
│  │                  │                                               │
│  │  while not shutdown:                                             │
│  │    changes = collect_and_debounce()                              │
│  │                                                                  │
│  │    if has_code_changes(changes):                                 │
│  │      restart_worker_pool()                                       │
│  │                                                                  │
│  │    stages = get_affected_stages(changes)                         │
│  │                                                                  │
│  │    if stages:                                                    │
│  │      results = executor.run(stages)  # BLOCKING                  │
│  │      report_to_tui(results)                                      │
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

### 1. Blocking Executor Serialization

The coordinator calls `executor.run()` which **blocks** until execution completes. This provides natural serialization:

- Changes accumulate in queue during execution
- Code changes can only be processed between execution waves
- No coordination logic needed to prevent mid-execution interference

```
Time →

Watcher:    [change1]     [change2]     [code.py]     [change4]
               │             │              │               │
               ▼             ▼              ▼               ▼
Queue:      accumulates while executor.run() blocks...
               │
Coordinator:   └──── collect batch ────┘
                      │
                      ▼
                   executor.run({affected stages})
                      │ (BLOCKING)
                      │
                      ▼
                   [done] → process next batch (including code.py)
```

### 2. Worker Restart on Code Changes

When Python files change, workers are restarted rather than hot-reloaded:

```python
def _restart_worker_pool(self) -> None:
    """Kill workers and spawn fresh ones that reimport all modules."""
    self._pool = loky.get_reusable_executor(
        max_workers=self._max_workers,
        kill_workers=True  # Terminates existing workers
    )
```

**Why restart instead of hot reload?**

| Approach | Reliability | Latency |
|----------|-------------|---------|
| `importlib.reload()` | ~80% (module staleness issues) | ~50ms |
| Worker restart | ~99% (fresh interpreter) | ~300ms |

Hot reload via `importlib.reload()` has fundamental issues:

- **Import staleness**: Modules that import the reloaded module still have old references
- **cloudpickle caching**: May serve cached pickles with old code
- **Module-level side effects**: Re-execute on reload

Worker restart avoids these by starting fresh Python interpreters that reimport everything.

### 3. Bounded Queue with Coalescing

The change queue is bounded to prevent memory exhaustion during long executions:

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

Changes coalesce in the watcher thread if the queue is full.

### 4. Debouncing with Maximum Wait

Debouncing prevents rapid file saves from triggering multiple runs:

```python
def _collect_and_debounce(self, max_wait_s: float = 5.0) -> set[Path]:
    """Collect changes with debounce, but don't wait forever."""
    changes: set[Path] = set()
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        try:
            batch = self._change_queue.get(timeout=0.1)
            changes.update(batch)
        except queue.Empty:
            if changes:
                return changes  # Quiet period, return batch

    return changes  # Hit deadline
```

The maximum wait prevents indefinite blocking if files continuously change (e.g., log rotation).

## Change Detection

### What Triggers Re-execution

| Change Type | Detection | Action |
|-------------|-----------|--------|
| **Stage code (.py)** | watchfiles event | Restart workers, run affected stages |
| **Helper functions (.py)** | watchfiles event | Restart workers, fingerprint check |
| **Input files (deps)** | watchfiles event | Run stages with changed deps |
| **params.yaml** | watchfiles event | Run stages with changed params |
| **Output files** | Filtered out | No action (prevents loops) |

### Output Filtering

Stage outputs are filtered from the watcher to prevent infinite loops:

```python
def _create_watch_filter(self) -> Callable[[str], bool]:
    """Filter that excludes all registered stage outputs."""
    outputs = set()
    for stage in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage)
        outputs.update(info["outs"])

    def should_watch(path: str) -> bool:
        return path not in outputs

    return should_watch
```

## Error Handling

### Execution Errors

Stage execution errors are displayed in the TUI without stopping the reactive loop:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stages:                                                             │
│    ✓ preprocess    [completed]  0.5s                                │
│    ✗ train         [failed]     error: CUDA out of memory           │
│    ○ evaluate      [skipped]    upstream failed                     │
│                                                                      │
│  Watching for changes...                                             │
└─────────────────────────────────────────────────────────────────────┘
```

Fix the error and save - the pipeline automatically re-runs.

### Invalid Pipeline Errors

If code changes make the pipeline invalid (syntax errors, circular dependencies), the error is displayed and the system waits for a fix:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠ PIPELINE ERROR                                                   │
│                                                                      │
│  SyntaxError in src/stages/train.py:42                              │
│    unexpected indent                                                 │
│                                                                      │
│  Pipeline paused. Fix the error and save to continue.               │
└─────────────────────────────────────────────────────────────────────┘
```

The system keeps the last valid configuration and resumes when the error is fixed.

## Graceful Shutdown

The watcher thread uses `stop_event` for clean shutdown:

```python
for changes in watchfiles.watch(
    *self._watch_paths,
    stop_event=self._shutdown  # Allows interruption
):
    ...
```

On `Ctrl+C`:

1. Shutdown flag is set
2. Current execution completes (not interrupted)
3. Watcher thread exits cleanly
4. Resources are released

## Performance Characteristics

| Operation | Latency |
|-----------|---------|
| File change detection | <50ms (watchfiles Rust layer) |
| Debounce quiet period | 100-300ms (configurable) |
| Worker restart | ~300ms (process spawn + imports) |
| Total code change → execution start | ~500ms |

## Limitations

- **Worker restart latency**: Code changes have ~300ms overhead for worker restart
- **No cancellation**: Long-running stages cannot be interrupted mid-execution
- **Single machine**: Not designed for distributed execution
- **Memory**: Long-running reactive sessions should be restarted periodically

## Future Work

See [GitHub Issue #110: Hot Reload Exploration](https://github.com/sjawhar/pivot/issues/110) for exploration of faster code change handling via `importlib.reload()`.

## See Also

- [Watch Mode](../guide/watch.md) - User guide for watch mode
- [Execution Model](./execution.md) - Batch execution architecture
