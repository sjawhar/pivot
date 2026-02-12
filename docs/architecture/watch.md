# Watch Mode Architecture

Watch mode provides continuous pipeline monitoring and automatic re-execution when dependencies change.

## Overview

Watch mode uses the same Engine as batch mode, with a FilesystemSource for continuous event production:

```bash
pivot repro --watch
```

The Engine monitors:

- **Stage function code** - Python files (`.py`) defining stages
- **Input data files** - Files declared as `deps`
- **Configuration files** - `pivot.yaml`, `pivot.yml`, `pipeline.py`, `params.yaml`, `params.yml`, `.pivotignore`

When changes are detected, only affected stages and their downstream dependencies re-run.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            ENGINE                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │ FilesystemSource │  ← Watches via watchfiles (Rust-backed)       │
│  │                  │                                               │
│  │  watchfiles.awatch()                                             │
│  │       │                                                          │
│  │       ▼                                                          │
│  │  send(event)                                                     │
│  └──────────────────┘                                               │
│           │                                                          │
│           │ Input channel (anyio)                                   │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ run(exit_on_completion=False)  ← Processes events until shutdown │
│  │  + WatchCoordinator policy                                       │
│  │                                                                  │
│  │  1. DataArtifactChanged → compute affected stages                │
│  │  2. CodeOrConfigChanged → reload registry, run all               │
│  │  3. CancelRequested → stop starting new stages                   │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Unified Architecture

The same Engine code handles both batch and watch mode:

| Mode | Entry Point | Event Source |
|------|-------------|--------------|
| Batch (`pivot repro`) | `engine.run(exit_on_completion=True)` | OneShotSource |
| Watch (`pivot repro --watch`) | `engine.run(exit_on_completion=False)` | FilesystemSource |

This unified architecture eliminates divergent code paths between batch and watch modes. Both modes use identical sink configuration, ensuring flags like `--quiet` work consistently.

## Event Flow

### Data Artifact Changes

When a dependency file changes:

1. FilesystemSource emits `DataArtifactChanged(paths=[...])`
2. WatchCoordinator computes affected stages from bipartite graph
3. Engine executes affected stages and their downstream dependencies
4. StageStarted/StageCompleted events emitted to sinks

### Code or Config Changes

When Python files or `pivot.yaml` change:

1. FilesystemSource emits `CodeOrConfigChanged(paths=[...])`
2. Engine invalidates caches and reloads registry
3. Engine rebuilds bipartite graph and updates WatchCoordinator
4. Engine updates FilesystemSource watch paths
5. Engine re-runs all stages

### Output Filtering

Stage outputs are filtered to prevent infinite loops. The Engine tracks stage execution state:

- Outputs of PREPARING/WAITING_ON_LOCK/RUNNING stages are filtered
- Changes are deferred and processed after COMPLETED

## JSON Output Mode

For IDE integrations and automation:

```bash
pivot repro --watch --json
```

This uses a JsonlSink to emit newline-delimited JSON events:

| Event Type | Description |
|------------|-------------|
| `stage_start` | Stage began execution (stage, index, total) |
| `stage_complete` | Stage finished (stage, status, reason, duration_ms, index, total) |

Note: JsonlSink translates internal engine events to the existing `pivot repro --json` format for backwards compatibility. Other engine events (state changes, log lines, pipeline reloads) are not emitted in JSON mode.

## Worker Pool Management

### Code Change Handling

When Python files change, the worker pool is restarted via `executor_core.restart_workers()`:

**Why restart instead of hot reload?**

Hot reload via `importlib.reload()` has fundamental issues:

- **Import staleness**: Modules that import the reloaded module still have old references
- **cloudpickle caching**: May serve cached pickles with old code
- **Module-level side effects**: Re-execute on reload

The Engine performs a **full module clear** before restart:

1. Removes all project modules from `sys.modules`
2. Calls `importlib.invalidate_caches()`
3. Restarts workers with fresh Python interpreters

### Warm Workers

Workers are warm within a single run (stages reuse the same pool). Watch mode creates a fresh pool per run; code reloads can trigger a restart via `executor_core.restart_workers()` (only in parallel mode, per WatchCoordinator policy).

## Debouncing

Changes are debounced to prevent rapid file saves from triggering multiple runs:

- **Quiet period:** Configurable via `--debounce` CLI flag or `watch.debounce` config
- **Maximum wait:** 5 seconds (prevents indefinite blocking during continuous saves)

## Error Handling

### Execution Errors

Stage failures don't stop the watch loop. Fix the error and save - the pipeline automatically re-runs.

### Invalid Pipeline Errors

Syntax errors or circular dependencies are reported and the Engine waits for a fix. The previous valid stage list remains until the error is resolved.

## Graceful Shutdown

On `Ctrl+C`:

1. Engine sets shutdown flag
2. Current execution completes (not interrupted)
3. FilesystemSource stops watching
4. Resources are released

## Performance

| Operation | Latency |
|-----------|---------|
| File change detection | <50ms (watchfiles Rust layer) |
| Debounce quiet period | 300ms default |
| Worker restart | ~300ms (process spawn + imports) |
| Total code change → execution start | ~500ms |

## Limitations

- **Worker restart latency**: Code changes have ~300ms overhead for worker restart
- **No mid-stage cancellation**: Running stages complete before cancellation takes effect
- **Single machine**: Not designed for distributed execution

## See Also

- [Watch Mode Guide](../guides/watch-mode.md) - User guide for watch mode
- [Execution Model](./execution.md) - Batch execution architecture
