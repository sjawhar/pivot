# Watch Mode Architecture

Watch mode provides continuous pipeline monitoring and automatic re-execution when dependencies change.

## Overview

Watch mode uses the same Engine as batch mode, with a FilesystemSource for continuous event production:

```bash
pivot run --watch
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
│  │  watchfiles.watch()                                              │
│  │       │                                                          │
│  │       ▼                                                          │
│  │  engine.submit(event)                                            │
│  └──────────────────┘                                               │
│           │                                                          │
│           │ Event Queue (thread-safe)                               │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ run(exit_on_completion=False)  ← Processes events until shutdown │
│  │                  │                                               │
│  │  1. Handle DataArtifactChanged → run affected stages             │
│  │  2. Handle CodeOrConfigChanged → reload registry, run all        │
│  │  3. Handle CancelRequested → stop starting new stages            │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Unified Architecture

The same Engine code handles both batch and watch mode:

| Mode | Entry Point | Event Source |
|------|-------------|--------------|
| Batch (`pivot run`) | `engine.run(exit_on_completion=True)` | OneShotSource |
| Watch (`pivot run --watch`) | `engine.run(exit_on_completion=False)` | FilesystemSource |

This unified architecture eliminates divergent code paths between batch and watch modes. Both modes use identical sink configuration, ensuring flags like `--quiet` work consistently.

## Event Flow

### Data Artifact Changes

When a dependency file changes:

1. FilesystemSource emits `DataArtifactChanged(paths=[...])`
2. Engine queries bipartite graph for affected stages
3. Engine executes affected stages and their downstream dependencies
4. StageStarted/StageCompleted events emitted to sinks

### Code or Config Changes

When Python files or `pivot.yaml` change:

1. FilesystemSource emits `CodeOrConfigChanged(paths=[...])`
2. Engine invalidates caches and reloads registry
3. Engine rebuilds bipartite graph
4. Engine updates FilesystemSource watch paths
5. Engine re-runs all stages

### Output Filtering

Stage outputs are filtered to prevent infinite loops. The Engine tracks stage execution state:

- Outputs of PREPARING/RUNNING stages are filtered
- Changes are deferred and processed after COMPLETED

## JSON Output Mode

For IDE integrations and automation:

```bash
pivot run --watch --json
```

This uses a JsonlSink instead of TuiSink, emitting newline-delimited JSON events:

| Event Type | Description |
|------------|-------------|
| `stage_start` | Stage began execution (stage, index, total) |
| `stage_complete` | Stage finished (stage, status, reason, duration_ms, index, total) |

Note: JsonlSink translates internal engine events to the existing `pivot run --json` format for backwards compatibility. Other engine events (state changes, log lines, pipeline reloads) are not emitted in JSON mode.

## Worker Pool Management

### Code Change Handling

When Python files change, the worker pool is restarted via `loky.get_reusable_executor(kill_workers=True)`:

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

Workers stay alive between execution waves (when no code changes). Expensive imports (numpy, pandas) only happen once per worker restart.

## Debouncing

Changes are debounced to prevent rapid file saves from triggering multiple runs:

- **Quiet period:** Configurable via `--debounce` CLI flag (default 300ms)
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

- [Watch Mode Reference](../reference/watch.md) - User guide for watch mode
- [Watch Mode Tutorial](../tutorial/watch.md) - Getting started with watch mode
- [Execution Model](./execution.md) - Batch execution architecture
