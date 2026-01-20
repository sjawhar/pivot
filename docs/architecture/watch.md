# Watch Execution Engine

The watch execution engine provides continuous pipeline monitoring and automatic re-execution when dependencies change.

## Overview

Unlike batch execution (`pivot run`), watch mode keeps the pipeline running and automatically responds to file changes:

```bash
pivot run --watch
```

The engine monitors:

- **Stage function code** - Python files (`.py`) defining stages
- **Input data files** - Files declared as `deps`
- **Configuration files** - `pivot.yaml`, `pivot.yml`, `pipeline.py`, `params.yaml`, `params.yml`, `.pivotignore`

When changes are detected, only affected stages and their downstream dependencies re-run.

## JSON Output Mode

For IDE integrations and automation, watch mode supports JSON output:

```bash
pivot run --watch --json
```

This emits newline-delimited JSON (JSONL) events to stdout:

| Event Type | Description | Fields |
|------------|-------------|--------|
| `status` | Status messages | `message`, `is_error` |
| `files_changed` | File change detection | `paths`, `code_changed` |
| `affected_stages` | Stages that will run | `stages`, `count` |
| `execution_result` | Stage execution results | `stages` (map of name → status/reason) |

Example output:

```json
{"type": "status", "message": "Running initial pipeline...", "is_error": false}
{"type": "files_changed", "paths": ["src/train.py"], "code_changed": true}
{"type": "affected_stages", "stages": ["train", "evaluate"], "count": 2}
{"type": "execution_result", "stages": {"train": {"status": "ran", "reason": "code changed"}}}
{"type": "status", "message": "Watching for changes...", "is_error": false}
```

When `--json` is enabled, the TUI is disabled and all output goes to stdout as JSON events.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          WATCH ENGINE                                │
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
│  │  1. Collect and debounce changes                                 │
│  │  2. If code changed: reload registry, restart workers            │
│  │  3. Determine affected stages                                    │
│  │  4. Execute stages (BLOCKING)                                    │
│  │  5. Report results to TUI                                        │
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

When Python files change, the worker pool is restarted rather than hot-reloaded via `_restart_worker_pool()` in `src/pivot/watch/engine.py`.

**Why restart instead of hot reload?**

Hot reload via `importlib.reload()` has fundamental issues:

- **Import staleness**: Modules that import the reloaded module still have old references
- **cloudpickle caching**: May serve cached pickles with old code
- **Module-level side effects**: Re-execute on reload

Instead of hot-reloading, the engine performs a **full module clear** via `_clear_project_modules()`:

1. Removes all project modules from `sys.modules`
2. Deletes `.pyc` bytecode files from `__pycache__/` directories
3. Calls `importlib.invalidate_caches()`

Worker restart then starts fresh Python interpreters that reimport everything cleanly.

### 3. Bounded Queue with Coalescing

The change queue is bounded to prevent memory exhaustion during long executions. Changes coalesce in the watcher thread if the queue is full—they accumulate locally until the queue has room.

If pending changes exceed 10,000 files, a "full rebuild" sentinel replaces the change set. This prevents unbounded memory growth when many files change at once (e.g., `git checkout` switching branches). The sentinel triggers code reload and full pipeline re-evaluation, letting the executor's change detection determine which stages actually need to run.

### 4. Debouncing with Maximum Wait

Debouncing prevents rapid file saves from triggering multiple runs. The coordinator waits for a quiet period after the last change before processing:

- **Quiet period:** 300ms default, configurable via `--debounce` CLI flag (in milliseconds)
- **Maximum wait:** 5 seconds (non-configurable) prevents indefinite blocking during continuous saves (e.g., auto-save editors)

If changes keep arriving within the quiet period, the coordinator accumulates them until either the quiet period elapses with no new changes, or the 5-second maximum is reached.

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

Stage outputs are filtered from the watcher to prevent infinite loops. The `OutputFilter` class in `src/pivot/watch/_watch_utils.py` tracks output paths and filters them during execution.

**Execution state tracking:** The filter uses execution state (running vs idle) to distinguish Pivot-written outputs from external modifications. Outputs are only filtered while Pivot is actively executing stages—during the window between `mark_execution_start()` and `mark_execution_end()`. Changes to output files while idle are treated as external modifications and trigger re-runs of downstream stages.

## Error Handling

### Execution Errors

Stage execution errors are displayed in the TUI without stopping the watch loop:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stages (3) ●1 ✗1 ⊘1                   │  train ✗                    │
│  ─────────────────────────────────────┼──────────────────────────────│
│    ● preprocess         0.5s           │  [12:34:56] CUDA error...   │
│    ✗ train              1.2s           │  [12:34:57] out of memory   │
│    ⊘ evaluate                          │                              │
│                                        │                              │
│  Watching for changes...               │                              │
└─────────────────────────────────────────────────────────────────────┘
```

Fix the error and save - the pipeline automatically re-runs.

### Invalid Pipeline Errors

If code changes make the pipeline invalid (syntax errors, circular dependencies), the error is displayed in the TUI title bar and the system waits for a fix. The previous valid stage list remains visible until the error is resolved.

## Graceful Shutdown

The watcher thread uses a `stop_event` passed to `watchfiles.watch()` for clean shutdown.

On `Ctrl+C`:

1. Shutdown flag is set
2. Current execution completes (not interrupted)
3. Watcher thread exits cleanly
4. Resources are released

## Performance Characteristics

| Operation | Latency |
|-----------|---------|
| File change detection | <50ms (watchfiles Rust layer) |
| Debounce quiet period | 300ms default (`--debounce` to configure) |
| Worker restart | ~300ms (process spawn + imports) |
| Total code change → execution start | ~500ms |

## Limitations

- **Worker restart latency**: Code changes have ~300ms overhead for worker restart
- **No cancellation**: Long-running stages cannot be interrupted mid-execution
- **Single machine**: Not designed for distributed execution
- **Memory**: Long-running watch sessions should be restarted periodically
- **Intermediate file detection gap**: See below

### Intermediate File Detection Gap

External changes to files that are both outputs and downstream inputs are **not detected** by watch mode.

**Why this happens:** Stage outputs are filtered from the watcher to prevent infinite loops (stage runs → writes output → triggers watch → stage runs again). This filtering applies to ALL outputs, including those that are also inputs to downstream stages.

**Example scenario:**

```
preprocess → data/clean.csv → train
```

If an external tool (not Pivot) modifies `data/clean.csv`, watch mode won't detect it because `data/clean.csv` is filtered as an output of `preprocess`.

**Workaround:** Force a re-run with `pivot run --force` or modify an upstream input file to trigger the pipeline.

## Future Work

See [GitHub Issue #110: Hot Reload Exploration](https://github.com/sjawhar/pivot/issues/110) for exploration of faster code change handling via `importlib.reload()`.

## See Also

- [Watch Mode Reference](../reference/watch.md) - User guide for watch mode
- [Watch Mode Tutorial](../tutorial/watch.md) - Getting started with watch mode
- [Execution Model](./execution.md) - Batch execution architecture
