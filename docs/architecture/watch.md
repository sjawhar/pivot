# Watch Mode Architecture

Watch mode provides continuous pipeline monitoring and automatic re-execution when dependencies change.

## Overview

Watch mode uses the same Engine as batch mode, with a FilesystemSource for continuous event production:

```bash
pivot repro --watch
```

The Engine monitors:

- **Stage function code** - Python files (`.py`) defining stages
- **Input data files** - Files declared with `pipeline.input()`
- **Configuration files** - `pipeline.py`, `params.yaml`, `params.yml`, `.pivotignore` (`engine/sources.py` `_CONFIG_FILE_NAMES`)

When changes are detected, only affected stages and their downstream dependencies re-run. A
change inside a `pipeline.input()` directory at a path the project's `.pivotignore` names affects
no stage, since the directory's hash leaves it out (`WatchCoordinator._is_ignored_input_change`,
through the engine's one `ignore.IgnoreFilter`). A change inside a stage's directory output is
never left out: a downstream stage depends on that output's whole hash.

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
| Watch (`pivot repro --watch`) | `engine.run(exit_on_completion=False)` | OneShotSource (startup run) + FilesystemSource |

This unified architecture eliminates divergent code paths between batch and watch modes. Both modes use identical sink configuration, ensuring flags like `--quiet` work consistently.

## Event Flow

### Startup

A watched engine fingerprints every stage before it starts any other source or handles any input event, because a later reload diffs the edited pipeline against those fingerprints. Only the RPC server starts first, so a TUI or agent connects at once. While the snapshot runs, `run` and `commit` are refused at once with error `-32004` ("Engine is starting"), and `explain`, which fingerprints its stage, waits for it. The startup run and the file watcher start after it.

### Data Artifact Changes

When a dependency file changes:

1. FilesystemSource emits `DataArtifactChanged(paths=[...])` for a batch of changes that touches no code or config file
2. WatchCoordinator computes affected stages from bipartite graph
3. Engine executes affected stages and their downstream dependencies
4. StageStarted/StageCompleted events emitted to sinks

A change that arrives while a run is in progress cancels that run and queues one restart. The restart covers the cancelled run's stages and the new change's stages (all stages if either was all), so it never narrows what was going to run. A change that arrives while a run is still cancelling (a running stage has not finished) starts nothing alongside it: it joins the queued restart. An explicit RPC `cancel`, whether it comes before or after those changes, keeps the cancelled run's own stages out of the restart, so only the changes replay. If a code reload was deferred to the end of that run, the restart is folded into the reload's re-run, without stages the reload removed.

### Code or Config Changes

When Python files or `pipeline.py` change:

1. FilesystemSource emits `CodeOrConfigChanged(paths=[...], data_paths=[...])`, one event for the whole batch: `data_paths` holds the batch's other changed files
2. Engine invalidates caches and reloads registry
3. Engine rebuilds bipartite graph and updates WatchCoordinator
4. Engine updates FilesystemSource watch paths
5. Engine re-runs all stages

The files in `data_paths` changed before the reload starts. If the reload starts a re-run, that re-run checks them and they start no run of their own: an editor that saves several buffers at once, or a checkout that changes `pipeline.py` and an input together, runs each stage once. If the reload starts no re-run, `data_paths` is handled as a data change once the reload ends: a pipeline left invalid drops it with a debug log (see below), and a valid pipeline with no stages finds no stage to run. If a run is in progress, the reload waits for that run to end, and `data_paths` is handled at once as a data change.

A reload re-imports `pipeline.py`, fingerprints every stage and checks `params.yaml` in one daemon thread, so the RPC server, the TUI's event polling and cancellation keep running. That thread does every `sys.modules` change the reload makes (dropping project modules, importing the edited ones); the event loop then swaps in the new pipeline, emits `PipelineReloaded` and starts the re-run. Until the re-run has started, `run` and `commit` are refused at once with error `-32004` ("Engine is reloading"), and `explain` waits for the new pipeline. Reloads run one at a time. Ctrl-C stops watch mode at once, even while an import is slow or never returns: the engine abandons the reload thread, and interpreter shutdown does not wait for a daemon thread. Cancelling the `FilesystemSource` also sets its watcher's stop event, so the watch thread ends within one poll step instead of at its 5 s timeout.

A pipeline is invalid when it fails to load (a syntax error, no `pipeline` variable), a stage fails to fingerprint, or `params.yaml` fails its checks (it does not parse, a key names no stage, a field is not one of the stage's params, or a value is one its params field rejects), whether the startup snapshot finds it or an edit introduces it. A run marks the pipeline invalid too, without running anything, when it loads a broken `params.yaml` before the watcher reports the edit, when a stage requests more of a resource than its capacity, or when a missing input leaves it no stage to run (`Stage '<stage>' depends on input '<name>' (path <path>), which does not exist`), as when an edit adds a stage before the file it reads exists. The Engine then logs `Pipeline invalid: <error>` and `Pipeline invalid - waiting for fix`, and emits `PipelineReloaded` with `error` set and the stages of the pipeline it still holds. While the pipeline is invalid, nothing runs and nothing is committed: RPC `run` and `commit` are refused with error `-32003`, and data changes and the startup run are dropped with a debug log, unless an input is missing (below). Data edited meanwhile therefore still differs from the stage locks, so the next edit that loads a good pipeline clears the error, reloads and reruns those stages.

A missing input is fixed by creating the file. Every input lies inside the watched project root, so the watcher reports the creation as a data change, though a file that arrives inside a new directory (`mkdir -p` then a write, `mv` of a tree, `cp -r`, `tar -x`) is reported as the directory alone, which maps to no stage. While a missing input keeps the pipeline invalid, the Engine decides admission once after filtering the data-event batch: it checks the inputs again only if at least one path is the missing input or a directory that would contain it. Every other batch, including one that maps to an existing stage, is handled as when the pipeline is invalid on main: its run request is ignored. For an input reached through an existing symlink in the project, the Engine compares both the declared path and the physical path with existing symlink components resolved, so moving the input tree into either spelling's directory also checks the input. Any other change, such as the session's own output written inside the project, is handled as when the pipeline is valid. If missing inputs still leave no stage to run, the Engine reports the pipeline invalid again and runs nothing; otherwise it emits `PipelineReloaded` with `error` unset and runs the stages it can, skipping the rest with a warning. That run covers the stages of the run the missing input stopped, with the changed files' stages.

### Output Filtering

Stage outputs are filtered to prevent infinite loops. The Engine tracks stage execution state:

- Outputs of PREPARING/WAITING_ON_LOCK/RUNNING stages are filtered
- Changes are deferred and processed after COMPLETED

## JSON Output Mode

For IDE integrations and automation:

```bash
pivot repro --watch --json
```

`pivot repro` first prints a `schema_version` line, then a JsonlSink emits newline-delimited JSON events on stdout:

| Event Type | Description |
|------------|-------------|
| `schema_version` | First line (version) |
| `stage_start` | Stage began execution (stage, index, total) |
| `stage_complete` | Stage finished (stage, status, reason, duration_ms, index, total) |
| `stages_expanded` | A mapped stage's template (`pipeline.map`) added its instances to the run (template, stages: the instance names `<template>@<key>` in key order). It comes before any of those instances starts; the `total` of later `stage_start`/`stage_complete` events counts them |
| `engine_state_changed` | The Engine became `active` or `idle` (state) |
| `pipeline_reloaded` | The pipeline was reloaded, or became invalid or valid again (stages, stages_added, stages_removed, stages_modified, error, seq, run_id). `error` is set while the pipeline is invalid and null once it is valid, including when it becomes valid again without a reload |

Note: JsonlSink renames the engine's `stage_started` and `stage_completed` to the existing `pivot repro --json` names for backwards compatibility, forwards `stages_expanded` with only its template and stages, and forwards `pipeline_reloaded` as the Engine emits it. Log lines (on stderr) and the other engine events are not emitted in JSON mode.

## Worker Pool Management

### Code Change Handling

When Python files change, `WorkerPool.restart()` replaces the workers, starting the reloaded
run's width of fresh workers together (see [Pool Size](execution.md#pool-size)). Across all
pipelines with `--keep-going`, the reload re-runs only the stages a missing dependency leaves
runnable:

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

One watch Engine keeps its workers warm across re-runs. Before reusing those workers, Pivot
checks the user-code modules they imported and replaces the pool when a source file changed.
Code reloads also replace workers so they execute the reloaded pipeline.

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

On `Ctrl+C` at the terminal:

1. Running stages are interrupted mid-stage: each gets `WorkerTerminated` (a `KeyboardInterrupt`), its cleanup runs, and nothing it produced is committed (see [Stopping mid-stage](./execution.md#stopping-mid-stage))
2. Pivot waits up to 10 s for the workers to exit, then kills any left with their child processes
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
- **Stage-level cancellation over RPC**: an RPC `cancel` lets running stages complete; only `Ctrl+C` at the terminal, or the death of the Pivot process, stops them mid-stage
- **Single machine**: Not designed for distributed execution

## See Also

- [Watch Mode Guide](../guides/watch-mode.md) - User guide for watch mode
- [Execution Model](./execution.md) - Batch execution architecture
