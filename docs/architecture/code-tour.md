# Code Tour

This guide maps Pivot's architectural concepts to actual file paths, helping you find where to start reading.

## Entry Points

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `pivot repro` | `packages/pivot/src/pivot/cli/repro.py` → `Engine.run(exit_on_completion=True)` | DAG-aware batch execution |
| `pivot run` | `packages/pivot/src/pivot/cli/run.py` → `Engine.run(exit_on_completion=True)` | Single-stage execution |
| `pivot list` | `packages/pivot/src/pivot/cli/list.py` | Stage listing |
| `pivot status --explain` | `packages/pivot/src/pivot/cli/status.py` → `status.get_pipeline_explanations()` | Change detection explanation |
| `pivot repro --watch` | `packages/pivot/src/pivot/cli/repro.py` → `Engine.run(exit_on_completion=False)` | Watch mode |

## Core Subsystems

### Pipeline Definition & Discovery

**Key files:**

- `packages/pivot/src/pivot/discovery.py` - Finds and loads `pipeline.py`
- `packages/pivot/src/pivot/compose.py` - `@stage`, `Pipeline`, `ArtifactHandle`, output format inference
- `packages/pivot/src/pivot/registry.py` - `StageRegistry` and `RegistryStageInfo` (built by `Pipeline` when its `with` block exits)
- `packages/pivot/src/pivot/stage_def.py` - `StageParams` and output-writing helpers

**How it works:**

1. `discovery.discover_pipeline()` finds `pipeline.py` in the current directory or the project root
2. It executes the file; each `@stage` call inside `with Pipeline(...)` records a node and returns an `ArtifactHandle`
3. On exit, `Pipeline._materialize_stages()` turns the nodes into `RegistryStageInfo` entries with `ArtifactRef` deps and outs (identities, formats, tags)
4. Discovery returns the module's `pipeline` variable

**Start reading:** `packages/pivot/src/pivot/compose.py:Pipeline`

### DAG Construction

**Key files:**

- `packages/pivot/src/pivot/engine/graph.py` - Builds bipartite artifact-stage graph from stages
- `packages/pivot/src/pivot/dag/` - DAG rendering (ASCII, DOT, Mermaid)

**How it works:**

1. `build_graph()` takes registered stages
2. Creates artifact nodes and stage nodes (bipartite graph)
3. Adds edges from each stage's output artifacts to the stages whose deps reference the same artifact identity
4. Returns a NetworkX DiGraph (use `get_execution_order()` for sorted order)

**Start reading:** `packages/pivot/src/pivot/engine/graph.py:build_graph()`

### Code Fingerprinting

**Key files:**

- `packages/pivot/src/pivot/fingerprint.py` - Main fingerprinting logic
- `packages/pivot/src/pivot/ast_utils.py` - AST manipulation helpers

**How it works:**

1. `get_stage_fingerprint()` starts from stage function
2. Inspects closure variables via `inspect.getclosurevars()`
3. Parses AST to find `module.function` patterns
4. Recursively fingerprints all dependencies
5. Normalizes and hashes the combined code

**Start reading:** `packages/pivot/src/pivot/fingerprint.py:get_stage_fingerprint()`

### Execution

**Key files:**

- `packages/pivot/src/pivot/engine/engine.py` - Central coordinator (Engine class)
- `packages/pivot/src/pivot/engine/graph.py` - Bipartite artifact-stage graph
- `packages/pivot/src/pivot/engine/types.py` - Event types and stage states
- `packages/pivot/src/pivot/engine/sources.py` - Event sources (FilesystemSource, OneShotSource)
- `packages/pivot/src/pivot/engine/sinks.py` - Event sinks (StaticConsoleSink, LiveConsoleSink, ResultCollectorSink)
- `packages/pivot/src/pivot/engine/worker_pool.py` - Engine-lifetime worker pool and queue lifecycle
- `packages/pivot/src/pivot/executor/core.py` - Reusable loky executor and child-process policy
- `packages/pivot/src/pivot/executor/worker.py` - Worker process code
- `packages/pivot/src/pivot/storage/store.py` - `WorkspaceStore`: maps artifact identities to workspace paths (`data/<pipeline>/<stage>.<ext>`)

**How it works:**

1. CLI creates Engine and registers sinks/sources
2. `Engine.run(exit_on_completion=True)` for batch mode; `Engine.run(exit_on_completion=False)` for watch mode
3. Engine builds bipartite graph and orchestrates execution
4. Workers execute stages via `worker.execute_stage()`
5. Lock files updated after each stage

**Start reading:** `packages/pivot/src/pivot/engine/engine.py:Engine.run()`

### Caching & Storage

**Key files:**

- `packages/pivot/src/pivot/storage/cache.py` - Content-addressable file cache (module-level functions)
- `packages/pivot/src/pivot/storage/lock.py` - Per-stage lock files
- `packages/pivot/src/pivot/storage/state.py` - LMDB state database
- `packages/pivot/src/pivot/storage/restore.py` - Restoring outputs from cache
- `packages/pivot/src/pivot/run_history.py` - Run cache entries and manifests

**How it works:**

1. `save_to_cache()` hashes and stores file contents by xxhash64
2. `LockFile` records stage fingerprint + output hashes
3. `StateDB` provides fast key-value storage for runtime state
4. `restore_from_cache()` retrieves cached files on cache hit
5. `run_history` manages run cache for skip detection across branches

**Start reading:** `packages/pivot/src/pivot/storage/cache.py:save_to_cache()`

### Engine Event System

**Key files:**

- `packages/pivot/src/pivot/engine/engine.py` - Event processing loop
- `packages/pivot/src/pivot/engine/types.py` - Input/output event definitions
- `packages/pivot/src/pivot/engine/sources.py` - Event producers
- `packages/pivot/src/pivot/engine/sinks.py` - Event consumers

**How it works:**

1. Sources submit events to Engine via `engine.submit()`
2. Engine processes events in `run()` (with `exit_on_completion` controlling batch vs watch mode)
3. Engine emits output events to registered sinks
4. Sinks handle display (TUI, console, JSON)

**Start reading:** `packages/pivot/src/pivot/engine/engine.py:Engine._handle_input_event()`

### TUI (pivot-tui package)

The TUI is a **pure RPC client** in a separate package (`pivot-tui`). It has zero imports of pivot runtime modules — only `pivot.types` is allowed.

**Key files:**

- `packages/pivot-tui/src/pivot_tui/run.py` - Main Textual app (`PivotApp`)
- `packages/pivot-tui/src/pivot_tui/client.py` - `PivotRpc`/`PivotClient` protocols
- `packages/pivot-tui/src/pivot_tui/rpc_client_impl.py` - `RpcPivotClient` (JSON-RPC 2.0 over Unix socket)
- `packages/pivot-tui/src/pivot_tui/event_poller.py` - `EventPoller` (polls events, converts to TUI messages)
- `packages/pivot-tui/src/pivot_tui/diff_panels.py` - Input/Output diff panel renderers
- `packages/pivot-tui/src/pivot_tui/widgets/` - UI components (stage list, panels, logs, debug)
- `packages/pivot-tui/src/pivot_tui/screens/` - Modal screens (help, history, confirm dialogs)
- `packages/pivot-tui/src/pivot_tui/testing/fake_server.py` - FakeRpcServer test double
- `packages/pivot/src/pivot/engine/agent_rpc.py` - JSON-RPC server (engine side)
- `packages/pivot/src/pivot/cli/_run_common.py` - 3-thread TUI launch coordinator

**How it works:**

1. CLI creates engine + `PivotApp`, passes `socket_path` to the app
2. `run_tui_with_engine()` coordinates 3 threads: engine, poller, and TUI
3. Engine thread starts RPC socket server; poller thread connects and polls events
4. TUI connects its own `RpcPivotClient` in `on_mount()` for UI commands
5. `EventPoller` converts engine events to typed `TuiStatusMessage`/`TuiLogMessage` etc.
6. Diff panels are pure renderers consuming server-provided explanation + output_summary data

**Start reading:** `packages/pivot-tui/src/pivot_tui/run.py:PivotApp`

### Remote Storage

**Key files:**

- `packages/pivot/src/pivot/remote/protocol.py` - `RemoteStorage` interface and `open_remote()` factory
- `packages/pivot/src/pivot/remote/storage.py` - S3 backend (`S3Remote`)
- `packages/pivot/src/pivot/remote/gcs.py` - GCS backend (`GCSRemote`)
- `packages/pivot/src/pivot/remote/keys.py` - Object key layout and download integrity checks both backends share
- `packages/pivot/src/pivot/remote/sync.py` - Push/pull logic
- `packages/pivot/src/pivot/remote/config.py` - Remote configuration and URL validation

**How it works:**

1. `open_remote()` picks `S3Remote` (`s3://`) or `GCSRemote` (`gs://`) behind the `RemoteStorage` interface
2. `push()` uploads cache files the remote lacks
3. `pull()` downloads missing cache files by hash, verifying each against its hash; the CLI then checks them out

**Start reading:** `packages/pivot/src/pivot/remote/sync.py:push()`

## Data Flow: `pivot repro`

```
CLI (run.py)
    │
    ▼
Discovery (discovery.py)
    │
    ▼
Pipeline (compose.py, built by running the user's pipeline.py)
    │
    ▼
Engine (engine/engine.py)
    │
    ├──► Build Graph (engine/graph.py)
    │
    ├──► Orchestrate Execution
    │         │
    │         ▼
    │    Worker Pool (executor/core.py)
    │         │
    │         ▼
    │    Workers (executor/worker.py)
    │         │
    │         ▼
    │    Stage Function
    │         │
    │         ▼
    │    Cache (storage/cache.py)
    │
    ├──► Emit Events to Sinks
    │
    ▼
Lock File Update (storage/lock.py)
```

## Key Design Patterns

### Module-Level Functions

All stage-related functions must be module-level for pickling. See `packages/pivot/src/pivot/fingerprint.py` for how we detect and handle this.

### Content-Addressable Storage

Files are stored by hash, enabling deduplication. See `packages/pivot/src/pivot/storage/cache.py:save_to_cache()`.

### Per-Stage Lock Files

Each stage has its own lock file for O(n) updates instead of O(n²). See `packages/pivot/src/pivot/storage/lock.py`.

### Reusable Worker Pool

Workers stay warm across executions. See `packages/pivot/src/pivot/executor/core.py` use of `loky.get_reusable_executor()`.

## Testing

Test structure mirrors source:

| Source | Tests |
|--------|-------|
| `packages/pivot/src/pivot/fingerprint.py` | `packages/pivot/tests/fingerprint/` |
| `packages/pivot/src/pivot/executor/` | `packages/pivot/tests/execution/test_executor.py` |
| `packages/pivot/src/pivot/cli/` | `packages/pivot/tests/integration/test_cli_*.py` |

See `tests/CLAUDE.md` for testing guidelines.

## Adding Features

### New CLI Command

1. Create `packages/pivot/src/pivot/cli/mycommand.py`
2. Use `@cli_decorators.pivot_command()` decorator
3. Add to `packages/pivot/src/pivot/cli/__init__.py`
4. Add tests in `packages/pivot/tests/integration/test_cli_mycommand.py`

See [CLI Development](../contributing/cli.md)

### New Loader Type

1. Add to `packages/pivot/src/pivot/loaders.py`
2. Choose the appropriate base class:
   - `Reader[R]` - read-only (for dependencies), implement `load() -> R`
   - `Writer[W]` - write-only (for outputs), implement `save(data: W, ...)`
   - `Loader[W, R]` - bidirectional, implement both; use `Loader[T]` when W == R (symmetric)
3. Implement required methods
4. Add tests

See [Adding Loaders](../contributing/loaders.md)

### New Output Tag

1. Add the tag to `packages/pivot/src/pivot/compose.py` (beside `metric`, `plot`, `incremental` and `directory`) and its `types.ArtifactTag`
2. Handle the tag where outputs are placed, written and cached (`storage/store.py`, `executor/worker.py`)
3. Add tests

## See Also

- [Architecture Overview](overview.md) - High-level design
- [Fingerprinting](fingerprinting.md) - How code tracking works
- [Execution Model](execution.md) - Parallel execution details
