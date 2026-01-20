# Code Tour

This guide maps Pivot's architectural concepts to actual file paths, helping you find where to start reading.

## Entry Points

| Command | Entry Point | Description |
|---------|-------------|-------------|
| `pivot run` | `src/pivot/cli/run.py` | Main execution flow |
| `pivot list` | `src/pivot/cli/list.py` | Stage listing |
| `pivot explain` | `src/pivot/cli/run.py:explain_cmd()` | Change detection explanation |
| `pivot run --watch` | `src/pivot/watch/engine.py` | Watch mode |

## Core Subsystems

### Pipeline Discovery & Registration

**Key files:**

- `src/pivot/discovery.py` - Auto-discovers `pivot.yaml`, `pipeline.py`
- `src/pivot/pipeline/yaml.py` - Parses `pivot.yaml` into internal structures
- `src/pivot/registry.py` - Global stage registry (`REGISTRY`)
- `src/pivot/stage_def.py` - Stage definition classes (`StageDef`, `StageParams`)

**How it works:**

1. `discovery.discover_and_register(project_root)` finds pipeline definition (defaults to current project root)
2. For YAML: `pipeline.yaml.register_from_pipeline_file()` parses and registers stages
3. For Python: module is imported, which calls `REGISTRY.register()`

**Start reading:** `src/pivot/discovery.py:discover_and_register()`

### DAG Construction

**Key files:**

- `src/pivot/dag.py` - Builds dependency graph from stages

**How it works:**

1. `build_dag()` takes registered stages
2. Creates nodes for each stage
3. Adds edges based on output→input path relationships
4. Returns a NetworkX DiGraph (use `dag.get_execution_order()` for sorted order)

**Start reading:** `src/pivot/dag.py:build_dag()`

### Code Fingerprinting

**Key files:**

- `src/pivot/fingerprint.py` - Main fingerprinting logic
- `src/pivot/ast_utils.py` - AST manipulation helpers

**How it works:**

1. `get_stage_fingerprint()` starts from stage function
2. Inspects closure variables via `inspect.getclosurevars()`
3. Parses AST to find `module.function` patterns
4. Recursively fingerprints all dependencies
5. Normalizes and hashes the combined code

**Start reading:** `src/pivot/fingerprint.py:get_stage_fingerprint()`

### Execution

**Key files:**

- `src/pivot/executor/core.py` - Main executor logic
- `src/pivot/executor/worker.py` - Worker process code
- `src/pivot/executor/commit.py` - Post-execution output handling
- `src/pivot/outputs.py` - Output type definitions (`Out`, `Metric`, `Plot`, `IncrementalOut`)

**How it works:**

1. `run()` function is the main entry point
2. Uses `loky.get_reusable_executor()` for warm worker pool
3. Workers execute stages via `worker.execute_stage()`
4. Lock files updated after each stage; `commit.commit_pending()` promotes deferred locks

**Start reading:** `src/pivot/executor/core.py:run()`

### Caching & Storage

**Key files:**

- `src/pivot/storage/cache.py` - Content-addressable file cache
- `src/pivot/storage/lock.py` - Per-stage lock files
- `src/pivot/storage/state.py` - LMDB state database
- `src/pivot/storage/restore.py` - Restoring outputs from cache
- `src/pivot/run_history.py` - Run cache entries and manifests

**How it works:**

1. `CacheStore` hashes and stores file contents by xxhash64
2. `LockFile` records stage fingerprint + output hashes
3. `StateDB` provides fast key-value storage for runtime state
4. `restore_outputs()` retrieves cached files on cache hit
5. `run_history` manages run cache for skip detection across branches

**Start reading:** `src/pivot/storage/cache.py:CacheStore`

### Watch Engine

**Key files:**

- `src/pivot/watch/engine.py` - Main watch loop
- `src/pivot/watch/_watch_utils.py` - Helper utilities (output filtering)
- `src/pivot/ignore.py` - `.pivotignore` file parsing

**How it works:**

1. `WatchEngine` starts watcher thread using `watchfiles`
2. Changes are debounced in coordinator loop
3. Code changes trigger worker pool restart
4. Affected stages are re-executed
5. `IgnoreFilter` excludes paths matching `.pivotignore` patterns

**Start reading:** `src/pivot/watch/engine.py:WatchEngine`

### TUI

**Key files:**

- `src/pivot/tui/run.py` - Main Textual app (`PivotApp`)
- `src/pivot/tui/widgets/` - UI components (stage list, panels, logs, debug)
- `src/pivot/tui/screens/` - Modal screens (help, history list, confirm dialogs)
- `src/pivot/tui/console.py` - Plain-text console output (non-TUI mode)
- `src/pivot/tui/agent_server.py` - JSON-RPC server for external control
- `src/pivot/tui/diff_panels.py` - Input/output diff visualization

**How it works:**

1. `PivotApp` is the main Textual application (supports both run and watch mode)
2. `StageListPanel` displays scrollable stage list with grouping
3. `TabbedDetailPanel` shows Logs/Input/Output tabs for selected stage
4. `RunDisplay` handles plain-text output in non-TUI mode
5. `AgentServer` provides JSON-RPC endpoint for programmatic control

**Start reading:** `src/pivot/tui/run.py:PivotApp`

### Remote Storage

**Key files:**

- `src/pivot/remote/storage.py` - S3 operations
- `src/pivot/remote/sync.py` - Push/pull logic
- `src/pivot/remote/config.py` - Remote configuration

**How it works:**

1. `RemoteStorage` abstracts S3 operations
2. `push_outputs()` uploads cache files to S3
3. `pull_outputs()` downloads cache files by hash

**Start reading:** `src/pivot/remote/sync.py:push_outputs()`

## Data Flow: `pivot run`

```
CLI (run.py)
    │
    ▼
Discovery (discovery.py)
    │
    ▼
Registry (registry.py) ◄── YAML Parser (pipeline/yaml.py)
    │
    ▼
DAG Builder (dag.py)
    │
    ▼
Executor (executor/core.py)
    │
    ├──► Worker Pool (executor/worker.py)
    │         │
    │         ▼
    │    Stage Function
    │         │
    │         ▼
    │    Commit Outputs (executor/commit.py)
    │         │
    │         ▼
    │    Cache (storage/cache.py)
    │
    ▼
Lock File Update (storage/lock.py)
```

## Key Design Patterns

### Module-Level Functions

All stage-related functions must be module-level for pickling. See `src/pivot/fingerprint.py` for how we detect and handle this.

### Content-Addressable Storage

Files are stored by hash, enabling deduplication. See `src/pivot/storage/cache.py:CacheStore.store()`.

### Per-Stage Lock Files

Each stage has its own lock file for O(n) updates instead of O(n²). See `src/pivot/storage/lock.py`.

### Reusable Worker Pool

Workers stay warm across executions. See `src/pivot/executor/core.py` use of `loky.get_reusable_executor()`.

## Testing

Test structure mirrors source:

| Source | Tests |
|--------|-------|
| `src/pivot/fingerprint.py` | `tests/fingerprint/` |
| `src/pivot/executor/` | `tests/execution/test_executor.py` |
| `src/pivot/cli/` | `tests/integration/test_cli_*.py` |

See `tests/CLAUDE.md` for testing guidelines.

## Adding Features

### New CLI Command

1. Create `src/pivot/cli/mycommand.py`
2. Use `@cli_decorators.pivot_command()` decorator
3. Add to `src/pivot/cli/__init__.py`
4. Add tests in `tests/integration/test_cli_mycommand.py`

See [CLI Development](../contributing/cli.md)

### New Loader Type

1. Add to `src/pivot/loaders.py`
2. Extend `Loader[T]` base class
3. Implement `load()` and `save()`
4. Add tests

See [Adding Loaders](../contributing/loaders.md)

### New Output Type

1. Add to `src/pivot/outputs.py`
2. Define handling in `src/pivot/executor/commit.py`
3. Add tests

## See Also

- [Architecture Overview](overview.md) - High-level design
- [Fingerprinting](fingerprinting.md) - How code tracking works
- [Execution Model](execution.md) - Parallel execution details
