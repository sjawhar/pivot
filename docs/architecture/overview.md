# Architecture Overview

Pivot is designed for high-performance pipeline execution with automatic code change detection.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (pipeline.py or pivot.yaml)             │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage Registry → Bipartite Graph → Engine (Coordinator)     │
│  Automatic fingerprinting | Scheduler | WatchCoordinator     │
└─────────────────────────────────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌──────────┐      ┌──────────┐      ┌──────────┐
│  Event   │      │  Event   │      │  Warm    │
│  Sources │      │  Sinks   │      │  Workers │
│ (Input)  │      │ (Output) │      │ (Exec)   │
└──────────┘      └──────────┘      └──────────┘
```

## Core Components

### Stage Registry

The `StageRegistry` maintains all registered stages:

- Validates stage definitions
- Builds dependency graph
- Stores stage metadata (function, deps, outs, params)

### Bipartite Graph

The Engine maintains a bipartite graph with two node types:

- **Artifact nodes** - Files (dependencies and outputs)
- **Stage nodes** - Python functions

```
[data.csv] ──→ [preprocess] ──→ [cleaned.csv] ──→ [train] ──→ [model.pkl]
 (artifact)      (stage)         (artifact)       (stage)      (artifact)
```

This graph enables:

1. **Execution** - "This file changed → which stages need to run?"
2. **Queries** - "What would run if I executed now?"

The legacy stage-only DAG is derived from the bipartite graph via `get_stage_dag()`.

#### Edge Direction

The bipartite graph follows data flow:

- **Artifact → Stage** for dependencies (consumes)
- **Stage → Artifact** for outputs (produces)

When extracting the stage-only DAG, edges are reversed to **consumer → producer** so `nx.dfs_postorder_nodes()` yields `[preprocess, train]` without an extra reverse step:

```
preprocess (produces data/clean.csv)
    ↑
  train (consumes data/clean.csv)
```

#### Path Resolution

The graph builder uses two strategies to match dependencies to outputs:

1. **Exact match (O(1)):** Dictionary lookup via `_build_outputs_map()`
   - Maps each output path to its producing stage
   - Handles the common case of explicit path dependencies

2. **Directory overlap (O(log n)):** pygtrie prefix tree for parent/child relationships
   - `has_subtrie()`: Dependency is parent of outputs (`data/` depends on `data/file.csv`)
   - `shortest_prefix()`: Dependency is child of output (`data/file.csv` depends on `data/`)

This handles cases where a stage declares a directory output and another stage depends on a file within that directory (or vice versa).

### Scheduler

Coordinates deterministic scheduling (sync, no IO):

- Maintains per-stage execution state
- Tracks upstream/downstream completion
- Enforces mutex groups (including exclusive `*`)
- Decides which stages are eligible to start

Implemented in `packages/pivot/src/pivot/engine/scheduler.py` and owned by the Engine.

### WatchCoordinator

Watch-mode policy planner extracted from the Engine:

- Computes affected stages for changed paths
- Filters events for outputs produced by in-flight stages
- Decides whether worker pools should restart after code reloads

Implemented in `packages/pivot/src/pivot/engine/watch.py`.

### Engine

The Engine is the async coordinator for all execution paths. It:

- Processes input events (file changes, run requests, cancellation)
- Delegates scheduling decisions to `Scheduler`
- Manages run lifecycle and stage state transitions
- Emits output events (stage started/completed, log lines)
- Owns the bipartite artifact-stage graph and `WatchCoordinator`

All code paths (CLI run, watch mode, agent RPC) route through the Engine.

### Event Sources

Sources produce input events:

- **FilesystemSource** - Watches files via watchfiles, emits `DataArtifactChanged` and `CodeOrConfigChanged`
- **OneShotSource** - Emits single `RunRequested` for batch mode
- **AgentRpcSource** - Receives JSON-RPC commands (`run`, `cancel`) and emits input events

### Event Sinks

Sinks consume output events for display:

- **StaticConsoleSink / LiveConsoleSink** - Rich-formatted terminal output (buffered for CI, live for TTY)
- **ResultCollectorSink** - Collects `StageCompleted` events for programmatic access
- **BroadcastEventSink** - Pub-sub event delivery for connected agents
- **EventBuffer** - Ring buffer for `events_since` polling (RPC clients)
- **JsonlSink** - Newline-delimited JSON for tooling integration (CLI helper)

### Executor

Runs stages in worker processes:

- `WorkerPool` wraps a loky `ProcessPoolExecutor` with `spawn` context
- Uses a manager-backed output queue for worker log streaming
- Warm workers with preloaded imports (reusable executor)
- True parallelism (not limited by GIL)

### Lock Files

Per-stage lock files (`.pivot/stages/<name>.lock`) enable fast, parallel writes. Each lock file records:

- **Code manifest** - Hashes of the stage function and its transitive dependencies
- **Parameters** - Current parameter values
- **Dependency hashes** - Content hashes of input files (with manifests for directories)
- **Output hashes** - Content hashes of output files

Lock files use relative paths for portability across machines.

## Data Flow

1. **Discovery** - CLI discovers pipeline (pipeline.py or pivot.yaml)
2. **Registration** - Stages registered from Python code (or YAML config)
3. **DAG Construction** - Build dependency graph from outputs/inputs
4. **Fingerprinting** - Hash code, params, and dependency content
5. **Comparison** - Compare fingerprints with lock files
6. **Scheduling** - Determine execution order respecting dependencies
7. **Execution** - Run stages in parallel workers
8. **Caching** - Store outputs in content-addressable cache
9. **Lock Update** - Write new fingerprints to lock files

## Cache Structure

```
.pivot/
├── cache/
│   └── files/           # Content-addressable storage
│       ├── ab/
│       │   └── cdef...  # Files keyed by xxhash64
│       └── ...
├── stages/              # Per-stage lock files
│   ├── preprocess.lock
│   └── train.lock
├── config.yaml          # Remote configuration
└── state.lmdb/          # LMDB database (hash cache, generations, run cache, remote index)
```

## Key Design Decisions

### Per-Stage Lock Files

**Problem:** DVC writes entire `dvc.lock` on every stage completion (O(n²) overhead).

**Solution:** Each stage writes only its own lock file. Parallel writes without contention.

### Content-Addressable Cache

Files are stored by their content hash:

- Deduplication across stages
- Fast restoration via hardlinks
- Simple remote synchronization

### Automatic Code Fingerprinting

**Problem:** Manual code dependency declarations are error-prone and tedious.

**Solution:** Automatic detection using:

- `inspect.getclosurevars()` for closure dependencies
- AST parsing for `module.function` patterns
- Recursive fingerprinting for transitive dependencies

### Warm Worker Pool

**Problem:** Importing numpy/pandas takes seconds per stage.

**Solution:** `loky.get_reusable_executor()` keeps workers alive across stage executions within a run. The first stage execution imports heavy dependencies; subsequent stages reuse those imports. In watch mode, each run creates a fresh pool, and code reloads trigger an explicit restart via `executor_core.restart_workers()`.

### Trie for Path Validation

**Problem:** Simple string matching can't detect path overlaps (`data/` vs `data/train.csv`).

**Solution:** Prefix trie data structure (pygtrie) validates path declarations:

- Detects when a file is inside a declared directory
- Prevents conflicting output declarations
- O(k) lookup where k is path depth

## See Also

- [Engine Architecture](engine.md) - Event-driven architecture, sources, sinks, and API
- [Execution Model](execution.md) - Parallel execution, skip detection, caching
- [Watch Mode](watch.md) - Continuous pipeline monitoring
- [Code Tour](code-tour.md) - Navigate the codebase
