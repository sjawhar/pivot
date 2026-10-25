# Architecture Overview

Pivot is designed for high-performance pipeline execution with automatic code change detection.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (pipeline.py: @stage + Pipeline)        │
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

#### Dependency Resolution

The graph builder matches each dependency to its producer by artifact identity: `_build_outputs_map()` maps every output's `ArtifactIdentity` (producing stage plus output key) to that stage, and a dependency handle carries the identity of the output it was created from. A dependency on one file of a directory output carries its own identity (the directory's key with the file's path appended) and hangs from the directory's node (`types.source_identity`), so the edge exists while its lock records only that file. A dependency with no producer is a pipeline input, resolved to its bound workspace path by the store. Paths play no part in matching, so two stages can never be linked by an accidental path prefix. Paths still decide where outputs land, so `StageRegistry.add_existing` refuses an output whose workspace path (`storage/store.py` `relative_output_path`) equals, contains, or lies inside another registered output's path.

### Scheduler

Coordinates deterministic scheduling (sync, no IO):

- Maintains per-stage execution state
- Tracks upstream/downstream completion
- Holds per-stage resource permits until completion
- Decides which stages are eligible to start

Each stage holds one CPU unit by default. A decorator can request additional resources with
`@stage(resources={"gpu": 1})`; non-CPU resources default to capacity 1 and can be configured
per run. Implemented in `packages/pivot/src/pivot/engine/scheduler.py` and owned by the Engine.

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
- Warm workers retain stage imports across runs in one Engine
- True parallelism (not limited by GIL)

### Lock Files

Per-stage lock files (`<state dir>/stages/<name>.lock`) enable fast, parallel writes. The state
directory comes from `core.state_dir` and defaults to `.pivot`. Each lock file records:

- **Code manifest** - Hashes of the stage function and its transitive dependencies
- **Parameters** - Current parameter values
- **Dependency hashes** - Content hashes of input files (with manifests for directories)
- **Output hashes** - Content hashes of output files

Lock files use relative paths for portability across machines.

## Data Flow

1. **Discovery** - CLI finds and runs `pipeline.py`
2. **Construction** - `@stage` calls inside `with Pipeline(...)` record stages; the block's exit builds the `StageRegistry`
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
└── config.yaml          # Project configuration

<state dir>/             # .pivot by default
├── cache/
│   └── files/           # Content-addressable storage
│       ├── ab/
│       │   └── cdef...  # Files keyed by xxhash64
│       └── ...
├── stages/              # Per-stage lock files, one directory per pipeline
│   └── demo/
│       ├── preprocess.lock
│       └── train.lock
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

**Problem:** Importing numpy/pandas takes seconds per worker.

**Solution:** `loky.get_reusable_executor()` keeps imports available for a worker's lifetime.
A run's first submission starts the run's resource width of workers, and later stages start
more only when every live worker is busy or a parent's consumers are about to be ready.
One-shot Engines release idle workers after the idle timeout. Long-lived Engines keep workers
warm between runs. Each run gets a fresh output queue, and Pivot replaces workers before a
later run when a user-code module they imported changed.

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
