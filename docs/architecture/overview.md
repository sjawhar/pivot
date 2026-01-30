# Architecture Overview

Pivot is designed for high-performance pipeline execution with automatic code change detection.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (pivot.yaml + typed Python functions)   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage Registry → Bipartite Graph → Engine                   │
│  Automatic fingerprinting | Artifact-Stage DAG | Event loop │
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

Edges point **consumer → producer** (the stage that USES an artifact points to the stage that PRODUCES it):

```
preprocess (produces data/clean.csv)
    ↑
  train (consumes data/clean.csv)
```

This convention may seem counter-intuitive, but it enables natural execution ordering: `nx.dfs_postorder_nodes()` returns `[preprocess, train]` without needing to reverse the graph.

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

Coordinates parallel execution:

- Greedy scheduling - runs stages as soon as dependencies complete
- Mutex handling - prevents concurrent execution of conflicting stages
- Ready queue - tracks which stages can run

### Engine

The Engine is the central coordinator for all execution paths. It:

- Processes input events (file changes, run requests, cancellation)
- Manages stage execution state machine
- Emits output events (stage started/completed, log lines)
- Maintains the bipartite artifact-stage graph

All code paths (CLI run, watch mode, agent RPC) route through the Engine.

### Event Sources

Sources produce input events:

- **FilesystemSource** - Watches files via watchfiles, emits `DataArtifactChanged` and `CodeOrConfigChanged`
- **OneShotSource** - Emits single `RunRequested` for batch mode

### Event Sinks

Sinks consume output events for display:

- **ConsoleSink** - Rich-formatted terminal output
- **TuiSink** - Forwards to Textual TUI
- **JsonlSink** - Newline-delimited JSON for tooling integration
- **WatchSink** - Handles engine state changes for watch mode

### Executor

Runs stages in worker processes:

- Uses `ProcessPoolExecutor` with `spawn` context
- Warm workers with preloaded imports
- True parallelism (not limited by GIL)

### Lock Files

Per-stage lock files (`.pivot/stages/<name>.lock`) enable fast, parallel writes. Each lock file records:

- **Code manifest** - Hashes of the stage function and its transitive dependencies
- **Parameters** - Current parameter values
- **Dependency hashes** - Content hashes of input files (with manifests for directories)
- **Output hashes** - Content hashes of output files
- **Dependency generations** - Generation counters for O(1) skip detection

Lock files use relative paths for portability across machines.

## Data Flow

1. **Discovery** - CLI discovers pipeline (pivot.yaml)
2. **Registration** - Stages registered from YAML config
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

**Solution:** `loky.get_reusable_executor()` keeps workers alive across calls. The first stage execution imports heavy dependencies; subsequent stages reuse those imports. Workers persist between execution waves in watch mode, eliminating repeated import overhead.

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
