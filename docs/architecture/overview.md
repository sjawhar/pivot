# Architecture Overview

Pivot is designed for high-performance pipeline execution with automatic code change detection.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (pivot.yaml + typed Python functions)   │
│  stages:                                                    │
│    train:                                                   │
│      python: stages.train                                   │
│      deps: {data: data.csv}                                 │
│      outs: {model: model.pkl}                               │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage Registry → DAG Builder → Scheduler                    │
│  Automatic fingerprinting | Topological sort | Ready queue  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Warm Workers / Interpreters                                 │
│  Preloaded numpy/pandas | True parallelism                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Per-Stage Lock Files (.pivot/stages/<name>.lock)        │
│  Code manifest | Params | Deps/Outs | Fast parallel writes │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Stage Registry

The `StageRegistry` maintains all registered stages:

- Validates stage definitions
- Builds dependency graph
- Stores stage metadata (function, deps, outs, params)

### DAG Builder

Constructs a directed acyclic graph from stage dependencies:

- Parses output paths to find implicit dependencies
- Validates no cycles exist
- Enables topological ordering

#### Edge Direction

Edges point **consumer → producer** (the stage that USES an artifact points to the stage that PRODUCES it):

```
preprocess (produces data/clean.csv)
    ↑
  train (consumes data/clean.csv)
```

This convention may seem counter-intuitive, but it enables natural execution ordering: `nx.dfs_postorder_nodes()` returns `[preprocess, train]` without needing to reverse the graph.

#### Path Resolution

The DAG builder uses two strategies to match dependencies to outputs:

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

