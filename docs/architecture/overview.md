# Architecture Overview

Pivot is designed for high-performance pipeline execution with automatic code change detection.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Pipeline Code (Python decorators)                     │
│  @stage(deps=['data.csv'], outs=['model.pkl'])              │
│  def train(lr: float = 0.01): ...                          │
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

### Scheduler

Coordinates parallel execution:

- Greedy scheduling - runs stages as soon as dependencies complete
- Mutex handling - prevents concurrent execution of conflicting stages
- Ready queue - tracks which stages can run

### Executor

Runs stages in worker processes:

- Uses `ProcessPoolExecutor` with `forkserver` context
- Warm workers with preloaded imports
- True parallelism (not limited by GIL)

### Lock Files

Per-stage lock files enable fast, parallel writes:

```yaml
# .pivot/stages/train.lock
code_manifest:
  func:train: "abc123"
  func:helper: "def456"
params:
  learning_rate: 0.01
deps:
  - path: data.csv
    hash: "789abc..."
outs:
  - path: model.pkl
    hash: "012def..."
dep_generations: {}
```

## Data Flow

1. **Registration** - `@stage` decorator registers function with metadata
2. **Discovery** - CLI discovers pipeline (pivot.yaml or pipeline.py)
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
└── state.lmdb/          # LMDB database for hash caching
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

**Problem:** Manual `deps: [file.py]` declarations are error-prone and tedious.

**Solution:** Automatic detection using:

- `inspect.getclosurevars()` for closure dependencies
- AST parsing for `module.function` patterns
- Recursive fingerprinting for transitive dependencies

### Warm Worker Pool

**Problem:** Importing numpy/pandas takes seconds per stage.

**Solution:** `loky.get_reusable_executor()` keeps workers alive across calls:

```python
# loky maintains a pool of reusable workers
executor = loky.get_reusable_executor(max_workers=4)

# First run: workers import numpy, pandas (~10s total)
executor.submit(stage_func)

# Second run: imports already loaded (~1s)
executor.submit(stage_func)
```

Workers persist between `pivot run` calls in watch mode, eliminating repeated import overhead.

### Trie for Path Validation

**Problem:** Simple string matching can't detect path overlaps (`data/` vs `data/train.csv`).

**Solution:** Prefix trie data structure (pygtrie) validates path declarations:

- Detects when a file is inside a declared directory
- Prevents conflicting output declarations
- O(k) lookup where k is path depth

## Module Organization

| Module | Responsibility |
|--------|----------------|
| `registry` | Stage registration and validation |
| `executor` | Parallel stage execution |
| `fingerprint` | Code change detection |
| `cache` | Content-addressable storage |
| `lock` | Per-stage lock file management |
| `state` | LMDB database for hash caching |
| `dag` | Dependency graph construction |
| `parameters` | Pydantic parameter handling |
