# Execution Model

Pivot uses a parallel execution model with warm worker pools for maximum performance.

## Execution Flow

```
┌──────────────┐
│ pivot repro  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Engine          │  ← Central coordinator
│  (run)           │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Build Bipartite │
│  Graph           │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Check           │
│  Fingerprints    │
│  vs Lock Files   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Engine          │
│  Orchestration   │──────────────┐
└──────┬───────────┘              │
       │                          │
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│  Worker 1    │          │  Worker N    │
│  (Process)   │   ...    │  (Process)   │
└──────┬───────┘          └──────┬───────┘
       │                          │
       ▼                          ▼
┌──────────────────────────────────────────┐
│  Per-Stage Lock Files                    │
│  (Parallel writes, no contention)        │
└──────────────────────────────────────────┘
```

## Engine Orchestration

The Engine uses event-driven orchestration for maximum parallelism:

1. **Stage States** - Each stage has its own state (PENDING → READY → PREPARING → RUNNING → COMPLETED)
2. **Ready Queue** - Stages with all dependencies satisfied
3. **Mutex Handling** - Prevents conflicting stages from running concurrently
4. **Event Emission** - StageStarted/StageCompleted events to sinks

As stages complete, their downstream stages become ready. The Engine handles both batch (`exit_on_completion=True`) and continuous (`exit_on_completion=False`) execution through the same orchestration code.

## Stage Execution States

Stages have individual states tracked by the Engine:

```python
class StageExecutionState(IntEnum):
    PENDING = 0      # Not yet considered
    BLOCKED = 1      # Upstream failed
    READY = 2        # Can run, waiting for worker
    PREPARING = 3    # Engine clearing outputs
    RUNNING = 4      # Stage function executing
    COMPLETED = 5    # Terminal
```

The IntEnum allows ordered comparisons (e.g., `state >= PREPARING` means execution has begun).

### State Transitions

```
PENDING ──(deps complete)──▶ READY ──(worker available)──▶ PREPARING ──▶ RUNNING ──▶ COMPLETED
    │                                                           │
    └──(upstream failed)──▶ BLOCKED                             └──(failed)──▶ COMPLETED
```

### Output Filtering by State

During watch mode, the Engine filters filesystem events based on stage state:

- **PREPARING**: Silence events for this stage's outputs (Engine is preparing them)
- **RUNNING**: Defer events for outputs (collect, don't act yet)
- **COMPLETED**: Process deferred events, compare output hashes, trigger downstream

## Worker Pool

Pivot uses `loky.get_reusable_executor()` for warm workers with `spawn` context.

### Why ProcessPoolExecutor?

- **True parallelism** - Not limited by Python's GIL
- **Isolation** - Each stage runs in its own process
- **Memory efficiency** - Workers can be recycled

### Why Spawn?

- **Safety** - Avoids fork() issues with threads (Python 3.13+ deprecates fork in multithreaded contexts)
- **Compatibility** - Works on macOS and Linux
- **Clean state** - Each worker starts fresh without inherited state

### Warm Workers

Workers stay alive between stages, so expensive imports (numpy, pandas) only happen once per worker, not once per stage.

## Mutex Handling

Mutex groups prevent concurrent execution:

```yaml
# pivot.yaml
stages:
  train_model_a:
    python: stages.train_model_a
    mutex:
      - gpu

  train_model_b:
    python: stages.train_model_b
    mutex:
      - gpu  # Won't run while train_model_a is running
```

### Exclusive Mutex

Use `mutex: ["*"]` to run a stage with no other stages executing concurrently:

```yaml
stages:
  database_migration:
    python: stages.migrate_db
    mutex:
      - "*"  # Runs exclusively - no other stages run at the same time
```

This is useful for stages that require exclusive access to shared resources like databases or file locks.

### Implementation

1. Track active mutex groups (including `*` for exclusive)
2. Before scheduling, check for conflicts
3. Wait for conflicting stages to complete

## Stage Execution

Each stage execution:

1. **Acquire Execution Lock** - Prevent concurrent execution of same stage
2. **Read Lock Data** - Get previous fingerprint and hashes
3. **Hash Dependencies** - Compute current dependency hashes
4. **Check Skip Conditions** - Compare fingerprints, params, deps
5. **Restore or Execute** - Either restore from cache or run function
6. **Cache Outputs** - Store in content-addressable cache
7. **Write Lock File** - Record new fingerprint and hashes
8. **Release Lock**

## Three-Tier Skip Detection

Pivot uses a three-tier skip detection system to minimize unnecessary work:

```
                Worker receives stage
                       │
                       ▼
          ┌────────────────────────┐
          │  Lookup dep hashes     │
          │  (StateDB cache: O(1)  │
          │   when metadata match) │
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
Tier 1:   │  Check generation O(1) │──── Match? ──▶ SKIP
          │  (StateDB lookup)      │
          └────────────┬───────────┘
                       │ No match
                       ▼
          ┌────────────────────────┐
Tier 2:   │  Compare lock file     │──── Unchanged? ─▶ SKIP
          │  (fingerprint+params+  │
          │   dep_hashes)          │
          └────────────┬───────────┘
                       │ Changed
                       ▼
          ┌────────────────────────┐
Tier 3:   │  Check run cache       │──── Hit? ─────▶ SKIP
          │  (input_hash → outputs)│                (restore from cache)
          └────────────┬───────────┘
                       │ No hit
                       ▼
                ┌──────────────┐
                │   EXECUTE    │
                │  stage func  │
                └──────────────┘
```

### Dependency Hash Lookup

Before skip checks, dependency hashes are looked up via StateDB. Each cached hash entry stores file metadata:

- `mtime_ns` - Modification time in nanoseconds
- `size` - File size in bytes
- `inode` - Filesystem inode number

When all three match the current file's `stat()` result, the cached hash is returned in O(1) without re-reading the file. The inode check detects file replacement (delete + create with same name), which keeps mtime/size but changes inode. Only files with changed metadata require actual hashing.

### Tier 1: Generation Check (O(1))

The fastest check uses monotonic generation counters to detect changes without comparing hashes:

- Each output file has a **generation counter** in StateDB, incremented every time that file is written
- When a stage runs, it **records the generation** of each dependency at that moment
- On next run: if `recorded_generation == current_generation` for all deps → nothing changed → skip

**Example:** Stage `train` depends on `data/clean.csv` (produced by `preprocess`).

1. `preprocess` runs, writes `data/clean.csv` → generation becomes 5
2. `train` runs, records `dep_generations: {"data/clean.csv": 5}`
3. Next run: `clean.csv` generation is still 5 → generations match → skip `train`
4. `preprocess` runs again → `clean.csv` generation becomes 6
5. Next run: `train` recorded 5, current is 6 → mismatch → fall through to Tier 2

This provides instant O(1) skip detection when nothing has changed.

**File metadata verification:** In addition to generation comparison, Tier 1 optionally verifies that cached file hashes still match the current file metadata (mtime, size, inode). This catches external modifications that generation tracking alone wouldn't detect—for example, if an external tool modified a dependency file without going through Pivot.

### Tier 2: Lock File Comparison

If generations don't match, perform a full comparison:

- Code fingerprint (AST hash of function + dependencies)
- Parameters (Pydantic model values)
- Dependency hashes (content hashes of all input files)

If all match the lock file → skip.

### Tier 3: Run Cache Lookup

If the current inputs differ from the lock file but match a previous execution:

- Compute `input_hash` from fingerprint + params + dep_hashes
- Look up in run cache: `input_hash → cached_outputs`
- If found → restore outputs from cache, skip execution

This enables skipping even when switching between branches or reverting changes.

## StateDB Architecture

StateDB is an LMDB-backed key-value store for all pipeline state. Keys use prefixes to namespace different data types:

| Prefix | Purpose |
|--------|---------|
| `hash:` | File hash cache (uses resolved paths, follows symlinks for deduplication) |
| `gen:` | Output generation counters (uses normalized paths, preserves symlinks) |
| `dep:` | Stage dependency generations (stage name + dep path) |
| `runcache:` | Run cache entries (stage name + input hash) |
| `run:` | Run history manifests |
| `remote:` | Remote index entries |

**Path handling strategies:** Hash keys use `resolve()` (follows symlinks) so that symlinked files deduplicate to the same cache entry. Generation keys use `normpath()` (preserves symlinks) so that logical path identity is maintained for dependency tracking.

### Multi-process safety

- Workers open StateDB in `readonly=True` mode (no write contention)
- Workers collect `DeferredWrites` and return them to the coordinator
- Coordinator applies all writes atomically in a single LMDB transaction

### Generation tracking

Generation counters enable the O(1) skip check. If a stage recorded dependencies at generations `[5, 3, 7]` and current generations are still `[5, 3, 7]`, skip without further comparison. If any generation differs, fall back to full hash comparison.

## Concurrency Safety

Pivot uses a "check-lock-recheck" pattern to prevent TOCTOU (Time-of-Check-Time-of-Use) race conditions in parallel execution.

### The Problem

Without proper locking, parallel stage execution can race: two processes could both read the lock file, both determine the stage is unchanged, both start executing, and conflict on output writes.

### The Solution: Execution Locks

All change detection and cache operations happen inside an execution lock. The lock is acquired before reading the lock file and held through execution and lock file update.

### Lock Implementation

Execution locks use PID-based sentinel files with atomic creation via `O_CREAT | O_EXCL`. See `src/pivot/storage/lock.py:execution_lock()`.

Key properties:

- **Atomic creation** - `O_CREAT | O_EXCL` guarantees only one process wins
- **Crash recovery** - Stale locks detected via PID checking
- **Cross-platform** - Works on Linux and macOS
- **Visible state** - Lock files can be inspected for debugging

### Design Decision: Simple Locks vs Reader-Writer Locks

Two approaches were considered for TOCTOU prevention:

| Approach | Description | Overhead |
|----------|-------------|----------|
| **Simple (chosen)** | Move operations inside existing execution lock | ~0ms |
| **RWLock** | Separate reader-writer locks per path (like DVC) | Higher |

**Benchmark results** (57-stage pipeline, 10 runs each, warmup excluded):

| Metric | Baseline | With TOCTOU Fix |
|--------|----------|-----------------|
| Mean | 8.944s ± 0.333s | 8.899s ± 0.282s |
| Overhead | - | -0.5% (not significant) |

The simple approach was chosen because:

1. **Zero measurable overhead** - Lock acquisition via `O_CREAT|O_EXCL` is ~μs
2. **No new dependencies** - Uses existing OS primitives
3. **Simpler code** - No separate lock coordination layer
4. **DVC's RWLock has issues** - JSON-based lock file requires full rewrite on every operation; LMDB alternative has global write lock that serializes all lock operations

The RWLock approach would only benefit workloads with many concurrent readers of the same paths, which is rare in practice since Pivot's DAG ensures dependencies complete before dependents run.

## Multi-Process Safety

### Concurrent `pivot repro` is Safe

Multiple simultaneous `pivot repro` invocations on the same project are safe and supported:

- Each invocation gets its own loky worker pool and StateDB instances
- LMDB enforces at most one writer at a time (via mutex), unlimited concurrent readers
- Read transactions use MVCC snapshots—readers never block or see partial writes
- Writes are per-stage lock files (distinct files) plus centralized StateDB (atomic updates)

### State Database Access Pattern

The coordinator-worker pattern ensures multi-process safety:

1. **Worker processes** (readonly): Open StateDB in readonly mode, see consistent MVCC
   snapshots. Collect deferred writes locally instead of writing directly.
2. **Coordinator process** (read-write): Applies all deferred writes in a single atomic
   transaction after worker completes via `apply_deferred_writes()`.

This avoids write contention between workers while maintaining consistency.

### Concurrent Scenarios

| Scenario | Result |
|----------|--------|
| Two `pivot repro`, same stage | Execution lock prevents concurrent execution |
| Two `pivot repro`, different stages | Both execute independently, writes serialize |
| `pivot repro --no-commit` + `pivot commit` | Independent operations, no lock needed |
| Cache writes by both processes | Idempotent (check exists before writing) |

## Error Handling

Two error modes:

| Mode | Behavior |
|------|----------|
| `fail` (default) | Stop on first error |
| `keep_going` | Continue with independent stages |

See `pivot.types.OnError` for the enum definition.

## Cancellation

The executor supports graceful cancellation via a `cancel_event` (a `threading.Event`). When set:

1. **Running stages complete** - The currently executing stage finishes normally
2. **Pending stages are skipped** - No new stages are started
3. **Results include cancellation** - Skipped stages report reason "cancelled"

Cancellation is **stage-level**, not mid-stage. This ensures outputs are always in a consistent state (either fully written or not started).

In watch mode, the Agent Server's `cancel` RPC method sets this event, allowing external tools to stop execution between stages. The TUI also uses this for `Ctrl+C` handling.

## Explain Mode

Preview what would run and why:

```bash
pivot status --explain [STAGES...]
# or
pivot repro --explain [STAGES...]
```

Shows:

- Code changes
- Parameter changes
- Dependency changes

## Checkout Missing Mode

The `--checkout-missing` flag restores tracked output files from cache before running:

```bash
# Restore missing tracked files, then run pipeline
pivot repro --checkout-missing
```

This is useful when:

- Switching branches where outputs were generated on another branch
- After `git clean` or accidental deletion of output files
- Cloning a repo where lock files exist but outputs don't

Without this flag, Pivot validates that all tracked outputs exist before running. If files are missing, it fails with an error suggesting either `pivot checkout --only-missing` (to restore without running) or `pivot repro --checkout-missing` (to restore and run).

**How it works:** Files are restored using the hashes recorded in existing lock files—no stages are re-executed during restoration. The cache must contain the files (push/pull from remote if needed). After restoration, the normal execution flow continues and may skip stages if nothing else changed.

## Deferred Commit Mode

The `--no-commit` flag runs stages without writing durable state:

```bash
# Run stages — outputs land on disk but no locks, cache, or StateDB updates
pivot repro --no-commit

# Inspect results, then snapshot current workspace state
pivot commit
```

### How It Works

1. `--no-commit`: stages execute and outputs are written to disk. No lock files, no cache copies, no StateDB updates. Output hashes are computed for the `StageResult` but nothing is persisted.
2. `pivot commit [stage_names...]`: computes current workspace state (fingerprints code, hashes deps and outputs), writes production lock files, saves outputs to cache, and updates StateDB. Without arguments, commits all stale stages. With stage names, unconditionally commits those stages.

### Use Case

This workflow avoids caching overhead during rapid iteration. `pivot commit` is also the "trust me" path — if you change code but know outputs are still correct, `pivot commit` records the current state without re-running stages.

## See Also

- [Architecture Overview](overview.md) - System architecture
- [Fingerprinting](fingerprinting.md) - Code change detection
