# Execution Model

Pivot uses a parallel execution model with warm worker pools for maximum performance.

## Execution Flow

```
┌──────────────┐
│  pivot run   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Build DAG       │
│  (topological    │
│   sort)          │
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
│  Greedy          │
│  Scheduler       │──────────────┐
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

## Greedy Scheduling

Pivot uses greedy scheduling for maximum parallelism:

1. **Ready Queue** - Stages with all dependencies satisfied
2. **Running Set** - Currently executing stages
3. **Completed Set** - Finished stages

```python
while not all_completed:
    # Find stages that can run
    ready = [s for s in pending if all_deps_complete(s)]

    # Respect mutex groups
    ready = filter_by_mutex(ready, running)

    # Submit to workers
    for stage in ready:
        submit(stage)
```

## Worker Pool

Pivot uses `loky.get_reusable_executor()` for warm workers:

```python
executor = loky.get_reusable_executor(
    max_workers=cpu_count(),
    context='forkserver',
)
```

### Why ProcessPoolExecutor?

- **True parallelism** - Not limited by Python's GIL
- **Isolation** - Each stage runs in its own process
- **Memory efficiency** - Workers can be recycled

### Why Forkserver?

- **Safety** - Avoids fork() issues with threads
- **Compatibility** - Works on macOS and Linux
- **Clean state** - Each worker starts from a clean fork

### Warm Workers

Workers stay alive between stages:

```python
# First stage: imports numpy, pandas (slow)
# Second stage: already imported (fast)
```

This avoids repeated import overhead for heavy dependencies.

## Mutex Handling

Mutex groups prevent concurrent execution:

```python
@stage(mutex=['gpu'])
def train_model_a(): pass

@stage(mutex=['gpu'])
def train_model_b(): pass  # Won't run while train_model_a is running
```

Implementation:

1. Track active mutex groups
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

## Concurrency Safety

Pivot uses a "check-lock-recheck" pattern to prevent TOCTOU (Time-of-Check-Time-of-Use) race conditions in parallel execution.

### The Problem

Without proper locking, parallel stage execution can race:

```
Process A                     Process B
─────────                     ─────────
Read lock file                Read lock file
Check: unchanged              Check: unchanged
                              Start executing...
Start executing...            ...
Write output                  Write output (CONFLICT!)
```

### The Solution: Execution Locks

All change detection and cache operations happen inside an execution lock:

```python
with execution_lock(stage_name, cache_dir):
    lock_data = stage_lock.read()           # Read inside lock
    dep_hashes = hash_dependencies(deps)     # Hash inside lock

    if can_skip(lock_data, fingerprint, dep_hashes):
        restore_outputs_from_cache(...)
        return SKIPPED

    run_stage_function()
    save_outputs_to_cache()
    stage_lock.write(new_lock_data)
```

### Lock Implementation

Execution locks use PID-based sentinel files with atomic creation:

```python
# Atomic lock acquisition
fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
with os.fdopen(fd, 'w') as f:
    f.write(f"pid: {os.getpid()}\n")
```

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

### Concurrent `pivot run` is Safe

Multiple simultaneous `pivot run` invocations on the same project are safe and supported:

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
| Two `pivot run`, same stage | Execution lock prevents concurrent execution |
| Two `pivot run`, different stages | Both execute independently, writes serialize |
| `pivot run --no-commit` + `pivot commit` | `pending_state_lock` coordinates |
| Cache writes by both processes | Idempotent (check exists before writing) |

## Error Handling

Three error modes:

| Mode | Behavior |
|------|----------|
| `fail` (default) | Stop on first error |
| `keep_going` | Continue with independent stages |
| `ignore` | Log errors, continue all |

```python
from pivot.executor import run
from pivot.types import OnError

run(on_error=OnError.KEEP_GOING)
```

## Timeouts

Stage-level timeouts prevent runaway execution:

```python
run(stage_timeout=3600)  # 1 hour per stage
```

## Dry Run Mode

Preview execution without running:

```python
run(dry_run=True)
```

Returns what would run and why.

## Explain Mode

Detailed breakdown of why stages run:

```python
run(explain_mode=True)
```

Shows:

- Code changes
- Parameter changes
- Dependency changes

## See Also

- [API Reference: executor](../reference/pivot/executor.md) - Full API documentation
