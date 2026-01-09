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

1. **Restore IncrementalOut** - If using incremental outputs
2. **Execute Function** - Run the user's code
3. **Hash Outputs** - Compute content hashes
4. **Cache Outputs** - Store in content-addressable cache
5. **Write Lock File** - Record fingerprint

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
