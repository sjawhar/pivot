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

1. **Stage States** - Each stage has its own state (PENDING → READY → PREPARING → WAITING_ON_LOCK → RUNNING → COMPLETED)
2. **Scheduler** - `Scheduler` owns stage state, upstream/downstream sets, and resource permits
3. **Ready Queue** - Engine asks Scheduler for eligible stages
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
    WAITING_ON_LOCK = 4  # Worker waiting for artifact locks
    RUNNING = 5      # Stage function executing
    COMPLETED = 6    # Terminal
```

The IntEnum allows ordered comparisons (e.g., `state >= PREPARING` means execution has begun).

### State Transitions

```
PENDING ──(deps complete)──▶ READY ──(worker available)──▶ PREPARING ──▶ WAITING_ON_LOCK ──▶ RUNNING ──▶ COMPLETED
    │                                                           │
    └──(upstream failed)──▶ BLOCKED                             └──(failed)──▶ COMPLETED
```

### Output Filtering by State

During watch mode, the Engine filters filesystem events based on stage state:

- **PREPARING**: Silence events for this stage's outputs (Engine is preparing them)
- **WAITING_ON_LOCK**: Defer events while worker waits on artifact locks
- **RUNNING**: Defer events for outputs (collect, don't act yet)
- **COMPLETED**: Process deferred events, compare output hashes, trigger downstream

## Worker Pool

Pivot uses `loky.get_reusable_executor()` with `spawn` context. `WorkerPool` owns workers and
the manager process for the Engine lifetime. Each run gets its own manager-backed output queue
and drain shutdown event, so logs cannot cross run boundaries. A manager that cannot be reached
when a run starts (it died, or its socket was removed) is replaced then.

### Why ProcessPoolExecutor?

- **True parallelism** - Not limited by Python's GIL
- **Isolation** - Each stage runs in its own process
- **Memory efficiency** - Workers can be recycled

### Why Spawn?

- **Safety** - Avoids fork() issues with threads (Python 3.13+ deprecates fork in multithreaded contexts)
- **Compatibility** - Works on macOS and Linux
- **Clean state** - Each worker starts fresh without inherited state

### Pool Size

A run starts one worker per stage it can run at once, up to `core.max_workers`: the width of
the run's stage graph, the largest set of its stages none of which consumes, directly or through
other stages of the run, another's output (`engine/graph.py` `get_width`). A chain of stages runs
on one worker; five stages that read one source stage run on five. The width counts every stage
of the run, including stages that will skip, so a partial run starts as many workers as a full
one. The widest topological level can be smaller: with stages A and S, and B and C that both
read S, the levels {A, S} and {B, C} hold two stages each, but A, B and C can all run together,
so the width is three. The widest level is still a lower bound, so once it reaches
`core.max_workers` the pool takes the cap without computing the width. Below the cap, the search
for the width takes a fixed number of steps; a pipeline it does not settle in that many gets an
upper bound on the width instead, never fewer workers than its stages that can run at once.

A TUI pre-warms the same number of workers before Textual starts. It restores missing tracked
files first when `--checkout-missing` asks for it, and it counts the stages the engine will run
(`get_runnable_stages`): with `--all --keep-going`, a stage with a missing dependency and
everything downstream of it are left out of both. `pivot run` runs its named stages in the given
order and starts a stage once the named stages it consumes from have finished, so its width is
that of the named stages alone.

The width bounds the pool, not the schedule: stage `resources` still limit how many stages run
together.

### Warm Workers

Workers stay alive across runs in one Engine, so imports from earlier stages remain available.
Before a later run reuses workers, Pivot replaces them if a user-code module they imported
changed. Worker capacity grows for a wider run but does not shrink for a narrower one: the
scheduler still limits that run's parallelism, while a later wide run reuses every warm worker.
A watch-mode code reload restarts the pool the same way, at the larger of the reloaded
pipeline's width and the pool's earlier size, so the run after it starts no other worker.

## Stage resources

Stages can reserve scheduler-held resources while they execute:

```python
@stage(resources={"gpu": 1})
def train(data: pd.DataFrame) -> Model: ...


@stage(resources={"cpu": "all"})
def compact(data: pd.DataFrame) -> pd.DataFrame: ...
```

Resource names are case-insensitive. Every scheduled stage, including bare `@stage` functions, holds one `cpu` unit. The `cpu` capacity is the configured worker count (`core.max_workers`), not the pool size, so `cpu: "all"` runs without another stage. Other resources default to capacity 1, which serializes stages sharing that resource.

Set a project capacity in `.pivot/config.yaml` or override it for one command:

```yaml
resources:
  gpu: 2
```

```bash
pivot repro --resources gpu=2
pivot run --resources gpu=2 report/train
```

The scheduler acquires every permit immediately before submitting a stage and releases them when it completes. A stage waiting for a resource does not consume a worker. Resource declarations tune concurrency only; they are not part of stage fingerprints.

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

- Each output has a **generation counter** in StateDB, keyed by its artifact identity and incremented every time the output is written
- When a stage runs, it **records the generation** of each dependency at that moment
- On next run: if `recorded_generation == current_generation` for all deps → nothing changed → skip

A dependency on one file of a directory output follows its directory's counter, which moves whenever any of its files are written; when it moved, the file's own hash in Tier 2 decides.

**Example:** Stage `train` depends on `demo/preprocess`'s output.

1. `preprocess` runs and writes its output → generation becomes 5
2. `train` runs, records `dep_generations: {"demo/preprocess": 5}`
3. Next run: the generation is still 5 → generations match → skip `train`
4. `preprocess` runs again → generation becomes 6
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

Pivot uses a "check-lock-recheck" pattern to prevent TOCTOU (Time-of-Check-Time-of-Use) race conditions between concurrent `pivot` processes on one project.

### The Problem

Without locking, two processes could both read a stage's lock file, both decide the stage must run (or that a missing output must be restored), and conflict on output writes; or one could overwrite an output while another process's stage is reading it.

### Artifact Locks

Each artifact has a lock file under the stage state directory's `locks/`, taken with `flock` (`storage/artifact_lock.py`): READ is `LOCK_SH`, WRITE is `LOCK_EX`. `expand_lock_requests` turns a stage's deps and outs into requests, one per artifact at the strongest mode asked for, and `acquire_many` takes them in sorted key order (no lock-order deadlocks), retrying a non-blocking `flock` until it succeeds.

| Who | Deps | Outs | Held |
|-----|------|------|------|
| A worker running a stage | READ | WRITE | for the whole run, through the lock file write |
| The coordinator deciding whether a cached stage skips | READ | READ | for the check |
| The coordinator restoring a cached stage's outputs from the cache | READ | WRITE | for the check and the restore |

A stage lock is read only after its artifact locks are held, so a waiting process evaluates the state committed by the writer ahead of it.

A cached stage's skip check takes READ on its outputs, so it does not wait for the runs reading them as deps: a widely read output is held READ by every running consumer, and WRITE would wait until none is running. When the check finds an output that differs from the lock but is in the cache, it releases its locks, takes WRITE on the outputs, and decides again from the start, since another process may have restored or rebuilt them in between. Locks are never upgraded in place: two processes each holding READ and waiting for WRITE would wait on each other.

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
| Two `pivot repro`, same stage | WRITE on its outputs lets one run it; the other waits, then finds it cached |
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

The engine supports graceful cancellation via a `CancelRequested` event. When set:

1. **Running stages complete** - The currently executing stage finishes normally
2. **Pending stages are skipped** - No new stages are started
3. **Results include cancellation** - Skipped stages report reason "cancelled"

This cancellation is **stage-level**, not mid-stage. This ensures outputs are always in a consistent state (either fully written or not started).

In watch mode, the Agent RPC `cancel` command sends a `CancelRequested` event, allowing external tools to stop execution between stages. The TUI also uses this for `Ctrl+C` handling.

### Stopping mid-stage

`Ctrl+C` at a terminal (outside the TUI), and the death of the Pivot process (`SIGKILL`, or `SIGTERM`, which Pivot does not handle outside the `pivot` command's `repro --watch --serve`; that serve mode handles SIGTERM as `Ctrl+C`), stop running stages mid-stage. Each running stage gets `pivot.executor.worker.WorkerTerminated`, a `KeyboardInterrupt`, raised where it is, so its `except KeyboardInterrupt` cleanup, `finally` blocks and context managers run, as they would for `Ctrl+C` in a plain Python program. Then:

- The stage fails and nothing it produced is committed: no output enters the cache and no lock is written, even if the stage caught the exception and returned a value.
- The worker exits once the stage has unwound, and no later than `worker.TERMINATION_GRACE_S` (10 s) after it was told to stop, whatever the stage does. Before exiting, it kills every process still descended from it. A process that detached (double-forked, a daemon, or orphaned when its parent exited) is not reached; the stage's cleanup must stop it, as it must stop a container, which the Docker daemon runs.
- Pivot itself (on `Ctrl+C`) waits up to the same 10 s for its workers, then kills any left with their child processes.

A cleanup that must finish therefore has 10 s: remove a container with `docker rm --force`, which takes about a second, not `docker stop`, which waits 10 s for the container by default.

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
