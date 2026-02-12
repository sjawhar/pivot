# Unified Skip Detection Design

Supersedes `2026-02-11-unify-skip-detection-design.md` (problem statement retained as reference).

## Problem

Two independent code paths answer "will this stage run?" — the worker path
(`executor/worker.py`) and the explain path (`explain.py`). They use different
code for Tier 2 comparison and can give different answers. The explain path
also doesn't check output path changes and doesn't implement Tier 3 (run cache).

Additionally, `engine/agent_rpc.py` uses the global `config_io.get_state_dir()`
instead of the per-stage state_dir, reading the wrong lock file for composed
pipelines.

## Design

### One function: `skip.check_stage()`

A new module `pivot/skip.py` contains the single source of truth for "will this
stage run?" Both the engine (during `repro`) and the explain/status path call
this function.

```python
def check_stage(
    lock_data: LockData | None,
    fingerprint: dict[str, str],
    params: dict[str, Any],
    dep_hashes: dict[str, HashInfo],
    out_paths: list[str],
    *,
    explain: bool = False,
    force: bool = False,
) -> ChangeDecision:
```

The function implements Tier 2 (lock file comparison). Tier 1 (generation check)
is invoked by callers before this function when a StateDB is available. Tier 3
(run cache) is handled by callers after this function when `changed=True`.

When `explain=False` (engine/repro), the function uses cheap dict equality and
short-circuits at the first detected change — no detailed diffs are computed.
When `explain=True` (status/explain/agent_rpc), it evaluates all comparisons
exhaustively and returns detailed diffs.

```python
    if lock_data is None:
        return ChangeDecision(changed=True, reason="No previous run")

    # Fast path: cheap dict equality, no detailed diffs
    if not explain:
        if lock_data["code_manifest"] != fingerprint:
            return ChangeDecision(changed=True, reason="Code changed")
        if lock_data["params"] != params:
            return ChangeDecision(changed=True, reason="Params changed")
        ...
        return ChangeDecision(changed=False, reason="")

    # Explain path: full diffs for all categories
    code_changes = diff_code_manifests(lock_data["code_manifest"], fingerprint)
    param_changes = diff_params(lock_data["params"], params)
    dep_changes = diff_dep_hashes(lock_data["dep_hashes"], dep_hashes)
    ...
    return ChangeDecision(
        changed=changed,
        reason=_first_reason(...),
        code_changes=code_changes,
        param_changes=param_changes,
        dep_changes=dep_changes,
    )
```

Tier 3 (run cache) stays outside `check_stage` — it involves output restoration
which is a side effect, not a comparison.

### Skip detection moves to the engine

The engine decides skip/run in `_start_ready_stages()` before dispatching to
the worker pool. Skipped stages never cross the process boundary.

```
Engine _start_ready_stages():
  for each ready stage:
    acquire scheduler mutexes
    set state → PREPARING
    acquire artifact locks (flock via anyio.to_thread.run_sync)
    
    fingerprint (just-in-time, deferred until stage is ready)
    hash dependencies
    read lock file
    decision = check_stage(explain=False)
    
    if NOT CHANGED:
      restore outputs from cache (async I/O, under flock)
      handle completion inline (emit events, release mutexes, release flock)
      continue
    
    if CHANGED and run cache hit:
      restore from cache, commit lock (async I/O, under flock)
      handle completion inline
      continue
    
    if CHANGED and must run:
      prepare WorkerStageInfo (fingerprint already computed)
      submit to worker pool
      keep flock held (released in _handle_stage_completion)
      set state → RUNNING
      emit StageStarted(explanation from check_stage)
```

Fingerprinting is just-in-time: each stage is fingerprinted when it becomes
ready, not up front. This avoids fingerprinting stages 50-150 before stage 1
starts, and ensures fingerprints reflect current source state (important in
watch mode where source files change mid-run).

### Artifact locking: engine holds flock for full duration

The engine acquires artifact locks (WRITE on outs, READ on deps) via
`LocalFlockLockService.acquire_many()` and holds them for the entire
skip-check-through-completion duration:

- **SKIP path**: engine acquires flock → check → restore → release flock
- **RUN path**: engine acquires flock → check → dispatch to worker → worker
  executes under engine's flock protection → engine releases flock in
  `_handle_stage_completion()`

The flock fd is stored in `self._artifact_locks[stage_name]` and released in
the completion handler. The worker does not acquire artifact locks at all —
the engine's flock protects it.

This eliminates the TOCTOU gap: the lock is held continuously from decision
through action. No hand-off across process boundaries (which flock can't do).
Crash recovery is automatic — the kernel releases all flocks when the engine
process dies.

For concurrent `pivot repro`: Engine B trying to acquire flock on the same
outputs blocks until Engine A releases in its completion handler. This
serializes concurrent access correctly.

### Worker simplification

The worker no longer contains skip detection or artifact locking. It receives
only stages that genuinely need execution:

```
worker.execute_stage():
    chdir(project_root)
    run stage function
    hash outputs, save to cache
    commit lock file
    return StageResult with deferred writes
```

Removed from worker:
- `can_skip_via_generation()` call (Tier 1)
- `_check_skip_or_run()` (Tier 2)
- `_try_skip_via_run_cache()` (Tier 3)
- Output restoration for skipped stages
- Artifact lock acquisition
- StateDB open in readonly mode for skip detection

### State_dir bug fix

`engine/agent_rpc.py` must use `registry.get_stage_state_dir(stage_info,
default_state_dir)` instead of `config_io.get_state_dir()`. This is a one-line
fix, independent of the rest of this design.

### Event handling

Skipped stages emit `StageCompleted(status=SKIPPED)` without a preceding
`StageStarted`. This is already the pattern for blocked/cancelled stages
(`_emit_skipped_stage`). The TUI handles this — `_finalize_history_entry`
creates synthetic entries for stages without a prior IN_PROGRESS. The console
sink handles missing `stage_start` gracefully (duration renders as None).

### What moves where

| From | To | What |
|------|----|------|
| `executor/worker.py` | `skip.py` | `can_skip_via_generation()` |
| `explain.py` | `skip.py` | `diff_code_manifests()`, `diff_params()`, `diff_dep_hashes()` |
| `storage/lock.py` | `skip.py` | `is_changed_with_lock_data()` logic (absorbed into `check_stage`) |
| `executor/worker.py` | deleted | `_check_skip_or_run()`, `_try_skip_via_run_cache()`, artifact lock code |
| `explain.py` | deleted | Tier 2 inline comparison logic (lines 280-315) |
| `executor/worker.py` | `engine.py` | Output restoration for skipped/cached stages |

### Callers

| Call site | Mode | Purpose |
|-----------|------|---------|
| `engine.py:_start_ready_stages()` | `explain=False` | Repro skip decision |
| `status.py:_get_explanations_in_parallel()` | `explain=True` | `pivot status` display |
| `engine/agent_rpc.py` | `explain=True` | TUI explain panel |

## Design decisions

**One function, not two.** The `explain` flag controls short-circuit vs
exhaustive. The check ordering is the same in both modes — no divergence
possible. An early version of this design proposed two functions (`should_run`
and `explain_changes`) but the control flow was identical except for the
short-circuit, making a single function with a flag cleaner.

**Pre-computed fingerprint, not lazy.** Profiling on a 173-stage pipeline
showed fingerprinting is always needed (for lock file comparison if skipping,
for lock file commit if running). Deferring it via a callable adds API
complexity without saving work. Fingerprinting is just-in-time per stage
(computed when the stage becomes ready), which naturally interleaves with
execution. See GitHub issue METR/eval-pipeline#749 for a potential future
optimization to avoid fingerprinting entirely when source file stats are
unchanged.

**Engine holds flock, not worker.** The worker can't inherit a flock from
the engine (flock is per-fd, per-process, can't be transferred across process
boundaries). Having the engine hold the flock for the full duration eliminates
the hand-off problem and removes artifact locking code from the worker.

**Run cache (Tier 3) outside check_stage.** Run cache involves output
restoration (side effects), not just comparison. The engine handles it after
`check_stage` returns `changed=True`.

**Concurrent `pivot repro` race.** If two engines evaluate the same stage,
both may decide "must run" and dispatch workers. Workers are serialized by the
engine's flock (Engine B blocks until Engine A's completion handler releases).
In the rare case where both engines dispatch before either acquires the flock,
both workers execute. This produces correct outputs (second write is identical
or supersedes) but wastes work. Acceptable for a pre-alpha tool.

## Profiling results (173-stage eval-pipeline)

| Operation | Total (warm) | Mean per stage | % of total |
|-----------|-------------|----------------|------------|
| Fingerprinting | 824ms | 4,762µs | 48% |
| Dep hashing | 422ms | 2,441µs | 25% |
| Lock file read | 452ms | 2,613µs | 26% |
| Lock comparison | 3ms | 20µs | <1% |
| Generation check | 0ms | 0µs | <1% |

Fingerprint stat redundancy: 3,521 stat calls for 636 unique source files
(5.5x). Cold start (first invocation): 4.3s total, dominated by AST walking.

## Out of scope

- Fingerprint cache optimization (filed as METR/eval-pipeline#749)
- Batch stat deduplication across stages sharing source files
- Parallel skip evaluation in `_start_ready_stages` (sequential is fine for now;
  the engine's event loop isn't doing useful work during dispatch anyway)
- Watch-mode-specific optimizations (file-watcher-based fingerprint invalidation)

## Related

- `2026-02-11-unify-skip-detection-design.md` — problem statement (retained)
- `2026-02-11-engine-hardening.md` — engine hardening plan (in progress,
  includes WatchCoordinator and scheduler improvements)
- `docs/solutions/2026-02-08-skip-detection-invariants.md` — three-tier
  algorithm documentation
