# Skip Detection: Problem Statement

## Bug: agent_rpc uses wrong state_dir

`engine/agent_rpc.py:284` passes `config_io.get_state_dir()` (the global default) to `explain.get_stage_explanation()`. It doesn't look up the per-stage `state_dir` via `registry.get_stage_state_dir()`.

Every other call site resolves per-stage state_dir correctly:
- `status.py:121` — `registry.get_stage_state_dir(stage_info, default_state_dir)`
- `executor/core.py:323` — same pattern for worker dispatch
- `engine/engine.py:636` — same pattern for deferred writes
- `cli/verify.py:75` — same pattern
- `cli/checkout.py:51` — same pattern

The agent_rpc path is the only one that skips this lookup. If a stage has a custom `state_dir` (from a composed pipeline), the explain query reads the wrong lock file and StateDB.

## Structural issue: Two independent implementations of "will this stage run?"

The question "will this stage run?" is answered by two completely separate code paths that can drift:

### Path 1: Worker (during `pivot repro`)

`executor/worker.py:execute_stage()` — lines 164-434

The engine submits **every stage** in execution_order to the worker pool unconditionally (`engine.py:_start_ready_stages()` at line 927). There is no coordinator-side filtering. The worker itself decides whether to skip:

1. **Tier 1** — `can_skip_via_generation()` (line 220) — O(1) generation counter comparison
2. **Tier 2** — `_check_skip_or_run()` (line 266) → `lock.StageLock.is_changed_with_lock_data()` — hashes all deps, compares fingerprint/params/dep_hashes/out_paths against lock file. Returns `(changed: bool, reason: str)`.
3. **Tier 3** — `_try_skip_via_run_cache()` (line 303) — checks if same input configuration was previously executed

### Path 2: Explain (during `pivot status` / `pivot explain`)

`explain.py:get_stage_explanation()` — lines 158-327

Called from `status.py:_get_explanations_in_parallel()` and `engine/agent_rpc.py`.

1. **Tier 1** — `can_skip_via_generation()` (line 214) — same function as worker (shared, good)
2. **Tier 2** — Independent implementation (lines 235-315):
   - Hashes deps via `worker.hash_dependencies()` (shared)
   - But then does its own field-by-field comparison: `diff_code_manifests()`, `diff_params()`, `diff_dep_hashes()` — producing detailed change lists
   - Does **not** call `is_changed_with_lock_data()`
3. **Tier 3** — Not implemented (explain doesn't check run cache)

### Where they diverge

| Aspect | Worker | Explain |
|--------|--------|---------|
| Tier 1 (generation) | `can_skip_via_generation()` | Same function (shared) |
| Tier 2 decision | `is_changed_with_lock_data()` | Independent diff logic |
| Tier 2 output | `(bool, str)` | Detailed change lists |
| Tier 3 (run cache) | Yes | No |
| Output restoration | Restores symlinks for skipped stages | N/A |
| Runs under execution lock | Yes | No |
| Output path comparison | Compares normalized out_paths | Does not compare out_paths |

The Tier 2 logic is the core problem. Both paths answer the same question ("did fingerprint/params/deps change?") but through different code. The worker delegates to `lock.StageLock.is_changed_with_lock_data()` (`storage/lock.py:228`). The explain path does its own `diff_code_manifests` / `diff_params` / `diff_dep_hashes` comparisons. If one path is updated and the other isn't, `pivot status` could predict a different outcome than `pivot repro`.

Notable: the explain path doesn't check output path changes at all — `is_changed_with_lock_data()` does (line 246-249), but the explain path skips this comparison entirely.

### Why the worker has skip logic at all

The engine doesn't pre-filter. `_start_ready_stages()` (engine.py:927-1002) iterates over ready stages and submits every one to the worker pool. The worker is the only place the run/skip decision happens during execution.

The worker checks skip **under the execution lock** (`lock.execution_lock()` at line 213). This prevents TOCTOU races — between a coordinator check and actual execution, another `pivot repro` process could change outputs.

Every skipped stage still pays the cost of: pickling `WorkerStageInfo` across the process boundary, worker process setup (`chdir`, logging), StateDB open in readonly mode, and the generation check itself.

## Key files

| File | Lines | Role |
|------|-------|------|
| `executor/worker.py` | 164-434 | Worker skip detection + execution |
| `executor/worker.py` | 1017-1084 | `can_skip_via_generation()` (shared) |
| `executor/worker.py` | 465-496 | `_check_skip_or_run()` (worker-only) |
| `explain.py` | 158-327 | `get_stage_explanation()` (explain-only) |
| `storage/lock.py` | 228-251 | `is_changed_with_lock_data()` (used by worker, not explain) |
| `engine/engine.py` | 927-1002 | `_start_ready_stages()` (submits all stages unconditionally) |
| `engine/engine.py` | 570-577 | Per-state_dir StateDB cache |
| `engine/agent_rpc.py` | 265-287 | Agent RPC explain handler (state_dir bug) |
| `status.py` | 102-155 | `_get_explanations_in_parallel()` (correctly resolves per-stage state_dir) |

## Related plans

- `2026-02-11-engine-coordinator-backend-refactor.md` — refactors `_start_ready_stages()` and engine orchestration loop
- `2026-02-07-rearchitect-commit-design.md` — removes pending locks, simplifies `--no-commit` (orthogonal to skip detection)
- `docs/solutions/2026-02-08-skip-detection-invariants.md` — documents the three-tier algorithm and its invariants
