# Lazy Fingerprinting: Decouple Graph-Building from Fingerprinting

## Problem

Pipeline startup is slow because fingerprinting happens eagerly during stage registration, before the graph is built or any stage can execute. For a 108-stage pipeline, `registry.register()` spends ~666ms on fingerprinting — blocking graph availability and delaying execution.

Fingerprinting and graph topology are independent concerns:
- **Graph topology** ("what depends on what") only changes when stage definitions change (new deps/outputs/stages)
- **Fingerprints** ("has the code changed") change whenever any code in a stage's dependency tree changes

Watch mode demonstrates this clearly: after editing a helper function, fingerprints change but the graph stays the same.

## Solution

Make fingerprint computation lazy. Remove fingerprinting from `registry.register()` and compute fingerprints on first access via an accessor on `StageRegistry`. The graph becomes available immediately after registration, and fingerprints are computed just-in-time when the engine is about to dispatch each stage.

**Target:** Eliminate ~666ms of fingerprinting from the registration path. Fingerprinting cost shifts to execution time, where it's pipelined with stage execution (later waves fingerprint while earlier waves run).

## Design

### RegistryStageInfo

Change `fingerprint` from required to nullable:

```python
# Before
fingerprint: dict[str, str]

# After
fingerprint: dict[str, str] | None
```

`register()` sets `fingerprint=None` instead of computing it.

### Accessor: `StageRegistry.ensure_fingerprint()`

Add `ensure_fingerprint()` to `StageRegistry`:

```python
def ensure_fingerprint(self, stage_name: str) -> dict[str, str]:
    info = self._stages[stage_name]
    if info["fingerprint"] is None:
        info["fingerprint"] = _compute_fingerprint(stage_name, info)
    return info["fingerprint"]
```

### Fingerprint computation

`_compute_fingerprint` replaces the inline logic currently at `register()` lines 470–473. It uses `dep_specs` and `out_specs` from the stored `RegistryStageInfo` directly, which lets us inline the loader fingerprinting rather than calling the three-argument `_get_annotation_loader_fingerprints(dep_specs, return_out_specs, single_out_spec)` helper (since `out_specs` already merges both return and single output specs):

```python
def _compute_fingerprint(stage_name: str, info: RegistryStageInfo) -> dict[str, str]:
    try:
        fp = fingerprint.get_stage_fingerprint(info["func"])
        for spec in info["dep_specs"].values():
            fp.update(fingerprint.get_loader_fingerprint(spec.loader))
        for out in info["out_specs"].values():
            fp.update(fingerprint.get_loader_fingerprint(out.loader))
        return fp
    except Exception as exc:
        raise exceptions.PivotError(
            f"Stage '{stage_name}': fingerprinting failed: {exc}"
        ) from exc
```

This may allow removing `_get_annotation_loader_fingerprints` if no other callers remain.

### Error handling

Fingerprint errors shift from registration time to first access (execution, status, watch reload). This is acceptable for the goal of fast graph building. Commands that only need graph topology (`pivot list`, `pivot dag`, tab-completion) no longer surface fingerprint problems — they work without computing fingerprints at all.

`ensure_fingerprint()` wraps failures in `PivotError` with the stage name so the user sees which stage failed and why.

### Concurrency

`pivot.fingerprint` caches are not thread-safe. To maintain that invariant, fingerprints must be computed on the main thread before any parallel work:

- **`status.get_pipeline_explanations`**: precompute fingerprints for all stages in `execution_order` on the main thread before submitting to the `ThreadPoolExecutor`.
- **Engine execution**: compute the fingerprint in `_start_ready_stages` (on the event loop thread, before submitting the worker to the process pool). This is safe because `_start_ready_stages` is `async` but runs on a single-threaded event loop — no concurrent `ensure_fingerprint()` calls.
- **Watch reload**: runs on the event loop thread, same single-threaded guarantee.

No lock is needed. The computation is synchronous and idempotent, and all call sites are single-threaded.

### Watch mode reload

`_emit_reload_event` compares old and new fingerprints to detect modified stages. With lazy fingerprinting, both old and new `RegistryStageInfo` may have `fingerprint=None` at comparison time.

The fix: call `ensure_fingerprint()` on both old and new stage infos for the intersection of stage names. If fingerprinting fails during reload, log a warning and conservatively treat the stage as modified (keeps the UI informative, prevents engine crashes):

```python
for stage_name in sorted(old_stage_names & new_stages_set):
    try:
        old_fp = old_registry.ensure_fingerprint(stage_name)
    except exceptions.PivotError:
        old_fp = None
    try:
        new_fp = self._registry.ensure_fingerprint(stage_name)
    except exceptions.PivotError:
        new_fp = None
    if old_fp != new_fp:
        modified.append(stage_name)
```

Note: `snapshot()` returns a shallow copy — it shares references to the same `RegistryStageInfo` dicts. If `ensure_fingerprint()` was called on the live registry before snapshotting, the snapshot gets the materialized fingerprint for free. If not, calling `ensure_fingerprint()` on the old registry during reload still works because it mutates the shared dict in place. Either way, comparison is correct.

Selective re-fingerprinting (only fingerprint stages affected by changed files) is a future optimization tracked in #358.

### `add_existing()`

Stages added via `add_existing()` may arrive with `fingerprint=None` (from another lazy registry) or pre-computed. `ensure_fingerprint()` handles both — it checks for `None` and computes if needed.

### Consumers

All sites that access `stage_info["fingerprint"]` switch to `registry.ensure_fingerprint(stage_name)`:

| File | Context | Notes |
|------|---------|-------|
| `engine/engine.py` | Watch mode comparison | Call on both old and new registries |
| `engine/engine.py` | `_start_ready_stages` | Compute before worker dispatch |
| `executor/core.py` | Building `WorkerStageInfo` | Receives pre-computed fingerprint from engine |
| `tui/run.py` | Building `WorkerStageInfo` | Call before building WorkerStageInfo |
| `tui/diff_panels.py` | Building `WorkerStageInfo` | Call before building WorkerStageInfo |
| `status.py` | `get_pipeline_explanations` | Precompute all before entering thread pool |
| `engine/agent_rpc.py` | Agent RPC | Call before building WorkerStageInfo |

**No changes needed:**
- `executor/worker.py` — receives the computed fingerprint via `WorkerStageInfo`
- `fingerprint.py` — computation logic unchanged, just called from a different place
- `engine/graph.py` — graph building doesn't touch fingerprints

### Tests and metrics

Existing tests cover all consumer paths — they exercise the full pipeline through execution. The `registry.register` metric will naturally get faster (that's the point). The existing `fingerprint.get_stage_fingerprint` metric inside `fingerprint.py` still fires when `ensure_fingerprint()` triggers computation, so fingerprinting cost remains independently measurable.

Add a test for the lazy computation behavior: register a stage, verify `fingerprint` is `None`, call `ensure_fingerprint()`, verify it returns a non-empty dict and caches it.

## What This Does NOT Change

- **Fingerprint computation logic** — `fingerprint.py` stays the same
- **Graph construction** — already fingerprint-free, no changes
- **Worker skip detection** — workers receive fingerprints via `WorkerStageInfo`, unchanged
- **Fingerprint caching** — in-memory and persistent StateDB caches still work

## Known Limitations

- Watch mode fingerprints all stages on re-discovery, not just affected ones (#358).
