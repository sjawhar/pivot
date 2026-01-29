# Determinism by Default

**Status:** Ready for review
**Date:** 2026-01-29

## Goal

Bit-for-bit reproducibility by default for all Pivot stages. Same inputs + same code = identical outputs across machines and runs.

## Design

### What Pivot Controls

**Environment variables (set before worker processes spawn):**
- `PYTHONHASHSEED=0` — Makes Python's `hash()` function deterministic

**Random seeds (set before each stage execution):**
- `random.seed(0)` — stdlib random module
- `numpy.random.seed(0)` — numpy's legacy random (only if numpy is importable)

**Seed value:** `0` — The conventional "deterministic" value, consistent across all controls.

### What Pivot Doesn't Control

- **PyTorch/TensorFlow seeds** — Users of these frameworks are typically seed-aware
- **Environment sanitization** — Too fragile, may break legitimate use cases
- **numpy's `default_rng()`** — Users of this modern API are already managing their own seeds

## Implementation

### PYTHONHASHSEED

This environment variable must be set before the Python interpreter starts — it cannot be changed mid-process. Since Pivot uses `loky.get_reusable_executor()` for worker pools, workers inherit environment from the parent process.

**Location:** `src/pivot/executor/core.py` (before executor creation)

```python
os.environ.setdefault("PYTHONHASHSEED", "0")
```

Using `setdefault` respects any value the user explicitly sets before running Pivot.

### Random Seeds

Seeds are set before each stage execution, not just at worker startup. This ensures determinism even when workers are reused across stages.

**Location:** `src/pivot/executor/worker.py`

```python
def _set_deterministic_seeds() -> None:
    """Set random seeds for reproducible stage execution."""
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except ImportError:
        pass
```

Called immediately before invoking the user's stage function.

## User Overrides

Users who need different behavior can override at two levels:

1. **Hash seed:** Set `PYTHONHASHSEED=X` before running `pivot run`
2. **Random seeds:** Call `random.seed(X)` or `np.random.seed(X)` in stage code

No special annotations or configuration needed — just write code.

## Edge Cases

**Warm worker pools:** Loky reuses workers across stages. Since seeds reset before each stage execution, previous random state doesn't affect subsequent stages.

**Parallel stages:** Multiple concurrent stages each start with seed 0. Identical stages with identical inputs produce identical random sequences — this is correct for reproducibility.

**User-set PYTHONHASHSEED:** `setdefault` preserves existing values, so power users retain control.

## Code Changes

| File | Change |
|------|--------|
| `src/pivot/executor/core.py` | Set `PYTHONHASHSEED=0` before executor creation |
| `src/pivot/executor/worker.py` | Add `_set_deterministic_seeds()`, call before each stage |
