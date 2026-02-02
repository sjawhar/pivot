---
tags: [python, fingerprinting, naming-conventions]
category: gotcha
module: fingerprint
symptoms: ["unexpected re-runs when private function changes", "confusion about what gets fingerprinted"]
---

# Single Underscore Functions ARE Tracked in Fingerprints

## Problem

When a private helper function (prefixed with single underscore `_`) changes, stages that call it re-run. This surprises developers who assume "private" means "not tracked."

```python
def _normalize_data(df: DataFrame) -> DataFrame:
    """Private helper - changes here WILL trigger re-runs."""
    return df.dropna()  # Any change here invalidates dependent stages

def train(
    params: TrainParams,
    data: Annotated[DataFrame, Dep("input.csv", CSV())],
) -> Annotated[DataFrame, Out("output.csv", CSV())]:
    return _normalize_data(data)
```

Changing `_normalize_data` (even just whitespace outside strings) causes `train` to re-run because the helper's AST is included in `train`'s fingerprint.

## Solution

This is intentional, not a bug. Pivot's fingerprinting correctly tracks single-underscore functions because they affect behavior.

The filtering logic in `fingerprint.py` explicitly only skips **dunders** (double-underscore names like `__name__`, `__init__`):

```python
# From _process_closure_values()
if skip_dunders and name.startswith("__"):
    continue  # Only filters dunders, not single-underscore
```

Dunders are filtered because:
- `__name__`, `__doc__`, `__module__` are metadata, not behavior
- `__init__` on builtins/stdlib types is not user code
- These are injected by Python, not explicitly referenced in your code

Single-underscore functions (`_helper`, `_normalize`, `_validate`) are tracked because:
- They contain implementation logic that affects outputs
- Changing them should invalidate caches (correct behavior)
- "Private" is a convention for API consumers, not a signal about code impact

**If you don't want a helper tracked**, move it to a separate module that isn't imported by your stage. But this is rarely the right choice - if changing the helper changes your output, you want re-runs.

## Key Insight

Python's underscore prefix conventions have no bearing on fingerprinting. The single underscore (`_func`) is an API hint meaning "internal, don't call directly" - it says nothing about whether code changes matter.

Pivot tracks all user code that could affect stage outputs:
- Single underscore (`_helper`) = **tracked** (implementation detail that matters)
- Double underscore dunder (`__name__`) = **not tracked** (Python-injected metadata)

If a function is in your stage's closure and could affect the result, it should be fingerprinted. The naming convention doesn't change that.
