---
tags: [python, fingerprinting, lambda, caching]
category: gotcha
module: fingerprint
symptoms: ["stages re-run every session", "fingerprint changes without code changes", "cache invalidation on restart"]
---

# Fingerprinting Callables Without Source

## Problem

Lambdas created via `eval()`, `exec()`, or in interactive sessions have no retrievable source, so `inspect.getsource()` fails and the AST hash is unavailable. `fingerprint._compute_function_hash()` then falls back in two steps:

```python
try:
    source = inspect.getsource(func)
except (OSError, TypeError):
    if hasattr(func, "__code__"):
        # marshal captures bytecode and constants; the source path is replaced first
        return _hash_code_object(func.__code__)
    # no source and no code object: id(func), which differs between sessions
    return xxhash.xxh64(str(id(func)).encode()).hexdigest()
```

- A lambda has `__code__`, so it is hashed by `marshal.dumps` of its code object after `_replace_code_filenames` swaps the source path for a fixed name. The hash is the same in every interpreter session, and editing the expression (`x + 1` to `x + 100`) changes it. `test_fingerprint_gaps.py::test_eval_lambda_is_hashed_by_its_code_object` checks both across `pivot repro` runs.
- Only a callable with neither source nor `__code__` (a callable instance of a class whose source is unavailable) reaches the `id(func)` fallback. Its fingerprint changes every session, so its stage reruns every time; Pivot logs a warning when that callable is user code.

## Solution

Use named module-level functions for stages and the helpers they call. A code-object hash is stable, but it is sensitive to anything the compiler sees, while the AST hash ignores formatting and comments:

```python
import pandas as pd

from pivot import Pipeline, stage


@stage
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()


with Pipeline("my_pipeline") as pipeline:
    process_data(pipeline.input("input.csv"))
```

## Key Insight

`inspect.getsource()` needs a filename and line numbers. Code without them is still fingerprinted from its code object, which is portable once the embedded source path is removed; only an object with no code object at all has nothing stable to hash.
