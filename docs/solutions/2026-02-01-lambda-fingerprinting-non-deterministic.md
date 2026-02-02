---
tags: [python, fingerprinting, lambda, caching]
category: gotcha
module: fingerprint
symptoms: ["stages re-run every session", "fingerprint changes without code changes", "cache invalidation on restart"]
---

# Lambda Fingerprinting Is Non-Deterministic

## Problem

Lambdas defined inline (especially those created via `eval()`, `exec()`, or in interactive sessions) often lack retrievable source code. When `inspect.getsource()` fails, the fingerprinting system falls back to `id(func)`, which is a memory address that changes every interpreter session.

```python
# This lambda has no recoverable source file
dynamic_lambda = eval("lambda x: x * 2")

# Fingerprinting falls back to id(func)
hash1 = fingerprint.hash_function_ast(dynamic_lambda)

# After restart, id() returns different value
# hash2 != hash1  -- stage re-runs unnecessarily
```

The fallback path in `fingerprint._compute_function_hash()`:

```python
try:
    source = inspect.getsource(func)
except (OSError, TypeError):
    if hasattr(func, "__code__"):
        # Uses marshal.dumps(__code__) - deterministic within same session
        return xxhash.xxh64(marshal.dumps(func.__code__)).hexdigest()
    # KNOWN ISSUE: id(func) is non-deterministic across runs
    return xxhash.xxh64(str(id(func)).encode()).hexdigest()
```

Even when `__code__` exists (most lambdas have it), the `marshal.dumps(__code__)` approach is only deterministic **within** a session. Across interpreter restarts, the serialized code object may differ due to:

1. Memory layout changes affecting object identity
2. Code object internals that vary by interpreter session
3. Different compilation paths for dynamically created code

Lambdas defined in actual source files (like `my_lambda = lambda x: x * 2` in a `.py` file) **do** have source available and fingerprint correctly. The issue affects dynamically created lambdas.

## Solution

Use named module-level functions instead of lambdas in stage definitions:

```python
from typing import Annotated
import pandas as pd
from pivot import Dep, Out
from pivot.loaders import CSV
from pivot.pipeline import Pipeline

# GOOD: Named module-level function has stable fingerprint
def process_data(
    data: Annotated[pd.DataFrame, Dep("input.csv", CSV())],
) -> Annotated[pd.DataFrame, Out("output.csv", CSV())]:
    return data.dropna()

pipeline = Pipeline("my_pipeline")
pipeline.register(process_data)
```

**Key requirement:** Stage functions must be module-level named functions, not lambdas or closures. This ensures `inspect.getsource()` can retrieve the source code for deterministic fingerprinting.

For Pydantic model defaults, use named functions for `default_factory`:

```python
# BAD: Lambda default_factory
class Params(pydantic.BaseModel):
    items: list[str] = pydantic.Field(default_factory=lambda: ["a", "b"])

# GOOD: Named function default_factory
def default_items():
    return ["a", "b"]

class Params(pydantic.BaseModel):
    items: list[str] = pydantic.Field(default_factory=default_items)
```

For callbacks and transformations passed to stages:

```python
# BAD: Inline lambda in stage parameter
def train(
    params: TrainParams,
    transform: Callable = lambda x: x.lower(),  # Unstable
):
    ...

# GOOD: Module-level named function
def default_transform(x):
    return x.lower()

def train(
    params: TrainParams,
    transform: Callable = default_transform,  # Stable
):
    ...
```

## Key Insight

Python's `inspect.getsource()` needs a filename and line numbers to retrieve source code. Lambdas created dynamically (via `eval`, `exec`, or interactive REPL) don't have this metadata, forcing fingerprinting to fall back to non-deterministic hashing. Named functions defined in source files always have this metadata, making their fingerprints stable across interpreter sessions.

The broader principle: anything that participates in fingerprinting must be defined in a way that makes its source recoverable. This means module-level definitions in `.py` files, not inline constructs in dynamic contexts.
