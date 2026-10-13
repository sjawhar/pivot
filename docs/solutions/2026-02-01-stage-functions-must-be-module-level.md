---
tags: [python, typing, multiprocessing, pickling]
category: gotcha
module: compose
symptoms: ["name 'X' is not defined when loading pipeline.py", "TypedDict return type not resolved", "stage configuration hidden in a closure"]
---

# Stage Functions and TypedDicts Belong at Module Level

## Problem

Two things go wrong when stage functions or their return `TypedDict`s are defined inside other functions.

**1. Type hint resolution can fail for nested TypedDicts with postponed annotations:**

```python
from __future__ import annotations


def create_pipeline():
    class TrainResult(TypedDict):
        model: Annotated[dict, loaders.Pickle()]
        metrics: Annotated[dict, metric]

    @stage
    def train(data: pd.DataFrame) -> TrainResult:  # NameError when resolving hints
        ...
```

Pivot reads a stage's outputs from its return annotation with `get_type_hints()` (`compose._analyze_return_type`). With postponed/string annotations, `get_type_hints()` resolves forward references in `func.__module__`'s namespace; for a nested definition, `TrainResult` exists only in the enclosing function's local scope. Loading `pipeline.py` then fails with:

```
Error: Failed to load …/pipeline.py: name 'TrainResult' is not defined
```

Without postponed annotations Python 3.13 can preserve a function-local `TypedDict`, but module-level definitions work in both modes and are the project convention.

**2. Configuration hidden in a closure:**

```python
def make_stage(threshold: float):
    @stage
    def keep(data: pd.DataFrame) -> pd.DataFrame:
        return data[data["value"] > threshold]

    return keep
```

This does run: workers receive the function through loky's cloudpickle, and the captured `threshold` is fingerprinted (changing it marks the stage "Code changed"). But the value bypasses `StageParams`: `pivot params show` does not list it, `params.yaml` cannot override it, and the lock file records it as code rather than as a parameter.

## Solution

Define stage functions, their return `TypedDict`s, their `StageParams` and custom loaders at module level, and pass configuration as `StageParams`:

```python
# stages/train.py
from typing import Annotated, TypedDict

import pandas as pd

from pivot import StageParams, loaders, metric, stage


class TrainResult(TypedDict):
    model: Annotated[dict, loaders.Pickle()]
    metrics: Annotated[dict, metric]


class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100


@stage
def train(data: pd.DataFrame, params: TrainParams) -> TrainResult:
    model = fit_model(data, params.learning_rate, params.epochs)
    return TrainResult(model=model, metrics={"accuracy": 0.95})
```

For test files, the same rule applies: put test stages and their `TypedDict`s at module level, and call them inside a `with Pipeline(...)` block in the test.

## Key Insight

Python's `get_type_hints()` does name resolution at call time, not definition time, and with postponed annotations looks names up in the function's `__module__` namespace. Nested class definitions are not there.

The rule: **if Pivot needs to introspect it, define it at module level; if it is configuration, put it in `StageParams`**. This covers:

- Stage functions
- Return `TypedDict`s
- Custom loaders
- `StageParams` subclasses

To run one stage function with several configurations, call it several times with different `StageParams` (a Python loop, optionally inside `pipeline.variant()`), not a factory.
