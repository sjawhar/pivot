---
tags: [python, typing, multiprocessing, pickling]
category: gotcha
module: registry
symptoms: ["NameError in get_type_hints", "cannot pickle local object", "KeyError in type resolution"]
---

# Stage Functions and TypedDicts Must Be Module-Level

## Problem

Pivot's stage definition system fails when stage functions or their return TypedDicts are defined inside other functions. Two mechanisms break:

**1. Type hint resolution fails:**

```python
def create_pipeline():
    class TrainOutputs(TypedDict):
        model: Annotated[Path, Out("model.pkl", Pickle())]

    def train() -> TrainOutputs:  # NameError when resolving hints
        ...
```

`get_type_hints()` resolves forward references by looking up names in `func.__module__`'s namespace. For nested definitions, `TrainOutputs` exists only in the enclosing function's local scope, not the module's global namespace. When Pivot calls `get_type_hints(train)`, it searches `sys.modules[train.__module__].__dict__` and fails to find `TrainOutputs`.

The actual error from `_get_type_hints_safe()` in `stage_def.py`:

```
Failed to resolve type hints for train: name 'TrainOutputs' is not defined
```

**2. Multiprocessing pickling fails:**

```python
def create_pipeline():
    def train():  # Cannot pickle - not importable
        return {"model": model_data}

    registry.register("train", train)
```

loky (Pivot's process pool) serializes functions by reference: `module_name.function_name`. For a nested function, there's no importable path - you can't do `from mymodule import create_pipeline.<locals>.train`. cloudpickle attempts to serialize the bytecode, but this often fails for closures that reference outer-scope variables.

```
_pickle.PicklingError: Can't pickle <function create_pipeline.<locals>.train>:
it's not found as mymodule.create_pipeline.<locals>.train
```

## Solution

Define all stage functions and their return TypedDicts at module level:

```python
# stages/train.py

from typing import Annotated, TypedDict
from pathlib import Path
from pivot import loaders, outputs
from pivot.stage_def import StageParams

# TypedDict at module level - resolvable by get_type_hints()
class TrainOutputs(TypedDict):
    model: Annotated[Path, outputs.Out("model.pkl", loaders.Pickle())]
    metrics: Annotated[dict[str, float], outputs.Out("metrics.json", loaders.JSON())]

class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100

# Function at module level - picklable by reference
def train(
    params: TrainParams,
    data: Annotated[pd.DataFrame, Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    model = fit_model(data, params.learning_rate, params.epochs)
    return TrainOutputs(
        model=model,
        metrics={"accuracy": 0.95},
    )
```

For test files, the same rule applies:

```python
# tests/test_stages.py

# Module-level TypedDict for test stage output
class _TestOutputs(TypedDict):
    result: Annotated[str, outputs.Out("result.txt", loaders.Text())]

# Module-level test stage function
def _test_stage() -> _TestOutputs:
    return _TestOutputs(result="test output")

def test_stage_execution():
    # Use the module-level definitions in tests
    manifest = fingerprint.get_stage_fingerprint(_test_stage)
    assert "result.txt" in str(manifest)
```

## Key Insight

Python's `get_type_hints()` does name resolution at call time, not definition time. It looks up type names in the function's `__module__` namespace, which for nested definitions doesn't include local class definitions. Similarly, multiprocessing pickles functions by their importable path (`module.name`), which doesn't exist for closures.

The rule is simple: **if Pivot needs to introspect or serialize it, define it at module level**. This includes:

- Stage functions
- Return TypedDicts
- Custom loaders
- StageParams subclasses

Factory functions that dynamically create stages are incompatible with Pivot's architecture. If you need parameterized stages, use `StageParams` with configuration, not factory patterns.
