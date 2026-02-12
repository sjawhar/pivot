# Parameters

Stage parameters are the knobs you turn when experimenting: learning rates,
thresholds, feature flags. Pivot tracks them as first-class inputs so that
changing a parameter automatically invalidates downstream stages, just like
changing code or data would.

## Defining Parameters

Subclass `StageParams` (a Pydantic `BaseModel`) and add it as a typed
argument to your stage function:

```python
import pivot

class TrainParams(pivot.StageParams):
    learning_rate: float = 0.01
    batch_size: int = 32
    dropout: float = 0.1
```

```python
from typing import Annotated
from pandas import DataFrame

def train(
    params: TrainParams,
    data: Annotated[DataFrame, pivot.Dep("features.csv", pivot.loaders.CSV())],
) -> Annotated[object, pivot.Out("model.pkl", pivot.loaders.Pickle())]:
    model = fit(data, lr=params.learning_rate, bs=params.batch_size)
    return model
```

`StageParams` gives you Pydantic validation for free — type coercion,
range checks, custom validators all work as expected:

```python
import pydantic
import pivot

class GridParams(pivot.StageParams):
    n_splits: int = 5
    scoring: str = "accuracy"

    @pydantic.field_validator("n_splits")
    @classmethod
    def at_least_two(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_splits must be >= 2")
        return v
```

Pivot discovers the params argument automatically — the name doesn't
matter, only the type annotation. A stage can have at most one
`StageParams` parameter.

## Providing Values

### At Registration

Pass an instance or class when registering:

```python
pipeline = pivot.Pipeline("my_pipeline")

# Instance with specific values
pipeline.register(train, params=TrainParams(learning_rate=0.001))

# Class — instantiates with defaults
pipeline.register(train, params=TrainParams)

# Omit — Pivot infers the class from the type hint
pipeline.register(train)
```

All three forms work. When you omit `params`, Pivot reads the type hint,
instantiates the class with defaults, and uses that. If any field lacks a
default, Pivot raises a `ParamsError` at registration time.

### Overriding via params.yaml

Create `params.yaml` at the project root to override values without
touching code:

```yaml
train:
  learning_rate: 0.005
  batch_size: 64
```

Overrides are applied on top of the registered defaults at execution time.
Only the fields you list are changed; everything else keeps its default.

For **matrix stages** (name format `base@variant`), overrides cascade:

```yaml
# Base overrides apply to all variants
process:
  threshold: 0.5

# Variant-specific overrides layer on top
process@high:
  threshold: 0.9
```

## Change Detection

Pivot serialises parameters to a deterministic JSON dict and stores them in
the per-stage lock file. On the next run, the worker compares the current
params dict against the locked version. Any difference triggers re-execution.

This means:

- Adding a new field with a default → **no change** (same serialised dict)
- Changing a default value → **triggers re-run**
- Overriding a value in `params.yaml` → **triggers re-run**
- Changing the Pydantic schema (field type, validators) → detected via
  [fingerprinting](fingerprinting.md), not params comparison

## Inspecting Parameters

### `pivot params show`

Display current effective values (defaults + overrides merged):

```
$ pivot params show
$ pivot params show train        # specific stage
$ pivot params show --json       # machine-readable
```

### `pivot params diff`

Compare workspace parameters against `git HEAD`:

```
$ pivot params diff
$ pivot params diff train --md   # Markdown table output
```

This is useful before committing to see what parameter changes you're
about to lock in.

## Matrix Stages

When the same logic needs to run with different parameter sets, register
variants instead of duplicating stages:

```python
for variant, lr in [("fast", 0.01), ("slow", 0.001)]:
    pipeline.register(
        train,
        name=f"train@{variant}",
        params=TrainParams(learning_rate=lr),
        variant=variant,
    )
```

Each variant gets its own lock file and cache entry. The `params.yaml`
override rules described above let you tweak individual variants without
changing code.

## Relationship to Other Concepts

| Input type | Tracked by | Triggers re-run when |
|------------|------------|----------------------|
| Parameters | Lock file params dict | Value changes |
| Code | [Fingerprint](fingerprinting.md) manifest | AST changes |
| Data | Dependency hash | File content changes |

Parameters, code fingerprints, and dependency hashes are all inputs to the
[caching](caching.md) skip-detection algorithm. A stage is skipped only
when **all three** match the previous run.
