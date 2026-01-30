# Parameters

Pivot supports parameters defined as Pydantic models for type-safe stage configuration.

## Basic Usage

Define a `StageParams` subclass and use it as a function parameter:

```python
# pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


def train(
    params: TrainParams,
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    print(f"Training with lr={params.learning_rate}")
    print(f"Epochs: {params.epochs}")
    ...
```

Override defaults at runtime using `params.yaml` at your project root:

```yaml
# params.yaml - git-ignore this file
train:
  learning_rate: 0.05
  epochs: 200
```

## Parameter Precedence

Parameters can come from multiple sources. Here's the precedence (highest to lowest):

1. **`params.yaml`** file at project root (for runtime overrides)
2. **Python `StageParams` defaults**

Example:

```python
class TrainParams(StageParams):
    learning_rate: float = 0.01  # Default (committed)
```

```yaml
# params.yaml
train:
  learning_rate: 0.001  # Overrides Python default (not committed)
```

This layering lets you:

- Define sensible defaults in Python (committed, type-checked)
- Override for local testing via `params.yaml` (git-ignored)

## Parameter Change Detection

Pivot tracks parameter changes and re-runs stages when parameters change:

```bash
# After changing params in Python or params.yaml
$ pivot status --explain train
Stage: train
  Status: WILL RUN
  Reason: Parameters changed

  Param changes:
    learning_rate: 0.01 -> 0.005
```

## Viewing Parameters

```bash
# Show current parameter values
pivot params show

# JSON output
pivot params show --json

# Compare with git HEAD
pivot params diff
```

## Matrix Stage Parameters

When using `pivot.yaml` for matrix expansion, each variant can have different parameters:

```yaml
# pivot.yaml (for matrix expansion)
stages:
  train:
    python: stages.train
    deps:
      data: data/${dataset}.csv
    outs:
      model: models/${model}.pkl
    params:
      epochs: 100
    matrix:
      model:
        small:
          params:
            epochs: 10
        large:
          params:
            epochs: 1000
      dataset: [train, test]
```

See [Matrix Stages](matrix.md) for more details on YAML matrix expansion.

## Testing with Parameters

Stage functions are directly testable:

```python
def test_train():
    params = TrainParams(learning_rate=0.5, epochs=10)
    test_data = pandas.DataFrame({"value": [1, 2, 3]})
    result = train(params, test_data)
    assert "model" in result
```

## Troubleshooting

### Parameters Not Taking Effect

**Symptom:** Changed parameters but stage doesn't re-run.

**Cause:** Possibly editing wrong file or precedence issue.

**Solution:** Check active parameter values and precedence:

```bash
# Show what values are actually used
pivot params show train

# Check explain output
pivot status --explain train
```

Remember: `params.yaml` > Python defaults

## See Also

- [Defining Pipelines](pipelines.md) - Stage definition patterns
- [Matrix Stages](matrix.md) - Parameter variations
