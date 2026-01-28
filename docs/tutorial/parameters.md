# Parameters & Experiments

This tutorial shows how to add parameters to your stages for running experiments with different configurations.

## Prerequisites

Complete the [Quick Start](../getting-started/quickstart.md). You should have a working pipeline with preprocess and train stages.

## Add Parameters to a Stage

Update `stages.py` to add a `TrainParams` class:

```python
# stages.py
import pathlib
import pickle
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.stage_def import StageParams


# ... preprocess function stays the same ...


class TrainParams(StageParams):
    """Training hyperparameters."""
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]


def train(
    params: TrainParams,
    data: Annotated[pathlib.Path, outputs.Dep("processed.parquet", loaders.PathOnly())],
) -> TrainOutputs:
    """Train a model with configurable parameters."""
    df = pandas.read_parquet(data)

    # Use parameters in your training logic
    model = {
        'rows': len(df),
        'cols': len(df.columns),
        'learning_rate': params.learning_rate,
        'epochs': params.epochs,
        'batch_size': params.batch_size,
    }

    model_path = pathlib.Path("model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Return metrics for tracking
    return {
        "model": model_path,
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05,
            "learning_rate": params.learning_rate,
        }
    }
```

Note the changes:

1. Import `StageParams` from `pivot.stage_def`
2. Define a Pydantic model extending `StageParams`
3. Add `params: TrainParams` as the first parameter
4. Added a `Metric` output for tracking

## Override Parameters in YAML

Update `pivot.yaml` to override defaults:

```yaml
# pivot.yaml
stages:
  preprocess:
    python: stages.preprocess
    deps:
      raw: data.csv
    outs:
      clean: processed.parquet

  train:
    python: stages.train
    deps:
      data: processed.parquet
    outs:
      model: model.pkl
    metrics:
      metrics: metrics.json
    params:
      learning_rate: 0.05  # Override default
      epochs: 200
```

## Run and See Parameter Change Detection

```bash
pivot run
```

Now change a parameter in `pivot.yaml`:

```yaml
    params:
      learning_rate: 0.001  # Changed!
      epochs: 200
```

```bash
pivot explain train
```

Output shows what changed:

```
Stage: train
  Status: WILL RUN
  Reason: Parameters changed

  Param changes:
    learning_rate: 0.05 -> 0.001
```

## View Current Parameters

```bash
# Show all parameters
pivot params show

# JSON output for scripting
pivot params show --json

# Specific stage
pivot params show train
```

## Compare Parameters

```bash
# Compare with git HEAD
pivot params diff
```

## Parameter Precedence

Parameters can come from multiple sources. Precedence (highest to lowest):

1. **`params.yaml`** at project root (git-ignored for local experiments)
2. **`pivot.yaml`** `params:` section
3. **Python `StageParams` defaults**

Create `params.yaml` for local overrides:

```yaml
# params.yaml - git-ignore this file
train:
  learning_rate: 0.001
  epochs: 10  # Quick local test
```

This lets you:

- Define sensible defaults in Python
- Configure experiments in `pivot.yaml` (committed)
- Override for local testing via `params.yaml` (not committed)

## Add More Metrics

Metrics are git-tracked JSON files, perfect for experiment tracking:

```python
def train(params: TrainParams, ...) -> TrainOutputs:
    # ... training code ...

    return {
        "model": model_path,
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05,
            "f1_score": 0.93,
            "params": {
                "learning_rate": params.learning_rate,
                "epochs": params.epochs,
            }
        }
    }
```

View metrics:

```bash
pivot metrics show
pivot metrics diff  # Compare with git HEAD
```

## Testing Parameterized Stages

Stage functions are directly testable:

```python
def test_train():
    # Create test data
    test_df = pandas.DataFrame({"value": [1, 2, 3]})
    test_path = pathlib.Path("test_processed.parquet")
    test_df.to_parquet(test_path)

    # Test with specific parameters
    params = TrainParams(learning_rate=0.5, epochs=10, batch_size=16)
    result = train(params, test_path)

    assert "model" in result
    assert result["metrics"]["learning_rate"] == 0.5
```

## Next Steps

- [Remote Storage & CI](remote.md) - Share caches and run in CI
- [Parameters Reference](../reference/parameters.md) - Full parameter documentation
- [Matrix Stages](../reference/matrix.md) - Run same stage with different parameter combinations
