# Pipeline Configuration

Pivot pipelines can be defined programmatically in Python or declaratively in YAML. Stage functions use annotations to declare their dependencies and outputs.

**Discovery order:** `pipeline.py` → `pivot.yaml` → `pivot.yml`

## Programmatic Registration (Primary Method)

For all-Python pipelines, use `REGISTRY.register()` directly in a `pipeline.py` file:

```python
# pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.registry import REGISTRY
from pivot.stage_def import StageParams


# Define parameter classes
class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


# Define output types
class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("data/clean.csv", loaders.PathOnly())]


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics/train.json")]


# Define stage functions
def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data/raw.csv", loaders.CSV())],
) -> PreprocessOutputs:
    """Load raw data, clean it, return path to output."""
    clean_df = raw.dropna()
    out_path = pathlib.Path("data/clean.csv")
    clean_df.to_csv(out_path, index=False)
    return PreprocessOutputs(clean=out_path)


def train(
    params: TrainParams,
    data: Annotated[pandas.DataFrame, outputs.Dep("data/clean.csv", loaders.CSV())],
) -> TrainOutputs:
    """Train model with injected data and params."""
    model_path = pathlib.Path("models/model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    model_path.write_text(f"model_lr={params.learning_rate}")

    return TrainOutputs(
        model=model_path,
        metrics={"accuracy": 0.95, "loss": 0.05},
    )


# Register stages
REGISTRY.register(preprocess)
REGISTRY.register(train)
```

Run with: `pivot run`

### Single Output Shorthand

For stages with one output, annotate the return type directly:

```python
def transform(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> Annotated[pandas.DataFrame, outputs.Out("output.csv", loaders.CSV())]:
    return data.dropna()

REGISTRY.register(transform)
```

### Path Overrides

Override annotation paths at registration time:

```python
REGISTRY.register(
    train,
    dep_path_overrides={"data": "custom/input.csv"},
    out_path_overrides={"model": {"path": "custom/model.pkl"}},
)
```

### Matrix Stages (Variants)

Register variants manually for matrix-like behavior:

```python
for dataset in ["train", "test"]:
    REGISTRY.register(
        train,
        name=f"train@{dataset}",
        variant=dataset,
        dep_path_overrides={"data": f"data/{dataset}.csv"},
        out_path_overrides={"model": {"path": f"models/{dataset}_model.pkl"}},
    )
```

### Testing Stage Functions

Test functions directly without framework setup:

```python
def test_train():
    test_df = pandas.DataFrame({"value": [1, 2, 3]})
    params = TrainParams(learning_rate=0.5)
    result = train(params, test_df)
    assert "model" in result
    assert "metrics" in result
```

## YAML Configuration

Define pipelines in `pivot.yaml`:

```yaml
# pivot.yaml
stages:
  preprocess:
    python: stages.preprocess    # Module path to function
    deps:
      raw: data.csv              # Named dependencies (override annotation paths)
    outs:
      clean: processed.parquet   # Named outputs (override annotation paths)

  train:
    python: stages.train
    deps:
      data: processed.parquet
    outs:
      model: model.pkl
    metrics:
      metrics: metrics.json      # Metric outputs (git-tracked)
    params:
      learning_rate: 0.01
```

### YAML Schema

```yaml
stages:
  stage_name:
    python: module.function      # Required: function to call
    deps:                        # Optional: path overrides for deps
      dep_name: path/to/file
    outs:                        # Optional: path overrides for outputs
      out_name: path/to/output
    metrics:                     # Optional: metric outputs (git-tracked)
      metric_name: metrics.json
    plots:                       # Optional: plot outputs
      plot_name: plot.png
    params:                      # Optional: parameter overrides
      key: value
    mutex:                       # Optional: mutex groups
      - gpu
    cwd: subdir/                 # Optional: working directory
    matrix:                      # Optional: matrix expansion
      dimension:
        variant1: {}
        variant2: {}
```

### Matrix in YAML

```yaml
stages:
  train:
    python: stages.train
    deps:
      data: "data/${dataset}.csv"
    outs:
      model: "models/${model}_${dataset}.pkl"
    matrix:
      model: [bert, gpt]
      dataset: [train, test]
```

Generates: `train@bert_train`, `train@bert_test`, `train@gpt_train`, `train@gpt_test`

## Discovery Order

Pivot searches for pipeline definitions:

1. `pipeline.py` - Module calling `REGISTRY.register()`
2. `pivot.yaml` - YAML configuration
3. `pivot.yml` - YAML configuration (alternative extension)

The first method found is used.

## See Also

- [Defining Stages](stages.md) - Stage definition patterns
- [Output Types](outputs.md) - Output types and options
- [Parameters](parameters.md) - Parameter handling
