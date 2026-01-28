# Defining Pipelines

Pipelines can be defined declaratively in YAML or programmatically in Python.

## Discovery Order

Pivot searches for pipeline definitions in this order:

1. `pivot.yaml` - YAML configuration (most common)
2. `pivot.yml` - YAML configuration (alternative extension)
3. `pipeline.py` - Python module calling `REGISTRY.register()`

The first method found is used.

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

### Stage Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `python` | `str` | Module path to function (required) |
| `deps` | `dict[str, str]` | Named dependency path overrides |
| `outs` | `dict[str, str]` | Named output path overrides |
| `metrics` | `dict[str, str]` | Metric outputs (git-tracked) |
| `plots` | `dict[str, str]` | Plot outputs |
| `params` | `dict` | Stage parameters |
| `mutex` | `list[str]` | Mutex groups for exclusive execution |
| `cwd` | `str` | Working directory for paths |
| `matrix` | `dict` | Matrix expansion configuration |

### How YAML and Python Work Together

Your Python function's annotations define *what* the stage needs (types and default paths). The YAML file lets you override those paths without editing Python code.

- If YAML specifies a path, it overrides the annotation's default
- If YAML doesn't specify a path, the annotation's default is used
- YAML `deps:`/`outs:` keys must match the Python parameter/output names

## Python Registration

For all-Python pipelines, use `REGISTRY.register()` directly in a `pipeline.py` file:

```python
# pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.registry import REGISTRY
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("data/clean.csv", loaders.PathOnly())]


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics/train.json")]


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

### Path Overrides at Registration

Override annotation paths at registration time:

```python
REGISTRY.register(
    train,
    dep_path_overrides={"data": "custom/input.csv"},
    out_path_overrides={"model": {"path": "custom/model.pkl"}},
)
```

### Matrix Stages via Python

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

## Single Output Shorthand

For stages with **one output**, annotate the return type directly:

```python
def transform(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> Annotated[pandas.DataFrame, outputs.Out("output.csv", loaders.CSV())]:
    return data.dropna()
```

For stages with **multiple outputs**, use a TypedDict:

```python
class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]

def train(...) -> TrainOutputs:
    return {"model": model_path, "metrics": metrics_dict}
```

## Function Requirements

Stage functions must be **pure and serializable** for multiprocessing.

### Why Serialization Matters

Pivot uses `ProcessPoolExecutor` with `forkserver` context for true parallelism. Worker processes are separate Python interpreters that receive serialized (pickled) functions. This means:

1. Functions must be defined at module level (not inside other functions)
2. Functions cannot capture local variables (closures)
3. The module containing the function must be importable

### Valid Stage Functions

```python
# Module-level function - works
def process_data(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> ProcessOutputs:
    ...
```

### Invalid Stage Functions

```python
# Lambda - not picklable
process = lambda: ...

# Closure captures variable
def make_stage(threshold):
    def process():
        if value > threshold:  # Captures threshold!
            pass
    return process

# Defined in __main__
if __name__ == '__main__':
    def my_stage():  # Can't be pickled!
        pass
```

Use parameters instead of closures to configure stage behavior.

!!! warning "Pickle Error"
    If you see `Could not pickle the task to send it to the workers`, your function captures a variable from its enclosing scope. Move the function to module level and pass values through parameters instead.

## Mutex Groups

Prevent stages from running concurrently:

```yaml
stages:
  train_gpu:
    python: stages.train_gpu
    deps:
      data: data.csv
    outs:
      model: gpu_model.pkl
    mutex:
      - gpu

  train_gpu_2:
    python: stages.train_gpu_2
    deps:
      data: data.csv
    outs:
      model: gpu_model_2.pkl
    mutex:
      - gpu   # Won't run at same time as train_gpu
```

## Working Directory

Set a working directory for path resolution:

```yaml
stages:
  process:
    python: stages.process
    deps:
      data: data.csv
    outs:
      output: output.csv
    cwd: subproject/
```

## Testing Stage Functions

Stage functions are directly testable without framework setup:

```python
def test_train():
    test_df = pandas.DataFrame({"value": [1, 2, 3]})
    params = TrainParams(learning_rate=0.5)
    result = train(params, test_df)
    assert "model" in result
    assert "metrics" in result
```

## See Also

- [Dependencies & Loaders](dependencies.md) - Declaring inputs
- [Output Types](outputs.md) - Output types and caching
- [Parameters](parameters.md) - Parameter handling
- [Matrix Stages](matrix.md) - Creating stage variants
