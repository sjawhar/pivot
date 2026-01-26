# Defining Stages

Stages are defined in `pivot.yaml` which references Python functions. The functions use annotations to declare their dependencies and outputs.

## Basic Usage

```yaml
# pivot.yaml
stages:
  process:
    python: stages.process
    deps:
      raw: input.csv
    outs:
      clean: output.parquet
```

```python
# stages.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class ProcessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("output.parquet", loaders.PathOnly())]


def process(
    raw: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> ProcessOutputs:
    df = raw.dropna()
    out_path = pathlib.Path("output.parquet")
    df.to_parquet(out_path)
    return {"clean": out_path}
```

## YAML Stage Parameters

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

## Dependencies

Dependencies are declared as annotated function parameters:

```python
def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data/raw.csv", loaders.CSV())],
    config: Annotated[dict, outputs.Dep("config/settings.yaml", loaders.YAML())],
) -> PreprocessOutputs:
    ...
```

YAML provides path overrides (must match parameter names):

```yaml
stages:
  preprocess:
    python: stages.preprocess
    deps:
      raw: data/raw.csv
      config: config/settings.yaml
```

Pivot automatically tracks Python code dependencies. You don't need to list `.py` files in `deps`.

## Outputs

Outputs are declared in the return type annotation:

```python
class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]
    plot: Annotated[pathlib.Path, outputs.Plot("loss.png", loaders.PathOnly())]


def train(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    # ... training code ...
    return {"model": model_path, "metrics": metrics_dict, "plot": plot_path}
```

YAML provides path overrides:

```yaml
stages:
  train:
    python: stages.train
    deps:
      data: data.csv
    outs:
      model: model.pkl
    metrics:
      metrics: metrics.json
    plots:
      plot: loss.png
```

See [Output Types](outputs.md) for details on each type.

## Parameters

Parameters are declared as a `StageParams` subclass:

```python
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    learning_rate: float = 0.01
    batch_size: int = 32


def train(
    params: TrainParams,
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    print(f"Learning rate: {params.learning_rate}")
    ...
```

YAML overrides defaults:

```yaml
stages:
  train:
    python: stages.train
    params:
      learning_rate: 0.05
      batch_size: 64
```

See [Parameters](parameters.md) for details.

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

## Function Requirements

Stage functions must be **pure and serializable** for multiprocessing.

### Why Serialization Matters

Pivot uses `ProcessPoolExecutor` with `forkserver` context for true parallelism. Worker processes are separate Python interpreters that receive serialized (pickled) functions. This means:

1. Functions must be defined at module level (not inside other functions)
2. Functions cannot capture local variables (closures)
3. The module containing the function must be importable

### Do

```python
# Module-level function in importable module
# stages.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class ProcessOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


def process_data(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> ProcessOutputs:
    out_path = pathlib.Path("output.csv")
    data.to_csv(out_path)
    return {"output": out_path}
```

### Don't

```python
# Lambda (not picklable)
# Error: Can't pickle <lambda>
process = lambda: ...

# Closure capturing local variable
# Error: Could not pickle the task to send it to the workers
def make_stage(threshold):
    def process():
        if value > threshold:  # Captures threshold!
            pass
    return process

# Defined in __main__
# Error: Can't get attribute 'my_stage' on <module '__main__'>
if __name__ == '__main__':
    def my_stage():  # Can't be pickled!
        pass
```

Use parameters instead of closures to configure stage behavior.

> **Error you'll see:** `Could not pickle the task to send it to the workers`
>
> This means your function captures a variable from its enclosing scope.
> Move the function to module level and pass values through parameters instead.

See [Troubleshooting](troubleshooting.md) for more common errors and solutions.

## See Also

- [Configuration](configuration.md) - Pipeline configuration
- [Output Types](outputs.md) - Output types and options
- [Parameters](parameters.md) - Parameter handling
