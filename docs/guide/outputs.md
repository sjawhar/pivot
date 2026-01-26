# Output Types

Pivot provides several output types for different use cases.

## Overview

| Type | Cached | Git-Tracked | Use Case |
|------|--------|-------------|----------|
| `Out` | Yes | No | Large data files, models |
| `Metric` | No | Yes | Small JSON metrics |
| `Plot` | Yes | No | Visualization files |
| `IncrementalOut` | Yes | No | Append-only files |

## Defining Outputs

Outputs are declared in the function's return type using a TypedDict with annotated fields:

```python
import pathlib
from typing import Annotated, TypedDict

from pivot import loaders, outputs


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]
    plot: Annotated[pathlib.Path, outputs.Plot("loss.png", loaders.PathOnly())]


def train(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    # ... training code ...
    return {
        "model": model_path,
        "metrics": {"accuracy": 0.95},
        "plot": plot_path,
    }
```

YAML provides path overrides:

```yaml
stages:
  train:
    python: stages.train
    outs:
      model: models/model.pkl
    metrics:
      metrics: metrics/train.json
    plots:
      plot: plots/loss.png
```

## Regular Outputs (`Out`)

Regular cached output for large files:

```python
class ProcessOutputs(TypedDict):
    data: Annotated[pandas.DataFrame, outputs.Out("data.parquet", loaders.CSV())]
```

Options (set on the `Out` instance):
- `cache=True` (default) - Store in content-addressable cache
- `persist=False` (default) - Keep in cache after workspace cleanup

## Metrics

Small files tracked in git (not cached):

```python
class TrainOutputs(TypedDict):
    metrics: Annotated[dict, outputs.Metric("metrics.json")]


def train(...) -> TrainOutputs:
    metrics = {'accuracy': 0.95, 'loss': 0.05}
    return {"metrics": metrics}  # Automatically saved as JSON
```

Use metrics for:
- Training metrics (accuracy, loss, F1)
- Data statistics (row counts, distributions)
- Any small JSON you want to track in git

View metrics:

```bash
pivot metrics show
pivot metrics diff  # Compare with git HEAD
```

## Plots

Visualization files that you create manually:

```python
class TrainOutputs(TypedDict):
    plot: Annotated[pathlib.Path, outputs.Plot("loss.png", loaders.PathOnly())]


def train(...) -> TrainOutputs:
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plot_path = pathlib.Path("loss.png")
    plt.savefig(plot_path)
    return {"plot": plot_path}
```

View plots:

```bash
pivot plots show          # Generate HTML gallery
pivot plots show --open   # Open in browser
pivot plots diff          # Show which plots changed
```

## Incremental Outputs

Outputs that preserve state between runs:

```python
class AppendOutputs(TypedDict):
    database: Annotated[dict, outputs.IncrementalOut("cache.json", loaders.JSON())]


def append_records(...) -> AppendOutputs:
    # database.json is restored from cache BEFORE execution
    # Stage can modify it rather than recreating
    existing_data = ...  # Loaded from cache
    existing_data["new_key"] = new_value
    return {"database": existing_data}
```

**How it works:**

1. Before execution, previous version is restored from cache
2. Stage modifies the data
3. New version is cached after execution
4. Uses COPY mode (not symlinks) so writes are safe

Use cases:
- Append-only databases
- Cumulative logs
- Incremental data processing

## Single Output Shorthand

For functions with a single output, annotate the return type directly:

```python
def transform(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> Annotated[pandas.DataFrame, outputs.Out("output.csv", loaders.CSV())]:
    return data.dropna()
```

## Mixing Output Types

```python
class FullOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    cache: Annotated[dict, outputs.IncrementalOut("cache.json", loaders.JSON())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]
    plot: Annotated[pathlib.Path, outputs.Plot("loss.png", loaders.PathOnly())]
```

```yaml
stages:
  train:
    python: stages.train
    outs:
      model: models/model.pkl
      cache: cache/train.json
    metrics:
      metrics: metrics/train.json
    plots:
      plot: plots/loss.png
```

## See Also

- [Defining Stages](stages.md) - Stage definition patterns
- [Configuration](configuration.md) - Pipeline configuration
