# Quick Start

This guide walks you through creating and running your first Pivot pipeline.

## 1. Create a Pipeline

Create `pivot.yaml`:

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
```

Create `stages.py`:

```python
# stages.py
import pathlib
import pickle
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("processed.parquet", loaders.PathOnly())]


def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    """Load and clean the data."""
    df = raw.dropna()
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return {"clean": out_path}


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


def train(
    data: Annotated[pathlib.Path, outputs.Dep("processed.parquet", loaders.PathOnly())],
) -> TrainOutputs:
    """Train a simple model."""
    df = pandas.read_parquet(data)
    model = {'rows': len(df), 'cols': len(df.columns)}
    model_path = pathlib.Path("model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return {"model": model_path}
```

Stage functions must be defined at module level (not inside `if __name__ == '__main__':`) because Pivot uses multiprocessing and needs to serialize functions to worker processes.

> **Single vs Multiple Outputs**
>
> For stages with **one output**, annotate the return type directly:
> ```python
> def preprocess(
>     raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
> ) -> Annotated[pandas.DataFrame, outputs.Out("processed.csv", loaders.CSV())]:
>     return raw.dropna()
> ```
>
> For stages with **multiple outputs**, use a TypedDict (shown above).

> **How YAML and Python Work Together**
>
> Your Python function's annotations define *what* the stage needs (types and default paths).
> The YAML file lets you override those paths without editing Python code.
>
> - If YAML specifies a path, it overrides the annotation's default
> - If YAML doesn't specify a path, the annotation's default is used
> - YAML `deps:`/`outs:` keys must match the Python parameter/output names

## 2. Create Sample Data

```bash
echo "name,value
Alice,100
Bob,200
Charlie," > data.csv
```

## 3. Run the Pipeline

```bash
pivot run
```

Pivot will:

1. Discover `pivot.yaml` automatically
2. Build a dependency graph
3. Execute stages in the correct order
4. Cache outputs for future runs

## 4. Re-run (Cached)

```bash
pivot run
```

The second run completes instantly because nothing changed.

## 5. Modify and Re-run

Edit `stages.py` to change the `preprocess` function:

```python
def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    df['doubled'] = df['value'] * 2  # New line!
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return {"clean": out_path}
```

```bash
pivot run
```

Pivot automatically detects the code change and re-runs both stages.

## 6. See Why Stages Run

```bash
pivot explain
```

Shows detailed breakdown of what changed and why each stage would run.

## 7. Dry Run

Preview what would run without executing:

```bash
pivot dry-run
```

## Next Steps

- [Core Concepts](concepts.md) - Understand stages, dependencies, and caching
- [Defining Stages](../guide/stages.md) - Deep dive into stage definition
- [Output Types](../guide/outputs.md) - Learn about outputs, metrics, and plots
