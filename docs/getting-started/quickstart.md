# Quick Start

This guide walks you through creating and running your first Pivot pipeline.

## Mental Model

Think **artifact-first**, not **stage-first**. The DAG emerges from artifact dependencies:

- **Wrong:** "Stage A triggers Stage B"
- **Right:** "This file changed. What needs to happen because of that?"

Invalidation is content-addressed: same inputs + same code = same outputs.

## 1. Initialize the Project

```bash
pivot init
```

This creates:
- `.pivot/` - Directory for cache and state
- `.pivotignore` - Patterns for files to exclude from watching

## 2. Create a Pipeline

Create `pipeline.py`:

```python
# pipeline.py
import pathlib
import pickle
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.pipeline import Pipeline

pipeline = Pipeline("my_pipeline")


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("processed.parquet", loaders.PathOnly())]


def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    """Load and clean the data."""
    df = raw.dropna()
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return PreprocessOutputs(clean=out_path)


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
    return TrainOutputs(model=model_path)


# Register stages - Pivot discovers deps/outs from annotations
pipeline.register(preprocess)
pipeline.register(train)
```

## 3. Create Sample Data

```bash
echo "name,value
Alice,100
Bob,200
Charlie," > data.csv
```

## 4. Run the Pipeline

```bash
pivot repro
```

Pivot will:

1. Discover `pipeline.py` and import it (which registers stages)
2. Build a dependency graph from the annotations
3. Execute stages in the correct order
4. Cache outputs for future runs

## 5. Re-run (Cached)

```bash
pivot repro
```

The second run completes instantly because nothing changed.

## 6. Modify and Re-run

Edit `pipeline.py` to change the `preprocess` function:

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
pivot repro
```

Pivot automatically detects the code change and re-runs both stages.

## 7. See Why Stages Run

```bash
pivot status --explain
```

Shows detailed breakdown of what changed and why each stage would run.

## A Note on Loaders

In the examples above, `Dep()` and `Out()` take a loader like `loaders.CSV()` or `loaders.PathOnly()`. These loaders implement the `Reader` and `Writer` protocols respectively. All built-in loaders implement both, so you can use them interchangeably with `Dep` and `Out`.

## Next Steps

- [Watch Mode & Rapid Iteration](../tutorial/watch.md) - Develop faster with auto-rerun
- [Defining Pipelines](../reference/pipelines.md) - Deep dive into stage definition
- [Output Types](../reference/outputs.md) - Learn about outputs, metrics, and plots

> **Project Structure**: For larger projects, consider using [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) as a starting template. Its `data/raw/`, `data/processed/`, and `src/` layout works well with Pivot.
