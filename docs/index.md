# Pivot

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Define stages with typed Python functions and annotations, and Pivot figures out what needs to re-run—no manual dependency declarations, no stale caches.

```bash
pivot repro      # Run your pipeline
# edit a helper function...
pivot repro      # Pivot detects the change and re-runs affected stages
```

## Quick Example

```python
# pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs
from pivot.pipeline import Pipeline


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("processed.parquet", loaders.PathOnly())]


def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return PreprocessOutputs(clean=out_path)


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


def train(
    data: Annotated[pathlib.Path, outputs.Dep("processed.parquet", loaders.PathOnly())],
) -> TrainOutputs:
    df = pandas.read_parquet(data)
    model_path = pathlib.Path("model.pkl")
    # ... train model ...
    return TrainOutputs(model=model_path)


# Register stages - Pivot discovers deps/outs from annotations
pipeline = Pipeline("my_pipeline")
pipeline.register(preprocess)
pipeline.register(train)
```

```bash
pivot repro  # Runs both stages
pivot repro  # Instant - nothing changed
```

Modify `preprocess`, and Pivot automatically re-runs both stages. Modify `train`, and only `train` re-runs.

## What Makes Pivot Different

### Automatic Code Change Detection

Change a helper function, and Pivot knows to re-run stages that call it:

```python
def normalize(x):
    return x / x.max()  # Change this...

def process(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> ProcessOutputs:
    return {"result": normalize(data)}  # ...and Pivot re-runs process
```

No YAML to update (for code changes). No manual declarations. Pivot parses your Python and tracks what each stage actually calls.

### See Why Stages Run

```bash
$ pivot status --explain train

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Changes:
    func:normalize
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

### Watch Mode

Edit code, save, see results:

```bash
pivot repro --watch  # Re-runs automatically on file changes
```

## Getting Started

```bash
pip install pivot
```

See the [Quick Start](getting-started/quickstart.md) to build your first pipeline.

## Requirements

- Python 3.13+
- Unix only (Linux/macOS)

## Learn More

- [Tutorials](tutorial/watch.md) - Watch mode, parameters, CI integration
- [Reference](reference/pipelines.md) - Complete documentation by task
- [Migrating from DVC](migrating-from-dvc.md) - Step-by-step migration guide
- [Architecture](architecture/overview.md) - Design decisions and internals
- [Comparison](comparison.md) - How Pivot compares to DVC, Prefect, Dagster

## Roadmap

- **Web UI** - DAG visualization and execution monitoring
- **Additional remotes** - GCS, Azure, SSH
- **Cloud orchestration** - Integration with cloud schedulers
