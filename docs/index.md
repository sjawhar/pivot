# Pivot

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Define stages in YAML with typed Python functions, and Pivot figures out what needs to re-run—no manual dependency declarations, no stale caches.

```bash
pivot run        # Run your pipeline
# edit a helper function...
pivot run        # Pivot detects the change and re-runs affected stages
```

## Quick Example

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

```python
# stages.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("processed.parquet", loaders.PathOnly())]


def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return {"clean": out_path}


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


def train(
    data: Annotated[pathlib.Path, outputs.Dep("processed.parquet", loaders.PathOnly())],
) -> TrainOutputs:
    df = pandas.read_parquet(data)
    model_path = pathlib.Path("model.pkl")
    # ... train model ...
    return {"model": model_path}
```

```bash
pivot run  # Runs both stages
pivot run  # Instant - nothing changed
```

Modify `preprocess`, and Pivot automatically re-runs both stages. Modify `train`, and only `train` re-runs.

> **How YAML and Python Work Together**
>
> Your Python function's annotations define *what* the stage needs (types and default paths).
> The YAML file lets you override those paths without editing Python code.
>
> - If YAML specifies a path, it overrides the annotation's default
> - If YAML doesn't specify a path, the annotation's default is used
> - YAML `deps:`/`outs:` keys must match the Python parameter/output names

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
$ pivot explain train

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
pivot run --watch  # Re-runs automatically on file changes
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
