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
import pivot


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, pivot.Out("processed.parquet", pivot.loaders.PathOnly())]


def preprocess(
    raw: Annotated[pandas.DataFrame, pivot.Dep("data.csv", pivot.loaders.CSV())],
) -> PreprocessOutputs:
    df = raw.dropna()
    out_path = pathlib.Path("processed.parquet")
    df.to_parquet(out_path)
    return PreprocessOutputs(clean=out_path)


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, pivot.Out("model.pkl", pivot.loaders.PathOnly())]


def train(
    data: Annotated[pathlib.Path, pivot.Dep("processed.parquet", pivot.loaders.PathOnly())],
) -> TrainOutputs:
    df = pandas.read_parquet(data)
    model_path = pathlib.Path("model.pkl")
    # ... train model ...
    return TrainOutputs(model=model_path)


# Register stages - Pivot discovers deps/outs from annotations
pipeline = pivot.Pipeline("my_pipeline")
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
    data: Annotated[pandas.DataFrame, pivot.Dep("data.csv", pivot.loaders.CSV())],
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

**Start here:** Follow the [Concepts](concepts/index.md) guide — a linear learning path from
first principles to advanced caching.

Then explore task-oriented [Guides](guides/watch-mode.md) for specific workflows:
- [Watch Mode](guides/watch-mode.md) — Rapid iteration
- [Multi-Pipeline Projects](guides/multi-pipeline.md) — Large project organization
- [Remote Storage](guides/remote-storage.md) — Share cache across machines
- [CI Integration](guides/ci-integration.md) — Pipeline verification in CI

**Reference:**
- [CLI Reference](cli/index.md) — All commands and options
- [Architecture](architecture/overview.md) — For contributors
- [Comparison with DVC](comparison.md) — Feature comparison
