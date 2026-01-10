# Pivot

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Define stages with decorators, and Pivot figures out what needs to re-run—no manual dependency declarations, no stale caches.

## Quick Example

```python
from pivot import stage

@stage(deps=['data.csv'], outs=['processed.parquet'])
def preprocess():
    import pandas
    df = pandas.read_csv('data.csv')
    df = df.dropna()
    df.to_parquet('processed.parquet')

@stage(deps=['processed.parquet'], outs=['model.pkl'])
def train():
    import pandas
    import pickle
    df = pandas.read_parquet('processed.parquet')
    model = {'rows': len(df), 'columns': len(df.columns)}
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
```

```bash
pivot run  # Runs both stages
pivot run  # Instant - nothing changed
```

Modify `preprocess`, and Pivot automatically re-runs both stages. Modify `train`, and only `train` re-runs.

## What Makes Pivot Different

### Automatic Code Change Detection

Change a helper function, and Pivot knows to re-run stages that call it:

```python
def normalize(x):
    return x / x.max()  # Change this...

@stage(deps=['data.csv'], outs=['output.csv'])
def process():
    data = load('data.csv')
    return normalize(data)  # ...and Pivot re-runs process
```

No YAML to update. No manual declarations. Pivot parses your Python and tracks what each stage actually calls.

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

See the [Quick Start](getting-started/quickstart.md) to build your first pipeline in 5 minutes.

## Requirements

- Python 3.13+
- Unix only (Linux/macOS)

## Learn More

- [Core Concepts](getting-started/concepts.md) - Stages, dependencies, caching
- [Comparison](comparison.md) - How Pivot compares to DVC, Prefect, Dagster
- [Architecture](architecture/overview.md) - Design decisions and internals

## Roadmap

- **Web UI** - DAG visualization and execution monitoring
- **Additional remotes** - GCS, Azure, SSH
- **Cloud orchestration** - Integration with cloud schedulers
