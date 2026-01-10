# Pivot: High-Performance Python Pipeline Tool

Pivot is a Python-native pipeline tool designed to eliminate DVC's performance bottlenecks while maintaining compatibility.

## Key Features

- **32x faster lock file operations** through per-stage lock files
- **Automatic code change detection** using Python introspection (no manual declarations!)
- **Warm worker pools** with preloaded imports (numpy/pandas loaded once)
- **DVC compatibility** via YAML export for code reviews

## Performance

| Component         | DVC          | Pivot       | Improvement     |
| ----------------- | ------------ | ----------- | --------------- |
| Lock file writes  | 289s (23.8%) | ~9s (0.7%)  | **32x faster**  |
| Total overhead    | 301s (24.8%) | ~20s (1.6%) | **15x faster**  |
| **Total runtime** | **1214s**    | **~950s**   | **1.3x faster** |

*Benchmarked on monitoring-horizons pipeline with 176 stages*

## Quick Example

```python
from pivot import stage

@stage(deps=['data.csv'], outs=['processed.parquet'])
def preprocess():
    import pandas as pd
    df = pd.read_csv('data.csv')
    df = df.dropna()
    df.to_parquet('processed.parquet')

@stage(deps=['processed.parquet'], outs=['model.pkl'])
def train():
    import pandas as pd
    import pickle
    df = pd.read_parquet('processed.parquet')
    model = {'data': df.shape}
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
```

```bash
pivot run  # Execute the pipeline
```

## Why Pivot?

### Automatic Code Change Detection

No more manual dependency declarations! Pivot automatically detects when your Python functions change:

```python
def helper(x):
    return x * 2  # Change this...

@stage(deps=['data.csv'])
def process():
    data = load('data.csv')
    return helper(data)  # ...and Pivot knows to re-run!
```

### Explain Mode

See **why** a stage would run:

```bash
$ pivot explain train

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
```

### Per-Stage Lock Files

DVC's bottleneck: Every stage writes the entire `dvc.lock` file (O(n²) behavior)

Pivot's solution: Each stage gets its own lock file for parallel writes without contention.

## Getting Started

Ready to try Pivot? Check out the [Installation Guide](getting-started/installation.md) and [Quick Start Tutorial](getting-started/quickstart.md).

## Requirements

- Python 3.13+
- Linux/macOS (Windows support via spawn context)

## Roadmap

Planned features:

- **Web UI** - DAG visualization and execution monitoring
- **Additional remote backends** - GCS, Azure, SSH
- **Cloud orchestration** - Integration with cloud schedulers
