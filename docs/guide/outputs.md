# Output Types

Pivot provides several output types for different use cases.

## Overview

| Type | Cached | Git-Tracked | Use Case |
|------|--------|-------------|----------|
| `Out` | Yes | No | Large data files, models |
| `Metric` | No | Yes | Small JSON metrics |
| `Plot` | Yes | No | Visualization files |
| `IncrementalOut` | Yes | No | Append-only files |

## Out (Default)

Regular cached output for large files:

```python
from pivot import stage, Out

@stage(deps=['data.csv'], outs=[Out('model.pkl')])
def train():
    pass

# Shorthand (strings become Out automatically)
@stage(deps=['data.csv'], outs=['model.pkl'])
def train():
    pass
```

Options:

- `cache=True` (default) - Store in content-addressable cache
- `persist=False` (default) - Keep in cache after workspace cleanup

## Metric

Small files tracked in git (not cached):

```python
from pivot import stage, Metric

@stage(
    deps=['data.csv'],
    outs=[
        'model.pkl',
        Metric('metrics.json'),
    ]
)
def train():
    import json
    metrics = {'accuracy': 0.95, 'loss': 0.05}
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
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

## Plot

Visualization files with optional axis configuration:

```python
from pivot import stage, Plot

@stage(
    deps=['data.csv'],
    outs=[
        Plot('loss_curve.png'),
        Plot('training.csv', x='epoch', y='loss'),
    ]
)
def train():
    pass
```

Options:

- `x` - X-axis column name (for CSV/JSON data)
- `y` - Y-axis column name (for CSV/JSON data)
- `template` - Visualization template

View plots:

```bash
pivot plots show          # Generate HTML gallery
pivot plots show --open   # Open in browser
pivot plots diff          # Show which plots changed
```

## IncrementalOut

Outputs that preserve state between runs:

```python
from pivot import stage, IncrementalOut

@stage(deps=['new_data.csv'], outs=[IncrementalOut('database.db')])
def append_to_database():
    # database.db is restored from cache BEFORE execution
    # Stage can append to it rather than recreating
    import sqlite3
    conn = sqlite3.connect('database.db')
    # ... append new records ...
```

**How it works:**

1. Before execution, previous version is restored from cache
2. Stage modifies the file in place
3. New version is cached after execution
4. Uses COPY mode (not symlinks) so writes are safe

Use cases:

- Append-only databases
- Cumulative logs
- Incremental data processing

## Mixing Output Types

```python
@stage(
    deps=['data.csv'],
    outs=[
        Out('model.pkl'),           # Large model file
        Metric('metrics.json'),     # Small metrics (git-tracked)
        Plot('loss.png'),           # Visualization
        IncrementalOut('log.txt'),  # Append-only log
    ]
)
def train():
    pass
```

## See Also

- [API Reference: outputs](../reference/pivot/outputs.md) - Full API documentation
