# Quick Start

This guide walks you through creating and running your first Pivot pipeline in under 5 minutes.

## 1. Create a Pipeline

Create a file called `pipeline.py`:

```python
import pickle

import pandas

from pivot import stage

@stage(deps=['data.csv'], outs=['processed.parquet'])
def preprocess():
    """Load and clean the data."""
    df = pandas.read_csv('data.csv')
    df = df.dropna()
    df.to_parquet('processed.parquet')

@stage(deps=['processed.parquet'], outs=['model.pkl'])
def train():
    """Train a simple model."""
    df = pandas.read_parquet('processed.parquet')
    model = {'rows': len(df), 'cols': len(df.columns)}
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
```

!!! note "Module-Level Functions"
    Stage functions must be defined at module level (not inside `if __name__ == '__main__':`) because Pivot uses multiprocessing and needs to serialize functions to worker processes.

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

1. Discover `pipeline.py` automatically
2. Build a dependency graph
3. Execute stages in the correct order
4. Cache outputs for future runs

## 4. Re-run (Cached)

```bash
pivot run
```

The second run completes instantly because nothing changed.

## 5. Modify and Re-run

Edit `pipeline.py` to change the `preprocess` function:

```python
@stage(deps=['data.csv'], outs=['processed.parquet'])
def preprocess():
    df = pandas.read_csv('data.csv')
    df = df.dropna()
    df['doubled'] = df['value'] * 2  # New line!
    df.to_parquet('processed.parquet')
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
- [Defining Stages](../guide/stages.md) - Deep dive into the `@stage` decorator
- [Output Types](../guide/outputs.md) - Learn about `Out`, `Metric`, `Plot`, and `IncrementalOut`
- [CLI Reference](../cli/index.md) - All available commands
