# Pivot

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Stages are plain Python functions; you wire them together by calling them inside a `Pipeline` block, and Pivot works out what needs to re-run.

```bash
pivot repro      # Run your pipeline
# edit a helper function...
pivot repro      # Pivot detects the change and re-runs affected stages
```

## Quick Example

```python
# pipeline.py
import pandas as pd

from pivot import Pipeline, stage


def normalize(values: pd.Series) -> pd.Series:
    return values / values.max()


@stage
def preprocess(records: pd.DataFrame) -> pd.DataFrame:
    clean = records.dropna()
    return clean.assign(value=normalize(clean["value"]))


@stage
def summarize(rows: pd.DataFrame) -> dict:
    return {"count": len(rows), "mean": float(rows["value"].mean())}


with Pipeline("example") as pipeline:
    records = pipeline.input("records.csv")  # reads data/raw/records.csv
    summarize(preprocess(records))
```

```bash
pivot repro  # Runs both stages
pivot repro  # Instant - nothing changed
```

Edit `normalize` and both stages re-run, because `preprocess` calls it and `summarize` consumes `preprocess`'s output. Edit `summarize` and only `summarize` re-runs.

## What Makes Pivot Different

### Automatic Code Change Detection

Pivot fingerprints each stage function and every helper it calls, so a change to `normalize` above re-runs `preprocess` without any declaration. No YAML to update, no list of source files to maintain.

### No Paths in Pipeline Code

Stages take data and return data. Pivot loads each input, saves each output under `data/<pipeline>/`, and builds the dependency graph from the handles you pass between stage calls.

### See Why Stages Run

```bash
pivot status --explain
```

### Watch Mode

Edit code, save, see results:

```bash
pivot repro --watch  # Re-runs automatically on file changes
```

## Getting Started

```bash
uv add pivot
```

See the [Quick Start](getting-started/quickstart.md) to build your first pipeline.

## Requirements

- Python 3.13+
- Unix only (Linux/macOS)

## Learn More

**Start here:** Follow the [Concepts](concepts/index.md) guide, a linear path from the first stage to caching.

Then explore task-oriented [Guides](guides/watch-mode.md) for specific workflows:

- [Watch Mode](guides/watch-mode.md): rapid iteration
- [Multi-Pipeline Projects](guides/multi-pipeline.md): large project organization
- [Remote Storage](guides/remote-storage.md): share cache across machines
- [CI Integration](guides/ci-integration.md): pipeline verification in CI

**Reference:**

- [CLI Reference](cli/index.md): all commands and options
- [Architecture](architecture/overview.md): for contributors
- [Comparison with DVC](comparison.md): feature comparison
