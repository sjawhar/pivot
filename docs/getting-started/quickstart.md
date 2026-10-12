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
- `.pivot/` — Directory for cache and state
- `.pivotignore` — Patterns for files to exclude from watching

Add to your `.gitignore`:

```
.pivot/cache/
.pivot/state.lmdb/
```

Commit the lock files — they track what ran and what outputs were produced:

```
# These go in git:
.pivot/stages/*.lock
```

## 2. Create a Pipeline

Create `pipeline.py`:

```python
# pipeline.py
import json
import pathlib
from typing import Annotated, TypedDict

import pivot

pipeline = pivot.Pipeline("my_pipeline")


class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, pivot.Out("clean.json", pivot.loaders.PathOnly())]


def preprocess(
    raw: Annotated[dict, pivot.Dep("data.json", pivot.loaders.JSON())],
) -> PreprocessOutputs:
    """Remove entries with missing values."""
    clean = [row for row in raw["records"] if all(row.values())]
    out_path = pathlib.Path("clean.json")
    out_path.write_text(json.dumps({"records": clean}, indent=2))
    return PreprocessOutputs(clean=out_path)


class SummarizeOutputs(TypedDict):
    summary: Annotated[pathlib.Path, pivot.Out("summary.json", pivot.loaders.PathOnly())]


def summarize(
    data: Annotated[dict, pivot.Dep("clean.json", pivot.loaders.JSON())],
) -> SummarizeOutputs:
    """Compute summary statistics."""
    records = data["records"]
    values = [r["value"] for r in records]
    summary = {
        "count": len(values),
        "mean": sum(values) / len(values) if values else 0,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
    }
    out_path = pathlib.Path("summary.json")
    out_path.write_text(json.dumps(summary, indent=2))
    return SummarizeOutputs(summary=out_path)


# Register stages — Pivot discovers deps/outs from annotations
pipeline.register(preprocess)
pipeline.register(summarize)
```

## 3. Create Sample Data

```bash
cat > data.json << 'EOF'
{
  "records": [
    {"name": "Alice", "value": 100},
    {"name": "Bob", "value": 200},
    {"name": "Charlie", "value": null}
  ]
}
EOF
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
    raw: Annotated[dict, pivot.Dep("data.json", pivot.loaders.JSON())],
) -> PreprocessOutputs:
    """Remove entries with missing values and add a 'processed' flag."""
    clean = [
        {**row, "processed": True}
        for row in raw["records"]
        if all(row.values())
    ]
    out_path = pathlib.Path("clean.json")
    out_path.write_text(json.dumps({"records": clean}, indent=2))
    return PreprocessOutputs(clean=out_path)
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

In the examples above, `Dep()` and `Out()` take a loader like `pivot.loaders.JSON()` or `pivot.loaders.PathOnly()`. Loaders implement the `Reader` and `Writer` protocols:

- **`Reader`** — knows how to load a file into a Python object (used by `Dep`)
- **`Writer`** — knows how to save a Python object to a file (used by `Out`)

All built-in loaders implement both, so you can use them interchangeably. `PathOnly()` skips automatic I/O — you handle file reads/writes yourself and just return the path.

## Next Steps

- [Watch Mode & Rapid Iteration](../guides/watch-mode.md) — Develop faster with auto-rerun
- [Pipelines](../concepts/pipelines.md) — Deep dive into stage definition
- [Outputs](../concepts/outputs.md) — Learn about outputs, metrics, and plots

> **Project Structure**: For larger projects, consider using [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) as a starting template. Its `data/raw/`, `data/processed/`, and `src/` layout works well with Pivot.
