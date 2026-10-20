# Pivot: High-Performance Python Pipeline Tool

**Change your code. Pivot knows what to run.**

Pivot is a Python pipeline tool with automatic code change detection. Stages are plain
Python functions; you wire them together by calling them inside a `Pipeline` block, and
Pivot works out what needs to re-run.

- **Automatic code change detection** using Python introspection
- **No paths in pipeline code**: Pivot chooses where every output is stored
- **Per-stage lock files** for fast parallel writes
- **Warm worker pools** with preloaded imports
- **Content-addressable caching** with S3 or Google Cloud Storage remotes

**Python:** 3.13+ | **Platform:** Unix (Linux/macOS)

## Quick Start

```bash
pip install pivot
mkdir my_project && cd my_project
git init && pivot init
```

Put a raw input under `data/raw/`:

```bash
mkdir -p data/raw
printf 'name,value\nalice,100\nbob,200\ncarol,\n' > data/raw/records.csv
```

Define stages and the pipeline in `pipeline.py`:

```python
# pipeline.py
import pandas as pd

from pivot import Pipeline, StageParams, stage


class CleanParams(StageParams):
    min_value: int = 0


@stage
def clean(records: pd.DataFrame, params: CleanParams) -> pd.DataFrame:
    kept = records.dropna()
    return kept[kept["value"] >= params.min_value]


@stage
def summarize(rows: pd.DataFrame) -> dict:
    return {"count": len(rows), "total": int(rows["value"].sum())}


with Pipeline("demo") as pipeline:
    records = pipeline.input("records.csv")
    cleaned = clean(records, params=CleanParams(min_value=50))
    summarize(cleaned)
```

Then run it; Pivot resolves the DAG from the handles passed between stages:

```bash
pivot repro           # runs clean, then summarize
pivot repro           # nothing changed: both stages are cached
pivot repro --watch   # re-run automatically on file changes
```

Pivot writes outputs to `data/demo/clean.jsonl` and `data/demo/summarize.yaml`.

## Learn More

- [Documentation](https://sjawhar.github.io/pivot/)
- [Source code](https://github.com/sjawhar/pivot)
- Interactive TUI: `pip install pivot[tui]`, then `pivot repro --tui`
