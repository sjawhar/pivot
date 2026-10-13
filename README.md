# Pivot: High-Performance Python Pipeline Tool

**Change your code. Pivot knows what to run.**

**Python:** 3.13+ | **Platform:** Unix (Linux/macOS) | **Coverage:** 90%+

---

## What is Pivot?

Pivot is a Python pipeline tool with automatic code change detection. Stages are plain Python functions; you wire them together by calling them inside a `Pipeline` block, and Pivot works out what needs to re-run.

- **Automatic code change detection** using Python introspection
- **No paths in pipeline code**: Pivot chooses where every output is stored
- **Per-stage lock files** for fast parallel writes
- **Warm worker pools** with preloaded imports
- **Content-addressable caching** with S3 remote storage

---

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

```bash
pivot repro  # runs clean, then summarize
pivot repro  # nothing changed: both stages are cached
```

Pivot writes outputs to `data/demo/clean.jsonl` and `data/demo/summarize.yaml`. Edit the body of `summarize` and only `summarize` re-runs. A change to `min_value` or `clean` re-runs `clean`; downstream stages run only when `clean` produces different output bytes.

The [Quick Start guide](https://sjawhar.github.io/pivot/getting-started/quickstart/) walks through this pipeline step by step.

---

## Key Features

### Automatic Code Change Detection

Pivot fingerprints each stage function and every helper it calls. Edit a helper and the stages that use it re-run; edit a comment and nothing does.

```python
def normalize(df):
    return df / df.max()  # change this...


@stage
def process(raw: pd.DataFrame) -> pd.DataFrame:
    return normalize(raw)  # ...and process re-runs
```

### Explain Mode

See why a stage would run:

```bash
pivot status --explain
```

### Typed Stage Parameters

Each stage takes its configuration as a `StageParams` subclass (a Pydantic model) defined next to it. Parameter values are recorded in the stage's lock file, so changing one re-runs that stage. A `params.yaml` at the project root can override values per stage without editing code.

### S3 Remote Cache

Share outputs across machines and CI:

```bash
pivot config set remotes.origin s3://my-bucket/pivot-cache
pivot push                 # Upload to remote
pivot pull                 # Download and restore outputs
```

### Data Diff

Compare data file changes interactively:

```bash
pivot diff data/demo/clean.jsonl
```

---

## Common Commands

Stage names are qualified by their pipeline (`demo/clean`); `pivot list` prints them.

```bash
pivot repro                    # Run entire pipeline (DAG-aware)
pivot repro demo/summarize     # Run demo/summarize and everything it depends on
pivot repro --watch            # Watch mode - re-run on file changes
pivot repro --show-output      # Stream stage stdout/stderr to terminal
pivot run demo/clean           # Run ONLY demo/clean (no dependency resolution)
pivot repro -n                 # Dry run - see what would execute

pivot status                   # Show pipeline status
pivot status --explain         # Understand why stages are stale
pivot list --deps              # List stages with dependencies
pivot dag --mermaid            # Visualize DAG as Mermaid diagram

pivot metrics show             # Display metric values
pivot params show              # Display effective parameters
pivot params diff              # Compare params against HEAD

pivot push                     # Push cached outputs to remote
pivot pull                     # Pull and restore from remote
pivot verify --allow-missing   # CI gate: verify reproducibility
```

See the full [CLI Reference](https://sjawhar.github.io/pivot/cli/) for all commands and options.

---

## Documentation

Full documentation at [sjawhar.github.io/pivot/](https://sjawhar.github.io/pivot/).

- [Quick Start](https://sjawhar.github.io/pivot/getting-started/quickstart/): build your first pipeline
- [Concepts](https://sjawhar.github.io/pivot/concepts/): stages, artifacts, parameters, fingerprinting, caching
- [CLI Reference](https://sjawhar.github.io/pivot/cli/): all available commands
- [Architecture](https://sjawhar.github.io/pivot/architecture/overview/): design decisions and internals

Agents writing pipelines should load [`skills/writing-pivot-stages/SKILL.md`](skills/writing-pivot-stages/SKILL.md).

---

## Development

```bash
uv sync --active                                                     # Install deps
uv run pytest packages/pivot/tests packages/pivot-tui/tests -n auto  # Test
uv run ruff format . && uv run ruff check . && uv run basedpyright   # Quality
```

---

## License

TBD
