# Pivot: High-Performance Python Pipeline Tool

**Change your code. Pivot knows what to run.**

**Python:** 3.13+ | **Platform:** Unix (Linux/macOS) | **Coverage:** 90%+

---

## What is Pivot?

Pivot is a Python pipeline tool with automatic code change detection. Define stages with typed Python functions and annotations, and Pivot figures out what needs to re-run — no manual dependency declarations, no stale caches.

- **Automatic code change detection** using Python introspection
- **Per-stage lock files** for fast parallel writes (32x faster than monolithic locks)
- **Warm worker pools** with preloaded imports
- **Content-addressable caching** with S3 remote storage
- **DVC compatibility** via YAML export

---

## Quick Start

```bash
pip install pivot
pivot init
```

### Python-First Pipeline Definition

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


# Create pipeline and register stages
pipeline = pivot.Pipeline("my_pipeline")
pipeline.register(preprocess)
pipeline.register(train)
```

```bash
pivot repro  # Runs both stages
pivot repro  # Instant - nothing changed
```

Modify `preprocess`, and Pivot automatically re-runs both stages. Modify `train`, and only `train` re-runs.

---

## Installation

```bash
pip install pivot
```

**Requirements:** Python 3.13+, Unix only (Linux/macOS)

---

## Key Features

### Automatic Code Change Detection

Pivot detects when your Python functions change — no manual `deps:` declarations:

```python
def helper(x):
    return x * 2  # Change this...

def process():
    data = load("data.csv")
    return helper(data)  # ...and Pivot knows to re-run!
```

Uses `inspect.getclosurevars()` + AST extraction with recursive fingerprinting for transitive dependencies.

### Explain Mode

See *why* a stage would run:

```bash
pivot repro --explain

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed
  Changes:
    func:helper_a  Old: 5995c853  New: a1b2c3d4
```

### Stage Parameters

Type-safe parameters via Pydantic:

```python
import pivot

class TrainParams(pivot.StageParams):
    learning_rate: float = 0.01
    epochs: int = 100

def train(params: TrainParams, data: Annotated[...]) -> TrainOutputs:
    ...
```

Parameter changes are tracked in lock files and trigger re-execution.

### S3 Remote Cache

Share outputs across machines and CI:

```bash
pivot config set remotes.origin s3://my-bucket/pivot-cache
pivot push                 # Upload to remote
pivot pull train_model     # Download specific stage outputs
```

### Import Artifacts

Import artifacts from remote Pivot repositories:

```bash
pivot import https://github.com/org/ml-models model.pkl --rev v1.0
pivot update               # Check and apply updates
```

### Incremental Outputs

Outputs that preserve state between runs for append-only workloads. Before execution, previous versions are restored from cache; the stage modifies in place and the new version is cached.

### Data Diff

Compare data file changes interactively:

```bash
pivot diff output.csv                    # Interactive TUI
pivot diff output.csv --key id --json    # Key-based matching, JSON output
```

---

## Common Commands

```bash
pivot repro                    # Run entire pipeline (DAG-aware)
pivot repro train evaluate     # Run specific stages + dependencies
pivot repro --watch            # Watch mode - re-run on file changes
pivot repro --show-output      # Stream stage stdout/stderr to terminal
pivot run my_stage             # Run ONLY my_stage (no dep resolution)
pivot repro -n                 # Dry run - see what would execute

pivot status                   # Show pipeline status
pivot status --explain train   # Understand why a stage is stale
pivot list --deps              # List stages with dependencies
pivot dag --mermaid            # Visualize DAG as Mermaid diagram

pivot diff output.csv          # Compare data files vs git HEAD
pivot metrics show             # Display metric values
pivot params diff              # Compare params against HEAD

pivot push                     # Push cached outputs to remote
pivot pull                     # Pull and restore from remote
pivot fetch                    # Fetch to local cache only
pivot verify --allow-missing   # CI gate: verify reproducibility

pivot import REPO PATH         # Import artifact from remote repo
pivot update --dry-run         # Check for import updates
pivot fingerprint reset        # Clear cached fingerprints
```

See the full [CLI Reference](https://sjawhar.github.io/pivot/cli/) for all commands and options.

---

## Documentation

Full documentation at [sjawhar.github.io/pivot/](https://sjawhar.github.io/pivot/).

- [Quick Start](https://sjawhar.github.io/pivot/getting-started/quickstart/) — Build your first pipeline in 5 minutes
- [Concepts](https://sjawhar.github.io/pivot/concepts/) — Linear learning path from first principles to advanced caching
- [CLI Reference](https://sjawhar.github.io/pivot/cli/) — All available commands
- [Architecture](https://sjawhar.github.io/pivot/architecture/overview/) — Design decisions and internals

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

