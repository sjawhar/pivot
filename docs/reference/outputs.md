# Outputs & Caching

Pivot provides several output types for different use cases, all backed by content-addressable caching.

## Overview

| Type | Cached | Git-Tracked | Use Case |
|------|--------|-------------|----------|
| `Out` | Yes | No | Large data files, models |
| `Metric` | No | Yes | Small JSON metrics |
| `Plot` | Yes | No | Visualization files |
| `IncrementalOut` | Yes | No | Append-only files |

## Defining Outputs

Outputs are declared in the function's return type using a TypedDict with annotated fields:

```python
import pathlib
from typing import Annotated, TypedDict

from pivot import loaders, outputs


class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]
    metrics: Annotated[dict, outputs.Metric("metrics.json")]
    plot: Annotated[pathlib.Path, outputs.Plot("loss.png", loaders.PathOnly())]


def train(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    # ... training code ...
    return {
        "model": model_path,
        "metrics": {"accuracy": 0.95},
        "plot": plot_path,
    }
```

Optionally, `pivot.yaml` can override paths without changing Python code:

```yaml
# pivot.yaml (optional - for path overrides)
stages:
  train:
    python: stages.train
    outs:
      model: models/model.pkl
    metrics:
      metrics: metrics/train.json
    plots:
      plot: plots/loss.png
```

## Regular Outputs (`Out`)

Cached outputs for large files:

```python
class ProcessOutputs(TypedDict):
    data: Annotated[pandas.DataFrame, outputs.Out("data.parquet", loaders.CSV())]
```

Options:

- `cache=True` (default) - Store in content-addressable cache

## Metrics

Small files tracked in git (not cached):

```python
class TrainOutputs(TypedDict):
    metrics: Annotated[dict, outputs.Metric("metrics.json")]


def train(...) -> TrainOutputs:
    metrics = {'accuracy': 0.95, 'loss': 0.05}
    return {"metrics": metrics}  # Automatically saved as JSON
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

## Plots

Visualization files that you create manually:

```python
class TrainOutputs(TypedDict):
    plot: Annotated[pathlib.Path, outputs.Plot("loss.png", loaders.PathOnly())]


def train(...) -> TrainOutputs:
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plot_path = pathlib.Path("loss.png")
    plt.savefig(plot_path)
    return {"plot": plot_path}
```

View plots:

```bash
pivot plots show          # Generate HTML gallery
pivot plots show --open   # Open in browser
pivot plots diff          # Show which plots changed
```

## Incremental Outputs

Outputs that preserve state between runs:

```python
class AppendOutputs(TypedDict):
    database: Annotated[dict, outputs.IncrementalOut("cache.json", loaders.JSON())]


def append_records(...) -> AppendOutputs:
    # database.json is restored from cache BEFORE execution
    existing_data = ...  # Loaded from cache
    existing_data["new_key"] = new_value
    return {"database": existing_data}
```

**How it works:**

1. Before execution, previous version is restored from cache
2. Stage modifies the data
3. New version is cached after execution
4. Uses COPY mode (not symlinks) so writes are safe

Use cases:

- Append-only databases
- Cumulative logs
- Incremental data processing

## Single Output Shorthand

For functions with a single output, annotate the return type directly:

```python
def transform(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> Annotated[pandas.DataFrame, outputs.Out("output.csv", loaders.CSV())]:
    return data.dropna()
```

## Choosing Output Types

Use this decision tree:

```
Is your output a visualization (chart, graph, image)?
├── Yes → Plot
└── No → Is it computed numbers for tracking over time?
         ├── Yes → Metric (git-tracked JSON/YAML)
         └── No → Regular output (cached, not git-tracked)
```

| Scenario | Type | Rationale |
|----------|------|-----------|
| Trained model file | `Out` | Large, cached, not tracked |
| Accuracy/loss numbers | `Metric` | Small, tracked for comparison |
| Training loss curve | `Plot` | Visualization for review |
| Intermediate data | `Out` | Large, cached |
| Cumulative log | `IncrementalOut` | Preserves state |

## Caching

### How Caching Works

Pivot uses **content-addressable storage**:

```
.pivot/
├── cache/
│   └── files/
│       ├── ab/
│       │   └── cdef0123...  # File content keyed by hash
│       └── ...
└── stages/
    ├── preprocess.lock      # Per-stage lock file
    └── train.lock
```

When a stage runs:

1. **Outputs are created** by your function
2. **Hash computed** - xxhash64 of file content
3. **Stored in cache** - content-addressable by hash
4. **Lock file updated** - records fingerprint (code + params + deps + output hashes)

On subsequent runs:

1. **Fingerprint compared** - code, params, deps checked
2. **If match** - restore outputs from cache (skip execution)
3. **If changed** - re-run stage

### Skip Conditions

A stage is **skipped** when:

- Code fingerprint matches
- Parameters match
- All input dependencies match
- All outputs exist in cache

A stage **runs** when any of these change.

### Checkout Modes

When restoring from cache:

| Mode | Description | Use Case |
|------|-------------|----------|
| `hardlink` | Hard link to cache (default) | Fast, space-efficient |
| `symlink` | Symbolic link to cache | Visual clarity |
| `copy` | Full copy from cache | When modification needed |

```bash
pivot checkout --checkout-mode copy
```

!!! warning "IncrementalOut Uses Copy Mode"
    `IncrementalOut` always uses copy mode internally. Since the stage modifies the file in-place, using hardlinks or symlinks would corrupt the cache.

### Why xxhash64?

Pivot uses xxhash64 for content hashing:

- **10x faster** than MD5 with equivalent collision resistance for caching
- Non-cryptographic (not for security, just deduplication)
- 64-bit hash provides sufficient uniqueness for cache keys

### File Change Detection

Before hashing file contents, Pivot checks if a file has changed using three filesystem attributes:

| Attribute | What It Detects |
|-----------|-----------------|
| **mtime** | In-place edits (content modified without replacing file) |
| **size** | Fast filter for obvious changes |
| **inode** | Atomic replacement (vim backup mode, some IDEs, `mv`) |

This combination provides defense-in-depth:

- **In-place edits** (e.g., `echo "new" > file`) change mtime and size but keep the same inode
- **Atomic replacement** (e.g., `mv tmp file`) assigns a new inode but may preserve mtime
- **Editors vary** - vim with backup mode does atomic replacement; nano does in-place edits

Using all three attributes catches both patterns.

!!! note "Why Not Just Hash?"
    Hashing every file on every run would be expensive for large outputs. The mtime+size+inode check is O(1) per file (just a `stat()` call), letting us skip rehashing unchanged files.

## Troubleshooting

### Stage Runs But Output Not Cached

**Symptom:** Stage executes successfully but runs again next time.

**Cause:** Output file not declared or declared incorrectly.

**Solution:** Verify outputs are declared in the function's return type and paths match:

```python
class ProcessOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.csv", loaders.PathOnly())]


def process(...) -> ProcessOutputs:
    out_path = pathlib.Path("output.csv")
    # ... create the file ...
    return {"output": out_path}  # Must return the declared output
```

Check that the file was actually created:

```bash
ls -la output.csv
```

### Stage Reruns Unexpectedly

**Symptom:** A stage runs even though you didn't change it.

**Cause:** Pivot detected a change in code, parameters, or dependencies.

**Solution:** Use `pivot status --explain` to see what changed:

```bash
$ pivot status --explain train

Stage: train
  Status: WILL RUN
  Reason: Code dependency changed

  Code changes:
    func:helper_a
      Old: 5995c853
      New: a1b2c3d4
      File: src/utils.py:15
```

Common triggers:

- Modified a helper function the stage calls
- Changed default argument values
- Updated a module the stage imports

## See Also

- [Defining Pipelines](pipelines.md) - Stage definition patterns
- [Dependencies & Loaders](dependencies.md) - Declaring inputs
- [Configuration](configuration.md) - Remote storage setup
