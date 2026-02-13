# Outputs

Outputs declare the files a stage writes. They form the other half of the
[DAG](artifacts-and-dag.md) — every output path can become a
[dependency](dependencies.md) for downstream stages.

## The return-type pattern

Outputs are declared in the stage function's **return type**, not its parameters.
For multiple outputs, use a `TypedDict` where each field is annotated with an
output spec:

```python
from typing import Annotated, TypedDict
from pandas import DataFrame
import pivot

class TrainOutputs(TypedDict):
    predictions: Annotated[DataFrame, pivot.Out("predictions.csv", pivot.loaders.CSV())]
    metrics: Annotated[dict, pivot.Metric("metrics.json")]

def train(
    data: Annotated[DataFrame, pivot.Dep("features.csv", pivot.loaders.CSV())],
) -> TrainOutputs:
    model = fit(data)
    return {
        "predictions": model.predict(data),
        "metrics": {"accuracy": 0.95, "f1": 0.91},
    }
```

The stage returns a plain dict matching the TypedDict keys. Pivot serializes
each value to its declared path using the specified [loader](loaders.md).

### Single-output shorthand

When a stage produces exactly one file, skip the TypedDict and annotate the
return type directly:

```python
def clean(
    raw: Annotated[DataFrame, pivot.Dep("raw.csv", pivot.loaders.CSV())],
) -> Annotated[DataFrame, pivot.Out("clean.csv", pivot.loaders.CSV())]:
    return raw.dropna()
```

The stage returns the value directly (not wrapped in a dict). Pivot saves it
to the declared path.

## Output types at a glance

| Type | Syntax | Cached | Use case |
|------|--------|--------|----------|
| `Out[W]` | `Out(path, writer, cache=True)` | Yes | Standard file output |
| `Metric` | `Metric(path)` | No | JSON metrics (git-tracked) |
| `Plot[W]` | `Plot(path, writer)` | Yes | Visualization files |
| `IncrementalOut[W,R]` | `IncrementalOut(path, loader)` | Yes | State carried across runs |
| `DirectoryOut[T]` | `DirectoryOut(path + "/", writer)` | Yes | Dynamic file sets |

All output types are frozen dataclasses. They are immutable, picklable, and
their code is fingerprinted for change detection.

## `Out` — standard file output

The workhorse output type. Writes a single file (or multiple files) using a
[Writer](loaders.md):

```python
class Results(TypedDict):
    report: Annotated[dict, pivot.Out("report.json", pivot.loaders.JSON())]
    data: Annotated[DataFrame, pivot.Out("processed.csv", pivot.loaders.CSV())]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| list[str] \| tuple[str, ...]` | required | Output file path(s) |
| `loader` | `Writer[W]` | required | Serializer for the data (named `loader` for consistency with `Dep`; accepts any `Writer`) |
| `cache` | `bool` | `True` | Whether to cache in `.pivot/cache` |

### Multi-file outputs

A single output key can write multiple files:

```python
class ShardOutputs(TypedDict):
    shards: Annotated[list[dict], pivot.Out(["shard_a.json", "shard_b.json"], pivot.loaders.JSON())]

def split(...) -> ShardOutputs:
    return {"shards": [data_a, data_b]}  # list matches path list
```

The return value must be a list/tuple with the same length as the path list.
Each element is written to the corresponding path.

### Disabling caching

Set `cache=False` to skip caching for outputs that shouldn't be stored
(e.g., temporary files or files you want to always regenerate):

```python
debug_log: Annotated[str, pivot.Out("debug.log", pivot.loaders.Text(), cache=False)]
```

## `Metric` — git-tracked JSON metrics

`Metric` is a specialized `Out` for small JSON values that should be
version-controlled rather than cached:

```python
class EvalOutputs(TypedDict):
    scores: Annotated[dict, pivot.Metric("metrics/eval.json")]
```

**Defaults that differ from `Out`:**

| Parameter | `Out` default | `Metric` default |
|-----------|---------------|------------------|
| `loader` | required | `JSON()` (automatic) |
| `cache` | `True` | `False` |

Metrics are not cached by default because they're small and belong in git for
tracking experiment history. The loader defaults to `JSON()` — you typically
don't need to specify it.

`Metric` accepts any JSON-serializable value: dicts, lists, strings, numbers,
booleans, or `None`.

## `Plot` — visualization output

`Plot` extends `Out` with optional metadata for visualization tools:

```python
import pathlib
from matplotlib.figure import Figure
import pivot

class TrainOutputs(TypedDict):
    # Automatic: Pivot saves the figure and closes it
    loss_curve: Annotated[Figure, pivot.Plot("plots/loss.png", pivot.loaders.MatplotlibFigure())]

    # Manual: stage creates the file, Pivot just tracks it
    roc_plot: Annotated[pathlib.Path, pivot.Plot("plots/roc.png", pivot.loaders.PathOnly())]
```

**Extra parameters** (beyond `Out`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `str \| None` | `None` | X-axis column name (for structured plot data) |
| `y` | `str \| None` | `None` | Y-axis column name (for structured plot data) |
| `template` | `str \| None` | `None` | Plot template identifier |

`MatplotlibFigure()` is a write-only loader that calls `fig.savefig()` and then
`plt.close(fig)` to prevent memory leaks. Format is inferred from the file
extension (`.png`, `.pdf`, `.svg`).

## `IncrementalOut` — state across runs {#incrementalout}

`IncrementalOut` is for outputs that accumulate state across runs. Before the
stage executes, Pivot restores the previous output from cache so the stage can
read, modify, and write it back.

```python
from typing import Annotated, TypedDict
import pivot

class CacheOutputs(TypedDict):
    cache: Annotated[dict, pivot.IncrementalOut("cache.json", pivot.loaders.JSON())]

def update_cache(
    cache: Annotated[dict, pivot.IncrementalOut("cache.json", pivot.loaders.JSON())],
    new_data: Annotated[DataFrame, pivot.Dep("new_data.csv", pivot.loaders.CSV())],
) -> CacheOutputs:
    # `cache` contains the previous run's output (or {} on first run)
    cache["latest"] = process(new_data)
    return {"cache": cache}
```

### How it works

1. Before execution, Pivot checks for a cached previous output
2. If found, restores it and injects as the parameter value
3. If not found (first run), uses the loader's `empty()` method (e.g., `{}` for
   `JSON`, empty `DataFrame` for `CSV`)
4. Stage reads, modifies, and returns the updated value
5. Pivot writes and re-caches the result

### Rules

- The `IncrementalOut` must appear on **both** a parameter and a return field
- Path and loader must match between input and output
- For TypedDict returns, the parameter name must match the return field name
- For single-output stages, only one `IncrementalOut` parameter is allowed
- The input does **not** create a DAG edge (it's self-referential)
- Requires a `Loader` (not just `Reader` or `Writer`) since it reads and writes

## `DirectoryOut` — dynamic file sets

When the number or names of output files aren't known until runtime, use
`DirectoryOut`:

```python
from typing import Annotated, TypedDict
import pivot

class TaskMetrics(TypedDict):
    accuracy: float
    loss: float

class ProcessOutputs(TypedDict):
    results: Annotated[dict[str, TaskMetrics], pivot.DirectoryOut("metrics/tasks/", pivot.loaders.YAML())]

def process_tasks(...) -> ProcessOutputs:
    return {
        "results": {
            "task_a.yaml": TaskMetrics(accuracy=0.95, loss=0.12),
            "task_b.yaml": TaskMetrics(accuracy=0.87, loss=0.31),
        }
    }
```

**How it works:**

- Path **must** end with `/` (enforced at construction time)
- Return value is `dict[str, T]` where keys are relative paths within the
  directory and values are the data to serialize
- Each key must include a file extension
- Keys are validated: no absolute paths, no `..` traversal, no empty names
- Duplicate keys after normalization are rejected
- Case collisions are detected (for cross-platform safety)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` (ending with `/`) | required | Output directory |
| `loader` | `Writer[T]` | required | Serializer for each file |
| `cache` | `bool` | `True` | Whether to cache files |

Each file in the directory is cached individually by content hash. Downstream
stages can depend on the directory (`Dep("metrics/tasks/")`) or on specific
files within it (`Dep("metrics/tasks/task_a.yaml")`).

## Overriding output paths

Like [dependencies](dependencies.md), output paths can be overridden at
registration time:

```python
# Simple path override
pipeline.register(train, out_path_overrides={
    "predictions": "v2/predictions.csv",
})

# Override with options
pipeline.register(train, out_path_overrides={
    "predictions": {"path": "v2/predictions.csv", "cache": False},
})
```

For single-output stages, the override key can be any string (there's only one
output to override):

```python
pipeline.register(clean, out_path_overrides={
    "output": "cleaned_v2.csv",
})
```

`IncrementalOut` paths cannot be overridden (input and output paths must match).

## Decision tree: which output type?

```
Is this a JSON metric you want in git?
  └─ Yes → Metric("path.json")

Is this a visualization?
  └─ Yes → Plot("path.png", MatplotlibFigure())

Does this output carry state between runs?
  └─ Yes → IncrementalOut("path", loader)

Are the output files determined at runtime?
  └─ Yes → DirectoryOut("dir/", writer)

Everything else:
  └─ Out("path", writer)
```

## How Pivot writes outputs

After a stage returns, Pivot:

1. Validates all TypedDict keys are present in the return value
2. Resolves each output path relative to the project root
3. Creates parent directories as needed
4. Writes each file sequentially using the output's loader
5. Hashes the written files for cache storage
6. Updates the stage's lock file with output hashes

Validation (step 1) happens before any writes begin, so missing keys or type
mismatches are caught early. However, there is no rollback — if the third of
five writes fails, the first two are already on disk.

## Multi-file paths and expansion

Internally, Pivot expands multi-file outputs (`Out(["a.csv", "b.csv"], ...)`)
into individual single-path output specs for DAG construction and caching. Each
file is tracked and cached independently. This means fine-grained cache hits —
if only one file in a multi-file output changes, only that file is re-cached.

## Summary

| Pattern | When to use |
|---------|------------|
| `TypedDict` with `Out` fields | Multiple outputs from one stage |
| `Annotated[T, Out(...)]` return | Single output |
| `Metric("path.json")` | Small JSON values for git tracking |
| `Plot("path.png", writer)` | Visualizations |
| `IncrementalOut("path", loader)` | State that accumulates across runs |
| `DirectoryOut("dir/", writer)` | Runtime-determined file sets |
| `cache=False` | Outputs that shouldn't be cached |

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Dependencies](dependencies.md) | [Loaders](loaders.md)