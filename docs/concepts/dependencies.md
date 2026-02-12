# Dependencies

Dependencies declare the files a stage reads. Pivot uses them to build the
[DAG](artifacts-and-dag.md), load data before execution, and detect when a stage
needs to re-run.

## The `Dep` pattern

Annotate a function parameter with `Annotated[T, Dep(path, loader)]` to declare
a file dependency:

```python
from typing import Annotated
from pandas import DataFrame
import pivot

def clean(
    raw: Annotated[DataFrame, pivot.Dep("raw.csv", pivot.loaders.CSV())],
) -> ...:
    # `raw` is a DataFrame loaded from raw.csv
    return raw.dropna()
```

At runtime, Pivot reads `raw.csv` using the `CSV()` [loader](loaders.md) and
injects the resulting `DataFrame` as the `raw` argument. The type annotation
(`DataFrame`) is for your editor and type checker — the loader controls what
actually gets loaded.

### Testing is natural

Because dependencies are just function parameters, testing requires no mocking:

```python
def test_clean():
    test_df = DataFrame({"a": [1, None, 3]})
    result = clean(test_df)  # pass data directly, no Pivot machinery
    assert len(result) == 2
```

## Dependency types at a glance

| Type | Syntax | DAG edge? | Use case |
|------|--------|-----------|----------|
| `Dep[R]` | `Dep(path, loader)` | Yes | Standard file dependency |
| `PlaceholderDep[R]` | `PlaceholderDep(loader)` | Yes | Path supplied at registration |
| `IncrementalOut` as input | `Annotated[T, IncrementalOut(...)]` on a parameter | No | Self-referential incremental state |

## Single-file dependencies

The most common pattern — one file, one parameter:

```python
def train(
    data: Annotated[DataFrame, pivot.Dep("features.csv", pivot.loaders.CSV())],
    config: Annotated[dict, pivot.Dep("config.json", pivot.loaders.JSON())],
) -> ...:
    ...
```

Each `Dep` creates a DAG edge from the file's producer (or marks it as an
external input if no stage produces it). The loader's type parameter `R`
determines what `load()` returns — see [Loaders](loaders.md) for the full list.

## Multi-file dependencies

For multiple files of the same type, use a list (variable-length) or tuple
(fixed-length) path:

```python
# Variable-length: any number of shards
def merge(
    shards: Annotated[list[DataFrame], pivot.Dep(["shard_0.csv", "shard_1.csv"], pivot.loaders.CSV())],
) -> ...:
    combined = pd.concat(shards)
    ...

# Fixed-length: exactly two files (tuple preserves length in the type)
def compare(
    pair: Annotated[tuple[DataFrame, DataFrame], pivot.Dep(("baseline.csv", "experiment.csv"), pivot.loaders.CSV())],
) -> ...:
    baseline, experiment = pair
    ...
```

Each path in the list/tuple creates its own DAG edge. At runtime, Pivot loads
every file with the same loader and injects the results as a list or tuple
matching the path type.

## `PlaceholderDep` — path at registration time

When a stage is generic and the dependency path isn't known until the pipeline
is assembled, use `PlaceholderDep`:

```python
import pivot

def compare(
    baseline: Annotated[DataFrame, pivot.PlaceholderDep(pivot.loaders.CSV())],
    experiment: Annotated[DataFrame, pivot.PlaceholderDep(pivot.loaders.CSV())],
) -> ...:
    ...

pipeline = pivot.Pipeline("analysis")
pipeline.register(
    compare,
    dep_path_overrides={
        "baseline": "model_a/results.csv",
        "experiment": "model_b/results.csv",
    },
)
```

`PlaceholderDep` has no default path — registration **fails** if
`dep_path_overrides` doesn't include every placeholder. This is intentional:
it prevents accidentally running a stage with an undefined input.

Pivot provides helpful error messages when overrides are missing, including
typo suggestions based on edit distance.

### When to use `PlaceholderDep` vs `Dep`

| Scenario | Use |
|----------|-----|
| Path is always the same | `pivot.Dep("fixed/path.csv", pivot.loaders.CSV())` |
| Path varies per pipeline or registration | `pivot.PlaceholderDep(pivot.loaders.CSV())` + `dep_path_overrides` |
| Path is known but you want to override it occasionally | `pivot.Dep("default.csv", pivot.loaders.CSV())` + optional `dep_path_overrides` |

`Dep` paths can also be overridden at registration time — `dep_path_overrides`
works for both `Dep` and `PlaceholderDep`. The difference is that `Dep` has a
sensible default while `PlaceholderDep` requires an explicit override.

## Directory dependencies

A `Dep` path can point to a directory rather than a file. Pivot resolves
directory deps using prefix matching against all registered output paths:

```python
def summarize(
    reports: Annotated[pathlib.Path, pivot.Dep("reports/", pivot.loaders.PathOnly())],
) -> ...:
    # `reports` is a Path object pointing to the reports/ directory
    for csv_file in reports.glob("*.csv"):
        ...
```

This creates DAG edges from every stage that writes into `reports/` (via
`Out` or `DirectoryOut`) to the `summarize` stage. The reverse also works —
a `Dep("reports/data.csv")` creates an edge from any stage whose
`DirectoryOut("reports/")` contains that file.

Use `PathOnly()` as the loader for directory deps since there's no single file
to deserialize — the stage receives a `pathlib.Path` and handles reading
manually.

## Path resolution

Paths in `Dep` annotations are resolved relative to the **pipeline root** (the
directory containing the `pipeline.py` file). For a pipeline at
`pipelines/training/pipeline.py`:

```python
# This resolves to: <project_root>/pipelines/training/data/input.csv
pivot.Dep("data/input.csv", pivot.loaders.CSV())
```

Paths are resolved relative to the pipeline root directory. Parent directory
references (`..`) are resolved during registration — `pivot.Dep("../shared/data.csv")`
becomes an absolute path before validation. The resolved path must stay within
the project root.

Dependencies may reference absolute paths outside the project (e.g.,
`/data/external/dataset.csv`), though this reduces portability. Outputs must
always resolve within the project root.

## Tracking files outside the pipeline

Files that exist outside any pipeline's output set (raw data, external configs)
are **external inputs**. Pivot treats them as leaf nodes in the DAG — they have
no producer stage, so they must exist on disk before execution.

Pivot still hashes external inputs for change detection. If `raw.csv` changes,
every stage downstream of it re-runs.

To explicitly track external files for visibility in `pivot status`:

```bash
pivot track raw_data/        # Track all files in a directory
pivot track config.yaml      # Track a single file
```

Tracked files appear in the DAG visualization and are included in dependency
validation — Pivot won't warn about "dependency not found" for tracked files.

## How dependencies create DAG edges

When you register a stage, Pivot:

1. Extracts all `Dep` and `PlaceholderDep` annotations from the function signature
2. Resolves paths relative to the pipeline root
3. For each dependency path, checks if any registered stage produces it (exact
   match or directory prefix match)
4. If a producer exists, creates a directed edge from producer to consumer
5. If no producer exists, the path is an external input (validated at DAG build
   time unless it exists on disk or is tracked)

This happens transparently — you never wire stages together manually.

```python
# These two registrations automatically create the edge clean -> train
# because pivot.Out("clean.csv") matches pivot.Dep("clean.csv")
pipeline.register(clean)   # pivot.Out("clean.csv")
pipeline.register(train)   # pivot.Dep("clean.csv")
```

## `IncrementalOut` as input

`IncrementalOut` is a special [output type](outputs.md#incrementalout) that also
appears as an input parameter. When a stage declares an `IncrementalOut` on both
a parameter and a return field, Pivot restores the previous output before the
stage runs. This does **not** create a DAG edge (it's self-referential, not a
cross-stage dependency).

```python
class CacheOutputs(TypedDict):
    cache: Annotated[dict, pivot.IncrementalOut("cache.json", pivot.loaders.JSON())]

def update_cache(
    cache: Annotated[dict, pivot.IncrementalOut("cache.json", pivot.loaders.JSON())],
    new_data: Annotated[DataFrame, pivot.Dep("new_data.csv", pivot.loaders.CSV())],
) -> CacheOutputs:
    cache["latest"] = new_data.to_dict()
    return {"cache": cache}
```

See [Outputs — IncrementalOut](outputs.md#incrementalout) for the full pattern.

## Summary

| Feature | How |
|---------|-----|
| Declare a dependency | `Annotated[T, pivot.Dep("path", loader)]` on a parameter |
| Multiple files | `pivot.Dep(["a.csv", "b.csv"], pivot.loaders.CSV())` or `pivot.Dep(("a.csv", "b.csv"), pivot.loaders.CSV())` |
| Defer path to registration | `pivot.PlaceholderDep(loader)` + `dep_path_overrides={}` |
| Directory dependency | `pivot.Dep("dir/", pivot.loaders.PathOnly())` |
| Override any dep path | `pipeline.register(fn, dep_path_overrides={"name": "new.csv"})` |
| Self-referential input | `pivot.IncrementalOut("path", loader)` on parameter (no DAG edge) |

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Outputs](outputs.md) | [Loaders](loaders.md)