# Pipelines

A pipeline groups stages into a unit with its own root directory and state.
You create one with `Pipeline(name)`, register stages on it, and Pivot
handles dependency resolution, DAG construction, and cross-pipeline
discovery.

!!! tip "pipeline.py vs pivot.yaml"
    **Start with `pipeline.py`** (recommended). You get full IDE support, type checking,
    composition with `include()`, matrix stages, and dynamic registration.

    Use `pivot.yaml` when you need to override paths/params without touching Python,
    or when migrating from DVC. See [The YAML Alternative](#the-yaml-alternative) below.

## Creating a Pipeline

```python
# pipeline.py
import pivot

pipeline = pivot.Pipeline("my_pipeline")
```

The `Pipeline` constructor takes a name and an optional `root`:

```python
Pipeline(
    name: str,           # Identifier — alphanumeric, underscore, hyphen
    root: Path | None,   # Defaults to the directory containing this file
)
```

When you omit `root`, Pivot inspects the call stack to find the caller's
`__file__` and uses its parent directory. This means `pipeline.py` files
are self-locating — the pipeline root is always the directory the file
lives in.

### Pipeline Root and Paths

All artifact paths in annotations are resolved relative to the pipeline
root, not the project root. If your pipeline lives at
`pipelines/training/pipeline.py`, then `pivot.Dep("data.csv", pivot.loaders.CSV())` resolves
to `pipelines/training/data.csv`.

This is what makes pipelines relocatable. Move the directory and
everything still works.

## Registering Stages

```python
from typing import Annotated
from pandas import DataFrame
import pivot

def clean(
    raw: Annotated[DataFrame, pivot.Dep("raw.csv", pivot.loaders.CSV())],
) -> Annotated[DataFrame, pivot.Out("clean.csv", pivot.loaders.CSV())]:
    return raw.dropna()

def transform(
    data: Annotated[DataFrame, pivot.Dep("clean.csv", pivot.loaders.CSV())],
) -> Annotated[DataFrame, pivot.Out("features.csv", pivot.loaders.CSV())]:
    return data.assign(log_price=data["price"].apply(math.log))

pipeline.register(clean)
pipeline.register(transform)
```

`register()` extracts everything from annotations — dependencies, outputs,
parameter types — in a single pass over the function's type hints. You
rarely need to pass anything beyond the function itself.

### Full Signature

```python
pipeline.register(
    func,                              # Stage function (must be module-level)
    name=None,                         # Override name (default: func.__name__)
    params=None,                       # StageParams class or instance
    mutex=None,                        # Mutex groups for exclusive execution
    variant=None,                      # Variant name for matrix stages
    dep_path_overrides=None,           # Override dependency paths
    out_path_overrides=None,           # Override output paths
)
```

### Path Overrides

Override annotation paths at registration time to reuse the same function
with different inputs/outputs:

```python
pipeline.register(
    clean,
    name="clean_2024",
    dep_path_overrides={"raw": "raw_2024.csv"},
    out_path_overrides={"result": "clean_2024.csv"},
)
```

Override values can be a simple path string or a dict with options:

```python
pipeline.register(
    train,
    out_path_overrides={
        "model": {"path": "models/v2.pkl", "cache": False},
    },
)
```

### PlaceholderDep

For stages that are always registered with overrides, use `PlaceholderDep`
to declare the dependency shape without a default path:

```python
def process(
    data: Annotated[DataFrame, pivot.PlaceholderDep(pivot.loaders.CSV())],
) -> Annotated[DataFrame, pivot.Out("processed.csv", pivot.loaders.CSV())]:
    ...

# Must provide override — PlaceholderDep has no default path
pipeline.register(process, dep_path_overrides={"data": "input.csv"})
```

## Pipeline Composition with include()

`include()` merges all stages from another pipeline into the current one:

```python
# project_root/pipeline.py
import pivot
from pipelines.training.pipeline import pipeline as training
from pipelines.evaluation.pipeline import pipeline as evaluation

pipeline = pivot.Pipeline("main")
pipeline.include(training)
pipeline.include(evaluation)
```

Each included stage keeps its original `state_dir`, so lock files and
state databases remain isolated. The include is a point-in-time snapshot —
later changes to the source pipeline are not reflected.

### Name Collisions

If an included pipeline has stage names that conflict with existing ones,
Pivot automatically prefixes all incoming stages with `{pipeline.name}/`:

```python
# Both have a "clean" stage
pipeline.include(training)     # clean
pipeline.include(evaluation)   # evaluation/clean (auto-prefixed)
```

## Transitive Dependency Discovery

When a stage depends on a file that no local stage produces, Pivot
searches for the producer automatically using three-tier discovery:

1. **Traverse-up** — walk from the dependency file's directory up to
   project root, loading any `pipeline.py` or `pivot.yaml` found along
   the way
2. **Output index** — check the cached output index at
   `.pivot/cache/outputs/` for a hint about which pipeline produces this
   file
3. **Full scan** — scan all pipeline config files in the project

When a producer is found, it (and its own dependencies, transitively) are
included into the current pipeline. This is how multi-pipeline projects
work without explicit wiring:

```
project/
  pipelines/
    data/
      pipeline.py      # produces data/clean.csv
    training/
      pipeline.py      # depends on data/clean.csv → auto-discovered
```

The training pipeline doesn't need to `include()` the data pipeline
explicitly. Pivot discovers the dependency chain at `build_dag()` time.

### Discovery Order

When Pivot looks for a pipeline config in a directory, it checks:

1. `pivot.yaml` (or `pivot.yml`)
2. `pipeline.py`

If both exist in the same directory, Pivot raises a `DiscoveryError`.
Only one config format per directory is allowed.

## State and Cache Layout

By default, all pipelines share the project-level `.pivot/` directory:

```
<project_root>/
  .pivot/
    cache/             # Content-addressable cache (shared across pipelines)
    stages/            # Per-stage lock files
    state.lmdb/        # LMDB database (generations, hashes)
    locks/             # Artifact locks for concurrent execution
```

The cache is always project-wide. Lock files and StateDB live under
`core.state_dir` (default: `.pivot`). Pipelines can override `state_dir`
at construction time to isolate state (useful for multi-pipeline projects
with independent release cycles).

## The YAML Alternative

For simpler projects, you can define stages in `pivot.yaml` instead of
`pipeline.py`:

```yaml
stages:
  clean:
    cmd: pipelines.data.stages.clean
  transform:
    cmd: pipelines.data.stages.transform
```

This creates an implicit `Pipeline` from the YAML config. The Python-first
approach with `pipeline.py` is more flexible — it supports composition,
matrix stages, dynamic registration, and path overrides that YAML cannot
express. Run `pivot schema` to output the JSON Schema for `pivot.yaml`.
See the [parameters](parameters.md) page for how params work with both
approaches.

## DAG Construction

After all stages are registered (and external deps discovered), call
`build_dag()` to get the dependency graph:

```python
dag = pipeline.build_dag()
```

This builds a bipartite graph (stages + artifacts), validates there are no
cycles, checks for output overlaps, and returns a NetworkX `DiGraph` with
stage names as nodes.

The DAG is the input to the execution engine. `pivot repro` builds the
DAG and executes stages in topological order, parallelising independent
branches.

## Relationship to Other Concepts

- Stages declare their inputs via [dependencies](dependencies.md) and
  produce [outputs](outputs.md)
- Stage code is tracked via [fingerprinting](fingerprinting.md)
- Stage parameters are tracked via [parameters](parameters.md)
- Execution and skip logic use the [caching](caching.md) system
