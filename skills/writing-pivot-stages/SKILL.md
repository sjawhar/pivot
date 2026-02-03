---
name: writing-pivot-stages
description: Use when writing Pivot pipeline stages, seeing annotation errors (Dep, Out, Annotated), loader mismatches, "cannot pickle" errors, DirectoryOut validation failures, or IncrementalOut path mismatches
---

# Writing Pivot Stages

## Overview

Pivot stages are pure Python functions declaring file I/O via type annotations. The framework handles loading, saving, caching, and DAG construction.

**Core principle:** Annotations handle all file I/O. Functions receive pre-loaded data and return data to be saved.

## Imports

```python
from typing import Annotated, TypedDict
from pivot.outputs import Dep, Out, Metric, Plot, PlaceholderDep, IncrementalOut, DirectoryOut
from pivot.loaders import CSV, JSON, JSONL, YAML, Text, Pickle, PathOnly, MatplotlibFigure
from pivot.loaders import Reader, Writer, Loader  # Base classes for custom loaders
from pivot.stage_def import StageParams
from pivot.pipeline import Pipeline
```

## Stage Anatomy

```python
class MyParams(StageParams):
    threshold: float = 0.5

class MyOutputs(TypedDict):
    result: Annotated[pd.DataFrame, Out("output.csv", CSV())]
    metrics: Annotated[dict, Metric("metrics.json")]

def my_stage(
    params: MyParams,
    data: Annotated[pd.DataFrame, Dep("input.csv", CSV())],
) -> MyOutputs:
    filtered = data[data["score"] > params.threshold]
    return {"result": filtered, "metrics": {"count": len(filtered)}}

pipeline = Pipeline("my_pipeline")
pipeline.register(my_stage, params=MyParams(threshold=0.3))
```

**Single output:** Annotate return directly instead of TypedDict:

```python
def transform(
    data: Annotated[pd.DataFrame, Dep("input.csv", CSV())],
) -> Annotated[pd.DataFrame, Out("output.csv", CSV())]:
    return data.dropna()
```

## Loader Hierarchy

The loader system has three base classes:

| Base Class | Methods | Use Case |
|------------|---------|----------|
| `Reader[R]` | `load() -> R` | Read-only (dependencies) |
| `Writer[W]` | `save(data: W, ...)` | Write-only (outputs) |
| `Loader[W, R]` | Both `load()` and `save()` | Bidirectional (incremental outputs) |

**Type constraints:**
- `Dep.loader` accepts `Reader[R]` (or `Loader`, which extends `Reader`)
- `Out.loader` accepts `Writer[W]` (or `Loader`, which extends `Writer`)
- `IncrementalOut.loader` requires `Loader[W, R]` (needs both read and write)
- `DirectoryOut.loader` accepts `Writer[T]`

## Built-in Loaders

| Loader | Base | Data Type | Options | `empty()` |
|--------|------|-----------|---------|-----------|
| `CSV()` | `Loader` | DataFrame | `index_col`, `sep`, `dtype` | Yes |
| `JSON()` | `Loader` | dict/list | `indent=2`, `empty_factory=dict` | Yes |
| `JSONL()` | `Loader` | list[dict] | — | Yes |
| `YAML()` | `Loader` | dict/list | `empty_factory=dict` | Yes |
| `Text()` | `Loader` | str | — | Yes |
| `Pickle()` | `Loader` | Any | `protocol` | No |
| `PathOnly()` | `Loader` | Path | — | No |
| `MatplotlibFigure()` | `Writer` | Figure | `dpi=150`, `bbox_inches`, `transparent` | N/A |

**Notes:**
- Loaders with `empty()` support are required for `IncrementalOut`.
- `MatplotlibFigure` is `Writer[Figure]` (write-only) because images cannot be loaded back as Figure objects.

## Output Types

| Type | Default Cache | Git-Tracked | Use Case |
|------|---------------|-------------|----------|
| `Out` | True | No | Standard outputs |
| `Metric` | False | Yes | Small JSON metrics (path must end `.json`) |
| `Plot` | True | No | Visualizations |
| `IncrementalOut` | True | No | Builds on previous run's output |
| `DirectoryOut` | True | No | Dynamic set of files in directory |

## Multi-File Dependencies/Outputs

```python
# Variable-length list (count can change between runs)
shards: Annotated[list[pd.DataFrame], Dep(["a.csv", "b.csv"], CSV())]

# Fixed-length tuple (exact count enforced)
pair: Annotated[tuple[pd.DataFrame, pd.DataFrame], Dep(("x.csv", "y.csv"), CSV())]
```

## IncrementalOut

Previous output restored from cache before stage runs. Use for append-only state:

```python
class CacheOutputs(TypedDict):
    cache: Annotated[dict, IncrementalOut("cache.json", JSON())]

def incremental_stage(
    cache: Annotated[dict | None, IncrementalOut("cache.json", JSON())],  # Input
) -> CacheOutputs:
    existing = cache or {}
    existing["new_key"] = "value"
    return {"cache": existing}
```

**Rules:** Same path in input and output annotations. Loader must support `empty()`.

## DirectoryOut

For dynamic file sets determined at runtime:

```python
class TaskOutputs(TypedDict):
    results: Annotated[dict[str, dict], DirectoryOut("results/", JSON())]

def process_tasks(...) -> TaskOutputs:
    return {"results": {
        "task_a.json": {"accuracy": 0.95},
        "task_b.json": {"accuracy": 0.87},
    }}
```

**Rules:**
- Path must end with `/`
- Keys must have extensions, no path traversal (`../`), no absolute paths
- Dict must be non-empty

## PlaceholderDep

Dependency with no default path—must be overridden at registration:

```python
def compare(
    baseline: Annotated[pd.DataFrame, PlaceholderDep(CSV())],
) -> CompareOutputs: ...

pipeline.register(compare, dep_path_overrides={"baseline": "model_a/results.csv"})
```

## Matplotlib Plots

Plots require all three parts in the annotation:

```python
Annotated[
    matplotlib.figure.Figure,              # 1. The type (must be Figure, not Axes)
    Plot("plots/my_plot.png",              # 2. The output type (Plot, not Out)
         MatplotlibFigure(dpi=150))        # 3. The loader (required, handles save/close)
]
```

**Note:** `MatplotlibFigure` is a `Writer[Figure]` (not a full `Loader`) because saved images cannot be loaded back as matplotlib Figure objects. It only has a `save()` method.

**Full example:**

```python
import matplotlib.figure
import matplotlib.pyplot as plt

class PlotOutputs(TypedDict):
    plot: Annotated[matplotlib.figure.Figure, Plot("plots/my_plot.png", MatplotlibFigure())]

def make_plot(
    data: Annotated[pd.DataFrame, Dep("input.csv", CSV())],
) -> PlotOutputs:
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    return {"plot": fig}  # Return Figure, not Axes. Framework saves and closes.
```

## Path Overrides

```python
pipeline.register(my_stage, name="my_stage@v2", out_path_overrides={"result": "v2/output.csv"})
```

## Custom Loaders

Create custom loaders by extending the appropriate base class:

**Reader (read-only)** - for dependencies that only need loading:

```python
import dataclasses
import pathlib
from pivot.loaders import Reader

@dataclasses.dataclass(frozen=True)
class ImageReader(Reader[np.ndarray]):
    def load(self, path: pathlib.Path) -> np.ndarray:
        from PIL import Image
        return np.array(Image.open(path))
```

**Writer (write-only)** - for outputs that cannot be loaded back:

```python
@dataclasses.dataclass(frozen=True)
class HTMLWriter(Writer[str]):
    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)
```

**Loader (bidirectional)** - for `IncrementalOut` or symmetric I/O:

```python
@dataclasses.dataclass(frozen=True)
class NPY(Loader[np.ndarray, np.ndarray]):
    def load(self, path: pathlib.Path) -> np.ndarray:
        return np.load(path)

    def save(self, data: np.ndarray, path: pathlib.Path) -> None:
        np.save(path, data)

    def empty(self) -> np.ndarray:  # Required for IncrementalOut
        return np.array([])
```

**Rules:**
- Use `@dataclasses.dataclass(frozen=True)` for immutability and pickling
- Loaders must be module-level classes (not closures)
- Implement `empty()` only if the loader will be used with `IncrementalOut`

## Critical Rules

1. **All paths relative to project root** — not relative to stage file
2. **No manual file I/O** — no `pd.read_csv()`, `to_csv()`, `open()` in stage body
3. **File paths in annotations, not params** — `StageParams` for config only
4. **Stages must be module-level functions** — lambdas/closures fail pickling
5. **TypedDict outputs must be module-level** — not defined inside functions

## Running Stages

```bash
pivot repro                  # Run entire pipeline (DAG-aware)
pivot repro my_stage         # Run my_stage AND all dependencies
pivot repro --dry-run        # Validate DAG without executing

pivot run my_stage           # Run ONLY my_stage (no dependency resolution)
```

## Pipeline Composition

Include stages from sub-pipelines while preserving their state directories:

```python
# Define sub-pipeline
preprocessing = Pipeline("preprocessing")
preprocessing.register(clean_data, name="clean")
preprocessing.register(normalize, name="normalize")

# Include in main pipeline
main = Pipeline("main")
main.include(preprocessing)  # Deep-copies stages, preserves state_dir
main.register(train, name="train")  # Can depend on preprocessing outputs
```

**Behavior:**
- Included stages keep their original `state_dir` (for lock files, state.db)
- Stages are deep-copied: mutations don't propagate between pipelines
- `include()` is a point-in-time snapshot; later registrations in source don't propagate
- Including empty pipeline is a no-op
- Including same pipeline twice raises (name collision)
- Transitive: if B includes C, then A includes B, A gets C's stages (already in B's registry)

**Rules:**
- Stage name collisions raise `PipelineConfigError`
- Cannot include a pipeline into itself

**Security Note:** Only include pipelines from trusted sources. Included stages execute with the same privileges as your pipeline.

## Testing

Pass data directly (annotations are bypassed):

```python
def test_my_stage():
    result = my_stage(
        params=MyParams(threshold=0.5),
        data=pd.DataFrame({"score": [0.3, 0.7, 0.9]}),
    )
    assert len(result["result"]) == 2
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot pickle` | Closure/lambda as stage | Move to module-level function |
| `PlaceholderDep requires override` | Missing path | Add `dep_path_overrides` |
| `IncrementalOut path mismatch` | Input/output paths differ | Use same path in both annotations |
| `DirectoryOut path must end with '/'` | Missing trailing slash | Add `/` to path |
| `DirectoryOut key must have extension` | Key like `"task_a"` | Use `"task_a.json"` |
| `loader is required` | `Out("file.json")` without loader | Add loader: `Out("file.json", JSON())` |
| `TypedDict field missing Out annotation` | Field without `Out`/`Metric`/`Plot` | Add annotation to all fields |
| `stage 'X' already exists` | Name collision in `include()` | Rename stage with `name=` at registration |
| `cannot include itself` | Self-include attempted | Use a separate Pipeline instance |

## Checklist

- [ ] No manual file I/O in stage function
- [ ] No file paths in `StageParams`
- [ ] All Dep/Out paths relative to project root
- [ ] Stage is module-level function (not closure)
- [ ] TypedDict outputs defined at module level
- [ ] Ran `pivot run` and verified outputs exist
