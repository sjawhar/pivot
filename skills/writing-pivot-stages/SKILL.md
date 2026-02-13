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
import pivot  # Single import — access everything via pivot.*

from typing import Annotated, TypedDict
```

All Pivot types are accessed via the `pivot` namespace:

| What | Access |
|------|--------|
| Dependencies | `pivot.Dep`, `pivot.PlaceholderDep` |
| Outputs | `pivot.Out`, `pivot.Metric`, `pivot.Plot`, `pivot.IncrementalOut`, `pivot.DirectoryOut` |
| Loaders | `pivot.loaders.CSV()`, `pivot.loaders.JSON()`, `pivot.loaders.YAML()`, etc. |
| Base classes | `pivot.loaders.Reader`, `pivot.loaders.Writer`, `pivot.loaders.Loader` |
| Params | `pivot.StageParams` |
| Pipeline | `pivot.Pipeline` |
| Decorators | `pivot.no_fingerprint` |

## Stage Anatomy

```python
import pivot

class MyParams(pivot.StageParams):
    threshold: float = 0.5

class MyOutputs(TypedDict):
    result: Annotated[pd.DataFrame, pivot.Out("output.csv", pivot.loaders.CSV())]
    metrics: Annotated[dict, pivot.Metric("metrics.json")]

def my_stage(
    params: MyParams,
    data: Annotated[pd.DataFrame, pivot.Dep("input.csv", pivot.loaders.CSV())],
) -> MyOutputs:
    filtered = data[data["score"] > params.threshold]
    return {"result": filtered, "metrics": {"count": len(filtered)}}

pipeline = pivot.Pipeline("my_pipeline")
pipeline.register(my_stage, params=MyParams(threshold=0.3))
```

**Single output:** Annotate return directly instead of TypedDict:

```python
def transform(
    data: Annotated[pd.DataFrame, pivot.Dep("input.csv", pivot.loaders.CSV())],
) -> Annotated[pd.DataFrame, pivot.Out("output.csv", pivot.loaders.CSV())]:
    return data.dropna()
```

## Loader Hierarchy

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
| `DataFrameJSONL()` | `Loader` | DataFrame | — | Yes |
| `YAML()` | `Loader` | dict/list | `empty_factory=dict` | Yes |
| `Text()` | `Loader` | str | — | Yes |
| `Pickle()` | `Loader` | Any | `protocol` | No |
| `PathOnly()` | `Loader` | Path | — | No |
| `MatplotlibFigure()` | `Writer` | Figure | `dpi=150`, `bbox_inches`, `transparent` | N/A |

**Notes:**
- `DataFrameJSONL()` reads/writes DataFrames as JSON Lines (orient=records). Preferred over `CSV()` for non-trivial DataFrames (preserves types, handles nested data).
- Loaders with `empty()` support are required for `IncrementalOut`.
- `MatplotlibFigure` is `Writer[Figure]` (write-only) — images can't be loaded back as Figure objects.

## Output Types

| Type | Default Cache | Git-Tracked | Use Case |
|------|---------------|-------------|----------|
| `Out` | True | No | Standard outputs |
| `Metric` | False | Yes | Small YAML/JSON metrics |
| `Plot` | True | No | Visualizations |
| `IncrementalOut` | True | No | Builds on previous run's output |
| `DirectoryOut` | True | No | Dynamic set of files in directory |

## Multi-File Dependencies/Outputs

```python
# Variable-length list (count can change between runs)
shards: Annotated[list[pd.DataFrame], pivot.Dep(["a.csv", "b.csv"], pivot.loaders.CSV())]

# Fixed-length tuple (exact count enforced)
pair: Annotated[tuple[pd.DataFrame, pd.DataFrame], pivot.Dep(("x.csv", "y.csv"), pivot.loaders.CSV())]
```

## IncrementalOut

Previous output restored from cache before stage runs. Use for append-only state:

```python
class CacheOutputs(TypedDict):
    cache: Annotated[dict, pivot.IncrementalOut("cache.json", pivot.loaders.JSON())]

def incremental_stage(
    cache: Annotated[dict | None, pivot.IncrementalOut("cache.json", pivot.loaders.JSON())],
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
    results: Annotated[dict[str, dict], pivot.DirectoryOut("results/", pivot.loaders.JSON())]

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

Dependency with no default path — must be overridden at registration:

```python
def compare(
    baseline: Annotated[pd.DataFrame, pivot.PlaceholderDep(pivot.loaders.CSV())],
) -> CompareOutputs: ...

pipeline.register(compare, dep_path_overrides={"baseline": "model_a/results.csv"})
```

## Matplotlib Plots

Plots require all three parts in the annotation:

```python
Annotated[
    matplotlib.figure.Figure,                       # 1. Type (must be Figure, not Axes)
    pivot.Plot("plots/my_plot.png",                 # 2. Output type (Plot, not Out)
               pivot.loaders.MatplotlibFigure())    # 3. Writer (handles save/close)
]
```

**Full example:**

```python
import matplotlib.figure
import matplotlib.pyplot as plt
import pivot

class PlotOutputs(TypedDict):
    plot: Annotated[matplotlib.figure.Figure, pivot.Plot("plots/my.png", pivot.loaders.MatplotlibFigure())]

def make_plot(
    data: Annotated[pd.DataFrame, pivot.Dep("input.csv", pivot.loaders.CSV())],
) -> PlotOutputs:
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    return {"plot": fig}  # Return Figure, not Axes. Framework saves and closes.
```

## Path Overrides and Variants

Override paths at registration time — useful for running same stage with different inputs/outputs:

```python
# Simple override
pipeline.register(my_stage, name="my_stage@v2", out_path_overrides={"result": "v2/output.csv"})

# Variant pattern: register same function multiple times with different paths
for variant, suffix in [("current", ""), ("legacy", "_legacy")]:
    pipeline.register(
        merge_data,
        name=f"merge_data@{variant}",
        dep_path_overrides={"input": f"data/raw/input{suffix}.jsonl"},
        out_path_overrides={"output": f"data/processed/output{suffix}.jsonl"},
        params=MergeParams(suffix=suffix),
    )
```

## Path Resolution

- Paths in annotations are **relative to the pipeline root** (defaults to directory of the file calling `Pipeline()`)
- Parent references (`..`) are resolved during registration — `Dep("../shared/data.csv")` works
- Resolved path must stay within the **project root** (the top-most directory with `.pivot/`)
- Dependencies may use absolute paths (reduces portability). Outputs must resolve within project root.
- For multi-pipeline projects, set `root=pivot.project.get_project_root()` to share a common base:

```python
pipeline = pivot.Pipeline("my_pipeline", root=pivot.project.get_project_root())
```

## Custom Loaders

Extend the appropriate base class. Use `@dataclasses.dataclass(frozen=True)` for immutability and pickling.

```python
import dataclasses
import pathlib
import pivot

# Reader (read-only) — for Dep
@dataclasses.dataclass(frozen=True)
class ImageReader(pivot.loaders.Reader[np.ndarray]):
    def load(self, path: pathlib.Path) -> np.ndarray:
        from PIL import Image
        return np.array(Image.open(path))

# Writer (write-only) — for Out/Plot
@dataclasses.dataclass(frozen=True)
class HTMLWriter(pivot.loaders.Writer[str]):
    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)

# Loader (bidirectional) — for IncrementalOut or symmetric I/O
@dataclasses.dataclass(frozen=True)
class NPY(pivot.loaders.Loader[np.ndarray, np.ndarray]):
    def load(self, path: pathlib.Path) -> np.ndarray:
        return np.load(path)

    def save(self, data: np.ndarray, path: pathlib.Path) -> None:
        np.save(path, data)

    def empty(self) -> np.ndarray:  # Required for IncrementalOut
        return np.array([])
```

**Rules:**
- Loaders must be module-level classes (not closures)
- Implement `empty()` only if used with `IncrementalOut`

## StageParams Defaults

Pydantic deep-copies all defaults — lists, dicts, nested models. **Never use `default_factory`**, `.model_copy()`, or `.copy()`:

```python
# WRONG — unnecessary complexity
class MyParams(pivot.StageParams):
    exclude: list[str] = pydantic.Field(default_factory=list)
    base_cfg: PlotConfig = pydantic.Field(default_factory=lambda: PlotConfig())

# CORRECT — Pydantic handles safely
class MyParams(pivot.StageParams):
    exclude: list[str] = []
    styling: dict[str, Any] = {}
    percents: list[int] = [50, 80]
    nested: SubModel = SubModel()
```

Only use `pydantic.Field()` when you need its features (`alias`, `description`, `ge`), not for defaults. When cleaning up `default_factory`, also strip redundant `pydantic.Field()` wrappers and unused `pydantic` imports in the same pass.

Don't add `field_validator` or `model_validator` unless strictly necessary — keep params as plain data. Field types must match what the corresponding `Dep` inputs provide. Don't leave placeholder fields that bypass a param pathway — wire it fully or remove it.

## Pipeline Composition

```python
preprocessing = pivot.Pipeline("preprocessing")
preprocessing.register(clean_data, name="clean")

main = pivot.Pipeline("main")
main.include(preprocessing)  # Deep-copies stages
main.register(train, name="train")
```

**Behavior:**
- Included stages keep their original `state_dir`
- Deep-copied: mutations don't propagate
- Point-in-time snapshot; later registrations in source don't propagate
- Name collisions are auto-prefixed with `{other.name}/` to disambiguate
- Cannot include a pipeline into itself

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

## Critical Rules

1. **All paths relative to project root** — not relative to stage file
2. **No manual file I/O** — no `pd.read_csv()`, `to_csv()`, `open()` in stage body
3. **File paths in annotations, not params** — `StageParams` for config only
4. **No `default_factory` for mutable defaults** — Pydantic deep-copies; use `= []`, `= {}`, `= Model()` directly
5. **Stages must be module-level functions** — lambdas/closures fail pickling
6. **TypedDict outputs must be module-level** — not defined inside functions
7. **Use `import pivot`** — then `pivot.Dep`, `pivot.Out`, `pivot.loaders.CSV()`, etc.
8. **StageParams field types must match Dep types** — don't leave placeholder fields that bypass a param pathway

## Running Stages

```bash
pivot repro                  # Run entire pipeline (DAG-aware)
pivot repro my_stage         # Run my_stage AND all dependencies
pivot repro --dry-run        # Validate DAG without executing
pivot run my_stage           # Run ONLY my_stage (no dependency resolution)
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot pickle` | Closure/lambda as stage | Move to module-level function |
| `PlaceholderDep requires override` | Missing path | Add `dep_path_overrides` |
| `IncrementalOut path mismatch` | Input/output paths differ | Use same path in both annotations |
| `DirectoryOut path must end with '/'` | Missing trailing slash | Add `/` to path |
| `DirectoryOut key must have extension` | Key like `"task_a"` | Use `"task_a.json"` |
| `loader is required` | `Out("file.json")` without loader | Add loader: `pivot.Out("file.json", pivot.loaders.JSON())` |
| `TypedDict field missing Out annotation` | Field without `Out`/`Metric`/`Plot` | Add annotation to all fields |
| `stage 'X' already exists` | Duplicate registration | Use distinct `name=` at registration. Note: `include()` auto-prefixes on collision (`{pipeline.name}/stage`) |
| `resolves outside base directory` | Output path escapes project root | Keep output paths within project |

## Checklist

- [ ] Using `import pivot` (not `from pivot.outputs import ...`)
- [ ] No manual file I/O in stage function
- [ ] No file paths in `StageParams`
- [ ] No `default_factory` in `StageParams`
- [ ] All Dep/Out paths relative to project root
- [ ] Stage is module-level function (not closure)
- [ ] TypedDict outputs defined at module level
- [ ] No `field_validator`/`model_validator` in `StageParams` unless strictly needed
- [ ] Ran `pivot repro --dry-run` to validate DAG
