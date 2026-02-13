# Multi-Pipeline Projects

Split large projects into multiple pipelines that Pivot automatically wires together through [artifact dependencies](../concepts/artifacts-and-dag.md). Each pipeline lives in its own directory with its own `pipeline.py`, and cross-pipeline dependencies are discovered at run time — no explicit configuration needed.

## When to Split

- **Team boundaries** — different teams own different parts of the workflow
- **Reusable data prep** — one pipeline produces datasets consumed by several analysis pipelines
- **Independent iteration** — each pipeline can run and test in isolation
- **Large monorepos** — subdirectories that should feel like separate projects

## Project Root

Pivot identifies the project root by walking up from the current directory to find the top-most `.pivot/` directory. Every `pipeline.py` under that root is discoverable.

```
my_project/
├── .pivot/              # ← project root marker
├── pipeline.py          # root pipeline
├── shared/
│   └── data.csv
└── analysis/
    └── pipeline.py      # child pipeline
```

Initialize once at the project root:

```bash
mkdir my_project && cd my_project
pivot init
```

## Parent/Child Pattern

The most common layout: shared data preparation at the root, specialized analysis in subdirectories.

### Root Pipeline — Produce Shared Data

```python
# my_project/pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pivot

pipeline = pivot.Pipeline("data_prep")


class PrepareOutputs(TypedDict):
    data: Annotated[pathlib.Path, pivot.Out("shared/data.csv", pivot.loaders.PathOnly())]


def prepare() -> PrepareOutputs:
    out = pathlib.Path("shared/data.csv")
    out.parent.mkdir(exist_ok=True)
    out.write_text("id,value\n1,100\n2,200\n3,300\n")
    return PrepareOutputs(data=out)


pipeline.register(prepare)
```

### Child Pipeline — Consume via Dep

```python
# my_project/analysis/pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pivot

pipeline = pivot.Pipeline("analysis")


class AnalyzeOutputs(TypedDict):
    report: Annotated[pathlib.Path, pivot.Out("report.txt", pivot.loaders.PathOnly())]


def analyze(
    data: Annotated[pathlib.Path, pivot.Dep("../shared/data.csv", pivot.loaders.PathOnly())],
) -> AnalyzeOutputs:
    content = data.read_text()
    lines = len(content.strip().split("\n")) - 1
    out = pathlib.Path("report.txt")
    out.write_text(f"Processed {lines} records\n")
    return AnalyzeOutputs(report=out)


pipeline.register(analyze)
```

### Run from the Child Directory

```bash
cd analysis
pivot repro
```

**What Pivot does:**

1. Walks up to `my_project/` (the project root with `.pivot/`)
2. Sees `analyze` needs `../shared/data.csv`
3. Searches from that path upward and finds `my_project/pipeline.py`
4. Discovers that `prepare` produces `shared/data.csv`
5. Runs `prepare` → `analyze` in dependency order

No `include()` calls, no configuration — the [DAG](../concepts/artifacts-and-dag.md) emerges from artifact paths.

## Sibling Pattern

Pipelines at the same directory level that depend on each other.

```
my_project/
├── .pivot/
└── pipelines/
    ├── features/
    │   └── pipeline.py      # produces output.csv
    └── model/
        └── pipeline.py      # consumes ../features/output.csv
```

### Feature Pipeline

```python
# pipelines/features/pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pivot

pipeline = pivot.Pipeline("features")


class FeatureOutputs(TypedDict):
    output: Annotated[pathlib.Path, pivot.Out("output.csv", pivot.loaders.PathOnly())]


def compute_features() -> FeatureOutputs:
    out = pathlib.Path("output.csv")
    out.write_text("feature,value\nf1,10\nf2,20\n")
    return FeatureOutputs(output=out)


pipeline.register(compute_features)
```

### Model Pipeline

```python
# pipelines/model/pipeline.py
import pathlib
from typing import Annotated, TypedDict

import pivot

pipeline = pivot.Pipeline("model")


class ModelOutputs(TypedDict):
    result: Annotated[pathlib.Path, pivot.Out("predictions.csv", pivot.loaders.PathOnly())]


def train_model(
    features: Annotated[pathlib.Path, pivot.Dep("../features/output.csv", pivot.loaders.PathOnly())],
) -> ModelOutputs:
    data = features.read_text()
    out = pathlib.Path("predictions.csv")
    out.write_text(f"# Model trained on features\n{data}")
    return ModelOutputs(result=out)


pipeline.register(train_model)
```

### Run

```bash
cd pipelines/model
pivot repro
```

Pivot discovers `features/pipeline.py` because the [dependency](../concepts/dependencies.md) `../features/output.csv` points into the `features/` directory.

## Running All Pipelines

From any directory, use `--all` to discover and run every pipeline under the project root:

```bash
pivot repro --all
pivot list --all
pivot status --all
```

This is useful for CI where you want to ensure the entire project is up to date.

## How Discovery Works

When a stage declares a `Dep` on a path outside its own pipeline directory, Pivot:

1. Resolves the path relative to the stage's working directory
2. Walks from that path upward looking for `pipeline.py` (or `pivot.yaml`)
3. Loads the discovered pipeline and checks if any stage produces the requested file
4. Adds that stage to the execution graph

This happens recursively — if the discovered stage itself has cross-pipeline dependencies, those are resolved too.

## Guidelines

**Split at natural boundaries.** Data prep, feature engineering, model training, and reporting are good candidates for separate pipelines.

**Each pipeline should run independently.** For testing and CI, any pipeline should work when run in isolation (assuming its upstream dependencies are cached).

**Use descriptive directory names.** `data_prep/`, `model_training/`, `reports/` — the directory name communicates intent. Pipeline names (the string passed to `Pipeline()`) should match purpose, not location.

**Commit lock files.** Lock files live under `.pivot/stages/` (the project-level state directory by default). Commit them so that `pivot pull` on another machine can restore outputs without re-running.

**Keep dependencies explicit.** Cross-pipeline wiring happens through `Dep` paths. If you can't express the dependency as a file path, the stages probably belong in the same pipeline.

## Related

- [Artifacts & DAG](../concepts/artifacts-and-dag.md) — how the dependency graph emerges from artifacts
- [Dependencies](../concepts/dependencies.md) — `Dep`, `Out`, and how stages connect
- [Pipelines](../concepts/pipelines.md) — `Pipeline` class, registration, and discovery
- [Watch Mode](./watch-mode.md) — auto-rerun across pipeline boundaries
