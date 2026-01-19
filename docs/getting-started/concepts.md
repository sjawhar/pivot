# Core Concepts

Understanding these concepts will help you get the most out of Pivot.

## Stages

A **stage** is a unit of work in your pipeline. Each stage:

- Has **dependencies** (inputs it reads)
- Produces **outputs** (files it writes)
- Contains a **function** that does the actual work

```yaml
# pivot.yaml
stages:
  process:
    python: stages.process
    deps:
      raw: input.csv
    outs:
      clean: output.parquet
```

```python
# stages.py
import pathlib
from typing import Annotated, TypedDict

import pandas
from pivot import loaders, outputs


class ProcessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("output.parquet", loaders.PathOnly())]


def process(
    raw: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> ProcessOutputs:
    df = raw.dropna()
    out_path = pathlib.Path("output.parquet")
    df.to_parquet(out_path)
    return {"clean": out_path}
```

## Dependency Graph (DAG)

Pivot builds a **Directed Acyclic Graph** from your stages:

```
data.csv
    │
    ▼
preprocess ──► processed.parquet
                    │
                    ▼
               train ──► model.pkl
```

Stages run in **topological order** - a stage only runs after all its dependencies are ready.

## Automatic Code Fingerprinting

Pivot tracks changes to your code automatically:

```python
def helper(x):
    return x * 2  # Change detected!

def process(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> ProcessOutputs:
    result = helper(data)  # Pivot knows helper is used
    ...
```

**How it works:**

1. Inspects your function's closure variables
2. Parses AST for `module.function` patterns
3. Recursively fingerprints all dependencies
4. Hashes the normalized code

## Caching

Pivot uses **content-addressable storage**:

```
.pivot/
├── cache/
│   └── files/
│       ├── ab/cdef0123...  # File content by hash
│       └── ...
└── stages/
    ├── preprocess.lock     # Per-stage lock file
    └── train.lock
```

When a stage runs:

1. Outputs are hashed and stored in `.pivot/cache/`
2. Lock file records the fingerprint (code + params + deps)
3. On next run, Pivot compares fingerprints to decide if re-execution is needed

## Skip Conditions

A stage is **skipped** when:

- Code fingerprint matches
- Parameters match
- All input dependencies match
- All outputs exist in cache

A stage **runs** when any of these change.

## Parallel Execution

Pivot runs independent stages in parallel using a **warm worker pool**:

```
Stage A ─────────┐
                 ├──► Stage C
Stage B ─────────┘
```

Stages A and B run simultaneously. Stage C waits for both to complete.

## Output Types

| Type | Cached | Git-Tracked | Use Case |
|------|--------|-------------|----------|
| `Out` | Yes | No | Large data files |
| `Metric` | No | Yes | Small metrics (JSON) |
| `Plot` | Yes | No | Visualization files |
| `IncrementalOut` | Yes | No | Append-only files |

## Parameters

Parameters are defined as Pydantic models:

```python
# stages.py
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    learning_rate: float = 0.01
    epochs: int = 100


def train(
    params: TrainParams,
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> TrainOutputs:
    print(f"LR: {params.learning_rate}")
    ...
```

Override defaults in YAML:

```yaml
# pivot.yaml
stages:
  train:
    python: stages.train
    deps:
      data: data.csv
    outs:
      model: model.pkl
    params:
      learning_rate: 0.05
      epochs: 200
```

## Next Steps

- [Defining Stages](../guide/stages.md) - Deep dive into stage configuration
- [Parameters](../guide/parameters.md) - Learn about parameters
- [Output Types](../guide/outputs.md) - Learn about output types
