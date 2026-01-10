# Core Concepts

Understanding these concepts will help you get the most out of Pivot.

## Stages

A **stage** is a unit of work in your pipeline. Each stage:

- Has **dependencies** (inputs it reads)
- Produces **outputs** (files it writes)
- Contains a **function** that does the actual work

```python
@stage(deps=['input.csv'], outs=['output.parquet'])
def process():
    # This function is the stage's work
    pass
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

@stage(deps=['data.csv'])
def process():
    return helper(load_data())  # Pivot knows helper is used
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

Type-safe parameters using Pydantic:

```python
from pydantic import BaseModel

class TrainParams(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 100

@stage(deps=['data.csv'], params=TrainParams)
def train(params: TrainParams):
    print(f"LR: {params.learning_rate}")
```

Override via `params.yaml`:

```yaml
train:
  learning_rate: 0.001
```

## Next Steps

- [Defining Stages](../guide/stages.md) - Deep dive into stage configuration
- [Parameters](../guide/parameters.md) - Learn about Pydantic parameters
- [Caching & Remote Storage](../guide/caching.md) - Share cache with your team
