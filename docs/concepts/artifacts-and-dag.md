# Artifacts & the DAG

Pivot pipelines are **artifact-first**: a stage is a function from data to data,
and the execution graph is whatever falls out of passing one stage's result to
another. There is no separate list of edges and no file path to keep in sync
between producer and consumer.

## The mental model

Think about data, not stages:

> "This input changed. What needs to re-run because of that?"

Every value a stage receives is an **input artifact**; every value it returns
is an **output artifact**. Pivot loads the inputs before the stage runs and saves
the outputs after it returns.

```
  scores.csv        (input — a file under data/raw/)
    │
    ▼
  clean             (stage — receives scores.csv as a DataFrame)
    │
    ▼
  etl/clean         (artifact — clean's return value)
    │
    ▼
  train             (stage — receives etl/clean)
    │
    ├──────────────┐
    ▼              ▼
  train:model    train:metrics
```

## Defining a pipeline

A pipeline is a `with Pipeline("name") as pipeline:` block in `pipeline.py`.
Inside the block, calling a `@stage` function does not run it: it records a
stage and returns a **handle** to the stage's output. Passing that handle to
another stage creates the edge.

```python
# pipeline.py
from typing import Annotated, TypedDict

import pandas as pd

from pivot import Pipeline, loaders, metric, stage


@stage
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values."""
    return raw.dropna()


class TrainResult(TypedDict):
    model: Annotated[dict, loaders.Pickle()]
    metrics: Annotated[dict, metric]


@stage
def train(data: pd.DataFrame) -> TrainResult:
    """Fit a trivial model and report its size."""
    model = {"mean": float(data["score"].mean())}
    return {"model": model, "metrics": {"rows": len(data)}}


with Pipeline("etl") as pipeline:
    raw = pipeline.input("scores.csv")
    cleaned = clean(raw)
    train(cleaned)
```

The DAG has one input, two stages and three output artifacts. `pivot repro`
runs `etl/clean` and then `etl/train`.

## Artifacts and where they live

Each artifact is identified by its producer and, for multi-output stages, a key:

| Artifact | Identity | Stored at |
|----------|----------|-----------|
| `pipeline.input("scores.csv")` | `scores.csv` | `data/raw/scores.csv` |
| `clean`'s return value | `etl/clean` | `data/etl/clean.jsonl` |
| `train`'s `model` field | `etl/train:model` | `data/etl/train/model.pkl` |
| `train`'s `metrics` field (tagged `metric`) | `etl/train:metrics` | `metrics/etl/train/metrics.yaml` |
| `split`'s return value (tagged `directory`) | `etl/split` | `data/etl/split/` |
| file `train.csv` of `split`'s directory, as a dependency | `etl/split:train.csv` | `data/etl/split/train.csv` |

Pivot chooses the storage path from the pipeline name, the stage name, the
output key and the [format](outputs.md). You never write a path for a stage
output, and a stage never opens a file to read an input or write an output.

A [directory output](outputs.md#directory-outputs) is one artifact with one
producer. A stage that takes one of its files (`fit(parts["train.csv"])`)
gets an edge from the producer in the DAG like any consumer, but its lock
records only that file's hash, so a change to another file in the directory
does not re-run it.

## What triggers a re-run

Invalidation is content-addressed. A stage re-runs when **any** of these change:

- Content hash of any input artifact
- Stage function code and the helpers it calls ([fingerprinting](fingerprinting.md))
- [Loader](loaders.md) code or configuration for its inputs and outputs
- [Parameter](parameters.md) values

Same inputs + same code + same parameters = same outputs. Pivot skips the stage.

## Visualizing the DAG

```bash
pivot dag                  # ASCII art of artifacts (default)
pivot dag --stages         # Stages as nodes instead of artifacts
pivot dag --mermaid        # Mermaid flowchart syntax
pivot dag --dot            # Graphviz DOT format
pivot dag --md             # Mermaid wrapped in markdown fences
pivot dag etl/train        # Subgraph: etl/train and its upstream
```

Pipe `--dot` to Graphviz for a rendered image:

```bash
pivot dag --dot | dot -Tpng -o dag.png
```

## Execution order

`pivot repro` runs dependencies before the stages that consume them and runs
independent branches in parallel. To run a stage **and** its upstream:

```bash
pivot repro etl/train      # Runs etl/clean first if needed, then etl/train
```

To run a stage **without** its dependencies:

```bash
pivot run etl/train        # Runs ONLY etl/train
```

Stage names are qualified by their pipeline (`etl/train`); `pivot list` prints them.

### Resource limits

Independent branches run in parallel when worker and resource capacity permit it. Decorate a stage with `@stage(resources={"gpu": 1})` to reserve a named resource while it runs. Non-CPU resources have capacity 1 by default; set `resources: {gpu: 2}` in `.pivot/config.yaml` or pass `--resources gpu=2` to `pivot repro` or `pivot run` to allow two GPU stages at once. Every stage holds one CPU unit, so `resources={"cpu": "all"}` runs without another stage. Resource declarations do not affect cache invalidation.

Pivot uses joblib's threading backend inside a stage by default. Python-level CPU work submitted through `joblib.Parallel` then runs one task at a time because of the GIL. For that workload, set `PIVOT_NESTED_PARALLELISM=processes` and declare `@stage(resources={"cpu": "all"})` so nested process pools do not overlap:

```bash
PIVOT_NESTED_PARALLELISM=processes pivot repro
```

In a 65-stage analysis project, this combination reduced an internally parallel bootstrap stage from 528 s to 28 s.

## Key takeaways

1. **Pass handles, not paths.** The DAG is the call graph of the `with` block.
2. **One producer per artifact.** Each handle refers to exactly one stage output or input file.
3. **Inputs live under `data/raw/`.** Anything a stage does not produce is declared with `pipeline.input()`.
4. **Content-addressed.** Same inputs + same code + same params = skip. No timestamps.

---

**Next:** [Dependencies](dependencies.md) | [Outputs](outputs.md) | [Loaders](loaders.md)
