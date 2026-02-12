# Artifacts & the DAG

Pivot pipelines are **artifact-first**: you declare what files each stage reads and
writes, and Pivot derives the execution graph automatically. There is no explicit
stage-to-stage wiring — the DAG emerges from matching output paths to dependency
paths.

## The mental model

Think about files, not stages:

> "This CSV changed. What needs to re-run because of that?"

Every file that a stage reads is a **dependency**. Every file it writes is an
**output**. When one stage's output path matches another stage's dependency path,
Pivot creates a directed edge in the DAG. The result is a bipartite graph of
artifacts and stages that encodes exactly which work must happen, in which order.

```
  raw.csv          (artifact — no producer, input file)
    │
    ▼
  clean            (stage — reads raw.csv, writes clean.csv)
    │
    ▼
  clean.csv        (artifact — produced by clean)
    │
    ▼
  train            (stage — reads clean.csv, writes model.pkl + metrics.json)
    │
    ├──────────┐
    ▼          ▼
  model.pkl  metrics.json
```

## Defining a pipeline

A pipeline is a Python module that creates a `Pipeline` object and registers
stage functions. Pivot discovers it via `pipeline.py` (or `pivot.yaml`) at the
project root or in subdirectories.

```python
# pipeline.py
from typing import Annotated, TypedDict

from pandas import DataFrame

import pivot

pipeline = pivot.Pipeline("etl")


def clean(
    raw: Annotated[DataFrame, pivot.Dep("raw.csv", pivot.loaders.CSV())],
) -> Annotated[DataFrame, pivot.Out("clean.csv", pivot.loaders.CSV())]:
    """Drop rows with missing values."""
    return raw.dropna()


class TrainOutputs(TypedDict):
    model: Annotated[bytes, pivot.Out("model.pkl", pivot.loaders.Pickle())]
    metrics: Annotated[dict, pivot.Out("metrics.json", pivot.loaders.JSON())]


def train(
    data: Annotated[DataFrame, pivot.Dep("clean.csv", pivot.loaders.CSV())],
) -> TrainOutputs:
    """Train a model and emit metrics."""
    ...


pipeline.register(clean)
pipeline.register(train)
```

That's it. The DAG has two stages and three artifacts, with `clean.csv` bridging
`clean` to `train`. No YAML wiring, no adjacency lists.

## How edges form

Pivot builds edges with a simple rule:

| Condition | Result |
|-----------|--------|
| Stage A has `Out("X")`, Stage B has `Dep("X")` | A -> B edge |
| Stage A has `DirectoryOut("dir/")`, Stage B has `Dep("dir/file.csv")` | A -> B edge |
| Stage A has `Out("dir/file.csv")`, Stage B has `Dep("dir/")` | A -> B edge |
| No stage produces path X, Stage B has `Dep("X")` | X is an **external input** (must exist on disk) |

Directory dependencies use prefix matching — a `Dep` on a directory creates
edges to every stage that writes files inside that directory, and vice versa.

### What triggers a re-run

Pivot uses content-addressed invalidation. A stage re-runs when **any** of
these change:

- Content hash of any dependency file
- Stage function code (AST fingerprint)
- [Loader](loaders.md) code (AST fingerprint of Reader/Writer classes)
- Parameter values (`StageParams` fields)

Same inputs + same code = same outputs. Pivot skips the stage entirely.

## Registering stages

`Pipeline.register()` extracts [dependencies](dependencies.md) and
[outputs](outputs.md) from type annotations — no separate configuration needed:

```python
pipeline.register(clean)                    # name inferred from function: "clean"
pipeline.register(clean, name="clean_v2")   # explicit name
```

## Composing pipelines

Split large projects into multiple `Pipeline` objects and combine them with
`include()`:

```python
# pipelines/training/pipeline.py
import pivot
training = pivot.Pipeline("training")
# ... register stages ...

# pipeline.py  (root)
import pivot
from pipelines.training.pipeline import training

root = pivot.Pipeline("main")
root.include(training)
```

`include()` copies stages from the source pipeline. If stage names collide, the
incoming stages are automatically prefixed with the source pipeline's name
(e.g. `training/clean`). Cross-pipeline dependencies resolve through the same
path-matching rules — if `training` produces `models/output.pkl` and a root
stage depends on it, the edge forms automatically.

## Visualizing the DAG

The `pivot dag` command renders the graph in several formats:

```bash
pivot dag                  # ASCII art (default)
pivot dag --stages         # Show stage nodes instead of artifacts
pivot dag --mermaid        # Mermaid flowchart syntax
pivot dag --dot            # Graphviz DOT format
pivot dag --md             # Mermaid wrapped in markdown fences
pivot dag train            # Subgraph: train + its upstream deps
```

**ASCII output** (default) draws boxes connected by lines:

```
+----------+
| raw.csv  |
+----------+
      *
      *
+-----------+
| clean.csv |
+-----------+
      *
      *
+------------+     +--------------+
| model.pkl  |     | metrics.json |
+------------+     +--------------+
```

Use `--stages` to see stage names instead of file paths. Use `--dot` and pipe
to Graphviz for publication-quality renders:

```bash
pivot dag --dot | dot -Tpng -o dag.png
```

## Cycle detection

Pivot rejects circular dependencies at DAG build time. If stage A depends on
an output of stage B, and B depends on an output of A, registration succeeds
but `pivot repro` (or any command that builds the DAG) raises a
`CyclicGraphError` with the cycle path.

## Execution order

`pivot repro` uses depth-first post-order traversal to determine execution
order — dependencies always run before the stages that consume them. To run
a specific stage **and** its upstream dependencies:

```bash
pivot repro train          # Runs clean first, then train
```

To run a stage **without** its dependencies (useful for re-running a single
stage when inputs haven't changed):

```bash
pivot run train            # Runs ONLY train
```

## Key takeaways

1. **Declare artifacts, not edges.** The DAG is a consequence of `Dep` and `Out` paths matching.
2. **One output, one producer.** Each file path can be produced by at most one stage.
3. **External inputs are fine.** Dependencies without a producer are expected to exist on disk.
4. **Content-addressed.** Same inputs + same code = skip. No timestamps, no manual flags.

---

**Next:** [Dependencies](dependencies.md) | [Outputs](outputs.md) | [Loaders](loaders.md)