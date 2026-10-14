# Multi-Pipeline Projects

Split a large project into several pipelines, each in its own directory with its own `pipeline.py`. One pipeline consumes another's output by importing the handle, and Pivot runs whatever upstream stages that handle needs.

## When to Split

- **Team boundaries**: different teams own different parts of the workflow
- **Reusable data prep**: one pipeline produces datasets consumed by several analysis pipelines
- **Independent iteration**: each pipeline can run on its own
- **Large monorepos**: subdirectories that should feel like separate projects

## Project Root and Imports

The project root is the top-most directory containing `.pivot/`. Every `pipeline.py` under it is discoverable, and imports resolve from it: a root `pipeline.py` imports sibling modules directly (`from stages import clean`); a pipeline in a subdirectory imports with the full path (`from report.helpers import x`, `from prep.pipeline import kept`).

```
my_project/
├── .pivot/              # project root marker
├── data/raw/scores.csv
├── prep/
│   └── pipeline.py      # pipeline "prep"
└── report/
    └── pipeline.py      # pipeline "report", consumes prep's output
```

```bash
mkdir my_project && cd my_project
git init && pivot init
mkdir -p data/raw prep report
printf 'name,score\na,0.2\nb,0.7\nc,0.9\n' > data/raw/scores.csv
```

## Upstream Pipeline

```python
# prep/pipeline.py
import pandas as pd

from pivot import Pipeline, StageParams, stage


class DropLowParams(StageParams):
    min_score: float = 0.5


@stage
def drop_low(raw: pd.DataFrame, params: DropLowParams) -> pd.DataFrame:
    return raw[raw["score"] >= params.min_score]


with Pipeline("prep") as pipeline:
    kept = drop_low(pipeline.input("scores.csv"), params=DropLowParams())
```

`kept` is a module-level variable (a `with` block does not create a scope), so other modules can import it.

## Downstream Pipeline

```python
# report/pipeline.py
from typing import Annotated

import pandas as pd

from pivot import Pipeline, metric, stage
from prep.pipeline import kept


@stage
def count(rows: pd.DataFrame) -> Annotated[dict, metric]:
    return {"rows": len(rows)}


with Pipeline("report") as pipeline:
    count(kept)
```

### Run from the Downstream Directory

```bash
cd report
pivot repro
```

Pivot loads `report/pipeline.py`, sees that `count` consumes `prep/drop_low`'s output, adds `prep/drop_low` to the run, and executes `prep/drop_low` then `report/count`. Each stage keeps its own pipeline prefix and writes under its own pipeline's directories (`data/prep/drop_low.jsonl`, `metrics/report/count.yaml`).

## Running All Pipelines

From anywhere in the project, `--all` loads every `pipeline.py` under the root and runs them as one graph:

```bash
pivot repro --all
pivot list --all
pivot status --all
```

## One Root Pipeline with `include()`

To make `pivot repro` at the root run everything without `--all`, give the root its own `pipeline.py` that includes the others:

```python
# pipeline.py
from pivot import Pipeline
from prep.pipeline import pipeline as prep
from report.pipeline import pipeline as report

with Pipeline("all") as pipeline:
    pipeline.include(prep)
    pipeline.include(report)
```

`include()` can be called inside the `with` block or after it. Included stages keep their names (`prep/drop_low`, `report/count`) and state; stages included twice (here `prep/drop_low`, which `report` already pulls in) are added once. Pipeline names must be unique within a project, so two stages under one name are refused unless they are one function called with the same inputs, outputs, params, resources and variant. The error names the first difference and the source files.

## Guidelines

**Split at natural boundaries.** Data prep, feature engineering, model training and reporting are good candidates for separate pipelines.

**Share data through handles, not paths.** A downstream pipeline imports the handle; it never reads `data/prep/...` itself.

**Give each pipeline a distinct name.** The name prefixes every stage and output directory.

**Commit lock files.** Lock files live under `.pivot/stages/<pipeline>/`. Commit them so that `pivot pull` on another machine can restore outputs without re-running.

## Related

- [Artifacts & DAG](../concepts/artifacts-and-dag.md): how handles form the graph
- [Pipelines](../concepts/pipelines.md): `Pipeline`, variants and `include()`
- [Watch Mode](./watch-mode.md): auto-rerun during development
