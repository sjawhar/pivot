# Migrating from DVC

This guide ports a DVC pipeline to Pivot. A port is complete when the project
has no `dvc.yaml`, no `dvc.lock`, no `.dvc/`, no `import dvc.api`, and no stage
that opens a file itself. A half-ported pipeline, with Pivot stages that still
read `params.yaml` or pass file paths around, gets the drawbacks of both tools
and the change detection of neither.

## How DVC concepts map to Pivot

| DVC | Pivot |
|-----|-------|
| `dvc.yaml` stage with `cmd: python script.py` | A `@stage` function |
| `deps:` listing data files | Function parameters that receive handles |
| `deps:` listing `.py` files | Nothing: Pivot fingerprints the function and the helpers it calls |
| `outs:` paths | The function's return value; Pivot chooses the path |
| A path written by one stage and read by another | A handle passed from one stage call to the next |
| Raw data tracked with `dvc add` | `data/raw/<name>`, declared with `pipeline.input("<name>")` |
| `params.yaml` + `params:` + `dvc.api.params_show()` | One `StageParams` subclass per stage, passed at the call site |
| `metrics:` | `Annotated[dict, metric]` on the return type |
| `plots:` | `Annotated[Figure, plot]` on the return type |
| `foreach` / `matrix` stages | A Python loop in the `with Pipeline` block, optionally inside `pipeline.variant(name)` |
| `vars:` and `${...}` interpolation | Python values in `pipeline.py` and `StageParams` fields |
| Several `dvc.yaml` files | Several `pipeline.py` files; one imports another's handles |
| `dvc.lock` | Per-stage lock files in `.pivot/stages/` |
| `dvc remote add` / `dvc push` / `dvc pull` | `pivot config set remotes.<name>` / `pivot push` / `pivot pull` |
| `dvc repro`, `dvc status`, `dvc dag` | `pivot repro`, `pivot status`, `pivot dag` |
| `dvc metrics show`, `dvc plots show`, `dvc params diff` | `pivot metrics show`, `pivot plots show`, `pivot params diff` |

## A worked example

### Before: DVC

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw/runs.csv
      - src/prepare.py
    params:
      - prepare.min_score
    outs:
      - data/prepared.csv
  summarize:
    foreach: [headline, all]
    do:
      cmd: python src/summarize.py ${item}
      deps:
        - data/prepared.csv
        - src/summarize.py
      params:
        - summarize
      metrics:
        - metrics/${item}.json:
            cache: false
```

```python
# src/summarize.py
import json
import sys

import dvc.api
import pandas as pd

subset = sys.argv[1]
params = dvc.api.params_show()["summarize"]
df = pd.read_csv("data/prepared.csv")
if subset == "headline":
    df = df[df["agent"].isin(params["headline_agents"])]
json.dump({"runs": len(df), "mean": df["score"].mean()}, open(f"metrics/{subset}.json", "w"))
```

### After: Pivot

```python
# pipeline.py
from typing import Annotated

import pandas as pd

from pivot import Pipeline, StageParams, metric, stage


class PrepareParams(StageParams):
    min_score: float = 0.0


@stage
def prepare(runs: pd.DataFrame, params: PrepareParams) -> pd.DataFrame:
    return runs[runs["score"] >= params.min_score]


class SummarizeParams(StageParams):
    agents: list[str] | None = None


@stage
def summarize(runs: pd.DataFrame, params: SummarizeParams) -> Annotated[dict, metric]:
    if params.agents is not None:
        runs = runs[runs["agent"].isin(params.agents)]
    return {"runs": len(runs), "mean": float(runs["score"].mean())}


with Pipeline("report") as pipeline:
    runs = pipeline.input("runs.csv")
    prepared = prepare(runs, params=PrepareParams(min_score=0.1))
    subsets = {"headline": ["gpt", "claude"], "all": None}
    for name, agents in subsets.items():
        with pipeline.variant(name):
            summarize(prepared, params=SummarizeParams(agents=agents))
```

```bash
mkdir -p data/raw
printf 'agent,score\ngpt,0.5\nclaude,0.7\nother,0.05\nother,0.3\n' > data/raw/runs.csv
pivot repro
pivot metrics show
```

What changed:

- The script's top-level code became a function. `sys.argv` became a
  `StageParams` field, and the `foreach` became a loop with
  `pipeline.variant()`, giving stages `report/summarize@headline` and
  `report/summarize@all`.
- `pd.read_csv`, `json.dump` and every path are gone. The stage receives a
  DataFrame and returns a dict; Pivot writes `metrics/report/summarize@all.yaml`.
- `dvc.api.params_show()` and the shared `params.yaml` section became one
  `StageParams` class per stage, defined beside it. The variant-specific values
  are passed at the call site.
- `src/*.py` no longer appears as a dependency: editing `summarize` or a
  helper it calls re-runs it.

## Porting steps

1. **Inputs.** Move every file that no stage produces under `data/raw/` and
   declare it with `pipeline.input("<file name>")`. Files from other projects
   go under `data/external/` (`pipeline.input(name, external=True)`).
2. **Stages.** For each DVC stage, write a `@stage` function: one parameter per
   data dependency, return the outputs (a `TypedDict` for several). Delete the
   file reads and writes from the body.
3. **Parameters.** For each stage, define a `StageParams` subclass beside it
   with the values that stage reads from `params.yaml`, with the current
   values as defaults or passed at the call site. YAML anchors become ordinary
   Python values.
4. **Wiring.** In `pipeline.py`, call the stages inside
   `with Pipeline("<name>") as pipeline:`, passing handles where DVC had
   matching `outs`/`deps` paths. Replace `foreach`/`matrix` with loops.
5. **Check.** `pivot list` shows every expected stage; `pivot repro` runs them;
   compare the outputs with the DVC outputs.
6. **Remove DVC.** Delete `dvc.yaml`, `dvc.lock`, `.dvc/`, `.dvcignore`,
   the `params.yaml` sections the stages no longer read, and the `dvc`
   dependency. Configure the remote:

```bash
pivot config set remotes.origin s3://mybucket/cache
pivot config set default_remote origin
pivot push
```

## FAQs

### Can I keep `params.yaml`?

Only as an override file. Pivot reads `params.yaml` at the project root keyed by
qualified stage name (`report/prepare: {min_score: 0.2}`) and applies it on top
of the values in code; stages never read it themselves. The values belong in
`StageParams`.

### What about my existing DVC cache?

Pivot has its own cache. Run `pivot repro` once to populate it; the DVC cache
is not used.

### Where do module-level config dicts go?

Into `StageParams`. A stage may read a module-level `bool`, `int`, `float`,
`str`, `bytes` or `None`; reading any other module-level value (a dict of
settings, a list of agents, a tuple of columns) is an error.
