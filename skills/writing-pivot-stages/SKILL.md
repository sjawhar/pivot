---
name: writing-pivot-stages
description: Use when writing, reviewing or debugging a Pivot pipeline (pipeline.py, @stage, StageParams, params.yaml), porting a DVC pipeline to Pivot, or when Pivot fails to load or run one (an import error, an unsupported parameter type, an unknown serialization format, a captured variable).
---

# Writing Pivot stages

A stage is a pure, module-level function decorated with `@stage`: its parameters are data, its return value is data,
and its settings are one `StageParams` object. A pipeline is a `with Pipeline("name") as pipeline:` block that calls
stages; the DAG comes from those calls. Pivot chooses every path, loads every input and writes every output, so
pipeline code holds no file paths, no file I/O and no settings outside `StageParams`.

## Example

```python
# pipeline.py, in a git repo where `pivot init` ran once; input at data/raw/scores.csv
from typing import Annotated, TypedDict

import pandas as pd

from pivot import Pipeline, StageParams, loaders, metric, stage


class CleanParams(StageParams):
    min_score: float = 0.5


class CleanResult(TypedDict):
    rows: pd.DataFrame
    summary: Annotated[dict, metric]


@stage
def clean(raw: pd.DataFrame, params: CleanParams) -> CleanResult:
    kept = raw[raw["score"] >= params.min_score]
    return {"rows": kept, "summary": {"kept": len(kept)}}


class ScaleParams(StageParams):
    factor: int = 1


@stage
def scale(rows: pd.DataFrame, params: ScaleParams) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return rows.assign(score=rows["score"] * params.factor)


@stage
def combine(tables: list[pd.DataFrame]) -> Annotated[dict, metric]:
    return {"rows": len(pd.concat(tables))}


with Pipeline("scores") as pipeline:
    cleaned = clean(pipeline.input("scores.csv"), params=CleanParams(min_score=0.5))
    scaled = []
    for factor in [2, 10]:
        with pipeline.variant(f"x{factor}"):
            scaled.append(scale(cleaned.rows, params=ScaleParams(factor=factor)))
    combine(scaled)
```

`pivot list` shows `scores/clean`, `scores/scale@x2`, `scores/scale@x10` and `scores/combine`; a second
`pivot repro` runs nothing.

## Stages

- Decorate with `@stage` and annotate the return type. Loader and tag annotations (`Annotated[...]`) go only on the
  return type and `TypedDict` fields, never on a parameter; keep the `StageParams` parameter's annotation, which is
  how Pivot finds it.
- Each parameter receives a handle (from `pipeline.input()` or a stage call), a list or tuple of handles, at most one
  `StageParams` instance (never `Optional` or a union), or `pipeline.previous()` for the stage's own `incremental`
  output. A plain value (`factor=3`) is refused: put it in `StageParams`. `*args`, `/` and `**kwargs` are refused.
- Define stages, their `StageParams`, their return `TypedDict`s and custom loaders at module level: workers receive
  them by pickling.
- Outside a `with Pipeline` block a stage is an ordinary function, so test it directly.
- A multi-output handle must be narrowed before use: `cleaned.rows` or `cleaned["rows"]`.
- A stage that starts something outside itself (a detached container, a server) stops it in a `finally` block:
  `Ctrl+C` raises `WorkerTerminated` in the stage, and 10 seconds later the worker kills its own process tree, which
  holds no detached process or container. Use `docker rm --force`, not `docker stop`.
- `@stage(resources={"gpu": 1})` limits how many such stages run at once (capacities under `resources:` in
  `.pivot/config.yaml`); `{"cpu": "all"}` runs the stage alone. For a process pool inside a stage, declare
  `{"cpu": "all"}` and run `PIVOT_NESTED_PARALLELISM=processes pivot repro` (`joblib.Parallel` uses threads).

## Inputs and outputs

- `pipeline.input("scores.csv")` reads `data/raw/scores.csv`; `external=True` reads `data/external/`, `path=` sets
  the path, `t=pathlib.Path` or `t=str` passes the path or text, `format=` sets the loader. Otherwise the extension
  decides what the stage receives: `.csv` and `.jsonl` a DataFrame, `.json`, `.yaml` and `.yml` the parsed value,
  `.txt`, `.sql` and `.jinja` text, `.pkl` and `.pickle` the unpickled object, anything else a `pathlib.Path`.
- The return type decides the format: `pd.DataFrame` JSON Lines, `dict`/`list` YAML, `str` text, `pathlib.Path` a
  file or directory the stage made outside the project (Pivot moves it in), `Annotated[T, loaders.X()]` that loader,
  a `TypedDict` one output per field, and `-> None` no output. `metric`, `plot`, `incremental` and `directory` are
  tags imported from `pivot`. Pivot picks every output path (`data/<pipeline>/<stage>.<ext>`); do not write there.
- Before a stage returns a `directory` (`Annotated[dict[str, T], directory]`, one file per key) or an `incremental`
  value, or you write a custom loader, read https://sjawhar.github.io/pivot/concepts/outputs/. A directory handle is
  not a collection: pass it whole, or index one file by its path, extension included (`parts["train.jsonl"]`).
- An input that names another object by its content hash is read with `pivot.read_by_hash(h)` inside the stage;
  read "Objects named by hash" at https://sjawhar.github.io/pivot/concepts/dependencies/ first.

## Settings: one `StageParams` per stage

Each stage has its own `StageParams` subclass, defined beside it, with typed fields and defaults. Unknown fields are
refused, in code and in `params.yaml`. Changing a value reruns the stage. `params.yaml` at the project root
overrides values by qualified stage name (`scores/scale:` for every variant, `scores/scale@x10:` for one);
`pivot params show` prints the effective values.

## Module-level values

Every setting a stage uses is a `StageParams` field. Pivot fingerprints a module-level scalar, tuple or frozenset of
scalars, or compiled regex a stage reads, so editing one reruns it, and refuses any other module-level value (a list,
dict, set, enum member, instance or bound method), naming it. `@pivot.fingerprint_by_name` hides a helper's changes
from Pivot: read its section at https://sjawhar.github.io/pivot/concepts/fingerprinting/ before you use it.

## Variants and several pipelines

- Calling a stage again in one pipeline makes `name@1`, `name@2`; `with pipeline.variant("x2"):` names it
  `scale@x2`. Variant names use letters, digits, `_`, `.` and `-`. Fan in by passing a list of handles.
- When the set of items is known only after a stage runs, map over its `directory` output with `pipeline.map`; read
  https://sjawhar.github.io/pivot/concepts/outputs/#one-stage-per-file-pipelinemap first.
- Imports resolve from the project root (the directory holding `.pivot/`): `report/pipeline.py` imports
  `from report.helpers import x`. Another pipeline can import a handle (`from prep.pipeline import kept`), and
  `pivot repro --all` runs every `pipeline.py`: see https://sjawhar.github.io/pivot/guides/multi-pipeline/.

## Running

```bash
pivot repro                 # whole pipeline
pivot repro scores/combine  # one stage and its upstream
pivot run scores/clean      # one stage only
pivot status --explain      # why each stage would run
```

Commit `pipeline.py`, the `data/raw/` inputs you own and the lock directory (`.pivot/stages/`, or `stages/` under
`core.state_dir` when that is set). A tracked directory also needs its `<path>.pvt.listing` beside its `.pvt`.

When loading or running fails, or when porting from DVC, read `references/common-errors.md`.
