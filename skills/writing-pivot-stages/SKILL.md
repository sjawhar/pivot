---
name: writing-pivot-stages
description: Use when writing, editing, reviewing, or debugging a Pivot pipeline (pipeline.py, @stage functions, Pipeline, pipeline.input, StageParams, params.yaml overrides), porting a DVC pipeline (dvc.yaml, dvc.api.params_show) to Pivot, or seeing Pivot errors such as "module 'pivot' has no attribute", "received handle from multi-output stage", "has unsupported type", "Cannot infer serialization format", "captures non-scalar variable", "does not define a 'pipeline' variable", or "No module named" when loading pipeline.py
---

# Writing Pivot Stages

## Overview

A Pivot stage is a pure Python function decorated with `@stage`. Its parameters are data, its return value is data, and its configuration is one `StageParams` object. A pipeline is a `with Pipeline("name") as pipeline:` block that calls stages and passes the returned handles to other stages. The DAG comes from those calls. Pivot chooses every file path, loads every input, and writes every output.

**Core principle:** pipeline code contains no file paths, no file I/O, and no configuration outside `StageParams`.

## A complete pipeline

Project layout:

```
my_project/            # a git repo; run `pivot init` here once
  pipeline.py
  data/raw/scores.csv  # columns: name,score
```

```python
# pipeline.py
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
    return {"rows": kept, "summary": {"kept": len(kept), "dropped": len(raw) - len(kept)}}


class ScaleParams(StageParams):
    factor: int = 1


@stage
def scale(rows: pd.DataFrame, params: ScaleParams) -> Annotated[pd.DataFrame, loaders.CSV()]:
    out = rows.copy()
    out["score"] = out["score"] * params.factor
    return out


@stage
def combine(tables: list[pd.DataFrame]) -> Annotated[dict, metric]:
    merged = pd.concat(tables, ignore_index=True)
    return {"rows": len(merged), "max_score": float(merged["score"].max())}


with Pipeline("scores") as pipeline:
    raw = pipeline.input("scores.csv")
    cleaned = clean(raw, params=CleanParams(min_score=0.5))
    scaled = []
    for factor in [2, 10]:
        with pipeline.variant(f"x{factor}"):
            scaled.append(scale(cleaned.rows, params=ScaleParams(factor=factor)))
    combine(scaled)
```

```bash
git init && pivot init     # once per project
pivot list                 # scores/clean, scores/scale@x2, scores/scale@x10, scores/combine
pivot repro                # runs all four stages in dependency order
pivot repro                # everything cached
```

Outputs: `data/scores/clean/rows.jsonl`, `metrics/scores/clean/summary.yaml`, `data/scores/scale@x2.csv`, `data/scores/scale@x10.csv`, `metrics/scores/combine.yaml`.

## Stages

- Decorate with `@stage` and annotate the return type (required).
- Every parameter receives one of:
  - an **artifact handle** (from `pipeline.input()` or another stage call); the function receives the loaded value,
  - a **list or tuple of handles**; the function receives a list or tuple of loaded values (`tables: list[pd.DataFrame]`),
  - a **`StageParams` instance**, at most one per stage; never `Optional[...]` or a union,
  - a **previous-output marker** from `pipeline.previous()` or `pipeline.previous("key")`; the function receives its own prior output tagged `incremental`.
- Plain values (`factor=3`, a string, a path) are rejected: `parameter 'factor' has unsupported type 'int'`. Put them in `StageParams`.
- `*args`, positional-only parameters (`/`) and `**kwargs` are rejected: workers call stages with named keyword arguments. Use a `list[...]` parameter for several artifact inputs and ordinary named parameters for all other inputs.

- Define stages, their `StageParams`, their return `TypedDict`s and custom loaders at module level. Workers receive the function by pickling. With postponed/string annotations (for example, `from __future__ import annotations`), return annotations are resolved in the module namespace, so a `TypedDict` defined inside a function fails with `name 'Result' is not defined`.
- Outside a `with Pipeline` block a stage is an ordinary function, so test it directly: `clean(df, CleanParams(min_score=0.9))`.

## Resources and concurrency

Use `resources=` on `@stage` when a stage needs a constrained scheduler resource:

```python
@stage(resources={"gpu": 1})
def train(rows: pd.DataFrame, params: TrainParams) -> dict:
    return {"rows": len(rows)}
```

Every scheduled stage holds one CPU unit. `@stage(resources={"cpu": "all"})` reserves every CPU unit, as many as the configured worker count (`core.max_workers`), and therefore runs alone. Other resource names default to capacity 1. Set capacities in `.pivot/config.yaml`:

```yaml
resources:
  gpu: 2
```

or for one run:

```bash
pivot repro --resources gpu=2
```

Resource names are case-insensitive. Resources control admission only: a waiting stage does not consume a worker, and changing a resource declaration does not invalidate cached outputs.

- `joblib.Parallel` defaults to threads inside a stage; for CPU-bound Python work, run `PIVOT_NESTED_PARALLELISM=processes pivot repro` and declare `@stage(resources={"cpu": "all"})` so nested process pools do not overlap.

## Cleanup when a run is stopped

`Ctrl+C` at the terminal, or the death of the `pivot` process, raises `pivot.executor.worker.WorkerTerminated` (a `KeyboardInterrupt`) inside every running stage. A stage that starts something outside itself (a detached container, a server, a child process it does not wait for) stops it in a `finally` block or a context manager:

```python
subprocess.run(["docker", "run", "--detach", "--rm", "--name", name, image, "sleep", "infinity"], check=True)
try:
    ...
finally:
    subprocess.run(["docker", "rm", "--force", name], check=False)
```

- The cleanup has 10 seconds; then the worker exits regardless and kills every process still descended from it. A detached process (double-forked, a daemon, or orphaned when its parent exited) and a container are not among those: stop them in the cleanup. Use `docker rm --force`, not `docker stop`, which waits 10 seconds by default.
- Do not catch `BaseException` or `KeyboardInterrupt` to carry on: the stage fails anyway, and nothing it returns is committed.
- An RPC or TUI cancel is different: it lets running stages finish and skips the rest.

## Pipelines and inputs

- `pipeline.py` must bind the pipeline to a module variable named `pipeline`. Pivot finds `pipeline.py` in the current directory or the project root (the top-most directory containing `.pivot/`).
- `pipeline.input("scores.csv")` declares a raw input at `data/raw/scores.csv`. Options:
  - `external=True` reads `data/external/<name>` instead,
  - `path="some/file.csv"` sets the path explicitly (relative to the project root),
  - `t=pathlib.Path` passes the stage a `Path` instead of loading the file; `t=str` passes the file's text,
  - `format=loaders.CSV(sep=";")` sets the reader explicitly.
- Input format from the extension: `.csv` CSV (DataFrame), `.jsonl` DataFrame, `.json` JSON, `.yaml`/`.yml` YAML, `.txt`/`.sql`/`.jinja` text, `.pkl` pickle, anything else a `pathlib.Path`.
- Stage names are `<pipeline>/<function>`, and the CLI takes them qualified: `pivot run scores/clean`, `pivot repro scores/combine`.

## Outputs

The return type decides the format. Annotate only the **return type**; parameters are never annotated with loaders.

| Return type | Format | File |
|---|---|---|
| `pd.DataFrame` | JSON Lines | `.jsonl` |
| `dict`, `list` | YAML | `.yaml` |
| `str` | text | `.txt` |
| `pathlib.Path` | a file or directory created outside the project, moved by `PathOnly()` | `.dat` (or a directory) |
| `Annotated[T, loaders.CSV()]` (any loader instance) | that loader | its extension |
| `Annotated[dict, metric]` | YAML, under `metrics/` | `.yaml` |
| `Annotated[Figure, plot]` (matplotlib) | PNG, under `plots/` | `.png` |
| `Annotated[T, incremental]` | the type's `Loader`, restored before the stage runs | its extension |
| `Annotated[dict[str, T], directory]` | a directory, one file per key, each written as a `T` output would be (or by `Annotated[..., directory, loaders.X()]`) | none: `data/<pipeline>/<stage>/` |
| a `TypedDict` | one output per field, each field typed as above | one file per field |

Any other type fails with `Cannot infer serialization format for type int`; wrap it (`-> dict`) or annotate a loader (`Annotated[int, loaders.JSON()]`).

- Import tags from `pivot`: `from pivot import directory, incremental, metric, plot`.
- Paths are chosen by Pivot: `data/<pipeline>/<stage>.<ext>`, or `data/<pipeline>/<stage>/<field>.<ext>` for `TypedDict` fields; `metrics/` and `plots/` replace `data/` for tagged outputs. Do not write directly to those managed paths. Two outputs may not resolve to the same path, or one inside another's directory (pipeline `demo` stage `x` field `y` and pipeline `demo/x` stage `y` both map to `data/demo/x/y.jsonl`), and a `pipeline.input()` may not be bound to an output's path (pipeline `raw`'s stage `s` writes `data/raw/s.jsonl`, the default path of `pipeline.input("s.jsonl")`); pass the stage's handle instead. Loading such a pipeline fails with `Output path conflict` naming both.
- An incremental output receives the loader's `empty()` value on the first run and a writable copy of its previous cached value on later runs. Bind it with `pipeline.previous()` for a single output or `pipeline.previous("key")` for a tagged `TypedDict` field. The marker creates no DAG edge, can bind one parameter only, and requires a full `Loader` with `load()`, `save()`, and `empty()`.
- For a `pathlib.Path` return, create a file or directory outside the project (for example in a temporary directory) and return that path. Pivot moves it to the managed output path; returning a path already inside the project is rejected because Pivot does not take ownership of user files.
- A `directory` output returns a dict from each file's path inside the directory to its value: `return {"train.jsonl": df_a, "test.jsonl": df_b}`. The loader comes from `T` or the explicit loader, never from a key's extension, so name keys to match it. Keys are relative paths with an extension (subdirectories allowed); an empty dict, `..`, an absolute path, a backslash, colon or null byte, two keys that normalize to one path, two keys that differ only by case, and a key inside another key's path (`a.json` and `a.json/b.json`) fail the stage before any output is written, so the previous run's outputs stay. The directory holds exactly the returned files each run, and is cached, restored, pushed and pulled as one artifact. `directory` cannot be combined with `metric`, `plot` or `incremental`.
- A stage takes a directory output whole or one file of it. Passing the handle (`count(parts)`) gives the stage a `dict[str, T]` read with the producer's loader, and the stage reruns when any file changes. Indexing the handle by a file's path (`fit(parts["train.jsonl"])`, or `built.parts["train.jsonl"]` for a `TypedDict` field) gives the stage that one file's value (its path, for `pathlib.Path` files), and the stage reruns only when that file changes. A handle is not a collection: do not iterate it or test membership (`for part in parts`, `"train.jsonl" in parts`); name a file or pass the handle itself instead. The path is checked when the pipeline is built by the rules for keys; a run where the producer did not write it fails with `missing deps:` naming the producer and the file, and a path naming a directory inside the output fails with `unreadable deps:`. A `TypedDict` field named like a file of a directory field is refused.
- A stage cannot take an output whose loader only writes (a `plot`, or anything saved with `MatplotlibFigure()`), or one file of a directory of them; the pipeline fails to build, naming both stages and the loader.
- A multi-output handle must be narrowed before use: `cleaned.rows` or `cleaned["rows"]`. Passing `cleaned` itself fails with `received handle from multi-output stage ... without selecting an output`.
- A `dict` or `list` output is written as YAML: numpy scalars and arrays are converted to plain Python numbers and lists. A DataFrame, a dataclass, or any other object inside it is written in a form Pivot cannot read back, with a warning under the stage in the `pivot repro` report; convert it first (`df.to_dict()`, `dataclasses.asdict(x)`). A later release will fail the stage instead.
- Loaders: `pivot.loaders.CSV`, `JSON`, `YAML`, `Text`, `JSONL`, `DataFrameJSONL`, `Pickle`, `MatplotlibFigure` (`pip install "pivot[matplotlib]"`). A custom loader is a module-level `@dataclasses.dataclass(frozen=True)` subclass of `loaders.Writer[T]` (`save`) or `loaders.Loader[T]` (`load`, `save`, and `empty`).

## Configuration: one `StageParams` per stage

Each stage gets its own `StageParams` subclass, defined directly above the stage that uses it. Fields are typed, with defaults; nested Pydantic models, lists and dicts are fine (`exclude: list[str] = []`, no `default_factory`). Unknown fields are refused, in code and in `params.yaml`; in `params.yaml` a typo inside a nested model is refused too, unless that model sets `extra="allow"`. Pass the instance at the call site: `scale(rows, params=ScaleParams(factor=2))`. A stage whose params all have defaults may declare `params: CleanParams = CleanParams()`.

Changing a param value re-runs that stage. `params.yaml` at the project root overrides values without editing code; keys are qualified stage names, and a key without `@...` applies to every variant:

```yaml
scores/scale:
  factor: 3
scores/scale@x10:
  factor: 100
```

`pivot params show` prints the effective values. A key with no pipeline part, or one that names no stage of a loaded pipeline, is refused with the names it could mean; a field the class does not have is refused with the fields it could mean, and a value the field rejects is refused naming the field and value. In watch mode these errors make the pipeline invalid until `params.yaml` is fixed.

## Module-level constants

A stage is a pure function of its inputs and its `StageParams`: every setting it uses is a `StageParams` field, never a module-level value. Pivot fingerprints a module-level scalar, tuple or frozenset of scalars, or compiled regex that a stage reads, so editing one re-runs the stage. It refuses any other module-level value (list, dict, set, a tuple holding a tuple, list or object, enum member, class instance, or a method bound to an instance such as `np.random.default_rng(SEED).normal`), read directly, as `module.NAME`, or as a default argument (`cols=COLUMNS`), with an error that names the variable. Neither is how a pipeline carries settings.


## Helpers that do not affect output

A helper that cannot change a stage's output — process bookkeeping, file
locks, log rotation — can carry `@pivot.fingerprint_by_name` so Pivot
records it by its module and qualified name only:

```python
from pivot import fingerprint_by_name


@fingerprint_by_name
def process_start():
    """Read /proc for a running process."""
    ...
```

A body change to a marked helper does not rerun calling stages. A move to
another module, a rename, or removing the mark reruns them once. The
decorator accepts module-level functions and `functools.wraps` /
`contextlib.contextmanager` wrappers. It refuses a method, a lambda, a
helper that imports a user module in its body, and a helper that uses
`importlib.import_module` or `__import__`.

A `functools.cache` or `functools.lru_cache` decorator above the mark keeps
the record by name. A decorator with a body of its own above the mark adds
that body to the fingerprint. The marked function below it stays a name.

Pivot does not walk code that only a marked helper calls. If that code
imports a user module in its body, the run-time check counts the import,
and the stage warns and reruns each run. Move that import to the top of its
module to clear it.

Soundness is the user's choice: if a marked helper can change a stage's
output, Pivot does not see the change.

## Variants and fan-out

Variants are Python loops. Calling the same stage again in one pipeline creates `name@1`, `name@2`, and so on. For readable names, wrap the call in `with pipeline.variant("x2"):` to get `scale@x2`; nested variant blocks give `name@outer@inner`. Variant names use letters, digits, `_`, `.` and `-` only. Fan in by passing a list of handles to one stage.

## Several modules and pipelines

Imports resolve from the project root (the directory holding `.pivot/`): a root `pipeline.py` imports sibling modules directly (`from stages import clean`); a pipeline in a subdirectory imports with the full path (`from report.helpers import x`, `from prep.pipeline import kept`). Larger projects keep one module per stage, each holding its own `StageParams` (`stages/clean.py` with `CleanParams` and `clean`), and `pipeline.py` only imports and wires them.

A handle is an ordinary Python object, so a downstream pipeline imports it and Pivot runs the upstream stages first:

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

From `report/`, `pivot repro` runs `prep/drop_low` then `report/count`. From the project root, `pivot repro --all` runs every `pipeline.py` under it. Pipeline names must be unique within a project: a name is the directory of its locks and outputs (`.pivot/stages/<pipeline>/`, `data/<pipeline>/`), so `--all` refuses two `pipeline.py` files with the same `Pipeline("name")`. To make one root pipeline, `include()` the others (inside the `with` block or after it):

```python
# pipeline.py
from pivot import Pipeline
from prep.pipeline import pipeline as prep
from report.pipeline import pipeline as report

pipeline = Pipeline("all")
pipeline.include(prep)
pipeline.include(report)
```

## Anti-patterns

| Don't | Do |
|---|---|
| One shared `params.py` (or config module) holding every stage's settings | One `StageParams` subclass beside each stage |
| A half-migration from DVC: `dvc.api.params_show()`, reading `params.yaml` in a stage, `dvc.yaml` or a YAML stage list next to `pipeline.py`, path strings passed between stages, `pd.read_csv`/`to_csv` inside stages | Port every stage to `@stage`, every input to `pipeline.input()`, every edge to a handle, every setting to `StageParams`; delete the DVC files |
| Mutable configuration in module-level constants (dicts of settings, lists of variants) | `StageParams` fields; loop over values inside the `with` block |
| Loader annotations on parameters, `Dep(...)`/`Out(...)` markers, path overrides, a `register` call | The function signature above: handles in, data out |

## Running

```bash
pivot repro                 # whole pipeline
pivot repro scores/combine  # one stage and its upstream
pivot run scores/clean      # one stage only
pivot status --explain      # why each stage would run
pivot repro --dry-run       # validate the DAG without running
```

Commit `pipeline.py`, `data/raw/` inputs you own, and `.pivot/stages/` (lock files). `pivot init` already ignores `.pivot/cache/` and `.pivot/state.lmdb/`.

## Common errors

| Error | Fix |
|---|---|
| `cannot import name 'Out' from 'pivot'` or `module 'pivot' has no attribute 'Out'` (or `Dep`, `Metric`) | Those markers do not exist; use the return type and `metric`/`plot` tags |
| `does not define a 'pipeline' variable` | `with Pipeline("name") as pipeline:` |
| `No module named '<yours>'` | Import from the project root: `from report.helpers import x`, not `from helpers import x`, in `report/pipeline.py` |
| `parameter 'x' has unsupported type` | Pass handles or `StageParams`; move plain values into `StageParams` |
| `received handle from multi-output stage` | Select a field: `handle.field` |
| `Cannot infer serialization format` | Return a supported type or `Annotated[T, loaders.X()]` |
| `captures non-scalar variable 'X'` | Move `X` into `StageParams` |
| `depends on input '<name>' (path data/raw/<name>), which does not exist` | Put the file at `data/raw/<name>` or pass `path=` |
| `must have a return type annotation` | Annotate the return; `-> None` stages have no outputs |
| `Invalid variant name 'big one': use only letters, digits and the characters _ . -` | Rename the variant; `/`, `@` and spaces are not allowed |
| `is tagged directory but its type is <type>` | A `directory` output's type is `dict[str, T]`, from file path to value |
| `whose loader <loader> cannot read it back` | Give that output a full `Loader`, or stop passing its handle to a stage |
| `depends on file 'x' of stage '<stage>': key must include file extension` | Index a directory handle by the file's path as the producer writes it, extension included: `parts["train.jsonl"]` |
| `Input binding conflict: inputs 'a' and 'b' are both bound to '<path>'` | Use one `pipeline.input()` name per file |
| `Output path conflict: 'p/x/y' and 'p/x:y' both write data/p/x/y.jsonl` | Rename a stage, pipeline or `TypedDict` field so each output has its own path |
| `wrote its YAML output in a form Pivot cannot read back` (a warning) | Convert the value before returning it: `df.to_dict()`, `dataclasses.asdict(x)`, or pick a format with `Annotated[T, loaders.X()]` |

## Checklist

- [ ] Every stage is a module-level `@stage` function with a return annotation
- [ ] Each stage has its own `StageParams` class beside it (or none)
- [ ] No file paths, `open()`, `read_csv`, or `to_csv` in stages; no `dvc` imports
- [ ] Inputs come from `pipeline.input()`; edges are handles
- [ ] `Annotated[...]` appears only on return types and `TypedDict` fields
- [ ] Stages read no module-level configuration; every setting is a `StageParams` field
- [ ] `pivot list` shows the expected stages, then `pivot repro` passes twice (second run: all cached)
