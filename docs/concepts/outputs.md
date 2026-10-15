# Outputs

A stage's outputs are its return value. Pivot saves whatever the stage returns,
chooses the file path, and makes the saved artifact available to downstream
stages through the handle the stage call returned.

## The return type decides the format

The return annotation is required (`-> None` means the stage has no outputs).
Pivot picks a writer from it:

| Return type | Saved as | Extension |
|-------------|----------|-----------|
| `pandas.DataFrame` | JSON Lines (`DataFrameJSONL()`) | `.jsonl` |
| `dict`, `list` (including `dict[str, int]` etc.) | YAML (`YAML()`) | `.yaml` |
| `str` | text (`Text()`) | `.txt` |
| `pathlib.Path` | a file or directory created outside the project, moved by `PathOnly()` | `.dat` (or a directory) |
| `Annotated[T, <loader>]` | that loader | the loader's extension |
| `Annotated[dict, metric]` | YAML, stored under `metrics/` | `.yaml` |
| `Annotated[Figure, plot]` | PNG via `MatplotlibFigure()`, stored under `plots/` | `.png` |
| `Annotated[T, incremental]` | the type's default `Loader`, restored before the stage runs | the loader's extension |
| a `TypedDict` | one output per field, each field typed as above | per field |

Any other type fails when the pipeline is built:
`Cannot infer serialization format for type int. Use Annotated[T, format] on
the return type to specify explicitly.`

`Annotated` appears only on return types and `TypedDict` fields. Stage
parameters are never annotated with loaders; inputs take their format from
the producing stage or from `pipeline.input()`.

## Where outputs are stored

| Output | Path |
|--------|------|
| Single output of stage `s` in pipeline `p` | `data/p/s.<ext>` |
| Field `f` of a `TypedDict` output | `data/p/s/f.<ext>` |
| Tagged `metric` | `metrics/p/...` instead of `data/p/...` |
| Tagged `plot` | `plots/p/...` instead of `data/p/...` |

Extensions come from the writer: `jsonl`, `csv`, `yaml`, `json`, `txt`, `pkl`,
`png`, and `dat` for custom writers. Variant stages keep their `@` suffix in the
file name (`data/p/scale@x2.csv`).

For a `pathlib.Path` return, the stage must create a file or directory outside
the project and return its path. Pivot moves it to the managed output path. A
returned path inside the project is rejected because Pivot must not take
ownership of a file it did not create.

## Example

```python
# pipeline.py
from typing import Annotated, TypedDict

import matplotlib.figure
import pandas as pd

from pivot import Pipeline, loaders, metric, plot, stage


class SplitResult(TypedDict):
    train: pd.DataFrame
    test: Annotated[pd.DataFrame, loaders.CSV()]
    sizes: Annotated[dict, metric]


@stage
def split(raw: pd.DataFrame) -> SplitResult:
    cut = len(raw) // 2
    return {
        "train": raw.iloc[:cut],
        "test": raw.iloc[cut:],
        "sizes": {"train": cut, "test": len(raw) - cut},
    }


@stage
def chart(train: pd.DataFrame) -> Annotated[matplotlib.figure.Figure, plot]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(train["id"], train["value"])
    return fig


@stage
def report(test: pd.DataFrame) -> str:
    return f"{len(test)} test rows\n"


with Pipeline("out") as pipeline:
    parts = split(pipeline.input("values.csv"))
    chart(parts.train)
    report(parts["test"])
```

After `pivot repro`:

```
data/out/split/train.jsonl
data/out/split/test.csv
metrics/out/split/sizes.yaml
plots/out/chart.png
data/out/report.txt
```

The plot needs matplotlib: `pip install "pivot[matplotlib]"`.

## Multi-output stages

A `TypedDict` return type declares several outputs. The stage returns a dict
with exactly those keys (`SplitResult(train=..., ...)` works too). Downstream,
select one field from the handle with `parts.train` or `parts["train"]`; each
field is a separate artifact with its own hash, so a change that only affects
`sizes` does not re-run stages that consume `train`.

Define the `TypedDict` at module level. The field name `_single` is reserved.

## Directory artifacts

A stage can return a `pathlib.Path` to a directory it created outside the
project. `PathOnly()` moves that directory to Pivot's managed output path and
Pivot caches every regular-file descendant as part of the artifact. Names that
project input tracking ignores, such as `build/` and `.venv/`, are still part of
the directory artifact. Pivot accepts a descendant symlink only when it is a
cache-derived checkout link; any other directory-artifact symlink is refused.

A directory artifact, input or output, is hashed and cached whole. Every file
under it counts, including names that code fingerprinting or watch mode may
ignore, such as `__pycache__/`, `build/`, `.git/` or `*.pyc`. Editing any of
them changes the directory hash, and all of them return on restore.

The declarative `DirectoryOut` annotation for a variable set of files remains
undecided; do not add that annotation to a compositional pipeline.

## Incremental outputs

Tag a return value with `incremental` when the stage needs its own prior output.
Bind the matching parameter to `pipeline.previous()`, or use
`pipeline.previous("key")` for a field of a `TypedDict` result. The first run
receives the loader's `empty()` value; later runs restore the cached output as a
writable copy before calling the stage. Previous-output parameters do not create
DAG edges.

```python
from typing import Annotated

from pivot import Pipeline, incremental, stage


@stage
def update_cache(previous: dict[str, int]) -> Annotated[dict[str, int], incremental]:
    return {"count": previous.get("count", 0) + 1}


with Pipeline("state") as pipeline:
    update_cache(previous=pipeline.previous())
```

An incremental output requires a full `Loader` with `load()`, `save()`, and a
concrete `empty()` implementation; a writer-only format such as
`MatplotlibFigure()` and loaders that inherit the default raising `empty()`
(including `PathOnly()` and `Pickle()`) are rejected. An existing incremental
output without a lock-file cache entry is refused unless
`pivot repro --allow-uncached-incremental` is explicit.

## Metrics and plots

`metric` and `plot` (imported from `pivot`) are tags, not paths:

- `Annotated[dict, metric]` saves YAML under `metrics/`. `pivot metrics show`
  and `pivot metrics diff` read every metric output. Metric files are not
  copied into the cache; commit them to git.
- `Annotated[Figure, plot]` saves a PNG under `plots/` and closes the figure.
  `pivot plots show` renders all plot outputs as an HTML gallery.

A tag can be combined with an explicit writer: `Annotated[pd.DataFrame, metric,
loaders.CSV()]` saves a CSV under `metrics/`.

A `dict` or `list` output is written as YAML. Numpy scalars and arrays are
converted to plain Python numbers and lists, so `{"mean": df["v"].mean()}`
works. A DataFrame, a dataclass, or any other object inside it is written with
Python-specific YAML tags that `pivot metrics show` and downstream stages cannot
read, and the `pivot repro` report prints a warning under the stage naming the
output, the type and where it sits. Convert it first (`df.to_dict()`,
`dataclasses.asdict(x)`); a later release will fail the stage instead.

## How Pivot writes outputs

After a stage returns, Pivot:

1. Checks that a `TypedDict` return value contains every declared key
2. Writes each value with its writer to the path above, creating directories
3. Hashes each file and copies it into `.pivot/cache/` (except metrics)
4. Records the hashes in the stage's lock file under `.pivot/stages/`

Pivot deletes the previous output before the stage runs, so a stage never sees
its own stale output.

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Dependencies](dependencies.md) | [Loaders](loaders.md)
