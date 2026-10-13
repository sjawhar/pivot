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
| `Annotated[T, <loader>]` | that loader | the loader's extension |
| `Annotated[dict, metric]` | YAML, stored under `metrics/` | `.yaml` |
| `Annotated[Figure, plot]` | PNG via `MatplotlibFigure()`, stored under `plots/` | `.png` |
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

## Not yet available

The compositional API has no way to declare a directory output (a variable set
of files decided at run time) or an incremental output (a stage that reads its
own previous result). Both designs are open; until they land, return a `dict`
or `TypedDict` instead.

## Metrics and plots

`metric` and `plot` (imported from `pivot`) are tags, not paths:

- `Annotated[dict, metric]` saves YAML under `metrics/`. `pivot metrics show`
  and `pivot metrics diff` read every metric output. Metric files are not
  copied into the cache; commit them to git.
- `Annotated[Figure, plot]` saves a PNG under `plots/` and closes the figure.
  `pivot plots show` renders all plot outputs as an HTML gallery.

A tag can be combined with an explicit writer: `Annotated[pd.DataFrame, metric,
loaders.CSV()]` saves a CSV under `metrics/`.

Convert numpy scalars to Python numbers (`float(df["v"].mean())`) before
returning them in a `dict` or `list`: the YAML writer serializes numpy types
with Python-specific tags that downstream stages cannot read back.

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
