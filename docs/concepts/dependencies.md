# Dependencies

A stage's dependencies are its function parameters. Pivot loads each one before
the stage runs, passes the loaded value in, and uses the dependency to build the
[DAG](artifacts-and-dag.md) and to decide when the stage must re-run.

A parameter can receive one of four values at the call site:

| Call-site value | The function receives | Creates a DAG edge |
|-----------------|-----------------------|--------------------|
| A handle (from `pipeline.input()` or another stage call) | The loaded artifact | Yes |
| A `list` or `tuple` of handles | A `list` or `tuple` of loaded artifacts | One per handle |
| A `StageParams` instance (at most one) | That instance, with `params.yaml` overrides applied | No (see [Parameters](parameters.md)) |
| `pipeline.previous()` or `pipeline.previous("key")` | The stage's prior tagged incremental output | No |

Anything else (an `int`, a string, a path) is rejected when the pipeline is
built: `parameter 'factor' has unsupported type 'int'. Use ArtifactHandle
(dependency), StageParams (config), or list/tuple of ArtifactHandle.` Put such
values in the stage's `StageParams`.

## Previous incremental outputs

`pipeline.previous()` is valid only as an argument to the stage call whose
return type has one output tagged `incremental`. For a `TypedDict` result, use
`pipeline.previous("key")` to select the tagged field. Each marker can bind one
parameter only; it is not a handle and does not add a self-edge to the DAG.

## Inputs: `pipeline.input()`

Files that no stage produces are declared once, by name:

```python
raw = pipeline.input("scores.csv")            # data/raw/scores.csv
```

| Argument | Effect |
|----------|--------|
| `name` | Input name; also the file name under `data/raw/` |
| `external=True` | Read `data/external/<name>` instead |
| `path="dir/file.csv"` | Read this path (relative to the project root) |
| `t=pathlib.Path` | Pass the stage a `pathlib.Path` instead of loading the file |
| `t=str` | Pass the stage the file's text |
| `format=loaders.CSV(sep=";")` | Load with this [reader](loaders.md) |

Without `t` or `format`, the extension picks the loader:

| Extension | Stage receives |
|-----------|----------------|
| `.csv` | `pandas.DataFrame` (`CSV()`) |
| `.jsonl` | `pandas.DataFrame` (`DataFrameJSONL()`) |
| `.json` | parsed JSON (`JSON()`) |
| `.yaml`, `.yml` | parsed YAML (`YAML()`) |
| `.txt`, `.sql`, `.jinja` | `str` (`Text()`) |
| `.pkl`, `.pickle` | unpickled object (`Pickle()`) |
| anything else | `pathlib.Path` (`PathOnly()`) |

An input whose file does not exist fails before anything runs:
`Stage 'demo/keep' depends on 'missing.csv' which is not produced by any stage
and does not exist on disk`.

## Example

```python
# pipeline.py
import pathlib

import pandas as pd

from pivot import Pipeline, StageParams, loaders, stage


class ShiftParams(StageParams):
    offset: int = 0


@stage
def shift(raw: pd.DataFrame, params: ShiftParams) -> pd.DataFrame:
    return raw.assign(value=raw["value"] + params.offset)


@stage
def stack(tables: tuple[pd.DataFrame, pd.DataFrame], lookup: dict) -> pd.DataFrame:
    left, right = tables
    combined = pd.concat([left, right], ignore_index=True)
    return combined.assign(value=combined["value"] + lookup["bonus"])


@stage
def describe(query: str, notes: pathlib.Path, rows: pd.DataFrame) -> dict:
    return {"query": query.strip(), "notes_bytes": notes.stat().st_size, "rows": len(rows)}


with Pipeline("deps") as pipeline:
    raw = pipeline.input("values.csv")
    lookup = pipeline.input("lookup.yaml", external=True)
    query = pipeline.input("query.sql")
    notes = pipeline.input("notes.md", t=pathlib.Path)
    semi = pipeline.input("semi.csv", format=loaders.CSV(sep=";"))

    low = shift(raw, params=ShiftParams(offset=1))
    high = shift(semi, params=ShiftParams(offset=100))
    stacked = stack((low, high), lookup)
    describe(query, notes, stacked)
```

with these files:

```bash
mkdir -p data/raw data/external
printf 'id,value\n1,10\n2,20\n' > data/raw/values.csv
printf 'id;value\n3;30\n' > data/raw/semi.csv
printf 'bonus: 5\n' > data/external/lookup.yaml
printf 'select 1\n' > data/raw/query.sql
printf '# notes\n' > data/raw/notes.md
```

The two calls to `shift` become the stages `deps/shift` and `deps/shift@1`
(see [variants](pipelines.md#variants)). `stack` receives a tuple of two
DataFrames because it was called with a tuple of two handles.

## Outputs of other stages

A stage call returns a handle to the stage's output. For a stage that returns a
`TypedDict` (several outputs), select one field before passing it on, with
attribute or item syntax:

```python
split = split_rows(raw)        # returns a TypedDict with fields "train" and "test"
fit(split.train)               # or split["train"]
```

Passing the whole handle fails: `Stage 'demo/fit' received handle from
multi-output stage 'demo/split_rows' without selecting an output. Use
handle.key or handle['key']. Available: ['train', 'test']`.

## Testing is natural

Outside a `with Pipeline` block a `@stage` function is an ordinary function, so
tests pass data directly:

```python
def test_shift() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    assert shift(df, ShiftParams(offset=1))["value"].tolist() == [2, 3]
```

## Files tracked outside the pipeline

`pivot track` caches files that are not stage outputs (large raw data, for
example) and writes a `.pvt` sidecar to commit in their place:

```bash
pivot track data/raw/values.csv
```

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Outputs](outputs.md) | [Loaders](loaders.md)
