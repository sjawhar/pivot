# Loaders

Loaders read and write artifact files. Pivot picks one for every
[input](dependencies.md) and [output](outputs.md) automatically; you name a
loader only to change the default, by putting an instance on a stage's
**return type** (`Annotated[pd.DataFrame, loaders.CSV()]`) or passing it to
`pipeline.input(..., format=...)`.

## The hierarchy

```
        Reader[R]            Writer[W]
        (abstract)           (abstract)
          │  load()            │  save()
          │                    │
          └───────┬────────────┘
                  │
              Loader[W, R]
              (abstract)
              load() + save() + empty()
```

All three are frozen dataclasses with abstract methods. They are:

- **Immutable**: fields are frozen, safe to share across processes
- **Picklable**: sent to worker processes with the stage
- **Fingerprinted**: their code and field values are hashed, so changing a
  loader re-runs the stages that use it

| Base class | Methods | Used for |
|------------|---------|----------|
| `Reader[R]` | `load(path) -> R` | `pipeline.input(..., format=...)` |
| `Writer[W]` | `save(data, path)` | Stage outputs that are not read back |
| `Loader[W, R]` | `load() + save() + empty()` | Dependencies and incremental outputs |

An output consumed by another stage needs a `Loader`: Pivot writes it with
`save()` and reads it back with `load()` for the consumer. An incremental output
also needs `empty()` for its first run, then uses `load()` after restoring its
previous cached value. A `Writer` alone is enough for leaf outputs such as
plots. `Loader[T]` (one type parameter) means `Loader[T, T]`.

## Built-in loaders

All built-in loaders live in `pivot.loaders`:

| Loader | Base | Type | Options | Extension |
|--------|------|------|---------|-----------|
| `CSV()` | `Loader` | `DataFrame` | `index_col`, `sep`, `dtype` | `.csv` |
| `JSON()` | `Loader` | any JSON value | `indent`, `empty_factory` | `.json` |
| `YAML()` | `Loader` | any YAML value | `empty_factory` | `.yaml` |
| `Text()` | `Loader` | `str` | — | `.txt` |
| `JSONL()` | `Loader` | `list[dict]` | — | `.dat` (no fixed extension) |
| `DataFrameJSONL()` | `Loader` | `DataFrame` | — | `.jsonl` |
| `Pickle()` | `Loader` | any picklable object | `protocol` | `.pkl` |
| `PathOnly()` | `Loader` | `pathlib.Path` | — | `.dat` (or a directory) |
| `MatplotlibFigure()` | `Writer` only | `Figure` | `dpi`, `bbox_inches`, `transparent` | `.png` |

Defaults Pivot chooses without an annotation: `DataFrame` → `DataFrameJSONL()`,
`dict`/`list` → `YAML()`, `str` → `Text()`, `plot` tag → `MatplotlibFigure()`,
`metric` tag → `YAML()`.

### DataFrameJSONL metadata header

A `DataFrameJSONL()` output starts with one metadata record, then ordinary row
records:

```json
{"__pivot_dataframe_metadata__":{"version":1,"columns":[{"name":"name","dtype":"object"},{"name":"created_at","dtype":"datetime64[ns, UTC]"}],"temporal_columns":{"created_at":{"kind":"datetime","timezone":"UTC"}}}}
{"name":"alice","created_at":"2026-10-13T12:00:00.000000000Z"}
```

`columns` lists every column's name and dtype in order, so a zero-row frame
keeps its columns and dtypes. An integer column label, such as the `0` and `1`
of a frame built from a numpy array, is written as its JSON key with
`"label":"int"` in its entry and comes back as an integer, so `m[0]` still works
in the next stage. Pivot uses `temporal_columns` to restore only the
columns it marked as temporal, retain the original timezone names, and preserve
all-null `datetime` and `timedelta` dtypes. It never guesses that an ISO-shaped
string column is temporal.

Non-Pivot JSONL readers see the explicit first record and must skip it before
reading data rows. Pivot treats the first line as the header only when it has
exactly this shape, so a column may have any name, including
`__pivot_dataframe_metadata__`.

A `.jsonl` file without the header, such as a raw input under `data/raw/` or a
file written by an earlier Pivot release, loads exactly as
`pd.read_json(path, lines=True, orient="records", convert_dates=False)` reads
it, so a stage gets the same values from an external file that it always did.
That includes pandas' inexact float parsing (`0.3` reads as
`0.30000000000000004`). Exact floats and restored dtypes apply to files Pivot
writes with the header.

Some frames cannot be written exactly. In that case Pivot writes the whole frame
in its previous headerless pandas form, with dates as epoch milliseconds, and
prints a warning under the stage naming the column. The cases are:

- `Decimal`, `set`, `bytes` or other object cells
- a timezone that cannot be rebuilt from its name, such as a `dateutil` offset
- datetimes with mixed UTC offsets in one column
- a date outside the nanosecond range (1677 to 2262), such as a `9999-12-31` sentinel
- a column label that is neither str nor int, or an int and a str label with the same text

Convert the column, or choose another format such as
`Annotated[pd.DataFrame, loaders.Pickle()]`; a later release will fail the stage
instead.

### Pickle is a trusted-code boundary

Loading a pickle can execute arbitrary code. Only load Pickle artifacts written
by code you trust. A cache or remote that can supply pickle bytes is a
trusted-code boundary: content hashes identify bytes, but they do not make
those bytes safe. `Pickle()` emits a `RuntimeWarning` when it loads as a
reminder; it is not a security control.

`PathOnly()` hands a stage an input path without loading content. For an output,
it takes ownership of a file or directory created outside the project and moves
it to Pivot's managed path; a path already inside the project is rejected. Use
it for a binary format or directory artifact that a stage creates itself, not
for a file under `data/<pipeline>/`.

## Overriding the format

```python
from typing import Annotated

import pandas as pd

from pivot import loaders, stage


@stage
def to_csv(rows: pd.DataFrame) -> Annotated[pd.DataFrame, loaders.CSV()]:
    return rows


@stage
def compact(rows: pd.DataFrame) -> Annotated[dict, loaders.JSON(indent=None)]:
    return {"n": len(rows)}
```

and for inputs:

```python
raw = pipeline.input("semicolons.csv", format=loaders.CSV(sep=";"))
```

## Writing custom loaders

A custom loader is a module-level frozen dataclass that extends `Reader`,
`Writer`, or `Loader`. Fields become options, and their values are part of
the fingerprint.

```python
# pipeline.py
import dataclasses
import pathlib
from typing import Annotated, override

import pandas as pd

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class TSV(loaders.Loader[pd.DataFrame]):
    """Tab-separated values."""

    float_format: str | None = None

    @override
    def load(self, path: pathlib.Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t")

    @override
    def save(self, data: pd.DataFrame, path: pathlib.Path) -> None:
        data.to_csv(path, sep="\t", index=False, float_format=self.float_format)

    @override
    def empty(self) -> pd.DataFrame:
        return pd.DataFrame()


@stage
def convert(raw: pd.DataFrame) -> Annotated[pd.DataFrame, TSV(float_format="%.2f")]:
    return raw.assign(ratio=raw["value"] / raw["value"].sum())


@stage
def total(table: pd.DataFrame) -> dict:
    return {"ratio_sum": round(float(table["ratio"].sum()), 2)}


with Pipeline("custom") as pipeline:
    total(convert(pipeline.input("values.csv")))
```

`total` reads `convert`'s output back through `TSV.load()`. Pivot names
outputs of custom loaders `<stage>.dat` (`data/custom/convert.dat`).

### Requirements for custom loaders

1. **Frozen dataclass, defined at module level**: workers unpickle it by
   reference.
2. **Picklable fields**: no lambdas, open handles or connections.
3. **Deterministic `save()`**: the same data should produce the same bytes, or
   downstream stages re-run needlessly.
4. **`Loader` for anything read back**: `Writer` alone only for leaf outputs.

## Choosing the right base class

```
Is the file only ever an input (pipeline.input)?
  └─ Yes → Reader[R]

Is it an output nobody reads back (an image, a report)?
  └─ Yes → Writer[W]

Otherwise → Loader[T]   (or Loader[WriteType, ReadType] if they differ)
```

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Dependencies](dependencies.md) | [Outputs](outputs.md)
