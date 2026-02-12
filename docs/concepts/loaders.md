# Loaders

Loaders handle serialization and deserialization of files. Every
[`Dep`](dependencies.md) needs a **Reader** (to load data), every
[`Out`](outputs.md) needs a **Writer** (to save data), and
[`IncrementalOut`](outputs.md#incrementalout) needs a full **Loader** (both).

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

- **Immutable** — fields are frozen, safe to share across processes
- **Picklable** — can be sent to worker processes
- **Fingerprinted** — their source code is hashed, so changes to a loader
  trigger stage re-runs

| Base class | Methods | Used by |
|------------|---------|---------|
| `Reader[R]` | `load(path) -> R` | `Dep`, `PlaceholderDep` |
| `Writer[W]` | `save(data, path)` | `Out`, `Plot`, `DirectoryOut` |
| `Loader[W, R]` | `load() + save() + empty()` | `IncrementalOut`, or anywhere both read and write are needed |

For symmetric loaders where the read and write types are the same,
`Loader[T]` (single type parameter) is equivalent to `Loader[T, T]`.

## Built-in loaders

All built-in loaders live in `pivot.loaders`:

| Loader | Base | Read type (`R`) | Write type (`W`) | Options |
|--------|------|-----------------|-------------------|---------|
| `CSV()` | `Loader` | `DataFrame` | `DataFrame` | `index_col`, `sep`, `dtype` |
| `JSON()` | `Loader` | any JSON type | any JSON type | `indent`, `empty_factory` |
| `YAML()` | `Loader` | any YAML type | any YAML type | `empty_factory` |
| `Text()` | `Loader` | `str` | `str` | — |
| `JSONL()` | `Loader` | `list[dict]` | `list[dict]` | — |
| `DataFrameJSONL()` | `Loader` | `DataFrame` | `DataFrame` | — |
| `Pickle()` | `Loader` | `T` | `T` | `protocol` |
| `PathOnly()` | `Loader` | `Path` | `Path` | — |
| `MatplotlibFigure()` | `Writer` only | — | `Figure` | `dpi`, `bbox_inches`, `transparent` |

### Usage examples

```python
from typing import Annotated
from pandas import DataFrame
import pivot

# CSV with options
data: Annotated[DataFrame, pivot.Dep("data.csv", pivot.loaders.CSV(sep="\t", index_col=0))]

# JSON (used by Dep and Out)
config: Annotated[dict, pivot.Dep("config.json", pivot.loaders.JSON())]
result: Annotated[dict, pivot.Out("result.json", pivot.loaders.JSON(indent=4))]

# YAML
params: Annotated[dict, pivot.Dep("params.yaml", pivot.loaders.YAML())]

# Plain text
readme: Annotated[str, pivot.Dep("README.md", pivot.loaders.Text())]

# JSONL (one JSON object per line)
records: Annotated[list[dict], pivot.Dep("events.jsonl", pivot.loaders.JSONL())]

# DataFrame from JSONL (uses pandas.read_json)
df: Annotated[DataFrame, pivot.Dep("data.jsonl", pivot.loaders.DataFrameJSONL())]

# Pickle (arbitrary Python objects)
model: Annotated[object, pivot.Out("model.pkl", pivot.loaders.Pickle())]

# PathOnly (no-op: stage gets/creates the file manually)
raw_file: Annotated[pathlib.Path, pivot.Dep("binary.dat", pivot.loaders.PathOnly())]

# MatplotlibFigure (write-only: saves and closes the figure)
plot: Annotated[Figure, pivot.Plot("loss.png", pivot.loaders.MatplotlibFigure(dpi=300))]

# Metric (defaults to JSON() — usually no loader needed)
scores: Annotated[dict, pivot.Metric("metrics.json")]
```

### `PathOnly` — manual file handling

`PathOnly()` is the escape hatch. On read, it returns the file's `pathlib.Path`
instead of loading content. On write, it validates the file exists (the stage
must create it). Use it for binary formats, custom parsers, or directory deps:

```python
def convert(
    source: Annotated[pathlib.Path, pivot.Dep("input.bin", pivot.loaders.PathOnly())],
) -> Annotated[pathlib.Path, pivot.Out("output.bin", pivot.loaders.PathOnly())]:
    output_path = pathlib.Path("output.bin")
    # Custom binary processing
    output_path.write_bytes(transform(source.read_bytes()))
    return output_path
```

### `MatplotlibFigure` — write-only

`MatplotlibFigure` is a `Writer` (not `Loader`) because image files can't be
loaded back as `Figure` objects. It saves the figure and calls `plt.close()` to
free memory:

```python
from matplotlib.figure import Figure
import pivot

# Options:
pivot.loaders.MatplotlibFigure()                           # defaults: 150 dpi, tight bbox
pivot.loaders.MatplotlibFigure(dpi=300, transparent=True)  # publication quality
```

Format is inferred from the file extension (`.png`, `.pdf`, `.svg`).

### `IncrementalOut` and `empty()`

`IncrementalOut` requires a `Loader` (not just `Reader` or `Writer`) because it
reads previous state and writes new state. On the first run, when no previous
output exists, Pivot calls `loader.empty()` to get a starting value:

| Loader | `empty()` returns |
|--------|-------------------|
| `JSON()` | `{}` (override with `empty_factory=list` for lists) |
| `YAML()` | `{}` (override with `empty_factory=list`) |
| `CSV()` | empty `DataFrame` |
| `Text()` | `""` |
| `JSONL()` | `[]` |
| `DataFrameJSONL()` | empty `DataFrame` |
| `Pickle()` | raises `NotImplementedError` |
| `PathOnly()` | raises `NotImplementedError` |

If you need `IncrementalOut` with a loader that doesn't support `empty()`,
override it in a custom loader.

## Writing custom loaders

Custom loaders are frozen dataclasses that extend `Reader`, `Writer`, or
`Loader`:

### Read-only loader

```python
import dataclasses
import pathlib
from typing import override
import pivot

@dataclasses.dataclass(frozen=True)
class Parquet(pivot.loaders.Reader["DataFrame"]):
    """Load Parquet files as DataFrames."""

    columns: list[str] | None = None

    @override
    def load(self, path: pathlib.Path) -> "DataFrame":
        import pandas
        return pandas.read_parquet(path, columns=self.columns)
```

Use with `Dep`:

```python
data: Annotated[DataFrame, pivot.Dep("features.parquet", Parquet(columns=["a", "b"]))]
```

### Write-only loader

```python
import dataclasses
import pathlib
from typing import override
import pivot

@dataclasses.dataclass(frozen=True)
class ParquetWriter(pivot.loaders.Writer["DataFrame"]):
    """Write DataFrames as Parquet files."""

    compression: str = "snappy"

    @override
    def save(self, data: "DataFrame", path: pathlib.Path) -> None:
        data.to_parquet(path, compression=self.compression)
```

Use with `Out`:

```python
result: Annotated[DataFrame, pivot.Out("output.parquet", ParquetWriter())]
```

### Full bidirectional loader

```python
import dataclasses
import pathlib
from typing import override
import pivot

@dataclasses.dataclass(frozen=True)
class Parquet(pivot.loaders.Loader["DataFrame"]):
    """Read and write Parquet files."""

    columns: list[str] | None = None
    compression: str = "snappy"

    @override
    def load(self, path: pathlib.Path) -> "DataFrame":
        import pandas
        return pandas.read_parquet(path, columns=self.columns)

    @override
    def save(self, data: "DataFrame", path: pathlib.Path) -> None:
        data.to_parquet(path, compression=self.compression)

    @override
    def empty(self) -> "DataFrame":
        import pandas
        return pandas.DataFrame()
```

Use with `Dep`, `Out`, or `IncrementalOut`:

```python
# As dependency
data: Annotated[DataFrame, pivot.Dep("input.parquet", Parquet())]

# As output
result: Annotated[DataFrame, pivot.Out("output.parquet", Parquet())]

# As incremental output (needs load + save + empty)
state: Annotated[DataFrame, pivot.IncrementalOut("state.parquet", Parquet())]
```

### Requirements for custom loaders

1. **Frozen dataclass** — use `@dataclasses.dataclass(frozen=True)`. This
   ensures immutability and enables pickling across worker processes.
2. **Picklable** — all fields must be serializable. Avoid lambdas, open file
   handles, or unpicklable objects as fields.
3. **Deterministic** — given the same input, `save()` should produce the same
   file content. This ensures correct cache behavior.
4. **Override decorator** — use `@override` on `load()`, `save()`, and
   `empty()` for clarity and type-checker support.

### Loader fingerprinting

Pivot fingerprints loader source code alongside stage function code. If you
change a loader's implementation (e.g., switch from `snappy` to `gzip`
compression), all stages using that loader re-run automatically. Loader
**field values** (like `dpi=300`) are part of the frozen dataclass identity and
are captured in the stage's parameter hash.

## Choosing the right base class

```
Does your stage only READ the file?
  └─ Yes → Reader[R]    (for Dep / PlaceholderDep)

Does your stage only WRITE the file?
  └─ Yes → Writer[W]    (for Out / Plot / DirectoryOut)

Does your stage both READ and WRITE (IncrementalOut)?
  └─ Yes → Loader[W, R] (implements load + save + empty)

Are the read and write types different?
  └─ Yes → Loader[WriteType, ReadType]
  └─ No  → Loader[T]    (shorthand for Loader[T, T])
```

## Summary

| Access | Purpose |
|--------|---------|
| `pivot.loaders.CSV` | pandas DataFrame CSV |
| `pivot.loaders.JSON` | JSON files (dict, list, etc.) |
| `pivot.loaders.YAML` | YAML files |
| `pivot.loaders.Text` | Plain text strings |
| `pivot.loaders.JSONL` | JSON Lines (list of dicts) |
| `pivot.loaders.DataFrameJSONL` | JSON Lines as DataFrame |
| `pivot.loaders.Pickle` | Arbitrary Python objects |
| `pivot.loaders.PathOnly` | No-op (manual file handling) |
| `pivot.loaders.MatplotlibFigure` | matplotlib figure (write-only) |
| `pivot.loaders.Reader` | Base class for read-only custom loaders |
| `pivot.loaders.Writer` | Base class for write-only custom loaders |
| `pivot.loaders.Loader` | Base class for bidirectional custom loaders |

---

**See also:** [Artifacts & the DAG](artifacts-and-dag.md) | [Dependencies](dependencies.md) | [Outputs](outputs.md)