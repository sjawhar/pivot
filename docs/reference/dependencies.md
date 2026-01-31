# Dependencies & Loaders

Dependencies are declared as annotated function parameters. Loaders define how Pivot reads your data.

## Declaring Dependencies

Dependencies use the `Annotated` type with `Dep`:

```python
from typing import Annotated

import pandas
from pivot import loaders, outputs


def preprocess(
    raw: Annotated[pandas.DataFrame, outputs.Dep("data/raw.csv", loaders.CSV())],
    config: Annotated[dict, outputs.Dep("config/settings.yaml", loaders.YAML())],
) -> PreprocessOutputs:
    ...
```

YAML provides path overrides (keys must match parameter names):

```yaml
stages:
  preprocess:
    python: stages.preprocess
    deps:
      raw: data/raw.csv
      config: config/settings.yaml
```

### Automatic Code Tracking

Pivot automatically tracks Python code dependencies. You don't need to list `.py` files in `deps` - changes to your stage function and any functions it calls are detected automatically.

## Available Loaders

| Loader | Type | Use for |
|--------|------|---------|
| `CSV()` | `pandas.DataFrame` | Tabular data (.csv) |
| `JSON()` | `dict` / `list` | Config, small structured data |
| `YAML()` | `dict` | Config files |
| `Pickle()` | `Any` | Python objects (not portable) |
| `PathOnly()` | `pathlib.Path` | When you handle I/O yourself |

## Usage Examples

```python
from typing import Annotated
import pathlib

import pandas
from pivot import loaders, outputs


def process(
    # CSV data loaded as DataFrame
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],

    # JSON config loaded as dict
    config: Annotated[dict, outputs.Dep("config.json", loaders.JSON())],

    # YAML config loaded as dict
    settings: Annotated[dict, outputs.Dep("settings.yaml", loaders.YAML())],

    # Path only - you load it yourself
    model: Annotated[pathlib.Path, outputs.Dep("model.h5", loaders.PathOnly())],
) -> OutputType:
    # data is already a DataFrame
    # config is already a dict
    # model is a Path - load it yourself
    loaded_model = load_model(model)
    ...
```

## Loader Options

### CSV

```python
loaders.CSV(
    index_col=None,   # Column to use as index (int or str)
    sep=",",          # Field separator
    dtype=None,       # Column types {"col": "int64"}
)
```

### JSON

```python
loaders.JSON(
    indent=2,         # Indentation for pretty printing (None for compact)
)
```

### Pickle

```python
loaders.Pickle(
    protocol=pickle.HIGHEST_PROTOCOL,  # Pickle protocol version
)
```

## When to Use PathOnly

Use `PathOnly()` when:

- Your library has its own load method (e.g., `keras.models.load_model()`)
- You want full control over the file format
- The file isn't a standard format supported by built-in loaders

```python
def train(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
    pretrained: Annotated[pathlib.Path, outputs.Dep("base_model.h5", loaders.PathOnly())],
) -> TrainOutputs:
    # Load model yourself
    model = keras.models.load_model(pretrained)
    ...
```

The `PathOnly` loader:

- On load: Returns the `pathlib.Path` so you can load it yourself
- On save: Validates the file exists (you must create it)

## Custom Loaders

### The Loader Type Hierarchy

Pivot has three loader protocols:

- **`Reader[R]`** - Read-only, requires `load()` method
- **`Writer[W]`** - Write-only, requires `save()` method
- **`Loader[W, R]`** - Bidirectional, inherits from both `Reader[R]` and `Writer[W]`

`Dep.loader` is typed as `Reader[R]`, so it accepts any loader that can read data. This includes:
- `Reader[R]` implementations (read-only)
- `Loader[T]` implementations (bidirectional, since `Loader` inherits from `Reader`)

Write-only loaders (`Writer[W]`) cannot be used with `Dep` since they lack a `load()` method.

> **Note:** `Loader[T]` is equivalent to `Loader[T, T]` thanks to a PEP 696 default type parameter. This means the write and read types are the same unless explicitly specified otherwise.

### Creating a Custom Loader

Extend `Loader[T]` with `load()` and `save()` methods:

```python
import dataclasses
import pathlib

import pandas
from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Parquet(loaders.Loader[pandas.DataFrame]):
    """Parquet file loader."""

    compression: str = "snappy"

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_parquet(path)

    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        data.to_parquet(path, compression=self.compression)
```

If you only need to read (not write), implement `Reader[R]` instead:

```python
@dataclasses.dataclass(frozen=True)
class ParquetReader(loaders.Reader[pandas.DataFrame]):
    """Read-only Parquet loader."""

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_parquet(path)
```

Use your custom loader:

```python
def process(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.parquet", Parquet())],
) -> Annotated[pandas.DataFrame, outputs.Out("output.parquet", Parquet(compression="gzip"))]:
    return data.dropna()
```

### Custom Loader Requirements

Custom loaders must be:

- **Immutable** (`@dataclasses.dataclass(frozen=True)`)
- **Module-level** (for pickling to worker processes)
- **Fingerprinted** (loader code changes trigger stage re-runs)

## Directory Dependencies

Depend on all files in a directory:

```yaml
stages:
  process:
    python: stages.process
    deps:
      data_dir: data/
```

Changes to any file in `data/` will trigger re-execution.

## Upstream Stage Outputs

When a stage depends on another stage's output, just reference the output path:

```yaml
stages:
  preprocess:
    python: stages.preprocess
    deps:
      raw: data.csv
    outs:
      clean: processed.parquet

  train:
    python: stages.train
    deps:
      data: processed.parquet  # References preprocess output
    outs:
      model: model.pkl
```

Pivot automatically builds the dependency graph from path relationships.

## See Also

- [Defining Pipelines](pipelines.md) - Stage definition patterns
- [Output Types](outputs.md) - Output types and caching
