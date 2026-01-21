# Loaders

Loaders define how Pivot reads and writes your data.

## Available Loaders

| Loader | Type | Use for |
|--------|------|---------|
| `CSV()` | `pandas.DataFrame` | Tabular data (.csv) |
| `JSON()` | `dict` / `list` | Config, small structured data |
| `YAML()` | `dict` | Config files |
| `Pickle()` | `Any` | Python objects (not portable) |
| `PathOnly()` | `pathlib.Path` | When you handle I/O yourself |

## Usage

```python
from typing import Annotated
from pivot import loaders, outputs

def process(
    data: Annotated[pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV())],
) -> Annotated[dict, outputs.Out("result.json", loaders.JSON())]:
    return {"rows": len(data)}
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

- Your library has its own save method (e.g., `model.save()`)
- You want full control over the file format
- The file isn't a standard format supported by built-in loaders

Example:

```python
def train(
    data: Annotated[pandas.DataFrame, outputs.Dep("data.csv", loaders.CSV())],
) -> Annotated[pathlib.Path, outputs.Out("model.h5", loaders.PathOnly())]:
    model = train_model(data)
    model.save("model.h5")  # You handle the save
    return pathlib.Path("model.h5")
```

The `PathOnly` loader:

- On load: Returns the `pathlib.Path` so you can load it yourself
- On save: Validates the file exists (you must create it)

## Custom Loaders

Extend `Loader[T]` with `load()` and `save()` methods:

```python
import dataclasses
import pathlib
from pivot import loaders


@dataclasses.dataclass(frozen=True)
class Parquet(loaders.Loader[pandas.DataFrame]):
    """Parquet file loader."""

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_parquet(path)

    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        data.to_parquet(path)
```

Custom loaders must be:

- **Immutable** (`@dataclasses.dataclass(frozen=True)`)
- **Module-level** (for pickling to worker processes)
- **Fingerprinted** (changes trigger stage re-runs)

## See Also

- [Defining Stages](stages.md) - Stage definition patterns
- [Output Types](outputs.md) - Output types and options
