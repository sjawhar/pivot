# Adding Loaders

Guide for adding new data loaders to Pivot.

## Loader Architecture

Pivot provides three base classes for data I/O, allowing you to implement exactly the capabilities you need:

```python
@dataclasses.dataclass(frozen=True)
class Reader[R](ABC):
    """Read-only - can load data from a file.

    Use for dependencies where you only need to read, not write.
    """

    @abstractmethod
    def load(self, path: pathlib.Path) -> R:
        """Load data from file path."""
        ...


@dataclasses.dataclass(frozen=True)
class Writer[W](ABC):
    """Write-only - can save data to a file.

    Use for outputs where you only need to write, not read.
    """

    @abstractmethod
    def save(self, data: W, path: pathlib.Path) -> None:
        """Save data to file path."""
        ...


@dataclasses.dataclass(frozen=True)
class Loader[W, R = W](Writer[W], Reader[R], ABC):
    """Bidirectional - can both save and load data.

    W is the write type (what save() accepts).
    R is the read type (what load() returns).
    For symmetric loaders where W == R, use a single type parameter: Loader[T].
    """
    ...
```

### When to Use Each Base Class

| Base Class | Use Case | Example |
|------------|----------|---------|
| `Reader[R]` | Dependencies only (read from disk) | Reading config files, external data |
| `Writer[W]` | Outputs only (write to disk) | `MatplotlibFigure` - images can't be read back as Figures |
| `Loader[T]` | Both directions (most common) | CSV, JSON, Pickle - symmetric read/write |
| `Loader[W, R]` | Asymmetric bidirectional | Write one type, read back a different type |

**Note:** `Loader[T]` uses PEP 696 type parameter defaults, so `Loader[T]` is equivalent to `Loader[T, T]`.

## Creating a Custom Loader

### Step 1: Choose the Right Base Class

- **`Reader[R]`** - Use when you only need to read data (e.g., for `Dep()` only)
- **`Writer[W]`** - Use when you only need to write data (e.g., for `Out()` only, like plots)
- **`Loader[T]`** - Use when you need both read and write (most common, required for `IncrementalOut`)

### Step 2: Define the Loader Class

Create a frozen dataclass that extends the appropriate base class:

```python
# src/pivot/loaders.py (or your own module)
import dataclasses
import pathlib

import pandas
from pivot.loaders import Loader


# Bidirectional loader (most common) - Loader[T] defaults R=W via PEP 696
@dataclasses.dataclass(frozen=True)
class Parquet(Loader[pandas.DataFrame]):
    """Parquet file loader."""

    compression: str = "snappy"
    engine: str = "auto"

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_parquet(path, engine=self.engine)

    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        data.to_parquet(path, compression=self.compression, engine=self.engine)
```

Here are examples using each base class:

```python
from pivot.loaders import Reader, Writer, Loader

# Reader-only: for dependencies you only read
@dataclasses.dataclass(frozen=True)
class ConfigReader(Reader[dict]):
    """Read-only config loader."""

    def load(self, path: pathlib.Path) -> dict:
        with open(path) as f:
            return json.load(f)


# Writer-only: for outputs you only write (like plots)
@dataclasses.dataclass(frozen=True)
class PlotWriter(Writer[Figure]):
    """Write-only plot saver."""
    dpi: int = 150

    def save(self, data: Figure, path: pathlib.Path) -> None:
        data.savefig(path, dpi=self.dpi)


# Bidirectional: for read/write (required for IncrementalOut)
@dataclasses.dataclass(frozen=True)
class Parquet(Loader[pandas.DataFrame]):
    """Parquet file loader - Loader[T] is shorthand for Loader[T, T]."""
    ...
```

### Step 3: Add Type Hints

Ensure the loader is properly typed:

```python
@dataclasses.dataclass(frozen=True)
class Parquet(Loader[pandas.DataFrame]):
    """Parquet file loader."""

    compression: str = "snappy"

    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_parquet(path)

    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        data.to_parquet(path, compression=self.compression)
```

### Step 4: Add Tests

```python
# tests/unit/test_loaders.py
import pathlib

import pandas
import pytest

from pivot.loaders import Parquet


def test_parquet_roundtrip(tmp_path: pathlib.Path) -> None:
    """Test Parquet loader can save and load data."""
    loader = Parquet()
    path = tmp_path / "test.parquet"

    # Create test data
    original = pandas.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # Save
    loader.save(original, path)
    assert path.exists()

    # Load
    loaded = loader.load(path)

    # Verify
    pandas.testing.assert_frame_equal(original, loaded)


def test_parquet_compression(tmp_path: pathlib.Path) -> None:
    """Test Parquet loader respects compression option."""
    loader = Parquet(compression="gzip")
    path = tmp_path / "test.parquet"

    data = pandas.DataFrame({"a": [1, 2, 3]})
    loader.save(data, path)

    # Verify compression was used (file size would differ)
    assert path.exists()
```

## Requirements

Custom loaders must be:

### 1. Immutable

Use `@dataclasses.dataclass(frozen=True)`:

```python
# Good - frozen
@dataclasses.dataclass(frozen=True)
class MyLoader(Loader[T]):
    option: str = "default"

# Bad - mutable
class MyLoader(Loader[T]):
    def __init__(self, option: str = "default"):
        self.option = option  # Mutable!
```

### 2. Module-Level

Define loaders at module level for pickling to worker processes:

```python
# Good - module level
@dataclasses.dataclass(frozen=True)
class MyLoader(Loader[T]):
    ...

# Bad - nested in function
def create_loader():
    @dataclasses.dataclass(frozen=True)
    class MyLoader(Loader[T]):  # Can't be pickled!
        ...
```

### 3. Fingerprinted

Loader code is automatically fingerprinted. Changes to loader code trigger stage re-runs.

Options are part of the fingerprint:

```python
# These create different fingerprints
Parquet(compression="snappy")
Parquet(compression="gzip")
```

## Usage Example

```python
from typing import Annotated, TypedDict

import pandas
from pivot import outputs
from myproject.loaders import Parquet


class ProcessOutputs(TypedDict):
    data: Annotated[pandas.DataFrame, outputs.Out("data.parquet", Parquet())]


def process(
    raw: Annotated[pandas.DataFrame, outputs.Dep("raw.parquet", Parquet())],
) -> ProcessOutputs:
    return {"data": raw.dropna()}
```

## Adding to Built-in Loaders

To add a loader to Pivot's built-in set:

1. Add to `src/pivot/loaders.py`
2. Export in `__all__`
3. Add documentation to `docs/reference/dependencies.md`
4. Add tests

## Checklist

- [ ] Uses `@dataclasses.dataclass(frozen=True)`
- [ ] Defined at module level
- [ ] Extends the appropriate base class (`Reader`, `Writer`, or `Loader`)
- [ ] Implements required methods (`load()` for Reader, `save()` for Writer, both for Loader)
- [ ] Has proper type hints
- [ ] Has unit tests for roundtrip (for bidirectional loaders)
- [ ] Has tests for all options
- [ ] Documented in README or docs (if adding to core)

## See Also

- [Dependencies & Loaders](../reference/dependencies.md) - Using loaders
- [Code Style](style.md) - Coding conventions
