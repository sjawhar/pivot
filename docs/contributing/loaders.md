# Adding Loaders

Guide for adding new data loaders to Pivot.

## Loader Architecture

Loaders define how Pivot reads and writes data. They implement the `Loader[T]` protocol:

```python
@dataclasses.dataclass(frozen=True)
class Loader(ABC, Generic[T]):
    """Base class for data loaders."""

    @abstractmethod
    def load(self, path: pathlib.Path) -> T:
        """Load data from path."""
        ...

    @abstractmethod
    def save(self, data: T, path: pathlib.Path) -> None:
        """Save data to path."""
        ...
```

## Creating a Custom Loader

### Step 1: Define the Loader Class

Create a frozen dataclass that extends `Loader[T]`:

```python
# src/pivot/loaders.py (or your own module)
import dataclasses
import pathlib
from typing import TypeVar

import pandas
from pivot.loaders import Loader


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

### Step 2: Add Type Hints

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

### Step 3: Add Tests

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
- [ ] Implements `load()` and `save()` methods
- [ ] Has proper type hints
- [ ] Has unit tests for roundtrip
- [ ] Has tests for all options
- [ ] Documented in README or docs (if adding to core)

## See Also

- [Dependencies & Loaders](../reference/dependencies.md) - Using loaders
- [Code Style](style.md) - Coding conventions
