from __future__ import annotations

import abc
import dataclasses
import json
import pathlib
import pickle
from typing import override

import pandas
import yaml


@dataclasses.dataclass(frozen=True)
class Loader[T](abc.ABC):
    """Base class for file loaders providing typed dependency/output access.

    Loaders are immutable, picklable, and their code is fingerprinted.
    Changes to loader implementation trigger stage re-runs.
    """

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> T:
        """Load data from file path."""
        ...

    @abc.abstractmethod
    def save(self, data: T, path: pathlib.Path) -> None:
        """Save data to file path."""
        ...


@dataclasses.dataclass(frozen=True)
class CSV[T](Loader[T]):
    """CSV file loader using pandas.

    Generic type parameter indicates the DataFrame type for type checking.
    Always returns pandas.DataFrame at runtime.
    """

    index_col: int | str | None = None
    sep: str = ","
    dtype: dict[str, str] | None = None

    @override
    def load(self, path: pathlib.Path) -> T:
        result = pandas.read_csv(  # pyright: ignore[reportUnknownMemberType] - pandas read_csv has complex overloads
            path,
            index_col=self.index_col,
            sep=self.sep,
            dtype=self.dtype,  # pyright: ignore[reportArgumentType] - dtype accepts more types at runtime
        )
        return result  # pyright: ignore[reportReturnType] - returns DataFrame, user specifies T for type checking

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        if not isinstance(data, pandas.DataFrame):
            raise TypeError(f"CSV loader expects DataFrame, got {type(data).__name__}")
        data.to_csv(path, index=self.index_col is not None)


@dataclasses.dataclass(frozen=True)
class JSON[T](Loader[T]):
    """JSON file loader.

    Generic type parameter indicates the expected dict type for type checking.
    Always returns dict at runtime.
    """

    indent: int | None = 2

    @override
    def load(self, path: pathlib.Path) -> T:
        with open(path) as f:
            return json.load(f)  # type: ignore[return-value] - json.load returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=self.indent)


@dataclasses.dataclass(frozen=True)
class YAML[T](Loader[T]):
    """YAML file loader.

    Generic type parameter indicates the expected type for type checking.
    Returns parsed YAML (typically dict) at runtime.
    """

    @override
    def load(self, path: pathlib.Path) -> T:
        with open(path) as f:
            return yaml.safe_load(f)  # type: ignore[return-value] - yaml returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclasses.dataclass(frozen=True)
class Pickle[T](Loader[T]):
    """Pickle file loader for arbitrary Python objects.

    WARNING: Loading pickle files from untrusted sources is a security risk.
    Pickle can execute arbitrary code during deserialization.
    """

    protocol: int = pickle.HIGHEST_PROTOCOL

    @override
    def load(self, path: pathlib.Path) -> T:
        with open(path, "rb") as f:
            return pickle.load(f)  # type: ignore[return-value] - pickle returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=self.protocol)


@dataclasses.dataclass(frozen=True)
class PathOnly(Loader[pathlib.Path]):
    """No-op loader that returns the path itself for manual loading.

    Use when you need custom loading logic that doesn't fit standard loaders.
    The save() method validates the file exists (user must create it manually).
    """

    @override
    def load(self, path: pathlib.Path) -> pathlib.Path:
        return path

    @override
    def save(self, data: pathlib.Path, path: pathlib.Path) -> None:
        _ = data  # PathOnly doesn't save data; just validates file exists
        if not path.exists():
            raise FileNotFoundError(f"Output file not created: {path}")
