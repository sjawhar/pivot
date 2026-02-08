from __future__ import annotations

import abc
import dataclasses
import json
import os
import pathlib
import pickle
import tempfile
from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

import pandas
import yaml


@dataclasses.dataclass(frozen=True)
class Reader[R](abc.ABC):
    """Read-only loader - can load data from a file.

    Use for dependencies where you only need to read, not write.
    Readers are immutable, picklable, and their code is fingerprinted.
    """

    @abc.abstractmethod
    def load(self, path: pathlib.Path) -> R:
        """Load data from file path."""
        ...


@dataclasses.dataclass(frozen=True)
class Writer[W](abc.ABC):
    """Write-only loader - can save data to a file.

    Use for outputs where you only need to write, not read.
    Writers are immutable, picklable, and their code is fingerprinted.
    """

    @abc.abstractmethod
    def save(self, data: W, path: pathlib.Path) -> None:
        """Save data to file path."""
        ...


@dataclasses.dataclass(frozen=True)
class Loader[W, R = W](Writer[W], Reader[R], abc.ABC):
    """Bidirectional loader - can both save and load data.

    W is the write type (what save() accepts).
    R is the read type (what load() returns).
    For symmetric loaders where W == R, use a single type parameter: Loader[T].

    Loaders are immutable, picklable, and their code is fingerprinted.
    Changes to loader implementation trigger stage re-runs.
    """

    def empty(self) -> R:
        """Return an empty instance of the loaded type.

        Used for IncrementalOut on first run when no previous output exists.
        Override in loaders that support IncrementalOut (JSON, CSV, YAML).
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot provide an empty instance. For IncrementalOut, use a loader with a known empty value (JSON, CSV, YAML)."
        )


@dataclasses.dataclass(frozen=True)
class CSV[T](Loader[T, T]):
    """CSV file loader using pandas.

    Generic type parameter indicates the DataFrame type for type checking.
    Always returns pandas.DataFrame at runtime.
    """

    index_col: int | str | list[int | str] | None = None
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

    @override
    def empty(self) -> T:
        return pandas.DataFrame()  # pyright: ignore[reportReturnType] - returns DataFrame, user specifies T


@dataclasses.dataclass(frozen=True)
class JSON[T](Loader[T, T]):
    """JSON file loader.

    Generic type parameter indicates the expected type for type checking.

    Args:
        indent: JSON indentation (default 2, None for compact).
        empty_factory: Callable returning empty value for IncrementalOut first run.
            Defaults to dict. Use list for list types: JSON(empty_factory=list).
    """

    indent: int | None = 2
    empty_factory: Callable[[], T] = dict  # pyright: ignore[reportAssignmentType] - dict is default

    @override
    def load(self, path: pathlib.Path) -> T:
        with open(path) as f:
            return json.load(f)  # type: ignore[return-value] - json.load returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=self.indent)

    @override
    def empty(self) -> T:
        return self.empty_factory()


@dataclasses.dataclass(frozen=True)
class YAML[T](Loader[T, T]):
    """YAML file loader.

    Generic type parameter indicates the expected type for type checking.

    Args:
        empty_factory: Callable returning empty value for IncrementalOut first run.
            Defaults to dict. Use list for list types: YAML(empty_factory=list).
    """

    empty_factory: Callable[[], T] = dict  # pyright: ignore[reportAssignmentType] - dict is default

    @override
    def load(self, path: pathlib.Path) -> T:
        with open(path) as f:
            return yaml.safe_load(f)  # type: ignore[return-value] - yaml returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @override
    def empty(self) -> T:
        return self.empty_factory()


@dataclasses.dataclass(frozen=True)
class Text(Loader[str, str]):
    """Plain text file loader.

    Saves atomically via temp file + rename to prevent corruption.
    """

    @override
    def load(self, path: pathlib.Path) -> str:
        return path.read_text()

    @override
    def save(self, data: str, path: pathlib.Path) -> None:
        if not isinstance(data, str):  # pyright: ignore[reportUnnecessaryIsInstance] - runtime validation for misuse
            raise TypeError(f"Text save expects str, got {type(data).__name__}")  # pyright: ignore[reportUnreachable]

        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".txt.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.rename(tmp_path_str, path)
        except Exception:
            tmp_path = pathlib.Path(tmp_path_str)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @override
    def empty(self) -> str:
        return ""


@dataclasses.dataclass(frozen=True)
class JSONL(Loader[list[dict[str, Any]], list[dict[str, Any]]]):
    """JSONL (JSON Lines) file loader - one JSON object per line.

    Saves atomically via temp file + rename. Reports line numbers on parse errors.
    """

    @override
    def load(self, path: pathlib.Path) -> list[dict[str, Any]]:
        results = list[dict[str, Any]]()
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
        return results

    @override
    def save(self, data: list[dict[str, Any]], path: pathlib.Path) -> None:
        if not isinstance(data, list):  # pyright: ignore[reportUnnecessaryIsInstance] - runtime validation for misuse
            raise TypeError(f"JSONL save expects list, got {type(data).__name__}")  # pyright: ignore[reportUnreachable]

        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            os.rename(tmp_path_str, path)
        except Exception:
            tmp_path = pathlib.Path(tmp_path_str)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @override
    def empty(self) -> list[dict[str, Any]]:
        return []


@dataclasses.dataclass(frozen=True)
class DataFrameJSONL(Loader[pandas.DataFrame, pandas.DataFrame]):
    """JSONL (JSON Lines) file loader that returns a pandas DataFrame.

    Uses pandas.read_json with lines=True for efficient loading.
    Saves atomically via temp file + rename.
    """

    @override
    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        return pandas.read_json(path, lines=True, orient="records", convert_dates=False)

    @override
    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        if not isinstance(data, pandas.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance] - runtime validation for misuse
            raise TypeError(f"DataFrameJSONL save expects DataFrame, got {type(data).__name__}")  # pyright: ignore[reportUnreachable]

        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                data.to_json(f, lines=True, orient="records")
            os.rename(tmp_path_str, path)
        except Exception:
            tmp_path = pathlib.Path(tmp_path_str)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @override
    def empty(self) -> pandas.DataFrame:
        return pandas.DataFrame()


@dataclasses.dataclass(frozen=True)
class Pickle[T](Loader[T, T]):
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
class PathOnly(Loader[pathlib.Path, pathlib.Path]):
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


@dataclasses.dataclass(frozen=True)
class MatplotlibFigure(Writer["Figure"]):
    """Save matplotlib figures as images. Write-only.

    Figures are closed after saving to prevent memory leaks.
    Format is inferred from path extension (.png, .pdf, .svg).

    This is a Writer (not Loader) because images cannot be loaded back as Figures.
    To read images, use a separate Reader like PathOnly() and load manually.
    """

    dpi: int = 150
    bbox_inches: Literal["tight"] | None = "tight"
    transparent: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.dpi <= 2400):
            raise ValueError(f"dpi must be between 1 and 2400, got {self.dpi}")

    @override
    def save(self, data: Figure, path: pathlib.Path) -> None:
        import matplotlib.pyplot as plt

        try:
            data.savefig(  # pyright: ignore[reportUnknownMemberType] - stubs incomplete
                path,
                dpi=self.dpi,
                bbox_inches=self.bbox_inches,
                transparent=self.transparent,
            )
        finally:
            plt.close(data)
