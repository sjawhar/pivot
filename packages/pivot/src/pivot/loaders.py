from __future__ import annotations

import abc
import collections.abc as collections_abc
import dataclasses
import datetime
import json
import math
import operator
import os
import pathlib
import pickle
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

import numpy
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

        Used by an incremental output on its first run.
        Override in loaders that support an empty value (JSON, CSV, YAML).
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot provide an empty instance. "
            + "Use an incremental output loader with a known empty value (JSON, CSV, YAML)."
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
        # pandas writes a zero-column frame as a bare line break, which read_csv rejects.
        if path.stat().st_size <= 2 and not path.read_bytes().strip():
            return self.empty()
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
        empty_factory: Callable returning the first-run value for an incremental output.
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
    """YAML file loader that writes only values safe_load can read.

    Numpy scalars and arrays are converted to equivalent Python values. Other
    arbitrary Python objects raise ``TypeError`` instead of being written as
    unsafe ``!!python/*`` tags.

    Args:
        empty_factory: Callable returning the first-run value for an incremental output.
            Defaults to dict. Use list for list types: YAML(empty_factory=list).
    """

    empty_factory: Callable[[], T] = dict  # pyright: ignore[reportAssignmentType] - dict is default

    @override
    def load(self, path: pathlib.Path) -> T:
        with open(path) as f:
            return yaml.safe_load(f)  # type: ignore[return-value] - yaml returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        def plain(value: object, location: str) -> object:
            if value is pandas.NaT or value is pandas.NA:
                return None
            if isinstance(value, numpy.ndarray):
                return plain(value.tolist(), location)
            if isinstance(value, numpy.generic):
                unwrapped = value.item()
                if isinstance(unwrapped, numpy.generic):
                    raise TypeError(
                        f"YAML output must contain plain data; got {type(value).__name__} at {location}. "
                        + "Convert extended-precision NumPy values explicitly."
                    )
                return plain(unwrapped, location)
            if type(value) in (
                type(None),
                bool,
                int,
                float,
                str,
                bytes,
                datetime.date,
                datetime.datetime,
            ):
                return value
            if isinstance(value, collections_abc.Mapping):
                mapping = cast("collections_abc.Mapping[object, object]", value)
                plain_mapping = dict[object, object]()
                for key, item in mapping.items():
                    plain_key = plain(key, f"{location} key")
                    plain_mapping[plain_key] = plain(item, f"{location}[{key!r}]")
                return plain_mapping
            if isinstance(value, (list, tuple)):
                items = cast("list[object] | tuple[object, ...]", value)
                return [plain(item, f"{location}[{index}]") for index, item in enumerate(items)]
            raise TypeError(
                f"YAML output must contain plain data; got {type(value).__name__} at {location}. "
                + "Convert DataFrames with .to_dict() and dataclasses with dataclasses.asdict()."
            )

        content = yaml.safe_dump(plain(data, "root"), default_flow_style=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

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


_DATAFRAME_METADATA_KEY = "__pivot_dataframe_metadata__"
_DATAFRAME_METADATA_VERSION = 1


def _json_default(value: object) -> object:
    """Convert the non-JSON scalars pandas hands back from object columns."""
    if value is pandas.NaT:
        return None
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return pandas.Timedelta(value).isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _replace_nonfinite(value: object) -> object:
    """Map NaN/inf anywhere inside a nested value to None (JSON has no NaN)."""
    if isinstance(value, (numpy.ndarray, numpy.generic)):
        value = cast("object", value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_replace_nonfinite(item) for item in items]
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {key: _replace_nonfinite(item) for key, item in mapping.items()}
    return value


def _iso_datetime_texts(series: pandas.Series[Any], suffix: str = "") -> list[str]:
    """JSON texts for a naive datetime64 column: quoted ISO 8601 with nanoseconds, NaT as null.

    ISO texts contain only digits, "-", ":", "T", "." and the suffix, so they need no escaping.
    """
    values = series.to_numpy(dtype="datetime64[ns]")
    texts: list[str] = numpy.datetime_as_string(values, unit="ns").tolist()
    return ["null" if text == "NaT" else f'"{text}{suffix}"' for text in texts]


def _encode_jsonl_column(
    series: pandas.Series[Any], encoder: json.JSONEncoder
) -> tuple[list[str], dict[str, object] | None]:
    """Encode one DataFrame column as JSON value texts and return temporal metadata."""
    dtype = series.dtype
    if pandas.api.types.is_float_dtype(dtype):
        if isinstance(dtype, numpy.dtype) and dtype.itemsize > numpy.dtype("float64").itemsize:
            raise TypeError(f"unsupported extended-precision floating dtype {dtype}")
        # repr() is the shortest text that parses back to the identical float;
        # pandas.to_json rounds to at most 15 significant digits.
        floats: list[float] = series.to_numpy(dtype="float64", na_value=math.nan).tolist()
        return [repr(v) if math.isfinite(v) else "null" for v in floats], None
    if isinstance(dtype, numpy.dtype) and dtype.kind in "iu":
        return series.astype(str).tolist(), None
    if isinstance(dtype, numpy.dtype) and dtype.kind == "b":
        return numpy.where(series.to_numpy(), "true", "false").tolist(), None
    if pandas.api.types.is_object_dtype(dtype) and pandas.api.types.infer_dtype(
        series, skipna=True
    ) in ("date", "datetime"):
        series = pandas.Series(pandas.to_datetime(series))
        dtype = series.dtype
    if isinstance(dtype, pandas.DatetimeTZDtype):
        # Values are ISO UTC text; metadata must rebuild the original zone exactly.
        timezone = str(dtype.tz)
        try:
            reconstructed_dtype = pandas.DatetimeTZDtype(tz=timezone)
        except (KeyError, TypeError, ValueError) as e:
            raise TypeError(f"cannot round-trip timezone {timezone!r}") from e
        if reconstructed_dtype != dtype:
            raise TypeError(f"cannot round-trip timezone {timezone!r}")
        metadata: dict[str, object] = {"kind": "datetime", "timezone": timezone}
        return (
            _iso_datetime_texts(series.dt.tz_convert("UTC").dt.tz_localize(None), suffix="Z"),
            metadata,
        )
    if pandas.api.types.is_datetime64_dtype(dtype):
        return _iso_datetime_texts(series), {"kind": "datetime", "timezone": None}
    if pandas.api.types.is_timedelta64_dtype(dtype):
        deltas = cast("list[pandas.Timedelta]", series.tolist())
        return (
            ["null" if delta is pandas.NaT else f'"{delta.isoformat()}"' for delta in deltas],
            {"kind": "timedelta"},
        )
    missing: list[bool] = series.isna().tolist()
    cells = [
        None if is_missing else cell
        for cell, is_missing in zip(cast("list[object]", series.tolist()), missing, strict=True)
    ]
    encoded = list[str]()
    for cell in cells:
        try:
            encoded.append(encoder.encode(cell))
        except ValueError:
            encoded.append(encoder.encode(_replace_nonfinite(cell)))
    return encoded, None


def _restore_temporal_jsonl_column(
    series: pandas.Series[Any], metadata: collections_abc.Mapping[str, object]
) -> pandas.Series[Any]:
    """Restore a writer-marked temporal column; never infer temporal dtypes from strings."""
    kind = metadata["kind"]
    if kind == "timedelta":
        return pandas.Series(pandas.to_timedelta(series))
    if kind != "datetime":
        raise ValueError(f"Unsupported DataFrameJSONL temporal metadata kind: {kind!r}")
    timezone = metadata["timezone"]
    if timezone is None:
        return pandas.Series(pandas.to_datetime(series, format="ISO8601"))
    if not isinstance(timezone, str):
        raise ValueError(f"Invalid DataFrameJSONL timezone metadata: {timezone!r}")
    parsed = pandas.to_datetime(series, format="ISO8601", utc=True)
    return pandas.Series(parsed.dt.tz_convert(timezone))


@dataclasses.dataclass(frozen=True)
class DataFrameJSONL(Loader[pandas.DataFrame, pandas.DataFrame]):
    """JSONL (JSON Lines) file loader that returns a pandas DataFrame.

    One JSON object per row; the index is not written. The first line is a
    reserved ``{"__pivot_dataframe_metadata__": ...}`` JSON object describing
    Pivot-written temporal columns. A non-Pivot JSONL reader sees that explicit
    metadata record followed by normal row records; it can skip the first line
    to read the data rows unchanged.

    - floats: written as the shortest exact text, read back bit-identical
      (NaN and +/-inf are written as ``null`` and read back as NaN).
    - ``datetime.date``/``datetime`` objects and ``datetime64`` columns: written as
      ISO 8601 with nanoseconds and restored only when the metadata marks the
      column temporal. Original tz-aware dtype zones are retained.
    - ``timedelta64`` columns: written as ISO 8601 durations and restored from
      metadata, including all-null columns.
    - strings remain strings, including ISO-shaped strings.

    Cells that JSON cannot hold (``Decimal``, ``set``, ``bytes``, arbitrary
    objects) raise ``TypeError`` naming the column. Saves atomically via temp
    file + rename.
    """

    @override
    def load(self, path: pathlib.Path) -> pandas.DataFrame:
        records = list[object]()
        temporal_columns = dict[str, collections_abc.Mapping[str, object]]()
        float_columns = dict[str, str]()
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = cast("object", json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
                if line_num == 1 and isinstance(record, dict):
                    metadata_record = cast("dict[object, object]", record)
                    if set(metadata_record) == {_DATAFRAME_METADATA_KEY}:
                        raw_metadata = metadata_record[_DATAFRAME_METADATA_KEY]
                        if not isinstance(raw_metadata, dict):
                            raise ValueError(
                                f"Invalid DataFrameJSONL metadata at {path}: expected object"
                            )
                        typed_metadata = cast("dict[object, object]", raw_metadata)
                        if typed_metadata.get("version") != _DATAFRAME_METADATA_VERSION:
                            raise ValueError(
                                f"Unsupported DataFrameJSONL metadata version at {path}"
                            )
                        raw_temporal = typed_metadata.get("temporal_columns", {})
                        if not isinstance(raw_temporal, dict):
                            raise ValueError(f"Invalid DataFrameJSONL temporal metadata at {path}")
                        temporal_mapping = cast("dict[object, object]", raw_temporal)
                        for name, metadata in temporal_mapping.items():
                            if not isinstance(name, str) or not isinstance(metadata, dict):
                                raise ValueError(
                                    f"Invalid DataFrameJSONL temporal column metadata at {path}"
                                )
                            temporal_columns[name] = cast(
                                "collections_abc.Mapping[str, object]", metadata
                            )
                        raw_float = typed_metadata.get("float_columns", {})
                        if not isinstance(raw_float, dict):
                            raise ValueError(f"Invalid DataFrameJSONL float metadata at {path}")
                        float_mapping = cast("dict[object, object]", raw_float)
                        for name, dtype in float_mapping.items():
                            if not isinstance(name, str) or not isinstance(dtype, str):
                                raise ValueError(
                                    f"Invalid DataFrameJSONL float column metadata at {path}"
                                )
                            float_columns[name] = dtype
                        continue
                records.append(cast("object", record))
        frame = pandas.DataFrame(records)
        for name, metadata in temporal_columns.items():
            if name not in frame.columns:
                frame[name] = pandas.Series(pandas.NaT, index=frame.index)
            frame[name] = _restore_temporal_jsonl_column(frame[name], metadata)
        for name, dtype in float_columns.items():
            if name not in frame.columns:
                frame[name] = pandas.Series(math.nan, index=frame.index)
            frame[name] = frame[name].astype(dtype)
        return frame

    @override
    def save(self, data: pandas.DataFrame, path: pathlib.Path) -> None:
        if not isinstance(data, pandas.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance] - runtime validation for misuse
            raise TypeError(f"DataFrameJSONL save expects DataFrame, got {type(data).__name__}")  # pyright: ignore[reportUnreachable]
        if not data.columns.is_unique:
            duplicated = sorted({str(c) for c in data.columns[data.columns.duplicated()]})
            raise ValueError(
                f"DataFrameJSONL requires unique column names, duplicated: {duplicated}"
            )
        if _DATAFRAME_METADATA_KEY in {str(name) for name in data.columns}:
            raise ValueError(
                f"DataFrameJSONL reserves column name {_DATAFRAME_METADATA_KEY!r} for metadata"
            )

        encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"), default=_json_default)
        keys = list[str]()
        columns = list[list[str]]()
        temporal_columns = dict[str, dict[str, object]]()
        float_columns = dict[str, str]()
        for name in data.columns:
            string_name = str(name)
            dtype = data[name].dtype
            try:
                column, temporal_metadata = _encode_jsonl_column(data[name], encoder)
            except TypeError as e:
                raise TypeError(
                    f"DataFrameJSONL cannot write column {string_name!r}: {e}. Convert it to "
                    + "str/int/float/bool/list/dict values, or pick another format for this "
                    + "output, e.g. Annotated[pd.DataFrame, Pickle()]."
                ) from e
            columns.append(column)
            keys.append(encoder.encode(string_name) + ":")
            if temporal_metadata is not None:
                temporal_columns[string_name] = temporal_metadata
            if pandas.api.types.is_float_dtype(dtype):
                float_columns[string_name] = str(dtype)

        metadata = {
            _DATAFRAME_METADATA_KEY: {
                "version": _DATAFRAME_METADATA_VERSION,
                "temporal_columns": temporal_columns,
                "float_columns": float_columns,
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(encoder.encode(metadata) + "\n")
                for cells in zip(*columns, strict=True):
                    f.write("{" + ",".join(map(operator.add, keys, cells)) + "}\n")
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

    Warning:
        Pickle executes code during deserialization. Treat every Pickle artifact
        from a cache or remote as trusted code: anyone who can write that
        storage can execute code on a machine that loads it.
    """

    protocol: int = pickle.HIGHEST_PROTOCOL

    @override
    def load(self, path: pathlib.Path) -> T:
        warnings.warn(
            "Loading a Pickle artifact can execute arbitrary code. Treat any cache or remote that supplied it as trusted code.",
            RuntimeWarning,
            stacklevel=2,
        )
        with open(path, "rb") as f:
            return pickle.load(f)  # type: ignore[return-value] - pickle returns Any, user specifies T

    @override
    def save(self, data: T, path: pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=self.protocol)


@dataclasses.dataclass(frozen=True)
class PathOnly(Loader[pathlib.Path, pathlib.Path]):
    """Loader that hands paths to the stage instead of loading content.

    On load, returns the artifact's path; it may be a file or a directory.
    On save, takes ownership of the file or directory the stage returned by
    moving it to the output path pivot chose. The stage writes its result
    anywhere outside the project (e.g. under ``tempfile.mkdtemp()``) and
    returns that path; it must not reuse the path afterwards.
    """

    @override
    def load(self, path: pathlib.Path) -> pathlib.Path:
        return path

    @override
    def save(self, data: pathlib.Path, path: pathlib.Path) -> None:
        if not data.exists():
            raise FileNotFoundError(f"Stage returned path {data}, which does not exist")
        if data.resolve() == path.resolve():
            return
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Cannot move returned path {data} to {path}: the output path already exists"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(data, path)


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


def format_extension(fmt: object) -> str:
    """Map loader type to file extension.

    Args:
        fmt: A loader instance (CSV, JSON, YAML, etc.)

    Returns:
        File extension without the dot (e.g., "csv", "json", "pkl")
    """
    match fmt:
        case DataFrameJSONL():
            return "jsonl"
        case CSV():
            return "csv"
        case YAML():
            return "yaml"
        case JSON():
            return "json"
        case Text():
            return "txt"
        case Pickle():
            return "pkl"
        case MatplotlibFigure():
            return "png"
        case _:
            return "dat"
