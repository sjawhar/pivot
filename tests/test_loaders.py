# pyright: reportAbstractUsage=false, reportImplicitAbstractClass=false, reportImplicitOverride=false, reportUnknownArgumentType=false

from __future__ import annotations

import dataclasses
import pathlib
import pickle
import typing
from typing import Any

import pandas
import pytest

from pivot import loaders

# ==============================================================================
# Loader base class tests
# ==============================================================================


def test_loader_is_abstract() -> None:
    """Loader base class cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        loaders.Loader()  # type: ignore[abstract]


def test_loader_requires_load_method() -> None:
    """Subclasses must implement load()."""

    @dataclasses.dataclass(frozen=True)
    class PartialLoader(loaders.Loader[str]):
        def save(self, data: str, path: pathlib.Path) -> None:
            pass

    with pytest.raises(TypeError, match="abstract"):
        PartialLoader()  # type: ignore[abstract]


def test_loader_requires_save_method() -> None:
    """Subclasses must implement save()."""

    @dataclasses.dataclass(frozen=True)
    class PartialLoader(loaders.Loader[str]):
        def load(self, path: pathlib.Path) -> str:
            return ""

    with pytest.raises(TypeError, match="abstract"):
        PartialLoader()  # type: ignore[abstract]


# ==============================================================================
# CSV loader tests
# ==============================================================================


def test_csv_loader_load(tmp_path: pathlib.Path) -> None:
    """CSV loader reads DataFrame from file."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")

    loader = loaders.CSV()
    df = loader.load(csv_file)

    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_csv_loader_save(tmp_path: pathlib.Path) -> None:
    """CSV loader writes DataFrame to file."""
    csv_file = tmp_path / "output.csv"
    df = pandas.DataFrame({"x": [1, 2], "y": [3, 4]})

    loader = loaders.CSV()
    loader.save(df, csv_file)

    assert csv_file.exists()
    loaded = pandas.read_csv(csv_file)
    assert list(loaded.columns) == ["x", "y"]


def test_csv_loader_with_index_col(tmp_path: pathlib.Path) -> None:
    """CSV loader respects index_col option."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("idx,val\na,1\nb,2\n")

    loader = loaders.CSV(index_col="idx")
    df = loader.load(csv_file)

    assert df.index.name == "idx"
    assert list(df.index) == ["a", "b"]


def test_csv_loader_with_sep(tmp_path: pathlib.Path) -> None:
    """CSV loader respects sep option."""
    csv_file = tmp_path / "data.tsv"
    csv_file.write_text("a\tb\n1\t2\n")

    loader = loaders.CSV(sep="\t")
    df = loader.load(csv_file)

    assert list(df.columns) == ["a", "b"]


# ==============================================================================
# JSON loader tests
# ==============================================================================


def test_json_loader_load(tmp_path: pathlib.Path) -> None:
    """JSON loader reads dict from file."""
    json_file = tmp_path / "data.json"
    json_file.write_text('{"key": "value", "num": 42}')

    loader = loaders.JSON()
    data = loader.load(json_file)

    assert data == {"key": "value", "num": 42}


def test_json_loader_save(tmp_path: pathlib.Path) -> None:
    """JSON loader writes dict to file."""
    json_file = tmp_path / "output.json"
    data = {"foo": [1, 2, 3]}

    loader = loaders.JSON()
    loader.save(data, json_file)

    assert json_file.exists()
    content = json_file.read_text()
    assert '"foo"' in content
    assert "[1, 2, 3]" in content or "[\n" in content


def test_json_loader_with_indent(tmp_path: pathlib.Path) -> None:
    """JSON loader respects indent option."""
    json_file = tmp_path / "output.json"
    data = {"a": 1}

    loader = loaders.JSON(indent=4)
    loader.save(data, json_file)

    content = json_file.read_text()
    assert "    " in content  # 4-space indent


# ==============================================================================
# YAML loader tests
# ==============================================================================


def test_yaml_loader_load(tmp_path: pathlib.Path) -> None:
    """YAML loader reads dict from file."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("key: value\nlist:\n  - a\n  - b\n")

    loader = loaders.YAML()
    data = loader.load(yaml_file)

    assert data == {"key": "value", "list": ["a", "b"]}


def test_yaml_loader_save(tmp_path: pathlib.Path) -> None:
    """YAML loader writes dict to file."""
    yaml_file = tmp_path / "output.yaml"
    data = {"setting": True, "items": [1, 2]}

    loader = loaders.YAML()
    loader.save(data, yaml_file)

    assert yaml_file.exists()
    content = yaml_file.read_text()
    assert "setting:" in content


# ==============================================================================
# Pickle loader tests
# ==============================================================================


def test_pickle_loader_load(tmp_path: pathlib.Path) -> None:
    """Pickle loader reads object from file."""
    pkl_file = tmp_path / "data.pkl"
    obj = {"complex": [1, 2, {"nested": True}]}
    pkl_file.write_bytes(pickle.dumps(obj))

    loader = loaders.Pickle[dict[str, Any]]()
    loaded = loader.load(pkl_file)

    assert loaded == obj


def test_pickle_loader_save(tmp_path: pathlib.Path) -> None:
    """Pickle loader writes object to file."""
    pkl_file = tmp_path / "output.pkl"
    obj = {"data": [1, 2, 3]}

    loader = loaders.Pickle[dict[str, Any]]()
    loader.save(obj, pkl_file)

    assert pkl_file.exists()
    loaded = pickle.loads(pkl_file.read_bytes())
    assert loaded == obj


# ==============================================================================
# PathOnly loader tests
# ==============================================================================


def test_pathonly_loader_load_returns_path(tmp_path: pathlib.Path) -> None:
    """PathOnly loader returns the path itself."""
    file = tmp_path / "file.bin"
    file.write_bytes(b"binary data")

    loader = loaders.PathOnly()
    result = loader.load(file)

    assert result == file
    assert isinstance(result, pathlib.Path)


def test_pathonly_loader_save_validates_exists(tmp_path: pathlib.Path) -> None:
    """PathOnly save validates file exists (user must create it)."""
    file = tmp_path / "output.bin"
    file.write_bytes(b"data")

    loader = loaders.PathOnly()
    loader.save(file, file)  # No error - file exists


def test_pathonly_loader_save_raises_if_missing(tmp_path: pathlib.Path) -> None:
    """PathOnly save raises if file doesn't exist."""
    file = tmp_path / "missing.bin"

    loader = loaders.PathOnly()
    with pytest.raises(FileNotFoundError):
        loader.save(file, file)


# ==============================================================================
# Pickling tests (required for ProcessPoolExecutor)
# ==============================================================================


def test_csv_loader_is_picklable() -> None:
    """CSV loader can be pickled and unpickled."""
    loader = loaders.CSV(index_col="id", sep=";")
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert restored.index_col == "id"
    assert restored.sep == ";"


def test_json_loader_is_picklable() -> None:
    """JSON loader can be pickled and unpickled."""
    loader = loaders.JSON(indent=4)
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert restored.indent == 4


def test_yaml_loader_is_picklable() -> None:
    """YAML loader can be pickled and unpickled."""
    loader = loaders.YAML()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.YAML)


def test_pickle_loader_is_picklable() -> None:
    """Pickle loader can be pickled and unpickled."""
    loader = loaders.Pickle[dict[str, int]]()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.Pickle)


def test_pathonly_loader_is_picklable() -> None:
    """PathOnly loader can be pickled and unpickled."""
    loader = loaders.PathOnly()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.PathOnly)


# ==============================================================================
# Generic type parameter tests
# ==============================================================================


def test_csv_generic_type_preserved() -> None:
    """CSV generic type parameter can be extracted."""
    # The type annotation CSV[pandas.DataFrame] should preserve DataFrame
    hint = loaders.CSV[pandas.DataFrame]
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    assert origin is loaders.CSV
    assert args == (pandas.DataFrame,)


def test_json_generic_type_preserved() -> None:
    """JSON generic type parameter can be extracted."""
    hint = loaders.JSON[dict[str, int]]
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    assert origin is loaders.JSON
    assert args == (dict[str, int],)


def test_pickle_generic_type_preserved() -> None:
    """Pickle generic type parameter can be extracted."""

    class MyModel:
        pass

    hint = loaders.Pickle[MyModel]
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    assert origin is loaders.Pickle
    assert args == (MyModel,)


# ==============================================================================
# Custom loader subclassing tests
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class _CustomTextLoader(loaders.Loader[str]):
    """Custom loader for testing - loads text with prefix."""

    prefix: str = ""

    def load(self, path: pathlib.Path) -> str:
        return self.prefix + path.read_text()

    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)


def test_custom_loader_works(tmp_path: pathlib.Path) -> None:
    """Custom loader subclass works correctly."""
    file = tmp_path / "test.txt"
    file.write_text("hello")

    loader = _CustomTextLoader(prefix="PREFIX:")
    result = loader.load(file)

    assert result == "PREFIX:hello"


def test_custom_loader_is_picklable() -> None:
    """Custom loader subclass can be pickled."""
    loader = _CustomTextLoader(prefix="TEST:")
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert restored.prefix == "TEST:"


def test_custom_loader_generic_type() -> None:
    """Custom loader is subclass of Loader."""
    assert issubclass(_CustomTextLoader, loaders.Loader)
