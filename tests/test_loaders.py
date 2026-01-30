# pyright: reportAbstractUsage=false, reportImplicitAbstractClass=false, reportImplicitOverride=false, reportUnknownArgumentType=false

from __future__ import annotations

import dataclasses
import json
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
# Text loader tests
# ==============================================================================


def test_text_loader_load(tmp_path: pathlib.Path) -> None:
    """Text loader reads string from file."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("hello world\nline two")

    loader = loaders.Text()
    data = loader.load(txt_file)

    assert data == "hello world\nline two"


def test_text_loader_save(tmp_path: pathlib.Path) -> None:
    """Text loader writes string to file."""
    txt_file = tmp_path / "output.txt"
    data = "some text content"

    loader = loaders.Text()
    loader.save(data, txt_file)

    assert txt_file.exists()
    assert txt_file.read_text() == "some text content"


def test_text_loader_save_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """Text loader creates parent directories if needed."""
    txt_file = tmp_path / "nested" / "dir" / "output.txt"

    loader = loaders.Text()
    loader.save("content", txt_file)

    assert txt_file.exists()
    assert txt_file.read_text() == "content"


def test_text_loader_save_type_error() -> None:
    """Text loader raises TypeError for non-string data."""
    loader = loaders.Text()
    with pytest.raises(TypeError, match="Text save expects str"):
        loader.save(123, pathlib.Path("test.txt"))  # pyright: ignore[reportArgumentType]


def test_text_loader_empty() -> None:
    """Text loader returns empty string for empty()."""
    loader = loaders.Text()
    assert loader.empty() == ""


def test_text_loader_is_picklable() -> None:
    """Text loader can be pickled and unpickled."""
    loader = loaders.Text()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.Text)


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

    def empty(self) -> str:
        return ""


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


# ==============================================================================
# MatplotlibFigure loader tests
# ==============================================================================


def test_matplotlib_figure_loader_save(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader saves figure to file and closes it."""
    import matplotlib.pyplot as plt

    png_file = tmp_path / "plot.png"
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    loader = loaders.MatplotlibFigure()
    loader.save(fig, png_file)

    assert png_file.exists()
    assert png_file.stat().st_size > 0
    # Verify figure was closed (can't plot on closed figure)
    assert fig.number not in plt.get_fignums()  # pyright: ignore[reportAttributeAccessIssue]


def test_matplotlib_figure_loader_save_pdf(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader saves to PDF format based on extension."""
    import matplotlib.pyplot as plt

    pdf_file = tmp_path / "plot.pdf"
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [3, 1, 2])

    loader = loaders.MatplotlibFigure()
    loader.save(fig, pdf_file)

    assert pdf_file.exists()
    # PDF files start with %PDF
    content = pdf_file.read_bytes()
    assert content.startswith(b"%PDF")


def test_matplotlib_figure_loader_save_svg(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader saves to SVG format based on extension."""
    import matplotlib.pyplot as plt

    svg_file = tmp_path / "plot.svg"
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [3, 1, 2])

    loader = loaders.MatplotlibFigure()
    loader.save(fig, svg_file)

    assert svg_file.exists()
    content = svg_file.read_text()
    assert "<svg" in content


def test_matplotlib_figure_loader_custom_dpi(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader respects dpi option."""
    import matplotlib.pyplot as plt

    png_file_low = tmp_path / "low.png"
    png_file_high = tmp_path / "high.png"

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    loaders.MatplotlibFigure(dpi=50).save(fig, png_file_low)

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    loaders.MatplotlibFigure(dpi=300).save(fig, png_file_high)

    # Higher DPI should produce larger file
    assert png_file_high.stat().st_size > png_file_low.stat().st_size


def test_matplotlib_figure_loader_transparent(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader respects transparent option."""
    import matplotlib.pyplot as plt
    from PIL import Image

    png_file = tmp_path / "transparent.png"
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    loaders.MatplotlibFigure(transparent=True).save(fig, png_file)

    # Transparent PNG has alpha channel
    img = Image.open(png_file)
    assert img.mode == "RGBA"


def test_matplotlib_figure_loader_bbox_inches_none(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader respects bbox_inches=None option."""
    import matplotlib.pyplot as plt

    png_file_tight = tmp_path / "tight.png"
    png_file_none = tmp_path / "none.png"

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    loaders.MatplotlibFigure(bbox_inches="tight").save(fig, png_file_tight)

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    loaders.MatplotlibFigure(bbox_inches=None).save(fig, png_file_none)

    # Both should produce valid files (just different sizes)
    assert png_file_tight.exists()
    assert png_file_none.exists()


def test_matplotlib_figure_loader_dpi_validation_low() -> None:
    """MatplotlibFigure loader validates dpi is at least 1."""
    with pytest.raises(ValueError, match="dpi must be between 1 and 2400"):
        loaders.MatplotlibFigure(dpi=0)


def test_matplotlib_figure_loader_dpi_validation_high() -> None:
    """MatplotlibFigure loader validates dpi is at most 2400."""
    with pytest.raises(ValueError, match="dpi must be between 1 and 2400"):
        loaders.MatplotlibFigure(dpi=2401)


def test_matplotlib_figure_loader_load_raises() -> None:
    """MatplotlibFigure loader raises on load (write-only)."""
    loader = loaders.MatplotlibFigure()
    with pytest.raises(NotImplementedError, match="write-only"):
        loader.load(pathlib.Path("test.png"))


def test_matplotlib_figure_loader_empty_raises() -> None:
    """MatplotlibFigure loader raises on empty (no incremental support)."""
    loader = loaders.MatplotlibFigure()
    with pytest.raises(NotImplementedError, match="cannot provide an empty instance"):
        loader.empty()


def test_matplotlib_figure_loader_is_picklable() -> None:
    """MatplotlibFigure loader can be pickled and unpickled."""
    loader = loaders.MatplotlibFigure(dpi=200, bbox_inches=None, transparent=True)
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert restored.dpi == 200
    assert restored.bbox_inches is None
    assert restored.transparent is True


def test_matplotlib_figure_loader_closes_on_error(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure loader closes figure even when savefig fails."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    loader = loaders.MatplotlibFigure()
    # Try to save to a path with non-existent parent directory
    bad_path = tmp_path / "nonexistent" / "plot.png"

    with pytest.raises(FileNotFoundError):
        loader.save(fig, bad_path)

    # Figure should still be closed
    assert fig.number not in plt.get_fignums()  # pyright: ignore[reportAttributeAccessIssue]


def test_matplotlib_figure_loader_with_out(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure works with Out for plot outputs."""
    import matplotlib.pyplot as plt

    from pivot import outputs

    # Create an Out with MatplotlibFigure loader (like Plot but with auto-save)
    out = outputs.Out("plot.png", loaders.MatplotlibFigure(dpi=100))

    # Verify the loader is correctly stored
    assert isinstance(out.loader, loaders.MatplotlibFigure)
    assert out.loader.dpi == 100
    assert isinstance(out.path, str)

    # Create and save a figure using the loader
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    plot_file = tmp_path / out.path
    out.loader.save(fig, plot_file)

    assert plot_file.exists()
    assert fig.number not in plt.get_fignums()  # pyright: ignore[reportAttributeAccessIssue]


# ==============================================================================
# JSONL loader tests
# ==============================================================================


def test_jsonl_loader_load(tmp_path: pathlib.Path) -> None:
    """JSONL loader reads list of dicts from file."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1}\n{"b": 2}\n')

    loader = loaders.JSONL()
    data = loader.load(jsonl_file)

    assert data == [{"a": 1}, {"b": 2}]


def test_jsonl_loader_load_skips_blank_lines(tmp_path: pathlib.Path) -> None:
    """JSONL loader skips blank lines."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1}\n\n{"b": 2}\n   \n')

    loader = loaders.JSONL()
    data = loader.load(jsonl_file)

    assert data == [{"a": 1}, {"b": 2}]


def test_jsonl_loader_load_invalid_json(tmp_path: pathlib.Path) -> None:
    """JSONL loader reports line number on invalid JSON."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1}\n{invalid}\n')

    loader = loaders.JSONL()
    with pytest.raises(ValueError, match=r"Invalid JSON at .+:2"):
        loader.load(jsonl_file)


def test_jsonl_loader_save(tmp_path: pathlib.Path) -> None:
    """JSONL loader writes list of dicts to file."""
    jsonl_file = tmp_path / "output.jsonl"
    data = [{"x": 1}, {"y": 2}]

    loader = loaders.JSONL()
    loader.save(data, jsonl_file)

    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"x": 1}
    assert json.loads(lines[1]) == {"y": 2}


def test_jsonl_loader_save_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """JSONL loader creates parent directories if needed."""
    jsonl_file = tmp_path / "nested" / "output.jsonl"

    loader = loaders.JSONL()
    loader.save([{"a": 1}], jsonl_file)

    assert jsonl_file.exists()


def test_jsonl_loader_save_type_error() -> None:
    """JSONL loader raises TypeError for non-list data."""
    loader = loaders.JSONL()
    with pytest.raises(TypeError, match="JSONL save expects list"):
        loader.save({"a": 1}, pathlib.Path("test.jsonl"))  # pyright: ignore[reportArgumentType]


def test_jsonl_loader_empty() -> None:
    """JSONL loader returns empty list for empty()."""
    loader = loaders.JSONL()
    assert loader.empty() == []


def test_jsonl_loader_is_picklable() -> None:
    """JSONL loader can be pickled and unpickled."""
    loader = loaders.JSONL()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.JSONL)


# ==============================================================================
# DataFrameJSONL loader tests
# ==============================================================================


def test_dataframe_jsonl_loader_load(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader reads DataFrame from file."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n')

    loader = loaders.DataFrameJSONL()
    df = loader.load(jsonl_file)

    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert df["a"].tolist() == [1, 3]


def test_dataframe_jsonl_loader_save(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader writes DataFrame to file."""
    jsonl_file = tmp_path / "output.jsonl"
    df = pandas.DataFrame({"x": [1, 2], "y": [3, 4]})

    loader = loaders.DataFrameJSONL()
    loader.save(df, jsonl_file)

    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_dataframe_jsonl_loader_roundtrip(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader roundtrips data correctly."""
    jsonl_file = tmp_path / "data.jsonl"
    df = pandas.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    loader = loaders.DataFrameJSONL()
    loader.save(df, jsonl_file)
    loaded = loader.load(jsonl_file)

    pandas.testing.assert_frame_equal(df, loaded)


def test_dataframe_jsonl_loader_save_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """DataFrameJSONL loader creates parent directories if needed."""
    jsonl_file = tmp_path / "nested" / "output.jsonl"
    df = pandas.DataFrame({"a": [1]})

    loader = loaders.DataFrameJSONL()
    loader.save(df, jsonl_file)

    assert jsonl_file.exists()


def test_dataframe_jsonl_loader_save_type_error() -> None:
    """DataFrameJSONL loader raises TypeError for non-DataFrame data."""
    loader = loaders.DataFrameJSONL()
    with pytest.raises(TypeError, match="DataFrameJSONL save expects DataFrame"):
        loader.save([{"a": 1}], pathlib.Path("test.jsonl"))  # pyright: ignore[reportArgumentType]


def test_dataframe_jsonl_loader_empty() -> None:
    """DataFrameJSONL loader returns empty DataFrame for empty()."""
    loader = loaders.DataFrameJSONL()
    result = loader.empty()
    assert isinstance(result, pandas.DataFrame)
    assert len(result) == 0


def test_dataframe_jsonl_loader_is_picklable() -> None:
    """DataFrameJSONL loader can be pickled and unpickled."""
    loader = loaders.DataFrameJSONL()
    pickled = pickle.dumps(loader)
    restored = pickle.loads(pickled)

    assert isinstance(restored, loaders.DataFrameJSONL)
