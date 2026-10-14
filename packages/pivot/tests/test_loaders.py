# pyright: reportAbstractUsage=false, reportImplicitAbstractClass=false, reportImplicitOverride=false, reportUnknownArgumentType=false

from __future__ import annotations

import dataclasses
import datetime
import decimal
import json
import math
import pathlib
import pickle
import typing
from typing import Any
from zoneinfo import ZoneInfo

import numpy
import pandas
import pytest
import pytz
import yaml
from dateutil import tz as dateutil_tz

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


def test_csv_loader_with_multi_index_col(tmp_path: pathlib.Path) -> None:
    """CSV loader supports list index_col for MultiIndex."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("idx1,idx2,val\na,x,1\nb,y,2\n")

    loader = loaders.CSV(index_col=[0, 1])
    df = loader.load(csv_file)

    assert df.index.names == ["idx1", "idx2"]
    assert list(df.index) == [("a", "x"), ("b", "y")]


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


def test_yaml_loader_roundtrips_numpy_values_as_plain_data(tmp_path: pathlib.Path) -> None:
    yaml_file = tmp_path / "metric.yaml"
    data: dict[str, object] = {
        "mean": numpy.float64(0.5),
        "count": numpy.int64(3),
        "valid": numpy.bool_(True),
        "samples": numpy.array([0.25, 0.75]),
    }

    loaders.YAML[dict[str, object]]().save(data, yaml_file)
    loaded = loaders.YAML[dict[str, object]]().load(yaml_file)

    assert loaded == {"mean": 0.5, "count": 3, "valid": True, "samples": [0.25, 0.75]}
    assert "python/" not in yaml_file.read_text()


def test_yaml_loader_writes_non_plain_data_as_before_with_a_warning(tmp_path: pathlib.Path) -> None:
    """Warn-first: the output keeps its previous bytes, and the warning says how to fix it."""
    yaml_file = tmp_path / "metric.yaml"
    data = {"frame": pandas.DataFrame({"value": [1.0]})}

    with pytest.warns(loaders.LossyOutputWarning, match=r"got DataFrame at root\['frame'\]"):
        loaders.YAML[object]().save(data, yaml_file)

    assert yaml_file.read_text() == yaml.dump(data, default_flow_style=False)


@dataclasses.dataclass
class _UnserializableMetric:
    value: float


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param(_UnserializableMetric(0.5), "_UnserializableMetric at root", id="dataclass"),
        pytest.param(
            {"value": numpy.datetime64("2024-01-01T00:00:00.000000001")},
            "datetime64",
            id="datetime64",
        ),
        pytest.param({"value": numpy.timedelta64(5, "ns")}, "timedelta64", id="timedelta64"),
        pytest.param(
            {"value": numpy.array(["2024-01-01T00:00:00.000000001"], dtype="datetime64[ns]")},
            "datetime64",
            id="datetime64-array",
        ),
        pytest.param({"value": numpy.longdouble("1.5")}, "longdouble", id="longdouble"),
        pytest.param({(1, 2): "pair"}, "tuple key at root", id="tuple-key"),
    ],
)
def test_yaml_loader_warns_for_values_safe_load_cannot_read(
    tmp_path: pathlib.Path, data: object, match: str
) -> None:
    yaml_file = tmp_path / "metric.yaml"

    with pytest.warns(loaders.LossyOutputWarning, match=match):
        loaders.YAML[object]().save(data, yaml_file)

    assert yaml_file.read_text() == yaml.dump(data, default_flow_style=False)


def test_yaml_loader_refuses_values_the_old_writer_also_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="generator"):
        loaders.YAML[object]().save({"rows": (n for n in range(2))}, tmp_path / "metric.yaml")


def test_pickle_loader_warns_about_cache_and_remote_trust(tmp_path: pathlib.Path) -> None:
    pickle_file = tmp_path / "artifact.pkl"
    pickle_file.write_bytes(pickle.dumps({"value": 1}))

    with pytest.warns(RuntimeWarning, match="cache or remote"):
        assert loaders.Pickle[dict[str, int]]().load(pickle_file) == {"value": 1}


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


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a\rb\r", id="cr"),
        pytest.param("a\r\nb\r\n", id="crlf"),
        pytest.param("a\nb\n", id="lf"),
        pytest.param("a\r\nb\rc\nd", id="mixed"),
    ],
)
def test_text_loader_roundtrips_line_endings_exactly(tmp_path: pathlib.Path, text: str) -> None:
    txt_file = tmp_path / "endings.txt"

    loaders.Text().save(text, txt_file)

    assert txt_file.read_bytes() == text.encode(), "save must not translate line endings"
    assert loaders.Text().load(txt_file) == text


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
    with pytest.warns(RuntimeWarning, match="cache or remote"):
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


def test_pathonly_loader_save_same_path_is_noop(tmp_path: pathlib.Path) -> None:
    """PathOnly save accepts a file already at the output path."""
    file = tmp_path / "output.bin"
    file.write_bytes(b"data")

    loader = loaders.PathOnly()
    loader.save(file, file)

    assert file.read_bytes() == b"data"


def test_pathonly_loader_save_raises_if_missing(tmp_path: pathlib.Path) -> None:
    """PathOnly save raises if the returned path doesn't exist, naming it."""
    missing = tmp_path / "missing.bin"

    loader = loaders.PathOnly()
    with pytest.raises(FileNotFoundError, match="missing.bin"):
        loader.save(missing, tmp_path / "stage.dat")


def test_pathonly_loader_save_moves_returned_file(tmp_path: pathlib.Path) -> None:
    """PathOnly save moves the file the stage returned to the output path."""
    returned = tmp_path / "scratch" / "report.bin"
    returned.parent.mkdir()
    returned.write_bytes(b"report")
    output = tmp_path / "data" / "stage.dat"

    loaders.PathOnly().save(returned, output)

    assert output.read_bytes() == b"report"
    assert not returned.exists(), "Returned file should be moved, not copied"


def test_pathonly_loader_save_moves_returned_directory(tmp_path: pathlib.Path) -> None:
    """PathOnly save moves a returned directory tree to the output path."""
    returned = tmp_path / "scratch" / "bundle"
    (returned / "sub").mkdir(parents=True)
    (returned / "a.txt").write_text("alpha")
    (returned / "sub" / "b.txt").write_text("beta")
    output = tmp_path / "data" / "stage.dat"

    loaders.PathOnly().save(returned, output)

    assert (output / "a.txt").read_text() == "alpha"
    assert (output / "sub" / "b.txt").read_text() == "beta"
    assert not returned.exists(), "Returned directory should be moved, not copied"


def test_pathonly_loader_save_refuses_existing_output(tmp_path: pathlib.Path) -> None:
    """PathOnly save never moves a path on top of an existing output."""
    returned = tmp_path / "report.bin"
    returned.write_bytes(b"new")
    output = tmp_path / "stage.dat"
    output.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="already exists"):
        loaders.PathOnly().save(returned, output)

    assert output.read_bytes() == b"old"
    assert returned.read_bytes() == b"new"


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


def test_matplotlib_figure_is_writer_only() -> None:
    """MatplotlibFigure is a Writer, not a Loader - no load() or empty() methods."""
    loader = loaders.MatplotlibFigure()
    # MatplotlibFigure is a Writer only - it has save() but not load() or empty()
    assert hasattr(loader, "save")
    assert not hasattr(loader, "load")
    assert not hasattr(loader, "empty")
    # Verify it's a Writer but not a Loader
    assert isinstance(loader, loaders.Writer)
    assert not isinstance(loader, loaders.Loader)


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


def test_matplotlib_figure_loader_saves_and_closes_figure(tmp_path: pathlib.Path) -> None:
    """MatplotlibFigure keeps its configured dpi, writes the figure, and closes it."""
    import matplotlib.pyplot as plt

    loader = loaders.MatplotlibFigure(dpi=100)
    assert loader.dpi == 100

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    plot_file = tmp_path / "plot.png"
    loader.save(fig, plot_file)

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
    assert len(lines) == 3
    assert json.loads(lines[0]) == {
        "__pivot_dataframe_metadata__": {
            "version": 1,
            "columns": [{"name": "x", "dtype": "int64"}, {"name": "y", "dtype": "int64"}],
            "temporal_columns": {},
        }
    }


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


# ==============================================================================
# Incremental first-run values
# ==============================================================================

_LOADERS_WITH_EMPTY_VALUES: list[loaders.Loader[Any, Any]] = [
    loaders.CSV(),
    loaders.JSON(),
    loaders.YAML(),
    loaders.Text(),
    loaders.JSONL(),
    loaders.DataFrameJSONL(),
]


def test_empty_value_round_trip_cases_cover_every_builtin_loader_with_empty() -> None:
    """Every built-in loader accepted for incremental outputs is checked below."""
    implemented = {
        value.__name__
        for value in vars(loaders).values()
        if isinstance(value, type)
        and issubclass(value, loaders.Loader)
        and value.empty is not loaders.Loader.empty
    }
    assert implemented == {type(loader).__name__ for loader in _LOADERS_WITH_EMPTY_VALUES}


@pytest.mark.parametrize(
    "loader",
    [pytest.param(loader, id=type(loader).__name__) for loader in _LOADERS_WITH_EMPTY_VALUES],
)
def test_incremental_loader_empty_value_round_trips(
    loader: loaders.Loader[Any, Any], tmp_path: pathlib.Path
) -> None:
    """An incremental first-run value can be saved and loaded on the next run."""
    path = tmp_path / "empty-value"
    empty = loader.empty()

    loader.save(empty, path)
    loaded = loader.load(path)

    if isinstance(empty, pandas.DataFrame):
        pandas.testing.assert_frame_equal(loaded, empty)
    else:
        assert loaded == empty


@pytest.mark.parametrize(
    ("content", "error"),
    [
        pytest.param('a,b\n1,"unterminated\n', pandas.errors.ParserError, id="unterminated-quote"),
        pytest.param("\n\n\n", pandas.errors.EmptyDataError, id="multiple-blank-lines"),
    ],
)
def test_csv_loader_rejects_files_other_than_the_empty_frame_encoding(
    content: str, error: type[Exception], tmp_path: pathlib.Path
) -> None:
    """Only the single line break written for a zero-column frame loads as empty."""
    path = tmp_path / "bad.csv"
    path.write_text(content)

    with pytest.raises(error):
        loaders.CSV().load(path)


def test_dataframe_jsonl_roundtrips_date_and_time_columns(tmp_path: pathlib.Path) -> None:
    """Date, datetime, tz-aware, NaT and timedelta values come back as datetimes, not ints."""
    jsonl_file = tmp_path / "dates.jsonl"
    df = pandas.DataFrame(
        {
            "date": [datetime.date(2024, 3, 4), datetime.date(2024, 3, 5), None],
            "ns": pandas.to_datetime(
                pandas.Series(
                    ["2024-03-04 12:34:56.123456789", "2024-03-05 00:00:00.000000001", None]
                )
            ),
            "tz": pandas.to_datetime(
                pandas.Series(["2024-03-04 12:00", "2024-03-05 08:00", None])
            ).dt.tz_localize("America/New_York"),
            "delta": pandas.to_timedelta(
                pandas.Series(["-1 days +00:00:01", "2 days 00:00:00.000000001", None])
            ),
        }
    )

    loaders.DataFrameJSONL().save(df, jsonl_file)
    jsonl_records = [json.loads(line) for line in jsonl_file.read_text().splitlines()]
    assert jsonl_records[0] == {
        "__pivot_dataframe_metadata__": {
            "version": 1,
            "columns": [
                {"name": "date", "dtype": "datetime64[ns]"},
                {"name": "ns", "dtype": "datetime64[ns]"},
                {"name": "tz", "dtype": "datetime64[ns, America/New_York]"},
                {"name": "delta", "dtype": "timedelta64[ns]"},
            ],
            "temporal_columns": {
                "date": {"kind": "datetime", "timezone": None},
                "ns": {"kind": "datetime", "timezone": None},
                "tz": {"kind": "datetime", "timezone": "America/New_York"},
                "delta": {"kind": "timedelta"},
            },
        }
    }
    assert jsonl_records[1]["date"] == "2024-03-04T00:00:00.000000000"
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    expected = pandas.DataFrame(
        {
            "date": pandas.to_datetime(pandas.Series(["2024-03-04", "2024-03-05", None])),
            "ns": df["ns"],
            "tz": df["tz"],
            "delta": df["delta"],
        }
    )
    pandas.testing.assert_frame_equal(loaded, expected)


def test_dataframe_jsonl_roundtrips_floats_exactly(tmp_path: pathlib.Path) -> None:
    """Float values come back bit-identical, including inside list cells."""
    jsonl_file = tmp_path / "floats.jsonl"
    values = [0.3, 0.35, 0.7, 0.1 + 0.2, 1 / 3, 1e-12, 2.0**-30, 123456789.123456789, math.nan]
    df = pandas.DataFrame(
        {
            "x": values,
            "whole": [float(i) for i in range(len(values))],
            "nested": [[v] for v in values[:-1]] + [None],
        }
    )

    loaders.DataFrameJSONL().save(df, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    pandas.testing.assert_frame_equal(loaded, df, check_exact=True)
    assert [f"run_{v}" for v in loaded["x"].tolist()[:3]] == ["run_0.3", "run_0.35", "run_0.7"]


def test_dataframe_jsonl_keeps_numeric_looking_strings_as_strings(tmp_path: pathlib.Path) -> None:
    jsonl_file = tmp_path / "strings.jsonl"
    df = pandas.DataFrame({"zip": ["02134", "10001", "00501"], "code": ["1", "2", "3"]})

    loaders.DataFrameJSONL().save(df, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    pandas.testing.assert_frame_equal(loaded, df)


def test_dataframe_jsonl_keeps_iso_shaped_strings_as_strings(tmp_path: pathlib.Path) -> None:
    jsonl_file = tmp_path / "iso-strings.jsonl"
    df = pandas.DataFrame(
        {
            "timestamp_text": [
                "2024-03-04T12:34:56.123456789",
                "2024-03-05T00:00:00.000000000",
            ]
        }
    )

    loaders.DataFrameJSONL().save(df, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    pandas.testing.assert_frame_equal(loaded, df)


def test_dataframe_jsonl_keeps_columns_of_zero_row_frame(tmp_path: pathlib.Path) -> None:
    frame = pandas.DataFrame(
        {"count": pandas.Series([], dtype="int64"), "label": pandas.Series([], dtype=object)}
    )
    loaders.DataFrameJSONL().save(frame, tmp_path / "out.jsonl")
    loaded = loaders.DataFrameJSONL().load(tmp_path / "out.jsonl")
    assert isinstance(loaded, pandas.DataFrame)
    assert list(loaded.columns) == ["count", "label"], "a filtered-to-empty frame keeps its schema"


def test_dataframe_jsonl_roundtrips_zero_row_frame_dtypes(tmp_path: pathlib.Path) -> None:
    jsonl_file = tmp_path / "empty.jsonl"
    frame = pandas.DataFrame(
        {
            "label": pandas.Series([], dtype=object),
            "count": pandas.Series([], dtype="int64"),
            "score": pandas.Series([], dtype="float32"),
            "flag": pandas.Series([], dtype="bool"),
            "when": pandas.Series([], dtype="datetime64[ns]"),
            "local": pandas.Series([], dtype=pandas.DatetimeTZDtype(tz="America/New_York")),
            "elapsed": pandas.Series([], dtype="timedelta64[ns]"),
        }
    )

    loaders.DataFrameJSONL().save(frame, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    pandas.testing.assert_frame_equal(loaded, frame)


def test_dataframe_jsonl_roundtrips_all_null_temporal_dtypes(tmp_path: pathlib.Path) -> None:
    jsonl_file = tmp_path / "all-null-temporal.jsonl"
    df = pandas.DataFrame(
        {
            "naive": pandas.Series([pandas.NaT, pandas.NaT], dtype="datetime64[ns]"),
            "tz": pandas.Series(
                [pandas.NaT, pandas.NaT], dtype=pandas.DatetimeTZDtype(tz="America/New_York")
            ),
            "delta": pandas.Series([pandas.NaT, pandas.NaT], dtype="timedelta64[ns]"),
        }
    )

    loaders.DataFrameJSONL().save(df, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    pandas.testing.assert_frame_equal(loaded, df)


def test_dataframe_jsonl_roundtrips_all_nan_float_dtype(tmp_path: pathlib.Path) -> None:
    jsonl_file = tmp_path / "all-nan-float.jsonl"
    frame = pandas.DataFrame({"score": pandas.Series([math.nan, math.nan], dtype="float64")})

    loaders.DataFrameJSONL().save(frame, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    pandas.testing.assert_frame_equal(loaded, frame)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(numpy.array([numpy.longdouble("1.0000000000000000009")]), id="float128"),
        pytest.param(pandas.period_range("2024-01", periods=2, freq="M"), id="period"),
        pytest.param([(n for n in range(2)), None], id="generator"),
    ],
)
def test_dataframe_jsonl_refuses_columns_the_old_writer_also_refused(
    tmp_path: pathlib.Path, values: object
) -> None:
    df = pandas.DataFrame({"x": values})

    with pytest.raises(TypeError, match="DataFrameJSONL cannot write column 'x'"):
        loaders.DataFrameJSONL().save(df, tmp_path / "refused.jsonl")

    assert not (tmp_path / "refused.jsonl").exists()


@pytest.mark.parametrize(
    "timezone",
    [
        pytest.param(dateutil_tz.tzoffset(None, 5 * 3600), id="dateutil-offset"),
        pytest.param(dateutil_tz.tzutc(), id="dateutil-utc"),
        pytest.param(dateutil_tz.gettz("Europe/London"), id="dateutil-tzfile"),
    ],
)
def test_dataframe_jsonl_writes_timezone_that_cannot_roundtrip_as_before_with_a_warning(
    tmp_path: pathlib.Path, timezone: datetime.tzinfo | None
) -> None:
    """Warn-first: the frame keeps the old headerless epoch-millisecond form."""
    assert timezone is not None
    jsonl_file = tmp_path / "dateutil.jsonl"
    frame = pandas.DataFrame(
        {
            "timestamp": pandas.to_datetime(
                pandas.Series(["2024-03-04 12:00", None])
            ).dt.tz_localize(timezone)
        }
    )

    with pytest.warns(loaders.LossyOutputWarning, match="column 'timestamp'.*timezone"):
        loaders.DataFrameJSONL().save(frame, jsonl_file)

    assert jsonl_file.read_text() == frame.to_json(lines=True, orient="records")


@pytest.mark.parametrize(
    "timezone",
    [
        pytest.param(ZoneInfo("Europe/Paris"), id="zoneinfo"),
        pytest.param(pytz.timezone("America/New_York"), id="pytz"),
        pytest.param(datetime.UTC, id="utc"),
    ],
)
def test_dataframe_jsonl_roundtrips_reconstructible_timezones(
    tmp_path: pathlib.Path, timezone: datetime.tzinfo
) -> None:
    jsonl_file = tmp_path / "timezone.jsonl"
    frame = pandas.DataFrame(
        {
            "timestamp": pandas.to_datetime(
                pandas.Series(["2024-03-04 12:00", "2024-08-04 12:00"])
            ).dt.tz_localize(timezone)
        }
    )

    loaders.DataFrameJSONL().save(frame, jsonl_file)
    loaded = loaders.DataFrameJSONL().load(jsonl_file)
    loaded_dtype = typing.cast("pandas.DatetimeTZDtype", loaded["timestamp"].dtype)
    frame_dtype = typing.cast("pandas.DatetimeTZDtype", frame["timestamp"].dtype)

    assert str(loaded_dtype.tz) == str(frame_dtype.tz)
    assert [timestamp.utcoffset() for timestamp in loaded["timestamp"].tolist()] == [
        timestamp.utcoffset() for timestamp in frame["timestamp"].tolist()
    ]


@dataclasses.dataclass
class _Cell:
    x: int


@pytest.mark.parametrize(
    ("cell", "type_name"),
    [
        pytest.param(decimal.Decimal("0.1"), "Decimal", id="decimal"),
        pytest.param({1, 2}, "set", id="set"),
        pytest.param(b"raw", "bytes", id="bytes"),
        pytest.param(_Cell(1), "_Cell", id="dataclass"),
        pytest.param(complex(1, 2), "complex", id="complex"),
    ],
)
def test_dataframe_jsonl_writes_cells_json_cannot_hold_as_before_with_a_warning(
    tmp_path: pathlib.Path, cell: object, type_name: str
) -> None:
    jsonl_file = tmp_path / "lossy.jsonl"
    df = pandas.DataFrame({"col": [cell, None], "n": [1, 2]})

    with pytest.warns(loaders.LossyOutputWarning, match=rf"column 'col'.*{type_name}"):
        loaders.DataFrameJSONL().save(df, jsonl_file)

    assert jsonl_file.read_text() == df.to_json(lines=True, orient="records")


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pandas.DataFrame({"__pivot_dataframe_metadata__": [1, 2]}), id="only"),
        pytest.param(
            pandas.DataFrame({"__pivot_dataframe_metadata__": ["a", "b"], "n": [1, 2]}),
            id="with-other",
        ),
        pytest.param(
            pandas.DataFrame({"__pivot_dataframe_metadata__": pandas.Series([], dtype="int64")}),
            id="zero-rows",
        ),
    ],
)
def test_dataframe_jsonl_roundtrips_a_column_named_like_the_header(
    tmp_path: pathlib.Path, frame: pandas.DataFrame
) -> None:
    jsonl_file = tmp_path / "header-named.jsonl"

    loaders.DataFrameJSONL().save(frame, jsonl_file)

    pandas.testing.assert_frame_equal(loaders.DataFrameJSONL().load(jsonl_file), frame)
    assert loaders.DataFrameJSONL().row_count(jsonl_file) == len(frame)


def test_dataframe_jsonl_reads_a_headerless_first_row_named_like_the_header(
    tmp_path: pathlib.Path,
) -> None:
    """Only a first line with the exact header shape is the header; anything else is a row."""
    jsonl_file = tmp_path / "old-format.jsonl"
    jsonl_file.write_text(
        '{"__pivot_dataframe_metadata__":1}\n{"__pivot_dataframe_metadata__":2}\n'
    )

    loaded = loaders.DataFrameJSONL().load(jsonl_file)

    assert loaded["__pivot_dataframe_metadata__"].tolist() == [1, 2]
    assert loaders.DataFrameJSONL().row_count(jsonl_file) == 2


def test_dataframe_jsonl_refuses_an_unknown_header_version(tmp_path: pathlib.Path) -> None:
    jsonl_file = tmp_path / "future.jsonl"
    header = {"version": 2, "columns": [], "temporal_columns": {}}
    jsonl_file.write_text(json.dumps({"__pivot_dataframe_metadata__": header}) + "\n")

    with pytest.raises(ValueError, match="Unsupported DataFrameJSONL metadata version"):
        loaders.DataFrameJSONL().load(jsonl_file)
