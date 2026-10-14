"""Property-based round-trip tests for the built-in loaders.

The contract a stage author relies on: what a stage returns is what the next stage loads,
``load(save(x)) == x``. Inputs a loader legitimately cannot represent are excluded from the
strategies, and each exclusion is named beside its strategy.
"""

from __future__ import annotations

import datetime
import pathlib
import tempfile
from typing import Any

import hypothesis
import hypothesis.extra.pandas as hpd
import numpy
import pandas
import pytest
from hypothesis import strategies as st

from pivot import loaders

_SETTINGS = hypothesis.settings(max_examples=100, deadline=None, derandomize=True, database=None)


def _round_trip(loader: loaders.Loader[Any, Any], value: object, filename: str) -> object:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / filename
        loader.save(value, path)
        return loader.load(path)


# NaN is excluded: NaN != NaN, so equality cannot express the property for it.
_json_scalars = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text()
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=12,
)

# Aware datetimes use whole-minute fixed offsets. Excluded: sub-minute offsets (historic LMT
# zones such as Africa/Abidjan before 1912), which PyYAML writes as "+00:16:08" and safe_load
# then cannot parse; and IANA zones, which load back as fixed offsets, so PEP 495 makes an
# ambiguous wall time compare unequal to the original.
_fixed_offsets = st.integers(min_value=-(23 * 60 + 59), max_value=23 * 60 + 59).map(
    lambda minutes: datetime.timezone(datetime.timedelta(minutes=minutes))
)
_yaml_scalars = (
    _json_scalars
    | st.dates()
    | st.datetimes()
    | st.datetimes(
        min_value=datetime.datetime(1900, 1, 1),
        max_value=datetime.datetime(2200, 1, 1),
        timezones=_fixed_offsets,
    )
)
_yaml_values = st.recursive(
    _yaml_scalars,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=12,
)


@_SETTINGS
@hypothesis.given(st.text(alphabet=st.characters(exclude_characters="\r")))
def test_text_round_trips_strings_without_carriage_returns(value: str) -> None:
    assert _round_trip(loaders.Text(), value, "out.txt") == value


@pytest.mark.xfail(strict=True, reason="red test for Task 10: Text.load translates newlines")
@pytest.mark.parametrize("value", ["line\r\nnext", "trailing\r"])
def test_text_round_trips_carriage_returns(value: str) -> None:
    assert _round_trip(loaders.Text(), value, "out.txt") == value


@_SETTINGS
@hypothesis.given(_json_values)
def test_json_round_trips_nested_values(value: object) -> None:
    assert _round_trip(loaders.JSON(), value, "out.json") == value


@_SETTINGS
@hypothesis.given(st.lists(st.dictionaries(st.text(), _json_values, max_size=4), max_size=4))
def test_jsonl_round_trips_records(value: list[dict[str, Any]]) -> None:
    assert _round_trip(loaders.JSONL(), value, "out.jsonl") == value


@_SETTINGS
@hypothesis.given(_yaml_values)
def test_yaml_round_trips_plain_values_dates_and_aware_datetimes(value: object) -> None:
    assert _round_trip(loaders.YAML(), value, "out.yaml") == value


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 10 (loader-fixes Slice 2 PR): "
    + "YAML.save writes python/* tags that YAML.load (safe_load) refuses",
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"n": numpy.int64(5)}, {"n": 5}, id="numpy-int64"),
        pytest.param({"mean": numpy.float64(0.5)}, {"mean": 0.5}, id="numpy-float64"),
        pytest.param({"pair": (1, 2)}, {"pair": [1, 2]}, id="tuple"),
    ],
)
def test_yaml_saves_numpy_scalars_and_tuples_as_loadable_plain_data(
    value: dict[str, object], expected: dict[str, object]
) -> None:
    assert _round_trip(loaders.YAML(), value, "out.yaml") == expected


_pickle_values = st.recursive(
    _json_scalars | st.binary() | st.dates() | st.datetimes() | st.complex_numbers(allow_nan=False),
    lambda children: st.lists(children, max_size=4)
    | st.tuples(children, children)
    | st.dictionaries(st.text(), children, max_size=4)
    | st.frozensets(st.integers(), max_size=4),
    max_leaves=12,
)


@_SETTINGS
@hypothesis.given(_pickle_values)
def test_pickle_round_trips_python_values(value: object) -> None:
    assert _round_trip(loaders.Pickle(), value, "out.pkl") == value


# Zero-row frames are covered by their own test below.
@_SETTINGS
@hypothesis.given(
    hpd.data_frames(
        columns=[
            hpd.column("count", dtype="int64"),
            hpd.column("flag", dtype=bool),
            hpd.column("label", dtype=object, elements=st.text()),
        ],
        index=hpd.range_indexes(min_size=1, max_size=6),
    )
)
def test_dataframe_jsonl_round_trips_int_bool_and_text_columns(frame: pandas.DataFrame) -> None:
    loaded = _round_trip(loaders.DataFrameJSONL(), frame, "out.jsonl")
    assert isinstance(loaded, pandas.DataFrame)
    pandas.testing.assert_frame_equal(loaded, frame)


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 10 (loader-fixes Slice 2 PR): "
    + "DataFrameJSONL.save rounds floats to 10 digits and int-like floats load as int64",
)
@_SETTINGS
@hypothesis.given(
    hpd.data_frames(
        columns=[
            hpd.column(
                "value",
                dtype="float64",
                elements=st.floats(allow_nan=True, allow_infinity=False),
            )
        ],
        index=hpd.range_indexes(min_size=1, max_size=6),
    )
)
def test_dataframe_jsonl_round_trips_float_columns(frame: pandas.DataFrame) -> None:
    loaded = _round_trip(loaders.DataFrameJSONL(), frame, "out.jsonl")
    assert isinstance(loaded, pandas.DataFrame)
    pandas.testing.assert_frame_equal(loaded, frame)


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 10 (loader-fixes Slice 2 PR): "
    + "DataFrameJSONL writes datetimes as epoch milliseconds (#451)",
)
@_SETTINGS
@hypothesis.given(
    hpd.data_frames(
        columns=[
            hpd.column(
                "when",
                dtype="datetime64[ns]",
                elements=st.datetimes(
                    min_value=datetime.datetime(1900, 1, 1),
                    max_value=datetime.datetime(2200, 1, 1),
                ),
            )
        ],
        index=hpd.range_indexes(min_size=1, max_size=5),
    )
)
def test_dataframe_jsonl_round_trips_datetime_columns(frame: pandas.DataFrame) -> None:
    loaded = _round_trip(loaders.DataFrameJSONL(), frame, "out.jsonl")
    assert isinstance(loaded, pandas.DataFrame)
    pandas.testing.assert_frame_equal(loaded, frame)


@pytest.mark.xfail(
    strict=True,
    reason="red test for Task 10 (loader-fixes Slice 2 PR): "
    + "DataFrameJSONL drops the columns of a zero-row frame",
)
def test_dataframe_jsonl_keeps_columns_of_zero_row_frame() -> None:
    frame = pandas.DataFrame(
        {"count": pandas.Series([], dtype="int64"), "label": pandas.Series([], dtype=object)}
    )
    loaded = _round_trip(loaders.DataFrameJSONL(), frame, "out.jsonl")
    assert isinstance(loaded, pandas.DataFrame)
    assert list(loaded.columns) == ["count", "label"], "a filtered-to-empty frame keeps its schema"


# Text cells are prefixed so they are never empty (CSV writes "" as an empty field, which
# read_csv reads as NaN) and never parse as numbers; NUL is excluded because pandas' C parser
# truncates a field at NUL.
_csv_text = st.text(alphabet=st.characters(exclude_characters="\x00")).map(lambda s: "s" + s)


@_SETTINGS
@hypothesis.given(
    hpd.data_frames(
        columns=[
            hpd.column("count", dtype="int64"),
            hpd.column(
                "value",
                dtype="float64",
                elements=st.floats(allow_nan=True, allow_infinity=False),
            ),
            hpd.column("label", dtype=object, elements=_csv_text),
            hpd.column("flag", dtype=bool),
        ],
        index=hpd.range_indexes(min_size=1, max_size=6),
    )
)
def test_csv_round_trips_typed_columns(frame: pandas.DataFrame) -> None:
    loaded = _round_trip(loaders.CSV(), frame, "out.csv")
    assert isinstance(loaded, pandas.DataFrame)
    pandas.testing.assert_frame_equal(loaded, frame)


def test_csv_empty_value_round_trips() -> None:
    """An incremental CSV output's first-run value can be read back on the next run."""
    loader = loaders.CSV[pandas.DataFrame]()
    loaded = _round_trip(loader, loader.empty(), "out.csv")
    assert isinstance(loaded, pandas.DataFrame)
    pandas.testing.assert_frame_equal(loaded, pandas.DataFrame())
