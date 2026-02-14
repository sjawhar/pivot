# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false
from __future__ import annotations

from typing import Annotated, TypedDict

import pandas as pd
import pytest

from pivot import loaders
from pivot.compose import (
    ArtifactHandle,
    _analyze_return_type,
    _format_extension,
    _infer_format,
    _infer_format_from_extension,
    _InputNode,
    _MetricTag,
    _OutputSpec,
    _StageNode,
    metric,
    plot,
    stage,
)


def test_infer_format_dataframe() -> None:
    fmt = _infer_format(pd.DataFrame)
    assert isinstance(fmt, loaders.DataFrameJSONL)


def test_infer_format_dict() -> None:
    fmt = _infer_format(dict)
    assert isinstance(fmt, loaders.YAML)


def test_infer_format_str() -> None:
    fmt = _infer_format(str)
    assert isinstance(fmt, loaders.Text)


def test_format_extension_known_formats() -> None:
    assert _format_extension(loaders.DataFrameJSONL()) == "jsonl"
    assert _format_extension(loaders.CSV()) == "csv"
    assert _format_extension(loaders.YAML()) == "yaml"
    assert _format_extension(loaders.JSON()) == "json"
    assert _format_extension(loaders.Text()) == "txt"
    assert _format_extension(loaders.Pickle()) == "pkl"
    assert _format_extension(loaders.MatplotlibFigure()) == "png"


def test_format_extension_fallback() -> None:
    assert _format_extension(loaders.PathOnly()) == "dat"


def test_artifact_handle_suboutput() -> None:
    def stage_func() -> None:
        return None

    pipeline = object()
    output_specs = [
        _OutputSpec(key="filtered_runs", python_type=list, format=loaders.JSONL()),
        _OutputSpec(key="raw_runs", python_type=dict, format=loaders.JSON()),
    ]
    stage_node = _StageNode(
        func=stage_func,
        original_func=stage_func,
        name="train",
        params=None,
        input_handles={},
        output_specs=output_specs,
        call_index=0,
    )
    handle = ArtifactHandle(
        pipeline=pipeline,
        source=stage_node,
        output_key=None,
        python_type=dict,
    )

    sub_handle = handle.filtered_runs
    assert sub_handle._output_key == "filtered_runs"
    assert sub_handle._python_type is list


def test_artifact_handle_missing_output_raises() -> None:
    def stage_func() -> None:
        return None

    pipeline = object()
    output_specs = [
        _OutputSpec(key="filtered_runs", python_type=list, format=loaders.JSONL()),
    ]
    stage_node = _StageNode(
        func=stage_func,
        original_func=stage_func,
        name="train",
        params=None,
        input_handles={},
        output_specs=output_specs,
        call_index=0,
    )
    handle = ArtifactHandle(
        pipeline=pipeline,
        source=stage_node,
        output_key=None,
        python_type=dict,
    )

    with pytest.raises(AttributeError):
        _ = handle.unknown_output


def test_artifact_handle_input_no_suboutputs() -> None:
    pipeline = object()
    input_node = _InputNode(
        name="source",
        python_type=str,
        path="data/input.txt",
        format=loaders.Text(),
    )
    handle = ArtifactHandle(
        pipeline=pipeline,
        source=input_node,
        output_key=None,
        python_type=str,
    )

    with pytest.raises(AttributeError):
        _ = handle.anything


# --- @stage decorator ---


def test_stage_direct_execution() -> None:
    @stage
    def add_one(x: int) -> int:
        return x + 1

    assert add_one(5) == 6


def test_stage_preserves_metadata() -> None:
    @stage
    def my_func(x: pd.DataFrame) -> pd.DataFrame:
        return x

    assert my_func.__name__ == "my_func"
    assert my_func._is_stage is True  # pyright: ignore[reportAttributeAccessIssue]
    assert my_func._original_func is not my_func  # pyright: ignore[reportAttributeAccessIssue]


# --- Return type analysis ---


def _helper_single_output() -> pd.DataFrame: ...
def _helper_annotated_csv() -> Annotated[pd.DataFrame, loaders.CSV()]: ...
def _helper_metric_tag() -> Annotated[dict, metric]: ...
def _helper_plot_tag() -> Annotated[pd.DataFrame, plot]: ...


class _HelperMultiOutput(TypedDict):
    data: pd.DataFrame
    weights: Annotated[dict, metric]


def _helper_multi_output() -> _HelperMultiOutput: ...


def test_analyze_return_type_single() -> None:
    specs = _analyze_return_type(_helper_single_output)
    assert len(specs) == 1
    assert specs[0].python_type is pd.DataFrame
    assert isinstance(specs[0].format, loaders.DataFrameJSONL)
    assert specs[0].tag is None


def test_analyze_return_type_annotated_override() -> None:
    specs = _analyze_return_type(_helper_annotated_csv)
    assert len(specs) == 1
    assert isinstance(specs[0].format, loaders.CSV)


def test_analyze_return_type_metric_tag() -> None:
    specs = _analyze_return_type(_helper_metric_tag)
    assert len(specs) == 1
    assert isinstance(specs[0].tag, _MetricTag)


def test_analyze_return_type_plot_tag() -> None:
    specs = _analyze_return_type(_helper_plot_tag)
    assert len(specs) == 1
    assert specs[0].tag is plot


def test_analyze_return_type_typeddict() -> None:
    specs = _analyze_return_type(_helper_multi_output)
    assert len(specs) == 2
    assert specs[0].key == "data"
    assert specs[0].python_type is pd.DataFrame
    assert specs[0].tag is None
    assert specs[1].key == "weights"
    assert isinstance(specs[1].tag, _MetricTag)


# --- Extension-based format inference ---


@pytest.mark.parametrize(
    ("path", "expected_type"),
    [
        pytest.param("data/foo.yaml", loaders.YAML, id="yaml"),
        pytest.param("data/foo.yml", loaders.YAML, id="yml"),
        pytest.param("data/foo.csv", loaders.CSV, id="csv"),
        pytest.param("data/foo.jsonl", loaders.DataFrameJSONL, id="jsonl"),
        pytest.param("data/foo.json", loaders.JSON, id="json"),
        pytest.param("sql/foo.sql", loaders.Text, id="sql"),
        pytest.param("sql/foo.jinja", loaders.Text, id="jinja"),
        pytest.param("data/foo.txt", loaders.Text, id="txt"),
        pytest.param("data/foo.pkl", loaders.Pickle, id="pkl"),
        pytest.param("data/foo.unknown", loaders.PathOnly, id="unknown"),
    ],
)
def test_infer_format_from_extension(path: str, expected_type: type) -> None:
    result = _infer_format_from_extension(path)
    assert isinstance(result, expected_type)
