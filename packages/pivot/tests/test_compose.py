# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false
from __future__ import annotations

import inspect
import pathlib
import typing
from typing import Annotated, TypedDict

import pandas as pd
import pytest

from pivot import loaders, outputs, stage_def
from pivot.compose import (
    ArtifactHandle,
    Pipeline,
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

    stage_func = typing.cast("typing.Any", my_func)

    assert my_func.__name__ == "my_func"
    assert stage_func._is_stage is True
    assert stage_func._original_func is not my_func


# --- Return type analysis ---


def _helper_single_output() -> pd.DataFrame: ...
def _helper_annotated_csv() -> Annotated[pd.DataFrame, loaders.CSV()]: ...
def _helper_metric_tag() -> Annotated[dict, metric]: ...
def _helper_plot_tag() -> Annotated[pd.DataFrame, plot]: ...


class _HelperMultiOutput(TypedDict):
    data: pd.DataFrame
    weights: Annotated[dict, metric]


def _helper_multi_output() -> _HelperMultiOutput: ...


@stage
def _helper_produce(params: stage_def.StageParams) -> pd.DataFrame:
    return pd.DataFrame()


@stage
def _helper_consume(data: pd.DataFrame) -> dict:
    return {}


@stage
def _helper_repeat(params: stage_def.StageParams) -> dict:
    return {}


@stage
def _helper_consume_dict(data: dict) -> dict:
    return data


@stage
def _helper_build_stage_a(
    params: stage_def.StageParams,
    raw: dict,
) -> pd.DataFrame:
    return pd.DataFrame()


@stage
def _helper_build_stage_b(
    data: pd.DataFrame,
) -> dict:
    return {"rows": len(data)}


class _HelperOptionalParams(stage_def.StageParams):
    x: int = 1


@stage
def _helper_optional_params_stage(
    data: pd.DataFrame,
    params: _HelperOptionalParams | None = None,
) -> pd.DataFrame:
    return data


@stage
def _helper_typing_optional_params_stage(
    data: pd.DataFrame,
    params: typing.Optional[_HelperOptionalParams] = None,  # noqa: UP045 - intentionally testing typing.Union path
) -> pd.DataFrame:
    return data


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


# --- Bug 1: p.input() t= parameter format inference ---


def test_input_t_pathlib_path_uses_pathonly_regardless_of_extension() -> None:
    """t=pathlib.Path should use PathOnly, not extension-inferred CSV."""
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        pipeline.input("data.csv", t=pathlib.Path, path="data/external/data.csv")

    node = pipeline._inputs["data.csv"]
    assert isinstance(node.format, loaders.PathOnly), (
        f"Expected PathOnly for t=pathlib.Path, got {type(node.format).__name__}"
    )


def test_input_t_str_uses_text_regardless_of_extension() -> None:
    """t=str should use Text, not extension-inferred CSV."""
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        pipeline.input("query.csv", t=str, path="data/external/query.csv")

    node = pipeline._inputs["query.csv"]
    assert isinstance(node.format, loaders.Text), (
        f"Expected Text for t=str, got {type(node.format).__name__}"
    )


def test_input_t_dataframe_still_uses_extension_inference() -> None:
    """t=pd.DataFrame should NOT override extension inference (CSV stays CSV)."""
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        pipeline.input("data.csv", t=pd.DataFrame, path="data/external/data.csv")

    node = pipeline._inputs["data.csv"]
    assert isinstance(node.format, loaders.CSV), (
        f"Expected CSV from extension for t=DataFrame, got {type(node.format).__name__}"
    )


def test_input_explicit_format_takes_priority_over_t() -> None:
    """Explicit format= parameter should override both t and extension."""
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        pipeline.input(
            "data.csv",
            t=pathlib.Path,
            path="data/external/data.csv",
            format=loaders.CSV(),
        )

    node = pipeline._inputs["data.csv"]
    assert isinstance(node.format, loaders.CSV), (
        f"Expected CSV from explicit format, got {type(node.format).__name__}"
    )


def test_input_no_t_uses_extension_inference() -> None:
    """Without t parameter, extension inference should work as before."""
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        pipeline.input("data.csv", path="data/raw/data.csv")

    node = pipeline._inputs["data.csv"]
    assert isinstance(node.format, loaders.CSV)


def test_pipeline_context_basic() -> None:
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        data = _helper_produce(params=stage_def.StageParams())
        _helper_consume(data)

    assert len(pipeline._stages) == 2
    assert pipeline._stages[0].name == "_helper_produce"
    assert pipeline._stages[1].name == "_helper_consume"
    assert "data" in pipeline._stages[1].input_handles


def test_pipeline_disambiguation() -> None:
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        _helper_repeat(params=stage_def.StageParams())
        _helper_repeat(params=stage_def.StageParams())
        _helper_repeat(params=stage_def.StageParams())

    assert pipeline._stages[0].name == "_helper_repeat"
    assert pipeline._stages[1].name == "_helper_repeat@1"
    assert pipeline._stages[2].name == "_helper_repeat@2"


def test_pipeline_input() -> None:
    with Pipeline("test", root=pathlib.Path("/tmp")) as pipeline:
        raw = pipeline.input("raw_data", path="data/raw/input.yaml", t=dict)
        _helper_consume_dict(raw)

    assert "raw_data" in pipeline._inputs
    assert "data" in pipeline._stages[0].input_handles


def test_pipeline_build_bridge(tmp_path: pathlib.Path) -> None:
    with Pipeline("compose_build", root=tmp_path) as pipeline:
        raw = pipeline.input("raw", path="data/raw.yaml", t=dict)
        data = _helper_build_stage_a(params=stage_def.StageParams(), raw=raw)
        _helper_build_stage_b(data)

    legacy = pipeline.build()

    assert legacy.list_stages() == ["_helper_build_stage_a", "_helper_build_stage_b"]

    stage_a = legacy.get("_helper_build_stage_a")
    stage_b = legacy.get("_helper_build_stage_b")

    assert stage_a["func"] is inspect.unwrap(_helper_build_stage_a)
    assert not hasattr(stage_a["func"], "_is_stage")
    assert stage_a["name"] == "_helper_build_stage_a"
    assert stage_a["deps"] == {"raw": "data/raw.yaml"}
    assert stage_a["deps_paths"] == ["data/raw.yaml"]
    assert stage_a["params"] is not None
    assert isinstance(stage_a["params"], stage_def.StageParams)
    assert stage_a["params_arg_name"] == "params"
    assert stage_a["mutex"] == []
    assert stage_a["variant"] is None
    assert stage_a["signature"] == inspect.signature(inspect.unwrap(_helper_build_stage_a))
    assert stage_a["state_dir"] == tmp_path / ".pivot"

    output_a_path = "data/compose_build/_helper_build_stage_a.jsonl"
    assert stage_a["outs_paths"] == [output_a_path]
    assert len(stage_a["outs"]) == 1
    assert isinstance(stage_a["outs"][0], outputs.Out)
    assert stage_a["outs"][0].path == output_a_path
    assert isinstance(stage_a["out_specs"][stage_def.SINGLE_OUTPUT_KEY], outputs.Out)
    assert stage_a["out_specs"][stage_def.SINGLE_OUTPUT_KEY].path == output_a_path

    dep_spec_a = stage_a["dep_specs"]["raw"]
    assert dep_spec_a.path == "data/raw.yaml"
    assert isinstance(dep_spec_a.loader, loaders.YAML)
    assert dep_spec_a.creates_dep_edge is True

    output_b_path = "data/compose_build/_helper_build_stage_b.yaml"
    assert stage_b["deps"] == {"data": output_a_path}
    assert stage_b["deps_paths"] == [output_a_path]
    assert stage_b["outs_paths"] == [output_b_path]
    assert isinstance(stage_b["out_specs"][stage_def.SINGLE_OUTPUT_KEY], outputs.Out)
    assert stage_b["out_specs"][stage_def.SINGLE_OUTPUT_KEY].path == output_b_path

    dep_spec_b = stage_b["dep_specs"]["data"]
    assert dep_spec_b.path == output_a_path
    assert dep_spec_b.loader == stage_a["out_specs"][stage_def.SINGLE_OUTPUT_KEY].loader


# --- Bug 2: Optional[StageParams] params detection ---


def test_build_rejects_stage_params_in_pipe_union(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        raw = pipeline.input("raw", path="data/raw.csv", t=pd.DataFrame)
        _helper_optional_params_stage(data=raw, params=_HelperOptionalParams(x=2))

    with pytest.raises(TypeError, match="StageParams must not be in a union"):
        pipeline.build()


def test_build_rejects_stage_params_in_typing_optional(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        raw = pipeline.input("raw", path="data/raw.csv", t=pd.DataFrame)
        _helper_typing_optional_params_stage(data=raw, params=_HelperOptionalParams(x=3))

    with pytest.raises(TypeError, match="StageParams must not be in a union"):
        pipeline.build()
