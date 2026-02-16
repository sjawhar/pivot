# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import inspect
import pathlib
import typing
from typing import Annotated, TypedDict

import pandas as pd
import pytest

from pivot import loaders, outputs, stage_def, types
from pivot.compose import (
    SINGLE_OUTPUT_KEY,
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

    pipeline = Pipeline("test", root=pathlib.Path("/tmp"))
    output_specs = [
        _OutputSpec(
            key="filtered_runs",
            python_type=list,
            format=typing.cast("loaders.Writer[object]", loaders.JSONL()),
        ),
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

    pipeline = Pipeline("test", root=pathlib.Path("/tmp"))
    output_specs = [
        _OutputSpec(
            key="filtered_runs",
            python_type=list,
            format=typing.cast("loaders.Writer[object]", loaders.JSONL()),
        ),
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
    pipeline = Pipeline("test", root=pathlib.Path("/tmp"))
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


class _HelperSingleFieldTypedDict(TypedDict):
    result: pd.DataFrame


@stage
def _helper_single_field_typeddict() -> _HelperSingleFieldTypedDict:
    return {"result": pd.DataFrame()}


class _HelperSingleMetricTypedDict(TypedDict):
    accuracy: Annotated[dict, metric]


class _HelperReservedKeyTypedDict(TypedDict):
    _single: str


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


def test_analyze_return_type_single_field_typeddict_with_metric() -> None:
    def _helper() -> _HelperSingleMetricTypedDict: ...

    specs = _analyze_return_type(_helper)
    assert len(specs) == 1
    assert specs[0].key == "accuracy"
    assert isinstance(specs[0].tag, _MetricTag)


def test_analyze_return_type_rejects_reserved_key() -> None:
    def _helper() -> _HelperReservedKeyTypedDict: ...

    with pytest.raises(ValueError, match=f"{SINGLE_OUTPUT_KEY!r} is reserved"):
        _analyze_return_type(_helper)


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
    assert stage_a["deps"]["raw"].identity == types.ArtifactIdentity(producer="raw", key=None)
    assert isinstance(stage_a["deps"]["raw"].format, loaders.YAML)
    assert stage_a["deps"]["raw"].tag is types.ArtifactTag.DATA
    assert stage_a["params"] is not None
    assert isinstance(stage_a["params"], stage_def.StageParams)
    assert stage_a["params_arg_name"] == "params"
    assert stage_a["mutex"] == []
    assert stage_a["variant"] is None
    assert stage_a["signature"] == inspect.signature(inspect.unwrap(_helper_build_stage_a))
    assert stage_a["state_dir"] == tmp_path / ".pivot"

    assert len(stage_a["outs"]) == 1
    assert isinstance(stage_a["outs"][0], types.ArtifactRef)
    assert stage_a["outs"][0].identity == types.ArtifactIdentity(
        producer="_helper_build_stage_a", key=None
    )
    assert stage_a["outs"][0].tag is types.ArtifactTag.DATA
    assert isinstance(stage_a["outs"][0].format, loaders.DataFrameJSONL)

    assert stage_b["deps"]["data"].identity == types.ArtifactIdentity(
        producer="_helper_build_stage_a", key=None
    )
    assert stage_b["deps"]["data"].tag is types.ArtifactTag.DATA
    assert isinstance(stage_b["outs"][0].format, loaders.YAML)
    assert stage_b["outs"][0].tag is types.ArtifactTag.DATA


def test_pipeline_build_single_field_typeddict_preserves_key(tmp_path: pathlib.Path) -> None:
    """Single-field TypedDict outputs must preserve their field name as the identity key."""
    with Pipeline("test_single_td", root=tmp_path) as pipeline:
        _helper_single_field_typeddict()

    legacy = pipeline.build()
    stage_info = legacy.get("_helper_single_field_typeddict")

    assert len(stage_info["outs"]) == 1
    out = stage_info["outs"][0]
    assert out.identity.key == "result", (
        f"Single-field TypedDict key should be 'result', not {out.identity.key!r}"
    )


def test_pipeline_build_single_field_typeddict_dep_key(tmp_path: pathlib.Path) -> None:
    """Dep references to single-field TypedDict outputs must use the field key."""
    with Pipeline("test_single_td_dep", root=tmp_path) as pipeline:
        data = _helper_single_field_typeddict()
        _helper_build_stage_b(data)  # pyright: ignore[reportArgumentType] - compose records deps, not real calls

    legacy = pipeline.build()
    consumer = legacy.get("_helper_build_stage_b")

    dep = consumer["deps"]["data"]
    assert dep.identity.key == "result", (
        f"Dep to single-field TypedDict should use key 'result', not {dep.identity.key!r}"
    )


def test_external_input_produces_bindings(tmp_path: pathlib.Path) -> None:
    project_root = tmp_path
    (project_root / "data" / "external").mkdir(parents=True)
    (project_root / "data" / "external" / "ext_data.jsonl").write_text("[]")

    with Pipeline("test", root=project_root) as pipeline:
        ext = pipeline.input("ext_data.jsonl", t=list, external=True)

        @stage
        def consume(data: list = ext) -> Annotated[list, outputs.Out("out.json", loaders.JSON())]:
            return data

    legacy = pipeline.build()
    bindings = legacy.input_bindings

    assert "ext_data.jsonl" in bindings
    assert bindings["ext_data.jsonl"] == "data/external/ext_data.jsonl"


class _HelperTaggedOutputs(TypedDict):
    data: pd.DataFrame
    metrics: Annotated[dict, metric]
    chart: Annotated[pd.DataFrame, plot]


@stage
def _helper_build_tagged() -> _HelperTaggedOutputs:
    return {
        "data": pd.DataFrame(),
        "metrics": {"rows": 1},
        "chart": pd.DataFrame(),
    }


def test_pipeline_build_artifact_refs_and_tags(tmp_path: pathlib.Path) -> None:
    with Pipeline("compose_tagged", root=tmp_path) as pipeline:
        _helper_build_tagged()

    legacy = pipeline.build()
    stage_info = legacy.get("_helper_build_tagged")

    outs = stage_info["outs"]
    assert all(isinstance(out, types.ArtifactRef) for out in outs)

    tags = {out.identity.key: out.tag for out in outs}
    assert tags["data"] is types.ArtifactTag.DATA
    assert tags["metrics"] is types.ArtifactTag.METRIC
    assert tags["chart"] is types.ArtifactTag.PLOT


def test_validate_artifact_identity_rejects_path_separators() -> None:
    # Slashes are allowed as namespace separators
    types.validate_artifact_identity(
        types.ArtifactIdentity(producer="blueprints/benchmark.yaml", key=None)
    )
    types.validate_artifact_identity(types.ArtifactIdentity(producer="data/raw/input", key=None))
    types.validate_artifact_identity(types.ArtifactIdentity(producer="ok", key="a/b/c"))

    # But backslashes are rejected
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="bad\\name", key=None))
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="ok", key="bad\\part"))

    # Null bytes are rejected
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="bad\0name", key=None))
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="ok", key="bad\0part"))

    # Path traversal segments (. or ..) are rejected
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="..", key=None))
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="a/../b", key=None))
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="a/./b", key=None))
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="ok", key="a/../b"))
    with pytest.raises(ValueError):
        types.validate_artifact_identity(types.ArtifactIdentity(producer="ok", key="a/./b"))


# --- Pipeline variant context manager ---


def test_pipeline_variant_basic(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_variant", root=tmp_path) as pipeline, pipeline.variant("gpt4"):
        _helper_produce(params=stage_def.StageParams())

    assert len(pipeline._stages) == 1
    assert pipeline._stages[0].name == "_helper_produce@gpt4"
    assert pipeline._stages[0].variant == "gpt4"


def test_pipeline_variant_nested(tmp_path: pathlib.Path) -> None:
    with (
        Pipeline("test_nested", root=tmp_path) as pipeline,
        pipeline.variant("base"),
        pipeline.variant("gpt4"),
    ):
        _helper_produce(params=stage_def.StageParams())

    assert len(pipeline._stages) == 1
    assert pipeline._stages[0].name == "_helper_produce@base@gpt4"
    assert pipeline._stages[0].variant == "base@gpt4"


def test_pipeline_variant_multiple_stages(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_multi", root=tmp_path) as pipeline, pipeline.variant("v1"):
        _helper_produce(params=stage_def.StageParams())
        _helper_consume_dict(data={})

    assert len(pipeline._stages) == 2
    assert pipeline._stages[0].name == "_helper_produce@v1"
    assert pipeline._stages[1].name == "_helper_consume_dict@v1"
    assert pipeline._stages[0].variant == "v1"
    assert pipeline._stages[1].variant == "v1"


def test_pipeline_variant_artifact_identity(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_artifact_id", root=tmp_path) as pipeline, pipeline.variant("gpt4"):
        data = _helper_produce(params=stage_def.StageParams())
        _helper_consume(data)

    legacy = pipeline.build()

    stage_a = legacy.get("_helper_produce@gpt4")
    stage_b = legacy.get("_helper_consume@gpt4")

    assert stage_a["variant"] == "gpt4"
    assert stage_b["variant"] == "gpt4"
    assert stage_a["outs"][0].identity.producer == "_helper_produce@gpt4"
    assert stage_b["deps"]["data"].identity.producer == "_helper_produce@gpt4"


def test_pipeline_variant_with_repeated_calls(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_repeat_variant", root=tmp_path) as pipeline, pipeline.variant("v1"):
        _helper_repeat(params=stage_def.StageParams())
        _helper_repeat(params=stage_def.StageParams())

    assert pipeline._stages[0].name == "_helper_repeat@v1"
    assert pipeline._stages[1].name == "_helper_repeat@1@v1"
    assert pipeline._stages[0].variant == "v1"
    assert pipeline._stages[1].variant == "v1"


def test_pipeline_variant_mixed_with_non_variant(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_mixed", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())
        with pipeline.variant("v1"):
            _helper_consume_dict(data={})
        _helper_repeat(params=stage_def.StageParams())

    assert pipeline._stages[0].name == "_helper_produce"
    assert pipeline._stages[1].name == "_helper_consume_dict@v1"
    assert pipeline._stages[2].name == "_helper_repeat"
    assert pipeline._stages[0].variant is None
    assert pipeline._stages[1].variant == "v1"
    assert pipeline._stages[2].variant is None


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
