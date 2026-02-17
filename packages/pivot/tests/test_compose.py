# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import inspect
import pathlib
import typing
from typing import Annotated, NotRequired, Required, TypedDict

import pandas as pd
import pytest

from pivot import fingerprint, loaders, outputs, stage_def, types
from pivot.compose import (
    SINGLE_OUTPUT_KEY,
    ArtifactHandle,
    CollectionKind,
    Pipeline,
    _handle_to_artifact_ref,
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
from pivot.decorators import no_fingerprint
from pivot.executor.worker import _reconstruct_list_kwargs
from pivot.registry import RegistryStageInfo, _compute_fingerprint


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


class _HelperRequiredNotRequired(TypedDict, total=False):
    data: Required[pd.DataFrame]
    config: NotRequired[Annotated[dict, loaders.YAML()]]


def _helper_required_not_required() -> _HelperRequiredNotRequired: ...


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
def _helper_base_a(params: stage_def.StageParams) -> pd.DataFrame:
    return pd.DataFrame()


@stage
def _helper_base_b(data: pd.DataFrame) -> dict:
    return {"rows": len(data)}


@stage
def _helper_base_c_unrelated(params: stage_def.StageParams) -> dict:
    return {"ok": True}


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

    assert legacy.list_stages() == [
        "compose_build/_helper_build_stage_a",
        "compose_build/_helper_build_stage_b",
    ]

    stage_a = legacy.get("compose_build/_helper_build_stage_a")
    stage_b = legacy.get("compose_build/_helper_build_stage_b")

    assert stage_a["func"] is inspect.unwrap(_helper_build_stage_a)
    assert not hasattr(stage_a["func"], "_is_stage")
    assert stage_a["name"] == "compose_build/_helper_build_stage_a"
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
        producer="compose_build/_helper_build_stage_a", key=None
    )
    assert stage_a["outs"][0].tag is types.ArtifactTag.DATA
    assert isinstance(stage_a["outs"][0].format, loaders.DataFrameJSONL)

    assert stage_b["deps"]["data"].identity == types.ArtifactIdentity(
        producer="compose_build/_helper_build_stage_a", key=None
    )
    assert stage_b["deps"]["data"].tag is types.ArtifactTag.DATA
    assert isinstance(stage_b["outs"][0].format, loaders.YAML)
    assert stage_b["outs"][0].tag is types.ArtifactTag.DATA


def test_pipeline_build_single_field_typeddict_preserves_key(tmp_path: pathlib.Path) -> None:
    """Single-field TypedDict outputs must preserve their field name as the identity key."""
    with Pipeline("test_single_td", root=tmp_path) as pipeline:
        _helper_single_field_typeddict()

    legacy = pipeline.build()
    stage_info = legacy.get("test_single_td/_helper_single_field_typeddict")

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
    consumer = legacy.get("test_single_td_dep/_helper_build_stage_b")

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
    stage_info = legacy.get("compose_tagged/_helper_build_tagged")

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
        result = _helper_produce(params=stage_def.StageParams())
        _helper_consume_dict(data=result)  # pyright: ignore[reportArgumentType] - compose records handles

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

    stage_a = legacy.get("test_artifact_id/_helper_produce@gpt4")
    stage_b = legacy.get("test_artifact_id/_helper_consume@gpt4")

    assert stage_a["variant"] == "gpt4"
    assert stage_b["variant"] == "gpt4"
    assert stage_a["outs"][0].identity.producer == "test_artifact_id/_helper_produce@gpt4"
    assert stage_b["deps"]["data"].identity.producer == "test_artifact_id/_helper_produce@gpt4"


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
        result = _helper_produce(params=stage_def.StageParams())
        with pipeline.variant("v1"):
            _helper_consume_dict(data=result)  # pyright: ignore[reportArgumentType] - compose records handles
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


# --- Bug 3: list[ArtifactHandle] support ---


@stage
def _helper_consume_list(
    main_data: pd.DataFrame,
    extra_data: list[pd.DataFrame],
) -> pd.DataFrame:
    return main_data


@stage
def _helper_consume_list_from_stages(
    data_files: list[pd.DataFrame],
) -> dict:
    return {"count": len(data_files)}


def test_record_stage_detects_list_artifact_handles(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=[inp, inp])

    node = pipeline._stages[0]
    assert "main_data" in node.input_handles
    assert "extra_data" in node.list_input_handles
    assert len(node.list_input_handles["extra_data"]) == 2
    assert node.collection_params["extra_data"] == CollectionKind.LIST


def test_record_stage_detects_tuple_artifact_handles(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=(inp, inp))  # pyright: ignore[reportArgumentType] - compose records handles

    node = pipeline._stages[0]
    assert "extra_data" in node.list_input_handles
    assert node.collection_params["extra_data"] == CollectionKind.TUPLE


def test_record_stage_detects_empty_list(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=[])

    node = pipeline._stages[0]
    assert "extra_data" in node.list_input_handles
    assert len(node.list_input_handles["extra_data"]) == 0
    assert node.collection_params["extra_data"] == CollectionKind.LIST


def test_build_expands_list_deps(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_list", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=[inp, inp, inp])

    legacy = pipeline.build()
    stage_info = legacy.get("test_list/_helper_consume_list")

    assert "main_data" in stage_info["deps"]
    assert "extra_data[0]" in stage_info["deps"]
    assert "extra_data[1]" in stage_info["deps"]
    assert "extra_data[2]" in stage_info["deps"]
    assert "extra_data" not in stage_info["deps"]

    for key in ["extra_data[0]", "extra_data[1]", "extra_data[2]"]:
        ref = stage_info["deps"][key]
        assert ref.identity.producer == "data.csv"


def test_build_list_deps_from_multiple_stages(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_multi_list", root=tmp_path) as pipeline:
        a = _helper_produce(params=stage_def.StageParams())
        b = _helper_produce(params=stage_def.StageParams())
        _helper_consume_list_from_stages(data_files=[a, b])

    legacy = pipeline.build()
    consumer = legacy.get("test_multi_list/_helper_consume_list_from_stages")

    assert "data_files[0]" in consumer["deps"]
    assert "data_files[1]" in consumer["deps"]
    assert consumer["deps"]["data_files[0]"].identity.producer == "test_multi_list/_helper_produce"
    assert (
        consumer["deps"]["data_files[1]"].identity.producer == "test_multi_list/_helper_produce@1"
    )


def test_build_list_deps_dag_has_correct_edges(tmp_path: pathlib.Path) -> None:
    with Pipeline("test_dag_list", root=tmp_path) as pipeline:
        a = _helper_produce(params=stage_def.StageParams())
        b = _helper_produce(params=stage_def.StageParams())
        _helper_consume_list_from_stages(data_files=[a, b])

    legacy = pipeline.build()

    from pivot.engine import graph

    bipartite = graph.build_graph(legacy._registry._stages)
    upstream = graph.get_upstream_stages(
        bipartite, "test_dag_list/_helper_consume_list_from_stages"
    )
    assert sorted(upstream) == [
        "test_dag_list/_helper_produce",
        "test_dag_list/_helper_produce@1",
    ]


# --- _reconstruct_list_kwargs ---


def test_reconstruct_list_kwargs_basic() -> None:
    kwargs = {
        "params": "some_params",
        "data_files[0]": "loaded_0",
        "data_files[1]": "loaded_1",
        "data_files[2]": "loaded_2",
    }
    result = _reconstruct_list_kwargs(kwargs, {"data_files": "list"})
    assert result == {
        "params": "some_params",
        "data_files": ["loaded_0", "loaded_1", "loaded_2"],
    }


def test_reconstruct_list_kwargs_preserves_order() -> None:
    kwargs = {
        "data_files[2]": "val_2",
        "data_files[0]": "val_0",
        "data_files[1]": "val_1",
    }
    result = _reconstruct_list_kwargs(kwargs, {"data_files": "list"})
    assert result["data_files"] == ["val_0", "val_1", "val_2"]


def test_reconstruct_list_kwargs_noop_without_lists() -> None:
    kwargs = {"x": 1, "y": 2}
    result = _reconstruct_list_kwargs(kwargs, {})
    assert result == {"x": 1, "y": 2}


def test_reconstruct_list_kwargs_multiple_lists() -> None:
    kwargs = {
        "a[0]": "a0",
        "a[1]": "a1",
        "b[0]": "b0",
        "plain": "val",
    }
    result = _reconstruct_list_kwargs(kwargs, {"a": "list", "b": "list"})
    assert result == {"a": ["a0", "a1"], "b": ["b0"], "plain": "val"}


def test_reconstruct_list_kwargs_tuple() -> None:
    kwargs = {"data[0]": "a", "data[1]": "b"}
    result = _reconstruct_list_kwargs(kwargs, {"data": "tuple"})
    assert result["data"] == ("a", "b")
    assert isinstance(result["data"], tuple)


def test_reconstruct_list_kwargs_empty_list() -> None:
    result = _reconstruct_list_kwargs({}, {"data_files": "list"})
    assert result == {"data_files": []}


def test_reconstruct_list_kwargs_empty_tuple() -> None:
    result = _reconstruct_list_kwargs({}, {"data_files": "tuple"})
    assert result == {"data_files": ()}
    assert isinstance(result["data_files"], tuple)


def test_build_collection_params_populated(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=[inp, inp])

    legacy = pipeline.build()
    stage_info = legacy.get("test/_helper_consume_list")
    assert stage_info["collection_params"] == {"extra_data": "list"}


def test_build_empty_list_no_deps_but_collection_params(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=[])

    legacy = pipeline.build()
    stage_info = legacy.get("test/_helper_consume_list")

    assert "extra_data[0]" not in stage_info["deps"]
    assert stage_info["collection_params"] == {"extra_data": "list"}


def test_build_single_element_list(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume_list(main_data=inp, extra_data=[inp])

    legacy = pipeline.build()
    stage_info = legacy.get("test/_helper_consume_list")
    assert "extra_data[0]" in stage_info["deps"]
    assert "extra_data[1]" not in stage_info["deps"]


def test_reconstruct_list_kwargs_single_element() -> None:
    kwargs = {"data[0]": "val"}
    result = _reconstruct_list_kwargs(kwargs, {"data": "list"})
    assert result == {"data": ["val"]}


# --- Multi-output dep resolution through build() ---


@stage
def _helper_consume_multi_output_dep(data: pd.DataFrame) -> pd.DataFrame:
    return data


def test_build_multi_output_dep_key(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        tagged = _helper_build_tagged()
        _helper_consume_multi_output_dep(data=tagged.data)

    legacy = pipeline.build()
    consumer = legacy.get("test/_helper_consume_multi_output_dep")
    ref = consumer["deps"]["data"]
    assert ref.identity.producer == "test/_helper_build_tagged"
    assert ref.identity.key == "data"


def test_build_multi_output_sub_handles_in_list(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        tagged = _helper_build_tagged()
        _helper_consume_list_from_stages(data_files=[tagged.data, tagged.data])

    legacy = pipeline.build()
    consumer = legacy.get("test/_helper_consume_list_from_stages")
    ref_0 = consumer["deps"]["data_files[0]"]
    assert ref_0.identity.producer == "test/_helper_build_tagged"
    assert ref_0.identity.key == "data"


def test_build_rejects_unsubscripted_multi_output_handle(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        tagged = _helper_build_tagged()
        _helper_consume_multi_output_dep(data=tagged)  # pyright: ignore[reportArgumentType] - intentional

    with pytest.raises(TypeError, match="without selecting an output"):
        pipeline.build()


# --- collection_params empty for non-list stages ---


def test_build_collection_params_empty_for_non_list_stage(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        a = _helper_produce(params=stage_def.StageParams())
        _helper_consume(data=a)

    legacy = pipeline.build()
    assert legacy.get("test/_helper_consume")["collection_params"] == {}


# --- Unrecognized argument detection ---


@stage
def _helper_with_plain_arg(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    return data


def test_record_stage_rejects_explicit_unsupported_arg(tmp_path: pathlib.Path) -> None:
    with (
        pytest.raises(ValueError, match="unsupported type"),
        Pipeline("test", root=tmp_path) as pipeline,
    ):
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_with_plain_arg(data=inp, verbose=True)  # pyright: ignore[reportArgumentType]


def test_record_stage_ignores_default_unsupported_arg(tmp_path: pathlib.Path) -> None:
    with Pipeline("test", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_with_plain_arg(data=inp)

    pipeline.build()


# --- Output format fingerprint detection ---


def test_loader_fingerprint_yaml_vs_csv_differ() -> None:
    """get_loader_fingerprint must produce different results for YAML vs CSV."""
    yaml_fp = fingerprint.get_loader_fingerprint(loaders.YAML())
    csv_fp = fingerprint.get_loader_fingerprint(loaders.CSV())
    assert yaml_fp != csv_fp, "YAML and CSV loaders should have different fingerprints"


def test_loader_fingerprint_csv_config_change_detected() -> None:
    """Changing CSV config (index_col) should change the fingerprint."""
    csv_default = fingerprint.get_loader_fingerprint(loaders.CSV())
    csv_custom = fingerprint.get_loader_fingerprint(loaders.CSV(index_col=[0, 1]))
    assert csv_default != csv_custom, "CSV() and CSV(index_col=[0, 1]) should differ"


def test_analyze_return_type_metric_with_explicit_csv() -> None:
    """Metric + explicit CSV loader should use CSV, not default YAML."""
    specs = _analyze_return_type(_helper_metric_with_csv)
    assert len(specs) == 1
    assert isinstance(specs[0].tag, _MetricTag), "Should be tagged as metric"
    assert isinstance(specs[0].format, loaders.CSV), "Should use explicit CSV, not default YAML"


def test_analyze_return_type_metric_default_yaml() -> None:
    """Metric without explicit loader should default to YAML."""
    specs = _analyze_return_type(_helper_metric_default_yaml)
    assert len(specs) == 1
    assert isinstance(specs[0].tag, _MetricTag)
    assert isinstance(specs[0].format, loaders.YAML), "Metric default should be YAML"


def _helper_metric_with_csv() -> Annotated[pd.DataFrame, metric, loaders.CSV(index_col=[0, 1])]:
    return pd.DataFrame()


def _helper_metric_default_yaml() -> Annotated[pd.DataFrame, metric]:
    return pd.DataFrame()


def _make_stage_info(
    func: typing.Any,
    name: str = "s",
    deps: dict[str, types.ArtifactRef] | None = None,
    outs: list[types.ArtifactRef] | None = None,
) -> RegistryStageInfo:
    return RegistryStageInfo(
        func=func,
        name=name,
        deps=deps or {},
        outs=outs or [],
        params=None,
        mutex=[],
        variant=None,
        signature=inspect.signature(func),
        fingerprint=None,
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


def _make_out(
    key: str,
    fmt: loaders.Writer[typing.Any] | loaders.Loader[typing.Any, typing.Any],
    producer: str = "s",
    tag: types.ArtifactTag = types.ArtifactTag.DATA,
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=fmt,
        python_type=pd.DataFrame,
        tag=tag,
    )


def _make_dep(
    key: str,
    fmt: loaders.Reader[typing.Any] | loaders.Loader[typing.Any, typing.Any],
    producer: str = "upstream",
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=fmt,
        python_type=pd.DataFrame,
        tag=types.ArtifactTag.DATA,
    )


def _dummy_stage() -> dict:
    return {}


@pytest.mark.parametrize(
    ("before_outs", "after_outs"),
    [
        pytest.param(
            [
                _make_out("a", loaders.CSV(index_col=[0, 1])),
                _make_out("b", loaders.CSV(index_col=0)),
            ],
            [_make_out("a", loaders.CSV(index_col=0)), _make_out("b", loaders.CSV(index_col=0))],
            id="same-class-different-config",
        ),
        pytest.param(
            [
                _make_out("savings", loaders.YAML(), tag=types.ArtifactTag.METRIC),
                _make_out("savings_nb", loaders.CSV(index_col=0), tag=types.ArtifactTag.METRIC),
            ],
            [
                _make_out("savings", loaders.CSV(index_col=[0, 1]), tag=types.ArtifactTag.METRIC),
                _make_out("savings_nb", loaders.CSV(index_col=0), tag=types.ArtifactTag.METRIC),
            ],
            id="yaml-to-csv-with-existing-csv",
        ),
    ],
)
def test_output_loader_change_detected(
    before_outs: list[types.ArtifactRef],
    after_outs: list[types.ArtifactRef],
) -> None:
    """Changing an output's loader config/class must change the stage fingerprint.

    Regression for key collision: without per-output namespacing, dict.update()
    merges loader keys by class name only, so two CSV outputs with different
    configs clobber each other's config hash.
    """
    fp_before = _compute_fingerprint("s", _make_stage_info(_dummy_stage, outs=before_outs))
    fp_after = _compute_fingerprint("s", _make_stage_info(_dummy_stage, outs=after_outs))
    assert fp_before != fp_after


def test_no_fingerprint_output_loader_collision(set_project_root: pathlib.Path) -> None:
    """@no_fingerprint() file-hash path must also prevent loader config collisions."""
    from pivot.decorators import no_fingerprint

    @no_fingerprint()
    def _nofp_stage() -> dict:
        return {}

    fp_before = _compute_fingerprint(
        "s",
        _make_stage_info(
            _nofp_stage,
            outs=[
                _make_out("a", loaders.CSV(index_col=[0, 1])),
                _make_out("b", loaders.CSV(index_col=0)),
            ],
        ),
    )
    fp_after = _compute_fingerprint(
        "s",
        _make_stage_info(
            _nofp_stage,
            outs=[
                _make_out("a", loaders.CSV(index_col=0)),
                _make_out("b", loaders.CSV(index_col=0)),
            ],
        ),
    )
    assert fp_before != fp_after


def test_dep_loader_config_collision() -> None:
    """Two deps with same loader class but different configs must both be fingerprinted."""
    fp_before = _compute_fingerprint(
        "s",
        _make_stage_info(
            _dummy_stage,
            deps={
                "train": _make_dep("train", loaders.CSV(index_col=[0, 1])),
                "test": _make_dep("test", loaders.CSV(index_col=0)),
            },
        ),
    )
    fp_after = _compute_fingerprint(
        "s",
        _make_stage_info(
            _dummy_stage,
            deps={
                "train": _make_dep("train", loaders.CSV(index_col=0)),
                "test": _make_dep("test", loaders.CSV(index_col=0)),
            },
        ),
    )
    assert fp_before != fp_after


# --- Cross-pipeline handle validation ---


def test_cross_pipeline_handle_accepted(tmp_path: pathlib.Path) -> None:
    p1 = Pipeline("pipe1", root=tmp_path)
    with p1:
        handle = p1.input("data", path="data/raw/data.csv", t=pd.DataFrame)

    p2 = Pipeline("pipe2", root=tmp_path)
    with p2:
        _helper_consume(data=handle)

    assert len(p2._stages) == 1
    assert "data" in p2._stages[0].input_handles
    assert p2._stages[0].input_handles["data"]._pipeline is p1


# --- build() validation without context manager ---


def test_build_without_context_manager_validates(tmp_path: pathlib.Path) -> None:
    """build() runs validation even without context manager."""
    p = Pipeline("test", root=tmp_path)
    p._validation_errors.append("test error for validation")
    with pytest.raises(ValueError, match="test error for validation"):
        p.build()


# --- Required/NotRequired unwrapping ---


def test_typeddict_required_fields_unwrapped() -> None:
    """Required/NotRequired wrappers don't break output parsing."""
    specs = _analyze_return_type(_helper_required_not_required)
    assert len(specs) == 2
    keys = {s.key for s in specs}
    assert keys == {"data", "config"}
    data_spec = next(s for s in specs if s.key == "data")
    assert data_spec.python_type is pd.DataFrame
    config_spec = next(s for s in specs if s.key == "config")
    assert config_spec.python_type is dict
    assert isinstance(config_spec.format, loaders.YAML)


# --- @no_fingerprint decorator stacking ---


# --- Stage name prefixing ---


def test_build_prefixes_stage_names_with_pipeline_name(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    legacy = pipeline.build()
    assert "my_pipeline/_helper_produce" in legacy.list_stages()
    assert "_helper_produce" not in legacy.list_stages()


def test_build_does_not_mutate_stage_node_names(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    assert pipeline._stages[0].name == "_helper_produce"
    pipeline.build()
    assert pipeline._stages[0].name == "_helper_produce"


def test_build_prefixes_output_identity_producer(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    legacy = pipeline.build()
    info = legacy.get("my_pipeline/_helper_produce")
    for out in info["outs"]:
        assert out.identity.producer == "my_pipeline/_helper_produce"


def test_handle_to_artifact_ref_qualifies_producer_name(tmp_path: pathlib.Path) -> None:
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        result = _helper_produce(params=stage_def.StageParams())

    ref = _handle_to_artifact_ref(result, "horizon/consumer")
    assert ref.identity.producer == "base/_helper_produce"


def test_build_prefixes_dep_identity_producer(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        result = _helper_produce(params=stage_def.StageParams())
        _helper_consume(data=result)

    legacy = pipeline.build()
    consumer = legacy.get("my_pipeline/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "my_pipeline/_helper_produce"


def test_cross_pipeline_dep_identity_qualified(tmp_path: pathlib.Path) -> None:
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        result = _helper_produce(params=stage_def.StageParams())

    p2 = Pipeline("horizon", root=tmp_path)
    with p2:
        _helper_consume(data=result)

    legacy = p2.build()
    consumer = legacy.get("horizon/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "base/_helper_produce"


def test_build_does_not_prefix_input_identity(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        inp = pipeline.input("data.csv", path="data/raw/data.csv", t=pd.DataFrame)
        _helper_consume(data=inp)

    legacy = pipeline.build()
    consumer = legacy.get("my_pipeline/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "data.csv"  # NOT "my_pipeline/data.csv"


def test_cross_pipeline_input_identity_not_qualified(tmp_path: pathlib.Path) -> None:
    p1 = Pipeline("base", root=tmp_path)
    with p1:
        data = p1.input("scores", path="data/raw/scores.csv", t=pd.DataFrame)

    p2 = Pipeline("horizon", root=tmp_path)
    with p2:
        _helper_consume(data=data)

    legacy = p2.build()
    consumer = legacy.get("horizon/_helper_consume")
    dep = consumer["deps"]["data"]
    assert dep.identity.producer == "scores"


def test_build_double_call_raises(tmp_path: pathlib.Path) -> None:
    with Pipeline("my_pipeline", root=tmp_path) as pipeline:
        _helper_produce(params=stage_def.StageParams())

    pipeline.build()
    with pytest.raises(RuntimeError, match="already built"):
        pipeline.build()


# --- @no_fingerprint decorator stacking ---


def test_no_fingerprint_outside_stage_decorator(set_project_root: pathlib.Path) -> None:
    """@no_fingerprint() applied outside @stage must be detected through compose → registry."""

    @no_fingerprint()
    @stage
    def my_stage() -> Annotated[str, outputs.Out("out.txt", loaders.Text())]:
        return "hello"

    p = Pipeline("test", root=set_project_root)
    with p:
        my_stage()
    built = p.build()

    fp = built._registry.ensure_fingerprint("test/my_stage")
    assert any(k.startswith("file:") for k in fp), (
        f"Expected file-hash fingerprint when @no_fingerprint is applied, got keys: {list(fp.keys())}"
    )


# --- Cross-pipeline closure collection ---


def test_build_includes_only_foreign_closure(tmp_path: pathlib.Path) -> None:
    """build() includes transitive foreign deps but excludes unrelated foreign stages."""
    base = Pipeline("base", root=tmp_path)
    with base:
        a = _helper_base_a(params=stage_def.StageParams())
        b = _helper_base_b(a)
        _helper_base_c_unrelated(params=stage_def.StageParams())

    consumer = Pipeline("consumer", root=tmp_path)
    with consumer:
        _helper_consume_dict(b)

    legacy = consumer.build()
    names = set(legacy.list_stages())
    assert "consumer/_helper_consume_dict" in names
    assert "base/_helper_base_b" in names
    assert "base/_helper_base_a" in names
    assert "base/_helper_base_c_unrelated" not in names
