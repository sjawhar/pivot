from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pydantic
import pytest
from pydantic import BaseModel, ValidationError

from pivot import exceptions, parameters, project, stage_def
from pivot.compose import Pipeline, stage

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


# -----------------------------------------------------------------------------
# Test Pydantic models
# -----------------------------------------------------------------------------


class TrainParams(BaseModel):
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


class RequiredParams(BaseModel):
    name: str  # Required - no default
    value: int = 10


class NestedParams(BaseModel):
    lr: float = 0.001
    optimizer: str = "adam"


class ComplexParams(BaseModel):
    training: NestedParams = NestedParams()
    debug: bool = False


class TupleParams(BaseModel):
    token_range: tuple[float, float] = (5000.0, 100000000.0)
    time_buckets: list[tuple[int, int]] = [(1, 15), (16, 60), (60, 240)]


# -----------------------------------------------------------------------------
# load_params_yaml tests
# -----------------------------------------------------------------------------


class _LoadParams(stage_def.StageParams):
    lr: float = 0.01


@stage
def _load_stage(params: _LoadParams) -> None:
    pass


def _pipeline(tmp_path: Path, name: str = "p") -> Pipeline:
    """A pipeline with stages '<name>/_load_stage' and '<name>/_load_stage@v'."""
    with Pipeline(name, root=tmp_path) as pipeline:
        _load_stage(params=_LoadParams())
        with pipeline.variant("v"):
            _load_stage(params=_LoadParams())
    return pipeline


def test_load_params_yaml_missing_file(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(project, "get_project_root", return_value=tmp_path)
    result = parameters.load_params_yaml(_pipeline(tmp_path))
    assert result == {}, "Missing params.yaml should return empty dict"


def test_load_params_yaml_from_explicit_path(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("p/_load_stage:\n  lr: 0.001\n")

    result = parameters.load_params_yaml(_pipeline(tmp_path), params_file)
    assert result == {"p/_load_stage": {"lr": 0.001}}


def test_load_params_yaml_from_project_root(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.object(project, "get_project_root", return_value=tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("p/_load_stage:\n  lr: 0.01\np/_load_stage@v:\n  lr: 0.5\n")

    result = parameters.load_params_yaml(_pipeline(tmp_path))
    assert result == {"p/_load_stage": {"lr": 0.01}, "p/_load_stage@v": {"lr": 0.5}}


def test_load_params_yaml_non_dict_root_raises(tmp_path: Path) -> None:
    """Non-dict root should raise ParamsError."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("- item1\n- item2\n")

    with pytest.raises(exceptions.ParamsError, match="root must be a dict"):
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)


def test_load_params_yaml_refuses_non_dict_stage_value(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("p/_load_stage: 3\n")

    with pytest.raises(
        exceptions.PipelineConfigError,
        match="params.yaml key 'p/_load_stage' must map parameter names to values, got int",
    ):
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)


def test_load_params_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    """Invalid YAML should raise ParamsError, not silently return empty dict."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("invalid: yaml: content: ::::")

    with pytest.raises(exceptions.ParamsError, match="Failed to parse"):
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)


def test_load_params_yaml_lists_every_bad_key(tmp_path: Path) -> None:
    """A bare key, a typo and a non-mapping value are all reported in one error."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("_load_stage:\n  lr: 1\np/_load_stgae:\n  lr: 2\np/_load_stage@v: [1]\n")

    with pytest.raises(exceptions.PipelineConfigError) as exc_info:
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)

    assert str(exc_info.value).splitlines() == [
        "params.yaml key '_load_stage' names no stage; keys are '<pipeline>/<stage>', "
        + "e.g. 'p/_load_stage'",
        "params.yaml key 'p/_load_stgae' names no stage in pipeline 'p'; "
        + "did you mean 'p/_load_stage' or 'p/_load_stage@v'?",
        "params.yaml key 'p/_load_stage@v' must map parameter names to values, got list",
    ]


def test_load_params_yaml_skips_key_for_pipeline_not_loaded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("other/train:\n  lr: 1\nnested/other/train:\n  lr: 2\n")

    with caplog.at_level("DEBUG", logger="pivot.parameters"):
        result = parameters.load_params_yaml(_pipeline(tmp_path), params_file)

    assert result == {"other/train": {"lr": 1}, "nested/other/train": {"lr": 2}}
    assert "pipeline 'other' is not loaded in this run; skipping" in caplog.text
    assert "pipeline 'nested/other' is not loaded in this run; skipping" in caplog.text


def test_load_params_yaml_typo_without_close_match_points_to_list(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("p/zzz:\n  lr: 1\n")

    with pytest.raises(
        exceptions.PipelineConfigError,
        match="names no stage in pipeline 'p'; run 'pivot list' to see its stages",
    ):
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)


def test_load_params_yaml_checks_keys_of_included_pipelines(tmp_path: Path) -> None:
    combined = Pipeline("all", root=tmp_path)
    combined.include(_pipeline(tmp_path, "a"))
    params_file = tmp_path / "params.yaml"
    params_file.write_text("a/_load_stage:\n  lr: 1\na/_load_stag:\n  lr: 2\n")

    with pytest.raises(exceptions.PipelineConfigError, match="names no stage in pipeline 'a'"):
        parameters.load_params_yaml(combined, params_file)


def test_load_params_yaml_refuses_bad_values_naming_stage_field_and_value(tmp_path: Path) -> None:
    """A value its field rejects is refused at load, with every bad key listed in one error."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "p/_load_stage:\n  lr: fast\np/_load_stage@v:\n  lr: [1]\np/_load_stgae:\n  lr: 2\n"
    )

    with pytest.raises(exceptions.PipelineConfigError) as exc_info:
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)

    assert str(exc_info.value).splitlines() == [
        "params.yaml key 'p/_load_stage': field 'lr' value 'fast' is invalid for _LoadParams: "
        + "Input should be a valid number, unable to parse string as a number",
        "params.yaml key 'p/_load_stage@v': field 'lr' value [1] is invalid for _LoadParams: "
        + "Input should be a valid number",
        "params.yaml key 'p/_load_stgae' names no stage in pipeline 'p'; "
        + "did you mean 'p/_load_stage' or 'p/_load_stage@v'?",
    ]


def test_load_params_yaml_accepts_a_bad_base_value_every_variant_replaces(
    tmp_path: Path,
) -> None:
    """A key is checked as each stage sees it: base value merged under the variant's key."""
    with Pipeline("q", root=tmp_path) as pipeline, pipeline.variant("v"):
        _load_stage(params=_LoadParams())
    params_file = tmp_path / "params.yaml"
    params_file.write_text("q/_load_stage:\n  lr: fast\nq/_load_stage@v:\n  lr: 0.25\n")

    assert parameters.load_params_yaml(pipeline, params_file) == {
        "q/_load_stage": {"lr": "fast"},
        "q/_load_stage@v": {"lr": 0.25},
    }


def test_load_params_yaml_refuses_an_unknown_field_with_close_matches(tmp_path: Path) -> None:
    """A field typo is refused naming the field and what it could mean, beside other errors."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("p/_load_stage:\n  lrr: 0.5\np/_load_stage@v:\n  zzz: 1\n  lr: x\n")

    with pytest.raises(exceptions.PipelineConfigError) as exc_info:
        parameters.load_params_yaml(_pipeline(tmp_path), params_file)

    assert str(exc_info.value).splitlines() == [
        "params.yaml key 'p/_load_stage': field 'lrr' is not a field of _LoadParams; "
        + "did you mean 'lr'?",
        "params.yaml key 'p/_load_stage@v': field 'lr' value 'x' is invalid for _LoadParams: "
        + "Input should be a valid number, unable to parse string as a number",
        "params.yaml key 'p/_load_stage@v': field 'zzz' is not a field of _LoadParams; "
        + "its fields are 'lr'",
    ]


def test_stage_params_refuses_an_unknown_field_in_code() -> None:
    """The same typo in code is refused too: StageParams forbids extra fields."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _LoadParams(lrr=0.5)  # pyright: ignore[reportCallIssue] - the typo under test


class _LooseParams(stage_def.StageParams):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")
    lr: float = 0.01


@stage
def _loose_stage(params: _LooseParams) -> None:
    pass


def test_stage_params_subclass_may_allow_extra_fields(tmp_path: Path) -> None:
    """A subclass that sets its own extra= keeps it; params.yaml extras then pass through."""
    with Pipeline("loose", root=tmp_path) as pipeline:
        _loose_stage(params=_LooseParams())
    params_file = tmp_path / "params.yaml"
    params_file.write_text("loose/_loose_stage:\n  extra: 1\n")

    assert parameters.load_params_yaml(pipeline, params_file) == {
        "loose/_loose_stage": {"extra": 1}
    }


class _DerivedParams(stage_def.StageParams):
    lr: float = 0.01

    @pydantic.computed_field
    @property
    def half_lr(self) -> float:
        return self.lr / 2


def test_apply_overrides_keeps_computed_fields_out_of_validation() -> None:
    """A computed field is dumped but is not an input; with extra='forbid' it must not be fed back."""
    result = parameters.apply_overrides(_DerivedParams(), "p/s", {"p/s": {"lr": 0.5}})

    assert result.half_lr == 0.25


class _OptimizerParams(stage_def.StageParams):
    momentum: float = 0.9


class _NestedLoadParams(stage_def.StageParams):
    optimizer: _OptimizerParams = _OptimizerParams()


@stage
def _nested_stage(params: _NestedLoadParams) -> None:
    pass


def test_load_params_yaml_names_the_nested_model_of_an_unknown_field(tmp_path: Path) -> None:
    with Pipeline("n", root=tmp_path) as pipeline:
        _nested_stage(params=_NestedLoadParams())
    params_file = tmp_path / "params.yaml"
    params_file.write_text("n/_nested_stage:\n  optimizer:\n    momentm: 0.5\n")

    with pytest.raises(
        exceptions.PipelineConfigError,
        match=r"^params\.yaml key 'n/_nested_stage': field 'optimizer\.momentm' is not a field "
        + r"of _OptimizerParams; did you mean 'momentum'\?$",
    ):
        parameters.load_params_yaml(pipeline, params_file)


# -----------------------------------------------------------------------------
# build_params_instance tests
# -----------------------------------------------------------------------------


def test_build_params_instance_defaults_only() -> None:
    instance = parameters.build_params_instance(TrainParams, "train", None)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.01
    assert instance.epochs == 100
    assert instance.batch_size == 32


def test_build_params_instance_with_yaml_overrides() -> None:
    yaml_overrides = {"train": {"learning_rate": 0.001, "epochs": 200}}
    instance = parameters.build_params_instance(TrainParams, "train", yaml_overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.001, "YAML should override default"
    assert instance.epochs == 200, "YAML should override default"
    assert instance.batch_size == 32, "Unspecified fields keep defaults"


def test_build_params_instance_missing_stage_in_yaml() -> None:
    yaml_overrides = {"other_stage": {"lr": 0.1}}
    instance = parameters.build_params_instance(TrainParams, "train", yaml_overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.01, "Should use defaults when stage not in YAML"


def test_build_params_instance_extra_yaml_fields_ignored() -> None:
    yaml_overrides = {"train": {"learning_rate": 0.002, "extra_field": "ignored"}}
    instance = parameters.build_params_instance(TrainParams, "train", yaml_overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.002
    assert not hasattr(instance, "extra_field"), "Extra fields should be ignored"


def test_build_params_instance_required_field_from_yaml() -> None:
    yaml_overrides = {"process": {"name": "my_process"}}
    instance = parameters.build_params_instance(RequiredParams, "process", yaml_overrides)
    assert isinstance(instance, RequiredParams)
    assert instance.name == "my_process"
    assert instance.value == 10


def test_build_params_instance_required_field_missing_raises() -> None:
    with pytest.raises(ValidationError):
        parameters.build_params_instance(RequiredParams, "process", {})


def test_build_params_instance_type_mismatch_raises() -> None:
    yaml_overrides = {"train": {"epochs": "not_an_int"}}
    with pytest.raises(ValidationError):
        parameters.build_params_instance(TrainParams, "train", yaml_overrides)


def test_build_params_instance_nested_model() -> None:
    yaml_overrides = {"complex": {"debug": True}}
    instance = parameters.build_params_instance(ComplexParams, "complex", yaml_overrides)
    assert isinstance(instance, ComplexParams)
    assert instance.debug is True
    assert instance.training.lr == 0.001
    assert instance.training.optimizer == "adam"


# -----------------------------------------------------------------------------
# get_effective_params tests
# -----------------------------------------------------------------------------


def test_get_effective_params_with_instance() -> None:
    """get_effective_params returns dict from instance."""
    instance = TrainParams(learning_rate=0.05, epochs=50)
    params_dict = parameters.get_effective_params(instance, "train", None)
    assert params_dict == {"learning_rate": 0.05, "epochs": 50, "batch_size": 32}


def test_get_effective_params_normalizes_tuples_to_lists() -> None:
    """get_effective_params converts tuples to lists via model_dump(mode='json')."""
    instance = TupleParams()
    result = parameters.get_effective_params(instance, "plot", None)

    # Top-level tuple should be converted to list
    assert isinstance(result["token_range"], list), "token_range should be a list"
    assert result["token_range"] == [5000.0, 100000000.0]

    # Nested tuples in list should be converted to lists
    assert isinstance(result["time_buckets"], list), "time_buckets should be a list"
    assert isinstance(result["time_buckets"][0], list), "time_buckets[0] should be a list"
    assert result["time_buckets"] == [[1, 15], [16, 60], [60, 240]]


# -----------------------------------------------------------------------------
# model_dump tests (Pydantic built-in, replacing params_to_dict)
# -----------------------------------------------------------------------------


def test_params_to_dict_simple() -> None:
    instance = TrainParams(learning_rate=0.05, epochs=50, batch_size=16)
    result = instance.model_dump()
    assert result == {"learning_rate": 0.05, "epochs": 50, "batch_size": 16}


def test_params_to_dict_nested() -> None:
    instance = ComplexParams(debug=True, training=NestedParams(lr=0.01, optimizer="sgd"))
    result = instance.model_dump()
    assert result == {
        "debug": True,
        "training": {"lr": 0.01, "optimizer": "sgd"},
    }


# -----------------------------------------------------------------------------
# validate_params_cls tests
# -----------------------------------------------------------------------------


def test_validate_params_cls_with_basemodel() -> None:
    assert parameters.validate_params_cls(TrainParams) is True


def test_validate_params_cls_with_non_class() -> None:
    assert parameters.validate_params_cls("not a class") is False


def test_validate_params_cls_with_regular_class() -> None:
    class RegularClass:
        pass

    assert parameters.validate_params_cls(RegularClass) is False


def test_validate_params_cls_with_dict() -> None:
    assert parameters.validate_params_cls(dict) is False


# -----------------------------------------------------------------------------
# apply_overrides tests (replacing extract_stage_params)
# -----------------------------------------------------------------------------


def test_apply_overrides_with_pydantic_model() -> None:
    instance = TrainParams()  # All defaults
    overrides = {"train": {"learning_rate": 0.005}}
    result = parameters.apply_overrides(instance, "train", overrides)
    assert isinstance(result, TrainParams)
    assert result.learning_rate == 0.005
    assert result.epochs == 100
    assert result.batch_size == 32


def test_apply_overrides_no_overrides() -> None:
    instance = TrainParams(learning_rate=0.02)
    result = parameters.apply_overrides(instance, "train", None)
    assert result is instance, "Should return same instance when no overrides"


def test_apply_overrides_preserves_unspecified_fields() -> None:
    instance = TrainParams(learning_rate=0.02, epochs=50)
    overrides = {"train": {"learning_rate": 0.005}}
    result = parameters.apply_overrides(instance, "train", overrides)
    assert result.learning_rate == 0.005
    assert result.epochs == 50, "Unspecified fields should be preserved"


def test_get_effective_params_with_overrides() -> None:
    instance = TrainParams()
    overrides = {"train": {"learning_rate": 0.002}}
    params_dict = parameters.get_effective_params(instance, "train", overrides)
    assert params_dict == {"learning_rate": 0.002, "epochs": 100, "batch_size": 32}


def test_get_effective_params_no_instance() -> None:
    params_dict = parameters.get_effective_params(None, "train", None)
    assert params_dict == {}


# -----------------------------------------------------------------------------
# List replacement tests (deepmerge should replace, not append)
# -----------------------------------------------------------------------------


class ListParams(BaseModel):
    files: list[str] = ["default.csv"]
    tags: list[str] = ["a", "b"]


def test_params_override_replaces_lists() -> None:
    """Override list replaces base list entirely, not append."""
    instance = ListParams()
    overrides = {"stage": {"files": ["override.csv"], "tags": ["x"]}}
    result = parameters.apply_overrides(instance, "stage", overrides)

    # Lists should be replaced, not appended
    assert result.files == ["override.csv"], "List should be replaced, not appended"
    assert result.tags == ["x"], "List should be replaced, not appended"


def test_params_deep_merge_replaces_nested_lists() -> None:
    """Deep merge replaces nested lists instead of appending."""

    class NestedListParams(BaseModel):
        config: dict[str, list[str]] = {"items": ["a", "b"]}

    instance = NestedListParams()
    overrides = {"stage": {"config": {"items": ["x", "y", "z"]}}}
    result = parameters.apply_overrides(instance, "stage", overrides)

    # Nested list should be replaced
    assert result.config["items"] == ["x", "y", "z"], "Nested list should be replaced"
