from __future__ import annotations

import hashlib
import json
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


class _PlainOptimizer(BaseModel):
    momentum: float = 0.9


class _AllowingOptimizer(BaseModel):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow")
    momentum: float = 0.9
    inner: _PlainOptimizer = _PlainOptimizer()


class _PlainNestedParams(stage_def.StageParams):
    optimizer: _PlainOptimizer = _PlainOptimizer()
    layers: list[_PlainOptimizer] = []
    named: dict[str, _PlainOptimizer] = {}
    loose: _AllowingOptimizer = _AllowingOptimizer()


@stage
def _plain_nested_stage(params: _PlainNestedParams) -> None:
    pass


def _plain_nested_pipeline(tmp_path: Path) -> Pipeline:
    with Pipeline("n", root=tmp_path) as pipeline:
        _plain_nested_stage(params=_PlainNestedParams())
    return pipeline


@pytest.mark.parametrize(
    ("params_yaml", "path"),
    [
        ("  optimizer:\n    momentm: 0.5\n", "optimizer.momentm"),
        ("  layers:\n  - momentm: 0.5\n", "layers.0.momentm"),
        ("  named:\n    a:\n      momentm: 0.5\n", "named.a.momentm"),
        ("  loose:\n    inner:\n      momentm: 0.5\n", "loose.inner.momentm"),
    ],
    ids=["model", "list-of-models", "dict-of-models", "inside-an-allowing-model"],
)
def test_load_params_yaml_refuses_a_typo_inside_a_plain_nested_model(
    tmp_path: Path, params_yaml: str, path: str
) -> None:
    """Pydantic ignores an unknown key in a plain BaseModel, so the typo would run the default."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("n/_plain_nested_stage:\n" + params_yaml)

    with pytest.raises(exceptions.PipelineConfigError) as exc_info:
        parameters.load_params_yaml(_plain_nested_pipeline(tmp_path), params_file)

    assert str(exc_info.value) == (
        f"params.yaml key 'n/_plain_nested_stage': field '{path}' is not a field of "
        + "_PlainOptimizer; did you mean 'momentum'?"
    )


def test_load_params_yaml_lets_a_nested_model_that_allows_extra_fields_take_one(
    tmp_path: Path,
) -> None:
    pipeline = _plain_nested_pipeline(tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("n/_plain_nested_stage:\n  loose:\n    extra: 1\n")

    overrides = parameters.load_params_yaml(pipeline, params_file)

    effective = parameters.get_effective_params(
        pipeline.get_stage("n/_plain_nested_stage")["params"], "n/_plain_nested_stage", overrides
    )
    assert effective["loose"] == {"momentum": 0.9, "inner": {"momentum": 0.9}, "extra": 1}


class _AliasedOptimizer(stage_def.StageParams):
    momentum: float = pydantic.Field(0.9, alias="mom")


class _AliasedParams(stage_def.StageParams):
    learning_rate: float = pydantic.Field(0.01, alias="lr")
    epochs: int = 3
    optimizer: _AliasedOptimizer = _AliasedOptimizer.model_validate({})


class _ByNameParams(stage_def.StageParams):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(populate_by_name=True)
    learning_rate: float = pydantic.Field(0.01, alias="lr")


@pytest.mark.parametrize(
    ("params", "override", "expected"),
    [
        (
            _AliasedParams(lr=0.5),
            {"epochs": 4},
            {"learning_rate": 0.5, "epochs": 4, "optimizer": {"momentum": 0.9}},
        ),
        (
            _AliasedParams(lr=0.5),
            {"lr": 0.2},
            {"learning_rate": 0.2, "epochs": 3, "optimizer": {"momentum": 0.9}},
        ),
        (
            _AliasedParams(lr=0.5),
            {"learning_rate": 0.2},
            {"learning_rate": 0.2, "epochs": 3, "optimizer": {"momentum": 0.9}},
        ),
        (
            _AliasedParams(lr=0.5),
            {"optimizer": {"mom": 0.1}},
            {"learning_rate": 0.5, "epochs": 3, "optimizer": {"momentum": 0.1}},
        ),
        (
            _AliasedParams(lr=0.5),
            {"optimizer": {"momentum": 0.1}},
            {"learning_rate": 0.5, "epochs": 3, "optimizer": {"momentum": 0.1}},
        ),
        (
            _ByNameParams.model_validate({"learning_rate": 0.5}),
            {"learning_rate": 0.2},
            {"learning_rate": 0.2},
        ),
        (_ByNameParams.model_validate({"learning_rate": 0.5}), {"lr": 0.2}, {"learning_rate": 0.2}),
    ],
    ids=[
        "other-field",
        "by-alias",
        "by-name",
        "nested-by-alias",
        "nested-by-name",
        "populate-by-name-by-name",
        "populate-by-name-by-alias",
    ],
)
def test_override_reaches_an_aliased_field_by_alias_or_by_name(
    params: stage_def.StageParams, override: dict[str, object], expected: dict[str, object]
) -> None:
    """The code's values for aliased fields survive, and either name in params.yaml applies."""
    assert parameters.get_effective_params(params, "p/s", {"p/s": override}) == expected


class _SplitAliasParams(stage_def.StageParams):
    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(serialize_by_alias=True)
    sep: str = pydantic.Field(",", validation_alias="sep_in", serialization_alias="sep_out")
    header: bool = True


def test_override_keeps_a_field_whose_input_and_output_aliases_differ() -> None:
    """The value from code survives an override of another field, whatever the field dumps as."""
    effective = parameters.apply_overrides(
        _SplitAliasParams.model_validate({"sep_in": ";"}), "p/s", {"p/s": {"header": False}}
    )

    assert (effective.sep, effective.header) == (";", False)


class _ChoiceParams(stage_def.StageParams):
    learning_rate: float = pydantic.Field(
        0.01, validation_alias=pydantic.AliasChoices("lr", "rate")
    )
    epochs: int = 3


class _PathParams(stage_def.StageParams):
    sep: str = pydantic.Field(",", validation_alias=pydantic.AliasPath("csv", "sep"))
    header: bool = True


@pytest.mark.parametrize(
    ("params", "override", "expected"),
    [
        (
            _ChoiceParams.model_validate({"rate": 0.5}),
            {"epochs": 4},
            {"learning_rate": 0.5, "epochs": 4},
        ),
        (
            _ChoiceParams.model_validate({"rate": 0.5}),
            {"rate": 0.2},
            {"learning_rate": 0.2, "epochs": 3},
        ),
        (
            _ChoiceParams.model_validate({"rate": 0.5}),
            {"lr": 0.2},
            {"learning_rate": 0.2, "epochs": 3},
        ),
        (
            _PathParams.model_validate({"csv": {"sep": ";"}}),
            {"header": False},
            {"sep": ";", "header": False},
        ),
        (
            _PathParams.model_validate({"csv": {"sep": ";"}}),
            {"sep": "|"},
            {"sep": "|", "header": True},
        ),
    ],
    ids=[
        "choices-other-field",
        "choices-second-choice",
        "choices-first-choice",
        "path-other-field",
        "path-by-name",
    ],
)
def test_override_reaches_a_field_with_alias_choices_or_an_alias_path(
    params: stage_def.StageParams, override: dict[str, object], expected: dict[str, object]
) -> None:
    assert parameters._override_errors(params, "p/s", {"p/s": override}) == []
    assert parameters.get_effective_params(params, "p/s", {"p/s": override}) == expected


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"rat": 0.2},
            "field 'rat' is not a field of _ChoiceParams; did you mean 'rate'?",
        ),
        (
            {"rate": 0.2, "lr": 0.3},
            "field 'learning_rate' is set twice, as 'rate' and as 'lr'",
        ),
    ],
    ids=["typo", "two-choices"],
)
def test_alias_choices_typo_and_double_setting_are_refused(
    override: dict[str, object], message: str
) -> None:
    params = _ChoiceParams.model_validate({"rate": 0.5})

    assert parameters._override_errors(params, "p/s", {"p/s": override}) == [
        ("p/s", f"params.yaml key 'p/s': {message}")
    ]


def test_aliased_field_typo_is_refused_suggesting_the_alias() -> None:
    errors = parameters._override_errors(_AliasedParams(lr=0.5), "p/s", {"p/s": {"lrr": 0.2}})

    assert errors == [
        (
            "p/s",
            "params.yaml key 'p/s': field 'lrr' is not a field of _AliasedParams; "
            + "did you mean 'lr'?",
        )
    ]


def test_variant_key_by_field_name_replaces_base_key_by_alias() -> None:
    """Both spellings name one field, so the variant's key wins over the base key's."""
    overrides = {"p/s": {"lr": 0.2}, "p/s@v": {"learning_rate": 0.3}}

    effective = parameters.get_effective_params(_AliasedParams(lr=0.5), "p/s@v", overrides)

    assert effective["learning_rate"] == 0.3


def test_one_key_setting_a_field_by_name_and_by_alias_is_refused() -> None:
    errors = parameters._override_errors(
        _AliasedParams(lr=0.5), "p/s", {"p/s": {"lr": 0.2, "learning_rate": 0.3}}
    )

    assert errors == [
        (
            "p/s",
            "params.yaml key 'p/s': field 'learning_rate' is set twice, "
            + "as 'lr' and as 'learning_rate'",
        )
    ]


@stage
def _aliased_stage(params: _AliasedParams) -> None:
    pass


def test_load_params_yaml_accepts_an_aliased_field_by_either_name(tmp_path: Path) -> None:
    with Pipeline("a", root=tmp_path) as pipeline:
        _aliased_stage(params=_AliasedParams(lr=0.5))
    params_file = tmp_path / "params.yaml"
    params_file.write_text("a/_aliased_stage:\n  lr: 0.2\n  optimizer:\n    momentum: 0.1\n")

    overrides = parameters.load_params_yaml(pipeline, params_file)

    assert parameters.get_effective_params(
        pipeline.get_stage("a/_aliased_stage")["params"], "a/_aliased_stage", overrides
    ) == {"learning_rate": 0.2, "epochs": 3, "optimizer": {"momentum": 0.1}}


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


class _SecretParams(stage_def.StageParams):
    token: pydantic.SecretStr = pydantic.SecretStr("code-token")
    raw: pydantic.SecretBytes = pydantic.SecretBytes(b"code-bytes")
    keys: list[pydantic.SecretStr] = []
    epochs: int = 3


def test_get_effective_params_records_a_secret_as_a_digest_of_its_value() -> None:
    """A masked secret ('**********') hid a changed value from change detection."""
    first = parameters.get_effective_params(
        _SecretParams(), "p/s", {"p/s": {"token": "first-token", "keys": ["k1"]}}
    )
    second = parameters.get_effective_params(
        _SecretParams(), "p/s", {"p/s": {"token": "second-token", "keys": ["k1"]}}
    )

    assert first["token"] == "sha256:" + hashlib.sha256(b"first-token").hexdigest()[:16]
    assert first["raw"] == "sha256:" + hashlib.sha256(b"code-bytes").hexdigest()[:16]
    assert first["keys"] == ["sha256:" + hashlib.sha256(b"k1").hexdigest()[:16]]
    assert first["epochs"] == 3
    assert first["token"] != second["token"]
    assert first["raw"] == second["raw"]
    text = json.dumps([first, second])
    for plaintext in ("first-token", "second-token", "code-bytes", 'k1"'):
        assert plaintext not in text


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
