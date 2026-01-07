"""Tests for parameter loading and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

from pivot import parameters

if TYPE_CHECKING:
    import pathlib

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


# -----------------------------------------------------------------------------
# load_params_yaml tests
# -----------------------------------------------------------------------------


def test_load_params_yaml_missing_file(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    mocker.patch("pivot.project.get_project_root", return_value=tmp_path)
    result = parameters.load_params_yaml()
    assert result == {}, "Missing params.yaml should return empty dict"


def test_load_params_yaml_from_explicit_path(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("train:\n  learning_rate: 0.001\n  epochs: 200\n")

    result = parameters.load_params_yaml(params_file)
    assert result == {"train": {"learning_rate": 0.001, "epochs": 200}}


def test_load_params_yaml_from_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    mocker.patch("pivot.project.get_project_root", return_value=tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("stage1:\n  lr: 0.01\nstage2:\n  batch: 64\n")

    result = parameters.load_params_yaml()
    assert result == {"stage1": {"lr": 0.01}, "stage2": {"batch": 64}}


def test_load_params_yaml_non_dict_root(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("- item1\n- item2\n")

    result = parameters.load_params_yaml(params_file)
    assert result == {}, "Non-dict root should return empty dict"


def test_load_params_yaml_filters_non_dict_values(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("valid:\n  key: value\ninvalid: just_a_string\n")

    result = parameters.load_params_yaml(params_file)
    assert result == {"valid": {"key": "value"}}, "Non-dict stage values should be filtered"


def test_load_params_yaml_invalid_yaml(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("invalid: yaml: content: ::::")

    result = parameters.load_params_yaml(params_file)
    assert result == {}, "Invalid YAML should return empty dict"


# -----------------------------------------------------------------------------
# build_params_instance tests
# -----------------------------------------------------------------------------


def test_build_params_instance_defaults_only() -> None:
    instance = parameters.build_params_instance(TrainParams, "train", None)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.01
    assert instance.epochs == 100
    assert instance.batch_size == 32


def test_build_params_instance_with_overrides() -> None:
    overrides = {"train": {"learning_rate": 0.001, "epochs": 200}}
    instance = parameters.build_params_instance(TrainParams, "train", overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.001, "YAML should override default"
    assert instance.epochs == 200, "YAML should override default"
    assert instance.batch_size == 32, "Unspecified fields keep defaults"


def test_build_params_instance_missing_stage_in_yaml() -> None:
    overrides = {"other_stage": {"lr": 0.1}}
    instance = parameters.build_params_instance(TrainParams, "train", overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.01, "Should use defaults when stage not in YAML"


def test_build_params_instance_extra_yaml_fields_ignored() -> None:
    overrides = {"train": {"learning_rate": 0.002, "extra_field": "ignored"}}
    instance = parameters.build_params_instance(TrainParams, "train", overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.002
    assert not hasattr(instance, "extra_field"), "Extra fields should be ignored"


def test_build_params_instance_required_field_from_yaml() -> None:
    overrides = {"process": {"name": "my_process"}}
    instance = parameters.build_params_instance(RequiredParams, "process", overrides)
    assert isinstance(instance, RequiredParams)
    assert instance.name == "my_process"
    assert instance.value == 10


def test_build_params_instance_required_field_missing_raises() -> None:
    with pytest.raises(ValidationError):
        parameters.build_params_instance(RequiredParams, "process", {})


def test_build_params_instance_type_mismatch_raises() -> None:
    overrides = {"train": {"epochs": "not_an_int"}}
    with pytest.raises(ValidationError):
        parameters.build_params_instance(TrainParams, "train", overrides)


def test_build_params_instance_nested_model() -> None:
    overrides = {"complex": {"debug": True}}
    instance = parameters.build_params_instance(ComplexParams, "complex", overrides)
    assert isinstance(instance, ComplexParams)
    assert instance.debug is True
    assert instance.training.lr == 0.001
    assert instance.training.optimizer == "adam"


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
# get_effective_params tests
# -----------------------------------------------------------------------------


def test_get_effective_params_with_instance() -> None:
    instance = TrainParams(learning_rate=0.01)
    overrides = {"train": {"learning_rate": 0.005}}
    result = parameters.get_effective_params(instance, "train", overrides)
    assert result == {"learning_rate": 0.005, "epochs": 100, "batch_size": 32}


def test_get_effective_params_no_instance() -> None:
    result = parameters.get_effective_params(None, "train", None)
    assert result == {}


def test_get_effective_params_with_overrides() -> None:
    instance = TrainParams()
    overrides = {"train": {"learning_rate": 0.002}}
    result = parameters.get_effective_params(instance, "train", overrides)
    assert result == {"learning_rate": 0.002, "epochs": 100, "batch_size": 32}


# -----------------------------------------------------------------------------
# apply_overrides tests (deep merging)
# -----------------------------------------------------------------------------


def test_apply_overrides_no_overrides() -> None:
    instance = TrainParams(learning_rate=0.05)
    result = parameters.apply_overrides(instance, "train", None)
    assert result.learning_rate == 0.05, "No overrides should return unchanged"


def test_apply_overrides_empty_stage() -> None:
    instance = TrainParams(learning_rate=0.05)
    result = parameters.apply_overrides(instance, "train", {"other": {"lr": 0.1}})
    assert result.learning_rate == 0.05, "Missing stage should return unchanged"


def test_apply_overrides_shallow_field() -> None:
    instance = TrainParams(learning_rate=0.05, epochs=50)
    result = parameters.apply_overrides(instance, "train", {"train": {"epochs": 200}})
    assert result.learning_rate == 0.05, "Unspecified field preserved"
    assert result.epochs == 200, "Specified field overridden"


def test_apply_overrides_deep_merge_nested() -> None:
    instance = ComplexParams(training=NestedParams(lr=0.001, optimizer="adam"), debug=False)
    overrides = {"train": {"training": {"lr": 0.01}}}
    result = parameters.apply_overrides(instance, "train", overrides)

    assert result.training.lr == 0.01, "Nested field should be overridden"
    assert result.training.optimizer == "adam", "Unspecified nested field preserved"
    assert result.debug is False, "Top-level field preserved"


def test_apply_overrides_deep_merge_multiple_nested_fields() -> None:
    instance = ComplexParams(training=NestedParams(lr=0.001, optimizer="adam"), debug=False)
    overrides = {"train": {"training": {"lr": 0.01, "optimizer": "sgd"}, "debug": True}}
    result = parameters.apply_overrides(instance, "train", overrides)

    assert result.training.lr == 0.01
    assert result.training.optimizer == "sgd"
    assert result.debug is True


def test_apply_overrides_replace_nested_entirely() -> None:
    """When override is not a dict, replace the entire nested model."""
    instance = ComplexParams(training=NestedParams(lr=0.001, optimizer="adam"))
    # Pass a complete nested dict - should work as full replacement
    overrides = {"train": {"training": {"lr": 0.05, "optimizer": "rmsprop"}}}
    result = parameters.apply_overrides(instance, "train", overrides)

    assert result.training.lr == 0.05
    assert result.training.optimizer == "rmsprop"


# -----------------------------------------------------------------------------
# Matrix stage override inheritance tests
# -----------------------------------------------------------------------------


def test_apply_overrides_matrix_base_name_applies_to_variant() -> None:
    """Base stage name override applies to all matrix variants."""
    instance = TrainParams(learning_rate=0.01, epochs=100)
    overrides = {"train": {"learning_rate": 0.05}}

    # Variant name includes @, base name override should apply
    result = parameters.apply_overrides(instance, "train@v1", overrides)
    assert result.learning_rate == 0.05, "Base name override should apply to variant"
    assert result.epochs == 100, "Other fields unchanged"


def test_apply_overrides_matrix_variant_specific() -> None:
    """Variant-specific override only applies to that variant."""
    instance = TrainParams(learning_rate=0.01, epochs=100)
    overrides = {"train@v1": {"learning_rate": 0.05}}

    # Exact variant match
    result_v1 = parameters.apply_overrides(instance, "train@v1", overrides)
    assert result_v1.learning_rate == 0.05

    # Different variant - no override
    result_v2 = parameters.apply_overrides(instance, "train@v2", overrides)
    assert result_v2.learning_rate == 0.01


def test_apply_overrides_matrix_variant_overrides_base() -> None:
    """Variant-specific override takes precedence over base name."""
    instance = TrainParams(learning_rate=0.01, epochs=100, batch_size=32)
    overrides = {
        "train": {"learning_rate": 0.05, "epochs": 200},  # Base applies to all
        "train@v1": {"learning_rate": 0.1},  # Variant-specific overrides base
    }

    result = parameters.apply_overrides(instance, "train@v1", overrides)
    assert result.learning_rate == 0.1, "Variant-specific should override base"
    assert result.epochs == 200, "Base name override still applies for non-overridden fields"
    assert result.batch_size == 32, "Default preserved"


def test_apply_overrides_matrix_nested_deep_merge() -> None:
    """Deep merge works with matrix stage inheritance."""
    instance = ComplexParams(training=NestedParams(lr=0.001, optimizer="adam"), debug=False)
    overrides = {
        "train": {"training": {"lr": 0.01}},  # Base: change lr
        "train@v1": {"debug": True},  # Variant: change debug
    }

    result = parameters.apply_overrides(instance, "train@v1", overrides)
    assert result.training.lr == 0.01, "Base name nested override applied"
    assert result.training.optimizer == "adam", "Nested field preserved"
    assert result.debug is True, "Variant-specific override applied"


def test_apply_overrides_non_matrix_stage_unchanged() -> None:
    """Regular stages (no @) work as before."""
    instance = TrainParams(learning_rate=0.01, epochs=100)
    overrides = {"train": {"learning_rate": 0.05}}

    result = parameters.apply_overrides(instance, "train", overrides)
    assert result.learning_rate == 0.05
