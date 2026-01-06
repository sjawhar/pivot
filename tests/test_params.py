"""Tests for parameter loading and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

from pivot import params

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
    result = params.load_params_yaml()
    assert result == {}, "Missing params.yaml should return empty dict"


def test_load_params_yaml_from_explicit_path(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("train:\n  learning_rate: 0.001\n  epochs: 200\n")

    result = params.load_params_yaml(params_file)
    assert result == {"train": {"learning_rate": 0.001, "epochs": 200}}


def test_load_params_yaml_from_project_root(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    mocker.patch("pivot.project.get_project_root", return_value=tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("stage1:\n  lr: 0.01\nstage2:\n  batch: 64\n")

    result = params.load_params_yaml()
    assert result == {"stage1": {"lr": 0.01}, "stage2": {"batch": 64}}


def test_load_params_yaml_non_dict_root(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("- item1\n- item2\n")

    result = params.load_params_yaml(params_file)
    assert result == {}, "Non-dict root should return empty dict"


def test_load_params_yaml_filters_non_dict_values(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("valid:\n  key: value\ninvalid: just_a_string\n")

    result = params.load_params_yaml(params_file)
    assert result == {"valid": {"key": "value"}}, "Non-dict stage values should be filtered"


def test_load_params_yaml_invalid_yaml(tmp_path: pathlib.Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("invalid: yaml: content: ::::")

    result = params.load_params_yaml(params_file)
    assert result == {}, "Invalid YAML should return empty dict"


# -----------------------------------------------------------------------------
# build_params_instance tests
# -----------------------------------------------------------------------------


def test_build_params_instance_defaults_only() -> None:
    instance = params.build_params_instance(TrainParams, "train", None)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.01
    assert instance.epochs == 100
    assert instance.batch_size == 32


def test_build_params_instance_with_yaml_overrides() -> None:
    yaml_overrides = {"train": {"learning_rate": 0.001, "epochs": 200}}
    instance = params.build_params_instance(TrainParams, "train", yaml_overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.001, "YAML should override default"
    assert instance.epochs == 200, "YAML should override default"
    assert instance.batch_size == 32, "Unspecified fields keep defaults"


def test_build_params_instance_missing_stage_in_yaml() -> None:
    yaml_overrides = {"other_stage": {"lr": 0.1}}
    instance = params.build_params_instance(TrainParams, "train", yaml_overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.01, "Should use defaults when stage not in YAML"


def test_build_params_instance_extra_yaml_fields_ignored() -> None:
    yaml_overrides = {"train": {"learning_rate": 0.002, "extra_field": "ignored"}}
    instance = params.build_params_instance(TrainParams, "train", yaml_overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.002
    assert not hasattr(instance, "extra_field"), "Extra fields should be ignored"


def test_build_params_instance_required_field_from_yaml() -> None:
    yaml_overrides = {"process": {"name": "my_process"}}
    instance = params.build_params_instance(RequiredParams, "process", yaml_overrides)
    assert isinstance(instance, RequiredParams)
    assert instance.name == "my_process"
    assert instance.value == 10


def test_build_params_instance_required_field_missing_raises() -> None:
    with pytest.raises(ValidationError):
        params.build_params_instance(RequiredParams, "process", {})


def test_build_params_instance_type_mismatch_raises() -> None:
    yaml_overrides = {"train": {"epochs": "not_an_int"}}
    with pytest.raises(ValidationError):
        params.build_params_instance(TrainParams, "train", yaml_overrides)


def test_build_params_instance_nested_model() -> None:
    yaml_overrides = {"complex": {"debug": True}}
    instance = params.build_params_instance(ComplexParams, "complex", yaml_overrides)
    assert isinstance(instance, ComplexParams)
    assert instance.debug is True
    assert instance.training.lr == 0.001
    assert instance.training.optimizer == "adam"


# -----------------------------------------------------------------------------
# params_to_dict tests
# -----------------------------------------------------------------------------


def test_params_to_dict_simple() -> None:
    instance = TrainParams(learning_rate=0.05, epochs=50, batch_size=16)
    result = params.params_to_dict(instance)
    assert result == {"learning_rate": 0.05, "epochs": 50, "batch_size": 16}


def test_params_to_dict_nested() -> None:
    instance = ComplexParams(debug=True, training=NestedParams(lr=0.01, optimizer="sgd"))
    result = params.params_to_dict(instance)
    assert result == {
        "debug": True,
        "training": {"lr": 0.01, "optimizer": "sgd"},
    }


# -----------------------------------------------------------------------------
# validate_params_cls tests
# -----------------------------------------------------------------------------


def test_validate_params_cls_with_basemodel() -> None:
    assert params.validate_params_cls(TrainParams) is True


def test_validate_params_cls_with_non_class() -> None:
    assert params.validate_params_cls("not a class") is False


def test_validate_params_cls_with_regular_class() -> None:
    class RegularClass:
        pass

    assert params.validate_params_cls(RegularClass) is False


def test_validate_params_cls_with_dict() -> None:
    assert params.validate_params_cls(dict) is False


# -----------------------------------------------------------------------------
# extract_stage_params tests
# -----------------------------------------------------------------------------


def test_extract_stage_params_with_pydantic_model() -> None:
    yaml_overrides = {"train": {"learning_rate": 0.005}}
    params_dict, instance = params.extract_stage_params(TrainParams, None, "train", yaml_overrides)
    assert params_dict == {"learning_rate": 0.005, "epochs": 100, "batch_size": 32}
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.005


def test_extract_stage_params_with_signature_defaults() -> None:
    import inspect

    def stage_func(lr: float = 0.01, epochs: int = 100) -> None:
        pass

    sig = inspect.signature(stage_func)
    params_dict, instance = params.extract_stage_params(None, sig, "train", None)
    assert params_dict == {"lr": 0.01, "epochs": 100}
    assert instance is None, "Legacy signature path returns no instance"


def test_extract_stage_params_no_params_cls_no_signature() -> None:
    params_dict, instance = params.extract_stage_params(None, None, "train", None)
    assert params_dict == {}
    assert instance is None


def test_extract_stage_params_pydantic_takes_precedence() -> None:
    import inspect

    def stage_func(lr: float = 0.99) -> None:
        pass

    sig = inspect.signature(stage_func)
    yaml_overrides = {"train": {"learning_rate": 0.002}}
    params_dict, instance = params.extract_stage_params(TrainParams, sig, "train", yaml_overrides)
    assert params_dict == {"learning_rate": 0.002, "epochs": 100, "batch_size": 32}
    assert isinstance(instance, TrainParams), "Pydantic model takes precedence over signature"


def test_extract_stage_params_signature_filters_empty_defaults() -> None:
    import inspect

    def stage_func(required_arg: str, optional: int = 42) -> None:
        pass

    sig = inspect.signature(stage_func)
    params_dict, instance = params.extract_stage_params(None, sig, "test", None)
    assert params_dict == {"optional": 42}, "Only params with defaults are extracted"
    assert "required_arg" not in params_dict
    assert instance is None
