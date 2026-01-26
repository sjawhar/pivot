from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

from pivot import exceptions, parameters

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


# -----------------------------------------------------------------------------
# load_params_yaml tests
# -----------------------------------------------------------------------------


def test_load_params_yaml_missing_file(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("pivot.project.get_project_root", return_value=tmp_path)
    result = parameters.load_params_yaml()
    assert result == {}, "Missing params.yaml should return empty dict"


def test_load_params_yaml_from_explicit_path(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("train:\n  learning_rate: 0.001\n  epochs: 200\n")

    result = parameters.load_params_yaml(params_file)
    assert result == {"train": {"learning_rate": 0.001, "epochs": 200}}


def test_load_params_yaml_from_project_root(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("pivot.project.get_project_root", return_value=tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("stage1:\n  lr: 0.01\nstage2:\n  batch: 64\n")

    result = parameters.load_params_yaml()
    assert result == {"stage1": {"lr": 0.01}, "stage2": {"batch": 64}}


def test_load_params_yaml_non_dict_root_raises(tmp_path: Path) -> None:
    """Non-dict root should raise ParamsError."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("- item1\n- item2\n")

    with pytest.raises(exceptions.ParamsError, match="root must be a dict"):
        parameters.load_params_yaml(params_file)


def test_load_params_yaml_filters_non_dict_values(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("valid:\n  key: value\ninvalid: just_a_string\n")

    result = parameters.load_params_yaml(params_file)
    assert result == {"valid": {"key": "value"}}, "Non-dict stage values should be filtered"


def test_load_params_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    """Invalid YAML should raise ParamsError, not silently return empty dict."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("invalid: yaml: content: ::::")

    with pytest.raises(exceptions.ParamsError, match="Failed to parse"):
        parameters.load_params_yaml(params_file)


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


def test_build_params_instance_with_overrides() -> None:
    """Alias for test_build_params_instance_with_yaml_overrides for compatibility."""
    overrides = {"train": {"learning_rate": 0.001, "epochs": 200}}
    instance = parameters.build_params_instance(TrainParams, "train", overrides)
    assert isinstance(instance, TrainParams)
    assert instance.learning_rate == 0.001, "YAML should override default"
    assert instance.epochs == 200, "YAML should override default"
    assert instance.batch_size == 32, "Unspecified fields keep defaults"


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
