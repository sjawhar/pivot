# pyright: reportUnusedFunction=false, reportPrivateUsage=false
"""Tests for return-based output specifications.

Stage functions can declare outputs via return type annotations:

    class ProcessOutputs(TypedDict):
        result: Annotated[dict[str, int], Out("output.json", JSON())]

    def process(params: ProcessParams) -> ProcessOutputs:
        return {"result": {"count": 42}}

The framework extracts output specs from the return annotation and saves
the return value to disk automatically.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib  # noqa: TC003 - needed for tmp_path type hint
import pickle
from typing import Annotated, TypedDict

import pytest
from typing_extensions import TypedDict as ExtTypedDict

from pivot import loaders, outputs, stage_def

# ==============================================================================
# Module-level TypedDicts for testing (required for type hint resolution)
# ==============================================================================


class _SingleOutputResult(TypedDict):
    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]


class _MultipleOutputsResult(TypedDict):
    model: Annotated[bytes, outputs.Out("model.pkl", loaders.Pickle[bytes]())]
    metrics: Annotated[
        dict[str, float], outputs.Out("metrics.json", loaders.JSON[dict[str, float]]())
    ]


class _InvalidMixedFieldsResult(TypedDict):
    """Invalid: has a field without Out annotation - should raise TypeError."""

    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]
    extra: str  # Not annotated with Out - this is invalid


class _NestedPathResult(TypedDict):
    result: Annotated[
        dict[str, int], outputs.Out("nested/dir/output.json", loaders.JSON[dict[str, int]]())
    ]


class _ListPathResult(TypedDict):
    items: Annotated[
        list[dict[str, int]], outputs.Out(["a.json", "b.json"], loaders.JSON[dict[str, int]]())
    ]


class _MixedPathTypesResult(TypedDict):
    single: Annotated[dict[str, int], outputs.Out("single.json", loaders.JSON[dict[str, int]]())]
    multi: Annotated[
        list[dict[str, int]], outputs.Out(["m1.json", "m2.json"], loaders.JSON[dict[str, int]]())
    ]


# TypedDict from typing_extensions for detection test
class _ExtensionsTypedDictResult(ExtTypedDict):
    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]


# Metadata class that's not an Out - for testing Annotated without Out
class _SomeOtherMetadata:
    pass


# Dataclass for testing invalid return types
@dataclasses.dataclass
class _DataclassResult:
    count: int


# ==============================================================================
# Test: Extract output specs from return annotation
# ==============================================================================


def test_get_output_specs_from_return_single_output() -> None:
    """Should extract a single output spec from TypedDict return annotation."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")

    assert len(specs) == 1
    assert "result" in specs
    assert specs["result"].path == "output.json"
    assert isinstance(specs["result"].loader, loaders.JSON)


def test_get_output_specs_from_return_multiple_outputs() -> None:
    """Should extract multiple output specs from TypedDict return annotation."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.get_output_specs_from_return(train, "test_stage")

    assert len(specs) == 2
    assert "model" in specs
    assert "metrics" in specs
    assert specs["model"].path == "model.pkl"
    assert isinstance(specs["model"].loader, loaders.Pickle)
    assert specs["metrics"].path == "metrics.json"
    assert isinstance(specs["metrics"].loader, loaders.JSON)


def test_get_output_specs_from_return_none_returns_empty() -> None:
    """Should return empty dict for None return type."""

    def process() -> None:
        pass

    specs = stage_def.get_output_specs_from_return(process, "test_stage")

    assert specs == {}


def test_get_output_specs_from_return_non_typeddict_returns_empty() -> None:
    """Non-TypedDict return types should return empty specs (no tracked outputs)."""

    def process() -> dict[str, int]:
        return {"count": 42}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    assert specs == {}


def test_get_output_specs_from_return_raises_on_unannotated_fields() -> None:
    """Should raise StageDefinitionError for TypedDict fields without Out annotation."""
    from pivot import exceptions

    def process() -> _InvalidMixedFieldsResult:
        return {"result": {"count": 42}, "extra": "ignored"}

    with pytest.raises(
        exceptions.StageDefinitionError, match="fields without Out annotations.*extra"
    ):
        stage_def.get_output_specs_from_return(process, "test_stage")


# ==============================================================================
# Test: Save return outputs to disk
# ==============================================================================


def test_save_return_outputs_writes_file(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write output files using loaders."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    assert json.loads(output_file.read_text()) == {"count": 42}


def test_save_return_outputs_multiple_files(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write multiple output files."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.get_output_specs_from_return(train, "test_stage")
    return_value: _MultipleOutputsResult = {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Check model file
    model_file = tmp_path / "model.pkl"
    assert model_file.exists()
    assert pickle.loads(model_file.read_bytes()) == b"model_bytes"

    # Check metrics file
    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists()
    assert json.loads(metrics_file.read_text()) == {"accuracy": 0.95}


def test_save_return_outputs_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should create parent directories."""

    def process() -> _NestedPathResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    return_value: _NestedPathResult = {"result": {"count": 42}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    output_file = tmp_path / "nested" / "dir" / "output.json"
    assert output_file.exists()


# ==============================================================================
# Test: List path Out (multiple files per output key)
# ==============================================================================


def test_get_output_specs_from_return_list_path() -> None:
    """Should extract list path from TypedDict return annotation."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")

    assert len(specs) == 1
    assert "items" in specs
    assert specs["items"].path == ["a.json", "b.json"]
    assert isinstance(specs["items"].loader, loaders.JSON)


def test_get_output_specs_from_return_mixed_path_types() -> None:
    """Should handle mixed single and list path types."""

    def process() -> _MixedPathTypesResult:
        return {"single": {"x": 1}, "multi": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")

    assert len(specs) == 2
    assert specs["single"].path == "single.json"
    assert specs["multi"].path == ["m1.json", "m2.json"]


def test_save_return_outputs_list_path(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write list path outputs to multiple files."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    return_value: _ListPathResult = {"items": [{"a": 1}, {"b": 2}]}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Both files should exist
    assert (tmp_path / "a.json").exists()
    assert (tmp_path / "b.json").exists()

    # Content should match list items
    assert json.loads((tmp_path / "a.json").read_text()) == {"a": 1}
    assert json.loads((tmp_path / "b.json").read_text()) == {"b": 2}


def test_save_return_outputs_mixed_path_types(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should handle mixed single and list paths."""

    def process() -> _MixedPathTypesResult:
        return {"single": {"x": 1}, "multi": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    return_value: _MixedPathTypesResult = {"single": {"x": 1}, "multi": [{"a": 1}, {"b": 2}]}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Single path output
    assert (tmp_path / "single.json").exists()
    assert json.loads((tmp_path / "single.json").read_text()) == {"x": 1}

    # List path outputs
    assert (tmp_path / "m1.json").exists()
    assert (tmp_path / "m2.json").exists()
    assert json.loads((tmp_path / "m1.json").read_text()) == {"a": 1}
    assert json.loads((tmp_path / "m2.json").read_text()) == {"b": 2}


# ==============================================================================
# Test: Validation errors
# ==============================================================================


def test_save_return_outputs_validates_missing_keys(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should raise when return value is missing declared keys."""

    def process() -> _MultipleOutputsResult:
        return {"model": b"data", "metrics": {"acc": 0.9}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    # Missing 'metrics' key
    return_value = {"model": b"data"}

    with pytest.raises(KeyError, match="Missing return output keys"):
        stage_def.save_return_outputs(return_value, specs, tmp_path)


def test_save_return_outputs_validates_list_value_length(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should raise when return list length doesn't match paths."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    # Return value has 3 items but spec declares 2 paths
    return_value = {"items": [{"a": 1}, {"b": 2}, {"c": 3}]}

    with pytest.raises(RuntimeError, match="has 2 paths but 3 values"):
        stage_def.save_return_outputs(return_value, specs, tmp_path)


# ==============================================================================
# Test: Extra keys warning
# ==============================================================================


def test_save_return_outputs_warns_on_extra_keys(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """save_return_outputs() should warn when return value has extra keys."""
    import logging

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    # Return value has extra keys not declared as outputs
    return_value = {"result": {"count": 42}, "undeclared": "data", "another": 123}

    with caplog.at_level(logging.WARNING):
        stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Output should still be written
    assert (tmp_path / "output.json").exists()

    # Warning should be logged
    assert "Extra keys in return value not declared as outputs" in caplog.text
    assert "another" in caplog.text
    assert "undeclared" in caplog.text


# ==============================================================================
# Test: TypedDict detection and strict return type validation
# ==============================================================================


def test_typing_extensions_typeddict_detected() -> None:
    """TypedDict from typing_extensions should be detected correctly."""

    def process() -> _ExtensionsTypedDictResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")

    assert len(specs) == 1
    assert "result" in specs
    assert specs["result"].path == "output.json"


def test_plain_dict_return_returns_empty() -> None:
    """Plain dict return type (not TypedDict) should return empty specs (no tracked outputs)."""

    def process() -> dict[str, int]:
        return {"count": 42}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    assert specs == {}


def test_annotated_without_out_returns_empty() -> None:
    """Annotated return type without Out should return empty specs (no tracked outputs)."""

    def process() -> Annotated[dict[str, int], _SomeOtherMetadata()]:
        return {"count": 42}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    assert specs == {}


def test_dataclass_return_returns_empty() -> None:
    """Dataclass return type should return empty specs (no tracked outputs)."""

    def process() -> _DataclassResult:
        return _DataclassResult(count=42)

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    assert specs == {}


def test_partial_out_annotations_raises_error() -> None:
    """TypedDict with only some fields having Out annotations should raise StageDefinitionError."""
    from pivot import exceptions

    # Use the module-level _InvalidMixedFieldsResult which has one Out and one plain field
    def process() -> _InvalidMixedFieldsResult:
        return {"result": {"count": 42}, "extra": "ignored"}

    with pytest.raises(
        exceptions.StageDefinitionError, match="fields without Out annotations.*extra"
    ):
        stage_def.get_output_specs_from_return(process, "test_stage")


def test_error_message_includes_stage_name() -> None:
    """Error messages should include the stage name for easier debugging.

    This tests that validation errors in TypedDict output annotations include the stage name.
    """
    from pivot import exceptions

    def my_custom_stage() -> _InvalidMixedFieldsResult:
        return {"result": {"count": 42}, "extra": "ignored"}

    with pytest.raises(exceptions.StageDefinitionError, match="Stage 'my_custom_stage'"):
        stage_def.get_output_specs_from_return(my_custom_stage, "my_custom_stage")
