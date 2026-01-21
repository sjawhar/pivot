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

import json
import pathlib  # noqa: TC003 - needed for tmp_path type hint
import pickle
from typing import Annotated, TypedDict

import pytest

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


# ==============================================================================
# Test: Extract output specs from return annotation
# ==============================================================================


def test_get_output_specs_from_return_single_output() -> None:
    """Should extract a single output spec from TypedDict return annotation."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process)

    assert len(specs) == 1
    assert "result" in specs
    assert specs["result"].path == "output.json"
    assert isinstance(specs["result"].loader, loaders.JSON)


def test_get_output_specs_from_return_multiple_outputs() -> None:
    """Should extract multiple output specs from TypedDict return annotation."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.get_output_specs_from_return(train)

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

    specs = stage_def.get_output_specs_from_return(process)

    assert specs == {}


def test_get_output_specs_from_return_non_typeddict_returns_empty() -> None:
    """Should return empty dict for non-TypedDict return types."""

    def process() -> dict[str, int]:
        return {"count": 42}

    specs = stage_def.get_output_specs_from_return(process)

    assert specs == {}


def test_get_output_specs_from_return_raises_on_unannotated_fields() -> None:
    """Should raise TypeError for TypedDict fields without Out annotation."""

    def process() -> _InvalidMixedFieldsResult:
        return {"result": {"count": 42}, "extra": "ignored"}

    with pytest.raises(TypeError, match="fields without Out annotations.*'extra'"):
        stage_def.get_output_specs_from_return(process)


# ==============================================================================
# Test: Save return outputs to disk
# ==============================================================================


def test_save_return_outputs_writes_file(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write output files using loaders."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    assert json.loads(output_file.read_text()) == {"count": 42}


def test_save_return_outputs_multiple_files(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write multiple output files."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.get_output_specs_from_return(train)
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

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _NestedPathResult = {"result": {"count": 42}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    output_file = tmp_path / "nested" / "dir" / "output.json"
    assert output_file.exists()


# ==============================================================================
# Test: Path overrides for return outputs
# ==============================================================================


def test_save_return_outputs_with_path_overrides(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should respect path overrides."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    # Override the default path
    overrides = {"result": "custom/path/output.json"}
    stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)

    # Default path should NOT exist
    assert not (tmp_path / "output.json").exists()

    # Custom path should exist
    output_file = tmp_path / "custom" / "path" / "output.json"
    assert output_file.exists()
    assert json.loads(output_file.read_text()) == {"count": 42}


def test_save_return_outputs_partial_path_overrides(tmp_path: pathlib.Path) -> None:
    """Path overrides can be partial - only override some outputs."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.get_output_specs_from_return(train)
    return_value: _MultipleOutputsResult = {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    # Only override model path
    overrides = {"model": "custom/model.pkl"}
    stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)

    # Model should be at custom path
    assert (tmp_path / "custom" / "model.pkl").exists()
    assert not (tmp_path / "model.pkl").exists()

    # Metrics should be at default path
    assert (tmp_path / "metrics.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"accuracy": 0.95}


# ==============================================================================
# Test: List path Out (multiple files per output key)
# ==============================================================================


def test_get_output_specs_from_return_list_path() -> None:
    """Should extract list path from TypedDict return annotation."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)

    assert len(specs) == 1
    assert "items" in specs
    assert specs["items"].path == ["a.json", "b.json"]
    assert isinstance(specs["items"].loader, loaders.JSON)


def test_get_output_specs_from_return_mixed_path_types() -> None:
    """Should handle mixed single and list path types."""

    def process() -> _MixedPathTypesResult:
        return {"single": {"x": 1}, "multi": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)

    assert len(specs) == 2
    assert specs["single"].path == "single.json"
    assert specs["multi"].path == ["m1.json", "m2.json"]


def test_save_return_outputs_list_path(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write list path outputs to multiple files."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)
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

    specs = stage_def.get_output_specs_from_return(process)
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


def test_save_return_outputs_list_path_override(tmp_path: pathlib.Path) -> None:
    """Path overrides for list paths should replace the entire list."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _ListPathResult = {"items": [{"a": 1}, {"b": 2}]}

    # Override with different paths
    overrides = {"items": ["custom/x.json", "custom/y.json"]}
    stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)

    # Default paths should NOT exist
    assert not (tmp_path / "a.json").exists()
    assert not (tmp_path / "b.json").exists()

    # Custom paths should exist
    assert (tmp_path / "custom" / "x.json").exists()
    assert (tmp_path / "custom" / "y.json").exists()


# ==============================================================================
# Test: Validation errors
# ==============================================================================


def test_save_return_outputs_validates_unknown_override_keys(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should raise on unknown keys in path_overrides."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    # Typo in override key
    overrides = {"rsult": "custom.json"}  # typo: 'rsult' instead of 'result'

    with pytest.raises(ValueError, match="Unknown return output names"):
        stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)


def test_save_return_outputs_validates_override_type_mismatch_str_to_list(
    tmp_path: pathlib.Path,
) -> None:
    """save_return_outputs() should raise when string override given for list path."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _ListPathResult = {"items": [{"a": 1}, {"b": 2}]}

    # Wrong type: string override for list path
    overrides = {"items": "single.json"}  # should be list

    with pytest.raises(TypeError, match="spec is sequence, override is str"):
        stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)


def test_save_return_outputs_validates_override_type_mismatch_list_to_str(
    tmp_path: pathlib.Path,
) -> None:
    """save_return_outputs() should raise when list override given for string path."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    # Wrong type: list override for string path
    overrides = {"result": ["a.json", "b.json"]}  # should be string

    with pytest.raises(TypeError, match="spec is str, override is sequence"):
        stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)


def test_save_return_outputs_allows_different_list_length_override(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should allow list override with different length."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)
    # Return value has 3 items to match the 3-path override
    return_value = {"items": [{"a": 1}, {"b": 2}, {"c": 3}]}

    # Override with 3 paths for 2-path spec - allowed for variable-size outputs
    overrides = {"items": ["x.json", "y.json", "z.json"]}

    stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)

    # Verify all three files were written
    assert (tmp_path / "x.json").exists()
    assert (tmp_path / "y.json").exists()
    assert (tmp_path / "z.json").exists()
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert json.loads((tmp_path / "y.json").read_text()) == {"b": 2}
    assert json.loads((tmp_path / "z.json").read_text()) == {"c": 3}


def test_save_return_outputs_validates_missing_keys(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should raise when return value is missing declared keys."""

    def process() -> _MultipleOutputsResult:
        return {"model": b"data", "metrics": {"acc": 0.9}}

    specs = stage_def.get_output_specs_from_return(process)
    # Missing 'metrics' key
    return_value = {"model": b"data"}

    with pytest.raises(KeyError, match="Missing return output keys"):
        stage_def.save_return_outputs(return_value, specs, tmp_path)


def test_save_return_outputs_validates_list_value_length(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should raise when return list length doesn't match paths."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.get_output_specs_from_return(process)
    # Return value has 3 items but spec declares 2 paths
    return_value = {"items": [{"a": 1}, {"b": 2}, {"c": 3}]}

    with pytest.raises(RuntimeError, match="has 2 paths but 3 values"):
        stage_def.save_return_outputs(return_value, specs, tmp_path)


def test_save_return_outputs_validates_path_traversal(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should reject paths that escape project root."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    # Override with path that escapes project root
    overrides = {"result": "../../../etc/passwd"}

    with pytest.raises(ValueError, match="escapes project root"):
        stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)


# ==============================================================================
# Test: Atomic validation (no partial writes on failure)
# ==============================================================================


class _ThreeItemsResult(TypedDict):
    items: Annotated[
        list[dict[str, int]],
        outputs.Out(["a.json", "b.json", "c.json"], loaders.JSON[dict[str, int]]()),
    ]


def test_save_return_outputs_validates_paths_upfront(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should validate all paths before writing any files.

    If the third path is invalid, no files should be written (not even the first two).
    """

    def process() -> _ThreeItemsResult:
        return {"items": [{"a": 1}, {"b": 2}, {"c": 3}]}

    specs = stage_def.get_output_specs_from_return(process)
    return_value: _ThreeItemsResult = {"items": [{"a": 1}, {"b": 2}, {"c": 3}]}

    # Third path escapes project root
    overrides = {"items": ["ok1.json", "ok2.json", "../../../escape.json"]}

    with pytest.raises(ValueError, match="Path escapes project root"):
        stage_def.save_return_outputs(return_value, specs, tmp_path, path_overrides=overrides)

    # No files should have been written (atomic failure)
    assert not (tmp_path / "ok1.json").exists(), (
        "First file should not be written on validation failure"
    )
    assert not (tmp_path / "ok2.json").exists(), (
        "Second file should not be written on validation failure"
    )


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

    specs = stage_def.get_output_specs_from_return(process)
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
