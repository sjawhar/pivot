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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs

    assert len(specs) == 1
    assert "result" in specs
    assert specs["result"].path == "output.json"
    assert isinstance(specs["result"].loader, loaders.JSON)


def test_get_output_specs_from_return_multiple_outputs() -> None:
    """Should extract multiple output specs from TypedDict return annotation."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.extract_stage_definition(train, "test_stage").out_specs

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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs

    assert specs == {}


def test_get_output_specs_from_return_non_typeddict_returns_empty() -> None:
    """Non-TypedDict return types should return empty specs (no tracked outputs)."""

    def process() -> dict[str, int]:
        return {"count": 42}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
    assert specs == {}


def test_get_output_specs_from_return_raises_on_unannotated_fields() -> None:
    """Should raise StageDefinitionError for TypedDict fields without Out annotation."""
    from pivot import exceptions

    def process() -> _InvalidMixedFieldsResult:
        return {"result": {"count": 42}, "extra": "ignored"}

    with pytest.raises(
        exceptions.StageDefinitionError, match="fields without Out annotations.*extra"
    ):
        stage_def.extract_stage_definition(process, "test_stage")


# ==============================================================================
# Test: Save return outputs to disk
# ==============================================================================


def test_save_return_outputs_writes_file(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write output files using loaders."""

    def process() -> _SingleOutputResult:
        return {"result": {"count": 42}}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
    return_value: _SingleOutputResult = {"result": {"count": 42}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    assert json.loads(output_file.read_text()) == {"count": 42}


def test_save_return_outputs_multiple_files(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write multiple output files."""

    def train() -> _MultipleOutputsResult:
        return {"model": b"model_bytes", "metrics": {"accuracy": 0.95}}

    specs = stage_def.extract_stage_definition(train, "test_stage").out_specs
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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs

    assert len(specs) == 1
    assert "items" in specs
    assert specs["items"].path == ["a.json", "b.json"]
    assert isinstance(specs["items"].loader, loaders.JSON)


def test_get_output_specs_from_return_mixed_path_types() -> None:
    """Should handle mixed single and list path types."""

    def process() -> _MixedPathTypesResult:
        return {"single": {"x": 1}, "multi": [{"a": 1}, {"b": 2}]}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs

    assert len(specs) == 2
    assert specs["single"].path == "single.json"
    assert specs["multi"].path == ["m1.json", "m2.json"]


def test_save_return_outputs_list_path(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write list path outputs to multiple files."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
    # Missing 'metrics' key
    return_value = {"model": b"data"}

    with pytest.raises(KeyError, match="Missing return output keys"):
        stage_def.save_return_outputs(return_value, specs, tmp_path)


def test_save_return_outputs_validates_list_value_length(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should raise when return list length doesn't match paths."""

    def process() -> _ListPathResult:
        return {"items": [{"a": 1}, {"b": 2}]}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
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

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs

    assert len(specs) == 1
    assert "result" in specs
    assert specs["result"].path == "output.json"


def test_plain_dict_return_returns_empty() -> None:
    """Plain dict return type (not TypedDict) should return empty specs (no tracked outputs)."""

    def process() -> dict[str, int]:
        return {"count": 42}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
    assert specs == {}


def test_annotated_without_out_returns_empty() -> None:
    """Annotated return type without Out should return empty specs (no tracked outputs)."""

    def process() -> Annotated[dict[str, int], _SomeOtherMetadata()]:
        return {"count": 42}

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
    assert specs == {}


def test_dataclass_return_returns_empty() -> None:
    """Dataclass return type should return empty specs (no tracked outputs)."""

    def process() -> _DataclassResult:
        return _DataclassResult(count=42)

    specs = stage_def.extract_stage_definition(process, "test_stage").out_specs
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
        stage_def.extract_stage_definition(process, "test_stage")


def test_error_message_includes_stage_name() -> None:
    """Error messages should include the stage name for easier debugging.

    This tests that validation errors in TypedDict output annotations include the stage name.
    """
    from pivot import exceptions

    def my_custom_stage() -> _InvalidMixedFieldsResult:
        return {"result": {"count": 42}, "extra": "ignored"}

    with pytest.raises(exceptions.StageDefinitionError, match="Stage 'my_custom_stage'"):
        stage_def.extract_stage_definition(my_custom_stage, "my_custom_stage")


# ==============================================================================
# Test: DirectoryOut in save_return_outputs
# ==============================================================================


def test_save_return_outputs_directory_out_writes_files(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should write DirectoryOut files using the loader."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {
        "task_results": {
            "task_a.json": {"accuracy": 95},
            "task_b.json": {"accuracy": 87},
        }
    }

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Check both files were created
    task_a = tmp_path / "results" / "task_a.json"
    task_b = tmp_path / "results" / "task_b.json"
    assert task_a.exists(), "task_a.json should exist"
    assert task_b.exists(), "task_b.json should exist"
    assert json.loads(task_a.read_text()) == {"accuracy": 95}
    assert json.loads(task_b.read_text()) == {"accuracy": 87}


def test_save_return_outputs_directory_out_creates_subdirs(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should create subdirectories for nested keys."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {
        "task_results": {
            "subdir/nested.json": {"value": 42},
        }
    }

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    nested = tmp_path / "results" / "subdir" / "nested.json"
    assert nested.exists(), "Nested file should be created"
    assert json.loads(nested.read_text()) == {"value": 42}


def test_save_return_outputs_directory_out_empty_dict_raises() -> None:
    """save_return_outputs() should raise ValueError for empty DirectoryOut dict."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value: dict[str, dict[str, dict[str, int]]] = {"task_results": {}}

    with pytest.raises(ValueError, match="dict must be non-empty"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_path_traversal_raises() -> None:
    """save_return_outputs() should raise ValueError for keys with path traversal."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": {"../escape.json": {"malicious": 1}}}

    with pytest.raises(ValueError, match="path traversal not allowed"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_absolute_path_raises() -> None:
    """save_return_outputs() should raise ValueError for absolute path keys."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": {"/etc/passwd.json": {"bad": 1}}}

    with pytest.raises(ValueError, match="absolute path not allowed"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_no_extension_raises() -> None:
    """save_return_outputs() should raise ValueError for keys without file extension."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": {"no_extension": {"value": 1}}}

    with pytest.raises(ValueError, match="must include file extension"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_empty_key_raises() -> None:
    """save_return_outputs() should raise ValueError for empty string key."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": {"": {"value": 1}}}

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_duplicate_after_normalization_raises() -> None:
    """save_return_outputs() should raise ValueError for keys that normalize to duplicates."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # Both keys normalize to "foo/bar.json"
    return_value = {"task_results": {"foo/bar.json": {"a": 1}, "foo//bar.json": {"b": 2}}}

    with pytest.raises(ValueError, match="duplicate key after normalization"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_non_dict_raises() -> None:
    """save_return_outputs() should raise RuntimeError for non-dict DirectoryOut value."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": [{"a": 1}]}  # List instead of dict

    with pytest.raises(RuntimeError, match="expects dict"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_non_string_key_raises() -> None:
    """save_return_outputs() should raise ValueError for non-string keys."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": {123: {"a": 1}}}  # type: ignore[dict-item] # Int key instead of string

    with pytest.raises(ValueError, match="keys must be strings"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_whitespace_only_key_raises() -> None:
    """save_return_outputs() should raise ValueError for whitespace-only filename."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    return_value = {"task_results": {"   .json": {"a": 1}}}  # Whitespace-only filename

    with pytest.raises(ValueError, match="filename cannot be empty or whitespace-only"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_unicode_keys(tmp_path: pathlib.Path) -> None:
    """save_return_outputs() should handle Unicode keys correctly (NFC normalized)."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # Use Unicode characters in keys
    return_value = {
        "task_results": {
            "données.json": {"count": 1},  # French accented chars
            "数据.json": {"count": 2},  # Chinese chars
            "café.json": {"count": 3},  # More accents
        }
    }

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Verify all files were created with correct Unicode names
    results_dir = tmp_path / "results"
    assert (results_dir / "données.json").exists()
    assert (results_dir / "数据.json").exists()
    assert (results_dir / "café.json").exists()

    # Verify contents
    with open(results_dir / "données.json") as f:
        assert json.load(f) == {"count": 1}


def test_save_return_outputs_directory_out_hidden_file_without_extension_raises() -> None:
    """save_return_outputs() should reject hidden files without real extensions."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # .hiddenfile has a leading dot but no extension (suffix is empty)
    return_value = {"task_results": {".hiddenfile": {"a": 1}}}

    with pytest.raises(ValueError, match="must include file extension"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_hidden_file_with_extension_allowed(
    tmp_path: pathlib.Path,
) -> None:
    """save_return_outputs() should allow hidden files that have real extensions."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # .metadata.json has a leading dot AND a .json extension
    return_value = {"task_results": {".metadata.json": {"version": 1}}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Verify hidden file was created
    assert (tmp_path / "results" / ".metadata.json").exists()


def test_save_return_outputs_directory_out_path_normalization_edge_cases(
    tmp_path: pathlib.Path,
) -> None:
    """save_return_outputs() normalizes paths correctly (redundant separators, etc)."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # Keys with redundant separators and leading ./ should be normalized
    return_value = {
        "task_results": {
            "foo//bar.json": {"a": 1},  # Double slash -> foo/bar.json
            "./baz.json": {"b": 2},  # Leading ./ -> baz.json
            "nested/./file.json": {"c": 3},  # Embedded ./ -> nested/file.json
        }
    }

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    # Verify files are created at normalized paths
    results_dir = tmp_path / "results"
    assert (results_dir / "foo" / "bar.json").exists()
    assert (results_dir / "baz.json").exists()
    assert (results_dir / "nested" / "file.json").exists()


def test_save_return_outputs_directory_out_path_normalization_duplicate_detection() -> None:
    """save_return_outputs() detects duplicates after path normalization."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # These keys normalize to the same path
    return_value = {
        "task_results": {
            "foo/bar.json": {"a": 1},
            "foo//bar.json": {"b": 2},  # Same as foo/bar.json after normalization
        }
    }

    with pytest.raises(ValueError, match="duplicate key after normalization"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_case_collision_raises() -> None:
    """save_return_outputs() should raise ValueError for keys that differ only by case."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # These keys would collide on case-insensitive filesystems (macOS, Windows)
    return_value = {
        "task_results": {
            "File.json": {"a": 1},
            "file.json": {"b": 2},
        }
    }

    with pytest.raises(ValueError, match="case-insensitive filesystems"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))


def test_save_return_outputs_directory_out_case_collision_nested_raises() -> None:
    """save_return_outputs() should raise ValueError for nested paths that differ only by case."""
    specs: dict[str, outputs.BaseOut] = {
        "task_results": outputs.DirectoryOut("results/", loaders.JSON[dict[str, int]]())
    }
    # Nested paths that would collide on case-insensitive filesystems
    return_value = {
        "task_results": {
            "Foo/bar.json": {"a": 1},
            "foo/bar.json": {"b": 2},
        }
    }

    with pytest.raises(ValueError, match="case-insensitive filesystems"):
        stage_def.save_return_outputs(return_value, specs, pathlib.Path("/tmp"))
