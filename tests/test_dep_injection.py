# pyright: reportUnusedFunction=false, reportPrivateUsage=false
"""Tests for dependency injection in stage functions.

This module tests the new pattern where dependencies are declared as function
arguments with Annotated type hints, enabling trivial testing:

    # Testing is natural - just pass the data directly:
    result = train(TrainParams(), test_df)

The framework handles injection at runtime using the Dep/Out annotations.
"""

from __future__ import annotations

import json
import pathlib  # noqa: TC003 - needed for tmp_path type hint
from typing import Annotated, TypedDict

import pandas
import pytest

from pivot import loaders, outputs, stage_def
from pivot.types import OutputMessage  # noqa: TC001 - needed at runtime for Queue

# ==============================================================================
# Module-level types for testing
# ==============================================================================


class _TrainParams(stage_def.StageParams):
    """Simple params class for testing."""

    learning_rate: float = 0.01


class _TrainOutputs(TypedDict):
    model: Annotated[dict[str, float], outputs.Out("model.json", loaders.JSON[dict[str, float]]())]


class _ProcessOutputs(TypedDict):
    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]


class _MultiDepOutputs(TypedDict):
    combined: Annotated[
        dict[str, int], outputs.Out("combined.json", loaders.JSON[dict[str, int]]())
    ]


# TypedDicts for Out annotation tests (must be module-level for get_type_hints)
class _OutTestOutputs(TypedDict):
    model: Annotated[dict[str, float], outputs.Out("model.json", loaders.JSON[dict[str, float]]())]
    metrics: Annotated[
        dict[str, float], outputs.Out("metrics.json", loaders.JSON[dict[str, float]]())
    ]


class _SingleOutTestOutputs(TypedDict):
    result: Annotated[dict[str, int], outputs.Out("output.json", loaders.JSON[dict[str, int]]())]


# ==============================================================================
# Test: Dep extraction from function signature
# ==============================================================================


def test_get_dep_specs_single_dep() -> None:
    """Should extract single Dep from function signature."""

    def process(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    specs = stage_def.get_dep_specs_from_signature(process)

    assert len(specs) == 1
    assert "data" in specs
    assert specs["data"].path == "input.csv"
    assert isinstance(specs["data"].loader, loaders.CSV)


def test_get_dep_specs_multiple_deps() -> None:
    """Should extract multiple Deps from function signature."""

    def process(
        train: Annotated[
            pandas.DataFrame, outputs.Dep("train.csv", loaders.CSV[pandas.DataFrame]())
        ],
        test: Annotated[pandas.DataFrame, outputs.Dep("test.csv", loaders.CSV[pandas.DataFrame]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"train": len(train), "test": len(test)}}

    specs = stage_def.get_dep_specs_from_signature(process)

    assert len(specs) == 2
    assert "train" in specs
    assert "test" in specs
    assert specs["train"].path == "train.csv"
    assert specs["test"].path == "test.csv"


def test_get_dep_specs_mixed_with_params() -> None:
    """Should extract Deps while ignoring non-Dep arguments."""

    def train(
        config: _TrainParams,
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _TrainOutputs:
        return {"model": {"weights": config.learning_rate}}

    specs = stage_def.get_dep_specs_from_signature(train)

    # Only 'data' should be extracted as a Dep, not 'config'
    assert len(specs) == 1
    assert "data" in specs
    assert "config" not in specs


def test_get_dep_specs_no_deps() -> None:
    """Should return empty dict for function with no Dep arguments."""

    def simple(config: _TrainParams) -> _TrainOutputs:
        return {"model": {"weights": config.learning_rate}}

    specs = stage_def.get_dep_specs_from_signature(simple)

    assert specs == {}


def test_get_dep_specs_list_path() -> None:
    """Should extract list path from Dep annotation."""

    def process(
        shards: Annotated[
            list[pandas.DataFrame],
            outputs.Dep(["shard1.csv", "shard2.csv"], loaders.CSV[pandas.DataFrame]()),
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": sum(len(df) for df in shards)}}

    specs = stage_def.get_dep_specs_from_signature(process)

    assert len(specs) == 1
    assert specs["shards"].path == ["shard1.csv", "shard2.csv"]


# ==============================================================================
# Test: Direct function calling (the main goal!)
# ==============================================================================


def test_direct_call_with_params_and_data() -> None:
    """Stage function should be callable directly with params and data."""

    def train(
        config: _TrainParams,
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _TrainOutputs:
        total = data["value"].sum() * config.learning_rate
        return {"model": {"weights": float(total)}}

    # Create test data
    test_df = pandas.DataFrame({"value": [10, 20, 30]})
    params = _TrainParams(learning_rate=0.5)

    # Direct call - just pass the data!
    result = train(params, test_df)

    assert result["model"]["weights"] == 30.0  # (10+20+30) * 0.5


def test_direct_call_without_params() -> None:
    """Stage function without params should also be directly callable."""

    def process(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    # Create test data
    test_df = pandas.DataFrame({"a": [1, 2, 3, 4, 5]})

    # Direct call - no params needed
    result = process(test_df)

    assert result["result"]["count"] == 5


def test_direct_call_multiple_deps() -> None:
    """Stage function with multiple deps should accept multiple data arguments."""

    def combine(
        left: Annotated[dict[str, int], outputs.Dep("left.json", loaders.JSON[dict[str, int]]())],
        right: Annotated[dict[str, int], outputs.Dep("right.json", loaders.JSON[dict[str, int]]())],
    ) -> _MultiDepOutputs:
        return {"combined": {"left": left["x"], "right": right["y"]}}

    # Direct call with test data
    result = combine({"x": 10}, {"y": 20})

    assert result["combined"]["left"] == 10
    assert result["combined"]["right"] == 20


# ==============================================================================
# Test: Single-output shorthand (no TypedDict wrapper)
# ==============================================================================


def test_single_output_shorthand() -> None:
    """Stage function can return single output without TypedDict wrapper."""

    def transform(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> Annotated[pandas.DataFrame, outputs.Out("output.csv", loaders.CSV[pandas.DataFrame]())]:
        return data.dropna()

    # Should work as a regular function
    test_df = pandas.DataFrame({"a": [1, None, 3]})
    result = transform(test_df)

    assert len(result) == 2


def test_get_output_spec_from_single_return() -> None:
    """Should extract single Out from return annotation (non-TypedDict)."""

    def transform(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> Annotated[pandas.DataFrame, outputs.Out("output.csv", loaders.CSV[pandas.DataFrame]())]:
        return data.dropna()

    spec = stage_def.get_single_output_spec_from_return(transform)

    assert spec is not None
    assert spec.path == "output.csv"
    assert isinstance(spec.loader, loaders.CSV)


def test_get_output_spec_from_single_return_none_for_typeddict() -> None:
    """Should return None for TypedDict return types (use get_output_specs_from_return instead)."""

    def process(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    spec = stage_def.get_single_output_spec_from_return(process)

    assert spec is None


# ==============================================================================
# Test: StageParams base class
# ==============================================================================


def test_stage_params_is_pydantic_model() -> None:
    """StageParams should be a Pydantic model for validation and serialization."""
    import pydantic

    assert issubclass(stage_def.StageParams, pydantic.BaseModel)


def test_stage_params_defaults() -> None:
    """StageParams subclass should work with defaults."""

    class MyParams(stage_def.StageParams):
        lr: float = 0.01
        batch_size: int = 32

    params = MyParams()
    assert params.lr == 0.01
    assert params.batch_size == 32


def test_stage_params_override() -> None:
    """StageParams subclass should accept overrides."""

    class MyParams(stage_def.StageParams):
        lr: float = 0.01

    params = MyParams(lr=0.1)
    assert params.lr == 0.1


# ==============================================================================
# Test: Out annotation (replacing Out)
# ==============================================================================


def test_out_annotation_extraction_from_typeddict() -> None:
    """Should extract Out from TypedDict return annotation (same as Out)."""

    def train(config: _TrainParams) -> _OutTestOutputs:
        return {"model": {"w": 1.0}, "metrics": {"loss": 0.1}}

    specs = stage_def.get_output_specs_from_return(train, "test_stage")

    assert len(specs) == 2
    assert "model" in specs
    assert "metrics" in specs
    assert specs["model"].path == "model.json"
    assert specs["metrics"].path == "metrics.json"


def test_save_outputs_with_out_annotation(tmp_path: pathlib.Path) -> None:
    """save_return_outputs should work with Out annotations (same as Out)."""

    def process() -> _SingleOutTestOutputs:
        return {"result": {"count": 42}}

    specs = stage_def.get_output_specs_from_return(process, "test_stage")
    return_value: _SingleOutTestOutputs = {"result": {"count": 42}}

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    output_file = tmp_path / "output.json"
    assert output_file.exists()
    assert json.loads(output_file.read_text()) == {"count": 42}


# ==============================================================================
# Test: Framework integration (worker loads deps and injects them)
# ==============================================================================


def test_load_deps_from_specs(tmp_path: pathlib.Path) -> None:
    """load_deps_from_specs should load files based on dep specs."""
    # Create test file
    input_file = tmp_path / "input.json"
    input_file.write_text('{"value": 100}')

    def process(
        data: Annotated[dict[str, int], outputs.Dep("input.json", loaders.JSON[dict[str, int]]())],
    ) -> _ProcessOutputs:
        return {"result": data}

    specs = stage_def.get_dep_specs_from_signature(process)
    loaded = stage_def.load_deps_from_specs(specs, tmp_path)

    assert "data" in loaded
    assert loaded["data"] == {"value": 100}


def test_load_deps_multiple_files(tmp_path: pathlib.Path) -> None:
    """load_deps_from_specs should load multiple dependency files."""
    # Create test files
    (tmp_path / "a.json").write_text('{"x": 1}')
    (tmp_path / "b.json").write_text('{"y": 2}')

    def process(
        left: Annotated[dict[str, int], outputs.Dep("a.json", loaders.JSON[dict[str, int]]())],
        right: Annotated[dict[str, int], outputs.Dep("b.json", loaders.JSON[dict[str, int]]())],
    ) -> _MultiDepOutputs:
        return {"combined": {**left, **right}}

    specs = stage_def.get_dep_specs_from_signature(process)
    loaded = stage_def.load_deps_from_specs(specs, tmp_path)

    assert loaded["left"] == {"x": 1}
    assert loaded["right"] == {"y": 2}


def test_load_deps_list_path(tmp_path: pathlib.Path) -> None:
    """load_deps_from_specs should load list paths as lists."""
    # Create test files
    (tmp_path / "s1.json").write_text('{"a": 1}')
    (tmp_path / "s2.json").write_text('{"b": 2}')

    def process(
        shards: Annotated[
            list[dict[str, int]],
            outputs.Dep(["s1.json", "s2.json"], loaders.JSON[dict[str, int]]()),
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(shards)}}

    specs = stage_def.get_dep_specs_from_signature(process)
    loaded = stage_def.load_deps_from_specs(specs, tmp_path)

    assert loaded["shards"] == [{"a": 1}, {"b": 2}]


# ==============================================================================
# Test: Identify params type in signature
# ==============================================================================


def test_find_params_type_in_signature() -> None:
    """Should find StageParams subclass in function signature."""

    def train(
        config: _TrainParams,
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _TrainOutputs:
        return {"model": {"weights": config.learning_rate}}

    params_type = stage_def.find_params_type_in_signature(train)

    assert params_type is _TrainParams


def test_find_params_type_no_params() -> None:
    """Should return None when no StageParams in signature."""

    def process(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    params_type = stage_def.find_params_type_in_signature(process)

    assert params_type is None


def test_find_params_arg_name() -> None:
    """Should find the argument name for StageParams."""

    def train(
        cfg: _TrainParams,  # Not 'config' or 'params', custom name
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _TrainOutputs:
        return {"model": {"weights": cfg.learning_rate}}

    arg_name, params_type = stage_def.find_params_in_signature(train)

    assert arg_name == "cfg"
    assert params_type is _TrainParams


# ==============================================================================
# Test: Worker integration (full end-to-end)
# ==============================================================================


def test_worker_injects_deps(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker should load deps from disk and inject them as function kwargs."""
    from multiprocessing import Queue

    from pivot import project
    from pivot.executor import worker

    # Setup: create input file
    input_file = tmp_path / "input.json"
    input_file.write_text('{"value": 42}')

    # Mock project root
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)

    # Stage function with dependency injection
    def process(
        data: Annotated[dict[str, int], outputs.Dep("input.json", loaders.JSON[dict[str, int]]())],
    ) -> _ProcessOutputs:
        return {"result": {"doubled": data["value"] * 2}}

    # Run the stage function through worker
    output_queue: Queue[OutputMessage] = Queue()
    output_lines: list[tuple[str, bool]] = []

    # Get dep specs and out specs for the worker
    dep_specs = stage_def.get_dep_specs_from_signature(process)
    out_specs = stage_def.get_output_specs_from_return(process, "test_stage")

    worker._run_stage_function_with_injection(
        process,
        "test_stage",
        output_queue,
        output_lines,
        params=None,
        dep_specs=dep_specs,
        project_root=tmp_path,
        out_specs=out_specs,
    )

    # Verify output was saved
    output_file = tmp_path / "output.json"
    assert output_file.exists(), "Output should be saved by worker"
    assert json.loads(output_file.read_text()) == {"doubled": 84}


def test_worker_injects_params_and_deps(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker should inject both params and deps as function kwargs."""
    from multiprocessing import Queue

    from pivot import project
    from pivot.executor import worker

    # Setup: create input file
    input_file = tmp_path / "input.json"
    input_file.write_text('{"value": 100}')

    # Mock project root
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)

    # Stage function with params and deps
    def train(
        config: _TrainParams,
        data: Annotated[dict[str, int], outputs.Dep("input.json", loaders.JSON[dict[str, int]]())],
    ) -> _TrainOutputs:
        return {"model": {"weights": data["value"] * config.learning_rate}}

    # Create params
    params = _TrainParams(learning_rate=0.5)

    # Run the stage function through worker
    output_queue: Queue[OutputMessage] = Queue()
    output_lines: list[tuple[str, bool]] = []

    dep_specs = stage_def.get_dep_specs_from_signature(train)
    out_specs = stage_def.get_output_specs_from_return(train, "test_stage")

    params_arg_name, _ = stage_def.find_params_in_signature(train)

    worker._run_stage_function_with_injection(
        train,
        "test_stage",
        output_queue,
        output_lines,
        params=params,
        dep_specs=dep_specs,
        project_root=tmp_path,
        out_specs=out_specs,
        params_arg_name=params_arg_name,
    )

    # Verify output was saved
    output_file = tmp_path / "model.json"
    assert output_file.exists()
    assert json.loads(output_file.read_text()) == {"weights": 50.0}


# ==============================================================================
# Test: PlaceholderDep (dependencies that must be overridden)
# ==============================================================================


def test_placeholder_dep_has_no_path() -> None:
    """PlaceholderDep should have loader but no path attribute."""
    placeholder = outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())

    assert hasattr(placeholder, "loader")
    assert isinstance(placeholder.loader, loaders.CSV)
    assert not hasattr(placeholder, "path")


def test_get_placeholder_dep_names_identifies_placeholders() -> None:
    """Should identify which parameters use PlaceholderDep."""

    def compare(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        config: Annotated[
            dict[str, int], outputs.Dep("config.json", loaders.JSON[dict[str, int]]())
        ],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": len(baseline) + len(experiment)}}

    placeholder_names = stage_def.get_placeholder_dep_names(compare)

    assert placeholder_names == {"baseline", "experiment"}


def test_get_placeholder_dep_names_returns_empty_for_no_placeholders() -> None:
    """Should return empty set when no PlaceholderDep annotations."""

    def process(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("input.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    placeholder_names = stage_def.get_placeholder_dep_names(process)

    assert placeholder_names == set()


def test_get_dep_specs_with_placeholder_and_overrides() -> None:
    """Should resolve PlaceholderDep using provided overrides."""

    def compare(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    overrides = {
        "baseline": "model_a/results.csv",
        "experiment": "model_b/results.csv",
    }
    specs = stage_def.get_dep_specs_from_signature(compare, overrides)

    assert specs["baseline"].path == "model_a/results.csv"
    assert specs["experiment"].path == "model_b/results.csv"
    assert isinstance(specs["baseline"].loader, loaders.CSV)


def test_get_dep_specs_mixed_placeholder_and_regular() -> None:
    """Should handle mix of PlaceholderDep and regular Dep."""

    def compare(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        config: Annotated[
            dict[str, int], outputs.Dep("config.json", loaders.JSON[dict[str, int]]())
        ],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    overrides = {"baseline": "model_a/results.csv"}
    specs = stage_def.get_dep_specs_from_signature(compare, overrides)

    assert specs["baseline"].path == "model_a/results.csv"
    assert specs["config"].path == "config.json"


def test_get_dep_specs_placeholder_without_override_raises() -> None:
    """Should raise when PlaceholderDep has no override."""

    def compare(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    with pytest.raises(ValueError, match="PlaceholderDep .* requires override"):
        stage_def.get_dep_specs_from_signature(compare, {})


def test_get_dep_specs_placeholder_none_overrides_raises() -> None:
    """Should raise when PlaceholderDep exists but overrides is None."""

    def compare(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    with pytest.raises(ValueError, match="PlaceholderDep .* requires override"):
        stage_def.get_dep_specs_from_signature(compare, None)


def test_placeholder_dep_list_path_override() -> None:
    """PlaceholderDep should work with list path overrides."""

    def process_shards(
        shards: Annotated[
            list[pandas.DataFrame], outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(shards)}}

    overrides = {"shards": ["shard1.csv", "shard2.csv", "shard3.csv"]}
    specs = stage_def.get_dep_specs_from_signature(process_shards, overrides)

    assert specs["shards"].path == ["shard1.csv", "shard2.csv", "shard3.csv"]


def test_placeholder_dep_tuple_path_override() -> None:
    """PlaceholderDep should work with tuple path overrides."""

    def compare_pair(
        pair: Annotated[
            tuple[pandas.DataFrame, pandas.DataFrame],
            outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]()),
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": 2}}

    overrides = {"pair": ("left.csv", "right.csv")}
    specs = stage_def.get_dep_specs_from_signature(compare_pair, overrides)

    assert specs["pair"].path == ("left.csv", "right.csv")


def test_get_dep_specs_regular_dep_with_override() -> None:
    """Should apply override to regular Dep when provided."""

    def process(
        data: Annotated[
            pandas.DataFrame, outputs.Dep("default.csv", loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": len(data)}}

    overrides = {"data": "custom/override.csv"}
    specs = stage_def.get_dep_specs_from_signature(process, overrides)

    assert specs["data"].path == "custom/override.csv"


def test_placeholder_dep_empty_string_override_raises() -> None:
    """PlaceholderDep should reject empty string override."""

    def process(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": 0}}

    with pytest.raises(ValueError, match="override cannot be empty"):
        stage_def.get_dep_specs_from_signature(process, {"baseline": ""})


def test_placeholder_dep_empty_list_override_raises() -> None:
    """PlaceholderDep should reject empty list override."""

    def process(
        shards: Annotated[
            list[pandas.DataFrame], outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": 0}}

    with pytest.raises(ValueError, match="override contains empty path"):
        stage_def.get_dep_specs_from_signature(process, {"shards": []})


def test_placeholder_dep_list_with_empty_element_raises() -> None:
    """PlaceholderDep should reject list with empty string element."""

    def process(
        shards: Annotated[
            list[pandas.DataFrame], outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _ProcessOutputs:
        return {"result": {"count": 0}}

    with pytest.raises(ValueError, match="override contains empty path"):
        stage_def.get_dep_specs_from_signature(process, {"shards": ["a.csv", "", "c.csv"]})


# ==============================================================================
# Error Scenario Tests
# ==============================================================================


def test_load_deps_missing_file_raises(tmp_path: pathlib.Path) -> None:
    """load_deps_from_specs raises RuntimeError for missing dependency files."""

    def process(
        data: Annotated[
            dict[str, int], outputs.Dep("missing.json", loaders.JSON[dict[str, int]]())
        ],
    ) -> _ProcessOutputs:
        return {"result": data}

    specs = stage_def.get_dep_specs_from_signature(process)

    with pytest.raises(RuntimeError, match="Failed to load dependency"):
        stage_def.load_deps_from_specs(specs, tmp_path)


def test_load_deps_invalid_json_raises(tmp_path: pathlib.Path) -> None:
    """load_deps_from_specs raises error for malformed JSON files."""
    # Create invalid JSON file
    (tmp_path / "bad.json").write_text("{ invalid json")

    def process(
        data: Annotated[dict[str, int], outputs.Dep("bad.json", loaders.JSON[dict[str, int]]())],
    ) -> _ProcessOutputs:
        return {"result": data}

    specs = stage_def.get_dep_specs_from_signature(process)

    with pytest.raises(RuntimeError, match="Failed to load dependency"):
        stage_def.load_deps_from_specs(specs, tmp_path)


def test_load_deps_wrong_type_in_file(tmp_path: pathlib.Path) -> None:
    """load_deps_from_specs handles type mismatches gracefully."""
    # Create JSON file with list instead of dict
    (tmp_path / "data.json").write_text(json.dumps([1, 2, 3]))

    def process(
        data: Annotated[dict[str, int], outputs.Dep("data.json", loaders.JSON[dict[str, int]]())],
    ) -> _ProcessOutputs:
        return {"result": data}

    specs = stage_def.get_dep_specs_from_signature(process)

    # Should load successfully - type checking is not enforced at load time
    loaded = stage_def.load_deps_from_specs(specs, tmp_path)
    # The type mismatch will be caught by the stage function if it expects dict operations
    assert loaded["data"] == [1, 2, 3]


def test_get_dep_specs_unresolved_placeholder_with_partial_overrides() -> None:
    """get_dep_specs raises when PlaceholderDep missing from overrides dict."""

    def compare(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _MultiDepOutputs:
        return {"combined": {"count": 0}}

    # Only provide override for one placeholder
    overrides = {"baseline": "model_a/results.csv"}

    with pytest.raises(ValueError, match="PlaceholderDep.*experiment.*requires override"):
        stage_def.get_dep_specs_from_signature(compare, overrides)
