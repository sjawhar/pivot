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
# Test: Path overrides (for matrices/YAML)
# ==============================================================================


def test_dep_spec_path_override() -> None:
    """DepSpec should support path overrides."""

    def process(
        data: Annotated[
            dict[str, int], outputs.Dep("default.json", loaders.JSON[dict[str, int]]())
        ],
    ) -> _ProcessOutputs:
        return {"result": data}

    specs = stage_def.get_dep_specs_from_signature(process)
    overridden = stage_def.apply_dep_path_overrides(specs, {"data": "custom/data.json"})

    assert overridden["data"].path == "custom/data.json"
    # Original spec unchanged
    assert specs["data"].path == "default.json"


def test_dep_spec_path_override_partial() -> None:
    """Path overrides can be partial (only override some deps)."""

    def process(
        left: Annotated[dict[str, int], outputs.Dep("left.json", loaders.JSON[dict[str, int]]())],
        right: Annotated[dict[str, int], outputs.Dep("right.json", loaders.JSON[dict[str, int]]())],
    ) -> _MultiDepOutputs:
        return {"combined": {**left, **right}}

    specs = stage_def.get_dep_specs_from_signature(process)
    overridden = stage_def.apply_dep_path_overrides(specs, {"left": "custom_left.json"})

    assert overridden["left"].path == "custom_left.json"
    assert overridden["right"].path == "right.json"  # Unchanged


def test_dep_spec_path_override_unknown_key_raises() -> None:
    """Path override with unknown key should raise ValueError."""

    def process(
        data: Annotated[dict[str, int], outputs.Dep("input.json", loaders.JSON[dict[str, int]]())],
    ) -> _ProcessOutputs:
        return {"result": data}

    specs = stage_def.get_dep_specs_from_signature(process)

    with pytest.raises(ValueError, match="Unknown dependency"):
        stage_def.apply_dep_path_overrides(specs, {"unknown": "path.json"})


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
