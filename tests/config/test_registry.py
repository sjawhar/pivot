# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportRedeclaration=false, reportIncompatibleVariableOverride=false, reportAssignmentType=false

import inspect
import math
import pathlib
from typing import Annotated, TypedDict

import pandas
import pytest
from pydantic import BaseModel

from helpers import register_test_stage
from pivot import exceptions, fingerprint, loaders, outputs, registry, stage_def
from pivot.exceptions import ParamsError, ValidationError
from pivot.pipeline.pipeline import Pipeline
from pivot.registry import RegistryStageInfo, StageRegistry

# =============================================================================
# Output TypedDicts for annotation-based stages
# =============================================================================


class _OutputTxt(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


class _OutCsv(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("out.csv", loaders.PathOnly())]


class _ModelPkl(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("model.pkl", loaders.PathOnly())]


class _ACsv(TypedDict):
    a: Annotated[pathlib.Path, outputs.Out("a.csv", loaders.PathOnly())]


class _BCsv(TypedDict):
    b: Annotated[pathlib.Path, outputs.Out("b.csv", loaders.PathOnly())]


# StageParams for testing
class PlainParams(stage_def.StageParams):
    learning_rate: float = 0.01


# =============================================================================
# Module-level helpers for fingerprint testing
# =============================================================================


# Module-level helper for testing module.attr capture (no leading underscore!)
def helper_uses_math() -> float:
    """Helper that uses math.pi for testing."""
    return math.pi * 2


# Module-level function that uses the helper (for testing transitive capture)
def stage_uses_helper() -> float:
    """Stage function that uses helper."""
    return helper_uses_math()


# Module-level stage functions for tests that need annotation-based deps/outs


def _process_with_dep(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> _OutputTxt:
    pathlib.Path("output.txt").write_text("done")
    return {"output": pathlib.Path("output.txt")}


def _stage1_with_dep(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
) -> None:
    pass


def _stage2_with_out() -> _ModelPkl:
    pathlib.Path("model.pkl").write_bytes(b"")
    return {"model": pathlib.Path("model.pkl")}


def _stage3_no_deps() -> None:
    pass


def _plain_stage(
    data: Annotated[pathlib.Path, outputs.Dep("data.csv", loaders.PathOnly())],
    params: PlainParams,
) -> _OutCsv:
    _ = params
    pathlib.Path("out.csv").write_text("")
    return {"output": pathlib.Path("out.csv")}


# =============================================================================
# Tests
# =============================================================================


def test_register_registers_function(
    test_pipeline: "Pipeline", set_project_root: pathlib.Path
) -> None:
    """Should register function when calling register_test_stage()."""
    register_test_stage(_process_with_dep, name="process")

    assert "process" in test_pipeline.list_stages()
    info = test_pipeline.get("process")
    assert info["name"] == "process"
    # Paths are normalized to absolute paths
    assert len(info["deps_paths"]) == 1
    assert info["deps_paths"][0].endswith("data.csv")
    # outs contains BaseOut objects, outs_paths contains string paths
    assert len(info["outs"]) == 1
    assert info["outs_paths"][0].endswith("output.txt")
    assert str(info["outs"][0].path).endswith("output.txt")


def test_register_defers_fingerprint_until_requested(test_pipeline: "Pipeline") -> None:
    """Should compute and cache fingerprints lazily."""

    def my_stage():
        return 42

    register_test_stage(my_stage)

    info = test_pipeline.get("my_stage")
    assert "fingerprint" in info
    assert info["fingerprint"] is None

    fingerprint = test_pipeline._registry.ensure_fingerprint("my_stage")
    assert isinstance(fingerprint, dict)
    assert "self:my_stage" in fingerprint
    assert info["fingerprint"] is fingerprint


def test_register_captures_signature(test_pipeline: "Pipeline") -> None:
    """Should capture function signature."""

    def my_stage(x: int, y: str = "default"):
        pass

    register_test_stage(my_stage)

    info = test_pipeline.get("my_stage")
    assert "signature" in info
    sig = info["signature"]
    assert isinstance(sig, inspect.Signature)
    assert "x" in sig.parameters
    assert "y" in sig.parameters
    assert sig.parameters["y"].default == "default"


def test_register_with_pydantic_params(test_pipeline: "Pipeline") -> None:
    """Should support Pydantic parameter models with params argument."""

    class TrainParams(stage_def.StageParams):
        learning_rate: float = 0.01
        epochs: int = 100

    def train(params: TrainParams):
        pass

    register_test_stage(train, params=TrainParams)

    info = test_pipeline.get("train")
    assert isinstance(info["params"], TrainParams), "Should store params as an instance"
    assert info["params"].learning_rate == 0.01
    assert info["params"].epochs == 100


def test_register_with_pydantic_params_instance(test_pipeline: "Pipeline") -> None:
    """Should support Pydantic parameter instance with custom values."""

    class TrainParams(stage_def.StageParams):
        learning_rate: float = 0.01
        epochs: int = 100

    def train_custom(params: TrainParams):
        pass

    register_test_stage(train_custom, params=TrainParams(learning_rate=0.05, epochs=50))

    info = test_pipeline.get("train_custom")
    assert info["params"] is not None
    assert isinstance(info["params"], TrainParams)
    assert info["params"].learning_rate == 0.05
    assert info["params"].epochs == 50


def test_register_params_cls_requires_params_argument(test_pipeline: "Pipeline") -> None:
    """Should raise ParamsError when params_cls provided but function has no params arg."""

    class MyParams(stage_def.StageParams):
        value: int = 10

    def process():
        pass

    with pytest.raises(ParamsError, match="must have a StageParams parameter"):
        register_test_stage(process, params=MyParams)


def test_register_params_cls_must_be_stageparams(test_pipeline: "Pipeline") -> None:
    """Should raise ParamsError when params_cls is not a StageParams subclass."""

    class NotAStageParams(BaseModel):
        pass

    def process(params: NotAStageParams):
        pass

    with pytest.raises(ParamsError, match="must have a StageParams parameter"):
        register_test_stage(process, params=NotAStageParams)  # pyright: ignore[reportArgumentType]


def test_register_infers_params_from_type_hint(test_pipeline: "Pipeline") -> None:
    """Should infer params class from function type hint when params not specified."""

    class InferredParams(stage_def.StageParams):
        value: int = 42
        name: str = "default"

    def process_inferred(params: InferredParams):
        pass

    register_test_stage(process_inferred)

    info = test_pipeline.get("process_inferred")
    assert info["params"] is not None
    assert isinstance(info["params"], InferredParams)
    assert info["params"].value == 42
    assert info["params"].name == "default"


def test_register_params_type_mismatch_raises_error(test_pipeline: "Pipeline") -> None:
    """Should raise ParamsError when params instance type doesn't match function type hint."""

    class ParamsA(stage_def.StageParams):
        value_a: int = 1

    class ParamsB(stage_def.StageParams):
        value_b: str = "x"

    def process_mismatch(params: ParamsB):
        pass

    with pytest.raises(ParamsError, match="does not match function type hint"):
        register_test_stage(process_mismatch, params=ParamsA())


def test_register_params_class_mismatch_raises_error(test_pipeline: "Pipeline") -> None:
    """Should raise ParamsError when params class doesn't match function type hint."""

    class ParamsA(stage_def.StageParams):
        value_a: int = 1

    class ParamsB(stage_def.StageParams):
        value_b: str = "x"

    def process_class_mismatch(params: ParamsB):
        pass

    with pytest.raises(ParamsError, match="does not match function type hint"):
        register_test_stage(process_class_mismatch, params=ParamsA)


def test_register_params_subclass_allowed(test_pipeline: "Pipeline") -> None:
    """Should allow params that is a subclass of the type hint."""

    class BaseParams(stage_def.StageParams):
        value: int = 1

    class DerivedParams(BaseParams):
        extra: str = "derived"

    def process_subclass(params: BaseParams):
        pass

    # Subclass should be accepted when base class is the type hint
    register_test_stage(process_subclass, params=DerivedParams())

    info = test_pipeline.get("process_subclass")
    assert info["params"] is not None
    assert isinstance(info["params"], DerivedParams)
    assert info["params"].extra == "derived"


def test_register_params_required_fields_raises_error(test_pipeline: "Pipeline") -> None:
    """Should raise ParamsError when inferred params have required fields without defaults."""

    class RequiredParams(stage_def.StageParams):
        required_value: int  # No default - must be provided

    def process_required(params: RequiredParams):
        pass

    with pytest.raises(ParamsError, match="validation error"):
        register_test_stage(process_required)


def test_register_with_additional_decorator(test_pipeline: "Pipeline") -> None:
    """Should work with functions that have additional decorators."""
    import functools
    from collections.abc import Callable
    from typing import Any, TypeVar

    class DecoratedParams(stage_def.StageParams):
        multiplier: int = 2

    T = TypeVar("T", bound=Callable[..., Any])

    def my_decorator(f: T) -> T:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        return wrapper  # pyright: ignore[reportReturnType]

    @my_decorator
    def decorated_stage(params: DecoratedParams) -> int:
        return params.multiplier * 10

    register_test_stage(decorated_stage)

    info = test_pipeline.get("decorated_stage")
    assert info["params"] is not None
    assert isinstance(info["params"], DecoratedParams)
    assert info["params"].multiplier == 2


def test_register_params_forward_ref_module_level(test_pipeline: "Pipeline") -> None:
    """Should handle forward reference to module-level class."""
    # ForwardRefParams is defined at module level below
    # This test verifies forward refs work when the class is importable

    def process_forward_ref(params: "ForwardRefParams"):
        pass

    register_test_stage(process_forward_ref)

    info = test_pipeline.get("process_forward_ref")
    assert info["params"] is not None
    assert isinstance(info["params"], ForwardRefParams)
    assert info["params"].ref_value == 99


# Module-level class for forward reference test
class ForwardRefParams(stage_def.StageParams):
    ref_value: int = 99


def test_register_defaults_to_function_name(test_pipeline: "Pipeline") -> None:
    """Should use function name as stage name by default."""

    def my_custom_stage():
        pass

    register_test_stage(my_custom_stage)

    assert "my_custom_stage" in test_pipeline.list_stages()


def test_register_with_no_deps_or_outs(test_pipeline: "Pipeline") -> None:
    """Should handle stages with no dependencies or outputs."""

    def simple():
        return 42

    register_test_stage(simple)

    info = test_pipeline.get("simple")
    assert info["deps"] == {}
    assert info["deps_paths"] == []
    assert info["outs"] == []


def test_register_captures_transitive_dependencies(test_pipeline: "Pipeline") -> None:
    """Should capture helper function fingerprints."""

    def helper(x: int) -> int:
        return x * 2

    def my_stage(x: int) -> int:
        return helper(x) + 1

    register_test_stage(my_stage)

    fp = test_pipeline._registry.ensure_fingerprint("my_stage")
    assert "self:my_stage" in fp
    assert "func:helper" in fp


def test_registry_get_stage():
    """Should retrieve stage info by name."""
    reg = StageRegistry()
    reg._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: 42,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )
    stage_info = reg.get("test")
    assert stage_info["name"] == "test"


def test_registry_get_nonexistent_stage_raises_keyerror():
    """Should raise KeyError if stage not found."""
    reg = StageRegistry()

    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_list_stages():
    """Should list all registered stage names."""
    reg = StageRegistry()
    reg._stages["stage1"] = RegistryStageInfo(
        name="stage1",
        func=lambda: None,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )
    reg._stages["stage2"] = RegistryStageInfo(
        name="stage2",
        func=lambda: None,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )
    stages = reg.list_stages()
    assert set(stages) == {"stage1", "stage2"}


def test_registry_list_stages_empty():
    """Should return empty list when no stages registered."""
    reg = StageRegistry()

    stages = reg.list_stages()
    assert stages == []


def test_registry_clear():
    """Should clear all registered stages."""
    reg = StageRegistry()
    reg._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: None,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )
    reg.clear()
    assert reg.list_stages() == []


def test_registry_register_with_annotation_based_deps():
    """Should register stage with annotation-based deps/outs."""
    reg = StageRegistry()

    reg.register(_process_with_dep, name="my_func")

    assert "my_func" in reg.list_stages()
    info = reg.get("my_func")
    assert info["func"] == _process_with_dep
    # Paths are extracted from annotations
    assert len(info["deps_paths"]) == 1
    assert info["deps_paths"][0].endswith("data.csv")
    assert len(info["outs"]) == 1
    assert info["outs_paths"][0].endswith("output.txt")


def test_registry_register_with_custom_name():
    """Should allow custom stage name."""
    reg = StageRegistry()

    def my_func():
        pass

    reg.register(my_func, name="custom_name")

    assert "custom_name" in reg.list_stages()
    info = reg.get("custom_name")
    assert info["name"] == "custom_name"


def test_stage_duplicate_registration_raises_error(test_pipeline: "Pipeline") -> None:
    """Should raise error when registering two stages with same name."""

    def func_one() -> None:
        pass

    def func_two() -> None:
        pass

    # Register first function with name "my_stage"
    register_test_stage(func_one, name="my_stage")

    # Registering different function with same name should raise error
    with pytest.raises(ValidationError, match="already registered"):
        register_test_stage(func_two, name="my_stage")


def test_multiple_stages_registered(
    test_pipeline: "Pipeline", set_project_root: pathlib.Path
) -> None:
    """Should register multiple stages independently."""
    register_test_stage(_stage1_with_dep, name="stage1")
    register_test_stage(_stage2_with_out, name="stage2")
    register_test_stage(_stage3_no_deps, name="stage3")

    stages = test_pipeline.list_stages()
    assert "stage1" in stages
    assert "stage2" in stages
    assert "stage3" in stages


def test_stage_captures_user_code_helpers():
    """Should capture user-code helper functions but not stdlib module attrs."""
    fp = fingerprint.get_stage_fingerprint(stage_uses_helper)

    # Should capture the helper (user code)
    assert "func:helper_uses_math" in fp
    # math.pi is stdlib - should NOT be in fingerprint
    assert "mod:math.pi" not in fp


def test_register_captures_constants(test_pipeline: "Pipeline") -> None:
    """Should capture constant values in fingerprint."""
    LEARNING_RATE = 0.01

    def uses_constant():
        return LEARNING_RATE * 100

    register_test_stage(uses_constant)

    fp = test_pipeline._registry.ensure_fingerprint("uses_constant")
    assert "const:LEARNING_RATE" in fp


def _stage_a_for_dag() -> _ACsv:
    pathlib.Path("a.csv").write_text("")
    return {"a": pathlib.Path("a.csv")}


def _stage_b_for_dag(
    a: Annotated[pathlib.Path, outputs.Dep("a.csv", loaders.PathOnly())],
) -> _BCsv:
    _ = a
    pathlib.Path("b.csv").write_text("")
    return {"b": pathlib.Path("b.csv")}


def test_registry_build_dag_integration(set_project_root: pathlib.Path) -> None:
    """Test registry build_dag integration."""
    reg = registry.StageRegistry()

    # Create test files
    (set_project_root / "a.csv").touch()

    # Register stages
    reg.register(_stage_a_for_dag, name="stage_a")
    reg.register(_stage_b_for_dag, name="stage_b")

    # Build DAG
    graph = reg.build_dag()

    # Check that DAG was built correctly
    assert "stage_a" in graph.nodes()
    assert "stage_b" in graph.nodes()
    assert graph.has_edge("stage_b", "stage_a")


# --- Snapshot/Restore Tests ---


def test_registry_snapshot_returns_copy() -> None:
    """snapshot() should return a shallow copy of stages dict."""
    reg = StageRegistry()
    reg._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: None,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )

    snapshot = reg.snapshot()

    assert snapshot == reg._stages
    assert snapshot is not reg._stages, "snapshot should be a copy, not the same object"


def test_registry_snapshot_empty() -> None:
    """snapshot() should work on empty registry."""
    reg = StageRegistry()

    snapshot = reg.snapshot()

    assert snapshot == {}


def test_registry_restore_replaces_stages() -> None:
    """restore() should replace all stages with snapshot contents."""
    reg = StageRegistry()
    reg._stages["current"] = RegistryStageInfo(
        name="current",
        func=lambda: None,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )

    backup = RegistryStageInfo(
        name="backup",
        func=lambda: 42,
        deps={"_0": "/tmp/dep"},
        deps_paths=["/tmp/dep"],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )
    snapshot = {"backup": backup}

    reg.restore(snapshot)

    assert reg.list_stages() == ["backup"]
    assert reg.get("backup")["deps_paths"] == ["/tmp/dep"]


def test_registry_restore_empty_snapshot() -> None:
    """restore() with empty snapshot should clear registry."""
    reg = StageRegistry()
    reg._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: None,
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )

    reg.restore({})

    assert reg.list_stages() == []


def test_registry_restore_preserves_metadata() -> None:
    """restore() should preserve all stage metadata."""
    reg = StageRegistry()

    original = RegistryStageInfo(
        name="original",
        func=lambda: "test",
        deps={"_0": "/tmp/a", "_1": "/tmp/b"},
        deps_paths=["/tmp/a", "/tmp/b"],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=["gpu"],
        variant="v1",
        signature=None,
        fingerprint={"self:original": "abc123"},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )
    snapshot = {"original": original}

    reg.restore(snapshot)

    assert reg.list_stages() == ["original"]
    restored = reg.get("original")
    assert restored["deps_paths"] == ["/tmp/a", "/tmp/b"]
    assert restored["mutex"] == ["gpu"]
    assert restored["variant"] == "v1"
    assert restored["fingerprint"] == {"self:original": "abc123"}


# ==============================================================================
# StageParams test
# ==============================================================================


def test_stageparams_work(test_pipeline: "Pipeline", set_project_root: pathlib.Path) -> None:
    """StageParams should work for stage parameters."""
    register_test_stage(_plain_stage, name="plain_stage", params=PlainParams())

    info = test_pipeline.get("plain_stage")
    assert isinstance(info["params"], PlainParams)
    assert info["params"].learning_rate == 0.01


# ==============================================================================
# out_path_overrides accepts simple strings
# ==============================================================================


def test_out_path_overrides_accepts_simple_string(
    test_pipeline: "Pipeline", set_project_root: pathlib.Path
) -> None:
    """out_path_overrides should accept simple strings, not just dicts with path key."""

    def my_stage() -> _OutputTxt:
        return {"output": pathlib.Path("output.txt")}

    # Simple string should work (not {"output": {"path": "override.txt"}})
    register_test_stage(my_stage, out_path_overrides={"output": "override.txt"})

    info = test_pipeline.get("my_stage")
    assert info["out_specs"]["output"].path == "override.txt"


def test_out_path_overrides_accepts_dict_with_options(
    test_pipeline: "Pipeline", set_project_root: pathlib.Path
) -> None:
    """out_path_overrides should accept dicts with path and options."""

    def my_stage2() -> _OutputTxt:
        return {"output": pathlib.Path("output.txt")}

    # Full dict with options should also work
    register_test_stage(
        my_stage2, out_path_overrides={"output": {"path": "override.txt", "cache": False}}
    )

    info = test_pipeline.get("my_stage2")
    assert info["out_specs"]["output"].path == "override.txt"
    assert info["out_specs"]["output"].cache is False


def test_out_path_overrides_accepts_list_paths(
    test_pipeline: Pipeline, set_project_root: pathlib.Path
) -> None:
    """out_path_overrides should accept list paths for multi-file outputs."""

    class MultiOutput(TypedDict):
        items: Annotated[list[str], outputs.Out(["a.txt", "b.txt"], loaders.PathOnly())]

    def my_stage3() -> MultiOutput:
        return {"items": []}

    # Simple list should work
    register_test_stage(my_stage3, out_path_overrides={"items": ["x.txt", "y.txt"]})

    info = test_pipeline.get("my_stage3")
    assert info["out_specs"]["items"].path == ["x.txt", "y.txt"]


# ==============================================================================
# PlaceholderDep validation tests
# ==============================================================================


class _PlaceholderCompareOutputs(TypedDict):
    result: Annotated[dict[str, int], outputs.Out("result.json", loaders.JSON[dict[str, int]]())]


def test_register_placeholder_dep_without_override_raises(
    test_registry: StageRegistry,
) -> None:
    """Registration should fail when PlaceholderDep has no override."""

    def _compare_no_override(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _PlaceholderCompareOutputs:
        return {"result": {"diff": 0}}

    with pytest.raises(
        exceptions.ValidationError, match="Placeholder dependencies missing overrides"
    ):
        test_registry.register(_compare_no_override, name="compare_test_no_override")


def test_register_placeholder_dep_partial_override_raises(
    test_registry: StageRegistry,
) -> None:
    """Registration should fail when only some PlaceholderDeps have overrides."""

    def _compare_partial(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _PlaceholderCompareOutputs:
        return {"result": {"diff": 0}}

    with pytest.raises(exceptions.ValidationError, match="experiment"):
        test_registry.register(
            _compare_partial,
            name="compare_partial_override",
            dep_path_overrides={"baseline": "model_a/results.csv"},
        )


def test_register_placeholder_dep_with_all_overrides_succeeds(
    test_registry: StageRegistry,
) -> None:
    """Registration should succeed when all PlaceholderDeps have overrides."""

    def _compare_success(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _PlaceholderCompareOutputs:
        return {"result": {"diff": 0}}

    test_registry.register(
        _compare_success,
        name="compare_success_test",
        dep_path_overrides={
            "baseline": "model_a/results.csv",
            "experiment": "model_b/results.csv",
        },
    )

    info = test_registry.get("compare_success_test")
    assert info is not None
    assert "baseline" in info["deps"]
    assert "experiment" in info["deps"]


def test_register_placeholder_dep_error_lists_all_missing(
    test_registry: StageRegistry,
) -> None:
    """Error message should list all missing placeholder overrides."""

    def _compare_many(
        a: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        b: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
        c: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    ) -> _PlaceholderCompareOutputs:
        return {"result": {"count": 0}}

    with pytest.raises(exceptions.ValidationError) as exc_info:
        test_registry.register(_compare_many, name="compare_many_test")

    # All three should be mentioned in the error
    error_msg = str(exc_info.value)
    assert "a" in error_msg
    assert "b" in error_msg
    assert "c" in error_msg


def test_register_placeholder_dep_typo_suggests_correction(
    test_registry: StageRegistry,
) -> None:
    """Error message should suggest correction for typos in override keys."""

    def _compare_typo(
        baseline: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
        experiment: Annotated[
            pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
        ],
    ) -> _PlaceholderCompareOutputs:
        return {"result": {"count": 0}}

    with pytest.raises(exceptions.ValidationError) as exc_info:
        test_registry.register(
            _compare_typo,
            name="compare_typo_test",
            dep_path_overrides={
                "basline": "model_a/results.csv",  # typo: basline -> baseline
                "experiment": "model_b/results.csv",
            },
        )

    error_msg = str(exc_info.value)
    assert "baseline" in error_msg  # missing param
    assert "Did you mean" in error_msg  # fuzzy suggestion
    assert "'basline' -> 'baseline'" in error_msg  # specific suggestion


# ==============================================================================
# state_dir field tests (multi-pipeline support)
# ==============================================================================


def test_registry_stage_info_has_state_dir() -> None:
    """RegistryStageInfo should include state_dir field."""
    reg = StageRegistry()

    def my_stage() -> None:
        pass

    state_dir = pathlib.Path("/tmp/test_pipeline/.pivot")
    reg.register(my_stage, name="my_stage", state_dir=state_dir)

    info = reg.get("my_stage")
    assert info["state_dir"] == state_dir


def test_registry_stage_info_state_dir_defaults_to_none() -> None:
    """state_dir should default to None when not specified."""
    reg = StageRegistry()

    def my_stage() -> None:
        pass

    reg.register(my_stage, name="my_stage")

    info = reg.get("my_stage")
    assert info["state_dir"] is None


# ==============================================================================
# add_existing() method tests (pipeline composition)
# ==============================================================================


def test_registry_add_existing_stage(tmp_path: pathlib.Path) -> None:
    """add_existing() should add a pre-built RegistryStageInfo."""
    reg = StageRegistry()

    # Create a stage info manually (simulating copy from another registry)
    info = RegistryStageInfo(
        func=_stage3_no_deps,
        name="added_stage",
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=tmp_path / ".pivot",
    )

    reg.add_existing(info)

    assert "added_stage" in reg.list_stages()
    assert reg.get("added_stage")["state_dir"] == tmp_path / ".pivot"


def test_registry_add_existing_collision_raises() -> None:
    """add_existing() should raise ValidationError on name collision."""
    reg = StageRegistry()
    reg.register(_stage3_no_deps, name="existing")

    info = RegistryStageInfo(
        func=_stage3_no_deps,
        name="existing",  # Collision!
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
    )

    with pytest.raises(exceptions.ValidationError, match="already registered"):
        reg.add_existing(info)


def test_registry_add_existing_invalidates_dag_cache(set_project_root: pathlib.Path) -> None:
    """add_existing() should invalidate the cached DAG."""
    reg = StageRegistry()

    # Create test file for stage dependency
    (set_project_root / "a.csv").touch()

    # Register initial stage and build DAG (caches it)
    reg.register(_stage_a_for_dag, name="stage_a")
    dag1 = reg.build_dag(validate=False)
    assert len(dag1.nodes) == 1

    # Add existing stage info
    info = RegistryStageInfo(
        func=_stage3_no_deps,
        name="added_stage",
        deps={},
        deps_paths=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=set_project_root / ".pivot",
    )
    reg.add_existing(info)

    # Subsequent build_dag should see new stage (cache was invalidated)
    dag2 = reg.build_dag(validate=False)
    assert len(dag2.nodes) == 2
    assert "added_stage" in dag2.nodes
