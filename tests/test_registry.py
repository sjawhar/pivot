"""Tests for stage registry and @stage decorator.

Tests the stage registration system that collects pipeline stages
and captures their metadata including code fingerprints.
"""
# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportRedeclaration=false

import inspect
import math
from pathlib import Path

import pytest
from pydantic import BaseModel

from pivot import fingerprint, registry, stage
from pivot.exceptions import ParamsError, ValidationError
from pivot.registry import REGISTRY, RegistryStageInfo, StageRegistry


# Module-level helper for testing module.attr capture (no leading underscore!)
def helper_uses_math() -> float:
    """Helper that uses math.pi for testing."""
    return math.pi * 2


# Module-level function that uses the helper (for testing transitive capture)
def stage_uses_helper() -> float:
    """Stage function that uses helper."""
    return helper_uses_math()


def test_stage_decorator_registers_function():
    """Should register function when decorated with @stage."""

    @stage(deps=["data.csv"], outs=["output.txt"])
    def process():
        return 42

    assert "process" in REGISTRY.list_stages()
    info = REGISTRY.get("process")
    assert info["name"] == "process"
    # Paths are normalized to absolute paths
    assert len(info["deps"]) == 1
    assert info["deps"][0].endswith("data.csv")
    # outs contains BaseOut objects, outs_paths contains string paths
    assert len(info["outs"]) == 1
    assert info["outs_paths"][0].endswith("output.txt")
    assert info["outs"][0].path.endswith("output.txt")


def test_stage_decorator_returns_unmodified_function():
    """Should return original function unmodified."""

    @stage()
    def my_func():
        """Original docstring."""
        return 42

    assert my_func() == 42
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "Original docstring."


def test_stage_captures_fingerprint():
    """Should capture code fingerprint on registration."""

    @stage()
    def my_stage():
        return 42

    info = REGISTRY.get("my_stage")
    assert "fingerprint" in info
    assert isinstance(info["fingerprint"], dict)
    assert "self:my_stage" in info["fingerprint"]


def test_stage_captures_signature():
    """Should capture function signature."""

    @stage()
    def my_stage(x: int, y: str = "default"):
        pass

    info = REGISTRY.get("my_stage")
    assert "signature" in info
    sig = info["signature"]
    assert isinstance(sig, inspect.Signature)
    assert "x" in sig.parameters
    assert "y" in sig.parameters
    assert sig.parameters["y"].default == "default"


def test_stage_with_pydantic_params():
    """Should support Pydantic parameter models with params argument."""

    class TrainParams(BaseModel):
        learning_rate: float = 0.01
        epochs: int = 100

    @stage(params_cls=TrainParams)
    def train(params: TrainParams):
        pass

    info = REGISTRY.get("train")
    assert info["params_cls"] == TrainParams


def test_stage_params_cls_requires_params_argument():
    """Should raise ParamsError when params_cls provided but function has no params arg."""

    class MyParams(BaseModel):
        value: int = 10

    with pytest.raises(ParamsError, match="function must have a 'params' parameter"):

        @stage(params_cls=MyParams)
        def process():
            pass


def test_stage_params_cls_must_be_basemodel():
    """Should raise ParamsError when params_cls is not a BaseModel subclass."""

    class NotAModel:
        pass

    with pytest.raises(ParamsError, match="must be a Pydantic BaseModel subclass"):

        @stage(params_cls=NotAModel)  # pyright: ignore[reportArgumentType]
        def process(params: NotAModel):
            pass


def test_stage_warns_on_orphaned_params_argument(caplog: pytest.LogCaptureFixture):
    """Should warn when function has params arg but no params_cls."""
    import logging

    caplog.set_level(logging.WARNING)

    @stage()
    def process(params: dict[str, str]):
        pass

    assert "has 'params' parameter but no params_cls specified" in caplog.text


def test_stage_defaults_to_function_name():
    """Should use function name as stage name by default."""

    @stage()
    def my_custom_stage():
        pass

    assert "my_custom_stage" in REGISTRY.list_stages()


def test_stage_with_no_deps_or_outs():
    """Should handle stages with no dependencies or outputs."""

    @stage()
    def simple():
        return 42

    info = REGISTRY.get("simple")
    assert info["deps"] == []
    assert info["outs"] == []


def test_stage_captures_transitive_dependencies():
    """Should capture helper function fingerprints."""

    def helper(x: int) -> int:
        return x * 2

    @stage()
    def my_stage(x: int) -> int:
        return helper(x) + 1

    info = REGISTRY.get("my_stage")
    fp = info["fingerprint"]
    assert "self:my_stage" in fp
    assert "func:helper" in fp


def test_registry_get_stage():
    """Should retrieve stage info by name."""
    registry = StageRegistry()
    registry._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: 42,
        deps=[],
        outs=[],
        outs_paths=[],
        params_cls=None,
        mutex=[],
        signature=None,
        fingerprint={},
    )
    stage_info = registry.get("test")
    assert stage_info["name"] == "test"


def test_registry_get_nonexistent_stage_raises_keyerror():
    """Should raise KeyError if stage not found."""
    registry = StageRegistry()

    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_registry_list_stages():
    """Should list all registered stage names."""
    registry = StageRegistry()
    registry._stages["stage1"] = RegistryStageInfo(
        name="stage1",
        func=lambda: None,
        deps=[],
        outs=[],
        outs_paths=[],
        params_cls=None,
        mutex=[],
        signature=None,
        fingerprint={},
    )
    registry._stages["stage2"] = RegistryStageInfo(
        name="stage2",
        func=lambda: None,
        deps=[],
        outs=[],
        outs_paths=[],
        params_cls=None,
        mutex=[],
        signature=None,
        fingerprint={},
    )
    stages = registry.list_stages()
    assert set(stages) == {"stage1", "stage2"}


def test_registry_list_stages_empty():
    """Should return empty list when no stages registered."""
    registry = StageRegistry()

    stages = registry.list_stages()
    assert stages == []


def test_registry_clear():
    """Should clear all registered stages."""
    registry = StageRegistry()
    registry._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: None,
        deps=[],
        outs=[],
        outs_paths=[],
        params_cls=None,
        mutex=[],
        signature=None,
        fingerprint={},
    )
    registry.clear()
    assert registry.list_stages() == []


def test_registry_register_directly():
    """Should register stage via direct register() call."""
    registry = StageRegistry()

    def my_func(x: int):
        return x * 2

    registry.register(my_func, deps=["input.txt"], outs=["output.txt"])

    assert "my_func" in registry.list_stages()
    info = registry.get("my_func")
    assert info["func"] == my_func
    # Paths are normalized to absolute paths
    assert len(info["deps"]) == 1
    assert info["deps"][0].endswith("input.txt")
    assert len(info["outs"]) == 1
    assert info["outs_paths"][0].endswith("output.txt")


def test_registry_register_with_custom_name():
    """Should allow custom stage name."""
    registry = StageRegistry()

    def my_func():
        pass

    registry.register(my_func, name="custom_name")

    assert "custom_name" in registry.list_stages()
    info = registry.get("custom_name")
    assert info["name"] == "custom_name"


def test_stage_duplicate_registration_raises_error():
    """Should raise error when registering two stages with same name."""

    def func_one():
        pass

    def func_two():
        pass

    # Register first function with name "my_stage"
    REGISTRY.register(
        func_one, name="my_stage", deps=["old.txt"], outs=list[str](), params_cls=None
    )

    # Registering different function with same name should raise error
    with pytest.raises(ValidationError, match="already registered"):
        REGISTRY.register(
            func_two, name="my_stage", deps=["new.txt"], outs=list[str](), params_cls=None
        )


def test_multiple_stages_registered():
    """Should register multiple stages independently."""

    @stage(deps=["data.csv"])
    def stage1():
        pass

    @stage(outs=["model.pkl"])
    def stage2():
        pass

    @stage()
    def stage3():
        pass

    stages = REGISTRY.list_stages()
    assert "stage1" in stages
    assert "stage2" in stages
    assert "stage3" in stages


def test_stage_captures_module_attrs():
    """Should capture module.attr patterns in fingerprint."""
    # Use fingerprinting directly on module-level function
    fp = fingerprint.get_stage_fingerprint(stage_uses_helper)

    # Should capture the helper which uses math.pi
    assert "func:helper_uses_math" in fp
    assert "mod:math.pi" in fp


def test_stage_captures_constants():
    """Should capture constant values in fingerprint."""
    LEARNING_RATE = 0.01

    @stage()
    def uses_constant():
        return LEARNING_RATE * 100

    info = REGISTRY.get("uses_constant")
    fp = info["fingerprint"]
    assert "const:LEARNING_RATE" in fp


def test_registry_build_dag_integration(tmp_path: Path) -> None:
    """Test registry build_dag integration."""
    reg = registry.StageRegistry()

    # Create test files
    (tmp_path / "a.csv").touch()

    def stage_a() -> None:
        pass

    def stage_b() -> None:
        pass

    # Register stages
    reg.register(stage_a, name="stage_a", deps=[], outs=[str(tmp_path / "a.csv")], params_cls=None)
    reg.register(
        stage_b,
        name="stage_b",
        deps=[str(tmp_path / "a.csv")],
        outs=[str(tmp_path / "b.csv")],
        params_cls=None,
    )

    # Build DAG
    graph = reg.build_dag()

    # Check that DAG was built correctly
    assert "stage_a" in graph.nodes()
    assert "stage_b" in graph.nodes()
    assert graph.has_edge("stage_b", "stage_a")
