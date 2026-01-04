"""Tests for stage registry and @stage decorator.

Tests the stage registration system that collects pipeline stages
and captures their metadata including code fingerprints.
"""
# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportRedeclaration=false

import inspect
import math

import pytest
from pydantic import BaseModel

from pivot import fingerprint, stage
from pivot.registry import REGISTRY, StageRegistry


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
    assert info["deps"] == ["data.csv"]
    assert info["outs"] == ["output.txt"]


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
    """Should support Pydantic parameter models."""

    class TrainParams(BaseModel):
        learning_rate: float = 0.01
        epochs: int = 100

    @stage(params_cls=TrainParams)
    def train(learning_rate: float, epochs: int):
        pass

    info = REGISTRY.get("train")
    assert info["params_cls"] == TrainParams


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


def test_stage_with_stage_dependencies():
    """Should handle stage:name dependencies."""

    @stage(deps=["stage:preprocess", "data.csv"], outs=["model.pkl"])
    def train():
        pass

    info = REGISTRY.get("train")
    assert "stage:preprocess" in info["deps"]
    assert "data.csv" in info["deps"]


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
    registry._stages["test"] = {"name": "test", "func": lambda: 42}  # pyright: ignore[reportPrivateUsage]

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
    registry._stages["stage1"] = {"name": "stage1"}  # pyright: ignore[reportPrivateUsage]
    registry._stages["stage2"] = {"name": "stage2"}  # pyright: ignore[reportPrivateUsage]

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
    registry._stages["test"] = {"name": "test"}  # pyright: ignore[reportPrivateUsage]

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
    assert info["deps"] == ["input.txt"]
    assert info["outs"] == ["output.txt"]


def test_registry_register_with_custom_name():
    """Should allow custom stage name."""
    registry = StageRegistry()

    def my_func():
        pass

    registry.register(my_func, name="custom_name")

    assert "custom_name" in registry.list_stages()
    info = registry.get("custom_name")
    assert info["name"] == "custom_name"


def test_stage_duplicate_registration_allowed():
    """Should allow re-registering a stage (overwrites previous)."""

    @stage(deps=["old.txt"])
    def my_stage():
        pass

    # Re-register with different deps
    @stage(deps=["new.txt"])
    def my_stage():
        pass

    info = REGISTRY.get("my_stage")
    assert info["deps"] == ["new.txt"]


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
