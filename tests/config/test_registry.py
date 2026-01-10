# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportRedeclaration=false, reportIncompatibleVariableOverride=false, reportAssignmentType=false

import inspect
import math
from pathlib import Path

import pandas
import pytest
from pydantic import BaseModel

from pivot import fingerprint, loaders, registry, stage, stage_def
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

    @stage(params=TrainParams)
    def train(params: TrainParams):
        pass

    info = REGISTRY.get("train")
    assert isinstance(info["params"], TrainParams), "Should store params as an instance"
    assert info["params"].learning_rate == 0.01
    assert info["params"].epochs == 100


def test_stage_with_pydantic_params_instance():
    """Should support Pydantic parameter instance with custom values."""

    class TrainParams(BaseModel):
        learning_rate: float = 0.01
        epochs: int = 100

    @stage(params=TrainParams(learning_rate=0.05, epochs=50))
    def train_custom(params: TrainParams):
        pass

    info = REGISTRY.get("train_custom")
    assert info["params"] is not None
    assert isinstance(info["params"], TrainParams)
    assert info["params"].learning_rate == 0.05
    assert info["params"].epochs == 50


def test_stage_params_cls_requires_params_argument():
    """Should raise ParamsError when params_cls provided but function has no params arg."""

    class MyParams(BaseModel):
        value: int = 10

    with pytest.raises(ParamsError, match="function must have a 'params' parameter"):

        @stage(params=MyParams)
        def process():
            pass


def test_stage_params_cls_must_be_basemodel():
    """Should raise ParamsError when params_cls is not a BaseModel subclass."""

    class NotAModel:
        pass

    with pytest.raises(ParamsError, match="must be a Pydantic BaseModel subclass"):

        @stage(params=NotAModel)  # pyright: ignore[reportArgumentType]
        def process(params: NotAModel):
            pass


def test_stage_infers_params_from_type_hint():
    """Should infer params class from function type hint when params not specified."""

    class InferredParams(BaseModel):
        value: int = 42
        name: str = "default"

    @stage()
    def process_inferred(params: InferredParams):
        pass

    info = REGISTRY.get("process_inferred")
    assert info["params"] is not None
    assert isinstance(info["params"], InferredParams)
    assert info["params"].value == 42
    assert info["params"].name == "default"


def test_stage_infer_params_requires_type_hint():
    """Should raise ParamsError when function has params arg but no type hint."""

    with pytest.raises(ParamsError, match="has 'params' parameter but no type hint"):

        @stage()
        def process_no_hint(params):  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
            pass


def test_stage_infer_params_requires_basemodel_hint():
    """Should raise ParamsError when params type hint is a generic type."""

    with pytest.raises(ParamsError, match="not a generic or union type"):

        @stage()
        def process_bad_hint(params: dict[str, str]):
            pass


def test_stage_infer_params_requires_basemodel_plain_class():
    """Should raise ParamsError when params type hint is a plain class (not BaseModel)."""

    class PlainClass:
        pass

    with pytest.raises(ParamsError, match="params type hint must be a Pydantic BaseModel"):

        @stage()
        def process_plain_class(params: PlainClass):
            pass


def test_stage_params_type_mismatch_raises_error():
    """Should raise ParamsError when params instance type doesn't match function type hint."""

    class ParamsA(BaseModel):
        value_a: int = 1

    class ParamsB(BaseModel):
        value_b: str = "x"

    with pytest.raises(ParamsError, match="does not match function type hint"):

        @stage(params=ParamsA())
        def process_mismatch(params: ParamsB):
            pass


def test_stage_params_class_mismatch_raises_error():
    """Should raise ParamsError when params class doesn't match function type hint."""

    class ParamsA(BaseModel):
        value_a: int = 1

    class ParamsB(BaseModel):
        value_b: str = "x"

    with pytest.raises(ParamsError, match="does not match function type hint"):

        @stage(params=ParamsA)
        def process_class_mismatch(params: ParamsB):
            pass


def test_stage_params_subclass_allowed():
    """Should allow params that is a subclass of the type hint."""

    class BaseParams(BaseModel):
        value: int = 1

    class DerivedParams(BaseParams):
        extra: str = "derived"

    # Subclass should be accepted when base class is the type hint
    @stage(params=DerivedParams())
    def process_subclass(params: BaseParams):
        pass

    info = REGISTRY.get("process_subclass")
    assert info["params"] is not None
    assert isinstance(info["params"], DerivedParams)
    assert info["params"].extra == "derived"


def test_stage_params_union_type_rejected():
    """Should reject Union type hints for params - must be concrete BaseModel."""
    from typing import Union  # pyright: ignore[reportDeprecated]

    class MyParams(BaseModel):
        value: int = 1

    with pytest.raises(ParamsError, match="not a generic or union type"):

        @stage()
        def process_union(params: Union[MyParams, None]):  # noqa: UP007  # pyright: ignore[reportDeprecated]
            pass


def test_stage_params_optional_type_rejected():
    """Should reject Optional type hints for params - must be concrete BaseModel."""
    from typing import Optional  # pyright: ignore[reportDeprecated]

    class MyParams(BaseModel):
        value: int = 1

    with pytest.raises(ParamsError, match="not a generic or union type"):

        @stage()
        def process_optional(params: Optional[MyParams]):  # noqa: UP045  # pyright: ignore[reportDeprecated]
            pass


def test_stage_params_pipe_none_type_rejected():
    """Should reject X | None type hints for params - must be concrete BaseModel."""

    class MyParams(BaseModel):
        value: int = 1

    with pytest.raises(ParamsError, match="not a generic or union type"):

        @stage()
        def process_pipe_none(params: MyParams | None):
            pass


def test_stage_params_generic_type_rejected():
    """Should reject generic type hints like list[MyParams]."""

    class MyParams(BaseModel):
        value: int = 1

    with pytest.raises(ParamsError, match="not a generic or union type"):

        @stage()
        def process_generic(params: list[MyParams]):
            pass


def test_stage_params_required_fields_raises_error():
    """Should raise ParamsError when inferred params have required fields without defaults."""

    class RequiredParams(BaseModel):
        required_value: int  # No default - must be provided

    with pytest.raises(ParamsError, match="validation error"):

        @stage()
        def process_required(params: RequiredParams):
            pass


def test_stage_with_additional_decorator():
    """Should work with functions that have additional decorators."""
    import functools
    from collections.abc import Callable
    from typing import Any, TypeVar

    class DecoratedParams(BaseModel):
        multiplier: int = 2

    T = TypeVar("T", bound=Callable[..., Any])

    def my_decorator(f: T) -> T:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        return wrapper  # pyright: ignore[reportReturnType]

    @stage()
    @my_decorator
    def decorated_stage(params: DecoratedParams) -> int:
        return params.multiplier * 10

    info = REGISTRY.get("decorated_stage")
    assert info["params"] is not None
    assert isinstance(info["params"], DecoratedParams)
    assert info["params"].multiplier == 2


def test_stage_params_forward_ref_module_level():
    """Should handle forward reference to module-level class."""
    # ForwardRefParams is defined at module level below
    # This test verifies forward refs work when the class is importable

    @stage()
    def process_forward_ref(params: "ForwardRefParams"):
        pass

    info = REGISTRY.get("process_forward_ref")
    assert info["params"] is not None
    assert isinstance(info["params"], ForwardRefParams)
    assert info["params"].ref_value == 99


# Module-level class for forward reference test
class ForwardRefParams(BaseModel):
    ref_value: int = 99


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
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
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
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
    )
    registry._stages["stage2"] = RegistryStageInfo(
        name="stage2",
        func=lambda: None,
        deps=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
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
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
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
    REGISTRY.register(func_one, name="my_stage", deps=["old.txt"], outs=list[str](), params=None)

    # Registering different function with same name should raise error
    with pytest.raises(ValidationError, match="already registered"):
        REGISTRY.register(
            func_two, name="my_stage", deps=["new.txt"], outs=list[str](), params=None
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


def test_registry_build_dag_integration(set_project_root: Path) -> None:
    """Test registry build_dag integration."""
    reg = registry.StageRegistry()

    # Create test files
    (set_project_root / "a.csv").touch()

    def stage_a() -> None:
        pass

    def stage_b() -> None:
        pass

    # Register stages
    reg.register(
        stage_a, name="stage_a", deps=[], outs=[str(set_project_root / "a.csv")], params=None
    )
    reg.register(
        stage_b,
        name="stage_b",
        deps=[str(set_project_root / "a.csv")],
        outs=[str(set_project_root / "b.csv")],
        params=None,
    )

    # Build DAG
    graph = reg.build_dag()

    # Check that DAG was built correctly
    assert "stage_a" in graph.nodes()
    assert "stage_b" in graph.nodes()
    assert graph.has_edge("stage_b", "stage_a")


# === Matrix Stage Tests ===


def test_stage_matrix_registers_multiple_stages() -> None:
    """Should register multiple stages for each variant in matrix."""
    from pivot import Variant

    @stage.matrix(
        [
            Variant(name="current", deps=["data/current.csv"], outs=["out/current.json"]),
            Variant(name="legacy", deps=["data/legacy.csv"], outs=["out/legacy.json"]),
        ]
    )
    def process_matrix(variant: str) -> None:
        pass

    assert "process_matrix@current" in REGISTRY.list_stages()
    assert "process_matrix@legacy" in REGISTRY.list_stages()

    current_info = REGISTRY.get("process_matrix@current")
    assert current_info["variant"] == "current"
    assert current_info["deps"][0].endswith("data/current.csv")

    legacy_info = REGISTRY.get("process_matrix@legacy")
    assert legacy_info["variant"] == "legacy"
    assert legacy_info["deps"][0].endswith("data/legacy.csv")


def test_stage_matrix_with_params() -> None:
    """Should support params in matrix variants."""
    from pivot import Variant

    class MyParams(BaseModel):
        mode: str = "default"
        threshold: float = 0.5

    @stage.matrix(
        [
            Variant(
                name="fast",
                deps=["data.csv"],
                outs=["out/fast.json"],
                params=MyParams(mode="fast", threshold=0.8),
            ),
            Variant(
                name="accurate",
                deps=["data.csv"],
                outs=["out/accurate.json"],
                params=MyParams(mode="accurate", threshold=0.2),
            ),
        ]
    )
    def train_matrix(variant: str, params: MyParams) -> None:
        pass

    fast_info = REGISTRY.get("train_matrix@fast")
    fast_params = fast_info["params"]
    assert fast_params is not None
    assert isinstance(fast_params, MyParams)
    assert fast_params.mode == "fast"
    assert fast_params.threshold == 0.8

    accurate_info = REGISTRY.get("train_matrix@accurate")
    accurate_params = accurate_info["params"]
    assert accurate_params is not None
    assert isinstance(accurate_params, MyParams)
    assert accurate_params.mode == "accurate"
    assert accurate_params.threshold == 0.2


def test_stage_matrix_invalid_variant_name_with_at() -> None:
    """Should raise error for variant names with '@'."""
    from pydantic import ValidationError as PydanticValidationError

    from pivot import Variant

    with pytest.raises(PydanticValidationError, match="alphanumeric"):
        Variant(name="invalid@name", deps=["x"], outs=["y"])


def test_stage_matrix_invalid_variant_name_with_slash() -> None:
    """Should raise error for variant names with slashes."""
    from pydantic import ValidationError as PydanticValidationError

    from pivot import Variant

    with pytest.raises(PydanticValidationError, match="alphanumeric"):
        Variant(name="invalid/name", deps=["x"], outs=["y"])


def test_stage_matrix_invalid_variant_name_empty() -> None:
    """Should raise error for empty variant names."""
    from pydantic import ValidationError as PydanticValidationError

    from pivot import Variant

    with pytest.raises(PydanticValidationError, match="cannot be empty"):
        Variant(name="", deps=["x"], outs=["y"])


def test_stage_matrix_invalid_variant_name_too_long() -> None:
    """Should raise error for variant names exceeding max length."""
    from pydantic import ValidationError as PydanticValidationError

    from pivot import Variant

    long_name = "a" * 65  # Max is 64
    with pytest.raises(PydanticValidationError, match="exceeds max length"):
        Variant(name=long_name, deps=["x"], outs=["y"])


def test_stage_matrix_duplicate_variant_names() -> None:
    """Should raise error for duplicate variant names in matrix."""
    from pivot import Variant

    with pytest.raises(ValidationError, match="Duplicate variant name"):

        @stage.matrix(
            [
                Variant(name="same", deps=["x"], outs=["y"]),
                Variant(name="same", deps=["a"], outs=["b"]),
            ]
        )
        def bad_matrix(variant: str) -> None:
            pass


def test_stage_matrix_empty_variants() -> None:
    """Should raise error for empty variant list."""
    with pytest.raises(ValidationError, match="cannot be empty"):

        @stage.matrix([])
        def bad_matrix(variant: str) -> None:
            pass


def test_stage_matrix_different_deps_per_variant() -> None:
    """Should support different deps/outs per variant."""
    from pivot import Variant

    @stage.matrix(
        [
            Variant(name="simple", deps=["a.csv"], outs=["simple.out"]),
            Variant(name="complex", deps=["a.csv", "b.csv", "c.csv"], outs=["complex.out"]),
        ]
    )
    def varied_deps(variant: str) -> None:
        pass

    simple_info = REGISTRY.get("varied_deps@simple")
    complex_info = REGISTRY.get("varied_deps@complex")

    assert len(simple_info["deps"]) == 1
    assert len(complex_info["deps"]) == 3


def test_stage_name_with_at_sign_rejected() -> None:
    """Should reject stage name containing @ (reserved for matrix variants)."""
    with pytest.raises(ValueError, match="cannot contain '@'"):

        @stage(name="invalid@name", deps=["x"], outs=["y"])
        def process() -> None:
            pass


def test_stage_func_name_with_at_sign_rejected() -> None:
    """Should reject custom stage name containing @ (reserved for matrix variants)."""
    with pytest.raises(ValueError, match="cannot contain '@'"):

        @stage(name="bad@stage", deps=["x"], outs=["y"])
        def normal_func() -> None:
            pass


# --- Snapshot/Restore Tests ---


def test_registry_snapshot_returns_copy() -> None:
    """snapshot() should return a shallow copy of stages dict."""
    reg = StageRegistry()
    reg._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: None,
        deps=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
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
        deps=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
    )

    backup = RegistryStageInfo(
        name="backup",
        func=lambda: 42,
        deps=["/tmp/dep"],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
    )
    snapshot = {"backup": backup}

    reg.restore(snapshot)

    assert reg.list_stages() == ["backup"]
    assert reg.get("backup")["deps"] == ["/tmp/dep"]


def test_registry_restore_empty_snapshot() -> None:
    """restore() with empty snapshot should clear registry."""
    reg = StageRegistry()
    reg._stages["test"] = RegistryStageInfo(
        name="test",
        func=lambda: None,
        deps=[],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=[],
        variant=None,
        signature=None,
        fingerprint={},
        cwd=None,
    )

    reg.restore({})

    assert reg.list_stages() == []


def test_registry_restore_preserves_metadata() -> None:
    """restore() should preserve all stage metadata."""
    reg = StageRegistry()

    original = RegistryStageInfo(
        name="original",
        func=lambda: "test",
        deps=["/tmp/a", "/tmp/b"],
        outs=[],
        outs_paths=[],
        params=None,
        mutex=["gpu"],
        variant="v1",
        signature=None,
        fingerprint={"self:original": "abc123"},
        cwd=None,
    )
    snapshot = {"original": original}

    reg.restore(snapshot)

    assert reg.list_stages() == ["original"]
    restored = reg.get("original")
    assert restored["deps"] == ["/tmp/a", "/tmp/b"]
    assert restored["mutex"] == ["gpu"]
    assert restored["variant"] == "v1"
    assert restored["fingerprint"] == {"self:original": "abc123"}


# ==============================================================================
# StageDef integration tests
# ==============================================================================


class _TestStageDef(stage_def.StageDef):
    """Module-level StageDef for testing (required for pickling)."""

    class deps:
        data: loaders.CSV[pandas.DataFrame] = "data/input.csv"

    class outs:
        result: loaders.JSON[dict[str, int]] = "output/result.json"

    threshold: float = 0.5


def test_stage_def_detected_from_type_hint() -> None:
    """Registry should detect StageDef from function params type hint."""

    @stage()
    def process_data(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("process_data")
    assert isinstance(info["params"], _TestStageDef)


def test_stage_def_deps_extracted_automatically() -> None:
    """Registry should extract deps from StageDef when not explicitly provided."""

    @stage()
    def auto_deps(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("auto_deps")
    assert len(info["deps"]) == 1
    assert info["deps"][0].endswith("data/input.csv")


def test_stage_def_outs_extracted_automatically() -> None:
    """Registry should extract outs from StageDef when not explicitly provided."""

    @stage()
    def auto_outs(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("auto_outs")
    assert len(info["outs_paths"]) == 1
    assert info["outs_paths"][0].endswith("output/result.json")


def test_stage_def_explicit_deps_override() -> None:
    """Explicit deps= argument should completely replace StageDef deps."""

    @stage(deps=["custom/data.csv", "extra/file.txt"])
    def override_deps(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("override_deps")
    assert len(info["deps"]) == 2
    assert any("custom/data.csv" in d for d in info["deps"])
    assert any("extra/file.txt" in d for d in info["deps"])
    assert not any("data/input.csv" in d for d in info["deps"])


def test_stage_def_explicit_outs_override() -> None:
    """Explicit outs= argument should completely replace StageDef outs."""

    @stage(outs=["custom/output.json"])
    def override_outs(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("override_outs")
    assert len(info["outs_paths"]) == 1
    assert info["outs_paths"][0].endswith("custom/output.json")
    assert not info["outs_paths"][0].endswith("output/result.json")


def test_stage_def_fingerprint_includes_loaders() -> None:
    """Fingerprint should include loader fingerprints for StageDef."""

    @stage()
    def with_loaders(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("with_loaders")
    fp = info["fingerprint"]

    # Should include loader fingerprints
    loader_keys = [k for k in fp.keys() if k.startswith("loader:")]
    assert len(loader_keys) > 0, "Should have loader fingerprints"


def test_plain_pydantic_params_still_work() -> None:
    """Regular Pydantic BaseModel params should still work (backward compat)."""

    class PlainParams(BaseModel):
        learning_rate: float = 0.01

    @stage(deps=["data.csv"], outs=["out.csv"])
    def plain_stage(params: PlainParams) -> None:
        pass

    info = REGISTRY.get("plain_stage")
    assert isinstance(info["params"], PlainParams)
    assert info["params"].learning_rate == 0.01


class _MultiDeps(stage_def.StageDef):
    """StageDef with multiple deps/outs for testing."""

    class deps:
        train: loaders.CSV[pandas.DataFrame] = "data/train.csv"
        test: loaders.CSV[pandas.DataFrame] = "data/test.csv"

    class outs:
        model: loaders.Pickle[dict[str, float]] = "models/model.pkl"
        metrics: loaders.JSON[dict[str, float]] = "metrics.json"


def test_stage_def_multiple_deps_outs() -> None:
    """StageDef with multiple deps/outs should register all paths."""

    @stage()
    def multi_io(params: _MultiDeps) -> None:
        pass

    info = REGISTRY.get("multi_io")
    assert len(info["deps"]) == 2
    assert len(info["outs_paths"]) == 2


def test_stage_def_params_values_preserved() -> None:
    """StageDef parameter values should be accessible on registered params."""

    @stage()
    def with_params(params: _TestStageDef) -> None:
        pass

    info = REGISTRY.get("with_params")
    assert info["params"] is not None
    assert info["params"].threshold == 0.5  # pyright: ignore[reportAttributeAccessIssue]
