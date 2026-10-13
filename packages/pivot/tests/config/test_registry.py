# pyright: reportUnusedFunction=false, reportUnusedParameter=false, reportRedeclaration=false
# No `from __future__ import annotations`: several tests define StageParams classes
# locally, and compose resolves stage annotations with typing.get_type_hints.

import functools
import inspect
import math
import pathlib
from collections.abc import Callable
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel
from tests import helpers

from pivot import compose, exceptions, fingerprint, loaders, registry, stage_def, types
from pivot.registry import PipelineLike, RegistryStageInfo, StageRegistry


# StageParams for testing
class PlainParams(stage_def.StageParams):
    learning_rate: float = 0.01


# =============================================================================
# Module-level helpers
# =============================================================================


# Module-level helper for testing module.attr capture (no leading underscore!)
def helper_uses_math() -> float:
    """Helper that uses math.pi for testing."""
    return math.pi * 2


# Module-level function that uses the helper (for testing transitive capture)
def stage_uses_helper() -> float:
    """Stage function that uses helper."""
    return helper_uses_math()


def _process_with_dep(data: pathlib.Path) -> str:
    return data.read_text()


def _stage1_with_dep(data: pathlib.Path) -> None:
    _ = data


def _stage2_with_out() -> dict[str, int]:
    return {"model": 1}


def _stage3_no_deps() -> None:
    pass


def _plain_stage(data: pathlib.Path, params: PlainParams) -> dict[str, float]:
    _ = data
    return {"lr": params.learning_rate}


def _stage_a_for_dag() -> dict[str, int]:
    return {"a": 1}


def _stage_b_for_dag(a: dict[str, int]) -> dict[str, int]:
    return {"b": a["a"]}


def _helper_info(
    name: str,
    *,
    func: Callable[..., Any] = _stage3_no_deps,
    deps: dict[str, types.ArtifactRef] | None = None,
    mutex: list[str] | None = None,
    variant: str | None = None,
    fingerprint_: dict[str, str] | None = None,
    state_dir: pathlib.Path | None = None,
) -> RegistryStageInfo:
    return RegistryStageInfo(
        func=func,
        name=name,
        deps=deps or {},
        outs=[],
        params=None,
        mutex=mutex or [],
        variant=variant,
        signature=None,
        fingerprint=fingerprint_,
        params_arg_name=None,
        state_dir=state_dir,
        collection_params={},
        no_fingerprint=False,
    )


def _helper_input_ref(name: str) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(name, None),
        format=loaders.PathOnly(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.DATA,
    )


# =============================================================================
# Registration through the compositional API
# =============================================================================


def test_register_registers_function(
    test_pipeline: PipelineLike, set_project_root: pathlib.Path
) -> None:
    """A stage call inside a Pipeline registers the stage with its dep and output."""
    helpers.register_test_stage(_process_with_dep, name="process", deps={"data": "data.csv"})

    assert "test/process" in test_pipeline.list_stages()
    info = test_pipeline.get_stage("test/process")
    assert info["name"] == "test/process"
    assert list(info["deps"]) == ["data"], "The input parameter becomes the stage's only dep"
    assert info["deps"]["data"].identity == types.ArtifactIdentity("data.csv", None)
    assert test_pipeline.input_bindings["data.csv"] == "data.csv"
    assert [out.identity for out in info["outs"]] == [types.ArtifactIdentity("test/process", None)]


def test_register_defers_fingerprint_until_requested(test_pipeline: PipelineLike) -> None:
    """Should compute and cache fingerprints lazily."""

    def my_stage() -> dict[str, int]:
        return {"answer": 42}

    helpers.register_test_stage(my_stage)

    info = test_pipeline.get_stage("test/my_stage")
    assert "fingerprint" in info
    assert info["fingerprint"] is None

    fp = test_pipeline.ensure_fingerprint("test/my_stage")
    assert isinstance(fp, dict)
    assert "self:my_stage" in fp
    assert info["fingerprint"] is fp


def test_register_captures_signature(test_pipeline: PipelineLike) -> None:
    """Should capture function signature."""

    def my_stage(x: str, y: str = "default") -> None:
        pass

    helpers.register_test_stage(my_stage, deps={"x": "x.txt"})

    info = test_pipeline.get_stage("test/my_stage")
    assert "signature" in info
    sig = info["signature"]
    assert isinstance(sig, inspect.Signature)
    assert "x" in sig.parameters
    assert "y" in sig.parameters
    assert sig.parameters["y"].default == "default"


def test_register_with_pydantic_params(tmp_path: pathlib.Path) -> None:
    """A StageParams instance passed to the stage call is stored on the stage."""

    class TrainParams(stage_def.StageParams):
        learning_rate: float = 0.01
        epochs: int = 100

    @compose.stage
    def train(params: TrainParams) -> None:
        pass

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        train(params=TrainParams())

    info = pipeline.get_stage("test/train")
    assert isinstance(info["params"], TrainParams), "Should store params as an instance"
    assert info["params"].learning_rate == 0.01
    assert info["params"].epochs == 100
    assert info["params_arg_name"] == "params"


def test_register_with_pydantic_params_instance(tmp_path: pathlib.Path) -> None:
    """Should support Pydantic parameter instance with custom values."""

    class TrainParams(stage_def.StageParams):
        learning_rate: float = 0.01
        epochs: int = 100

    @compose.stage
    def train_custom(params: TrainParams) -> None:
        pass

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        train_custom(params=TrainParams(learning_rate=0.05, epochs=50))

    info = pipeline.get_stage("test/train_custom")
    assert isinstance(info["params"], TrainParams)
    assert info["params"].learning_rate == 0.05
    assert info["params"].epochs == 50


def test_register_params_requires_params_argument(tmp_path: pathlib.Path) -> None:
    """Passing params to a stage without a params parameter is a definition-time error."""

    class MyParams(stage_def.StageParams):
        value: int = 10

    @compose.stage
    def process() -> None:
        pass

    with (
        pytest.raises(ValueError, match="unexpected keyword argument 'params'"),
        compose.Pipeline("test", root=tmp_path),
    ):
        process(params=MyParams())  # pyright: ignore[reportCallIssue] - the error under test


def test_register_params_must_be_stageparams(tmp_path: pathlib.Path) -> None:
    """A plain pydantic model is rejected; configuration must be a StageParams."""

    class NotAStageParams(BaseModel):
        pass

    @compose.stage
    def process(params: NotAStageParams) -> None:
        pass

    with (
        pytest.raises(ValueError, match="unsupported type 'NotAStageParams'"),
        compose.Pipeline("test", root=tmp_path),
    ):
        process(params=NotAStageParams())


def test_register_uses_default_params_value(tmp_path: pathlib.Path) -> None:
    """A StageParams default on the parameter is used when the call omits params."""

    class InferredParams(stage_def.StageParams):
        value: int = 42
        name: str = "default"

    _default = InferredParams()

    @compose.stage
    def process_inferred(params: InferredParams = _default) -> None:
        pass

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        process_inferred()

    info = pipeline.get_stage("test/process_inferred")
    assert isinstance(info["params"], InferredParams)
    assert info["params"].value == 42
    assert info["params"].name == "default"


@pytest.mark.xfail(
    strict=True,
    reason="red test (owner TBD): compose accepts a StageParams instance whose type does not "
    + "match the stage's params annotation (main raised ParamsError at registration)",
)
def test_register_params_type_mismatch_raises_error(tmp_path: pathlib.Path) -> None:
    """A params instance of an unrelated StageParams type is rejected at definition time."""

    class ParamsA(stage_def.StageParams):
        value_a: int = 1

    class ParamsB(stage_def.StageParams):
        value_b: str = "x"

    @compose.stage
    def process_mismatch(params: ParamsB) -> None:
        pass

    with (
        pytest.raises((exceptions.ParamsError, ValueError, TypeError)),
        compose.Pipeline("test", root=tmp_path),
    ):
        process_mismatch(params=ParamsA())  # pyright: ignore[reportArgumentType] - the error under test


def test_register_params_subclass_allowed(tmp_path: pathlib.Path) -> None:
    """Should allow params that is a subclass of the type hint."""

    class BaseParams(stage_def.StageParams):
        value: int = 1

    class DerivedParams(BaseParams):
        extra: str = "derived"

    @compose.stage
    def process_subclass(params: BaseParams) -> None:
        pass

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        process_subclass(params=DerivedParams())

    info = pipeline.get_stage("test/process_subclass")
    assert isinstance(info["params"], DerivedParams)
    assert info["params"].extra == "derived"


def test_register_with_additional_decorator(tmp_path: pathlib.Path) -> None:
    """Should work with functions that have additional decorators."""

    class DecoratedParams(stage_def.StageParams):
        multiplier: int = 2

    T = TypeVar("T", bound=Callable[..., Any])

    def my_decorator(f: T) -> T:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        return wrapper  # pyright: ignore[reportReturnType]

    @compose.stage
    @my_decorator
    def decorated_stage(params: DecoratedParams) -> None:
        _ = params.multiplier * 10

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        decorated_stage(params=DecoratedParams())

    info = pipeline.get_stage("test/decorated_stage")
    assert isinstance(info["params"], DecoratedParams)
    assert info["params"].multiplier == 2
    assert info["params_arg_name"] == "params"


def test_register_params_forward_ref_module_level(tmp_path: pathlib.Path) -> None:
    """Should handle forward reference to module-level class."""

    @compose.stage
    def process_forward_ref(params: "ForwardRefParams") -> None:
        pass

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        process_forward_ref(params=ForwardRefParams())

    info = pipeline.get_stage("test/process_forward_ref")
    assert isinstance(info["params"], ForwardRefParams)
    assert info["params"].ref_value == 99
    assert info["params_arg_name"] == "params"


# Module-level class for forward reference test
class ForwardRefParams(stage_def.StageParams):
    ref_value: int = 99


def test_register_defaults_to_function_name(test_pipeline: PipelineLike) -> None:
    """Should use function name as stage name by default."""

    def my_custom_stage() -> None:
        pass

    helpers.register_test_stage(my_custom_stage)

    assert "test/my_custom_stage" in test_pipeline.list_stages()


def test_register_with_no_deps_or_outs(test_pipeline: PipelineLike) -> None:
    """Should handle stages with no dependencies or outputs."""

    def simple() -> None:
        pass

    helpers.register_test_stage(simple)

    info = test_pipeline.get_stage("test/simple")
    assert info["deps"] == {}
    assert info["outs"] == []


def test_register_captures_transitive_dependencies(test_pipeline: PipelineLike) -> None:
    """Should capture helper function fingerprints."""

    def helper(x: int) -> int:
        return x * 2

    def my_stage(x: str) -> dict[str, int]:
        return {"value": helper(len(x)) + 1}

    helpers.register_test_stage(my_stage, deps={"x": "x.txt"})

    fp = test_pipeline.ensure_fingerprint("test/my_stage")
    assert "self:my_stage" in fp
    assert "func:helper" in fp


def test_multiple_stages_registered(
    test_pipeline: PipelineLike, set_project_root: pathlib.Path
) -> None:
    """Should register multiple stages independently."""
    helpers.register_test_stage(_stage1_with_dep, name="stage1", deps={"data": "data.csv"})
    helpers.register_test_stage(_stage2_with_out, name="stage2")
    helpers.register_test_stage(_stage3_no_deps, name="stage3")

    stages = test_pipeline.list_stages()
    assert "test/stage1" in stages
    assert "test/stage2" in stages
    assert "test/stage3" in stages


def test_stage_captures_user_code_helpers() -> None:
    """Should capture user-code helper functions but not stdlib module attrs."""
    fp = fingerprint.get_stage_fingerprint(stage_uses_helper)

    # Should capture the helper (user code)
    assert "func:helper_uses_math" in fp
    # math.pi is stdlib - should NOT be in fingerprint
    assert "mod:math.pi" not in fp


def test_register_captures_constants(test_pipeline: PipelineLike) -> None:
    """Should capture constant values in fingerprint."""
    LEARNING_RATE = 0.01

    def uses_constant() -> dict[str, float]:
        return {"value": LEARNING_RATE * 100}

    helpers.register_test_stage(uses_constant)

    fp = test_pipeline.ensure_fingerprint("test/uses_constant")
    assert "const:LEARNING_RATE" in fp


def test_registry_build_dag_integration(tmp_path: pathlib.Path) -> None:
    """A handle passed between stage calls becomes a DAG edge."""
    stage_a = compose.stage(_stage_a_for_dag)
    stage_b = compose.stage(_stage_b_for_dag)

    with compose.Pipeline("test", root=tmp_path) as pipeline:
        stage_b(stage_a())

    graph = pipeline.build_dag()

    assert "test/_stage_a_for_dag" in graph.nodes()
    assert "test/_stage_b_for_dag" in graph.nodes()
    assert graph.has_edge("test/_stage_b_for_dag", "test/_stage_a_for_dag")


def test_stageparams_work(test_pipeline: PipelineLike, set_project_root: pathlib.Path) -> None:
    """StageParams should work for stage parameters."""
    helpers.register_test_stage(
        _plain_stage, name="plain_stage", params=PlainParams(), deps={"data": "data.csv"}
    )

    info = test_pipeline.get_stage("test/plain_stage")
    assert isinstance(info["params"], PlainParams)
    assert info["params"].learning_rate == 0.01


def test_registry_stage_info_has_state_dir(tmp_path: pathlib.Path) -> None:
    """Stages record their pipeline's state directory."""
    with compose.Pipeline("test", root=tmp_path) as pipeline:
        compose.stage(_stage3_no_deps)()

    info = pipeline.get_stage("test/_stage3_no_deps")
    assert info["state_dir"] == tmp_path.resolve() / ".pivot"


# =============================================================================
# StageRegistry
# =============================================================================


def test_registry_get_stage() -> None:
    """Should retrieve stage info by name."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("test", func=lambda: 42, fingerprint_={}))
    stage_info = reg.get("test")
    assert stage_info["name"] == "test"


def test_registry_get_nonexistent_stage_raises_keyerror() -> None:
    """Should raise KeyError if stage not found."""
    reg = StageRegistry()

    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_list_stages() -> None:
    """Should list all registered stage names."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("stage1"))
    reg.add_existing(_helper_info("stage2"))
    stages = reg.list_stages()
    assert set(stages) == {"stage1", "stage2"}


def test_registry_list_stages_empty() -> None:
    """Should return empty list when no stages registered."""
    reg = StageRegistry()

    stages = reg.list_stages()
    assert stages == []


def test_registry_clear() -> None:
    """Should clear all registered stages."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("test"))
    reg.clear()
    assert reg.list_stages() == []


def test_registry_snapshot_returns_copy() -> None:
    """snapshot() should return a shallow copy of stages dict."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("test"))

    snapshot = reg.snapshot()

    assert snapshot == {"test": reg.get("test")}
    snapshot.clear()
    assert reg.list_stages() == ["test"], "snapshot should be a copy, not the same object"


def test_registry_snapshot_empty() -> None:
    """snapshot() should work on empty registry."""
    reg = StageRegistry()

    snapshot = reg.snapshot()

    assert snapshot == {}


def test_registry_restore_replaces_stages() -> None:
    """restore() should replace all stages with snapshot contents."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("current"))

    dep = _helper_input_ref("dep")
    backup = _helper_info("backup", func=lambda: 42, deps={"_0": dep})

    reg.restore({"backup": backup})

    assert reg.list_stages() == ["backup"]
    assert reg.get("backup")["deps"] == {"_0": dep}


def test_registry_restore_empty_snapshot() -> None:
    """restore() with empty snapshot should clear registry."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("test"))

    reg.restore({})

    assert reg.list_stages() == []


def test_registry_restore_preserves_metadata() -> None:
    """restore() should preserve all stage metadata."""
    reg = StageRegistry()
    deps = {"_0": _helper_input_ref("a"), "_1": _helper_input_ref("b")}
    original = _helper_info(
        "original",
        func=lambda: "test",
        deps=deps,
        mutex=["gpu"],
        variant="v1",
        fingerprint_={"self:original": "abc123"},
    )

    reg.restore({"original": original})

    assert reg.list_stages() == ["original"]
    restored = reg.get("original")
    assert restored["deps"] == deps
    assert restored["mutex"] == ["gpu"]
    assert restored["variant"] == "v1"
    assert restored["fingerprint"] == {"self:original": "abc123"}


def test_registry_add_existing_stage(tmp_path: pathlib.Path) -> None:
    """add_existing() should add a pre-built RegistryStageInfo."""
    reg = StageRegistry()

    reg.add_existing(_helper_info("added_stage", state_dir=tmp_path / ".pivot"))

    assert "added_stage" in reg.list_stages()
    assert reg.get("added_stage")["state_dir"] == tmp_path / ".pivot"


def test_registry_add_existing_collision_raises() -> None:
    """add_existing() should raise ValidationError on name collision."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("existing"))

    with pytest.raises(exceptions.ValidationError, match="already registered"):
        reg.add_existing(_helper_info("existing"))


def test_registry_add_existing_invalidates_dag_cache(tmp_path: pathlib.Path) -> None:
    """add_existing() should invalidate the cached DAG."""
    reg = StageRegistry()
    reg.add_existing(_helper_info("stage_a"))
    dag1 = reg.build_dag()
    assert len(dag1.nodes) == 1

    reg.add_existing(_helper_info("added_stage", state_dir=tmp_path / ".pivot"))

    # Subsequent build_dag should see new stage (cache was invalidated)
    dag2 = reg.build_dag()
    assert len(dag2.nodes) == 2
    assert "added_stage" in dag2.nodes


def test_get_stage_state_dir_returns_custom_when_set(tmp_path: pathlib.Path) -> None:
    """get_stage_state_dir returns the stage's state_dir when it is set.

    Stages from included pipelines have a per-stage state_dir pointing to
    the original pipeline's .pivot directory. This overrides the default.
    """
    custom_dir = tmp_path / "sub" / ".pivot"
    stage_info = _helper_info("test", state_dir=custom_dir)

    result = registry.get_stage_state_dir(stage_info, tmp_path / ".pivot")

    assert result == custom_dir


def test_get_stage_state_dir_returns_default_when_none(tmp_path: pathlib.Path) -> None:
    """get_stage_state_dir returns the default when stage's state_dir is None."""
    stage_info = _helper_info("test", state_dir=None)
    default_dir = tmp_path / ".pivot"

    result = registry.get_stage_state_dir(stage_info, default_dir)

    assert result == default_dir
