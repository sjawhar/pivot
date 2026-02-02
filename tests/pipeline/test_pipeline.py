from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest

from pivot import exceptions, loaders, outputs, project
from pivot.pipeline.pipeline import Pipeline
from pivot.pipeline.yaml import PipelineConfigError
from pivot.stage_def import StageParams

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# Module-level helper for stage registration tests
class _SimpleOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _simple_stage() -> _SimpleOutput:
    pathlib.Path("result.txt").write_text("done")
    return _SimpleOutput(result=pathlib.Path("result.txt"))


# Helper for params preservation test
class _TestIncludeParams(StageParams):
    value: int = 42


def _parameterized_stage_for_include(params: _TestIncludeParams) -> _SimpleOutput:
    pathlib.Path("result.txt").write_text(str(params.value))
    return _SimpleOutput(result=pathlib.Path("result.txt"))


def test_pipeline_creation_with_name() -> None:
    """Pipeline should be creatable with a name."""
    p = Pipeline("my_pipeline")
    assert p.name == "my_pipeline"


def test_pipeline_infers_root_from_caller() -> None:
    """Pipeline should infer root directory from caller's __file__."""
    p = Pipeline("test")

    # Should be the directory containing this test file
    expected = pathlib.Path(__file__).parent
    assert p.root == expected


def test_pipeline_accepts_explicit_root() -> None:
    """Pipeline should accept explicit root override."""
    custom_root = pathlib.Path("/custom/path")
    p = Pipeline("test", root=custom_root)
    assert p.root == custom_root


def test_pipeline_state_dir_derived_from_root() -> None:
    """Pipeline state_dir should be root/.pivot."""
    custom_root = pathlib.Path("/custom/path")
    p = Pipeline("test", root=custom_root)
    assert p.state_dir == custom_root / ".pivot"


# =============================================================================
# Pipeline Name Validation Tests
# =============================================================================


def test_pipeline_name_validation_valid_names() -> None:
    """Pipeline should accept valid names."""
    # Various valid patterns
    valid_names = [
        "pipeline",
        "MyPipeline",
        "pipeline_1",
        "my-pipeline",
        "A",
        "a123",
        "Pipeline_With_Underscores",
        "pipeline-with-hyphens",
    ]
    for name in valid_names:
        p = Pipeline(name)
        assert p.name == name


def test_pipeline_name_validation_empty_fails() -> None:
    """Pipeline should reject empty name."""
    from pivot.pipeline.yaml import PipelineConfigError

    with pytest.raises(PipelineConfigError, match="cannot be empty"):
        Pipeline("")


def test_pipeline_name_validation_invalid_patterns() -> None:
    """Pipeline should reject names with invalid patterns."""
    from pivot.pipeline.yaml import PipelineConfigError

    invalid_names = [
        "1pipeline",  # starts with number
        "_pipeline",  # starts with underscore
        "-pipeline",  # starts with hyphen
        "pipe.line",  # contains period
        "pipe/line",  # contains slash
        "pipe line",  # contains space
        "pipeline!",  # contains special char
        "../escape",  # path traversal attempt
    ]
    for name in invalid_names:
        with pytest.raises(PipelineConfigError, match="Invalid pipeline name"):
            Pipeline(name)


# Stage registration tests


def test_pipeline_register_stage(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Pipeline.register should register a stage with the pipeline's state_dir."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="my_stage")

    assert "my_stage" in p.list_stages()
    info = p.get("my_stage")
    assert info["state_dir"] == tmp_path / ".pivot"


def test_pipeline_stages_isolated(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Two pipelines can have stages with the same name."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p1 = Pipeline("pipeline1", root=tmp_path / "p1")
    p2 = Pipeline("pipeline2", root=tmp_path / "p2")

    p1.register(_simple_stage, name="train")
    p2.register(_simple_stage, name="train")

    assert "train" in p1.list_stages()
    assert "train" in p2.list_stages()

    # Each has its own state_dir
    assert p1.get("train")["state_dir"] == tmp_path / "p1" / ".pivot"
    assert p2.get("train")["state_dir"] == tmp_path / "p2" / ".pivot"


# =============================================================================
# Invalid Registration Tests
# =============================================================================


class _InvalidOutputMissingAnnotation(TypedDict):
    # Missing Annotated[] wrapper - should be caught
    result: pathlib.Path


def _stage_with_invalid_output_missing_annotation() -> _InvalidOutputMissingAnnotation:
    pathlib.Path("result.txt").write_text("done")
    return _InvalidOutputMissingAnnotation(result=pathlib.Path("result.txt"))


def test_pipeline_register_missing_annotation_fails(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Pipeline.register fails when output TypedDict field missing Annotated wrapper.

    Critical for catching user errors where they forget Annotated[] on TypedDict fields.
    This prevents silent registration of stages that won't work at runtime.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)

    with pytest.raises(exceptions.StageDefinitionError, match="without Out annotations"):
        p.register(_stage_with_invalid_output_missing_annotation, name="invalid")


# =============================================================================
# get() Tests
# =============================================================================


def test_pipeline_get_raises_keyerror_for_unknown_stage(tmp_path: pathlib.Path) -> None:
    """get() should raise KeyError with stage name for unknown stages."""
    p = Pipeline("test", root=tmp_path)

    with pytest.raises(KeyError, match="unknown_stage"):
        p.get("unknown_stage")


# =============================================================================
# build_dag() Tests
# =============================================================================


# Helper stages for DAG tests - defined at module level for fingerprinting
class _StageAOutput(TypedDict):
    data: Annotated[pathlib.Path, outputs.Out("a.txt", loaders.PathOnly())]


def _stage_a() -> _StageAOutput:
    pathlib.Path("a.txt").write_text("a")
    return _StageAOutput(data=pathlib.Path("a.txt"))


class _StageBOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("b.txt", loaders.PathOnly())]


def _stage_b(
    data: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
) -> _StageBOutput:
    pathlib.Path("b.txt").write_text(f"b depends on {data}")
    return _StageBOutput(result=pathlib.Path("b.txt"))


# Cycle test helpers - at module level for type hint resolution
class _CycleAOutput(TypedDict):
    data: Annotated[pathlib.Path, outputs.Out("cycle_a.txt", loaders.PathOnly())]


def _cycle_a(
    dep: Annotated[pathlib.Path, outputs.Dep("cycle_b.txt", loaders.PathOnly())],
) -> _CycleAOutput:
    return _CycleAOutput(data=pathlib.Path("cycle_a.txt"))


class _CycleBOutput(TypedDict):
    data: Annotated[pathlib.Path, outputs.Out("cycle_b.txt", loaders.PathOnly())]


def _cycle_b(
    dep: Annotated[pathlib.Path, outputs.Dep("cycle_a.txt", loaders.PathOnly())],
) -> _CycleBOutput:
    return _CycleBOutput(data=pathlib.Path("cycle_b.txt"))


def test_pipeline_build_dag_with_valid_stages(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """build_dag() should return a valid DAG for registered stages."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()  # Project root marker

    p = Pipeline("test", root=tmp_path)
    p.register(_stage_a, name="stage_a")
    p.register(_stage_b, name="stage_b")

    dag = p.build_dag(validate=True)

    # DAG should have both stages as nodes
    assert "stage_a" in dag.nodes
    assert "stage_b" in dag.nodes

    # stage_b depends on stage_a (edge from consumer to producer: b -> a)
    assert dag.has_edge("stage_b", "stage_a")


def test_pipeline_build_dag_validate_false_skips_dependency_check(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """build_dag(validate=False) should not raise for missing dependencies."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)
    # Only register stage_b which depends on a.txt (not provided by any stage)
    p.register(_stage_b, name="stage_b")

    # validate=False should not raise
    dag = p.build_dag(validate=False)
    assert "stage_b" in dag.nodes


def test_pipeline_build_dag_detects_cycles(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """build_dag() should raise CyclicGraphError for circular dependencies."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)
    p.register(_cycle_a, name="cycle_a")
    p.register(_cycle_b, name="cycle_b")

    with pytest.raises(exceptions.CyclicGraphError):
        p.build_dag(validate=True)


# =============================================================================
# snapshot() and restore() Tests
# =============================================================================


def test_pipeline_snapshot_captures_current_state(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """snapshot() should capture current registry state for rollback."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="stage1")

    snap = p.snapshot()

    assert "stage1" in snap
    assert snap["stage1"]["name"] == "stage1"
    assert snap["stage1"]["func"] == _simple_stage


def test_pipeline_restore_replaces_state(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """restore() should replace registry state from snapshot."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="stage1")

    # Take snapshot
    snap = p.snapshot()

    # Modify pipeline
    p.register(_stage_a, name="stage_a")
    assert "stage_a" in p.list_stages()

    # Restore from snapshot
    p.restore(snap)

    # Should have original state
    assert "stage1" in p.list_stages()
    assert "stage_a" not in p.list_stages()


def test_pipeline_clear_removes_all_stages(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """clear() should remove all registered stages."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="stage1")
    p.register(_stage_a, name="stage_a")

    assert len(p.list_stages()) == 2

    p.clear()

    assert len(p.list_stages()) == 0


# =============================================================================
# include() Tests
# =============================================================================


def test_pipeline_include_copies_stages(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """include() should copy all stages from the included pipeline."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="sub_stage")

    main.include(sub)

    assert "sub_stage" in main.list_stages()


def test_pipeline_include_preserves_state_dir(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Included stages should keep their original state_dir."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="sub_stage")
    main.include(sub)

    # Stage should have sub's state_dir, not main's
    info = main.get("sub_stage")
    assert info["state_dir"] == tmp_path / "sub" / ".pivot"


def test_pipeline_include_preserves_state_dir_transitively(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Transitive inclusion should preserve original state_dir through multiple levels.

    When A includes B and B includes C, C's stages in A should retain C's state_dir.
    Critical for ensuring lock files and state.db remain in correct locations.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    # Level 1: Base pipeline
    base = Pipeline("base", root=tmp_path / "base")
    base.register(_simple_stage, name="base_stage")

    # Level 2: Intermediate pipeline includes base
    intermediate = Pipeline("intermediate", root=tmp_path / "intermediate")
    intermediate.include(base)
    intermediate.register(_stage_a, name="intermediate_stage")

    # Level 3: Main pipeline includes intermediate (gets base transitively)
    main = Pipeline("main", root=tmp_path / "main")
    main.include(intermediate)

    # Verify base_stage retained base's state_dir (not intermediate's)
    base_info = main.get("base_stage")
    assert base_info["state_dir"] == tmp_path / "base" / ".pivot"

    # Verify intermediate_stage has intermediate's state_dir
    intermediate_info = main.get("intermediate_stage")
    assert intermediate_info["state_dir"] == tmp_path / "intermediate" / ".pivot"


def test_pipeline_include_name_collision_raises(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """include() should raise PipelineConfigError on stage name collision."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    main.register(_simple_stage, name="train")
    sub.register(_simple_stage, name="train")

    with pytest.raises(PipelineConfigError, match="train.*already exists"):
        main.include(sub)


def test_pipeline_include_collision_transitive(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Name collision should be detected even in transitively included stages."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="train")

    # Nested pipeline has same stage name
    subsub = Pipeline("subsub", root=tmp_path / "subsub")
    subsub.register(_stage_a, name="train")

    # Intermediate includes nested
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.include(subsub)

    # Should detect collision with transitively included stage
    with pytest.raises(PipelineConfigError, match="train.*already exists"):
        main.include(sub)


def test_pipeline_include_collision_is_atomic(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Include should be atomic: collision should not partially add stages.

    Important for preventing registry corruption when include() fails partway through.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="conflict")

    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_stage_a, name="safe")
    sub.register(_stage_b, name="conflict")

    initial_stages = set(main.list_stages())

    with pytest.raises(PipelineConfigError):
        main.include(sub)

    # Should be unchanged (not have "safe" added)
    assert set(main.list_stages()) == initial_stages


def test_pipeline_include_empty_pipeline(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """include() with empty pipeline should be a no-op."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="existing")

    empty = Pipeline("empty", root=tmp_path / "empty")

    main.include(empty)  # Should not raise

    assert set(main.list_stages()) == {"existing"}


def test_pipeline_include_self_raises(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Including a pipeline into itself should raise."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="stage")

    with pytest.raises(PipelineConfigError, match="cannot include itself"):
        main.include(main)


def test_pipeline_include_same_pipeline_twice_raises(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Including the same pipeline twice should raise on collision."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_simple_stage, name="sub_stage")

    main.include(sub)

    with pytest.raises(PipelineConfigError, match="sub_stage.*already exists"):
        main.include(sub)


def test_pipeline_include_multiple_different_pipelines(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Including multiple different pipelines should work."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub1 = Pipeline("sub1", root=tmp_path / "sub1")
    sub2 = Pipeline("sub2", root=tmp_path / "sub2")

    sub1.register(_simple_stage, name="stage1")
    sub2.register(_simple_stage, name="stage2")

    main.include(sub1)
    main.include(sub2)

    assert set(main.list_stages()) == {"stage1", "stage2"}


def test_pipeline_include_preserves_params(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Included stages should keep their params."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(
        _parameterized_stage_for_include,
        name="param_stage",
        params=_TestIncludeParams(value=100),
    )
    main.include(sub)

    info = main.get("param_stage")
    params = info["params"]
    assert params is not None
    assert isinstance(params, _TestIncludeParams)
    assert params.value == 100


def test_pipeline_include_isolates_mutations(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Mutations to included stages shouldn't affect source pipeline.

    Critical for preventing action-at-a-distance bugs where modifying one
    pipeline unexpectedly affects another.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="stage", mutex=["original"])
    main.include(sub)

    # Mutate main's copy
    main.get("stage")["mutex"].append("test_mutex")

    # Sub's copy should be unaffected
    assert "test_mutex" not in sub.get("stage")["mutex"]


def test_pipeline_include_is_independent_copy(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Stages are copied at include time; subsequent changes don't propagate."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="original")
    main.include(sub)

    # Modify sub after inclusion
    sub.register(_stage_a, name="new_stage")

    # Main should be unaffected
    assert "original" in main.list_stages()
    assert "new_stage" not in main.list_stages()


def test_pipeline_include_invalidates_dag_cache(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Include should invalidate cached DAG to ensure freshness."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path)
    main.register(_stage_a, name="stage_a")

    # Build DAG (caches it)
    dag1 = main.build_dag(validate=False)
    assert len(dag1.nodes) == 1

    # Include another pipeline
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_stage_b, name="stage_b")
    main.include(sub)

    # Subsequent build_dag should see new stage
    dag2 = main.build_dag(validate=False)
    assert len(dag2.nodes) == 2
    assert "stage_b" in dag2.nodes


def test_pipeline_include_dag_connects_across_pipelines(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """DAG should connect main stage depending on included sub-pipeline stage.

    When including a sub-pipeline, cross-pipeline dependencies work because:
    - Producer in sub outputs sub/a.txt (relative to project root)
    - Consumer in main can reference sub/a.txt to depend on it
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path)
    sub = Pipeline("sub", root=tmp_path / "sub")

    # Producer outputs a.txt (resolved to sub/a.txt project-relative)
    sub.register(_stage_a, name="producer")
    main.include(sub)

    # Consumer needs to reference sub/a.txt (the producer's actual output path)
    # Use dep_path_overrides to point to the correct path
    main.register(
        _stage_b,
        name="consumer",
        dep_path_overrides={"data": "sub/a.txt"},  # Reference producer's output
    )

    dag = main.build_dag(validate=True)

    assert "producer" in dag.nodes
    assert "consumer" in dag.nodes
    # consumer depends on producer (edge from consumer -> producer)
    assert dag.has_edge("consumer", "producer")


def test_pipeline_include_preserves_all_metadata(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Included stages should preserve all metadata fields (deps, outs, mutex, variant, fingerprint)."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    # Register stage with various metadata
    sub.register(
        _stage_b,
        name="complex_stage",
        mutex=["gpu", "memory"],
        variant="test_variant",
    )

    original_info = sub.get("complex_stage")
    main.include(sub)
    included_info = main.get("complex_stage")

    # Verify all critical fields preserved
    assert included_info["deps"] == original_info["deps"]
    assert included_info["deps_paths"] == original_info["deps_paths"]
    assert included_info["outs"] == original_info["outs"]
    assert included_info["outs_paths"] == original_info["outs_paths"]
    assert included_info["mutex"] == original_info["mutex"]
    assert included_info["variant"] == original_info["variant"]
    assert included_info["fingerprint"] == original_info["fingerprint"]
    assert included_info["signature"] == original_info["signature"]
    assert included_info["func"] == original_info["func"]


def test_pipeline_include_isolates_nested_mutations(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Deep copy should isolate mutations to nested structures (params object)."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(
        _parameterized_stage_for_include,
        name="stage",
        params=_TestIncludeParams(value=100),
    )

    main.include(sub)

    # Get params from both pipelines
    included_params = main.get("stage")["params"]
    original_params = sub.get("stage")["params"]

    assert isinstance(included_params, _TestIncludeParams)
    assert isinstance(original_params, _TestIncludeParams)

    # Params should be different objects (deep copied)
    assert included_params is not original_params

    # Verify values are equal but independent
    assert included_params.value == original_params.value == 100


def test_pipeline_include_multiple_collisions_atomic(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Multiple collisions should still be atomic (no partial adds)."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="collide_a")
    main.register(_stage_a, name="collide_b")

    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_stage_b, name="collide_a")  # First collision
    sub.register(_simple_stage, name="collide_b")  # Second collision
    sub.register(_stage_a, name="safe_stage")  # Would be safe if others didn't collide

    initial_stages = set(main.list_stages())

    with pytest.raises(PipelineConfigError, match="collide_a.*already exists"):
        main.include(sub)

    # Should be unchanged (safe_stage not added)
    assert set(main.list_stages()) == initial_stages


def test_pipeline_include_deepcopy_failure_is_atomic(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Include should be atomic even when deepcopy fails mid-operation.

    If deepcopy raises an exception on the Nth stage, no stages should be added.
    """
    import copy

    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    # Register 3 stages in sub
    sub.register(_simple_stage, name="stage1")
    sub.register(_stage_a, name="stage2")
    sub.register(_stage_b, name="stage3")

    initial_stages = set(main.list_stages())
    deepcopy_call_count = 0

    original_deepcopy = copy.deepcopy

    def failing_deepcopy(obj: object) -> object:
        nonlocal deepcopy_call_count
        deepcopy_call_count += 1
        if deepcopy_call_count == 2:  # Fail on second stage
            raise RuntimeError("Simulated deepcopy failure")
        return original_deepcopy(obj)

    mocker.patch.object(copy, "deepcopy", failing_deepcopy)

    with pytest.raises(RuntimeError, match="Simulated deepcopy failure"):
        main.include(sub)

    # Should be unchanged (no stages added)
    assert set(main.list_stages()) == initial_stages


# =============================================================================
# Pipeline Path Resolution Tests (register() resolves paths relative to pipeline root)
# =============================================================================


# Helper stages for path resolution tests - defined at module level for fingerprinting
class _PathResolutionOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("data/output.txt", loaders.PathOnly())]


def _path_resolution_stage(
    data: Annotated[pathlib.Path, outputs.Dep("data/input.txt", loaders.PathOnly())],
) -> _PathResolutionOutput:
    return _PathResolutionOutput(result=pathlib.Path("data/output.txt"))


def _single_output_stage(
    data: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]:
    return pathlib.Path("output.txt")


class _MultiPathDepOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("merged.txt", loaders.PathOnly())]


def _multi_path_dep_stage(
    files: Annotated[
        list[pathlib.Path], outputs.Dep(["data/a.txt", "data/b.txt"], loaders.PathOnly())
    ],
) -> _MultiPathDepOutput:
    return _MultiPathDepOutput(result=pathlib.Path("merged.txt"))


# Cross-pipeline dep helper (uses ../)
class _CrossPipelineOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _cross_pipeline_stage(
    data: Annotated[pathlib.Path, outputs.Dep("../shared/data.csv", loaders.PathOnly())],
) -> _CrossPipelineOutput:
    return _CrossPipelineOutput(result=pathlib.Path("result.txt"))


# Absolute path helper
class _AbsolutePathOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _absolute_path_stage(
    data: Annotated[pathlib.Path, outputs.Dep("/data/external/file.csv", loaders.PathOnly())],
) -> _AbsolutePathOutput:
    return _AbsolutePathOutput(result=pathlib.Path("result.txt"))


# Windows path helper
class _WindowsPathOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("out/result.txt", loaders.PathOnly())]


def _windows_path_stage(
    data: Annotated[pathlib.Path, outputs.Dep("data\\input.txt", loaders.PathOnly())],
) -> _WindowsPathOutput:
    return _WindowsPathOutput(result=pathlib.Path("out/result.txt"))


# Escape project root helper
class _EscapeOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _escape_stage(
    data: Annotated[pathlib.Path, outputs.Dep("../../outside_project.csv", loaders.PathOnly())],
) -> _EscapeOutput:
    return _EscapeOutput(result=pathlib.Path("result.txt"))


# Override stage helper
def _override_stage(
    data: Annotated[pathlib.Path, outputs.PlaceholderDep(loaders.PathOnly())],
) -> Annotated[pathlib.Path, outputs.Out("default_out.txt", loaders.PathOnly())]:
    return pathlib.Path("overridden_out.txt")


# Cache option stage helper
def _cache_option_stage() -> Annotated[
    pathlib.Path, outputs.Out("default.txt", loaders.PathOnly())
]:
    return pathlib.Path("output.txt")


def test_register_resolves_relative_to_pipeline_root(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Paths in annotations should be resolved relative to pipeline root, not project root.

    Pipeline at /project/subdir/ with annotation path "data/input.txt"
    should resolve to /project/subdir/data/input.txt, stored as project-relative
    "subdir/data/input.txt".
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    # Pipeline in subdirectory
    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(_path_resolution_stage, name="my_stage")

    info = p.get("my_stage")

    # Dep path should be project-relative: subdir/data/input.txt
    # (flattened deps_paths contains absolute paths after registration normalization)
    # The deps dict should have the resolved path
    assert "data" in info["deps"]
    # After normalization, paths are absolute in deps_paths
    expected_dep = str(project_root / "subdir" / "data" / "input.txt")
    assert expected_dep in info["deps_paths"]

    # Out path should also be project-relative
    expected_out = str(project_root / "subdir" / "data" / "output.txt")
    assert expected_out in info["outs_paths"]


def test_register_cross_pipeline_dep_with_dotdot(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Pipeline can depend on files outside its root using ../ traversal.

    Pipeline at /project/pipelines/eval/ depending on ../shared/data.csv
    should resolve to /project/pipelines/shared/data.csv.
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    # Create directory structure
    eval_pipeline = project_root / "pipelines" / "eval"
    eval_pipeline.mkdir(parents=True)
    shared_dir = project_root / "pipelines" / "shared"
    shared_dir.mkdir(parents=True)

    p = Pipeline("eval", root=eval_pipeline)
    p.register(_cross_pipeline_stage, name="cross_stage")

    info = p.get("cross_stage")

    # Dep should resolve to pipelines/shared/data.csv (project-relative)
    # After normalization, stored as absolute path
    expected_dep = str(project_root / "pipelines" / "shared" / "data.csv")
    assert expected_dep in info["deps_paths"]


def test_register_absolute_path_unchanged(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Absolute paths should be normalized but kept absolute.

    An absolute path like /data/external/file.csv should remain absolute
    (not converted to project-relative).
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(_absolute_path_stage, name="abs_stage")

    info = p.get("abs_stage")

    # Absolute path should be preserved (normalized, but still absolute)
    assert "/data/external/file.csv" in info["deps_paths"]


def test_register_windows_path_normalized(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Windows-style paths should be normalized to POSIX format.

    A path like "data\\subdir\\file.txt" should become "data/subdir/file.txt".
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(_windows_path_stage, name="win_stage")

    info = p.get("win_stage")

    # Paths should be normalized (no backslashes in final paths)
    for dep_path in info["deps_paths"]:
        assert "\\" not in dep_path, f"Backslash found in dep path: {dep_path}"
    for out_path in info["outs_paths"]:
        assert "\\" not in out_path, f"Backslash found in out path: {out_path}"


def test_register_overrides_are_pipeline_relative(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """dep_path_overrides and out_path_overrides should be resolved relative to pipeline root."""
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(
        _override_stage,
        name="override_stage",
        dep_path_overrides={"data": "custom/input.csv"},
        out_path_overrides={"_single": "custom/output.csv"},
    )

    info = p.get("override_stage")

    # Override paths should be resolved from pipeline root (subdir/)
    expected_dep = str(project_root / "subdir" / "custom" / "input.csv")
    assert expected_dep in info["deps_paths"]

    expected_out = str(project_root / "subdir" / "custom" / "output.csv")
    assert expected_out in info["outs_paths"]


def test_register_validates_after_normalization(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Validation should happen after path normalization.

    - ../foo should be allowed (collapses to valid path)
    - But escaping project root should be rejected
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    # Pipeline at project root level
    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)

    # Should raise because ../../outside_project.csv from subdir/ escapes project root
    with pytest.raises(
        (ValueError, exceptions.InvalidPathError, exceptions.SecurityValidationError)
    ):
        p.register(_escape_stage, name="escape_stage")


def test_register_single_output_stage(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Single-output stages (non-TypedDict return) should have paths resolved correctly."""
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(_single_output_stage, name="single_out")

    info = p.get("single_out")

    # Both dep and out should be resolved from pipeline root
    expected_dep = str(project_root / "subdir" / "input.txt")
    assert expected_dep in info["deps_paths"]

    expected_out = str(project_root / "subdir" / "output.txt")
    assert expected_out in info["outs_paths"]


def test_register_out_override_with_cache_option(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """OutOverride dict with cache option should preserve the cache flag."""
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(
        _cache_option_stage,
        name="cache_stage",
        out_path_overrides={"_single": {"path": "custom/output.txt", "cache": False}},
    )

    info = p.get("cache_stage")

    # Output path should be resolved from pipeline root
    expected_out = str(project_root / "subdir" / "custom" / "output.txt")
    assert expected_out in info["outs_paths"]

    # Cache option should be preserved
    assert len(info["outs"]) == 1
    assert info["outs"][0].cache is False


def test_register_multi_path_dep_resolved(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """Multi-path deps (list/tuple) should have each path resolved relative to pipeline root."""
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    pipeline_root = project_root / "subdir"
    pipeline_root.mkdir()

    p = Pipeline("test", root=pipeline_root)
    p.register(_multi_path_dep_stage, name="multi_dep")

    info = p.get("multi_dep")

    # Both paths should be resolved from pipeline root
    expected_a = str(project_root / "subdir" / "data" / "a.txt")
    expected_b = str(project_root / "subdir" / "data" / "b.txt")

    assert expected_a in info["deps_paths"]
    assert expected_b in info["deps_paths"]


# =============================================================================
# Cross-Pipeline DAG Integration Tests
# =============================================================================


# Helper stages for cross-pipeline tests - defined at module level for fingerprinting
class _SharedDataOutput(TypedDict):
    data: Annotated[pathlib.Path, outputs.Out("data.csv", loaders.PathOnly())]


def _shared_data_producer() -> _SharedDataOutput:
    pathlib.Path("data.csv").write_text("shared data")
    return _SharedDataOutput(data=pathlib.Path("data.csv"))


class _CrossPipelineConsumerOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _cross_pipeline_consumer(
    data: Annotated[pathlib.Path, outputs.Dep("../shared/data.csv", loaders.PathOnly())],
) -> _CrossPipelineConsumerOutput:
    return _CrossPipelineConsumerOutput(result=pathlib.Path("result.txt"))


def test_dag_connects_cross_pipeline_deps(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """DAG should correctly connect stages when dependency uses ../ to reference cross-pipeline output.

    When pipeline at /project/pipelines/eval/ depends on ../shared/data.csv,
    and pipeline at /project/pipelines/shared/ produces data.csv,
    the DAG should have an edge from consumer to producer.
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    # Create directory structure: pipelines/shared/ and pipelines/eval/
    shared_root = project_root / "pipelines" / "shared"
    shared_root.mkdir(parents=True)
    eval_root = project_root / "pipelines" / "eval"
    eval_root.mkdir(parents=True)

    # Main pipeline at project root - will include both sub-pipelines
    main = Pipeline("main", root=project_root)

    # Shared pipeline produces data.csv (resolved to pipelines/shared/data.csv)
    shared = Pipeline("shared", root=shared_root)
    shared.register(_shared_data_producer, name="producer")

    # Eval pipeline consumes ../shared/data.csv (resolved to pipelines/shared/data.csv)
    eval_pipeline = Pipeline("eval", root=eval_root)
    eval_pipeline.register(_cross_pipeline_consumer, name="consumer")

    # Include both pipelines into main
    main.include(shared)
    main.include(eval_pipeline)

    # Build DAG - should connect consumer -> producer
    dag = main.build_dag(validate=True)

    assert "producer" in dag.nodes
    assert "consumer" in dag.nodes
    # Edge from consumer to producer (consumer depends on producer)
    assert dag.has_edge("consumer", "producer"), (
        f"Expected edge consumer->producer. Edges: {list(dag.edges)}"
    )


def test_dag_missing_dep_raises(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """DAG validation should raise DependencyNotFoundError when dep doesn't match any output.

    When a stage depends on a path that:
    1. Is not produced by any registered stage
    2. Does not exist on disk
    The build_dag(validate=True) should raise DependencyNotFoundError.
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    eval_root = project_root / "pipelines" / "eval"
    eval_root.mkdir(parents=True)

    p = Pipeline("eval", root=eval_root)
    # Consumer depends on ../shared/data.csv but no producer registered
    p.register(_cross_pipeline_consumer, name="consumer")

    # Should raise because dependency doesn't exist and no stage produces it
    with pytest.raises(exceptions.DependencyNotFoundError) as exc_info:
        p.build_dag(validate=True)

    # Verify error message contains the missing path
    assert "consumer" in str(exc_info.value)
    assert "shared" in str(exc_info.value) or "data.csv" in str(exc_info.value)


def test_included_pipeline_paths_resolve_correctly(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """When using include(), included pipeline stages should have paths already resolved.

    The included stage's paths should be resolved relative to its original pipeline root,
    not the including pipeline's root.
    """
    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    # Sub-pipeline in subdir/
    sub_root = project_root / "subdir"
    sub_root.mkdir()

    sub = Pipeline("sub", root=sub_root)
    sub.register(_path_resolution_stage, name="sub_stage")

    # Main pipeline at project root
    main = Pipeline("main", root=project_root)
    main.include(sub)

    # Get the included stage's info
    info = main.get("sub_stage")

    # Paths should be resolved from sub_root (subdir/), not main's root
    # Dep: data/input.txt -> subdir/data/input.txt (absolute)
    expected_dep = str(project_root / "subdir" / "data" / "input.txt")
    assert expected_dep in info["deps_paths"], (
        f"Expected dep path {expected_dep} in {info['deps_paths']}"
    )

    # Out: data/output.txt -> subdir/data/output.txt (absolute)
    expected_out = str(project_root / "subdir" / "data" / "output.txt")
    assert expected_out in info["outs_paths"], (
        f"Expected out path {expected_out} in {info['outs_paths']}"
    )


def test_lockfile_contains_project_relative_paths(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Lock files should store project-relative paths, not absolute or pipeline-relative.

    This test verifies that when a lock file is written, the paths stored in it
    are relative to the project root, ensuring portability.
    """
    from pivot.storage import lock
    from pivot.types import LockData

    project_root = tmp_path
    mocker.patch.object(project, "_project_root_cache", project_root)
    (project_root / ".git").mkdir()

    # Create stages dir
    stages_dir = project_root / ".pivot" / "stages"
    stages_dir.mkdir(parents=True)

    # Create a StageLock and write data with absolute paths
    stage_lock = lock.StageLock("test_stage", stages_dir)

    # Simulate lock data with absolute paths (as used internally)
    abs_dep = str(project_root / "subdir" / "data" / "input.txt")
    abs_out = str(project_root / "subdir" / "data" / "output.txt")

    lock_data = LockData(
        code_manifest={"main": "abc123"},
        params={"key": "value"},
        dep_hashes={abs_dep: {"hash": "dep_hash_123"}},
        output_hashes={abs_out: {"hash": "out_hash_456"}},
        dep_generations={"producer": 1},
    )

    # Write the lock file
    stage_lock.write(lock_data)

    # Read the raw YAML to verify paths are project-relative
    import yaml

    raw_content = stage_lock.path.read_text()
    raw_data = yaml.safe_load(raw_content)

    # Deps should be project-relative (not absolute)
    assert len(raw_data["deps"]) == 1
    dep_path = raw_data["deps"][0]["path"]
    assert dep_path == "subdir/data/input.txt", (
        f"Expected project-relative path 'subdir/data/input.txt', got '{dep_path}'"
    )
    assert not dep_path.startswith("/"), "Dep path should not be absolute"

    # Outs should be project-relative (not absolute)
    assert len(raw_data["outs"]) == 1
    out_path = raw_data["outs"][0]["path"]
    assert out_path == "subdir/data/output.txt", (
        f"Expected project-relative path 'subdir/data/output.txt', got '{out_path}'"
    )
    assert not out_path.startswith("/"), "Out path should not be absolute"

    # Verify round-trip: read() should convert back to absolute paths
    read_back = stage_lock.read()
    assert read_back is not None
    assert abs_dep in read_back["dep_hashes"]
    assert abs_out in read_back["output_hashes"]


# =============================================================================
# Path Resolution Edge Case Tests
# =============================================================================


def test_resolve_path_rejects_empty_string(tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
    """_resolve_path should reject empty string paths."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)

    with pytest.raises(ValueError, match="cannot be empty"):
        p._resolve_path("")


def test_resolve_path_rejects_whitespace_only(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """_resolve_path should reject whitespace-only paths."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        p._resolve_path("   ")

    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        p._resolve_path("\t\n")


def test_resolve_path_rejects_root_only_paths(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """_resolve_path should reject root-only paths like '/' or 'C:\\'."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path)

    with pytest.raises(ValueError, match="cannot be a root directory"):
        p._resolve_path("/")

    with pytest.raises(ValueError, match="cannot be a root directory"):
        p._resolve_path("\\")

    # Windows drive roots
    with pytest.raises(ValueError, match="cannot be a root directory"):
        p._resolve_path("C:\\")

    with pytest.raises(ValueError, match="cannot be a root directory"):
        p._resolve_path("D:/")


def test_resolve_path_handles_windows_drive_letters(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """_resolve_path should recognize Windows drive letters as absolute paths.

    Windows paths like C:\\path or D:/path should be treated as absolute,
    not joined with the pipeline root.
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path / "subdir")

    # Windows paths should be preserved as absolute (converted to POSIX format)
    # Note: On Unix, these paths are still "absolute" in the sense they have a drive letter
    resolved = p._resolve_path("C:/data/file.csv")
    # On Unix, this becomes a path like "C:/data/file.csv" (POSIX format)
    # The key is that it's NOT joined with pipeline root
    assert "subdir" not in resolved, (
        f"Windows absolute path should not be joined with pipeline root: {resolved}"
    )

    resolved = p._resolve_path("D:\\path\\to\\file.txt")
    assert "subdir" not in resolved, (
        f"Windows absolute path should not be joined with pipeline root: {resolved}"
    )


# =============================================================================
# IncrementalOut Registration Tests
# =============================================================================


# Module-level helper for IncrementalOut tests
def incremental_cache_stage(
    existing: Annotated[
        dict[str, str], outputs.IncrementalOut("data/cache.yaml", loaders.YAML[dict[str, str]]())
    ],
) -> Annotated[
    dict[str, str], outputs.IncrementalOut("data/cache.yaml", loaders.YAML[dict[str, str]]())
]:
    """Stage with IncrementalOut - both input and output use same path/loader."""
    existing["new_key"] = "new_value"
    return existing


def test_pipeline_register_incremental_out_stage(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Pipeline.register() should succeed for stages using IncrementalOut.

    IncrementalOut paths cannot be overridden (registry rejects path overrides
    for them), so Pipeline must NOT pass IncrementalOut paths as overrides.
    The paths remain as specified in the annotations (normalized by registry).
    """
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    p = Pipeline("test", root=tmp_path / "subdir")

    # This should NOT raise - IncrementalOut paths are not overridden
    p.register(incremental_cache_stage)

    # Verify the stage was registered
    assert "incremental_cache_stage" in p.list_stages()
    info = p.get("incremental_cache_stage")

    # IncrementalOut path is preserved (not resolved relative to pipeline root)
    # Registry normalizes it to absolute based on project root
    assert info["outs_paths"][0].endswith("data/cache.yaml")
