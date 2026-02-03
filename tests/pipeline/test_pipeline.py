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


def test_pipeline_register_stage(tmp_path: pathlib.Path) -> None:
    """Pipeline.register should register a stage with the pipeline's state_dir."""
    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="my_stage")

    assert "my_stage" in p.list_stages()
    info = p.get("my_stage")
    assert info["state_dir"] == tmp_path / ".pivot"


def test_pipeline_stages_isolated(tmp_path: pathlib.Path) -> None:
    """Two pipelines can have stages with the same name."""
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


def test_pipeline_register_missing_annotation_fails(tmp_path: pathlib.Path) -> None:
    """Pipeline.register fails when output TypedDict field missing Annotated wrapper.

    Critical for catching user errors where they forget Annotated[] on TypedDict fields.
    This prevents silent registration of stages that won't work at runtime.
    """
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


def test_pipeline_snapshot_captures_current_state(tmp_path: pathlib.Path) -> None:
    """snapshot() should capture current registry state for rollback."""
    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="stage1")

    snap = p.snapshot()

    assert "stage1" in snap
    assert snap["stage1"]["name"] == "stage1"
    assert snap["stage1"]["func"] == _simple_stage


def test_pipeline_restore_replaces_state(tmp_path: pathlib.Path) -> None:
    """restore() should replace registry state from snapshot."""
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


def test_pipeline_clear_removes_all_stages(tmp_path: pathlib.Path) -> None:
    """clear() should remove all registered stages."""
    p = Pipeline("test", root=tmp_path)
    p.register(_simple_stage, name="stage1")
    p.register(_stage_a, name="stage_a")

    assert len(p.list_stages()) == 2

    p.clear()

    assert len(p.list_stages()) == 0


# =============================================================================
# include() Tests
# =============================================================================


def test_pipeline_include_copies_stages(tmp_path: pathlib.Path) -> None:
    """include() should copy all stages from the included pipeline."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="sub_stage")

    main.include(sub)

    assert "sub_stage" in main.list_stages()


def test_pipeline_include_preserves_state_dir(tmp_path: pathlib.Path) -> None:
    """Included stages should keep their original state_dir."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="sub_stage")
    main.include(sub)

    # Stage should have sub's state_dir, not main's
    info = main.get("sub_stage")
    assert info["state_dir"] == tmp_path / "sub" / ".pivot"


def test_pipeline_include_preserves_state_dir_transitively(tmp_path: pathlib.Path) -> None:
    """Transitive inclusion should preserve original state_dir through multiple levels.

    When A includes B and B includes C, C's stages in A should retain C's state_dir.
    Critical for ensuring lock files and state.db remain in correct locations.
    """
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


def test_pipeline_include_name_collision_raises(tmp_path: pathlib.Path) -> None:
    """include() should raise PipelineConfigError on stage name collision."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    main.register(_simple_stage, name="train")
    sub.register(_simple_stage, name="train")

    with pytest.raises(PipelineConfigError, match="train.*already exists"):
        main.include(sub)


def test_pipeline_include_collision_transitive(tmp_path: pathlib.Path) -> None:
    """Name collision should be detected even in transitively included stages."""
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


def test_pipeline_include_collision_is_atomic(tmp_path: pathlib.Path) -> None:
    """Include should be atomic: collision should not partially add stages.

    Important for preventing registry corruption when include() fails partway through.
    """
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


def test_pipeline_include_empty_pipeline(tmp_path: pathlib.Path) -> None:
    """include() with empty pipeline should be a no-op."""
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="existing")

    empty = Pipeline("empty", root=tmp_path / "empty")

    main.include(empty)  # Should not raise

    assert set(main.list_stages()) == {"existing"}


def test_pipeline_include_self_raises(tmp_path: pathlib.Path) -> None:
    """Including a pipeline into itself should raise."""
    main = Pipeline("main", root=tmp_path / "main")
    main.register(_simple_stage, name="stage")

    with pytest.raises(PipelineConfigError, match="cannot include itself"):
        main.include(main)


def test_pipeline_include_same_pipeline_twice_raises(tmp_path: pathlib.Path) -> None:
    """Including the same pipeline twice should raise on collision."""
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")
    sub.register(_simple_stage, name="sub_stage")

    main.include(sub)

    with pytest.raises(PipelineConfigError, match="sub_stage.*already exists"):
        main.include(sub)


def test_pipeline_include_multiple_different_pipelines(tmp_path: pathlib.Path) -> None:
    """Including multiple different pipelines should work."""
    main = Pipeline("main", root=tmp_path / "main")
    sub1 = Pipeline("sub1", root=tmp_path / "sub1")
    sub2 = Pipeline("sub2", root=tmp_path / "sub2")

    sub1.register(_simple_stage, name="stage1")
    sub2.register(_simple_stage, name="stage2")

    main.include(sub1)
    main.include(sub2)

    assert set(main.list_stages()) == {"stage1", "stage2"}


def test_pipeline_include_preserves_params(tmp_path: pathlib.Path) -> None:
    """Included stages should keep their params."""
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


def test_pipeline_include_isolates_mutations(tmp_path: pathlib.Path) -> None:
    """Mutations to included stages shouldn't affect source pipeline.

    Critical for preventing action-at-a-distance bugs where modifying one
    pipeline unexpectedly affects another.
    """
    main = Pipeline("main", root=tmp_path / "main")
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_simple_stage, name="stage", mutex=["original"])
    main.include(sub)

    # Mutate main's copy
    main.get("stage")["mutex"].append("test_mutex")

    # Sub's copy should be unaffected
    assert "test_mutex" not in sub.get("stage")["mutex"]


def test_pipeline_include_is_independent_copy(tmp_path: pathlib.Path) -> None:
    """Stages are copied at include time; subsequent changes don't propagate."""
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
    """DAG should connect main stage depending on included sub-pipeline stage."""
    mocker.patch.object(project, "_project_root_cache", tmp_path)
    (tmp_path / ".git").mkdir()

    main = Pipeline("main", root=tmp_path)
    sub = Pipeline("sub", root=tmp_path / "sub")

    sub.register(_stage_a, name="producer")
    main.include(sub)
    main.register(_stage_b, name="consumer")

    dag = main.build_dag(validate=True)

    assert "producer" in dag.nodes
    assert "consumer" in dag.nodes
    # consumer depends on producer (edge from consumer -> producer)
    assert dag.has_edge("consumer", "producer")


def test_pipeline_include_preserves_all_metadata(tmp_path: pathlib.Path) -> None:
    """Included stages should preserve all metadata fields (deps, outs, mutex, variant, fingerprint)."""
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


def test_pipeline_include_isolates_nested_mutations(tmp_path: pathlib.Path) -> None:
    """Deep copy should isolate mutations to nested structures (params object)."""
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


def test_pipeline_include_multiple_collisions_atomic(tmp_path: pathlib.Path) -> None:
    """Multiple collisions should still be atomic (no partial adds)."""
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
