from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest

from pivot import exceptions, loaders, outputs, project
from pivot.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# Module-level helper for stage registration tests
class _SimpleOutput(TypedDict):
    result: Annotated[pathlib.Path, outputs.Out("result.txt", loaders.PathOnly())]


def _simple_stage() -> _SimpleOutput:
    pathlib.Path("result.txt").write_text("done")
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
