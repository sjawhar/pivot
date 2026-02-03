# tests/integration/test_lazy_resolution.py
"""Integration tests for lazy pipeline dependency resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import stage_module_isolation
from pivot import discovery

if TYPE_CHECKING:
    import pathlib


# =============================================================================
# Helper functions for generating pipeline code
# =============================================================================


def _make_producer_pipeline_code(
    name: str,
    stage_name: str,
    output_path: str,
) -> str:
    """Generate pipeline code for a producer stage."""
    return f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("{name}")

class _Output(TypedDict):
    data: Annotated[Path, Out("{output_path}", loaders.PathOnly())]

def {stage_name}() -> _Output:
    Path("{output_path}").parent.mkdir(parents=True, exist_ok=True)
    Path("{output_path}").write_text("produced")
    return _Output(data=Path("{output_path}"))

pipeline.register({stage_name})
'''


def _make_consumer_pipeline_code(
    name: str,
    stage_name: str,
    dep_path: str,
    output_path: str,
) -> str:
    """Generate pipeline code for a consumer stage."""
    return f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out, Dep

pipeline = Pipeline("{name}")

class _Output(TypedDict):
    result: Annotated[Path, Out("{output_path}", loaders.PathOnly())]

def {stage_name}(
    data: Annotated[Path, Dep("{dep_path}", loaders.PathOnly())]
) -> _Output:
    return _Output(result=Path("{output_path}"))

pipeline.register({stage_name})
'''


def _make_transform_pipeline_code(
    name: str,
    stage_name: str,
    dep_path: str,
    output_path: str,
) -> str:
    """Generate pipeline code for a transform stage (consumes and produces)."""
    return f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out, Dep

pipeline = Pipeline("{name}")

class _Output(TypedDict):
    data: Annotated[Path, Out("{output_path}", loaders.PathOnly())]

def {stage_name}(
    input_data: Annotated[Path, Dep("{dep_path}", loaders.PathOnly())]
) -> _Output:
    Path("{output_path}").write_text("transformed")
    return _Output(data=Path("{output_path}"))

pipeline.register({stage_name})
'''


# =============================================================================
# Integration Tests
# =============================================================================


def test_lazy_resolution_builds_complete_dag(set_project_root: pathlib.Path) -> None:
    """Child pipeline should build complete DAG including parent stages.

    End-to-end test verifying that resolve_from_parents() enables build_dag()
    to succeed by including necessary producers from parent pipeline.
    """
    # Parent pipeline at root
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("parent", "producer", "data/output.txt")
    )

    # Child pipeline
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    # Load child pipeline and resolve
    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None, "Failed to load child pipeline"

    child.resolve_from_parents()
    dag = child.build_dag(validate=True)

    assert "producer" in dag.nodes, "Expected producer from parent in DAG"
    assert "consumer" in dag.nodes, "Expected consumer from child in DAG"
    assert dag.has_edge("consumer", "producer"), "Expected edge from consumer to producer"


def test_lazy_resolution_preserves_parent_state_dir(set_project_root: pathlib.Path) -> None:
    """Included parent stages should retain their original state_dir.

    Critical for correctness: lock files and state.db must remain in parent's
    .pivot directory, not child's, to avoid conflicts and enable proper
    incremental builds.
    """
    # Parent pipeline at root
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("parent", "producer", "data/output.txt")
    )

    # Child pipeline
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    # Load child pipeline and resolve
    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None, "Failed to load child pipeline"

    child.resolve_from_parents()

    # Producer's state_dir should be parent's .pivot, not child's
    producer_info = child.get("producer")
    assert producer_info["state_dir"] == set_project_root / ".pivot", (
        f"Expected producer state_dir to be {set_project_root / '.pivot'}, "
        f"got {producer_info['state_dir']}"
    )

    consumer_info = child.get("consumer")
    assert consumer_info["state_dir"] == child_dir / ".pivot", (
        f"Expected consumer state_dir to be {child_dir / '.pivot'}, "
        f"got {consumer_info['state_dir']}"
    )


def test_lazy_resolution_multilevel_hierarchy(set_project_root: pathlib.Path) -> None:
    """Should resolve dependencies through multiple levels of parent pipelines.

    Tests grandparent -> parent -> child dependency chain to ensure
    transitive resolution works across multiple directory levels.
    """
    # Grandparent produces raw.txt
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("grandparent", "extract", "raw.txt")
    )

    # Parent depends on raw.txt, produces processed.txt
    parent_dir = set_project_root / "parent"
    parent_dir.mkdir()
    (parent_dir / "pipeline.py").write_text(
        _make_transform_pipeline_code("parent", "process", "../raw.txt", "processed.txt")
    )

    # Child depends on processed.txt, produces final.txt
    child_dir = parent_dir / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "finalize", "../processed.txt", "final.txt")
    )

    # Load child and resolve
    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None, "Failed to load child pipeline"

    child.resolve_from_parents()

    # Should have all three stages
    stages = set(child.list_stages())
    assert stages == {"extract", "process", "finalize"}, (
        f"Expected all three stages to be included, got {stages}"
    )

    # Build DAG should succeed
    dag = child.build_dag(validate=True)

    # Verify dependency chain
    assert dag.has_edge("finalize", "process"), "Expected finalize -> process edge"
    assert dag.has_edge("process", "extract"), "Expected process -> extract edge"

    # Verify state_dirs are preserved
    assert child.get("extract")["state_dir"] == set_project_root / ".pivot"
    assert child.get("process")["state_dir"] == parent_dir / ".pivot"
    assert child.get("finalize")["state_dir"] == child_dir / ".pivot"


def test_lazy_resolution_with_pivot_yaml_parent(set_project_root: pathlib.Path) -> None:
    """Should resolve dependencies from parent defined in pivot.yaml.

    Tests that lazy resolution works with YAML-configured parent pipelines,
    not just pipeline.py.
    """
    # Create parent stages.py module
    parent_stages = set_project_root / "stages.py"
    parent_stages.write_text("""
from typing import Annotated, TypedDict
from pathlib import Path
from pivot import loaders
from pivot.outputs import Out

class ProducerOutput(TypedDict):
    data: Annotated[Path, Out("data.txt", loaders.PathOnly())]

def producer() -> ProducerOutput:
    Path("data.txt").write_text("from yaml parent")
    return ProducerOutput(data=Path("data.txt"))
""")

    # Parent pipeline via pivot.yaml
    pivot_yaml = set_project_root / "pivot.yaml"
    pivot_yaml.write_text("""
pipeline: yaml_parent
stages:
  producer:
    python: stages.producer
""")

    # Child pipeline.py depends on data.txt
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data.txt", "result.txt")
    )

    with stage_module_isolation(set_project_root):
        # Load child and resolve
        child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
        assert child is not None, "Failed to load child pipeline"

        child.resolve_from_parents()

    # Should include producer from YAML parent
    assert "producer" in child.list_stages(), "Expected to include producer from pivot.yaml parent"
    assert "consumer" in child.list_stages()

    # Build DAG should succeed
    dag = child.build_dag(validate=True)
    assert dag.has_edge("consumer", "producer")
