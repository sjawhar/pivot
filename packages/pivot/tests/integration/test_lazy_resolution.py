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

    End-to-end test verifying that resolve_external_dependencies() enables build_dag()
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

    child.resolve_external_dependencies()
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

    child.resolve_external_dependencies()

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

    child.resolve_external_dependencies()

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

        child.resolve_external_dependencies()

    # Should include producer from YAML parent
    assert "producer" in child.list_stages(), "Expected to include producer from pivot.yaml parent"
    assert "consumer" in child.list_stages()

    # Build DAG should succeed
    dag = child.build_dag(validate=True)
    assert dag.has_edge("consumer", "producer")


def test_resolve_external_dependencies_sibling_pipelines(
    set_project_root: pathlib.Path,
) -> None:
    """Should resolve dependencies from sibling pipeline directories.

    This is the core use case: time_horizon_1_0 depends on output from
    time_horizon_1_1, where both are siblings under model_reports/.
    """
    # Create sibling structure:
    # project_root/
    #   model_reports/
    #     sibling_a/pipeline.py  <- consumer, depends on ../sibling_b/data/output.txt
    #     sibling_b/pipeline.py  <- producer of data/output.txt

    model_reports = set_project_root / "model_reports"
    sibling_a = model_reports / "sibling_a"
    sibling_b = model_reports / "sibling_b"
    sibling_a.mkdir(parents=True)
    sibling_b.mkdir(parents=True)

    # Producer in sibling_b
    (sibling_b / "pipeline.py").write_text(
        _make_producer_pipeline_code("sibling_b", "producer", "data/output.txt")
    )

    # Consumer in sibling_a depends on sibling_b's output
    (sibling_a / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "sibling_a", "consumer", "../sibling_b/data/output.txt", "result.txt"
        )
    )

    # Load consumer pipeline and resolve
    consumer = discovery.load_pipeline_from_path(sibling_a / "pipeline.py")
    assert consumer is not None

    consumer.resolve_external_dependencies()

    # Should include producer from sibling
    assert "producer" in consumer.list_stages()
    assert "consumer" in consumer.list_stages()

    # Build DAG should succeed
    dag = consumer.build_dag(validate=True)
    assert dag.has_edge("consumer", "producer")

    # Producer's state_dir should be sibling_b's .pivot
    producer_info = consumer.get("producer")
    assert producer_info["state_dir"] == sibling_b / ".pivot"


def test_build_dag_auto_resolves_external_dependencies(
    set_project_root: pathlib.Path,
) -> None:
    """build_dag() should automatically resolve external dependencies.

    Users shouldn't need to call resolve_external_dependencies() explicitly.
    """
    # Same sibling structure as previous test
    model_reports = set_project_root / "model_reports"
    sibling_a = model_reports / "sibling_a"
    sibling_b = model_reports / "sibling_b"
    sibling_a.mkdir(parents=True)
    sibling_b.mkdir(parents=True)

    (sibling_b / "pipeline.py").write_text(
        _make_producer_pipeline_code("sibling_b", "producer", "data/output.txt")
    )
    (sibling_a / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "sibling_a", "consumer", "../sibling_b/data/output.txt", "result.txt"
        )
    )

    # Load consumer pipeline - do NOT call resolve_external_dependencies()
    consumer = discovery.load_pipeline_from_path(sibling_a / "pipeline.py")
    assert consumer is not None

    # build_dag should auto-resolve and succeed
    dag = consumer.build_dag(validate=True)

    # Should have included producer automatically
    assert "producer" in consumer.list_stages()
    assert dag.has_edge("consumer", "producer")


# =============================================================================
# Three-Tier Discovery Tests
# =============================================================================


def test_tier2_output_index_hit(set_project_root: pathlib.Path) -> None:
    """Tier 2: Output index cache should resolve deps without traversal.

    When the output index points to the correct pipeline, resolution should
    succeed via the cached hint without needing a full scan.
    """
    # Producer in a separate directory tree (not discoverable via traverse-up)
    producer_dir = set_project_root / "eval_pipeline" / "difficulty"
    producer_dir.mkdir(parents=True)
    (producer_dir / "pipeline.py").write_text(
        _make_producer_pipeline_code("difficulty", "compute", "data/difficulty/processed/task.yaml")
    )

    # Consumer in a different tree
    consumer_dir = set_project_root / "reports"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "reports",
            "report",
            "../eval_pipeline/difficulty/data/difficulty/processed/task.yaml",
            "report.txt",
        )
    )

    # Pre-populate output index (simulates a previous build_dag)
    index_path = (
        set_project_root
        / ".pivot"
        / "cache"
        / "outputs"
        / "eval_pipeline"
        / "difficulty"
        / "data"
        / "difficulty"
        / "processed"
        / "task.yaml"
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("eval_pipeline/difficulty")

    consumer = discovery.load_pipeline_from_path(consumer_dir / "pipeline.py")
    assert consumer is not None

    dag = consumer.build_dag(validate=True)
    assert "compute" in dag.nodes, "Tier 2 should resolve producer via index"
    assert dag.has_edge("report", "compute")


def test_tier2_stale_index_falls_through_to_tier3(set_project_root: pathlib.Path) -> None:
    """Tier 2 stale: Index points to wrong pipeline, should fall through to tier 3.

    When the index entry is stale (points to a pipeline that no longer produces
    the dep), resolution should fall through to tier 3 (full scan) and find
    the actual producer.
    """
    # Actual producer — outputs ../shared/output.csv relative to new_location/
    # which resolves to <root>/shared/output.csv
    producer_dir = set_project_root / "new_location"
    producer_dir.mkdir(parents=True)
    (producer_dir / "pipeline.py").write_text(
        _make_producer_pipeline_code("new_loc", "produce", "../shared/output.csv")
    )

    # Stale pipeline that no longer produces shared/output.csv
    stale_dir = set_project_root / "old_location"
    stale_dir.mkdir(parents=True)
    (stale_dir / "pipeline.py").write_text(
        _make_producer_pipeline_code("old_loc", "old_produce", "old/output.csv")
    )

    # Consumer — depends on ../shared/output.csv relative to consumer/
    # which also resolves to <root>/shared/output.csv
    consumer_dir = set_project_root / "consumer"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("consumer", "consume", "../shared/output.csv", "result.csv")
    )

    # Stale index pointing to old_location
    index_path = set_project_root / ".pivot" / "cache" / "outputs" / "shared" / "output.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("old_location")

    consumer = discovery.load_pipeline_from_path(consumer_dir / "pipeline.py")
    assert consumer is not None

    dag = consumer.build_dag(validate=True)
    assert "produce" in dag.nodes, "Tier 3 should find actual producer after stale tier 2"
    assert dag.has_edge("consume", "produce")


def test_tier3_cold_start_different_directory_tree(set_project_root: pathlib.Path) -> None:
    """Tier 3 cold start: No index, producer in different tree, found via full scan.

    This is the core motivation for three-tier discovery — code and data live
    in different directory trees, so traverse-up can't find the producer.
    """
    # Producer pipeline — code in eval_pipeline/, outputs in data/
    producer_dir = set_project_root / "eval_pipeline" / "difficulty"
    producer_dir.mkdir(parents=True)
    (producer_dir / "pipeline.py").write_text(
        _make_producer_pipeline_code(
            "difficulty", "compute_difficulty", "data/difficulty/processed/metrics.yaml"
        )
    )

    # Consumer pipeline — code in reports/, depends on data/ output
    consumer_dir = set_project_root / "reports" / "summary"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "summary",
            "summarize",
            "../../eval_pipeline/difficulty/data/difficulty/processed/metrics.yaml",
            "summary.txt",
        )
    )

    # No output index exists (cold start)
    consumer = discovery.load_pipeline_from_path(consumer_dir / "pipeline.py")
    assert consumer is not None

    dag = consumer.build_dag(validate=True)
    assert "compute_difficulty" in dag.nodes, "Tier 3 full scan should find cross-tree producer"
    assert dag.has_edge("summarize", "compute_difficulty")


def test_output_index_written_after_build_dag(set_project_root: pathlib.Path) -> None:
    """Output index should be written after successful build_dag().

    Verifies that .pivot/cache/outputs/ files are created with correct content
    pointing to the producing pipeline's directory.
    """
    # Simple producer
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("root", "produce", "data/output.txt")
    )
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consume", "../data/output.txt", "result.txt")
    )

    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None
    child.build_dag(validate=True)

    # Check output index files
    producer_index = set_project_root / ".pivot" / "cache" / "outputs" / "data" / "output.txt"
    assert producer_index.exists(), "Output index should be written for producer"
    assert producer_index.read_text() == ".", "Producer pipeline dir should be project root"

    consumer_index = set_project_root / ".pivot" / "cache" / "outputs" / "child" / "result.txt"
    assert consumer_index.exists(), "Output index should be written for consumer"
    assert consumer_index.read_text() == "child", "Consumer pipeline dir should be 'child'"


def _make_shared_root_pipeline_code(
    name: str,
    stage_name: str,
    output_path: str,
) -> str:
    """Generate pipeline code that uses root=project.get_project_root() (shared root)."""
    return f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders, project
from pivot.outputs import Out

pipeline = Pipeline("{name}", root=project.get_project_root())

class _Output(TypedDict):
    data: Annotated[Path, Out("{output_path}", loaders.PathOnly())]

def {stage_name}() -> _Output:
    Path("{output_path}").parent.mkdir(parents=True, exist_ok=True)
    Path("{output_path}").write_text("produced")
    return _Output(data=Path("{output_path}"))

pipeline.register({stage_name})
'''


def _make_shared_root_consumer_code(
    name: str,
    stage_name: str,
    dep_path: str,
    output_path: str,
) -> str:
    """Generate consumer pipeline code that uses root=project.get_project_root() (shared root)."""
    return f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders, project
from pivot.outputs import Out, Dep

pipeline = Pipeline("{name}", root=project.get_project_root())

class _Output(TypedDict):
    result: Annotated[Path, Out("{output_path}", loaders.PathOnly())]

def {stage_name}(
    data: Annotated[Path, Dep("{dep_path}", loaders.PathOnly())]
) -> _Output:
    return _Output(result=Path("{output_path}"))

pipeline.register({stage_name})
'''


def test_output_index_shared_root_pipelines(set_project_root: pathlib.Path) -> None:
    """Output index should distinguish pipelines sharing the same project root.

    When sub-pipelines use root=project.get_project_root(), they all share
    the same state_dir. The output index must still record the correct
    pipeline directory (derived from the function's source file, not state_dir).
    """
    # Two sub-pipelines both using root=project.get_project_root()
    sub_a = set_project_root / "sub_a"
    sub_b = set_project_root / "sub_b"
    sub_a.mkdir()
    sub_b.mkdir()

    # Producer in sub_a
    (sub_a / "pipeline.py").write_text(
        _make_shared_root_pipeline_code("sub_a", "produce", "data/output.txt")
    )

    # Consumer in sub_b depends on sub_a's output
    (sub_b / "pipeline.py").write_text(
        _make_shared_root_consumer_code("sub_b", "consume", "data/output.txt", "sub_b/result.txt")
    )

    consumer = discovery.load_pipeline_from_path(sub_b / "pipeline.py")
    assert consumer is not None

    dag = consumer.build_dag(validate=True)
    assert "produce" in dag.nodes
    assert dag.has_edge("consume", "produce")

    # Check output index — producer should point to sub_a, NOT "."
    producer_index = set_project_root / ".pivot" / "cache" / "outputs" / "data" / "output.txt"
    assert producer_index.exists(), "Output index should be written for producer"
    assert producer_index.read_text() == "sub_a", (
        f"Producer pipeline dir should be 'sub_a', got '{producer_index.read_text()}'"
    )

    # Consumer should point to sub_b
    consumer_index = set_project_root / ".pivot" / "cache" / "outputs" / "sub_b" / "result.txt"
    assert consumer_index.exists(), "Output index should be written for consumer"
    assert consumer_index.read_text() == "sub_b", (
        f"Consumer pipeline dir should be 'sub_b', got '{consumer_index.read_text()}'"
    )


def test_watch_mode_safety_fresh_state_per_call(set_project_root: pathlib.Path) -> None:
    """Watch mode: Each build_dag() call uses fresh per-call state.

    Simulates watch-mode reload by building twice — both builds should succeed,
    demonstrating per-call freshness of loaded_pipelines and all_pipeline_paths.
    """
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("root", "produce", "data/output.txt")
    )
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consume", "../data/output.txt", "result.txt")
    )

    # First build
    child1 = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child1 is not None
    dag1 = child1.build_dag(validate=True)
    assert "produce" in dag1.nodes

    # Second build (simulates watch-mode reload with fresh Pipeline)
    child2 = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child2 is not None
    dag2 = child2.build_dag(validate=True)
    assert "produce" in dag2.nodes


def test_dep_exists_on_disk_producer_still_found(set_project_root: pathlib.Path) -> None:
    """Producer should be found even when dep file exists on disk.

    The exists() check was previously skipping resolution when the output file
    was already on disk. With three-tier discovery, we always try to find the
    producer to ensure it's in the DAG for re-execution.
    """
    # Producer in separate tree
    producer_dir = set_project_root / "pipelines" / "etl"
    producer_dir.mkdir(parents=True)
    (producer_dir / "pipeline.py").write_text(
        _make_producer_pipeline_code("etl", "extract", "data/raw/input.csv")
    )

    # Consumer in different tree
    consumer_dir = set_project_root / "pipelines" / "analysis"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "analysis", "analyze", "../etl/data/raw/input.csv", "results.csv"
        )
    )

    # Create the output file on disk (simulates previous successful run)
    output_file = producer_dir / "data" / "raw" / "input.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("existing,data\n1,2\n")

    consumer = discovery.load_pipeline_from_path(consumer_dir / "pipeline.py")
    assert consumer is not None

    dag = consumer.build_dag(validate=True)
    assert "extract" in dag.nodes, "Producer should be found even when output exists on disk"
    assert dag.has_edge("analyze", "extract")


def test_resolve_external_dependencies_when_output_exists_on_disk(
    set_project_root: pathlib.Path,
) -> None:
    """Should resolve sibling dependencies even when output files already exist on disk.

    Regression test: if the producer's output file already exists (from a previous run),
    resolve_external_dependencies() must still find and include the producing stage.
    Without this, re-running a consumer pipeline skips the producer entirely.
    """
    model_reports = set_project_root / "model_reports"
    sibling_a = model_reports / "sibling_a"
    sibling_b = model_reports / "sibling_b"
    sibling_a.mkdir(parents=True)
    sibling_b.mkdir(parents=True)

    # Producer in sibling_b outputs data/output.txt (relative to sibling_b)
    (sibling_b / "pipeline.py").write_text(
        _make_producer_pipeline_code("sibling_b", "producer", "data/output.txt")
    )

    # Consumer in sibling_a depends on sibling_b's output
    (sibling_a / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "sibling_a", "consumer", "../sibling_b/data/output.txt", "result.txt"
        )
    )

    # Simulate a previous run: create the output file on disk
    output_dir = sibling_b / "data"
    output_dir.mkdir(parents=True)
    (output_dir / "output.txt").write_text("previously produced data")

    # Load consumer pipeline and resolve
    consumer = discovery.load_pipeline_from_path(sibling_a / "pipeline.py")
    assert consumer is not None, "Pipeline should load successfully"

    consumer.resolve_external_dependencies()

    # Should still include producer from sibling, even though the file exists
    assert "producer" in consumer.list_stages(), (
        "Producer stage should be included even when its output file exists on disk"
    )
    assert "consumer" in consumer.list_stages(), "Consumer stage should be in pipeline"

    dag = consumer.build_dag(validate=True)
    assert dag.has_edge("consumer", "producer"), "DAG should have edge from consumer to producer"


def test_resolution_skipped_after_first_build_dag(
    set_project_root: pathlib.Path,
) -> None:
    """Second build_dag() call should return cached DAG without rebuilding.

    The _external_deps_resolved flag avoids redundant O(n) stage iteration
    and filesystem traversal on repeated build_dag() calls.
    """
    # Simple parent-child setup
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("parent", "producer", "data/output.txt")
    )
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None, "Child pipeline should load successfully"

    # First build_dag resolves external deps
    dag1 = child.build_dag(validate=True)
    assert "producer" in child.list_stages(), "Producer stage should be resolved in child pipeline"

    # Second call should return the same cached DAG object
    dag2 = child.build_dag(validate=True)
    assert dag2 is dag1, "Second call should return cached DAG"
    assert "producer" in child.list_stages(), "Producer should still be present"


def test_resolution_resets_after_invalidate_dag_cache(
    set_project_root: pathlib.Path,
) -> None:
    """invalidate_dag_cache() should force re-resolution on next build_dag()."""
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("parent", "producer", "data/output.txt")
    )
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None, "Child pipeline should load successfully"

    # First build
    dag1 = child.build_dag(validate=True)
    assert "producer" in child.list_stages(), "Producer stage should be resolved after invalidation"

    # Invalidate and rebuild — should get a fresh DAG
    child.invalidate_dag_cache()
    dag2 = child.build_dag(validate=True)
    assert dag2 is not dag1, "Should rebuild DAG after invalidation"
    assert "producer" in child.list_stages(), "Producer stage should still be resolved"


def test_resolution_resets_after_clear(
    set_project_root: pathlib.Path,
) -> None:
    """clear() should force re-resolution on next build_dag().

    After clearing all stages and re-loading the pipeline, external
    dependency resolution must run again to discover the producer.
    """
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("parent", "producer", "data/output.txt")
    )
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None

    # First build resolves external deps and caches
    child.build_dag(validate=True)
    assert "producer" in child.list_stages()

    # Clear and re-load the pipeline from scratch
    child.clear()
    assert child.list_stages() == [], "Pipeline should have no stages after clear"

    # Re-load (simulates what watch-mode reload does)
    child2 = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child2 is not None

    # The re-loaded pipeline is a fresh instance, so resolution must run
    dag = child2.build_dag(validate=True)
    assert "producer" in child2.list_stages(), "Producer should be re-resolved after reload"
    assert dag.has_edge("consumer", "producer"), "DAG should have edge from consumer to producer"


def test_resolution_resets_after_restore(
    set_project_root: pathlib.Path,
) -> None:
    """restore() should force re-resolution on next build_dag().

    After restoring a snapshot that only contains the consumer (no producer),
    external dependency resolution must run again to re-discover the producer.
    """
    (set_project_root / "pipeline.py").write_text(
        _make_producer_pipeline_code("parent", "producer", "data/output.txt")
    )
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None

    # Snapshot BEFORE resolution (only has consumer)
    pre_resolution_snapshot = child.snapshot()
    assert "producer" not in pre_resolution_snapshot, "Snapshot should not contain producer yet"

    # Build DAG triggers resolution
    child.build_dag(validate=True)
    assert "producer" in child.list_stages()

    # Restore to pre-resolution state (consumer only)
    child.restore(pre_resolution_snapshot)
    assert "producer" not in child.list_stages(), "Producer should be gone after restore"

    # Rebuild should re-resolve and rediscover the producer
    dag = child.build_dag(validate=True)
    assert "producer" in child.list_stages(), "Producer should be re-resolved after restore"
    assert dag.has_edge("consumer", "producer"), "DAG should have edge from consumer to producer"


def test_resolution_resets_after_include(
    set_project_root: pathlib.Path,
) -> None:
    """include() should force re-resolution when new stages introduce unresolved deps.

    When a pipeline has resolved its external dependencies, then a new sub-pipeline
    is included that has its own unresolved deps, resolution must run again to
    discover producers for the newly-included consumer's dependencies.
    """
    # Parent pipeline at project root produces both data/output.txt and extra_data.txt
    parent_code = """
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out

pipeline = Pipeline("parent")

class _Output(TypedDict):
    data: Annotated[Path, Out("data/output.txt", loaders.PathOnly())]

class _Output2(TypedDict):
    extra: Annotated[Path, Out("extra_data.txt", loaders.PathOnly())]

def producer() -> _Output:
    Path("data/output.txt").parent.mkdir(parents=True, exist_ok=True)
    Path("data/output.txt").write_text("produced")
    return _Output(data=Path("data/output.txt"))

def extra_producer() -> _Output2:
    Path("extra_data.txt").write_text("extra")
    return _Output2(extra=Path("extra_data.txt"))

pipeline.register(producer)
pipeline.register(extra_producer)
"""
    (set_project_root / "pipeline.py").write_text(parent_code)

    # Child consumer depends on parent's data/output.txt
    child_dir = set_project_root / "child"
    child_dir.mkdir()
    (child_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code("child", "consumer", "../data/output.txt", "result.txt")
    )

    # Sibling pipeline has a consumer that depends on extra_data.txt from parent
    sibling_dir = set_project_root / "sibling"
    sibling_dir.mkdir()
    (sibling_dir / "pipeline.py").write_text(
        _make_consumer_pipeline_code(
            "sibling", "sibling_consumer", "../extra_data.txt", "sibling_result.txt"
        )
    )

    child = discovery.load_pipeline_from_path(child_dir / "pipeline.py")
    assert child is not None

    # First build resolves consumer -> producer
    child.build_dag(validate=True)
    assert "producer" in child.list_stages()

    # Include the sibling pipeline (which has an unresolved dep on extra_data.txt)
    sibling = discovery.load_pipeline_from_path(sibling_dir / "pipeline.py")
    assert sibling is not None
    child.include(sibling)

    # Rebuild should re-resolve and discover extra_producer for sibling_consumer's dep
    dag = child.build_dag(validate=True)
    assert "sibling_consumer" in dag.nodes, "Included stage should be in DAG"
    assert "extra_producer" in dag.nodes, "Producer for included consumer's dep should be resolved"
    assert "consumer" in dag.nodes, "Original consumer should still be in DAG"
    assert "producer" in dag.nodes, "Original producer should still be in DAG"


def test_resolve_does_not_reload_own_pipeline(set_project_root: pathlib.Path) -> None:
    """Resolution should not re-execute the pipeline's own module.

    When Tier 3 scans all pipeline.py files, it should skip the one that was
    already loaded (via pre-seeded cache). Without this fix, the pipeline's
    module is executed twice — once at discovery and again during resolution.
    """
    # Counter file tracks how many times the pipeline module executes
    counter_file = set_project_root / "load_count.txt"
    counter_file.write_text("0")

    # Producer in separate dir (unresolvable via traverse-up from consumer)
    producer_dir = set_project_root / "producer"
    producer_dir.mkdir()
    (producer_dir / "pipeline.py").write_text(
        _make_producer_pipeline_code("producer_pipe", "produce", "shared/data.txt")
    )

    # Consumer pipeline that increments a counter on each execution
    consumer_dir = set_project_root / "consumer"
    consumer_dir.mkdir()
    (consumer_dir / "pipeline.py").write_text(f'''
from typing import Annotated, TypedDict
from pathlib import Path
from pivot.pipeline import Pipeline
from pivot import loaders
from pivot.outputs import Out, Dep

# Track how many times this module is executed
_counter_path = Path("{counter_file}")
_count = int(_counter_path.read_text()) + 1
_counter_path.write_text(str(_count))

pipeline = Pipeline("consumer_pipe")

class _Output(TypedDict):
    result: Annotated[Path, Out("result.txt", loaders.PathOnly())]

def consume(
    data: Annotated[Path, Dep("../producer/shared/data.txt", loaders.PathOnly())]
) -> _Output:
    return _Output(result=Path("result.txt"))

pipeline.register(consume)
''')

    # Load pipeline (first execution of module)
    consumer = discovery.load_pipeline_from_path(consumer_dir / "pipeline.py")
    assert consumer is not None
    assert counter_file.read_text() == "1", "Module should be loaded exactly once"

    # build_dag triggers resolve_external_dependencies which scans all pipelines
    consumer.build_dag(validate=True)

    # Module should NOT be executed again (pre-seeded cache prevents reload)
    assert counter_file.read_text() == "1", (
        "Module should not be re-executed during resolution (was loaded twice)"
    )
