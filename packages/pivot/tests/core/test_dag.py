from pathlib import Path

import pytest

from pivot import loaders, outputs
from pivot.engine import graph as engine_graph
from pivot.exceptions import CyclicGraphError, DependencyNotFoundError
from pivot.registry import RegistryStageInfo
from pivot.storage.track import PvtData


def _create_stage(name: str, deps: list[str], outs: list[str]) -> RegistryStageInfo:
    """Create a stage dict for testing."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={f"_{i}": d for i, d in enumerate(deps)},
        deps_paths=deps,
        outs=[outputs.Out(path=out, loader=loaders.PathOnly()) for out in outs],
        outs_paths=outs,
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


# --- Basic DAG construction tests ---


def test_build_dag_simple_chain(tmp_path: Path) -> None:
    """Build DAG for simple chain A -> B -> C."""
    # Create files
    (tmp_path / "a.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_c": _create_stage("stage_c", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # Check nodes exist
    assert set(graph.nodes()) == {"stage_a", "stage_b", "stage_c"}

    # Check edges (consumer -> producer)
    assert graph.has_edge("stage_b", "stage_a")
    assert graph.has_edge("stage_c", "stage_b")


def test_build_dag_diamond(tmp_path: Path) -> None:
    """Build DAG for diamond dependency pattern.

         train
        /     \\
    preproc  features
        \\     /
          data
    """
    # Create source file
    (tmp_path / "data.csv").touch()

    stages = {
        "data": _create_stage("data", [], [str(tmp_path / "data.csv")]),
        "preproc": _create_stage(
            "preproc", [str(tmp_path / "data.csv")], [str(tmp_path / "clean.csv")]
        ),
        "features": _create_stage(
            "features", [str(tmp_path / "data.csv")], [str(tmp_path / "features.csv")]
        ),
        "train": _create_stage(
            "train",
            [str(tmp_path / "clean.csv"), str(tmp_path / "features.csv")],
            [str(tmp_path / "model.pkl")],
        ),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # Check all nodes
    assert set(graph.nodes()) == {"data", "preproc", "features", "train"}

    # Check edges
    assert graph.has_edge("preproc", "data")
    assert graph.has_edge("features", "data")
    assert graph.has_edge("train", "preproc")
    assert graph.has_edge("train", "features")


def test_build_dag_independent_stages(tmp_path: Path) -> None:
    """Build DAG with independent stages (no dependencies between them)."""
    (tmp_path / "a.csv").touch()
    (tmp_path / "x.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_x": _create_stage("stage_x", [str(tmp_path / "x.csv")], [str(tmp_path / "y.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # No edges between independent stages
    assert not graph.has_edge("stage_a", "stage_x")
    assert not graph.has_edge("stage_x", "stage_a")


def test_build_dag_empty() -> None:
    """Build DAG with no stages."""
    bipartite = engine_graph.build_graph({})
    graph = engine_graph.get_stage_dag(bipartite)
    assert len(list(graph.nodes())) == 0


# --- Dependency resolution tests ---


def test_file_dependency_resolution(tmp_path: Path) -> None:
    """Find producing stage by output file path."""
    (tmp_path / "data.csv").touch()

    stages = {
        "extract": _create_stage("extract", [], [str(tmp_path / "data.csv")]),
        "transform": _create_stage(
            "transform", [str(tmp_path / "data.csv")], [str(tmp_path / "clean.csv")]
        ),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # transform depends on extract
    assert graph.has_edge("transform", "extract")


def test_dependency_on_existing_file(tmp_path: Path) -> None:
    """Dependency exists on disk but not produced by any stage - no edge created."""
    (tmp_path / "external.csv").touch()

    stages = {
        "process": _create_stage(
            "process", [str(tmp_path / "external.csv")], [str(tmp_path / "output.csv")]
        )
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # No edges (external file is not a stage)
    assert len(list(graph.edges())) == 0


def test_missing_dependency_raises_error(tmp_path: Path) -> None:
    """Dependency not produced AND doesn't exist on disk - raise error."""
    stages = {
        "process": _create_stage(
            "process", [str(tmp_path / "missing.csv")], [str(tmp_path / "output.csv")]
        )
    }

    with pytest.raises(
        DependencyNotFoundError,
        match="depends on.*missing.csv.*not produced by any stage and does not exist on disk",
    ):
        engine_graph.build_graph(stages, validate=True)


def test_missing_dependency_with_validate_false(tmp_path: Path) -> None:
    """With validate=False, missing dependencies don't raise error."""
    stages = {
        "process": _create_stage(
            "process", [str(tmp_path / "missing.csv")], [str(tmp_path / "output.csv")]
        )
    }

    # Should not raise
    bipartite = engine_graph.build_graph(stages, validate=False)
    graph = engine_graph.get_stage_dag(bipartite)
    assert "process" in graph.nodes()


# --- Cycle detection tests ---


def test_circular_dependency_raises_error(tmp_path: Path) -> None:
    """Detect circular dependency A -> B -> A."""
    stages = {
        "stage_a": _create_stage("stage_a", [str(tmp_path / "b.csv")], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
    }

    with pytest.raises(CyclicGraphError, match="Circular dependency detected"):
        engine_graph.build_graph(stages)  # cycles always checked


def test_self_dependency_raises_error(tmp_path: Path) -> None:
    """Detect self-dependency A -> A."""
    stages = {
        "stage_a": _create_stage("stage_a", [str(tmp_path / "a.csv")], [str(tmp_path / "a.csv")])
    }

    with pytest.raises(CyclicGraphError, match="Circular dependency detected"):
        engine_graph.build_graph(stages)  # cycles always checked


def test_transitive_cycle_raises_error(tmp_path: Path) -> None:
    """Detect transitive cycle A -> B -> C -> A."""
    stages = {
        "stage_a": _create_stage("stage_a", [str(tmp_path / "c.csv")], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_c": _create_stage("stage_c", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
    }

    with pytest.raises(CyclicGraphError, match="Circular dependency detected"):
        engine_graph.build_graph(stages)  # cycles always checked


# --- Execution order tests ---


def test_execution_order_simple_chain(tmp_path: Path) -> None:
    """Verify execution order for simple chain."""
    (tmp_path / "a.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_c": _create_stage("stage_c", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)
    order = engine_graph.get_execution_order(graph)

    assert order == ["stage_a", "stage_b", "stage_c"]


def test_execution_order_diamond(tmp_path: Path) -> None:
    """Verify execution order for diamond dependency pattern."""
    (tmp_path / "data.csv").touch()

    stages = {
        "data": _create_stage("data", [], [str(tmp_path / "data.csv")]),
        "preproc": _create_stage(
            "preproc", [str(tmp_path / "data.csv")], [str(tmp_path / "clean.csv")]
        ),
        "features": _create_stage(
            "features", [str(tmp_path / "data.csv")], [str(tmp_path / "features.csv")]
        ),
        "train": _create_stage(
            "train",
            [str(tmp_path / "clean.csv"), str(tmp_path / "features.csv")],
            [str(tmp_path / "model.pkl")],
        ),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)
    order = engine_graph.get_execution_order(graph)

    # data must run first
    assert order[0] == "data"

    # preproc and features can run in any order (both after data)
    assert set(order[1:3]) == {"preproc", "features"}

    # train must run last
    assert order[3] == "train"


def test_execution_order_parallel_branches(tmp_path: Path) -> None:
    """Verify execution order for independent parallel branches."""
    (tmp_path / "a.csv").touch()
    (tmp_path / "x.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
        "stage_x": _create_stage("stage_x", [str(tmp_path / "x.csv")], [str(tmp_path / "y.csv")]),
        "stage_y": _create_stage("stage_y", [str(tmp_path / "y.csv")], [str(tmp_path / "z.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)
    order = engine_graph.get_execution_order(graph)

    # stage_a before stage_b
    assert order.index("stage_a") < order.index("stage_b")

    # stage_x before stage_y
    assert order.index("stage_x") < order.index("stage_y")


def test_execution_order_subset(tmp_path: Path) -> None:
    """Verify execution order for subset of stages."""
    (tmp_path / "a.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_c": _create_stage("stage_c", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # Execute only stage_b and its dependencies
    order = engine_graph.get_execution_order(graph, stages=["stage_b"])

    # Should include stage_a (dependency) and stage_b, but not stage_c
    assert set(order) == {"stage_a", "stage_b"}
    assert order == ["stage_a", "stage_b"]


# --- Subgraph extraction tests ---


def test_get_subgraph_single_stage(tmp_path: Path) -> None:
    """Get subgraph for single stage and its dependencies."""
    (tmp_path / "a.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_c": _create_stage("stage_c", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # Get execution order for just stage_b
    order = engine_graph.get_execution_order(graph, stages=["stage_b"])

    # Should include dependencies
    assert "stage_a" in order
    assert "stage_b" in order
    assert "stage_c" not in order


def test_get_subgraph_single_stage_with_shared_dependency(tmp_path: Path) -> None:
    """Get subgraph for a single stage with shared dependencies."""
    (tmp_path / "data.csv").touch()

    stages = {
        "data": _create_stage("data", [], [str(tmp_path / "data.csv")]),
        "preproc": _create_stage(
            "preproc", [str(tmp_path / "data.csv")], [str(tmp_path / "clean.csv")]
        ),
        "features": _create_stage(
            "features", [str(tmp_path / "data.csv")], [str(tmp_path / "features.csv")]
        ),
        "train": _create_stage(
            "train",
            [str(tmp_path / "clean.csv"), str(tmp_path / "features.csv")],
            [str(tmp_path / "model.pkl")],
        ),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # Get execution order for train (depends on preproc and features)
    order = engine_graph.get_execution_order(graph, stages=["train"])

    # Should include data (dependency), preproc, features, and train
    assert set(order) == {"data", "preproc", "features", "train"}


def test_get_downstream_stages(tmp_path: Path) -> None:
    """Get all stages that depend on given stage."""
    (tmp_path / "a.csv").touch()

    stages = {
        "stage_a": _create_stage("stage_a", [], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
        "stage_c": _create_stage("stage_c", [str(tmp_path / "b.csv")], [str(tmp_path / "c.csv")]),
    }

    bipartite = engine_graph.build_graph(stages)

    # Get all stages downstream of stage_a (uses bipartite graph)
    downstream = engine_graph.get_downstream_stages(bipartite, "stage_a")

    # stage_b and stage_c depend on stage_a (directly or transitively)
    # Note: engine_graph.get_downstream_stages does NOT include the source stage itself
    assert set(downstream) == {"stage_b", "stage_c"}


# --- Edge case tests ---


def test_stage_with_no_deps(tmp_path: Path) -> None:
    """Stage with no dependencies (leaf node)."""
    stages = {"stage_a": _create_stage("stage_a", [], [str(tmp_path / "a.csv")])}

    bipartite = engine_graph.build_graph(stages, validate=False)
    graph = engine_graph.get_stage_dag(bipartite)

    assert "stage_a" in graph.nodes()
    assert len(list(graph.edges())) == 0


def test_stage_with_no_outs(tmp_path: Path) -> None:
    """Stage with no outputs (terminal node)."""
    (tmp_path / "input.csv").touch()

    stages = {"stage_a": _create_stage("stage_a", [str(tmp_path / "input.csv")], [])}

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    assert "stage_a" in graph.nodes()


def test_multiple_stages_same_dependency(tmp_path: Path) -> None:
    """Multiple stages depending on same output (fan-in pattern)."""
    (tmp_path / "data.csv").touch()

    stages = {
        "extract": _create_stage("extract", [], [str(tmp_path / "data.csv")]),
        "analyze": _create_stage(
            "analyze", [str(tmp_path / "data.csv")], [str(tmp_path / "report.txt")]
        ),
        "visualize": _create_stage(
            "visualize", [str(tmp_path / "data.csv")], [str(tmp_path / "chart.png")]
        ),
    }

    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)

    # Both analyze and visualize depend on extract
    assert graph.has_edge("analyze", "extract")
    assert graph.has_edge("visualize", "extract")


# --- Directory dependency tests (BUG-007) ---


def test_directory_depends_on_file_outputs(tmp_path: Path) -> None:
    """Directory dependency waits for stages outputting files into it."""
    dir_path = tmp_path / "data" / "outputs"
    dir_path.mkdir(parents=True)

    stages = {
        "produce_file": _create_stage("produce_file", [], [str(dir_path / "file.csv")]),
        "consume_dir": _create_stage(
            "consume_dir", [str(dir_path)], [str(tmp_path / "result.csv")]
        ),
    }

    bipartite = engine_graph.build_graph(stages, validate=False)
    graph = engine_graph.get_stage_dag(bipartite)

    assert graph.has_edge("consume_dir", "produce_file"), (
        "Should create edge for dir->file dependency"
    )


def test_directory_depends_on_multiple_file_outputs(tmp_path: Path) -> None:
    """Directory dependency waits for ALL stages outputting files into it."""
    dir_path = tmp_path / "data" / "outputs"
    dir_path.mkdir(parents=True)

    stages = {
        "produce_a": _create_stage("produce_a", [], [str(dir_path / "a.csv")]),
        "produce_b": _create_stage("produce_b", [], [str(dir_path / "b.csv")]),
        "consume_dir": _create_stage(
            "consume_dir", [str(dir_path)], [str(tmp_path / "result.csv")]
        ),
    }

    bipartite = engine_graph.build_graph(stages, validate=False)
    graph = engine_graph.get_stage_dag(bipartite)

    assert graph.has_edge("consume_dir", "produce_a"), "Should depend on first producer"
    assert graph.has_edge("consume_dir", "produce_b"), "Should depend on second producer"


def test_nested_directory_dependency(tmp_path: Path) -> None:
    """Directory dependency detects files in nested subdirectories."""
    dir_path = tmp_path / "data"
    nested = dir_path / "sub" / "nested"
    nested.mkdir(parents=True)

    stages = {
        "produce_nested": _create_stage("produce_nested", [], [str(nested / "file.csv")]),
        "consume_parent": _create_stage(
            "consume_parent", [str(dir_path)], [str(tmp_path / "result.csv")]
        ),
    }

    bipartite = engine_graph.build_graph(stages, validate=False)
    graph = engine_graph.get_stage_dag(bipartite)

    assert graph.has_edge("consume_parent", "produce_nested"), "Should detect nested file outputs"


# --- Tracked file tests ---


def test_tracked_file_recognized_as_valid_source(tmp_path: Path) -> None:
    """Tracked files are valid dependency sources even if not on disk."""
    # Dependency doesn't exist on disk, but it's in tracked_files
    tracked_path = str(tmp_path / "tracked_input.csv")

    stages = {"process": _create_stage("process", [tracked_path], [str(tmp_path / "output.csv")])}

    # Without tracked_files, this would raise DependencyNotFoundError
    tracked_files: dict[str, PvtData] = {
        tracked_path: PvtData(path="tracked_input.csv", hash="abc123", size=100),
    }

    # Should not raise - tracked file is recognized as valid source
    bipartite = engine_graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    graph = engine_graph.get_stage_dag(bipartite)
    assert "process" in graph.nodes()


def test_tracked_file_inside_directory_recognized(tmp_path: Path) -> None:
    """File inside a tracked directory is recognized as valid source."""
    tracked_dir = str(tmp_path / "tracked_data")
    file_inside = str(tmp_path / "tracked_data" / "file.csv")

    stages = {"process": _create_stage("process", [file_inside], [str(tmp_path / "output.csv")])}

    # Directory is tracked
    tracked_files: dict[str, PvtData] = {
        tracked_dir: PvtData(
            path="tracked_data",
            hash="def456",
            size=500,
            num_files=3,
            manifest=[{"relpath": "file.csv", "hash": "ghi789", "size": 100, "isexec": False}],
        ),
    }

    # Should not raise - file inside tracked directory is valid
    bipartite = engine_graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    graph = engine_graph.get_stage_dag(bipartite)
    assert "process" in graph.nodes()


def test_dependency_on_tracked_directory(tmp_path: Path) -> None:
    """Directory dependency on tracked directory is valid."""
    tracked_dir = str(tmp_path / "tracked_data")

    stages = {"process": _create_stage("process", [tracked_dir], [str(tmp_path / "output.csv")])}

    tracked_files: dict[str, PvtData] = {
        tracked_dir: PvtData(path="tracked_data", hash="abc123", size=500, num_files=2),
    }

    bipartite = engine_graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    graph = engine_graph.get_stage_dag(bipartite)
    assert "process" in graph.nodes()


def test_untracked_missing_dependency_still_raises(tmp_path: Path) -> None:
    """Missing dependency not in tracked_files still raises error."""
    missing_path = str(tmp_path / "missing.csv")

    stages = {"process": _create_stage("process", [missing_path], [str(tmp_path / "output.csv")])}

    # Some other file is tracked, but not the one we depend on
    tracked_files: dict[str, PvtData] = {
        str(tmp_path / "other.csv"): PvtData(path="other.csv", hash="xyz", size=50),
    }

    with pytest.raises(DependencyNotFoundError):
        engine_graph.build_graph(stages, validate=True, tracked_files=tracked_files)
