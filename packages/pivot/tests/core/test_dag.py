# pyright: reportMissingImports=false
from pathlib import Path
from unittest.mock import Mock

import pytest

from pivot import loaders
from pivot.engine import graph as engine_graph
from pivot.exceptions import CyclicGraphError, DependencyNotFoundError
from pivot.registry import RegistryStageInfo
from pivot.storage.store import Store
from pivot.types import ArtifactIdentity, ArtifactRef, ArtifactTag


def _artifact_ref(identity: ArtifactIdentity) -> ArtifactRef:
    return ArtifactRef(
        identity=identity,
        format=loaders.PathOnly(),
        python_type=str,
        tag=ArtifactTag.DATA,
    )


def _create_stage(name: str, deps: list[str], outs: list[str]) -> RegistryStageInfo:
    """Create a stage dict for testing.

    Converts string paths to ArtifactIdentity objects using the path as producer.
    This matches the behavior of the graph building logic which uses paths for matching.
    """
    # Use the full path as the producer (key=None) to match path-based artifact matching
    dep_identities = [ArtifactIdentity(path, None) for path in deps]
    out_identities = [ArtifactIdentity(path, None) for path in outs]

    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={f"_{i}": _artifact_ref(dep) for i, dep in enumerate(dep_identities)},
        outs=[_artifact_ref(out) for out in out_identities],
        params=None,
        mutex=list[str](),
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
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
    """Dependency refers to known stage but missing output key raises error."""
    producer_out = ArtifactIdentity("producer", "output.csv")
    missing_dep = ArtifactIdentity("producer", "missing.csv")

    stages = {
        "producer": RegistryStageInfo(
            func=lambda: None,
            name="producer",
            deps={},
            outs=[_artifact_ref(producer_out)],
            params=None,
            mutex=list[str](),
            variant=None,
            signature=None,
            fingerprint=dict[str, str](),
            params_arg_name=None,
            state_dir=None,
            collection_params={},
            no_fingerprint=False,
        ),
        "consumer": RegistryStageInfo(
            func=lambda: None,
            name="consumer",
            deps={"_0": _artifact_ref(missing_dep)},
            outs=[_artifact_ref(ArtifactIdentity("consumer", "output.csv"))],
            params=None,
            mutex=list[str](),
            variant=None,
            signature=None,
            fingerprint=dict[str, str](),
            params_arg_name=None,
            state_dir=None,
            collection_params={},
            no_fingerprint=False,
        ),
    }

    with pytest.raises(
        DependencyNotFoundError,
        match="depends on.*producer:missing.csv.*not produced by any stage and does not exist on disk",
    ):
        engine_graph.validate_dependency_sources(stages)


def test_build_graph_does_not_validate_deps(tmp_path: Path) -> None:
    """build_graph doesn't validate dependency existence."""
    stages = {
        "process": _create_stage(
            "process", [str(tmp_path / "missing.csv")], [str(tmp_path / "output.csv")]
        )
    }

    # Should not raise
    bipartite = engine_graph.build_graph(stages)
    graph = engine_graph.get_stage_dag(bipartite)
    assert "process" in graph.nodes()


def test_validate_dependency_sources_uses_store() -> None:
    external_dep = _artifact_ref(ArtifactIdentity("input.sql", None))
    stages = {
        "consumer": RegistryStageInfo(
            func=lambda: None,
            name="consumer",
            deps={"_0": external_dep},
            outs=[_artifact_ref(ArtifactIdentity("output.csv", None))],
            params=None,
            mutex=list[str](),
            variant=None,
            signature=None,
            fingerprint=dict[str, str](),
            params_arg_name=None,
            state_dir=None,
            collection_params={},
            no_fingerprint=False,
        ),
    }

    store = Mock(spec=Store)
    store.exists.return_value = True
    engine_graph.validate_dependency_sources(stages, store=store)
    store.exists.assert_called_once_with(external_dep)

    store = Mock(spec=Store)
    store.exists.return_value = False
    with pytest.raises(DependencyNotFoundError):
        engine_graph.validate_dependency_sources(stages, store=store)


def test_validate_dependency_sources_no_store_skips_external() -> None:
    external_dep = _artifact_ref(ArtifactIdentity("external", "input.csv"))
    stages = {
        "consumer": RegistryStageInfo(
            func=lambda: None,
            name="consumer",
            deps={"_0": external_dep},
            outs=[_artifact_ref(ArtifactIdentity("consumer", "output.csv"))],
            params=None,
            mutex=list[str](),
            variant=None,
            signature=None,
            fingerprint=dict[str, str](),
            params_arg_name=None,
            state_dir=None,
            collection_params={},
            no_fingerprint=False,
        ),
    }

    engine_graph.validate_dependency_sources(stages, store=None)

    stages = {
        "producer": RegistryStageInfo(
            func=lambda: None,
            name="producer",
            deps={},
            outs=[_artifact_ref(ArtifactIdentity("producer", "output.csv"))],
            params=None,
            mutex=list[str](),
            variant=None,
            signature=None,
            fingerprint=dict[str, str](),
            params_arg_name=None,
            state_dir=None,
            collection_params={},
            no_fingerprint=False,
        ),
        "consumer": RegistryStageInfo(
            func=lambda: None,
            name="consumer",
            deps={"_0": _artifact_ref(ArtifactIdentity("producer", "missing.csv"))},
            outs=[_artifact_ref(ArtifactIdentity("consumer", "output.csv"))],
            params=None,
            mutex=list[str](),
            variant=None,
            signature=None,
            fingerprint=dict[str, str](),
            params_arg_name=None,
            state_dir=None,
            collection_params={},
            no_fingerprint=False,
        ),
    }

    with pytest.raises(DependencyNotFoundError):
        engine_graph.validate_dependency_sources(stages, store=None)


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

    bipartite = engine_graph.build_graph(stages)
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
