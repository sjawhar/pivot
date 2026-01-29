"""Tests for the bipartite artifact-stage graph."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest

from pivot import dag as legacy_dag
from pivot import loaders, outputs
from pivot.engine import graph, types
from pivot.registry import REGISTRY, RegistryStageInfo


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
        mutex=list[Any](),
        variant=None,
        signature=None,
        fingerprint=dict[str, Any](),
        dep_specs=dict[str, Any](),
        out_specs=dict[str, Any](),
        params_arg_name=None,
    )


# --- Node naming tests ---


def test_artifact_node_creates_prefixed_string() -> None:
    """artifact_node creates 'artifact:' prefixed string."""
    node = graph.artifact_node(Path("/data/input.csv"))
    assert node == "artifact:/data/input.csv"


def test_stage_node_creates_prefixed_string() -> None:
    """stage_node creates 'stage:' prefixed string."""
    node = graph.stage_node("train")
    assert node == "stage:train"


def test_parse_node_extracts_type_and_value() -> None:
    """parse_node extracts NodeType and value from prefixed string."""
    node_type, value = graph.parse_node("artifact:/data/input.csv")
    assert node_type == types.NodeType.ARTIFACT
    assert value == "/data/input.csv"

    node_type, value = graph.parse_node("stage:train")
    assert node_type == types.NodeType.STAGE
    assert value == "train"


def test_parse_node_handles_colons_in_path() -> None:
    """parse_node handles paths with colons (Windows, URLs)."""
    node_type, value = graph.parse_node("artifact:C:/data/input.csv")
    assert node_type == types.NodeType.ARTIFACT
    assert value == "C:/data/input.csv"


# --- Graph building tests ---


@pytest.fixture
def clean_registry() -> Generator[None]:
    """Ensure registry is clean before and after test."""
    REGISTRY.clear()
    yield
    REGISTRY.clear()


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_simple_chain(tmp_path: Path) -> None:
    """Build bipartite graph for simple chain: input -> A -> intermediate -> B -> output."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    # Register stages directly (bypass REGISTRY for isolated test)
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }

    g = graph.build_graph(stages)

    # Check we have both stage and artifact nodes
    stage_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.STAGE]
    artifact_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.ARTIFACT]

    assert len(stage_nodes) == 2
    assert len(artifact_nodes) == 3  # input, intermediate, output

    # Check edges: artifact -> stage (consumed by) and stage -> artifact (produces)
    assert g.has_edge(graph.artifact_node(input_file), graph.stage_node("stage_a"))
    assert g.has_edge(graph.stage_node("stage_a"), graph.artifact_node(intermediate))
    assert g.has_edge(graph.artifact_node(intermediate), graph.stage_node("stage_b"))
    assert g.has_edge(graph.stage_node("stage_b"), graph.artifact_node(output_file))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_diamond(tmp_path: Path) -> None:
    """Build bipartite graph for diamond pattern.

    input -> preprocess -> clean
          -> features -> feats
    clean + feats -> train -> model
    """
    input_file = tmp_path / "input.csv"
    clean = tmp_path / "clean.csv"
    feats = tmp_path / "feats.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(clean)]),
        "features": _create_stage("features", [str(input_file)], [str(feats)]),
        "train": _create_stage("train", [str(clean), str(feats)], [str(model)]),
    }

    g = graph.build_graph(stages)

    # Both preprocess and features consume input
    assert g.has_edge(graph.artifact_node(input_file), graph.stage_node("preprocess"))
    assert g.has_edge(graph.artifact_node(input_file), graph.stage_node("features"))

    # Train consumes both clean and feats
    assert g.has_edge(graph.artifact_node(clean), graph.stage_node("train"))
    assert g.has_edge(graph.artifact_node(feats), graph.stage_node("train"))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_empty() -> None:
    """Build graph with no stages returns empty graph."""
    g = graph.build_graph({})
    assert len(g.nodes()) == 0
    assert len(g.edges()) == 0


# --- Query function tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_dependent_stages(tmp_path: Path) -> None:
    """get_consumers returns stages that depend on an artifact."""
    input_file = tmp_path / "input.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(out_a)]),
        "stage_b": _create_stage("stage_b", [str(input_file)], [str(out_b)]),
    }

    g = graph.build_graph(stages)
    consumers = graph.get_consumers(g, input_file)

    assert set(consumers) == {"stage_a", "stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_consumers returns empty list for unknown path."""
    g = graph.build_graph({})
    consumers = graph.get_consumers(g, tmp_path / "unknown.csv")
    assert consumers == []


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_producing_stage(tmp_path: Path) -> None:
    """get_producer returns the stage that produces an artifact."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, output_file)

    assert producer == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_none_for_input_artifact(tmp_path: Path) -> None:
    """get_producer returns None for artifacts that are inputs (not produced by any stage)."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, input_file)

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_watch_paths_returns_all_artifacts(tmp_path: Path) -> None:
    """get_watch_paths returns all artifact paths."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    paths = graph.get_watch_paths(g)

    assert set(paths) == {input_file, intermediate, output_file}


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages(tmp_path: Path) -> None:
    """get_downstream_stages returns all transitively downstream stages."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert set(downstream) == {"stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_leaf(tmp_path: Path) -> None:
    """get_downstream_stages returns empty for leaf stage."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert downstream == []


# --- Incremental update tests ---


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_dep(tmp_path: Path) -> None:
    """update_stage adds new dependency edges."""
    input_a = tmp_path / "a.csv"
    input_b = tmp_path / "b.csv"
    output_file = tmp_path / "output.csv"
    input_a.touch()
    input_b.touch()

    # Initial: stage_a depends on input_a
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_a)], [str(output_file)]),
    }
    g = graph.build_graph(stages)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # Update: stage_a now also depends on input_b
    new_info = _create_stage("stage_a", [str(input_a), str(input_b)], [str(output_file)])
    graph.update_stage(g, "stage_a", new_info)

    assert set(graph.get_consumers(g, input_a)) == {"stage_a"}
    assert set(graph.get_consumers(g, input_b)) == {"stage_a"}


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_dep(tmp_path: Path) -> None:
    """update_stage removes old dependency edges and orphaned artifacts."""
    input_a = tmp_path / "a.csv"
    input_b = tmp_path / "b.csv"
    output_file = tmp_path / "output.csv"
    input_a.touch()
    input_b.touch()

    # Initial: stage_a depends on both inputs
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_a), str(input_b)], [str(output_file)]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only depends on input_a
    new_info = _create_stage("stage_a", [str(input_a)], [str(output_file)])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # input_b should be removed from graph (orphaned)
    assert graph.artifact_node(input_b) not in g


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_preserves_shared_artifacts(tmp_path: Path) -> None:
    """update_stage doesn't remove artifacts used by other stages."""
    shared_input = tmp_path / "shared.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    shared_input.touch()

    # Both stages depend on shared_input
    stages = {
        "stage_a": _create_stage("stage_a", [str(shared_input)], [str(out_a)]),
        "stage_b": _create_stage("stage_b", [str(shared_input)], [str(out_b)]),
    }
    g = graph.build_graph(stages)

    # Update stage_a to have no deps - shared_input should remain (used by stage_b)
    new_info = _create_stage("stage_a", [], [str(out_a)])
    graph.update_stage(g, "stage_a", new_info)

    # shared_input still in graph
    assert graph.artifact_node(shared_input) in g
    assert graph.get_consumers(g, shared_input) == ["stage_b"]


# --- Integration test ---


@pytest.mark.usefixtures("clean_registry")
def test_graph_consistent_with_legacy_dag(tmp_path: Path) -> None:
    """Bipartite graph encodes same relationships as legacy stage-only DAG."""
    # Build a pipeline with multiple stages
    input_file = tmp_path / "input.csv"
    clean = tmp_path / "clean.csv"
    feats = tmp_path / "feats.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(clean)]),
        "features": _create_stage("features", [str(input_file)], [str(feats)]),
        "train": _create_stage("train", [str(clean), str(feats)], [str(model)]),
    }

    # Build both graphs
    bipartite = graph.build_graph(stages)
    legacy = legacy_dag.build_dag(stages)

    # Verify same stage relationships
    # Legacy DAG has edge (consumer -> producer), meaning train -> preprocess, train -> features
    for stage_name in stages:
        legacy_deps = set(legacy.successors(stage_name))  # Stages this depends on
        bipartite_deps = set[str]()
        for dep_path in stages[stage_name]["deps_paths"]:
            producer = graph.get_producer(bipartite, Path(dep_path))
            if producer:
                bipartite_deps.add(producer)

        assert legacy_deps == bipartite_deps, f"Mismatch for {stage_name}"
