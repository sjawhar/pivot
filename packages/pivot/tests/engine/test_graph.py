"""Tests for the bipartite artifact-stage graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, loaders
from pivot.engine import graph, types
from pivot.registry import RegistryStageInfo
from pivot.types import ArtifactIdentity, ArtifactRef, ArtifactTag

if TYPE_CHECKING:
    from pathlib import Path


def _artifact_ref(identity: ArtifactIdentity) -> ArtifactRef:
    return ArtifactRef(
        identity=identity,
        format=loaders.PathOnly(),
        python_type=str,
        tag=ArtifactTag.DATA,
    )


def _create_stage(
    name: str,
    deps: list[ArtifactIdentity],
    outs: list[ArtifactIdentity],
) -> RegistryStageInfo:
    """Create a stage dict for testing."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={f"_{i}": _artifact_ref(dep) for i, dep in enumerate(deps)},
        outs=[_artifact_ref(out) for out in outs],
        params=None,
        mutex=list[str](),
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        params_arg_name=None,
        state_dir=None,
    )


# --- Node naming tests ---


def test_artifact_node_creates_prefixed_string() -> None:
    """artifact_node creates 'artifact:' prefixed string."""
    node = graph.artifact_node(ArtifactIdentity("source", None))
    assert node == "artifact:source"


def test_stage_node_creates_prefixed_string() -> None:
    """stage_node creates 'stage:' prefixed string."""
    node = graph.stage_node("train")
    assert node == "stage:train"


def test_parse_node_extracts_type_and_value() -> None:
    """parse_node extracts NodeType and value from prefixed string."""
    node_type, value = graph.parse_node("artifact:source:input.csv")
    assert node_type == types.NodeType.ARTIFACT
    assert value == "source:input.csv"

    node_type, value = graph.parse_node("stage:train")
    assert node_type == types.NodeType.STAGE
    assert value == "train"


def test_parse_node_handles_colons_in_path() -> None:
    """parse_artifact_identity parses producer-only strings."""
    identity = graph.parse_artifact_identity("external")
    assert identity == ArtifactIdentity("external", None)

    identity = graph.parse_artifact_identity("producer:key")
    assert identity == ArtifactIdentity("producer", "key")


# --- Graph building tests ---


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_simple_chain(tmp_path: Path) -> None:
    """Build bipartite graph for simple chain: input -> A -> intermediate -> B -> output."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    # Create stages dict directly for isolated graph test
    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    g = graph.build_graph(stages)

    # Check we have both stage and artifact nodes
    stage_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.STAGE]
    artifact_nodes = [n for n in g.nodes() if g.nodes[n]["type"] == types.NodeType.ARTIFACT]

    assert len(stage_nodes) == 2
    assert len(artifact_nodes) == 3  # input, intermediate, output

    # Check edges: artifact -> stage (consumed by) and stage -> artifact (produces)
    assert g.has_edge(graph.artifact_node(input_identity), graph.stage_node("stage_a"))
    assert g.has_edge(graph.stage_node("stage_a"), graph.artifact_node(intermediate))
    assert g.has_edge(graph.artifact_node(intermediate), graph.stage_node("stage_b"))
    assert g.has_edge(graph.stage_node("stage_b"), graph.artifact_node(output_identity))


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_diamond(tmp_path: Path) -> None:
    """Build bipartite graph for diamond pattern.

    input -> preprocess -> clean
          -> features -> feats
    clean + feats -> train -> model
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    clean = ArtifactIdentity("preprocess", "clean.csv")
    feats = ArtifactIdentity("features", "feats.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [clean]),
        "features": _create_stage("features", [input_identity], [feats]),
        "train": _create_stage("train", [clean, feats], [model]),
    }

    g = graph.build_graph(stages)

    # Both preprocess and features consume input
    assert g.has_edge(graph.artifact_node(input_identity), graph.stage_node("preprocess"))
    assert g.has_edge(graph.artifact_node(input_identity), graph.stage_node("features"))

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
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_b", "b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a]),
        "stage_b": _create_stage("stage_b", [input_identity], [out_b]),
    }

    g = graph.build_graph(stages)
    consumers = graph.get_consumers(g, input_identity)

    assert set(consumers) == {"stage_a", "stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_consumers returns empty list for unknown path."""
    g = graph.build_graph({})
    consumers = graph.get_consumers(g, ArtifactIdentity("unknown", "artifact"))
    assert consumers == []


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_producing_stage(tmp_path: Path) -> None:
    """get_producer returns the stage that produces an artifact."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, output_identity)

    assert producer == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_none_for_input_artifact(tmp_path: Path) -> None:
    """get_producer returns None for artifacts that are inputs (not produced by any stage)."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, input_identity)

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_watch_paths_returns_all_artifacts(tmp_path: Path) -> None:
    """get_watch_paths returns empty list with identity-based artifacts."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    g = graph.build_graph(stages)
    paths = graph.get_watch_paths(g)

    assert paths == []


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages(tmp_path: Path) -> None:
    """get_downstream_stages returns all transitively downstream stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert set(downstream) == {"stage_b"}


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_leaf(tmp_path: Path) -> None:
    """get_downstream_stages returns empty for leaf stage."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "stage_a")

    assert downstream == []


# --- Incremental update tests ---


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_dep(tmp_path: Path) -> None:
    """update_stage adds new dependency edges."""
    input_a = ArtifactIdentity("source", "a.csv")
    input_b = ArtifactIdentity("source", "b.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    # Initial: stage_a depends on input_a
    stages = {
        "stage_a": _create_stage("stage_a", [input_a], [output_identity]),
    }
    g = graph.build_graph(stages)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # Update: stage_a now also depends on input_b
    new_info = _create_stage("stage_a", [input_a, input_b], [output_identity])
    graph.update_stage(g, "stage_a", new_info)

    assert set(graph.get_consumers(g, input_a)) == {"stage_a"}
    assert set(graph.get_consumers(g, input_b)) == {"stage_a"}


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_dep(tmp_path: Path) -> None:
    """update_stage removes old dependency edges and orphaned artifacts."""
    input_a = ArtifactIdentity("source", "a.csv")
    input_b = ArtifactIdentity("source", "b.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    # Initial: stage_a depends on both inputs
    stages = {
        "stage_a": _create_stage("stage_a", [input_a, input_b], [output_identity]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only depends on input_a
    new_info = _create_stage("stage_a", [input_a], [output_identity])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_consumers(g, input_a) == ["stage_a"]
    assert graph.get_consumers(g, input_b) == []

    # input_b should be removed from graph (orphaned)
    assert graph.artifact_node(input_b) not in g


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_preserves_shared_artifacts(tmp_path: Path) -> None:
    """update_stage doesn't remove artifacts used by other stages."""
    shared_input = ArtifactIdentity("source", "shared.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_b", "b.csv")

    # Both stages depend on shared_input
    stages = {
        "stage_a": _create_stage("stage_a", [shared_input], [out_a]),
        "stage_b": _create_stage("stage_b", [shared_input], [out_b]),
    }
    g = graph.build_graph(stages)

    # Update stage_a to have no deps - shared_input should remain (used by stage_b)
    new_info = _create_stage("stage_a", [], [out_a])
    graph.update_stage(g, "stage_a", new_info)

    # shared_input still in graph
    assert graph.artifact_node(shared_input) in g
    assert graph.get_consumers(g, shared_input) == ["stage_b"]


# --- get_stage_dag tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_stage_dag_extracts_stage_only_graph(tmp_path: Path) -> None:
    """get_stage_dag returns stage-only DAG from bipartite graph."""
    input_identity = ArtifactIdentity("source", "input.csv")
    cleaned = ArtifactIdentity("preprocess", "cleaned.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [cleaned]),
        "train": _create_stage("train", [cleaned], [model]),
    }
    bipartite = graph.build_graph(stages)

    # Extract stage-only DAG
    stage_dag = graph.get_stage_dag(bipartite)

    # Should have stage nodes (not artifact:... or stage:... prefixed)
    assert "preprocess" in stage_dag
    assert "train" in stage_dag

    # Should NOT have artifact nodes or prefixed stage nodes
    for node in stage_dag.nodes():
        assert not node.startswith("artifact:")
        assert not node.startswith("stage:")

    # Edge direction: consumer -> producer (for DFS postorder execution)
    # train depends on preprocess, so edge goes train -> preprocess
    assert stage_dag.has_edge("train", "preprocess")


# --- get_artifact_consumers tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_artifact_consumers_returns_direct_and_downstream(tmp_path: Path) -> None:
    """get_artifact_consumers returns stages that depend on artifact."""
    # Build graph: input.csv -> preprocess -> cleaned.csv -> train -> model.pkl
    input_identity = ArtifactIdentity("source", "input.csv")
    cleaned = ArtifactIdentity("preprocess", "cleaned.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [cleaned]),
        "train": _create_stage("train", [cleaned], [model]),
    }
    g = graph.build_graph(stages)

    # Input change should affect both preprocess AND train
    consumers = graph.get_artifact_consumers(g, input_identity, include_downstream=True)

    assert "preprocess" in consumers
    assert "train" in consumers  # Downstream of preprocess

    # Without downstream, only direct consumers
    direct = graph.get_artifact_consumers(g, input_identity, include_downstream=False)

    assert "preprocess" in direct
    assert "train" not in direct


@pytest.mark.usefixtures("clean_registry")
def test_get_artifact_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_artifact_consumers returns empty list for unknown artifact."""
    g = graph.build_graph({})
    consumers = graph.get_artifact_consumers(g, ArtifactIdentity("unknown", "artifact"))
    assert consumers == []


# --- Validation tests ---


def test_build_graph_raises_on_cycle(tmp_path: Path) -> None:
    """build_graph raises CyclicGraphError when graph has cycles."""
    file_a = ArtifactIdentity("stage_a", "a.csv")
    file_b = ArtifactIdentity("stage_b", "b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [file_b], [file_a]),
        "stage_b": _create_stage("stage_b", [file_a], [file_b]),
    }

    with pytest.raises(exceptions.CyclicGraphError, match="Circular dependency"):
        graph.build_graph(stages)  # Cycles always checked


def test_build_graph_raises_on_missing_dependency(tmp_path: Path) -> None:
    """build_graph raises DependencyNotFoundError when validate=True."""
    missing_dep = ArtifactIdentity("stage_b", "missing.csv")

    stages = {
        "stage_a": _create_stage(
            "stage_a", [missing_dep], [ArtifactIdentity("stage_a", "out.csv")]
        ),
        "stage_b": _create_stage("stage_b", [], [ArtifactIdentity("stage_b", "other.csv")]),
    }

    with pytest.raises(exceptions.DependencyNotFoundError):
        graph.build_graph(stages, validate=True)


def test_build_graph_allows_missing_when_validate_false(tmp_path: Path) -> None:
    """build_graph allows missing deps when validate=False."""
    output_identity = ArtifactIdentity("stage_a", "output.csv")
    missing_dep = ArtifactIdentity("external", "missing.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [missing_dep], [output_identity]),
    }

    # Should not raise
    g = graph.build_graph(stages, validate=False)
    assert "stage:stage_a" in g


# --- get_upstream_stages tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_returns_producing_stages(tmp_path: Path) -> None:
    """get_upstream_stages returns stages that produce inputs for a stage."""
    input_identity = ArtifactIdentity("source", "input.csv")
    cleaned = ArtifactIdentity("preprocess", "cleaned.csv")
    features = ArtifactIdentity("extract", "features.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [cleaned]),
        "extract": _create_stage("extract", [input_identity], [features]),
        "train": _create_stage("train", [cleaned, features], [model]),
    }

    g = graph.build_graph(stages)
    upstream = graph.get_upstream_stages(g, "train")

    # train depends on outputs from preprocess and extract
    assert set(upstream) == {"preprocess", "extract"}


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_empty_for_root_stage(tmp_path: Path) -> None:
    """get_upstream_stages returns empty list for stage with no upstream dependencies."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    upstream = graph.get_upstream_stages(g, "stage_a")

    # stage_a has no upstream stages (only external input)
    assert upstream == []


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_empty_for_unknown_stage(tmp_path: Path) -> None:
    """get_upstream_stages returns empty list for unknown stage."""
    g = graph.build_graph({})
    upstream = graph.get_upstream_stages(g, "unknown_stage")
    assert upstream == []


# --- get_execution_order with single_stage tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_execution_order_single_stage_mode(tmp_path: Path) -> None:
    """get_execution_order with single_stage=True returns only requested stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Single stage mode - should return only stage_b, NOT its dependency stage_a
    order = graph.get_execution_order(stage_dag, stages=["stage_b"], single_stage=True)

    assert order == ["stage_b"]
    assert "stage_a" not in order


@pytest.mark.usefixtures("clean_registry")
def test_get_execution_order_single_stage_preserves_order(tmp_path: Path) -> None:
    """get_execution_order with single_stage=True preserves input order."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_b", "b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a]),
        "stage_b": _create_stage("stage_b", [input_identity], [out_b]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Request in specific order - should be preserved
    order = graph.get_execution_order(stage_dag, stages=["stage_b", "stage_a"], single_stage=True)

    assert order == ["stage_b", "stage_a"]


# --- Additional edge case tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_producer_returns_none_for_unknown_path(tmp_path: Path) -> None:
    """get_producer returns None for completely unknown artifact path."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, ArtifactIdentity("unknown", "artifact"))

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_unknown_stage(tmp_path: Path) -> None:
    """get_downstream_stages returns empty list for unknown stage."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "unknown_stage")

    assert downstream == []


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_out(tmp_path: Path) -> None:
    """update_stage adds new output edges."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_a", "b.csv")

    # Initial: stage_a produces only out_a
    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a]),
    }
    g = graph.build_graph(stages)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) is None

    # Update: stage_a now also produces out_b
    new_info = _create_stage("stage_a", [input_identity], [out_a, out_b])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_out(tmp_path: Path) -> None:
    """update_stage removes old output edges and orphaned artifacts."""
    input_identity = ArtifactIdentity("source", "input.csv")
    out_a = ArtifactIdentity("stage_a", "a.csv")
    out_b = ArtifactIdentity("stage_a", "b.csv")

    # Initial: stage_a produces both outputs
    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [out_a, out_b]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only produces out_a
    new_info = _create_stage("stage_a", [input_identity], [out_a])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) is None

    # out_b should be removed from graph (orphaned)
    assert graph.artifact_node(out_b) not in g


# --- Error path tests ---


@pytest.mark.usefixtures("clean_registry")
def test_cycle_detection_error_message_format(tmp_path: Path) -> None:
    """Cycle error message contains affected stage names."""
    stages = {
        "stage_a": _create_stage(
            "stage_a",
            [ArtifactIdentity("stage_b", "b.csv")],
            [ArtifactIdentity("stage_a", "a.csv")],
        ),
        "stage_b": _create_stage(
            "stage_b",
            [ArtifactIdentity("stage_a", "a.csv")],
            [ArtifactIdentity("stage_b", "b.csv")],
        ),
    }

    try:
        graph.build_graph(stages)
        pytest.fail("Should have raised CyclicGraphError")
    except exceptions.CyclicGraphError as e:
        # Error message should contain stage names
        assert "stage_a" in str(e) or "stage_b" in str(e)
        assert "Circular dependency" in str(e)


def test_get_execution_order_unknown_stage_raises_error(tmp_path: Path) -> None:
    """get_execution_order raises StageNotFoundError for unknown stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Should raise StageNotFoundError, not raw NetworkXError
    with pytest.raises(exceptions.StageNotFoundError, match="unknown_stage"):
        graph.get_execution_order(stage_dag, stages=["unknown_stage"])


def test_get_execution_order_mixed_known_unknown_stages(tmp_path: Path) -> None:
    """get_execution_order raises StageNotFoundError with all unknown stages."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Should raise with both unknown stages listed
    with pytest.raises(exceptions.StageNotFoundError, match="unknown"):
        graph.get_execution_order(stage_dag, stages=["stage_a", "unknown1", "unknown2"])


def test_get_execution_order_with_stages_returns_subgraph_order(tmp_path: Path) -> None:
    """get_execution_order with stages returns dependencies in correct order."""
    # Build diamond: input -> A, B -> C
    input_identity = ArtifactIdentity("source", "input.csv")
    a_out = ArtifactIdentity("stage_a", "a.csv")
    b_out = ArtifactIdentity("stage_b", "b.csv")
    c_out = ArtifactIdentity("stage_c", "c.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [a_out]),
        "stage_b": _create_stage("stage_b", [input_identity], [b_out]),
        "stage_c": _create_stage("stage_c", [a_out, b_out], [c_out]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Request only stage_c - should include its dependencies (a and b)
    order = graph.get_execution_order(stage_dag, stages=["stage_c"])

    assert "stage_a" in order
    assert "stage_b" in order
    assert "stage_c" in order
    # C must come after A and B
    assert order.index("stage_c") > order.index("stage_a")
    assert order.index("stage_c") > order.index("stage_b")


# --- extract_graph_view tests ---


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_empty() -> None:
    """extract_graph_view on empty graph returns empty lists."""
    g = graph.build_graph({})
    view = graph.extract_graph_view(g)

    assert view["stages"] == []
    assert view["artifacts"] == []
    assert view["stage_edges"] == []
    assert view["artifact_edges"] == []


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_single_stage(tmp_path: Path) -> None:
    """extract_graph_view extracts stage and artifact from single-stage graph."""
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert view["stages"] == ["stage_a"]
    assert set(view["artifacts"]) == {"source:input.csv", "stage_a:output.csv"}
    # Single stage with no downstream — no stage edges
    assert view["stage_edges"] == []
    # Artifact edges: input -> output (through stage_a)
    assert ("source:input.csv", "stage_a:output.csv") in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_linear_chain(tmp_path: Path) -> None:
    """extract_graph_view extracts correct edges for a linear chain."""
    input_identity = ArtifactIdentity("source", "input.csv")
    intermediate = ArtifactIdentity("stage_a", "intermediate.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [intermediate]),
        "stage_b": _create_stage("stage_b", [intermediate], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"stage_a", "stage_b"}
    assert set(view["artifacts"]) == {
        "source:input.csv",
        "stage_a:intermediate.csv",
        "stage_b:output.csv",
    }
    # stage_a -> stage_b (producer -> consumer, data-flow direction)
    assert ("stage_a", "stage_b") in view["stage_edges"]
    # artifact edges: input -> intermediate, intermediate -> output
    assert ("source:input.csv", "stage_a:intermediate.csv") in view["artifact_edges"]
    assert ("stage_a:intermediate.csv", "stage_b:output.csv") in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_diamond(tmp_path: Path) -> None:
    """extract_graph_view handles diamond DAG correctly."""
    input_identity = ArtifactIdentity("source", "input.csv")
    clean = ArtifactIdentity("preprocess", "clean.csv")
    feats = ArtifactIdentity("features", "feats.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [clean]),
        "features": _create_stage("features", [input_identity], [feats]),
        "train": _create_stage("train", [clean, feats], [model]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"preprocess", "features", "train"}
    # Stage edges (producer -> consumer)
    assert ("preprocess", "train") in view["stage_edges"]
    assert ("features", "train") in view["stage_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_stage_with_multiple_outputs(tmp_path: Path) -> None:
    """extract_graph_view deduplicates stage edges when multiple artifacts connect stages.

    Stage A produces [file1.csv, file2.csv], Stage B consumes both.
    Should create ONE stage edge (A->B) despite two artifact paths.
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    file1 = ArtifactIdentity("stage_a", "file1.csv")
    file2 = ArtifactIdentity("stage_a", "file2.csv")
    output_identity = ArtifactIdentity("stage_b", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [file1, file2]),
        "stage_b": _create_stage("stage_b", [file1, file2], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"stage_a", "stage_b"}

    # Stage edges: A->B appears ONCE (deduplicated despite two artifact connections)
    stage_edges = view["stage_edges"]
    assert stage_edges.count(("stage_a", "stage_b")) == 1, (
        "Stage edges should be deduplicated when multiple artifacts connect same stages"
    )
    assert len(stage_edges) == 1, f"Expected exactly 1 stage edge, got {len(stage_edges)}"

    # Artifact edges: input -> file1, input -> file2, file1 -> output, file2 -> output
    artifact_edges = set(view["artifact_edges"])
    assert ("source:input.csv", "stage_a:file1.csv") in artifact_edges
    assert ("source:input.csv", "stage_a:file2.csv") in artifact_edges
    assert ("stage_a:file1.csv", "stage_b:output.csv") in artifact_edges
    assert ("stage_a:file2.csv", "stage_b:output.csv") in artifact_edges
    assert len(artifact_edges) == 4


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_external_input_artifacts(tmp_path: Path) -> None:
    """extract_graph_view handles external inputs (no producer) correctly.

    External input files should appear in artifacts but have no incoming edges.
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    output_identity = ArtifactIdentity("stage_a", "output.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_identity], [output_identity]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # Check external input has NO incoming artifact edges
    artifact_edges = view["artifact_edges"]
    for _src, dst in artifact_edges:
        assert dst != "source:input.csv", (
            "External input source:input.csv should not be a destination in artifact edges"
        )

    # But it SHOULD have outgoing edges
    assert ("source:input.csv", "stage_a:output.csv") in artifact_edges


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_no_spurious_edges(tmp_path: Path) -> None:
    """extract_graph_view should not create edges between unconnected stages."""
    input_a = ArtifactIdentity("source", "input_a.csv")
    input_b = ArtifactIdentity("source", "input_b.csv")
    output_a = ArtifactIdentity("stage_a", "output_a.csv")
    output_b = ArtifactIdentity("stage_b", "output_b.csv")

    stages = {
        "stage_a": _create_stage("stage_a", [input_a], [output_a]),
        "stage_b": _create_stage("stage_b", [input_b], [output_b]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # No stage edges should exist (disconnected components)
    assert view["stage_edges"] == []

    # Only artifact edges within each component
    artifact_edges = set(view["artifact_edges"])
    assert ("source:input_a.csv", "stage_a:output_a.csv") in artifact_edges
    assert ("source:input_b.csv", "stage_b:output_b.csv") in artifact_edges

    # No cross-edges between components
    assert ("source:input_a.csv", "stage_b:output_b.csv") not in artifact_edges
    assert ("source:input_b.csv", "stage_a:output_a.csv") not in artifact_edges

    # Exact count
    assert len(artifact_edges) == 2


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_complex_diamond_with_edge_verification(tmp_path: Path) -> None:
    """extract_graph_view with strict edge verification for complex diamond.

    Verifies exact edge count and directionality for diamond pattern.
    """
    input_identity = ArtifactIdentity("source", "input.csv")
    clean = ArtifactIdentity("preprocess", "clean.csv")
    feats = ArtifactIdentity("features", "feats.csv")
    model = ArtifactIdentity("train", "model.pkl")

    stages = {
        "preprocess": _create_stage("preprocess", [input_identity], [clean]),
        "features": _create_stage("features", [input_identity], [feats]),
        "train": _create_stage("train", [clean, feats], [model]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # Verify stage edges with exact counts and no reverse edges
    stage_edges = view["stage_edges"]
    assert stage_edges.count(("preprocess", "train")) == 1
    assert stage_edges.count(("features", "train")) == 1
    assert ("train", "preprocess") not in stage_edges
    assert ("train", "features") not in stage_edges

    # Total stage edges should be exactly 2
    assert len(stage_edges) == 2

    # Verify artifact edges
    artifact_edges = set(view["artifact_edges"])
    # From input to intermediates
    assert ("source:input.csv", "preprocess:clean.csv") in artifact_edges
    assert ("source:input.csv", "features:feats.csv") in artifact_edges
    # From intermediates to model
    assert ("preprocess:clean.csv", "train:model.pkl") in artifact_edges
    assert ("features:feats.csv", "train:model.pkl") in artifact_edges

    # NO reverse edges
    assert ("preprocess:clean.csv", "source:input.csv") not in artifact_edges
    assert ("train:model.pkl", "preprocess:clean.csv") not in artifact_edges

    # Exact count
    assert len(artifact_edges) == 4
