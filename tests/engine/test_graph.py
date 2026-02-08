"""Tests for the bipartite artifact-stage graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from pivot import exceptions, loaders, outputs
from pivot.engine import graph, types
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
        mutex=list[str](),
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        dep_specs={},
        out_specs={},
        params_arg_name=None,
        state_dir=None,
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


@pytest.mark.usefixtures("clean_registry")
def test_build_graph_simple_chain(tmp_path: Path) -> None:
    """Build bipartite graph for simple chain: input -> A -> intermediate -> B -> output."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    # Create stages dict directly for isolated graph test
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


# --- get_stage_dag tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_stage_dag_extracts_stage_only_graph(tmp_path: Path) -> None:
    """get_stage_dag returns stage-only DAG from bipartite graph."""
    input_file = tmp_path / "input.csv"
    cleaned = tmp_path / "cleaned.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(cleaned)]),
        "train": _create_stage("train", [str(cleaned)], [str(model)]),
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
    input_file = tmp_path / "input.csv"
    cleaned = tmp_path / "cleaned.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(cleaned)]),
        "train": _create_stage("train", [str(cleaned)], [str(model)]),
    }
    g = graph.build_graph(stages)

    # Input change should affect both preprocess AND train
    consumers = graph.get_artifact_consumers(g, input_file, include_downstream=True)

    assert "preprocess" in consumers
    assert "train" in consumers  # Downstream of preprocess

    # Without downstream, only direct consumers
    direct = graph.get_artifact_consumers(g, input_file, include_downstream=False)

    assert "preprocess" in direct
    assert "train" not in direct


@pytest.mark.usefixtures("clean_registry")
def test_get_artifact_consumers_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    """get_artifact_consumers returns empty list for unknown artifact."""
    g = graph.build_graph({})
    consumers = graph.get_artifact_consumers(g, tmp_path / "unknown.csv")
    assert consumers == []


# --- Validation tests ---


def test_build_graph_raises_on_cycle(tmp_path: Path) -> None:
    """build_graph raises CyclicGraphError when graph has cycles."""
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"

    stages = {
        "stage_a": _create_stage("stage_a", [str(file_b)], [str(file_a)]),
        "stage_b": _create_stage("stage_b", [str(file_a)], [str(file_b)]),
    }

    with pytest.raises(exceptions.CyclicGraphError, match="Circular dependency"):
        graph.build_graph(stages)  # Cycles always checked


def test_build_graph_raises_on_missing_dependency(tmp_path: Path) -> None:
    """build_graph raises DependencyNotFoundError when validate=True."""
    output_file = tmp_path / "output.csv"
    missing_dep = tmp_path / "missing.csv"

    stages = {
        "stage_a": _create_stage("stage_a", [str(missing_dep)], [str(output_file)]),
    }

    with pytest.raises(exceptions.DependencyNotFoundError):
        graph.build_graph(stages, validate=True)


def test_build_graph_allows_missing_when_validate_false(tmp_path: Path) -> None:
    """build_graph allows missing deps when validate=False."""
    output_file = tmp_path / "output.csv"
    missing_dep = tmp_path / "missing.csv"

    stages = {
        "stage_a": _create_stage("stage_a", [str(missing_dep)], [str(output_file)]),
    }

    # Should not raise
    g = graph.build_graph(stages, validate=False)
    assert "stage:stage_a" in g


def test_build_graph_accepts_tracked_file(tmp_path: Path) -> None:
    """build_graph accepts tracked files as valid dependency sources."""
    output_file = tmp_path / "output.csv"
    tracked_input = tmp_path / "tracked.csv"

    tracked_files: dict[str, PvtData] = {
        str(tracked_input): PvtData(path="tracked.csv", hash="abc123", size=100)
    }

    stages = {
        "stage_a": _create_stage("stage_a", [str(tracked_input)], [str(output_file)]),
    }

    # Should not raise - tracked file is valid
    g = graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    assert "stage:stage_a" in g


def test_build_graph_directory_dependency(tmp_path: Path) -> None:
    """build_graph resolves directory dependencies via trie."""
    input_file = tmp_path / "input.csv"
    output_dir = tmp_path / "outputs"
    file_a = output_dir / "a.csv"
    input_file.touch()

    stages = {
        "producer": _create_stage("producer", [str(input_file)], [str(file_a)]),
        "consumer": _create_stage("consumer", [str(output_dir)], [str(tmp_path / "final.csv")]),
    }

    # Should not raise - output_dir contains file_a from producer
    g = graph.build_graph(stages, validate=True)
    assert "stage:producer" in g
    assert "stage:consumer" in g


# --- get_upstream_stages tests ---


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_returns_producing_stages(tmp_path: Path) -> None:
    """get_upstream_stages returns stages that produce inputs for a stage."""
    input_file = tmp_path / "input.csv"
    cleaned = tmp_path / "cleaned.csv"
    features = tmp_path / "features.csv"
    model = tmp_path / "model.pkl"
    input_file.touch()

    stages = {
        "preprocess": _create_stage("preprocess", [str(input_file)], [str(cleaned)]),
        "extract": _create_stage("extract", [str(input_file)], [str(features)]),
        "train": _create_stage("train", [str(cleaned), str(features)], [str(model)]),
    }

    g = graph.build_graph(stages)
    upstream = graph.get_upstream_stages(g, "train")

    # train depends on outputs from preprocess and extract
    assert set(upstream) == {"preprocess", "extract"}


@pytest.mark.usefixtures("clean_registry")
def test_get_upstream_stages_empty_for_root_stage(tmp_path: Path) -> None:
    """get_upstream_stages returns empty list for stage with no upstream dependencies."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
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
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
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
    input_file = tmp_path / "input.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(out_a)]),
        "stage_b": _create_stage("stage_b", [str(input_file)], [str(out_b)]),
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
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    producer = graph.get_producer(g, tmp_path / "completely_unknown.csv")

    assert producer is None


@pytest.mark.usefixtures("clean_registry")
def test_get_downstream_stages_empty_for_unknown_stage(tmp_path: Path) -> None:
    """get_downstream_stages returns empty list for unknown stage."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    g = graph.build_graph(stages)
    downstream = graph.get_downstream_stages(g, "unknown_stage")

    assert downstream == []


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_adds_new_out(tmp_path: Path) -> None:
    """update_stage adds new output edges."""
    input_file = tmp_path / "input.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    input_file.touch()

    # Initial: stage_a produces only out_a
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(out_a)]),
    }
    g = graph.build_graph(stages)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) is None

    # Update: stage_a now also produces out_b
    new_info = _create_stage("stage_a", [str(input_file)], [str(out_a), str(out_b)])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) == "stage_a"


@pytest.mark.usefixtures("clean_registry")
def test_update_stage_removes_old_out(tmp_path: Path) -> None:
    """update_stage removes old output edges and orphaned artifacts."""
    input_file = tmp_path / "input.csv"
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    input_file.touch()

    # Initial: stage_a produces both outputs
    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(out_a), str(out_b)]),
    }
    g = graph.build_graph(stages)

    # Update: stage_a now only produces out_a
    new_info = _create_stage("stage_a", [str(input_file)], [str(out_a)])
    graph.update_stage(g, "stage_a", new_info)

    assert graph.get_producer(g, out_a) == "stage_a"
    assert graph.get_producer(g, out_b) is None

    # out_b should be removed from graph (orphaned)
    assert graph.artifact_node(out_b) not in g


@pytest.mark.usefixtures("clean_registry")
def test_build_tracked_trie_empty() -> None:
    """build_tracked_trie handles empty tracked files dict."""
    trie = graph.build_tracked_trie({})
    assert len(trie) == 0


@pytest.mark.usefixtures("clean_registry")
def test_build_tracked_trie_single_file(tmp_path: Path) -> None:
    """build_tracked_trie creates trie from single file."""
    tracked_path = str(tmp_path / "file.csv")
    tracked_files: dict[str, PvtData] = {
        tracked_path: PvtData(path="file.csv", hash="abc123", size=100)
    }

    trie = graph.build_tracked_trie(tracked_files)

    # Trie should contain the path
    path_key = (tmp_path / "file.csv").parts
    assert trie[path_key] == tracked_path


@pytest.mark.usefixtures("clean_registry")
def test_build_tracked_trie_nested_files(tmp_path: Path) -> None:
    """build_tracked_trie creates trie from nested file structure."""
    file1 = str(tmp_path / "data" / "a.csv")
    file2 = str(tmp_path / "data" / "b.csv")
    tracked_files: dict[str, PvtData] = {
        file1: PvtData(path="data/a.csv", hash="abc", size=50),
        file2: PvtData(path="data/b.csv", hash="def", size=50),
    }

    trie = graph.build_tracked_trie(tracked_files)

    # Both files should be in trie
    assert trie[(tmp_path / "data" / "a.csv").parts] == file1
    assert trie[(tmp_path / "data" / "b.csv").parts] == file2


# --- Edge cases for directory dependency resolution ---


@pytest.mark.usefixtures("clean_registry")
def test_directory_dependency_parent_is_output(tmp_path: Path) -> None:
    """File depends on parent directory that is produced by a stage."""
    data_dir = tmp_path / "data"
    file_in_dir = data_dir / "file.csv"
    output_file = tmp_path / "output.csv"
    data_dir.mkdir()
    file_in_dir.touch()

    stages = {
        # Producer outputs the directory
        "produce_dir": _create_stage("produce_dir", [], [str(data_dir)]),
        # Consumer depends on a file inside the directory
        "consume_file": _create_stage("consume_file", [str(file_in_dir)], [str(output_file)]),
    }

    g = graph.build_graph(stages, validate=False)

    # Should create edge from consumer to producer
    stage_dag = graph.get_stage_dag(g)
    assert stage_dag.has_edge("consume_file", "produce_dir")


@pytest.mark.usefixtures("clean_registry")
def test_directory_dependency_with_seen_stages_dedupe(tmp_path: Path) -> None:
    """Directory dependency correctly deduplicates stages producing multiple files."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    stages = {
        # Producer outputs multiple files in the same directory
        "producer": _create_stage(
            "producer",
            [],
            [str(output_dir / "a.csv"), str(output_dir / "b.csv"), str(output_dir / "c.csv")],
        ),
        # Consumer depends on the directory
        "consumer": _create_stage("consumer", [str(output_dir)], [str(tmp_path / "result.csv")]),
    }

    g = graph.build_graph(stages, validate=False)
    stage_dag = graph.get_stage_dag(g)

    # Should have exactly one edge from consumer to producer (not three)
    edges_to_producer = list(stage_dag.successors("consumer"))
    assert edges_to_producer == ["producer"]


# --- Error path tests ---


@pytest.mark.usefixtures("clean_registry")
def test_cycle_detection_error_message_format(tmp_path: Path) -> None:
    """Cycle error message contains affected stage names."""
    stages = {
        "stage_a": _create_stage("stage_a", [str(tmp_path / "b.csv")], [str(tmp_path / "a.csv")]),
        "stage_b": _create_stage("stage_b", [str(tmp_path / "a.csv")], [str(tmp_path / "b.csv")]),
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
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Should raise StageNotFoundError, not raw NetworkXError
    with pytest.raises(exceptions.StageNotFoundError, match="unknown_stage"):
        graph.get_execution_order(stage_dag, stages=["unknown_stage"])


def test_get_execution_order_mixed_known_unknown_stages(tmp_path: Path) -> None:
    """get_execution_order raises StageNotFoundError with all unknown stages."""
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }

    bipartite = graph.build_graph(stages)
    stage_dag = graph.get_stage_dag(bipartite)

    # Should raise with both unknown stages listed
    with pytest.raises(exceptions.StageNotFoundError, match="unknown"):
        graph.get_execution_order(stage_dag, stages=["stage_a", "unknown1", "unknown2"])


def test_get_execution_order_with_stages_returns_subgraph_order(tmp_path: Path) -> None:
    """get_execution_order with stages returns dependencies in correct order."""
    # Build diamond: input -> A, B -> C
    input_file = tmp_path / "input.csv"
    a_out = tmp_path / "a.csv"
    b_out = tmp_path / "b.csv"
    c_out = tmp_path / "c.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(a_out)]),
        "stage_b": _create_stage("stage_b", [str(input_file)], [str(b_out)]),
        "stage_c": _create_stage("stage_c", [str(a_out), str(b_out)], [str(c_out)]),
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


def test_tracked_file_inside_directory_validates(tmp_path: Path) -> None:
    """Dependency inside a tracked directory is recognized as valid."""
    # Track a directory
    tracked_dir = tmp_path / "tracked_data"
    tracked_dir.mkdir()

    # Stage depends on a file INSIDE the tracked directory
    dep_inside = tracked_dir / "nested" / "data.csv"

    stages = {
        "consumer": _create_stage("consumer", [str(dep_inside)], [str(tmp_path / "out.csv")]),
    }

    # The tracked_files dict has the directory tracked
    tracked_files: dict[str, PvtData] = {
        str(tracked_dir): {"path": "tracked_data", "hash": "abc123", "size": 0}
    }

    # Should NOT raise - dependency is inside tracked directory
    g = graph.build_graph(stages, validate=True, tracked_files=tracked_files)
    assert "stage:consumer" in g


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
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert view["stages"] == ["stage_a"]
    assert set(view["artifacts"]) == {str(input_file), str(output_file)}
    # Single stage with no downstream — no stage edges
    assert view["stage_edges"] == []
    # Artifact edges: input -> output (through stage_a)
    assert (str(input_file), str(output_file)) in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_linear_chain(tmp_path: Path) -> None:
    """extract_graph_view extracts correct edges for a linear chain."""
    input_file = tmp_path / "input.csv"
    intermediate = tmp_path / "intermediate.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(intermediate)]),
        "stage_b": _create_stage("stage_b", [str(intermediate)], [str(output_file)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    assert set(view["stages"]) == {"stage_a", "stage_b"}
    assert set(view["artifacts"]) == {str(input_file), str(intermediate), str(output_file)}
    # stage_a -> stage_b (producer -> consumer, data-flow direction)
    assert ("stage_a", "stage_b") in view["stage_edges"]
    # artifact edges: input -> intermediate, intermediate -> output
    assert (str(input_file), str(intermediate)) in view["artifact_edges"]
    assert (str(intermediate), str(output_file)) in view["artifact_edges"]


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_diamond(tmp_path: Path) -> None:
    """extract_graph_view handles diamond DAG correctly."""
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
    input_file = tmp_path / "input.csv"
    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(file1), str(file2)]),
        "stage_b": _create_stage("stage_b", [str(file1), str(file2)], [str(output_file)]),
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
    assert (str(input_file), str(file1)) in artifact_edges
    assert (str(input_file), str(file2)) in artifact_edges
    assert (str(file1), str(output_file)) in artifact_edges
    assert (str(file2), str(output_file)) in artifact_edges
    assert len(artifact_edges) == 4


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_external_input_artifacts(tmp_path: Path) -> None:
    """extract_graph_view handles external inputs (no producer) correctly.

    External input files should appear in artifacts but have no incoming edges.
    """
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    input_file.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_file)], [str(output_file)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # Check external input has NO incoming artifact edges
    artifact_edges = view["artifact_edges"]
    for _src, dst in artifact_edges:
        assert dst != str(input_file), (
            f"External input {input_file} should not be a destination in artifact edges"
        )

    # But it SHOULD have outgoing edges
    assert (str(input_file), str(output_file)) in artifact_edges


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_no_spurious_edges(tmp_path: Path) -> None:
    """extract_graph_view should not create edges between unconnected stages."""
    input_a = tmp_path / "input_a.csv"
    input_b = tmp_path / "input_b.csv"
    output_a = tmp_path / "output_a.csv"
    output_b = tmp_path / "output_b.csv"
    input_a.touch()
    input_b.touch()

    stages = {
        "stage_a": _create_stage("stage_a", [str(input_a)], [str(output_a)]),
        "stage_b": _create_stage("stage_b", [str(input_b)], [str(output_b)]),
    }
    g = graph.build_graph(stages)
    view = graph.extract_graph_view(g)

    # No stage edges should exist (disconnected components)
    assert view["stage_edges"] == []

    # Only artifact edges within each component
    artifact_edges = set(view["artifact_edges"])
    assert (str(input_a), str(output_a)) in artifact_edges
    assert (str(input_b), str(output_b)) in artifact_edges

    # No cross-edges between components
    assert (str(input_a), str(output_b)) not in artifact_edges
    assert (str(input_b), str(output_a)) not in artifact_edges

    # Exact count
    assert len(artifact_edges) == 2


@pytest.mark.usefixtures("clean_registry")
def test_extract_graph_view_complex_diamond_with_edge_verification(tmp_path: Path) -> None:
    """extract_graph_view with strict edge verification for complex diamond.

    Verifies exact edge count and directionality for diamond pattern.
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
    assert (str(input_file), str(clean)) in artifact_edges
    assert (str(input_file), str(feats)) in artifact_edges
    # From intermediates to model
    assert (str(clean), str(model)) in artifact_edges
    assert (str(feats), str(model)) in artifact_edges

    # NO reverse edges
    assert (str(clean), str(input_file)) not in artifact_edges
    assert (str(model), str(clean)) not in artifact_edges

    # Exact count
    assert len(artifact_edges) == 4
