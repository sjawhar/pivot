"""Unit tests for DAG render functions (ASCII, Mermaid, DOT)."""

from __future__ import annotations

import pathlib

from pivot import dag, loaders, outputs
from pivot.engine import graph as engine_graph
from pivot.registry import RegistryStageInfo

# =============================================================================
# Helper functions for building test stages
# =============================================================================


def _noop_stage_func() -> None:
    """No-op function for test stages (must be module-level for fingerprinting)."""


def _create_stage(
    name: str,
    deps: list[str],
    outs: list[str],
) -> RegistryStageInfo:
    """Create a stage info dict for testing."""
    return RegistryStageInfo(
        func=_noop_stage_func,
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


# =============================================================================
# Empty pipeline tests
# =============================================================================


def test_render_ascii_empty_graph() -> None:
    """Empty graph returns placeholder text."""
    bipartite = engine_graph.build_graph({})
    view = engine_graph.extract_graph_view(bipartite)
    result = dag.render_ascii(view)
    assert result == "(empty graph)"


def test_render_ascii_empty_graph_stages() -> None:
    """Empty graph with stages=True returns placeholder text."""
    bipartite = engine_graph.build_graph({})
    view = engine_graph.extract_graph_view(bipartite)
    result = dag.render_ascii(view, stages=True)
    assert result == "(empty graph)"


def test_render_mermaid_empty_graph() -> None:
    """Empty graph returns valid empty Mermaid flowchart."""
    bipartite = engine_graph.build_graph({})
    view = engine_graph.extract_graph_view(bipartite)
    result = dag.render_mermaid(view)
    assert result == "flowchart TD"


def test_render_mermaid_empty_graph_stages() -> None:
    """Empty graph with stages=True returns valid empty Mermaid flowchart."""
    bipartite = engine_graph.build_graph({})
    view = engine_graph.extract_graph_view(bipartite)
    result = dag.render_mermaid(view, stages=True)
    assert result == "flowchart TD"


def test_render_dot_empty_graph() -> None:
    """Empty graph returns minimal DOT."""
    bipartite = engine_graph.build_graph({})
    view = engine_graph.extract_graph_view(bipartite)
    result = dag.render_dot(view)
    assert result == "digraph {\n}"


def test_render_dot_empty_graph_stages() -> None:
    """Empty graph with stages=True returns minimal DOT."""
    bipartite = engine_graph.build_graph({})
    view = engine_graph.extract_graph_view(bipartite)
    result = dag.render_dot(view, stages=True)
    assert result == "digraph {\n}"


# =============================================================================
# Single stage tests
# =============================================================================


def test_render_ascii_single_stage_artifact_view() -> None:
    """Single stage shows output artifact in artifact view."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=False)

    # Should contain the artifact path
    assert "data.csv" in result
    # Should have box characters
    assert "+" in result
    assert "|" in result


def test_render_ascii_single_stage_stage_view() -> None:
    """Single stage shows stage name in stage view."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=True)

    # Should contain the stage name
    assert "load" in result
    # Should have box characters
    assert "+" in result
    assert "|" in result


def test_render_mermaid_single_stage_artifact_view() -> None:
    """Single stage shows artifact in Mermaid."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=False)

    assert "flowchart TD" in result
    assert "data.csv" in result


def test_render_mermaid_single_stage_stage_view() -> None:
    """Single stage shows stage name in Mermaid."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    assert "flowchart TD" in result
    assert "load" in result


def test_render_dot_single_stage_artifact_view() -> None:
    """Single stage shows artifact in DOT."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=False)

    assert "digraph {" in result
    assert "data.csv" in result
    assert "}" in result


def test_render_dot_single_stage_stage_view() -> None:
    """Single stage shows stage name in DOT."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    assert "digraph {" in result
    assert "load" in result
    assert "}" in result


# =============================================================================
# Linear chain tests (A -> B -> C)
# =============================================================================


def test_render_ascii_linear_chain_artifact_view() -> None:
    """Linear chain shows artifact flow in ASCII."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=False)

    # Should contain all artifacts
    assert "raw.csv" in result
    assert "clean.csv" in result
    assert "output.csv" in result


def test_render_ascii_linear_chain_stage_view() -> None:
    """Linear chain shows stage flow in ASCII."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=True)

    # Should contain all stages
    assert "extract" in result
    assert "transform" in result
    assert "load" in result


def test_render_mermaid_linear_chain_artifact_view() -> None:
    """Linear chain shows artifact edges in Mermaid."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=False)

    # Should have flowchart and edges
    assert "flowchart TD" in result
    assert "-->" in result
    # Should have all artifacts
    assert "raw.csv" in result
    assert "clean.csv" in result
    assert "output.csv" in result


def test_render_mermaid_linear_chain_stage_view() -> None:
    """Linear chain shows stage edges in Mermaid."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    # Should have flowchart and edges
    assert "flowchart TD" in result
    assert "-->" in result
    # Should have all stages
    assert "extract" in result
    assert "transform" in result
    assert "load" in result


def test_render_dot_linear_chain_artifact_view() -> None:
    """Linear chain shows artifact edges in DOT."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=False)

    # Should have edges
    assert "->" in result
    # Should have all artifacts
    assert "raw.csv" in result
    assert "clean.csv" in result
    assert "output.csv" in result


def test_render_dot_linear_chain_stage_view() -> None:
    """Linear chain shows stage edges in DOT."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    # Should have edges
    assert "->" in result
    # Should have all stages
    assert "extract" in result
    assert "transform" in result
    assert "load" in result


# =============================================================================
# Diamond pattern tests (A -> B, A -> C, B -> D, C -> D)
# =============================================================================


def test_render_ascii_diamond_pattern_artifact_view() -> None:
    """Diamond pattern shows all artifacts in ASCII."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=False)

    # Should contain all artifacts
    assert "data.csv" in result
    assert "a.csv" in result
    assert "b.csv" in result
    assert "merged.csv" in result


def test_render_ascii_diamond_pattern_stage_view() -> None:
    """Diamond pattern shows all stages in ASCII."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=True)

    # Should contain all stages
    assert "source" in result
    assert "branch_a" in result
    assert "branch_b" in result
    assert "merge" in result


def test_render_mermaid_diamond_pattern_artifact_view() -> None:
    """Diamond pattern shows correct edges in Mermaid artifact view."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=False)

    # Should have all artifacts
    assert "data.csv" in result
    assert "a.csv" in result
    assert "b.csv" in result
    assert "merged.csv" in result
    # Should have multiple edges
    assert result.count("-->") >= 4


def test_render_mermaid_diamond_pattern_stage_view() -> None:
    """Diamond pattern shows correct edges in Mermaid stage view."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    # Should have all stages
    assert "source" in result
    assert "branch_a" in result
    assert "branch_b" in result
    assert "merge" in result
    # Should have 4 edges: source->a, source->b, a->merge, b->merge
    assert result.count("-->") >= 4


def test_render_dot_diamond_pattern_artifact_view() -> None:
    """Diamond pattern shows correct edges in DOT artifact view."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=False)

    # Should have all artifacts
    assert "data.csv" in result
    assert "a.csv" in result
    assert "b.csv" in result
    assert "merged.csv" in result
    # Should have multiple edges
    assert result.count("->") >= 4


def test_render_dot_diamond_pattern_stage_view() -> None:
    """Diamond pattern shows correct edges in DOT stage view."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    # Should have all stages
    assert "source" in result
    assert "branch_a" in result
    assert "branch_b" in result
    assert "merge" in result
    # Should have multiple edges
    assert result.count("->") >= 4


# =============================================================================
# Stage with no deps (leaf node) tests
# =============================================================================


def test_render_ascii_stage_no_deps() -> None:
    """Stage with no deps renders as isolated box."""
    stages = {"generate": _create_stage("generate", [], ["output.txt"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=True)

    assert "generate" in result
    assert "+" in result


def test_render_mermaid_stage_no_deps() -> None:
    """Stage with no deps renders as isolated node in Mermaid."""
    stages = {"generate": _create_stage("generate", [], ["output.txt"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    assert "generate" in result
    # No edges expected
    assert "-->" not in result


def test_render_dot_stage_no_deps() -> None:
    """Stage with no deps renders as isolated node in DOT."""
    stages = {"generate": _create_stage("generate", [], ["output.txt"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    assert "generate" in result
    # Isolated node, no edges
    assert "->" not in result


# =============================================================================
# Independent stages (multiple disconnected components) tests
# =============================================================================


def test_render_ascii_disconnected_components() -> None:
    """Disconnected stages are laid out in ASCII."""
    stages = {
        "task_a": _create_stage("task_a", [], ["a.txt"]),
        "task_b": _create_stage("task_b", [], ["b.txt"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_ascii(view, stages=True)

    # Both stages should appear
    assert "task_a" in result
    assert "task_b" in result


def test_render_mermaid_disconnected_components() -> None:
    """Disconnected stages are rendered in Mermaid."""
    stages = {
        "task_a": _create_stage("task_a", [], ["a.txt"]),
        "task_b": _create_stage("task_b", [], ["b.txt"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    assert "task_a" in result
    assert "task_b" in result
    # No edges between disconnected components
    assert "-->" not in result


def test_render_dot_disconnected_components() -> None:
    """Disconnected stages are rendered in DOT."""
    stages = {
        "task_a": _create_stage("task_a", [], ["a.txt"]),
        "task_b": _create_stage("task_b", [], ["b.txt"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    assert "task_a" in result
    assert "task_b" in result
    # No edges
    assert "->" not in result


# =============================================================================
# Special character handling tests
# =============================================================================


def test_render_mermaid_escapes_quotes_in_labels() -> None:
    """Mermaid output escapes quotes in artifact/stage labels using HTML entities."""
    stages = {"stage": _create_stage("stage", [], ['file"with"quotes.txt'])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=False)

    # Quotes should be escaped as HTML entities
    assert "&quot;" in result
    assert 'file"with"quotes.txt' not in result  # Original should be escaped


def test_render_dot_escapes_quotes_in_labels() -> None:
    """DOT output escapes quotes in artifact/stage labels."""
    stages = {"stage": _create_stage("stage", [], ['file"with"quotes.txt'])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=False)

    # Quotes should be escaped
    assert '\\"' in result


def test_render_mermaid_escapes_newlines_and_hashes() -> None:
    """Mermaid output escapes newlines and hash characters."""
    stages = {"stage": _create_stage("stage", [], ["file#v2\nwith\nnewlines.txt"])}
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=False)

    # Newlines should be replaced with spaces
    assert "\n" not in result.split('"')[1] if '"' in result else True
    # Hash should be escaped as HTML entity
    assert "&#35;" in result


# =============================================================================
# Subgraph rendering tests (filtered graph)
# =============================================================================


def test_render_ascii_subgraph() -> None:
    """Render a subgraph containing only part of the pipeline."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    g = engine_graph.build_graph(stages)

    # Get subgraph of just extract and transform
    subgraph = g.subgraph(
        [
            engine_graph.stage_node("extract"),
            engine_graph.stage_node("transform"),
            engine_graph.artifact_node(pathlib.Path("raw.csv")),
            engine_graph.artifact_node(pathlib.Path("clean.csv")),
        ]
    )
    view = engine_graph.extract_graph_view(subgraph)

    result = dag.render_ascii(view, stages=True)

    assert "extract" in result
    assert "transform" in result
    # load should not be present
    assert "load" not in result


def test_render_mermaid_subgraph() -> None:
    """Render a subgraph in Mermaid format."""
    stages = {
        "extract": _create_stage("extract", [], ["raw.csv"]),
        "transform": _create_stage("transform", ["raw.csv"], ["clean.csv"]),
        "load": _create_stage("load", ["clean.csv"], ["output.csv"]),
    }
    g = engine_graph.build_graph(stages)

    # Get subgraph of just extract and transform
    subgraph = g.subgraph(
        [
            engine_graph.stage_node("extract"),
            engine_graph.stage_node("transform"),
            engine_graph.artifact_node(pathlib.Path("raw.csv")),
            engine_graph.artifact_node(pathlib.Path("clean.csv")),
        ]
    )
    view = engine_graph.extract_graph_view(subgraph)

    result = dag.render_mermaid(view, stages=True)

    assert "extract" in result
    assert "transform" in result
    assert "load" not in result


# =============================================================================
# Mermaid/DOT format validation tests
# =============================================================================


def test_render_mermaid_format_is_valid() -> None:
    """Mermaid output follows valid format structure."""
    stages = {
        "a": _create_stage("a", [], ["out.txt"]),
        "b": _create_stage("b", ["out.txt"], ["final.txt"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    lines = result.split("\n")
    # First line should be flowchart directive
    assert lines[0] == "flowchart TD"
    # Should have node definitions with brackets
    node_lines = [line for line in lines if "[" in line and "]" in line]
    assert len(node_lines) >= 2
    # Should have edge definitions
    edge_lines = [line for line in lines if "-->" in line]
    assert len(edge_lines) >= 1


def test_render_dot_format_is_valid() -> None:
    """DOT output follows valid format structure."""
    stages = {
        "a": _create_stage("a", [], ["out.txt"]),
        "b": _create_stage("b", ["out.txt"], ["final.txt"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    lines = result.split("\n")
    # First line should open digraph
    assert "digraph {" in lines[0]
    # Last line should close it
    assert lines[-1] == "}"
    # Should have edge definitions
    edge_lines = [line for line in lines if "->" in line]
    assert len(edge_lines) >= 1


# =============================================================================
# Edge directionality verification tests
# =============================================================================


def test_render_mermaid_linear_chain_edge_direction() -> None:
    """Mermaid linear chain has correct edge direction (producer -> consumer)."""
    stages = {
        "stage_a": _create_stage("stage_a", [], ["intermediate.csv"]),
        "stage_b": _create_stage("stage_b", ["intermediate.csv"], ["final.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=True)

    # Parse node IDs from result
    lines = result.split("\n")
    node_map = dict[str, str]()
    for line in lines:
        if "[" in line and "]" in line and "node" in line:
            # Extract: node1["stage_a"] -> node_map["stage_a"] = "node1"
            node_id = line.split("[")[0].strip()
            label_start = line.index('["') + 2
            label_end = line.index('"]')
            label = line[label_start:label_end]
            node_map[label] = node_id

    # Verify edge direction: stage_a --> stage_b (not reversed)
    edge_line = f"    {node_map['stage_a']}-->{node_map['stage_b']}"
    assert edge_line in result, f"Expected edge {edge_line} in output"

    # Verify NO reverse edge
    reverse_edge = f"    {node_map['stage_b']}-->{node_map['stage_a']}"
    assert reverse_edge not in result, f"Unexpected reverse edge {reverse_edge}"


def test_render_dot_diamond_edge_completeness() -> None:
    """DOT diamond has all expected edges and no extras."""
    stages = {
        "source": _create_stage("source", [], ["data.csv"]),
        "branch_a": _create_stage("branch_a", ["data.csv"], ["a.csv"]),
        "branch_b": _create_stage("branch_b", ["data.csv"], ["b.csv"]),
        "merge": _create_stage("merge", ["a.csv", "b.csv"], ["merged.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_dot(view, stages=True)

    # Expected edges in stage view
    expected_edges = [
        ('"source"', '"branch_a"'),
        ('"source"', '"branch_b"'),
        ('"branch_a"', '"merge"'),
        ('"branch_b"', '"merge"'),
    ]

    for src, dst in expected_edges:
        edge_line = f"{src} -> {dst}"
        assert edge_line in result, f"Missing edge: {edge_line}"

    # Count total edges
    edge_count = result.count(" -> ")
    assert edge_count == 4, f"Expected 4 edges in diamond, got {edge_count}"


def test_render_mermaid_artifact_edges_match_data_flow() -> None:
    """Mermaid artifact view edges follow data flow (input -> output)."""
    stages = {
        "stage_a": _create_stage("stage_a", ["input.csv"], ["output.csv"]),
    }
    bipartite = engine_graph.build_graph(stages)
    view = engine_graph.extract_graph_view(bipartite)

    result = dag.render_mermaid(view, stages=False)

    # Parse to find which node is input vs output
    lines = result.split("\n")
    node_map = dict[str, str]()
    for line in lines:
        if "[" in line and "]" in line and "node" in line:
            node_id = line.split("[")[0].strip()
            label_start = line.index('["') + 2
            label_end = line.index('"]')
            label = line[label_start:label_end]
            node_map[label] = node_id

    # Edge should go input.csv -> output.csv (data flow direction)
    edge_line = f"    {node_map['input.csv']}-->{node_map['output.csv']}"
    assert edge_line in result, f"Expected data flow edge {edge_line}"

    # NOT reversed
    reverse_edge = f"    {node_map['output.csv']}-->{node_map['input.csv']}"
    assert reverse_edge not in result, f"Unexpected reverse edge {reverse_edge}"
