"""Unit tests for DAG render functions (ASCII, Mermaid, DOT)."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot import dag, loaders, outputs
from pivot.engine import graph as engine_graph
from pivot.registry import RegistryStageInfo

if TYPE_CHECKING:
    import networkx as nx


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
    )


def _build_graph(stages_dict: dict[str, RegistryStageInfo]) -> nx.DiGraph[str]:
    """Build bipartite graph from stages dict."""
    return engine_graph.build_graph(stages_dict)


# =============================================================================
# Empty pipeline tests
# =============================================================================


def test_render_ascii_empty_graph() -> None:
    """Empty graph returns placeholder text."""
    g = _build_graph({})
    result = dag.render_ascii(g)
    assert result == "(empty graph)"


def test_render_ascii_empty_graph_stages() -> None:
    """Empty graph with stages=True returns placeholder text."""
    g = _build_graph({})
    result = dag.render_ascii(g, stages=True)
    assert result == "(empty graph)"


def test_render_mermaid_empty_graph() -> None:
    """Empty graph returns valid empty Mermaid flowchart."""
    g = _build_graph({})
    result = dag.render_mermaid(g)
    assert result == "flowchart TD"


def test_render_mermaid_empty_graph_stages() -> None:
    """Empty graph with stages=True returns valid empty Mermaid flowchart."""
    g = _build_graph({})
    result = dag.render_mermaid(g, stages=True)
    assert result == "flowchart TD"


def test_render_dot_empty_graph() -> None:
    """Empty graph returns minimal DOT."""
    g = _build_graph({})
    result = dag.render_dot(g)
    assert result == "digraph {\n}"


def test_render_dot_empty_graph_stages() -> None:
    """Empty graph with stages=True returns minimal DOT."""
    g = _build_graph({})
    result = dag.render_dot(g, stages=True)
    assert result == "digraph {\n}"


# =============================================================================
# Single stage tests
# =============================================================================


def test_render_ascii_single_stage_artifact_view() -> None:
    """Single stage shows output artifact in artifact view."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=False)

    # Should contain the artifact path
    assert "data.csv" in result
    # Should have box characters
    assert "+" in result
    assert "|" in result


def test_render_ascii_single_stage_stage_view() -> None:
    """Single stage shows stage name in stage view."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=True)

    # Should contain the stage name
    assert "load" in result
    # Should have box characters
    assert "+" in result
    assert "|" in result


def test_render_mermaid_single_stage_artifact_view() -> None:
    """Single stage shows artifact in Mermaid."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=False)

    assert "flowchart TD" in result
    assert "data.csv" in result


def test_render_mermaid_single_stage_stage_view() -> None:
    """Single stage shows stage name in Mermaid."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=True)

    assert "flowchart TD" in result
    assert "load" in result


def test_render_dot_single_stage_artifact_view() -> None:
    """Single stage shows artifact in DOT."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=False)

    assert "digraph {" in result
    assert "data.csv" in result
    assert "}" in result


def test_render_dot_single_stage_stage_view() -> None:
    """Single stage shows stage name in DOT."""
    stages = {"load": _create_stage("load", [], ["data.csv"])}
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=False)

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
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=False)

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
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=False)

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
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=False)

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
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=False)

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
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=False)

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
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=True)

    assert "generate" in result
    assert "+" in result


def test_render_mermaid_stage_no_deps() -> None:
    """Stage with no deps renders as isolated node in Mermaid."""
    stages = {"generate": _create_stage("generate", [], ["output.txt"])}
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=True)

    assert "generate" in result
    # No edges expected
    assert "-->" not in result


def test_render_dot_stage_no_deps() -> None:
    """Stage with no deps renders as isolated node in DOT."""
    stages = {"generate": _create_stage("generate", [], ["output.txt"])}
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=True)

    # Both stages should appear
    assert "task_a" in result
    assert "task_b" in result


def test_render_mermaid_disconnected_components() -> None:
    """Disconnected stages are rendered in Mermaid."""
    stages = {
        "task_a": _create_stage("task_a", [], ["a.txt"]),
        "task_b": _create_stage("task_b", [], ["b.txt"]),
    }
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=False)

    # Quotes should be escaped as HTML entities
    assert "&quot;" in result
    assert 'file"with"quotes.txt' not in result  # Original should be escaped


def test_render_dot_escapes_quotes_in_labels() -> None:
    """DOT output escapes quotes in artifact/stage labels."""
    stages = {"stage": _create_stage("stage", [], ['file"with"quotes.txt'])}
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=False)

    # Quotes should be escaped
    assert '\\"' in result


def test_render_mermaid_escapes_newlines_and_hashes() -> None:
    """Mermaid output escapes newlines and hash characters."""
    stages = {"stage": _create_stage("stage", [], ["file#v2\nwith\nnewlines.txt"])}
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=False)

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
    g = _build_graph(stages)

    # Get subgraph of just extract and transform
    subgraph = g.subgraph(
        [
            engine_graph.stage_node("extract"),
            engine_graph.stage_node("transform"),
            engine_graph.artifact_node(pathlib.Path("raw.csv")),
            engine_graph.artifact_node(pathlib.Path("clean.csv")),
        ]
    )

    result = dag.render_ascii(subgraph, stages=True)

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
    g = _build_graph(stages)

    # Get subgraph of just extract and transform
    subgraph = g.subgraph(
        [
            engine_graph.stage_node("extract"),
            engine_graph.stage_node("transform"),
            engine_graph.artifact_node(pathlib.Path("raw.csv")),
            engine_graph.artifact_node(pathlib.Path("clean.csv")),
        ]
    )

    result = dag.render_mermaid(subgraph, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_mermaid(g, stages=True)

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
    g = _build_graph(stages)

    result = dag.render_dot(g, stages=True)

    lines = result.split("\n")
    # First line should open digraph
    assert "digraph {" in lines[0]
    # Last line should close it
    assert lines[-1] == "}"
    # Should have edge definitions
    edge_lines = [line for line in lines if "->" in line]
    assert len(edge_lines) >= 1


def test_render_ascii_wide_characters() -> None:
    """ASCII rendering handles wide characters (CJK, emoji) correctly."""
    # Use a CJK character which has display width 2
    stages = {
        "日本語": _create_stage("日本語", [], ["output.txt"]),
    }
    g = _build_graph(stages)

    result = dag.render_ascii(g, stages=True)

    # Should contain the label
    assert "日本語" in result
    # Box should be properly formed (has borders)
    assert "+" in result
    assert "|" in result
