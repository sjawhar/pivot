"""Bipartite artifact-stage graph built on NetworkX."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import networkx as nx

from pivot.engine.types import NodeType

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.registry import RegistryStageInfo

__all__ = [
    "artifact_node",
    "stage_node",
    "parse_node",
    "build_graph",
    "get_consumers",
    "get_producer",
    "get_watch_paths",
    "get_downstream_stages",
    "update_stage",
]


def artifact_node(path: Path) -> str:
    """Create artifact node ID from path."""
    return f"artifact:{path}"


def stage_node(name: str) -> str:
    """Create stage node ID from name."""
    return f"stage:{name}"


def parse_node(node: str) -> tuple[NodeType, str]:
    """Extract NodeType and value from node ID.

    Handles colons in paths by only splitting on the first colon.
    """
    prefix, value = node.split(":", 1)
    return NodeType(prefix), value


def build_graph(stages: dict[str, RegistryStageInfo]) -> nx.DiGraph[str]:
    """Build bipartite artifact-stage graph from stage definitions.

    Args:
        stages: Dict mapping stage name to RegistryStageInfo.

    Returns:
        Directed graph where:
        - Nodes are either artifacts (files) or stages (functions)
        - Edges go: artifact -> stage (consumed by) and stage -> artifact (produces)
    """
    g: nx.DiGraph[str] = nx.DiGraph()

    for stage_name, info in stages.items():
        stage = stage_node(stage_name)
        g.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact -> stage
        for dep_path in info["deps_paths"]:
            artifact = artifact_node(pathlib.Path(dep_path))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(artifact, stage)

        # Outs: stage -> artifact
        for out in info["outs"]:
            artifact = artifact_node(pathlib.Path(str(out.path)))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(stage, artifact)

    return g


def get_consumers(g: nx.DiGraph[str], path: Path) -> list[str]:
    """Get stages that depend on this artifact.

    Args:
        g: The bipartite graph.
        path: Path to the artifact.

    Returns:
        List of stage names that consume this artifact.
    """
    node = artifact_node(path)
    if node not in g:
        return []
    return [parse_node(n)[1] for n in g.successors(node) if g.nodes[n]["type"] == NodeType.STAGE]


def get_producer(g: nx.DiGraph[str], path: Path) -> str | None:
    """Get the stage that produces this artifact.

    Args:
        g: The bipartite graph.
        path: Path to the artifact.

    Returns:
        Stage name that produces this artifact, or None if it's an input.
    """
    node = artifact_node(path)
    if node not in g:
        return None
    for pred in g.predecessors(node):
        if g.nodes[pred]["type"] == NodeType.STAGE:
            return parse_node(pred)[1]
    return None


def get_watch_paths(g: nx.DiGraph[str]) -> list[Path]:
    """Get all artifact paths (for filesystem watcher).

    Args:
        g: The bipartite graph.

    Returns:
        List of all artifact paths in the graph.
    """
    return [
        pathlib.Path(parse_node(n)[1]) for n in g.nodes() if g.nodes[n]["type"] == NodeType.ARTIFACT
    ]


def get_downstream_stages(g: nx.DiGraph[str], stage_name: str) -> list[str]:
    """Get all stages transitively downstream of this one.

    Args:
        g: The bipartite graph.
        stage_name: Name of the stage.

    Returns:
        List of stage names that transitively depend on this stage's outputs.
    """
    node = stage_node(stage_name)
    if node not in g:
        return []

    downstream = list[str]()
    for descendant in nx.descendants(g, node):
        if g.nodes[descendant]["type"] == NodeType.STAGE:
            downstream.append(parse_node(descendant)[1])
    return downstream


def update_stage(g: nx.DiGraph[str], stage_name: str, new_info: RegistryStageInfo) -> None:
    """Incrementally update graph when a stage's definition changes.

    Efficiently diffs current and new deps/outs, adding and removing edges
    as needed. Removes orphaned artifact nodes (no longer connected to any stage).

    Args:
        g: The bipartite graph to modify in place.
        stage_name: Name of the stage to update.
        new_info: New stage definition from registry.
    """
    stage = stage_node(stage_name)

    # Get current deps and outs from graph
    current_deps = {
        pathlib.Path(parse_node(n)[1])
        for n in g.predecessors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }
    current_outs = {
        pathlib.Path(parse_node(n)[1])
        for n in g.successors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }

    # Get new deps and outs from info
    new_deps = {pathlib.Path(p) for p in new_info["deps_paths"]}
    new_outs = {pathlib.Path(str(out.path)) for out in new_info["outs"]}

    # Remove old deps
    for removed_dep in current_deps - new_deps:
        artifact = artifact_node(removed_dep)
        g.remove_edge(artifact, stage)
        if g.degree(artifact) == 0:
            g.remove_node(artifact)

    # Add new deps
    for added_dep in new_deps - current_deps:
        artifact = artifact_node(added_dep)
        if artifact not in g:
            g.add_node(artifact, type=NodeType.ARTIFACT)
        g.add_edge(artifact, stage)

    # Remove old outs
    for removed_out in current_outs - new_outs:
        artifact = artifact_node(removed_out)
        g.remove_edge(stage, artifact)
        if g.degree(artifact) == 0:
            g.remove_node(artifact)

    # Add new outs
    for added_out in new_outs - current_outs:
        artifact = artifact_node(added_out)
        if artifact not in g:
            g.add_node(artifact, type=NodeType.ARTIFACT)
        g.add_edge(stage, artifact)
