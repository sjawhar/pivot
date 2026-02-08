"""Bipartite artifact-stage graph built on NetworkX."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, TypedDict

import networkx as nx
import pygtrie

from pivot.engine.types import NodeType

if TYPE_CHECKING:
    from pathlib import Path

    from pivot.registry import RegistryStageInfo
    from pivot.storage.track import PvtData

__all__ = [
    "artifact_node",
    "stage_node",
    "build_graph",
    "build_tracked_trie",
    "get_consumers",
    "get_producer",
    "get_watch_paths",
    "get_downstream_stages",
    "get_upstream_stages",
    "get_stage_dag",
    "update_stage",
    "get_artifact_consumers",
    "get_execution_order",
    "GraphView",
    "extract_graph_view",
]


class GraphView(TypedDict):
    """Pre-extracted graph data for rendering.

    Decouples renderers from the internal bipartite graph representation.
    All node identifiers are plain strings (stage names, artifact paths)
    with no encoding prefixes.

    Edge direction is data-flow: producer -> consumer / input -> output.
    """

    stages: list[str]
    artifacts: list[str]
    stage_edges: list[tuple[str, str]]
    artifact_edges: list[tuple[str, str]]


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


def _build_outputs_map(stages: dict[str, RegistryStageInfo]) -> dict[str, str]:
    """Build mapping from output path to stage name.

    Returns:
        Dict of output_path -> stage_name

    Note:
        All paths are already normalized (absolute) by registry.py,
        so simple dict lookup is sufficient.
    """
    return {
        out_path: stage_name
        for stage_name, stage_info in stages.items()
        for out_path in stage_info["outs_paths"]
    }


def _build_outputs_trie(stages: dict[str, RegistryStageInfo]) -> pygtrie.Trie[tuple[str, str]]:
    """Build trie of output paths for directory dependency resolution."""
    trie: pygtrie.Trie[tuple[str, str]] = pygtrie.Trie()
    for stage_name, stage_info in stages.items():
        for out_path in stage_info["outs_paths"]:
            out_key = pathlib.Path(out_path).parts
            trie[out_key] = (stage_name, out_path)
    return trie


def _find_producers_for_path_with_artifacts(
    dep_path: str, outputs_trie: pygtrie.Trie[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Find stages and their output paths overlapping the dependency path.

    Returns:
        List of (stage_name, output_path) tuples for producers with outputs
        that overlap the dependency path (either as parent or child).
    """
    dep_key = pathlib.Path(dep_path).parts
    results = list[tuple[str, str]]()
    seen_stages = set[str]()

    # Case 1: Dependency is parent of outputs (dir depends on files inside)
    if outputs_trie.has_subtrie(dep_key):
        for stage_name, out_path in outputs_trie.values(prefix=dep_key):
            if stage_name not in seen_stages:
                results.append((stage_name, out_path))
                seen_stages.add(stage_name)

    # Case 2: Dependency is child of output (file depends on parent dir)
    prefix_item = outputs_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        stage_name, out_path = prefix_item.value
        if stage_name not in seen_stages:
            results.append((stage_name, out_path))

    return results


def _check_acyclic(g: nx.DiGraph[str]) -> None:
    """Check graph for cycles, raise if found."""
    from pivot import exceptions

    try:
        cycle = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return

    # Extract stage names from cycle for error message
    # Cycle is a list of (from_node, to_node, direction) tuples
    stages_in_cycle = list[str]()
    for from_node, _, _ in cycle:
        node_type, name = parse_node(from_node)
        if node_type == NodeType.STAGE and name not in stages_in_cycle:
            stages_in_cycle.append(name)

    if not stages_in_cycle:
        # Fallback if cycle is artifact-only (shouldn't happen in valid graph)
        # Extract readable names from the cycle edges
        nodes_in_cycle = list[str]()
        for from_node, _, _ in cycle:
            _, from_name = parse_node(from_node)
            if from_name not in nodes_in_cycle:
                nodes_in_cycle.append(from_name)
        stages_in_cycle = nodes_in_cycle if nodes_in_cycle else ["<unknown>"]

    raise exceptions.CyclicGraphError(
        f"Circular dependency detected: {' -> '.join(stages_in_cycle)}"
    )


def build_tracked_trie(tracked_files: dict[str, PvtData]) -> pygtrie.Trie[str]:
    """Build trie of tracked file paths for dependency checking.

    Keys are path tuples (from Path.parts), values are the absolute path string.
    """
    trie: pygtrie.Trie[str] = pygtrie.Trie()
    for abs_path in tracked_files:
        path_key = pathlib.Path(abs_path).parts
        trie[path_key] = abs_path
    return trie


def _is_tracked_path(dep: str, tracked_trie: pygtrie.Trie[str]) -> bool:
    """Check if dependency is a tracked file (exact match or inside tracked directory)."""
    dep_key = pathlib.Path(dep).parts

    # Exact match
    if dep_key in tracked_trie:
        return True

    # Dependency is inside a tracked directory
    prefix_item = tracked_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        return True

    # Dependency is a directory containing tracked files
    return tracked_trie.has_subtrie(dep_key)


def build_graph(
    stages: dict[str, RegistryStageInfo],
    validate: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
) -> nx.DiGraph[str]:
    """Build bipartite artifact-stage graph from stage definitions.

    Args:
        stages: Dict mapping stage name to RegistryStageInfo.
        validate: If True, validate that all dependencies exist.
        tracked_files: Dict of tracked file paths -> PvtData (from .pvt files).
            If provided, tracked files are recognized as valid dependency sources.

    Returns:
        Directed graph where:
        - Nodes are either artifacts (files) or stages (functions)
        - Edges go: artifact -> stage (consumed by) and stage -> artifact (produces)

    Raises:
        CyclicGraphError: If graph contains cycles (always checked)
        DependencyNotFoundError: If dependency doesn't exist (when validate=True)
    """
    from pivot import exceptions

    g: nx.DiGraph[str] = nx.DiGraph()

    # Build lookup structures - outputs_trie needed for directory dependency edges
    outputs_map = _build_outputs_map(stages)
    outputs_trie = _build_outputs_trie(stages)
    tracked_trie = build_tracked_trie(tracked_files) if tracked_files else None

    for stage_name, info in stages.items():
        stage = stage_node(stage_name)
        g.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact -> stage
        for dep_path in info["deps_paths"]:
            artifact = artifact_node(pathlib.Path(dep_path))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(artifact, stage)

            # Check for direct producer via exact match
            producer = outputs_map.get(dep_path)
            if producer:
                continue

            # Check for directory dependency via trie - add edges from output files
            # inside the directory to ensure proper stage DAG extraction
            producers_info = _find_producers_for_path_with_artifacts(dep_path, outputs_trie)
            if producers_info:
                for _, out_path in producers_info:
                    out_artifact = artifact_node(pathlib.Path(out_path))
                    # Ensure output artifact node exists and add edge to consuming stage
                    if out_artifact not in g:
                        g.add_node(out_artifact, type=NodeType.ARTIFACT)
                    g.add_edge(out_artifact, stage)
                continue

            # Validation: check dependency source exists (only when validate=True)
            if validate:
                # Check if exists on disk
                if pathlib.Path(dep_path).exists():
                    continue
                # Check if tracked file
                if tracked_trie and _is_tracked_path(dep_path, tracked_trie):
                    continue
                # Dependency not found
                raise exceptions.DependencyNotFoundError(
                    stage=stage_name,
                    dep=dep_path,
                    available_outputs=list(outputs_map.keys()),
                )

        # Outs: stage -> artifact
        for out in info["outs"]:
            artifact = artifact_node(pathlib.Path(str(out.path)))
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(stage, artifact)

    # Always check for cycles - a cyclic graph is never valid
    _check_acyclic(g)

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


def get_upstream_stages(g: nx.DiGraph[str], stage_name: str) -> list[str]:
    """Get stages whose outputs are consumed by this stage."""
    node = stage_node(stage_name)
    if node not in g:
        return []

    upstream = list[str]()
    for artifact in g.predecessors(node):
        if g.nodes[artifact]["type"] != NodeType.ARTIFACT:
            continue
        for producer in g.predecessors(artifact):
            if g.nodes[producer]["type"] == NodeType.STAGE:
                upstream.append(parse_node(producer)[1])
    return upstream


def get_stage_dag(g: nx.DiGraph[str]) -> nx.DiGraph[str]:
    """Extract stage-only DAG from bipartite graph.

    Returns a DAG with edges from consumer to producer. This allows
    get_execution_order() to work correctly with dfs_postorder_nodes traversal.
    """
    stage_dag: nx.DiGraph[str] = nx.DiGraph()

    for node in g.nodes():
        if g.nodes[node]["type"] == NodeType.STAGE:
            stage_name = parse_node(node)[1]
            stage_dag.add_node(stage_name)

    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.STAGE:
            continue
        stage_name = parse_node(node)[1]

        for artifact in g.successors(node):
            if g.nodes[artifact]["type"] != NodeType.ARTIFACT:
                continue
            for consumer in g.successors(artifact):
                if g.nodes[consumer]["type"] == NodeType.STAGE:
                    consumer_name = parse_node(consumer)[1]
                    # Edge from consumer to producer (for DFS postorder execution)
                    stage_dag.add_edge(consumer_name, stage_name)

    return stage_dag


def extract_graph_view(g: nx.DiGraph[str]) -> GraphView:
    """Extract a renderer-friendly view from the bipartite graph.

    Walks the bipartite graph, collecting stage names, artifact paths,
    and derived edges without exposing the internal node encoding.

    Edge semantics (data-flow direction):
    - stage_edges: (producer_stage, consumer_stage)
    - artifact_edges: (input_artifact, output_artifact)

    Args:
        g: Bipartite artifact-stage graph from build_graph().

    Returns:
        GraphView with plain-string nodes and edges.
    """
    stages = list[str]()
    artifacts = list[str]()
    stage_edges_set = set[tuple[str, str]]()
    artifact_edges_set = set[tuple[str, str]]()

    # Collect nodes by type
    for node in g.nodes():
        node_type, value = parse_node(node)
        if node_type == NodeType.STAGE:
            stages.append(value)
        else:
            artifacts.append(value)

    # Derive stage-to-stage edges (producer -> consumer)
    # Walk: stage -> artifact (produces) -> stage (consumes)
    # Use set to deduplicate edges (directory deps can create multiple paths)
    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.STAGE:
            continue
        _, producer_name = parse_node(node)
        for art_succ in g.successors(node):
            if g.nodes[art_succ]["type"] != NodeType.ARTIFACT:
                continue
            for consumer_node in g.successors(art_succ):
                if g.nodes[consumer_node]["type"] != NodeType.STAGE:
                    continue
                _, consumer_name = parse_node(consumer_node)
                stage_edges_set.add((producer_name, consumer_name))

    # Derive artifact-to-artifact edges (input -> output)
    # Walk: artifact -> stage (consumes) -> artifact (produces)
    # Use set to deduplicate edges (multiple stages can create same artifact flow)
    for node in g.nodes():
        if g.nodes[node]["type"] != NodeType.ARTIFACT:
            continue
        _, input_path = parse_node(node)
        for stage_succ in g.successors(node):
            if g.nodes[stage_succ]["type"] != NodeType.STAGE:
                continue
            for output_node in g.successors(stage_succ):
                if g.nodes[output_node]["type"] != NodeType.ARTIFACT:
                    continue
                _, output_path = parse_node(output_node)
                artifact_edges_set.add((input_path, output_path))

    return GraphView(
        stages=stages,
        artifacts=artifacts,
        stage_edges=sorted(stage_edges_set),
        artifact_edges=sorted(artifact_edges_set),
    )


def get_artifact_consumers(
    g: nx.DiGraph[str],
    path: Path,
    include_downstream: bool = True,
) -> list[str]:
    """Get all stages affected by a change to this artifact.

    Args:
        g: The bipartite graph.
        path: Path to the artifact.
        include_downstream: If True, include transitive dependents.

    Returns:
        Sorted list of stage names that would be affected (deterministic order).
    """
    direct = get_consumers(g, path)
    if not direct:
        return []

    if not include_downstream:
        return sorted(direct)

    all_affected = set(direct)
    for stage in direct:
        downstream = get_downstream_stages(g, stage)
        all_affected.update(downstream)

    return sorted(all_affected)


def get_execution_order(
    graph: nx.DiGraph[str],
    stages: list[str] | None = None,
    single_stage: bool = False,
) -> list[str]:
    """Get execution order using DFS postorder traversal.

    Args:
        graph: Stage-only DAG (from get_stage_dag)
        stages: Optional target stages to execute (default: all stages)
        single_stage: If True, run only the specified stages without dependencies.
            Stages are executed in the order provided, not DAG order.

    Returns:
        List of stage names in execution order (dependencies first, unless single_stage)
    """
    if stages:
        if single_stage:
            return stages
        subgraph = _get_subgraph(graph, stages)
        return list(nx.dfs_postorder_nodes(subgraph))

    return list(nx.dfs_postorder_nodes(graph))


def _get_subgraph(graph: nx.DiGraph[str], source_stages: list[str]) -> nx.DiGraph[str]:
    """Get subgraph containing sources and all their dependencies.

    Raises:
        StageNotFoundError: If any source stage is not in the graph.
    """
    from pivot import exceptions

    # Validate all stages exist before traversing
    graph_nodes = set(graph.nodes())
    unknown = [s for s in source_stages if s not in graph_nodes]
    if unknown:
        raise exceptions.StageNotFoundError(unknown, available_stages=list(graph_nodes))

    nodes = set[str]()
    for stage in source_stages:
        nodes.update(nx.dfs_postorder_nodes(graph, stage))
    return graph.subgraph(nodes)
