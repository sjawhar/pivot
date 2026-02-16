# pyright: reportImplicitRelativeImport=false, reportMissingModuleSource=false
"""Bipartite artifact-stage graph built on NetworkX."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, TypedDict, cast

import networkx as nx

from pivot import outputs
from pivot import types as pivot_types
from pivot.engine.types import NodeType

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo
    from pivot.storage.store import Store

__all__ = [
    "artifact_node",
    "parse_artifact_identity",
    "stage_node",
    "build_graph",
    "validate_dependency_sources",
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
    All node identifiers are plain strings (stage names, artifact identities)
    with no encoding prefixes.

    Edge direction is data-flow: producer -> consumer / input -> output.
    """

    stages: list[str]
    artifacts: list[str]
    stage_edges: list[tuple[str, str]]
    artifact_edges: list[tuple[str, str]]


def _coerce_identity(
    identity: pivot_types.ArtifactIdentity | pathlib.Path | str,
) -> pivot_types.ArtifactIdentity:
    if isinstance(identity, pivot_types.ArtifactIdentity):
        return identity
    if isinstance(identity, pathlib.Path):
        return _path_identity(str(identity))
    return _path_identity(identity)


def artifact_node(identity: pivot_types.ArtifactIdentity | pathlib.Path | str) -> str:
    """Create artifact node ID from identity."""
    resolved = _coerce_identity(identity)
    return f"artifact:{pivot_types.identity_key(resolved)}"


def stage_node(name: str) -> str:
    """Create stage node ID from name."""
    return f"stage:{name}"


def parse_node(node: str) -> tuple[NodeType, str]:
    """Extract NodeType and value from node ID.

    Splits on the first colon to preserve identity strings with colons.
    """
    prefix, value = node.split(":", 1)
    return NodeType(prefix), value


def parse_artifact_identity(value: str) -> pivot_types.ArtifactIdentity:
    if ":" not in value:
        return pivot_types.ArtifactIdentity(value, None)
    producer, key = value.split(":", 1)
    return pivot_types.ArtifactIdentity(producer, key)


def _path_identity(path: str) -> pivot_types.ArtifactIdentity:
    return pivot_types.ArtifactIdentity(path, None)


def _output_identities(
    out: pivot_types.ArtifactRef | outputs.BaseOut,
) -> list[pivot_types.ArtifactIdentity]:
    if isinstance(out, pivot_types.ArtifactRef):
        return [out.identity]
    path = out.path
    if isinstance(path, (list, tuple)):
        return [_path_identity(str(item)) for item in path]
    return [_path_identity(str(path))]


def _dep_identity(dep: pivot_types.ArtifactRef | str) -> pivot_types.ArtifactIdentity:
    if isinstance(dep, pivot_types.ArtifactRef):
        return dep.identity
    return _path_identity(dep)


def _build_outputs_map(
    stages: dict[str, RegistryStageInfo],
) -> dict[pivot_types.ArtifactIdentity, str]:
    """Build mapping from output identity to stage name."""
    outputs_map = dict[pivot_types.ArtifactIdentity, str]()
    for stage_name, stage_info in stages.items():
        for out in stage_info["outs"]:
            for identity in _output_identities(out):
                outputs_map[identity] = stage_name
    return outputs_map


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


def validate_dependency_sources(
    stages: dict[str, RegistryStageInfo],
    store: Store | None = None,
) -> None:
    """Check that every dependency is either produced by a stage or exists on disk.

    Raises :class:`~pivot.exceptions.DependencyNotFoundError` when a dep is
    unresolvable.  With *store=None*, only structural checks run (wrong output
    key on a known stage); external-dep existence is silently skipped.
    """
    from pivot import exceptions

    outputs_map = _build_outputs_map(stages)
    stage_names = set(stages.keys())

    for stage_name, info in stages.items():
        for dep in info["deps"].values():
            dep_identity = _dep_identity(dep)
            if dep_identity in outputs_map:
                continue
            if dep_identity.producer in stage_names:
                raise exceptions.DependencyNotFoundError(
                    stage=stage_name,
                    dep=pivot_types.identity_key(dep_identity),
                    available_outputs=[pivot_types.identity_key(out) for out in outputs_map],
                )
            if store is None:
                continue
            if store.exists(dep):
                continue
            raise exceptions.DependencyNotFoundError(
                stage=stage_name,
                dep=pivot_types.identity_key(dep_identity),
                available_outputs=[pivot_types.identity_key(out) for out in outputs_map],
            )


def build_graph(
    stages: dict[str, RegistryStageInfo],
) -> nx.DiGraph[str]:
    """Build bipartite artifact-stage graph from stage definitions.

    Args:
        stages: Dict mapping stage name to RegistryStageInfo.
    Returns:
        Directed graph where:
        - Nodes are either artifacts (identities) or stages (functions)
        - Edges go: artifact -> stage (consumed by) and stage -> artifact (produces)

    Raises:
        CyclicGraphError: If graph contains cycles (always checked)
    """
    g: nx.DiGraph[str] = nx.DiGraph()

    for stage_name, info in stages.items():
        stage = stage_node(stage_name)
        g.add_node(stage, type=NodeType.STAGE)

        # Deps: artifact -> stage
        for dep in info["deps"].values():
            dep_identity = _dep_identity(dep)
            artifact = artifact_node(dep_identity)
            g.add_node(artifact, type=NodeType.ARTIFACT)
            g.add_edge(artifact, stage)

        # Outs: stage -> artifact
        for out in info["outs"]:
            for identity in _output_identities(out):
                artifact = artifact_node(identity)
                g.add_node(artifact, type=NodeType.ARTIFACT)
                g.add_edge(stage, artifact)

    # Always check for cycles - a cyclic graph is never valid
    _check_acyclic(g)

    return g


def get_consumers(g: nx.DiGraph[str], identity: pivot_types.ArtifactIdentity) -> list[str]:
    """Get stages that depend on this artifact.

    Args:
        g: The bipartite graph.
        identity: Artifact identity.

    Returns:
        List of stage names that consume this artifact.
    """
    node = artifact_node(identity)
    if node not in g:
        return []
    return [parse_node(n)[1] for n in g.successors(node) if g.nodes[n]["type"] == NodeType.STAGE]


def get_producer(g: nx.DiGraph[str], identity: pivot_types.ArtifactIdentity) -> str | None:
    """Get the stage that produces this artifact.

    Args:
        g: The bipartite graph.
        identity: Artifact identity.

    Returns:
        Stage name that produces this artifact, or None if it's an input.
    """
    node = artifact_node(identity)
    if node not in g:
        return None
    for pred in g.predecessors(node):
        if g.nodes[pred]["type"] == NodeType.STAGE:
            return parse_node(pred)[1]
    return None


def get_watch_paths(g: nx.DiGraph[str]) -> list[str]:
    """Return watch paths (identity-based artifacts require Store resolution).

    TODO: Resolve ArtifactIdentity to filesystem paths via Store (Task 4).
    """
    _ = g
    return []


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
        parse_artifact_identity(parse_node(n)[1])
        for n in g.predecessors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }
    current_outs = {
        parse_artifact_identity(parse_node(n)[1])
        for n in g.successors(stage)
        if g.nodes[n]["type"] == NodeType.ARTIFACT
    }

    # Get new deps and outs from info
    new_deps = {ref.identity for ref in new_info["deps"].values()}
    new_outs = {ref.identity for ref in new_info["outs"]}

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

    Walks the bipartite graph, collecting stage names, artifact identities,
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
    # Use set to deduplicate edges
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
        stages=sorted(stages),
        artifacts=sorted(artifacts),
        stage_edges=sorted(stage_edges_set),
        artifact_edges=sorted(artifact_edges_set),
    )


def get_artifact_consumers(
    g: nx.DiGraph[str],
    identity: pivot_types.ArtifactIdentity,
    include_downstream: bool = True,
) -> list[str]:
    """Get all stages affected by a change to this artifact.

    Args:
        g: The bipartite graph.
        identity: Artifact identity.
        include_downstream: If True, include transitive dependents.

    Returns:
        Sorted list of stage names that would be affected (deterministic order).
    """
    direct = get_consumers(g, identity)
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
    return cast("nx.DiGraph[str]", graph.subgraph(nodes))
