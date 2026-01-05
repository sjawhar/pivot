"""DAG construction and traversal for pipeline stages.

Builds a directed acyclic graph from registered stages to determine execution order.
Uses networkx for graph operations and DFS postorder traversal.
"""

from __future__ import annotations

import pathlib
from typing import Any, cast

import networkx as nx

from pivot.exceptions import CyclicGraphError, DependencyNotFoundError


def build_dag(stages: dict[str, dict[str, Any]], validate: bool = True) -> nx.DiGraph[str]:
    """Build DAG from registered stages.

    Args:
        stages: Dict of stage_name -> stage_info (from registry._stages)
        validate: If True, validate that all dependencies exist

    Returns:
        DiGraph with edges from consumer to producer

    Raises:
        CyclicGraphError: If graph contains cycles
        DependencyNotFoundError: If dependency doesn't exist (when validate=True)

    Example:
        >>> stages = {
        ...     'preprocess': {'deps': ['/abs/data.csv'], 'outs': ['/abs/clean.csv']},
        ...     'train': {'deps': ['/abs/clean.csv'], 'outs': ['/abs/model.pkl']}
        ... }
        >>> graph = build_dag(stages)
        >>> list(nx.dfs_postorder_nodes(graph))
        ['preprocess', 'train']
    """
    graph: nx.DiGraph[str] = nx.DiGraph()

    for stage_name, stage_info in stages.items():
        graph.add_node(stage_name, **stage_info)

    outputs_map = _build_outputs_map(stages)

    for stage_name, stage_info in stages.items():
        for dep in stage_info.get("deps", []):
            producer = outputs_map.get(dep)
            if producer:
                graph.add_edge(stage_name, producer)
            elif validate and not pathlib.Path(dep).exists():
                raise DependencyNotFoundError(
                    f"Stage '{stage_name}' depends on '{dep}' which is not produced by "
                    + "any stage and does not exist on disk"
                )

    _check_acyclic(graph)

    return graph


def _build_outputs_map(stages: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build mapping from output path to stage name.

    Returns:
        Dict of output_path -> stage_name

    Note:
        All paths are already normalized (absolute) by registry.py,
        so simple dict lookup is sufficient. Prefers outs_paths (list of str)
        but falls back to outs for backward compatibility with direct dict creation.
    """
    outputs_map = dict[str, str]()
    for stage_name, stage_info in stages.items():
        # Prefer outs_paths (from registry), fallback to outs (direct dict)
        outs = stage_info.get("outs_paths") or stage_info.get("outs", [])
        for out in outs:
            # Handle both string paths and BaseOut objects
            path = out.path if hasattr(out, "path") else out
            outputs_map[path] = stage_name
    return outputs_map


def _check_acyclic(graph: nx.DiGraph[str]) -> None:
    """Check graph for cycles, raise if found."""
    try:
        # networkx stubs don't fully type find_cycle's return value
        cycle = cast(
            "list[tuple[str, str, str]]",
            nx.find_cycle(graph, orientation="original"),  # pyright: ignore[reportUnknownMemberType]
        )
    except nx.NetworkXNoCycle:
        return

    stages_in_cycle = list[str]()
    for from_node, _to_node, _ in cycle:
        stages_in_cycle.append(from_node)
    if cycle:
        stages_in_cycle.append(cycle[-1][1])

    raise CyclicGraphError(f"Circular dependency detected: {' -> '.join(stages_in_cycle)}")


def get_execution_order(graph: nx.DiGraph[str], stages: list[str] | None = None) -> list[str]:
    """Get execution order using DFS postorder traversal.

    Args:
        graph: DAG of stages
        stages: Optional list of stages to execute (default: all)

    Returns:
        List of stage names in execution order (dependencies first)

    Example:
        >>> # For a simple chain A -> B -> C
        >>> get_execution_order(graph)
        ['A', 'B', 'C']
    """
    if stages:
        subgraph = _get_subgraph(graph, stages)
        return list(nx.dfs_postorder_nodes(subgraph))

    return list(nx.dfs_postorder_nodes(graph))


def _get_subgraph(graph: nx.DiGraph[str], source_stages: list[str]) -> nx.DiGraph[str]:
    """Get subgraph containing sources and all their dependencies."""
    nodes = set[str]()
    for stage in source_stages:
        nodes.update(nx.dfs_postorder_nodes(graph, stage))
    # subgraph() returns a SubGraph view that behaves like DiGraph at runtime
    return cast("nx.DiGraph[str]", graph.subgraph(nodes))


def get_parallel_groups(graph: nx.DiGraph[str], stages: list[str] | None = None) -> list[list[str]]:
    """Get stages grouped by parallel execution levels.

    Uses topological generations to identify stages that can run in parallel.
    Each group contains stages with no dependencies on each other.

    Args:
        graph: DAG of stages
        stages: Optional list of stages to execute (default: all)

    Returns:
        List of groups, where each group is a list of stage names that can run
        in parallel. Groups must be executed in order (group 0 before group 1, etc).

    Example:
        >>> # For pipeline: A -> B, A -> C, B -> D, C -> D
        >>> get_parallel_groups(graph)
        [['A'], ['B', 'C'], ['D']]  # B and C can run in parallel
    """
    subgraph = _get_subgraph(graph, stages) if stages else graph

    # Reverse the graph because our edges go consumer->producer
    # but topological_generations expects producer->consumer
    reversed_graph = subgraph.reverse(copy=False)

    # Get generations (each generation can run in parallel)
    generations = list(nx.topological_generations(reversed_graph))

    return [list(gen) for gen in generations]


def get_downstream_stages(graph: nx.DiGraph[str], stage: str) -> list[str]:
    """Get all stages that depend on given stage (directly or transitively).

    Uses reverse graph to traverse from stage to dependents.

    Args:
        graph: DAG of stages
        stage: Source stage name

    Returns:
        List of stage names that depend on the source stage (includes source itself)

    Example:
        >>> # If B depends on A, and C depends on B
        >>> get_downstream_stages(graph, 'A')
        ['A', 'B', 'C']
    """
    reversed_graph = graph.reverse(copy=False)
    return list(nx.dfs_postorder_nodes(reversed_graph, stage))
