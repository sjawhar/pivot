from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, cast

import networkx as nx
import pygtrie

from pivot.exceptions import CyclicGraphError, DependencyNotFoundError

if TYPE_CHECKING:
    from pivot.registry import RegistryStageInfo


def build_dag(stages: dict[str, RegistryStageInfo], validate: bool = True) -> nx.DiGraph[str]:
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
    outputs_trie = _build_outputs_trie(stages)

    for stage_name, stage_info in stages.items():
        for dep in stage_info["deps"]:
            producer = outputs_map.get(dep)
            if producer:
                graph.add_edge(stage_name, producer)
            else:
                # Check for directory dependency on file outputs (or vice versa)
                producers = _find_producers_for_path(dep, outputs_trie)
                for prod in producers:
                    graph.add_edge(stage_name, prod)

                if not producers and validate and not pathlib.Path(dep).exists():
                    raise DependencyNotFoundError(
                        f"Stage '{stage_name}' depends on '{dep}' which is not produced by "
                        + "any stage and does not exist on disk"
                    )

    _check_acyclic(graph)

    return graph


def _build_outputs_map(stages: dict[str, RegistryStageInfo]) -> dict[str, str]:
    """Build mapping from output path to stage name.

    Returns:
        Dict of output_path -> stage_name

    Note:
        All paths are already normalized (absolute) by registry.py,
        so simple dict lookup is sufficient. Prefers outs_paths (list of str)
        but falls back to outs for backward compatibility with direct dict creation.
    """
    outputs_map = {
        out_path: stage_name
        for stage_name, stage_info in stages.items()
        for out_path in stage_info["outs_paths"]
    }
    return outputs_map


def _build_outputs_trie(stages: dict[str, RegistryStageInfo]) -> pygtrie.Trie:
    """Build trie of output paths for directory dependency resolution."""
    trie = pygtrie.Trie()
    for stage_name, stage_info in stages.items():
        for out_path in stage_info["outs_paths"]:
            out_key = pathlib.Path(out_path).parts
            trie[out_key] = (stage_name, out_path)
    return trie


def _find_producers_for_path(dep_path: str, outputs_trie: pygtrie.Trie) -> list[str]:
    """Find stages with outputs overlapping the dependency path (parent or child)."""
    dep_key = pathlib.Path(dep_path).parts
    producers = list[str]()

    # Case 1: Dependency is parent of outputs (dir depends on files inside)
    if outputs_trie.has_subtrie(dep_key):
        for stage_name, _ in outputs_trie.values(prefix=dep_key):
            if stage_name not in producers:
                producers.append(stage_name)

    # Case 2: Dependency is child of output (file depends on parent dir)
    prefix_item = outputs_trie.shortest_prefix(dep_key)
    if prefix_item is not None and prefix_item.value is not None:
        stage_name, _ = prefix_item.value
        if stage_name not in producers:
            producers.append(stage_name)

    return producers


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

    stages_in_cycle = [from_node for from_node, _, _ in cycle] + [cycle[-1][1]]
    raise CyclicGraphError(f"Circular dependency detected: {' -> '.join(stages_in_cycle)}")


def get_execution_order(
    graph: nx.DiGraph[str],
    stages: list[str] | None = None,
    single_stage: bool = False,
) -> list[str]:
    """Get execution order using DFS postorder traversal.

    Args:
        graph: DAG of stages
        stages: Optional target stages to execute (default: all stages)
        single_stage: If True, run only the specified stages without dependencies.
            Stages are executed in the order provided, not DAG order.

    Returns:
        List of stage names in execution order (dependencies first, unless single_stage)

    Example:
        >>> # For a simple chain A -> B -> C
        >>> get_execution_order(graph)
        ['A', 'B', 'C']
        >>> get_execution_order(graph, ['C'])
        ['A', 'B', 'C']
        >>> get_execution_order(graph, ['C'], single_stage=True)
        ['C']
        >>> # With single_stage, order matches input order (not DAG order)
        >>> get_execution_order(graph, ['C', 'A'], single_stage=True)
        ['C', 'A']
    """
    if stages:
        if single_stage:
            return stages
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


def get_parallel_groups(  # pragma: no cover
    graph: nx.DiGraph[str], stages: list[str] | None = None
) -> list[list[str]]:
    """Get stages grouped by parallel execution levels (unused with greedy execution)."""
    subgraph = _get_subgraph(graph, stages) if stages else graph
    reversed_graph = subgraph.reverse(copy=False)
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
