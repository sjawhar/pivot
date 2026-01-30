from __future__ import annotations

import logging

import click
import networkx as nx

from pivot import dag, project, registry
from pivot.cli import decorators as cli_decorators
from pivot.cli import targets as cli_targets
from pivot.engine import graph as engine_graph
from pivot.engine import types as engine_types

logger = logging.getLogger(__name__)


def _resolve_targets_to_stages(
    target_list: list[str],
    bipartite_graph: nx.DiGraph[str],
) -> tuple[set[str], list[str]]:
    """Resolve targets to stage names.

    Stage names are used directly. Artifact paths are resolved to stages that
    produce or consume them.

    Returns:
        Tuple of (resolved stage names, unresolved targets).
    """
    registered_stages = set(registry.REGISTRY.list_stages())
    result = set[str]()
    unresolved = list[str]()

    for target in target_list:
        if target in registered_stages:
            result.add(target)
        else:
            # Treat as artifact path - use absolute path to match graph node format
            # Only find the producer (for upstream-only semantics like stage targets)
            norm_path = project.normalize_path(target)
            producer = engine_graph.get_producer(bipartite_graph, norm_path)
            if producer:
                result.add(producer)
            else:
                unresolved.append(target)

    return result, unresolved


def _get_upstream_subgraph(
    bipartite_graph: nx.DiGraph[str],
    stage_names: set[str],
) -> nx.DiGraph[str]:
    """Get subgraph containing specified stages and all their upstream dependencies.

    Uses the bipartite graph to find all ancestor nodes (both stages and artifacts)
    for the specified stages.
    """
    nodes_to_include = set[str]()

    for stage_name in stage_names:
        stage = engine_graph.stage_node(stage_name)
        if stage not in bipartite_graph:
            continue

        # Include this stage
        nodes_to_include.add(stage)

        # Include all ancestors (upstream stages and their artifacts)
        ancestors = nx.ancestors(bipartite_graph, stage)
        nodes_to_include.update(ancestors)

        # Include artifacts produced by this stage
        for succ in bipartite_graph.successors(stage):
            if bipartite_graph.nodes[succ]["type"] == engine_types.NodeType.ARTIFACT:
                nodes_to_include.add(succ)

    # Use subgraph view (no copy method needed)
    return bipartite_graph.subgraph(nodes_to_include)


@cli_decorators.pivot_command("dag")
@click.argument("targets", nargs=-1)
@click.option("--dot", "output_format", flag_value="dot", help="Output Graphviz DOT format")
@click.option("--mermaid", "output_format", flag_value="mermaid", help="Output Mermaid format")
@click.option("--md", "output_format", flag_value="md", help="Output Mermaid wrapped in markdown")
@click.option(
    "--stages", "show_stages", is_flag=True, help="Show stages as nodes (default: artifacts)"
)
def dag_cmd(
    targets: tuple[str, ...],
    output_format: str | None,
    show_stages: bool,
) -> None:
    """Visualize the pipeline DAG.

    Shows the dependency graph of artifacts (default) or stages. Without targets,
    shows the entire graph. With targets, shows the subgraph containing those
    targets and their upstream dependencies.

    TARGETS can be stage names or artifact paths. Stage names take precedence
    when a name matches both a stage and a file path.
    """
    # Build bipartite graph from registry
    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    bipartite_graph = engine_graph.build_graph(all_stages)

    # Filter to subgraph if targets provided
    if targets:
        valid_targets = cli_targets.validate_targets(targets)
        stage_names, unresolved = _resolve_targets_to_stages(valid_targets, bipartite_graph)

        if unresolved:
            unresolved_str = ", ".join(f"'{t}'" for t in unresolved)
            logger.warning(f"Targets not found in DAG (ignored): {unresolved_str}")

        if not stage_names:
            targets_str = ", ".join(f"'{t}'" for t in valid_targets)
            msg = (
                f"No stages found for targets: {targets_str}. "
                + "Check that targets are valid stage names or artifact paths."
            )
            raise click.ClickException(msg)
        bipartite_graph = _get_upstream_subgraph(bipartite_graph, stage_names)

    # Render the graph
    match output_format:
        case "dot":
            output = dag.render_dot(bipartite_graph, stages=show_stages)
        case "mermaid":
            output = dag.render_mermaid(bipartite_graph, stages=show_stages)
        case "md":
            mermaid = dag.render_mermaid(bipartite_graph, stages=show_stages)
            output = f"```mermaid\n{mermaid}\n```"
        case _:
            # Default: ASCII
            output = dag.render_ascii(bipartite_graph, stages=show_stages)

    click.echo(output)
