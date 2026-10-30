from __future__ import annotations

import logging

import click
import networkx as nx

from pivot import dag, types
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets
from pivot.engine import graph as engine_graph
from pivot.engine import types as engine_types

logger = logging.getLogger(__name__)


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


NOT_EXPANDED_LABEL = " (not yet expanded)"
"""Appended to the node of a mapped stage whose producer is not current, and to its outputs."""


def _label_not_expanded(
    view: engine_graph.GraphView, templates: set[str]
) -> engine_graph.GraphView:
    """``view`` with each of ``templates`` (``Pipeline.map``, not expanded) and each of its
    outputs labelled ``NOT_EXPANDED_LABEL``: such a node stands for instances not yet known."""
    if not templates:
        return view

    def stage(name: str) -> str:
        return name + NOT_EXPANDED_LABEL if name in templates else name

    def artifact(key: str) -> str:
        return (
            key + NOT_EXPANDED_LABEL if types.identity_from_key(key).producer in templates else key
        )

    return engine_graph.GraphView(
        stages=[stage(name) for name in view["stages"]],
        artifacts=[artifact(key) for key in view["artifacts"]],
        stage_edges=[(stage(a), stage(b)) for a, b in view["stage_edges"]],
        artifact_edges=[(artifact(a), artifact(b)) for a, b in view["artifact_edges"]],
    )


@cli_decorators.pivot_command("dag", allow_all=True)
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
    when a name matches both a stage and a file path. A stage mapped over a directory output
    shows as its instances when its producer is current, else as one node labelled
    "(not yet expanded)".
    """
    # Build bipartite graph from pipeline, its mapped stages expanded as a run would
    result, _ = cli_helpers.expand_mapped_stages()
    graph_stages = cli_helpers.get_graph_stages(result["expanded"])
    bipartite_graph = engine_graph.build_graph(graph_stages)

    # Filter to subgraph if targets provided
    if targets:
        valid_targets = cli_targets.validate_targets(targets)
        # A mapped stage's declared output path names no stage, as `pivot repro` refuses it
        cli_helpers.refuse_template_output_targets(valid_targets)
        stage_names, unresolved = cli_targets.resolve_targets_to_stages(
            cli_helpers.mapped_target_names(valid_targets, result),
            bipartite_graph,
            stages=graph_stages,
        )

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

    # Extract view for rendering
    view = _label_not_expanded(
        engine_graph.extract_graph_view(bipartite_graph), set(result["not_expanded"])
    )

    # Render the graph
    match output_format:
        case "dot":
            output = dag.render_dot(view, stages=show_stages)
        case "mermaid":
            output = dag.render_mermaid(view, stages=show_stages)
        case "md":
            mermaid = dag.render_mermaid(view, stages=show_stages)
            output = f"```mermaid\n{mermaid}\n```"
        case _:
            # Default: ASCII
            output = dag.render_ascii(view, stages=show_stages)

    click.echo(output)
