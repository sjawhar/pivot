from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

import click

from pivot import outputs, project
from pivot.cli import helpers as cli_helpers
from pivot.engine import graph as engine_graph

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

    from pivot.show import plots as plots_mod

logger = logging.getLogger(__name__)


class TargetValidationError(click.ClickException):
    """Raised when target validation fails."""


class ResolvedTarget(TypedDict):
    """Result of resolving a single target."""

    target: str
    is_stage: bool
    is_file: bool
    norm_path: str


def validate_targets(targets: tuple[str, ...]) -> list[str]:
    """Filter empty/whitespace-only targets; raise if all are invalid."""
    if not targets:
        return []

    valid = [t for t in targets if t.strip()]
    invalid = [t for t in targets if not t.strip()]

    if invalid:
        logger.warning(f"Ignoring {len(invalid)} empty/whitespace-only target(s)")

    if targets and not valid:
        raise TargetValidationError("All targets are empty or whitespace-only")

    return valid


def resolve_targets_to_stages(
    targets: list[str],
    bipartite_graph: nx.DiGraph[str],
) -> tuple[set[str], list[str]]:
    """Resolve targets to stage names.

    Stage names are used directly. Artifact paths are resolved to the stages
    that produce them.

    Returns:
        Tuple of (resolved stage names, unresolved targets).
    """
    registered_stages = set(cli_helpers.list_stages())
    result = set[str]()
    unresolved = list[str]()

    for target in targets:
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


def _classify_targets(
    targets: list[str],
    proj_root: Path,
) -> list[ResolvedTarget]:
    """Classify each target as stage, file, both, or neither."""
    registered_stages = set(cli_helpers.list_stages())
    results = list[ResolvedTarget]()

    for target in targets:
        is_stage = target in registered_stages
        norm_path = project.to_relative_path(project.normalize_path(target), proj_root)
        is_file = (proj_root / norm_path).exists()

        if is_stage and is_file:
            logger.warning(
                f"Target '{target}' matches both a stage name and a file path. "
                + f"Using stage '{target}'. To use the file, specify a path like './{target}'."
            )

        results.append(
            ResolvedTarget(
                target=target,
                is_stage=is_stage,
                is_file=is_file,
                norm_path=norm_path,
            )
        )

    return results


def resolve_output_paths(
    targets: list[str],
    proj_root: Path,
    output_type: type[outputs.Metric] | type[outputs.Plot[Any]],
) -> tuple[set[str], list[str]]:
    """Resolve targets to output file paths.

    Returns (resolved_paths, unknown_targets).
    """
    resolved = set[str]()
    missing = list[str]()

    for item in _classify_targets(targets, proj_root):
        if item["is_stage"]:
            info = cli_helpers.get_stage(item["target"])
            for out in info["outs"]:
                if isinstance(out, output_type):
                    # Registry always stores single-file outputs (multi-file are expanded)
                    rel_path = project.to_relative_path(project.normalize_path(out.path), proj_root)
                    resolved.add(rel_path)
        elif item["is_file"]:
            resolved.add(item["norm_path"])
        else:
            missing.append(item["target"])

    return resolved, missing


def resolve_plot_infos(
    targets: list[str],
    proj_root: Path,
) -> tuple[list[plots_mod.PlotInfo], list[str]]:
    """Resolve targets to PlotInfo entries with full metadata.

    Returns (plot_list, unknown_targets).
    """
    from pivot.show import plots

    resolved = list[plots.PlotInfo]()
    missing = list[str]()

    for item in _classify_targets(targets, proj_root):
        if item["is_stage"]:
            info = cli_helpers.get_stage(item["target"])
            for out in info["outs"]:
                if isinstance(out, outputs.Plot):
                    # Registry always stores single-file outputs (multi-file are expanded)
                    resolved.append(
                        plots.PlotInfo(
                            path=project.to_relative_path(
                                project.normalize_path(out.path), proj_root
                            ),
                            stage_name=item["target"],
                            x=out.x,
                            y=out.y,
                            template=out.template,
                        )
                    )
        elif item["is_file"]:
            resolved.append(
                plots.PlotInfo(
                    path=item["norm_path"],
                    stage_name="(direct)",
                    x=None,
                    y=None,
                    template=None,
                )
            )
        else:
            missing.append(item["target"])

    return resolved, missing


def _format_unknown_targets_error(missing: list[str]) -> str:
    """Format error message for targets that couldn't be resolved."""
    if len(missing) == 1:
        return f"Target '{missing[0]}' is neither a registered stage nor an existing file"
    targets_str = ", ".join(f"'{t}'" for t in missing)
    return f"Targets {targets_str} are neither registered stages nor existing files"


def resolve_and_validate(
    targets: tuple[str, ...],
    proj_root: Path,
    output_type: type[outputs.Metric] | type[outputs.Plot[Any]],
) -> set[str] | None:
    """Validate targets and resolve to output paths.

    Returns None if no targets provided. Raises ClickException on errors.
    """
    if not targets:
        return None

    valid_targets = validate_targets(targets)
    if not valid_targets:
        return None

    paths, missing = resolve_output_paths(valid_targets, proj_root, output_type)
    if missing:
        raise click.ClickException(_format_unknown_targets_error(missing))

    return paths
