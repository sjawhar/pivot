from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import click

from pivot import discovery, outputs, project, types
from pivot.cli import helpers as cli_helpers
from pivot.engine import graph as engine_graph

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import networkx as nx

    from pivot.pipeline.pipeline import Pipeline
    from pivot.registry import RegistryStageInfo
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


class IdentityTarget(TypedDict):
    """CLI target resolved to a stage identity."""

    kind: Literal["identity"]
    identity: types.ArtifactIdentity
    stage_name: str
    refs: list[types.ArtifactRef]


class PvtTarget(TypedDict):
    """CLI target resolved to a .pvt-tracked file."""

    kind: Literal["pvt"]
    path: str


def parse_identity_target(target: str) -> types.ArtifactIdentity:
    """Parse a CLI target string into an ArtifactIdentity.

    Accepts ``"stage_name"`` or ``"stage_name:key"`` format.
    """
    return types.identity_from_key(target)


def resolve_cli_target(
    target: str,
    all_stages: Mapping[str, RegistryStageInfo],
    pvt_exists: Callable[[str], bool],
) -> IdentityTarget | PvtTarget:
    """Resolve a CLI target to an identity or pvt target.

    Resolution order:
    1. Parse as identity (``"stage"`` or ``"stage:key"``)
    2. If producer is a registered stage, return IdentityTarget
    3. If target has a ``.pvt`` sidecar, return PvtTarget
    4. Otherwise raise TargetValidationError

    Args:
        target: CLI target string.
        all_stages: All registered stages by name.
        pvt_exists: Callable that checks if a path has a .pvt sidecar.

    Returns:
        IdentityTarget or PvtTarget.

    Raises:
        TargetValidationError: If target cannot be resolved.
    """
    identity = parse_identity_target(target)

    if identity.producer in all_stages:
        stage_info = all_stages[identity.producer]
        stage_outs: list[types.ArtifactRef] = stage_info["outs"]

        if identity.key is not None:
            # Filter to matching key
            matching = [out for out in stage_outs if out.identity.key == identity.key]
            if not matching:
                available_keys = [types.identity_key(out.identity) for out in stage_outs]
                available = ", ".join(available_keys) or "(none)"
                message = (
                    f"Stage '{identity.producer}' has no output key '{identity.key}'. "
                    + f"Available: {available}"
                )
                raise TargetValidationError(message)
            refs = matching
        else:
            refs = stage_outs

        return IdentityTarget(
            kind="identity",
            identity=identity,
            stage_name=identity.producer,
            refs=refs,
        )

    # Not a stage — check if .pvt tracked
    if pvt_exists(target):
        return PvtTarget(kind="pvt", path=target)

    raise TargetValidationError(
        f"Target '{target}' is neither a registered stage nor a .pvt-tracked file"
    )


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


_PIPELINE_FILENAMES = frozenset((*discovery.PIVOT_YAML_NAMES, discovery.PIPELINE_PY_NAME))


def resolve_pipeline_file_targets(
    targets: list[str],
) -> tuple[set[str], list[str], list[Pipeline]]:
    """Resolve targets that are paths to pipeline config files.

    For each target, checks if it's an existing file whose name matches
    pipeline.py, pivot.yaml, or pivot.yml. If so, loads the pipeline and
    extracts all stage names.

    Returns:
        Tuple of (resolved stage names, remaining unresolved targets, loaded pipelines).
    """
    resolved = set[str]()
    remaining = list[str]()
    pipelines: list[Pipeline] = []

    for target in targets:
        path = pathlib.Path(target)
        if path.is_file() and path.name in _PIPELINE_FILENAMES:
            pipeline = discovery.load_pipeline_from_path(path)
            if pipeline is not None:
                for name in pipeline.list_stages():
                    resolved.add(name)
                pipelines.append(pipeline)
            else:
                remaining.append(target)
        else:
            remaining.append(target)

    return resolved, remaining, pipelines


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
            identity = types.identity_from_key(target)
            if identity.producer in registered_stages:
                result.add(identity.producer)
                continue
            # Treat as artifact path - use absolute path to match graph node format
            # Only find the producer (for upstream-only semantics like stage targets)
            norm_path = project.normalize_path(target)
            identity = engine_graph.parse_artifact_identity(str(norm_path))
            producer = engine_graph.get_producer(bipartite_graph, identity)
            if producer:
                result.add(producer)
            else:
                unresolved.append(target)

    return result, unresolved


def _classify_targets(
    targets: list[str],
    proj_root: pathlib.Path,
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
    proj_root: pathlib.Path,
    output_type: type[outputs.Metric] | type[outputs.Plot[Any]],
) -> tuple[set[str], list[str]]:
    """Resolve targets to output file paths.

    Returns (resolved_paths, unknown_targets).
    """
    resolved = set[str]()
    missing = list[str]()

    store = cli_helpers.get_workspace_store()

    for item in _classify_targets(targets, proj_root):
        if item["is_stage"]:
            info = cli_helpers.get_stage(item["target"])
            stage_outs = cast("list[object]", info["outs"])
            for out in stage_outs:
                if isinstance(out, types.ArtifactRef):
                    if (
                        output_type is outputs.Metric
                        and out.tag is types.ArtifactTag.METRIC
                        or output_type is outputs.Plot
                        and out.tag is types.ArtifactTag.PLOT
                    ):
                        if store is None:
                            resolved.add(types.identity_key(out.identity))
                        else:
                            resolved.add(
                                project.to_relative_path(store.resolve_display_path(out), proj_root)
                            )
                elif isinstance(out, output_type):
                    # Registry always stores single-file outputs (multi-file are expanded)
                    expanded = outputs.require_expanded(
                        cast("outputs.Metric | outputs.Plot[Any]", out)
                    )
                    rel_path = project.to_relative_path(
                        project.normalize_path(expanded.path), proj_root
                    )
                    resolved.add(rel_path)
        elif item["is_file"]:
            resolved.add(item["norm_path"])
        else:
            missing.append(item["target"])

    return resolved, missing


def resolve_plot_infos(
    targets: list[str],
    proj_root: pathlib.Path,
) -> tuple[list[plots_mod.PlotInfo], list[str]]:
    """Resolve targets to PlotInfo entries with full metadata.

    Returns (plot_list, unknown_targets).
    """
    from pivot.show import plots

    resolved = list[plots.PlotInfo]()
    missing = list[str]()

    store = cli_helpers.get_workspace_store()

    for item in _classify_targets(targets, proj_root):
        if item["is_stage"]:
            info = cli_helpers.get_stage(item["target"])
            stage_outs = cast("list[object]", info["outs"])
            for out in stage_outs:
                if isinstance(out, types.ArtifactRef):
                    if out.tag is types.ArtifactTag.PLOT:
                        if store is None:
                            path = types.identity_key(out.identity)
                        else:
                            path = project.to_relative_path(
                                store.resolve_display_path(out), proj_root
                            )
                        resolved.append(
                            plots.PlotInfo(
                                path=path,
                                stage_name=item["target"],
                                x=None,
                                y=None,
                                template=None,
                            )
                        )
                elif isinstance(out, outputs.Plot):
                    # Registry always stores single-file outputs (multi-file are expanded)
                    expanded = outputs.require_expanded(cast("outputs.Plot[Any]", out))
                    resolved.append(
                        plots.PlotInfo(
                            path=project.to_relative_path(
                                project.normalize_path(expanded.path), proj_root
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
    proj_root: pathlib.Path,
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
