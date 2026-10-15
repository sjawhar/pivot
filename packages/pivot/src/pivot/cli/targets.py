from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Literal, TypedDict

import click

from pivot import discovery, names, project, types
from pivot.cli import helpers as cli_helpers

if TYPE_CHECKING:
    from collections.abc import Mapping

    import networkx as nx

    from pivot.registry import PipelineLike, RegistryStageInfo
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
    refs: list[types.ArtifactRef]


class IdentityTarget(TypedDict):
    """CLI target resolved to a stage identity."""

    kind: Literal["identity"]
    identity: types.ArtifactIdentity
    stage_name: str
    refs: list[types.ArtifactRef]


def parse_identity_target(target: str) -> types.ArtifactIdentity:
    """Parse a CLI target string into an ArtifactIdentity.

    Accepts ``"stage_name"`` or ``"stage_name:key"`` format.
    """
    return types.identity_from_key(target)


def format_identity_display(identity: types.ArtifactIdentity, path: str) -> str:
    """Format a stage output's identity and workspace path for human-facing CLI output."""
    return f"{types.identity_key(identity)} → {path}"


def resolve_cli_target(
    target: str,
    all_stages: Mapping[str, RegistryStageInfo],
) -> IdentityTarget | None:
    """Resolve a CLI target naming a stage output: ``stage`` or ``stage:key``.

    The stage may be bare or pipeline-qualified. Returns None when the target names no
    registered stage, so the caller can treat it as a file path.

    Raises:
        TargetValidationError: If the stage exists but has no output with the given key.
    """
    identity = parse_identity_target(target)
    producer = names.resolve_stage_name(identity.producer, all_stages)
    if producer not in all_stages:
        return None
    identity = types.ArtifactIdentity(producer=producer, key=identity.key)
    stage_outs = all_stages[producer]["outs"]
    if identity.key is None:
        refs = stage_outs
    else:
        refs = [out for out in stage_outs if out.identity.key == identity.key]
        if not refs:
            available = ", ".join(types.identity_key(out.identity) for out in stage_outs)
            raise TargetValidationError(
                f"Stage '{producer}' has no output key '{identity.key}'. "
                + f"Available: {available or '(none)'}"
            )
    return IdentityTarget(kind="identity", identity=identity, stage_name=producer, refs=refs)


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


_PIPELINE_FILENAMES = frozenset((discovery.PIPELINE_PY_NAME,))


def resolve_pipeline_file_targets(
    targets: list[str],
) -> tuple[set[str], list[str], list[PipelineLike]]:
    """Resolve targets that are paths to pipeline config files.

    For each target, checks if it's an existing file whose name matches
    pipeline.py. If so, loads the pipeline and extracts all stage names.

    Returns:
        Tuple of (resolved stage names, remaining unresolved targets, loaded pipelines).
    """
    resolved = set[str]()
    remaining = list[str]()
    pipelines: list[PipelineLike] = []

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
    _bipartite_graph: nx.DiGraph[str],
) -> tuple[set[str], list[str]]:
    """Resolve targets to stage names.

    Stage names are used directly. Artifact paths are resolved to the stages
    that produce them. File paths are resolved by matching against workspace
    output paths of all registered stages.

    Returns:
        Tuple of (resolved stage names, unresolved targets).
    """
    all_stages = cli_helpers.get_all_stages()
    result = set[str]()
    unresolved = list[str]()
    # Workspace path -> producing stage, built only when a target is not a stage
    stage_by_output_path: dict[str, str] | None = None

    for target in targets:
        resolved = resolve_cli_target(target, all_stages)
        if resolved is not None:
            result.add(resolved["stage_name"])
            continue
        if stage_by_output_path is None:
            identities = cli_helpers.require_workspace_store().output_identities_by_path(
                out for info in all_stages.values() for out in info["outs"]
            )
            stage_by_output_path = {
                path: identity.producer for path, identity in identities.items()
            }
        proj_root = project.get_project_root()
        norm_path = project.to_relative_path(project.normalize_path(target), proj_root)
        if norm_path in stage_by_output_path:
            result.add(stage_by_output_path[norm_path])
        else:
            unresolved.append(target)

    return result, unresolved


def _classify_targets(
    targets: list[str],
    proj_root: pathlib.Path,
) -> list[ResolvedTarget]:
    """Classify each target as stage output, file, both, or neither."""
    all_stages = cli_helpers.get_all_stages()
    results = list[ResolvedTarget]()

    for target in targets:
        stage_target = resolve_cli_target(target, all_stages)
        norm_path = project.to_relative_path(project.normalize_path(target), proj_root)
        is_file = (proj_root / norm_path).exists()

        if stage_target is not None and is_file:
            logger.warning(
                f"Target '{target}' matches both a stage name and a file path. "
                + f"Using stage '{stage_target['stage_name']}'. To use the file, specify a path like './{target}'."
            )

        results.append(
            ResolvedTarget(
                target=target if stage_target is None else stage_target["stage_name"],
                is_stage=stage_target is not None,
                is_file=is_file,
                norm_path=norm_path,
                refs=[] if stage_target is None else stage_target["refs"],
            )
        )

    return results


def resolve_output_paths(
    targets: list[str],
    proj_root: pathlib.Path,
    tag: types.ArtifactTag,
) -> tuple[set[str], list[str]]:
    """Resolve targets to the workspace paths of their outputs carrying ``tag``.

    Returns (resolved_paths, unknown_targets).
    """
    resolved = set[str]()
    missing = list[str]()

    for item in _classify_targets(targets, proj_root):
        if item["is_stage"]:
            store = cli_helpers.require_workspace_store()
            resolved.update(
                project.to_relative_path(store.resolve_display_path(out), proj_root)
                for out in item["refs"]
                if out.tag is tag
            )
        elif item["is_file"]:
            resolved.add(item["norm_path"])
        else:
            missing.append(item["target"])

    return resolved, missing


def resolve_plot_infos(
    targets: list[str],
    proj_root: pathlib.Path,
) -> tuple[list[plots_mod.PlotInfo], list[str]]:
    """Resolve targets to the plots to render.

    Returns (plot_list, unknown_targets).
    """
    from pivot.show import plots

    resolved = list[plots.PlotInfo]()
    missing = list[str]()

    for item in _classify_targets(targets, proj_root):
        if item["is_stage"]:
            store = cli_helpers.require_workspace_store()
            resolved.extend(
                plots.PlotInfo(
                    path=project.to_relative_path(store.resolve_display_path(out), proj_root),
                    stage_name=item["target"],
                )
                for out in item["refs"]
                if out.tag is types.ArtifactTag.PLOT
            )
        elif item["is_file"]:
            resolved.append(plots.PlotInfo(path=item["norm_path"], stage_name="(direct)"))
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
    tag: types.ArtifactTag,
) -> set[str] | None:
    """Validate targets and resolve to output paths.

    Returns None if no targets provided. Raises ClickException on errors.
    """
    if not targets:
        return None

    valid_targets = validate_targets(targets)
    if not valid_targets:
        return None

    paths, missing = resolve_output_paths(valid_targets, proj_root, tag)
    if missing:
        raise click.ClickException(_format_unknown_targets_error(missing))

    return paths
