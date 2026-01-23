from __future__ import annotations

import asyncio
import logging
import pathlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from pivot import (
    config,
    dag,
    exceptions,
    explain,
    metrics,
    parameters,
    registry,
)
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, track
from pivot.storage import state as state_mod
from pivot.types import (
    CodeChange,
    DepChange,
    ParamChange,
    PipelineStatus,
    PipelineStatusInfo,
    RemoteSyncInfo,
    StageExplanation,
    TrackedFileInfo,
    TrackedFileStatus,
)

if TYPE_CHECKING:
    from networkx import DiGraph

logger = logging.getLogger(__name__)


def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
    """Get status for all stages, tracking upstream staleness."""
    with metrics.timed("status.get_pipeline_status"):
        graph = registry.REGISTRY.build_dag(validate=True)
        execution_order = dag.get_execution_order(graph, stages, single_stage=single_stage)

        if not execution_order:
            return [], graph

        state_dir = config.get_state_dir()
        overrides = parameters.load_params_yaml()

        # Compute explanations in parallel (I/O-bound: lock file reads, hashing)
        # ThreadPoolExecutor is appropriate since work is file I/O, not CPU
        max_workers = min(8, len(execution_order))
        explanations_by_name = dict[str, StageExplanation]()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = dict[Future[StageExplanation], str]()
            for stage_name in execution_order:
                stage_info = registry.REGISTRY.get(stage_name)
                future = pool.submit(
                    explain.get_stage_explanation,
                    stage_name,
                    stage_info["fingerprint"],
                    stage_info["deps_paths"],
                    stage_info["outs_paths"],
                    stage_info["params"],
                    overrides,
                    state_dir,
                )
                futures[future] = stage_name

            for future in as_completed(futures):
                stage_name = futures[future]
                try:
                    explanations_by_name[stage_name] = future.result()
                except Exception as e:
                    logger.warning(f"Failed to get explanation for {stage_name}: {e}")
                    explanations_by_name[stage_name] = StageExplanation(
                        stage_name=stage_name,
                        will_run=True,
                        is_forced=False,
                        reason=f"Error: {e}",
                        code_changes=list[CodeChange](),
                        param_changes=list[ParamChange](),
                        dep_changes=list[DepChange](),
                    )

        # Preserve original order for staleness propagation
        explanations = [explanations_by_name[name] for name in execution_order]

        return _compute_upstream_staleness(explanations, graph), graph


def _compute_upstream_staleness(
    explanations: list[StageExplanation],
    graph: DiGraph[str],
) -> list[PipelineStatusInfo]:
    """Process explanations and mark stages stale due to upstream dependencies."""
    stale_stages = set[str]()
    results = list[PipelineStatusInfo]()

    for exp in explanations:
        # DAG edges go from consumer -> producer, so successors() gives upstream (producer) stages
        upstream_stale = [
            succ for succ in graph.successors(exp["stage_name"]) if succ in stale_stages
        ]

        is_stale = exp["will_run"] or bool(upstream_stale)
        if is_stale:
            stale_stages.add(exp["stage_name"])

        if exp["will_run"]:
            reason = exp["reason"]
        elif upstream_stale:
            reason = f"Upstream stale ({', '.join(upstream_stale)})"
        else:
            reason = ""

        results.append(
            PipelineStatusInfo(
                name=exp["stage_name"],
                status=PipelineStatus.STALE if is_stale else PipelineStatus.CACHED,
                reason=reason,
                upstream_stale=upstream_stale,
            )
        )

    return results


def get_tracked_files_status(project_root: pathlib.Path) -> list[TrackedFileInfo]:
    """Get status for all tracked files."""
    tracked = track.discover_pvt_files(project_root)
    results = list[TrackedFileInfo]()

    for abs_path_str, track_data in sorted(tracked.items()):
        path = pathlib.Path(abs_path_str)
        rel_path = str(path.relative_to(project_root))

        try:
            if path.is_dir():
                current_hash, _ = cache.hash_directory(path)
            else:
                current_hash = cache.hash_file(path)
        except FileNotFoundError:
            results.append(
                TrackedFileInfo(
                    path=rel_path, status=TrackedFileStatus.MISSING, size=track_data["size"]
                )
            )
            continue

        results.append(
            TrackedFileInfo(
                path=rel_path,
                status=(
                    TrackedFileStatus.MODIFIED
                    if current_hash != track_data["hash"]
                    else TrackedFileStatus.CLEAN
                ),
                size=track_data["size"],
            )
        )

    return results


def get_remote_status(
    remote_name: str | None,
    cache_dir: pathlib.Path,
) -> RemoteSyncInfo:
    """Get remote sync status.

    Raises:
        RemoteNotConfiguredError: If no remotes are configured
        RemoteNotFoundError: If specified remote doesn't exist
        RemoteConnectionError: If connection to remote fails
    """
    remotes = remote_config.list_remotes()
    if not remotes:
        raise exceptions.RemoteNotConfiguredError("No remotes configured")

    s3_remote, resolved_name = transfer.create_remote_from_name(remote_name)
    url = remote_config.get_remote_url(resolved_name)
    local_hashes = transfer.get_local_cache_hashes(cache_dir)

    if not local_hashes:
        return RemoteSyncInfo(name=resolved_name, url=url, push_count=0, pull_count=0)

    with state_mod.StateDB(config.get_state_dir() / "state.db") as state_db:
        status = asyncio.run(
            transfer.compare_status(local_hashes, s3_remote, state_db, resolved_name)
        )

    return RemoteSyncInfo(
        name=resolved_name,
        url=url,
        push_count=len(status["local_only"]),
        pull_count=len(status["remote_only"]),
    )


def _pluralize(count: int, singular: str) -> str:
    """Return singular or plural form based on count."""
    return singular if count == 1 else f"{singular}s"


def get_suggestions(
    stale_count: int,
    modified_count: int,
    push_count: int,
    pull_count: int,
) -> list[str]:
    """Generate actionable suggestions based on current status."""
    suggestions = list[str]()

    if stale_count > 0:
        suggestions.append(
            f"Run `pivot run` to execute {stale_count} stale {_pluralize(stale_count, 'stage')}"
        )

    if modified_count > 0:
        suggestions.append(
            f"Run `pivot track` to update {modified_count} modified {_pluralize(modified_count, 'file')}"
        )

    if push_count > 0:
        suggestions.append(
            f"Run `pivot push` to upload {push_count} {_pluralize(push_count, 'file')}"
        )

    if pull_count > 0:
        suggestions.append(
            f"Run `pivot pull` to download {pull_count} {_pluralize(pull_count, 'file')}"
        )

    return suggestions
