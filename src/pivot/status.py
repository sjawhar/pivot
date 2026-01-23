from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import TYPE_CHECKING, Any

import pydantic

from pivot import (
    dag,
    exceptions,
    explain,
    parameters,
    project,
    registry,
)
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, lock, track
from pivot.storage import state as state_mod
from pivot.types import (
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


def _can_skip_via_generation(
    stage_name: str,
    fingerprint: dict[str, str],
    deps: list[str],
    current_params: dict[str, Any],
    cache_dir: pathlib.Path,
) -> bool:
    """Check if stage can skip using O(1) generation tracking.

    This mirrors the logic in executor/worker.py to ensure status is consistent with run.
    """
    stage_lock = lock.StageLock(stage_name, lock.get_stages_dir(cache_dir))
    lock_data = stage_lock.read()
    if lock_data is None:
        return False

    if lock_data["code_manifest"] != fingerprint:
        return False
    if lock_data["params"] != current_params:
        return False

    with state_mod.StateDB(cache_dir, readonly=True) as state_db:
        recorded_gens = state_db.get_dep_generations(stage_name)
        if recorded_gens is None:
            return False

        dep_paths = [pathlib.Path(d) for d in deps]
        current_gens = state_db.get_many_generations(dep_paths)

        for dep in deps:
            path = pathlib.Path(dep)
            normalized = str(project.normalize_path(dep))
            current_gen = current_gens.get(path)
            if current_gen is None:
                return False
            if current_gen != recorded_gens.get(normalized):
                return False

    return True


def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
    """Get status for all stages, tracking upstream staleness."""
    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages, single_stage=single_stage)

    if not execution_order:
        return [], graph

    resolved_cache_dir = cache_dir or project.get_project_root() / ".pivot" / "cache"
    overrides = parameters.load_params_yaml()

    explanations = list[StageExplanation]()
    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)

        # First check generation tracking (O(1)) - this is what `pivot run` uses
        # to decide whether to skip a stage, so status should match
        can_skip = False
        try:
            current_params = parameters.get_effective_params(
                stage_info["params"], stage_name, overrides
            )
            can_skip = _can_skip_via_generation(
                stage_name,
                stage_info["fingerprint"],
                stage_info["deps_paths"],
                current_params,
                resolved_cache_dir,
            )
        except pydantic.ValidationError:
            # Invalid params - fall through to full explanation which will report the error
            pass

        if can_skip:
            # Stage would be skipped by run, so report it as cached
            explanations.append(
                StageExplanation(
                    stage_name=stage_name,
                    will_run=False,
                    is_forced=False,
                    reason="",
                    code_changes=[],
                    param_changes=[],
                    dep_changes=[],
                )
            )
        else:
            # Fall back to full hash-based explanation
            explanation = explain.get_stage_explanation(
                stage_name,
                stage_info["fingerprint"],
                stage_info["deps_paths"],
                stage_info["params"],
                overrides,
                resolved_cache_dir,
            )
            explanations.append(explanation)

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

    with state_mod.StateDB(cache_dir) as state_db:
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
