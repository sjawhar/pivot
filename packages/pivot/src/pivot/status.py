"""Pipeline status queries using the bipartite artifact-stage graph.

This module provides the primary query API for determining pipeline status.
It orchestrates calls to explain.py for individual stage explanations.

Graph Parameter
---------------
Query functions accept an optional `graph` parameter:

- If provided, uses the bipartite artifact-stage graph directly
- If None, builds a bipartite graph from all_stages

The graph is used for:
1. Computing execution order via get_stage_dag() + get_execution_order()
2. Artifact path-based queries
3. Ensuring consistency between queries and execution

Module Relationships
--------------------
- status.py: Orchestrates queries, handles graph, computes execution order
- explain.py: Provides per-stage explanations (receives data from status.py)
- cli/verify.py: Verification commands (uses status.py internally)

Example::

    from pivot.engine import graph as engine_graph
    from pivot import status

    # Build bipartite graph from Pipeline
    all_stages = {name: pipeline.get_stage(name) for name in pipeline.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    # Use graph for status query
    statuses, dag = status.get_pipeline_status(
        ["train"],
        single_stage=False,
        all_stages=all_stages,
        pipeline=pipeline,
        graph=graph,
    )
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportImplicitRelativeImport=false

from __future__ import annotations

import asyncio
import enum
import logging
import pathlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, cast

from pivot import (
    config,
    exceptions,
    explain,
    import_artifact,
    metrics,
    parameters,
    project,
    registry,
    types,
)
from pivot.engine import graph as engine_graph
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, lock, track
from pivot.storage import state as state_mod
from pivot.storage import store as store_mod
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
    from collections.abc import Callable

    import networkx as nx
    import pygtrie
    from networkx import DiGraph

    from pivot.registry import RegistryStageInfo
    from pivot.storage.track import PvtData

logger = logging.getLogger(__name__)


def _discover_tracked_files(
    allow_missing: bool,
) -> tuple[dict[str, PvtData] | None, pygtrie.Trie[str] | None]:
    """Discover tracked files for .pvt hash lookup when allow_missing is set."""
    if not allow_missing:
        return None, None

    tracked_files = track.discover_pvt_files(project.get_project_root())
    return tracked_files, None


def _get_explanations_in_parallel(
    execution_order: list[str],
    overrides: parameters.ParamsOverrides | None,
    all_stages: dict[str, RegistryStageInfo],
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
    store: store_mod.Store | None = None,
) -> dict[str, StageExplanation]:
    """Compute stage explanations in parallel (I/O-bound: lock file reads, hashing)."""
    default_state_dir = config.get_state_dir()
    max_workers = min(8, len(execution_order))
    explanations_by_name = dict[str, StageExplanation]()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = dict[Future[StageExplanation], str]()
        for stage_name in execution_order:
            stage_info = all_stages[stage_name]
            fingerprint = cast("dict[str, str]", stage_info["fingerprint"])
            stage_state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
            future = pool.submit(
                explain.get_stage_explanation,
                stage_name,
                fingerprint,
                [types.identity_key(dep.identity) for dep in stage_info["deps"].values()],
                [types.identity_key(out.identity) for out in stage_info["outs"]],
                stage_info["params"],
                overrides,
                stage_state_dir,
                force=force,
                allow_missing=allow_missing,
                tracked_files=tracked_files,
                tracked_trie=tracked_trie,
                deps_refs=stage_info["deps"],
                store=store,
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
                    upstream_stale=[],
                )

    return explanations_by_name


def _make_workspace_store(pipeline: registry.PipelineLike) -> store_mod.WorkspaceStore:
    return store_mod.WorkspaceStore(
        project_root=project.get_project_root(),
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )


def get_pipeline_explanations(
    stages: list[str] | None,
    single_stage: bool,
    all_stages: dict[str, RegistryStageInfo],
    pipeline: registry.PipelineLike,
    force: bool = False,
    allow_missing: bool = False,
    graph: nx.DiGraph[str] | None = None,
) -> list[StageExplanation]:
    _t = metrics.start()
    try:
        tracked_files, tracked_trie = _discover_tracked_files(allow_missing)
        if graph is None:
            graph = engine_graph.build_graph(all_stages)
        stage_graph = engine_graph.get_stage_dag(graph)
        execution_order = engine_graph.get_execution_order(
            stage_graph, stages, single_stage=single_stage
        )

        if not execution_order:
            return []

        for stage_name in execution_order:
            pipeline.ensure_fingerprint(stage_name)
        overrides = parameters.load_params_yaml()

        store = _make_workspace_store(pipeline)
        explanations_by_name = _get_explanations_in_parallel(
            execution_order,
            overrides,
            all_stages=all_stages,
            force=force,
            allow_missing=allow_missing,
            tracked_files=tracked_files,
            tracked_trie=tracked_trie,
            store=store,
        )

        # Preserve original order for staleness propagation
        explanations = [explanations_by_name[name] for name in execution_order]

        return _compute_explanations_with_upstream(explanations, stage_graph)
    finally:
        metrics.end("status.get_pipeline_explanations", _t)


def _compute_explanations_with_upstream(
    explanations: list[StageExplanation],
    graph: DiGraph[str],
) -> list[StageExplanation]:
    """Process explanations and add upstream_stale field for stages stale due to upstream."""
    stale_stages = set[str]()
    results = list[StageExplanation]()

    for exp in explanations:
        # DAG edges go from consumer -> producer, so successors() gives upstream (producer) stages
        upstream_stale = [
            succ for succ in graph.successors(exp["stage_name"]) if succ in stale_stages
        ]

        is_stale = exp["will_run"] or bool(upstream_stale)
        if is_stale:
            stale_stages.add(exp["stage_name"])

        # Compute updated reason
        reason = (
            exp["reason"]
            if exp["will_run"]
            else (f"Upstream stale ({', '.join(upstream_stale)})" if upstream_stale else "")
        )

        # Create new explanation with upstream_stale populated
        # Note: Explicit field copy is required for TypedDict type safety
        new_exp = StageExplanation(
            stage_name=exp["stage_name"],
            will_run=is_stale,
            is_forced=exp["is_forced"],
            reason=reason,
            code_changes=exp["code_changes"],
            param_changes=exp["param_changes"],
            dep_changes=exp["dep_changes"],
            upstream_stale=upstream_stale,
        )
        results.append(new_exp)

    return results


def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
    all_stages: dict[str, RegistryStageInfo],
    pipeline: registry.PipelineLike,
    allow_missing: bool = False,
    graph: nx.DiGraph[str] | None = None,
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
    _t = metrics.start()
    try:
        tracked_files, tracked_trie = _discover_tracked_files(allow_missing)
        if graph is None:
            graph = engine_graph.build_graph(all_stages)
        stage_graph = engine_graph.get_stage_dag(graph)
        execution_order = engine_graph.get_execution_order(
            stage_graph, stages, single_stage=single_stage
        )

        if not execution_order:
            return [], stage_graph

        for stage_name in execution_order:
            pipeline.ensure_fingerprint(stage_name)
        overrides = parameters.load_params_yaml()

        store = _make_workspace_store(pipeline)
        explanations_by_name = _get_explanations_in_parallel(
            execution_order,
            overrides,
            all_stages=all_stages,
            allow_missing=allow_missing,
            tracked_files=tracked_files,
            tracked_trie=tracked_trie,
            store=store,
        )

        # Preserve original order for staleness propagation
        explanations = [explanations_by_name[name] for name in execution_order]

        # Reuse the shared upstream computation logic
        enriched = _compute_explanations_with_upstream(explanations, stage_graph)
        return _explanations_to_status(enriched), stage_graph
    finally:
        metrics.end("status.get_pipeline_status", _t)


def _explanations_to_status(explanations: list[StageExplanation]) -> list[PipelineStatusInfo]:
    """Convert enriched explanations to PipelineStatusInfo list."""
    return [
        PipelineStatusInfo(
            name=exp["stage_name"],
            status=PipelineStatus.STALE if exp["will_run"] else PipelineStatus.CACHED,
            reason=exp["reason"],
            upstream_stale=exp["upstream_stale"],
        )
        for exp in explanations
    ]


def get_tracked_files_status(
    project_root: pathlib.Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[TrackedFileInfo]:
    """Get status for all tracked files.

    Args:
        project_root: Project root directory.
        on_progress: Optional callback called with (completed, total) after each file.
    """
    tracked = track.discover_pvt_files(project_root)
    total = len(tracked)
    results = list[TrackedFileInfo]()

    # Use state_db for hash caching (mtime-based)
    with state_mod.StateDB(config.get_state_db_path()) as state_db:
        for i, (abs_path_str, track_data) in enumerate(sorted(tracked.items()), 1):
            path = pathlib.Path(abs_path_str)
            rel_path = str(path.relative_to(project_root))

            try:
                if path.is_dir():
                    current_hash, _ = cache.hash_directory(path, state_db)
                else:
                    current_hash, _ = cache.hash_file(path, state_db)
            except FileNotFoundError:
                results.append(
                    TrackedFileInfo(
                        path=rel_path, status=TrackedFileStatus.MISSING, size=track_data["size"]
                    )
                )
                if on_progress is not None:
                    on_progress(i, total)
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
            if on_progress is not None:
                on_progress(i, total)

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

    with state_mod.StateDB(config.get_state_db_path()) as state_db:
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


def find_orphaned_lock_files(
    registered_stages: set[str],
    state_dir: pathlib.Path | None = None,
) -> list[str]:
    if state_dir is None:
        state_dir = project.get_project_root() / ".pivot"
    stages_dir = lock.get_stages_dir(state_dir)
    return lock.find_orphaned_locks(stages_dir, registered_stages)


def get_suggestions(
    stale_count: int,
    modified_count: int,
    push_count: int,
    pull_count: int,
    orphan_count: int = 0,
) -> list[str]:
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

    if orphan_count > 0:
        suggestions.append(
            f"{orphan_count} orphaned lock {_pluralize(orphan_count, 'file')} from "
            f"renamed/removed stages. Old outputs may still exist on disk."
        )

    return suggestions


def what_if_changed(
    paths: list[pathlib.Path],
    all_stages: dict[str, RegistryStageInfo],
    graph: nx.DiGraph[str] | None = None,
) -> list[str]:
    """Determine which stages would run if these paths changed.

    Args:
        paths: Paths that hypothetically changed (relative or absolute).
        all_stages: Dict mapping stage names to RegistryStageInfo.
        graph: Optional bipartite graph from Engine.

    Returns:
        List of stage names that would be affected.
    """
    if graph is None:
        graph = engine_graph.build_graph(all_stages)

    _ = graph, paths
    return sorted(all_stages.keys())


class ImportCheckStatus(enum.Enum):
    UP_TO_DATE = "up to date"
    UPDATE_AVAILABLE = "update available"
    ERROR = "error"


class ImportStatusInfo:
    __slots__: tuple[str, ...] = ("path", "status", "current_rev", "latest_rev", "error")

    path: str
    status: ImportCheckStatus
    current_rev: str
    latest_rev: str
    error: str

    def __init__(
        self,
        path: str,
        status: ImportCheckStatus,
        current_rev: str = "",
        latest_rev: str = "",
        error: str = "",
    ) -> None:
        self.path = path
        self.status = status
        self.current_rev = current_rev
        self.latest_rev = latest_rev
        self.error = error


async def _check_imports_batch(
    import_pvts: dict[str, track.PvtData],
    project_root: pathlib.Path,
) -> list[ImportStatusInfo]:
    tasks = list[tuple[str, track.PvtData]]()
    for data_path, pvt_data in sorted(import_pvts.items()):
        try:
            rel_path = str(pathlib.Path(data_path).relative_to(project_root))
        except ValueError:
            rel_path = data_path
        tasks.append((rel_path, pvt_data))

    checks = await asyncio.gather(
        *[import_artifact.check_for_update(pvt_data) for _, pvt_data in tasks],
        return_exceptions=True,
    )

    results = list[ImportStatusInfo]()
    for (rel_path, _), check_or_exc in zip(tasks, checks, strict=True):
        if isinstance(check_or_exc, Exception):
            logger.warning("Failed to check import %s: %s", rel_path, check_or_exc)
            results.append(
                ImportStatusInfo(
                    path=rel_path, status=ImportCheckStatus.ERROR, error=str(check_or_exc)
                )
            )
        else:
            check = cast("import_artifact.UpdateCheck", check_or_exc)
            if check["available"]:
                results.append(
                    ImportStatusInfo(
                        path=rel_path,
                        status=ImportCheckStatus.UPDATE_AVAILABLE,
                        current_rev=check["current_rev"][:8],
                        latest_rev=check["latest_rev"][:8],
                    )
                )
            else:
                results.append(ImportStatusInfo(path=rel_path, status=ImportCheckStatus.UP_TO_DATE))
    return results


def get_import_status(project_root: pathlib.Path) -> list[ImportStatusInfo]:
    """Check update status for all imported artifacts concurrently."""
    import_pvts = track.discover_import_pvt_files(project_root)
    if not import_pvts:
        return []
    return asyncio.run(_check_imports_batch(import_pvts, project_root))
