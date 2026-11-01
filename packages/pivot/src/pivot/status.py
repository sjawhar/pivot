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

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import pathlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, cast

import pygtrie

from pivot import (
    config,
    display,
    exceptions,
    expansion,
    explain,
    import_artifact,
    metrics,
    parameters,
    project,
    registry,
)
from pivot import fingerprint as fingerprint_mod
from pivot import state_dir as state_dir_mod
from pivot.engine import graph as engine_graph
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import cache, listing, lock, member_skip, track
from pivot.storage import state as state_mod
from pivot.storage import store as store_mod
from pivot.types import (
    ArtifactIdentity,
    ArtifactTag,
    CodeChange,
    DepChange,
    FileHashEntry,
    ImportCheckStatus,
    ImportStatusInfo,
    ParamChange,
    PipelineStatus,
    PipelineStatusInfo,
    Recovery,
    RemoteSyncInfo,
    StageExplanation,
    TrackedFileInfo,
    TrackedFileStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping

    import networkx as nx
    from networkx import DiGraph

    from pivot.registry import RegistryStageInfo
    from pivot.storage.track import PvtData

logger = logging.getLogger(__name__)

UPSTREAM_STALE_REASON_PREFIX = "Upstream stale"
"""Reason prefix for a stage that is current on its own but has a stale upstream."""

NOT_EXPANDED_REASON_PREFIX = "Not yet expanded"
"""Reason prefix for a mapped stage (``Pipeline.map``) whose producer is not current."""


def _discover_tracked_files(
    allow_missing: bool, all_stages: dict[str, RegistryStageInfo]
) -> tuple[dict[str, PvtData] | None, pygtrie.Trie[str] | None]:
    """Discover tracked files for .pvt hash lookup when allow_missing is set. A stage's own
    ``*.pvt`` member of its output is none (``member_skip.own_member_skip``)."""
    if not allow_missing:
        return None, None

    project_root = project.get_project_root()
    tracked_files = track.discover_pvt_files(
        project_root,
        skip=member_skip.own_member_skip(
            project_root,
            all_stages,
            config.locate_state_dir(),
            config.locate_cache_dir() / "files",
        ),
    )
    tracked_trie: pygtrie.Trie[str] = pygtrie.Trie()
    for path in tracked_files:
        tracked_trie[pathlib.Path(path).parts] = path
    return tracked_files, tracked_trie


def _get_explanations_in_parallel(
    execution_order: list[str],
    overrides: parameters.ParamsOverrides | None,
    all_stages: dict[str, RegistryStageInfo],
    store: store_mod.WorkspaceStore,
    force: bool = False,
    allow_missing: bool = False,
    tracked_files: dict[str, PvtData] | None = None,
    tracked_trie: pygtrie.Trie[str] | None = None,
    *,
    known: Collection[str] = (),
) -> dict[str, StageExplanation]:
    """Compute stage explanations in parallel (I/O-bound: lock file reads, hashing), for each
    stage of ``execution_order`` but ``known``. A known stage's lock is still read: a stage that
    reads one file of its directory output finds that file's hash in it."""
    default_state_dir = config.locate_state_dir()
    pending = [name for name in execution_order if name not in known]
    max_workers = min(8, len(pending))
    locks = explain.read_locks(execution_order, all_stages, default_state_dir, store.cache_dir)
    unreadable = dict[ArtifactIdentity, str]()
    producer_hashes = explain.producer_output_hashes(
        locks,
        (ref for name in pending for ref in all_stages[name]["deps"].values()),
        {name: explain.stages_dir(all_stages[name], default_state_dir) for name in locks},
        store.cache_dir,
        unreadable,
    )
    explanations_by_name = dict[str, StageExplanation]()
    entries_per_stage = list[tuple[pathlib.Path, list[FileHashEntry]]]()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = dict[Future[StageExplanation], str]()
        for stage_name in pending:
            stage_info = all_stages[stage_name]
            fingerprint = cast("dict[str, str]", stage_info["fingerprint"])
            stage_state_dir = registry.get_stage_state_dir(stage_info, default_state_dir)
            entries = list[FileHashEntry]()
            entries_per_stage.append((stage_state_dir, entries))
            future = pool.submit(
                contextvars.copy_context().run,
                explain.get_stage_explanation,
                stage_name,
                fingerprint,
                stage_info["deps"],
                list(stage_info["outs"]),
                stage_info["params"],
                overrides,
                stage_state_dir,
                store,
                lock_data=locks[stage_name],
                force=force,
                allow_missing=allow_missing,
                tracked_files=tracked_files,
                tracked_trie=tracked_trie,
                producer_hashes=producer_hashes,
                file_hash_entries=entries,
                unreadable_listings=unreadable,
            )
            futures[future] = stage_name

        for future in as_completed(futures):
            stage_name = futures[future]
            try:
                explanations_by_name[stage_name] = future.result()
            except Exception as e:
                message = exceptions.describe_error(e)
                logger.warning(f"Failed to get explanation for {stage_name}: {message}")
                explanations_by_name[stage_name] = StageExplanation(
                    stage_name=stage_name,
                    will_run=True,
                    is_forced=False,
                    reason=f"Error: {message}",
                    code_changes=list[CodeChange](),
                    param_changes=list[ParamChange](),
                    dep_changes=list[DepChange](),
                    upstream_stale=[],
                )

    entries_by_state_dir = dict[pathlib.Path, list[FileHashEntry]]()
    for state_dir, entries in entries_per_stage:
        entries_by_state_dir.setdefault(state_dir, []).extend(entries)
    for state_dir, entries in entries_by_state_dir.items():
        state_mod.save_file_hashes(state_dir, entries, create=False)
    return explanations_by_name


def _make_workspace_store(pipeline: registry.PipelineLike) -> store_mod.WorkspaceStore:
    return store_mod.WorkspaceStore(
        project_root=project.get_project_root(),
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
        cache_dir=config.locate_cache_dir() / "files",
    )


def _explanations_for(
    execution_order: list[str],
    all_stages: dict[str, RegistryStageInfo],
    pipeline: registry.PipelineLike,
    *,
    force: bool,
    allow_missing: bool,
    known: Mapping[str, StageExplanation] | None,
) -> dict[str, StageExplanation]:
    """Each stage of ``execution_order``'s own explanation, before its upstream is counted
    (``_compute_explanations_with_upstream``): ``known``'s where it holds one, computed with the
    same ``force`` and ``allow_missing``, else read now."""
    tracked_files, tracked_trie = _discover_tracked_files(allow_missing, all_stages)
    with fingerprint_mod.fingerprint_pass():
        for stage_name in execution_order:
            pipeline.ensure_fingerprint(stage_name)
    overrides = parameters.load_params_yaml(pipeline)
    reused = {name: known[name] for name in execution_order if known is not None and name in known}
    pending = [name for name in execution_order if name not in reused]
    if not pending:
        return reused
    return {
        **reused,
        **_get_explanations_in_parallel(
            execution_order,
            overrides,
            all_stages=all_stages,
            force=force,
            allow_missing=allow_missing,
            tracked_files=tracked_files,
            tracked_trie=tracked_trie,
            store=_make_workspace_store(pipeline),
            known=reused.keys(),
        ),
    }


def get_pipeline_explanations(
    stages: list[str] | None,
    single_stage: bool,
    all_stages: dict[str, RegistryStageInfo],
    pipeline: registry.PipelineLike,
    force: bool = False,
    allow_missing: bool = False,
    graph: nx.DiGraph[str] | None = None,
    *,
    known: Mapping[str, StageExplanation] | None = None,
) -> list[StageExplanation]:
    """Explain each stage of ``stages``' execution order. ``known`` holds explanations already
    computed with the same ``force`` and ``allow_missing`` (``expand_current``'s)."""
    _t = metrics.start()
    try:
        with state_dir_mod.entry_invocation(pipeline.resolved_state_dirs):
            if graph is None:
                graph = engine_graph.build_graph(all_stages)
            stage_graph = engine_graph.get_stage_dag(graph)
            execution_order = engine_graph.get_execution_order(
                stage_graph, stages, single_stage=single_stage
            )

            if not execution_order:
                return []

            explanations_by_name = _explanations_for(
                execution_order,
                all_stages,
                pipeline,
                force=force,
                allow_missing=allow_missing,
                known=known,
            )
            explanations = [explanations_by_name[name] for name in execution_order]
            return _compute_explanations_with_upstream(explanations, stage_graph)
    finally:
        metrics.end("status.get_pipeline_explanations", _t)


def pullable_stages(stages: Iterable[tuple[str, Recovery | None, list[str]]]) -> set[str]:
    """The stages, of (name, recovery, stale upstream stages) in execution order with their
    upstream counted, that ``pivot pull`` brings up to date without recomputing anything.

    A stage is pullable when it has a recovery (``Recovery``: pull restores its outputs, fetches
    an object it reads by hash, or brings its stale upstream stages up to date) and every stale
    upstream stage is itself pullable. In a plain clone the downstream stages are stale only
    because their upstream's outputs are missing too, and pull restores all of them; a stage
    downstream of one that must recompute is recomputed by repro.
    """
    pullable = set[str]()
    for name, recovery, upstream_stale in stages:
        if recovery is not None and pullable.issuperset(upstream_stale):
            pullable.add(name)
    return pullable


def expand_current(
    pipeline: registry.PipelineLike,
    *,
    force: bool = False,
    allow_missing: bool = False,
    fetched: Mapping[str, listing.DirListing] | None = None,
) -> tuple[expansion.Expansion, dict[str, StageExplanation]]:
    """Expand each mapped stage (``Pipeline.map``) of ``pipeline`` whose producer is current,
    as a run would before running its instances (``expansion.expand_current``, which takes a
    listing the cache lacks from ``fetched``).

    Whether a producer is current is the explain path's decision (``get_pipeline_explanations``
    over the producers and their upstream stages): its explanation, its upstream counted, says it
    would not run; or, with a remote configured, only that it and its stale upstream stages are
    pullable (``pullable_stages``, by each explanation's ``recovery``): its lock matches, and pull
    restores what it lacks. Every other template stays unexpanded, naming what it waits for.

    Returns the expansion and the explanations it computed, before their upstream is counted,
    for the caller to pass on as ``known`` (computed with ``force`` and ``allow_missing``).
    """
    pipeline.reset_expansions()
    templates = pipeline.list_templates()
    if not templates:
        return expansion.Expansion(expanded={}, not_expanded={}, listings={}), {}
    _t = metrics.start()
    try:
        with state_dir_mod.entry_invocation(pipeline.resolved_state_dirs):
            all_stages = {name: pipeline.get_stage(name) for name in pipeline.list_stages()}
            producers = {pipeline.get_template(name)["source"].producer for name in templates}
            stage_graph = engine_graph.get_stage_dag(engine_graph.build_graph(all_stages))
            order = engine_graph.get_execution_order(
                stage_graph, sorted(producers & set(all_stages)), single_stage=False
            )
            raw = (
                _explanations_for(
                    order,
                    all_stages,
                    pipeline,
                    force=force,
                    allow_missing=allow_missing,
                    known=None,
                )
                if order
                else {}
            )
            enriched = _compute_explanations_with_upstream(
                [raw[name] for name in order], stage_graph
            )
            current = {exp["stage_name"] for exp in enriched if not exp["will_run"]}
            if remote_config.list_remotes():
                current |= pullable_stages(
                    (
                        exp["stage_name"],
                        exp["recovery"] if "recovery" in exp else None,
                        exp["upstream_stale"],
                    )
                    for exp in enriched
                )
            waiting = {
                exp["stage_name"]: exp["reason"] or "stale"
                for exp in enriched
                if exp["stage_name"] not in current
            }
            default_state_dir = config.locate_state_dir()
            files_dir = config.locate_cache_dir() / "files"
            locks = explain.read_locks(
                sorted(producers & current), all_stages, default_state_dir, files_dir
            )
            result = expansion.expand_current(
                pipeline,
                locks,
                waiting,
                default_state_dir=default_state_dir,
                cache_dir=files_dir,
                fetched=fetched,
            )
            return result, raw
    finally:
        metrics.end("status.expand_current", _t)


def not_expanded_status(
    result: expansion.Expansion, templates: Iterable[str] | None = None
) -> list[PipelineStatusInfo]:
    """A ``NOT_EXPANDED`` row for each template ``result`` left unexpanded, of ``templates``
    (None: every one), in the order the pipeline holds them."""
    wanted = None if templates is None else set(templates)
    return [
        PipelineStatusInfo(
            name=name,
            status=PipelineStatus.NOT_EXPANDED,
            reason=f"{NOT_EXPANDED_REASON_PREFIX}: {why}",
            upstream_stale=[],
        )
        for name, why in result["not_expanded"].items()
        if wanted is None or name in wanted
    ]


def _compute_explanations_with_upstream(
    explanations: list[StageExplanation],
    graph: DiGraph[str],
) -> list[StageExplanation]:
    """Process explanations and add upstream_stale field for stages stale due to upstream.

    ``recovery`` stays only when it brings the stage up to date without running it: a stage that
    will run keeps its own, and one current on its own gets ``Recovery.UPSTREAM``, but only when
    every stale upstream stage has a recovery too. A stage downstream of one that must run gets
    none, since that run's outputs reach it. ``explanations`` arrive in execution order."""
    stale_stages = set[str]()
    recovered = set[str]()
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
            else (
                f"{UPSTREAM_STALE_REASON_PREFIX} ({', '.join(upstream_stale)})"
                if upstream_stale
                else ""
            )
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
        if recovered.issuperset(upstream_stale):
            if exp["will_run"]:
                if "recovery" in exp:
                    new_exp["recovery"] = exp["recovery"]
            elif upstream_stale:
                new_exp["recovery"] = Recovery.UPSTREAM
        if "recovery" in new_exp:
            recovered.add(exp["stage_name"])
        results.append(new_exp)

    return results


def get_pipeline_status(
    stages: list[str] | None,
    single_stage: bool,
    all_stages: dict[str, RegistryStageInfo],
    pipeline: registry.PipelineLike,
    allow_missing: bool = False,
    graph: nx.DiGraph[str] | None = None,
    *,
    known: Mapping[str, StageExplanation] | None = None,
) -> tuple[list[PipelineStatusInfo], DiGraph[str]]:
    """The status of each stage of ``stages``' execution order; ``known`` as
    ``get_pipeline_explanations`` takes it."""
    _t = metrics.start()
    try:
        with state_dir_mod.entry_invocation(pipeline.resolved_state_dirs):
            if graph is None:
                graph = engine_graph.build_graph(all_stages)
            stage_graph = engine_graph.get_stage_dag(graph)
            execution_order = engine_graph.get_execution_order(
                stage_graph, stages, single_stage=single_stage
            )

            if not execution_order:
                return [], stage_graph

            explanations_by_name = _explanations_for(
                execution_order,
                all_stages,
                pipeline,
                force=False,
                allow_missing=allow_missing,
                known=known,
            )
            explanations = [explanations_by_name[name] for name in execution_order]
            enriched = _compute_explanations_with_upstream(explanations, stage_graph)
            return _explanations_to_status(enriched), stage_graph
    finally:
        metrics.end("status.get_pipeline_status", _t)


def _explanations_to_status(explanations: list[StageExplanation]) -> list[PipelineStatusInfo]:
    """Convert enriched explanations to PipelineStatusInfo list."""
    rows = list[PipelineStatusInfo]()
    for exp in explanations:
        row = PipelineStatusInfo(
            name=exp["stage_name"],
            status=PipelineStatus.STALE if exp["will_run"] else PipelineStatus.CACHED,
            reason=exp["reason"],
            upstream_stale=exp["upstream_stale"],
        )
        if "recovery" in exp:
            row["recovery"] = exp["recovery"]
        rows.append(row)
    return rows


def get_tracked_files_status(
    project_root: pathlib.Path,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    skip: track.PvtSkip | None = None,
) -> list[TrackedFileInfo]:
    """Get status for all tracked files.

    Args:
        project_root: Project root directory.
        on_progress: Optional callback called with (completed, total) after each file.
        skip: Leaves a .pvt out of the tracked files (``track.discover_pvt_files``): a stage's
            own member of its output (``member_skip.own_member_skip``).
    """
    tracked = track.discover_pvt_files(project_root, skip=skip)
    total = len(tracked)
    results = list[TrackedFileInfo]()

    # Look hashes up read-only and save what was read once, best-effort, so a hash cache that
    # cannot be opened or written never fails status; a damaged one still does. Both
    # directories are located, not created: the save makes the state directory #503's resolver
    # chose only when it has a hash to save.
    cache_dir = config.locate_cache_dir() / "files"
    state_dir = config.locate_state_dir()
    pending = list[tuple[pathlib.Path, os.stat_result, str]]()
    with state_mod.open_readonly(state_dir, create=False) as state_db:
        for i, (abs_path_str, track_data) in enumerate(sorted(tracked.items()), 1):
            path = pathlib.Path(abs_path_str)
            rel_path = str(path.relative_to(project_root))

            try:
                if path.is_dir():
                    current_hash = cache.hash_directory(
                        path, state_db, cache_dir=cache_dir, pending=pending
                    )["record"]["hash"]
                else:
                    current_hash, _ = cache.hash_file(path, state_db, pending=pending)
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

    state_mod.save_file_hashes(state_dir, pending, create=True)

    return results


def get_remote_status(
    remote_name: str | None,
    cache_dir: pathlib.Path,
) -> RemoteSyncInfo:
    """Get remote sync status: ``push_count`` objects an untargeted ``pivot push`` uploads and
    ``pull_count`` objects an untargeted ``pivot pull`` downloads (``compare_status``).

    Raises:
        RemoteNotConfiguredError: If no remotes are configured
        RemoteNotFoundError: If specified remote doesn't exist
        RemoteConnectionError: If connection to remote fails
    """
    remotes = remote_config.list_remotes()
    if not remotes:
        raise exceptions.RemoteNotConfiguredError("No remotes configured")

    remote_storage, resolved_name = transfer.create_remote_from_name(remote_name)
    url = display.redact_url_credentials(remote_config.get_remote_url(resolved_name))
    local_hashes = transfer.get_local_cache_hashes(cache_dir)

    with state_mod.StateDB(config.locate_state_dir()) as state_db:
        transfer.check_remote_url(state_db, resolved_name, remote_storage)
        status = asyncio.run(
            transfer.compare_status(local_hashes, remote_storage, state_db, resolved_name)
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
    all_stages: dict[str, RegistryStageInfo], pipeline: registry.PipelineLike
) -> list[str]:
    """Find lock and listing files in the registered stages' state dirs that belong to no
    registered stage, output or input.

    A file of an instance of a template (``Pipeline.map``) is none: an instance is registered
    only while a run has expanded its template, and a run decides which of them stay. Its lock is
    named by the instance, its listing ``<instance>=<key>.listing`` and its restart proof
    ``<instance>.incremental``, so each names the template (``registry.template_of``).
    """
    templates = pipeline.list_templates()
    default_state_dir = config.locate_state_dir()
    state_dirs = {default_state_dir} | {
        registry.get_stage_state_dir(stage_info, default_state_dir)
        for stage_info in all_stages.values()
    }
    store = _make_workspace_store(pipeline)
    registered_stages = dict[pathlib.Path, set[str]]()
    registered_listings = set[pathlib.Path]()
    for stage_name, stage_info in all_stages.items():
        stages_dir = explain.stages_dir(stage_info, default_state_dir)
        registered_stages.setdefault(stages_dir, set[str]()).add(stage_name)
        registered_listings.update(
            listing.output_listing_path(stages_dir, out.identity) for out in stage_info["outs"]
        )
        for ref in stage_info["deps"].values():
            if ref.tag is ArtifactTag.INPUT:
                registered_listings.add(
                    listing.input_listing_path(stages_dir, store.input_relpath(ref))
                )
    orphaned = set[str]()
    for state_dir in state_dirs:
        stages_dir = lock.get_stages_dir(state_dir)
        orphaned.update(
            lock.find_orphaned_locks(
                stages_dir, registered_stages.get(stages_dir, set[str]()), registered_listings
            )
        )
    return sorted(
        entry
        for entry in orphaned
        if registry.template_of(entry.split("=", 1)[0], templates) is None
    )


def get_suggestions(
    stale_count: int,
    modified_count: int,
    push_count: int,
    pull_count: int,
    orphan_count: int = 0,
    missing_outputs_count: int = 0,
    missing_reads_count: int = 0,
    remote_configured: bool = False,
) -> list[str]:
    suggestions = list[str]()

    # A clone whose locks are current but whose outputs are absent: pulling restores them
    # without recomputing, so it comes before the repro suggestion.
    if missing_outputs_count > 0 and remote_configured:
        suggestions.append(
            f"Run `pivot pull` to restore outputs of {missing_outputs_count} "
            + f"{_pluralize(missing_outputs_count, 'stage')} from the remote"
        )

    # Objects stages read by hash, missing from the cache: pulling fetches them, and the
    # stages are then current without running.
    if missing_reads_count > 0 and remote_configured:
        suggestions.append(
            f"Run `pivot pull` to fetch the objects {missing_reads_count} "
            + f"{_pluralize(missing_reads_count, 'stage')} read by hash from the remote"
        )

    if stale_count > 0:
        suggestions.append(
            f"Run `pivot repro` to execute {stale_count} stale {_pluralize(stale_count, 'stage')}"
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
            f"{orphan_count} orphaned lock or listing {_pluralize(orphan_count, 'file')} from "
            + "renamed/removed stages. Old outputs may still exist on disk."
        )

    return suggestions


async def _check_imports_batch(
    import_pvts: dict[str, track.PvtData],
    project_root: pathlib.Path,
) -> list[ImportStatusInfo]:
    tasks = list[tuple[str, pathlib.Path, track.PvtData]]()
    for data_path, pvt_data in sorted(import_pvts.items()):
        try:
            rel_path = str(pathlib.Path(data_path).relative_to(project_root))
        except ValueError:
            rel_path = data_path
        tasks.append((rel_path, track.get_pvt_path(pathlib.Path(data_path)), pvt_data))

    checks = await asyncio.gather(
        *[import_artifact.check_for_update(pvt_path, pvt_data) for _, pvt_path, pvt_data in tasks],
        return_exceptions=True,
    )

    results = list[ImportStatusInfo]()
    for (rel_path, _, _), check_or_exc in zip(tasks, checks, strict=True):
        if isinstance(check_or_exc, Exception):
            error = display.redact_url_credentials(exceptions.describe_error(check_or_exc))
            logger.warning(
                "Failed to check import %s: %s",
                display.escape_control(rel_path),
                display.escape_control(error),
            )
            results.append(
                ImportStatusInfo(
                    path=rel_path,
                    status=ImportCheckStatus.ERROR,
                    current_rev=None,
                    latest_rev=None,
                    error=error,
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
                        error=None,
                    )
                )
            else:
                results.append(
                    ImportStatusInfo(
                        path=rel_path,
                        status=ImportCheckStatus.UP_TO_DATE,
                        current_rev=None,
                        latest_rev=None,
                        error=None,
                    )
                )
    return results


def get_import_status(
    project_root: pathlib.Path, *, skip: track.PvtSkip | None = None
) -> list[ImportStatusInfo]:
    """Check update status for all imported artifacts concurrently (``skip``: as
    ``get_tracked_files_status``)."""
    import_pvts = track.discover_import_pvt_files(project_root, skip=skip)
    if not import_pvts:
        return []
    return asyncio.run(_check_imports_batch(import_pvts, project_root))
