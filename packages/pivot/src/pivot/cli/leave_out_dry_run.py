"""The dry run and ``--explain`` of ``pivot repro`` in leave-out mode
(``engine.graph.is_leave_out_mode``: ``--all --keep-going``, outside a watch session): the
decisions that run makes before it expands a mapped stage (``plan_run``), reading only, and what
it would then run (``explanations``), as ``Engine._orchestrate_execution`` decides both, with a
row for each mapped stage that does not expand, of the kind the run's decision about it gives
(``_explain_rows``). Any other mode is shown by ``cli/repro.py`` ``_get_explanations``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from pivot import config, exceptions, expansion, project
from pivot.cli import helpers as cli_helpers
from pivot.engine import graph as engine_graph
from pivot.types import OnError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    import networkx as nx

    from pivot.cli import console as tui_console
    from pivot.cli.status import Explained
    from pivot.executor.core import MissingTrackedFiles
    from pivot.registry import PipelineLike, RegistryStageInfo
    from pivot.types import ArtifactRef, StageExplanation

_logger = logging.getLogger(__name__)


class Plan(NamedTuple):
    """What a run in leave-out mode decides before it expands a mapped stage (``plan_run``)."""

    found: MissingTrackedFiles
    """The missing tracked files, split by whether the ``--checkout-missing`` restore brings them
    back (``executor.core.find_missing_tracked_files``, by the rule
    ``MissingTrackedFiles.restorable`` gives). A path it brings back counts as present, and
    nothing restores it; any other stays missing, as in the run."""
    present: list[tuple[str, ArtifactRef]]
    """Each dependency of the kept stages and templates that no stage produces, with its stage:
    each exists, or the restore creates it (``engine.graph.RunnableStages.present``)."""
    stages: dict[str, RegistryStageInfo]
    """The run's own stages and templates that it keeps."""
    left_out: dict[str, exceptions.DependencyNotFoundError]
    """Each stage and template the run leaves out, with its cause."""
    targets: list[str] | None
    """The targets the run keeps; None for every stage."""
    left_out_targets: dict[str, exceptions.DependencyNotFoundError]
    """Each target the run leaves out, as given, with its cause."""
    wanted: dict[str, dict[str, set[str]]]
    """The keys each template expands when the kept targets name instances
    (``expansion.mapped_targets``)."""


def plan_run(stages_list: list[str] | None, *, checkout_missing: bool) -> Plan:
    """The decisions ``pivot repro --all --keep-going`` makes before it expands a mapped stage,
    made in its order (``Engine._orchestrate_execution`` in leave-out mode), reading only.

    It refuses a cache whose object store is a mount of its own, as the run's registration in
    the cache does before anything else (``naming.require_shared_store``); finds the missing
    tracked files with the run's skip, built from the loaded stages
    (``executor.core.find_missing_tracked_files``; with ``checkout_missing`` each tracked path
    that the restore brings back counts as present, as ``Plan.found`` says, and nothing is
    restored); stops on one that is, lies inside or contains an output of a loaded stage
    (``ownership.tracked_output_refusal``), as the run start does; leaves out each stage with a
    missing input and every stage downstream of it, with one ``Skipping:`` warning per missing
    input (``engine.graph.get_runnable_stages``); and leaves out each target whose stage it
    leaves out (``expansion.keep_targets``, which the run calls too).

    Raises:
        CacheError: The cache's object store is the root of a mount of its own.
        TrackedFileMissingError: A missing tracked file that the restore does not bring back
            overlaps an output of a loaded stage.
        MultipleDependencyError: Every stage is left out.
    """
    from pivot.executor import core as executor_core
    from pivot.storage import member_skip, naming, ownership

    cache_dir = config.locate_cache_dir()
    naming.require_shared_store(cache_dir)
    pipeline = cli_helpers.get_pipeline()
    # The loaded stages, as the run start holds them: no instance
    pipeline.reset_expansions()
    loaded = cli_helpers.get_all_stages()
    templates = {name: pipeline.get_template(name) for name in pipeline.list_templates()}
    project_root = project.get_project_root()
    found = executor_core.find_missing_tracked_files(
        project_root,
        skip=member_skip.own_member_skip(
            project_root, loaded, config.locate_state_dir(), cache_dir / "files"
        ),
        checkout_missing=checkout_missing,
    )
    if found.missing.paths:
        refusal = ownership.tracked_output_refusal(
            found.missing, ownership.stage_output_paths(project_root, loaded), project_root
        )
        if refusal is not None:
            raise refusal
    runnable, errors, left_out, present = engine_graph.get_runnable_stages(
        cli_helpers.get_graph_stages(),
        cli_helpers.require_workspace_store(),
        all_pipelines=True,
        on_error=OnError.KEEP_GOING,
        missing_tracked=found.missing.paths,
        restored_tracked=found.restored,
    )
    for error in errors:
        _logger.warning("Skipping: %s", error.format_user_message())
    if runnable is None or (errors and not runnable):
        raise exceptions.MultipleDependencyError(errors)
    kept = expansion.keep_targets(stages_list or [], templates, left_out)
    return Plan(
        found=found,
        present=[(name, dep) for name, dep in present if name in runnable],
        stages=runnable,
        left_out=left_out,
        targets=None if stages_list is None else kept.targets,
        left_out_targets=kept.left_out_targets,
        wanted=kept.wanted,
    )


def _refused_upstream(
    template: str, refused: Collection[str], pipeline: PipelineLike
) -> str | None:
    """The first, sorted, of the mapped stages ``refused`` (``expansion.Expansion``'s) that
    mapped stage ``template`` is mapped over, directly or through other mapped stages
    (``cli_helpers.templates_and_upstream``), or None. The run fails each refused one, which
    blocks ``template``."""
    if not refused:
        return None
    upstream = set(cli_helpers.templates_and_upstream([template], pipeline)) - {template}
    found = sorted(upstream.intersection(refused))
    return found[0] if found else None


def _scheduled(
    plan: Plan, kept: expansion.Expansion, order: list[str]
) -> tuple[list[str] | None, list[str] | None]:
    """The stages the leave-out dry run explains for the targets it keeps, and the mapped stages
    it shows a row for, as the run's own execution order holds them: ``order``, the order the
    run schedules for those targets over the stages and mapped stages it keeps
    (``cli_helpers.run_order``, as ``Engine._orchestrate_execution`` orders them, each instance
    target standing for its mapped stage), every stage upstream of a mapped stage included.
    Each stage of it is explained, and so is every instance of each mapped stage of it that
    expands (``kept``), which the run adds and runs; each mapped stage of it that does not
    expand has a row. With no target the order holds every stage and mapped stage the run
    keeps, for which None and None stand: every stage explained, every row shown."""
    if plan.targets is None:
        return None, None
    templates = set(cli_helpers.get_pipeline().list_templates())
    stages = [name for name in order if name not in templates]
    stages += [
        instance
        for name in order
        if name in kept["expanded"]
        for instance in kept["expanded"][name]
    ]
    return stages, [name for name in order if name in templates and name not in kept["expanded"]]


def _log_restore_limits(
    not_expanded: Collection[str],
    plan: Plan,
    graph: nx.DiGraph[str],
    known: Mapping[str, StageExplanation],
    not_current: Collection[str],
    *,
    force: bool,
) -> None:
    """Log one notice for each mapped stage of ``not_expanded`` (shown as not yet expanded)
    whose producer is in ``not_current`` and has a limit from ``status.restore_limits`` (given
    ``graph``, ``known`` and ``force``); with no restorable file it reads nothing."""
    from pivot import status as status_mod

    if not plan.found.restorable or not not_expanded:
        return
    pipeline = cli_helpers.get_pipeline()
    # A template mapped over another one's instances shares its source: this is the producer
    # of the first template of its chain
    producers = {
        template: pipeline.get_template(template)["source"].producer for template in not_expanded
    }
    limits = status_mod.restore_limits(
        pipeline,
        {producer for producer in producers.values() if producer in not_current},
        graph,
        present=plan.present,
        restorable=plan.found.restorable,
        known=known,
        force=force,
    )
    for template, producer in producers.items():
        if producer not in limits:
            continue
        limit = limits[producer]
        reader = (
            f"Its producer '{producer}'"
            if limit.reader == producer
            else f"Stage '{limit.reader}', upstream of its producer '{producer}',"
        )
        path = project.to_relative_path(limit.path)
        tracked = project.to_relative_path(limit.tracked)
        _logger.warning(
            "Notice: %s",
            f"mapped stage '{template}' is shown as not yet expanded. {reader} reads '{path}', "
            + f"which changes when --checkout-missing restores tracked path '{tracked}'. The dry "
            + f"run explains '{limit.reader}' from the files on disk, so the run may restore "
            + f"'{tracked}', find '{producer}' current and add the instances of '{template}'.",
        )


def _explain_rows(
    kept: expansion.Expansion, shown: list[str] | None, pipeline: PipelineLike
) -> list[tui_console.ExplainRow]:
    """A row for each mapped stage of ``shown`` (None: every one) that ``kept`` leaves
    unexpanded, of the kind the run's decision about it gives: one of ``kept``'s ``refused``,
    which the run fails, with the reason ``kept`` gives; one the run blocks behind one of those
    (``_refused_upstream``), naming that one; any other one, which the run expands once its
    producer is current, with what it waits for. In the order of the kinds of
    ``status_cli.DRY_RUN_SHOWN``, each kind's rows in the order the pipeline holds them."""
    from pivot import status as status_mod
    from pivot.cli import console as tui_console_mod
    from pivot.cli import status as status_cli

    refused = kept["refused"]
    rows = list[tui_console_mod.ExplainRow]()
    for row in status_mod.not_expanded_status(kept, shown):
        name = row["name"]
        if name in refused:
            kind, reason = status_cli.REFUSED_ROW, kept["not_expanded"][name]
        elif (upstream := _refused_upstream(name, refused, pipeline)) is not None:
            kind, reason = status_cli.BLOCKED_ROW, f"upstream '{upstream}' would fail"
        else:
            kind, reason = status_cli.NOT_EXPANDED_ROW, row["reason"]
        rows.append(tui_console_mod.ExplainRow(kind, name, reason))
    kinds = list(status_cli.DRY_RUN_SHOWN)
    return sorted(rows, key=lambda row: kinds.index(row.kind))


def explanations(
    stages_list: list[str] | None,
    *,
    force: bool,
    no_commit: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> Explained:
    """What ``pivot repro --all --keep-going`` of ``stages_list`` would do, for ``--dry-run``
    and ``--explain``: the stage explanations, a row for each mapped stage that does not expand
    (``_explain_rows``), each orphaned instance the run would remove, mapped to its mapped stage
    (``status.orphaned_instances``: of each mapped stage the run expands without refusing,
    ``cli_helpers.eligible_templates``, unless it commits nothing), and each target it leaves
    out. The command refuses ``--allow-missing`` in this mode, so no missing input is allowed.

    The run's decisions before it expands (``plan_run``) leave out stages and targets; when
    targets were left out and none is left, nothing is explained, as the run runs nothing. The
    mapped stages expand as the run would expand them: the missing tracked files that the
    restore brings back count as present, each template adds the instances
    ``expansion.plan_instances`` plans over the run's checks, or the template fails with its
    reason, and with ``no_commit`` the keys are read as that run reads them. A template the run
    fails (``expansion.Expansion``'s ``refused``: any refusal but an unread listing of a producer
    current only by a pull, which the run runs again) has a row that says so, not a not-expanded
    one, and so does each template the run blocks behind it (``_refused_upstream``). The targets
    stand for the run's own execution order for them (``_scheduled``): each stage of it, every
    stage upstream of a mapped stage included, the instances of each mapped stage of it that
    expands, and a row for each mapped stage of it that does not. A left-out stage, every
    instance of a left-out mapped stage, its row and its orphaned instances are not shown, and a
    left-out stage changes no orphan the run would remove (``cli_helpers.eligible_templates``
    over the stages the run keeps). With ``checkout_missing``, a notice on stderr names each
    mapped stage shown as not yet expanded that the run may expand (``_log_restore_limits``).
    """
    from pivot import status as status_mod
    from pivot.cli import status as status_cli

    plan = plan_run(stages_list, checkout_missing=checkout_missing)
    if plan.left_out_targets and not plan.targets:
        # Every target is left out; no target would explain every stage
        return status_cli.Explained(
            explanations=[], rows=[], orphans={}, left_out_targets=plan.left_out_targets
        )
    pipeline = cli_helpers.get_pipeline()
    # Each stage of the producers' execution order that is not current
    not_current = set[str]()
    result, known = cli_helpers.expand_mapped_stages(
        force=force,
        restorable=plan.found.restorable,
        checks=expansion.InstanceChecks(
            wanted=plan.wanted,
            missing_tracked=plan.found.missing,
            allow_uncached_incremental=allow_uncached_incremental,
            force=force,
        ),
        no_commit=no_commit,
        not_current=not_current,
    )
    kept = expansion.without_templates(result, plan.left_out)
    # The run's own execution order for its targets, which the orphans it removes and the
    # stages shown for its targets both follow
    order = cli_helpers.run_order(plan.targets, plan.stages)
    removing = (
        []
        if no_commit
        else cli_helpers.eligible_templates(
            kept,
            order,
            force=force,
            allow_uncached_incremental=allow_uncached_incremental,
            stages=plan.stages,
        )
    )
    orphans = status_mod.orphaned_instances(pipeline, kept, removing)
    targets, shown_templates = _scheduled(plan, kept, order)
    # Each left-out stage and template, and every instance of a left-out template
    hidden = set(plan.left_out).union(
        *(result["expanded"][name] for name in plan.left_out if name in result["expanded"])
    )
    all_stages = {
        name: info for name, info in cli_helpers.get_all_stages().items() if name not in hidden
    }
    # No dependency check: ``plan_run`` already reported each missing dependency and left out
    # its stages
    graph = engine_graph.build_graph(all_stages)
    stage_explanations = status_mod.get_pipeline_explanations(
        targets,
        single_stage=False,
        all_stages=all_stages,
        pipeline=pipeline,
        force=force,
        graph=graph,
        known=known,
        restorable=plan.found.restorable,
    )
    rows = _explain_rows(kept, shown_templates, pipeline)
    _log_restore_limits(
        [row.name for row in rows if row.kind == status_cli.NOT_EXPANDED_ROW],
        plan,
        graph,
        known,
        not_current,
        force=force,
    )
    return status_cli.Explained(
        explanations=stage_explanations,
        rows=rows,
        orphans=orphans,
        left_out_targets=plan.left_out_targets,
    )
