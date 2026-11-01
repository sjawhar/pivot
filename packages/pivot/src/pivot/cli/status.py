from __future__ import annotations

import sys

import click
import tqdm

from pivot import config, display, exceptions, expansion, project
from pivot import status as status_mod
from pivot.cli import completion, console
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import graph as engine_graph
from pivot.remote import config as remote_config
from pivot.storage import ownership
from pivot.types import (
    ExplainOutput,
    ExplainStageJson,
    ImportCheckStatus,
    ImportStatusInfo,
    PipelineStatus,
    PipelineStatusInfo,
    Recovery,
    RemoteSyncInfo,
    StageExplanation,
    StatusOutput,
    TrackedFileInfo,
    TrackedFileStatus,
    dep_change_to_json,
)


@cli_decorators.pivot_command(allow_all=True)
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--verbose", "-v", is_flag=True, help="Show all stages, not just stale")
@click.option(
    "--explain", "-e", is_flag=True, help="Show detailed breakdown of why stages would run"
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON. Writes one document to stdout; notes go to stderr.",
)
@click.option("--stages-only", is_flag=True, help="Show only pipeline status")
@click.option("--tracked-only", is_flag=True, help="Show only tracked files")
@click.option("--remote-only", is_flag=True, help="Show only remote status")
@click.option("--remote", "-r", is_flag=True, help="Include remote sync status")
@click.option(
    "--check-imports",
    is_flag=True,
    help="Check for import updates (requires network). JSON adds an imports field.",
)
@click.pass_context
def status(
    ctx: click.Context,
    stages: tuple[str, ...],
    verbose: bool,
    explain: bool,
    output_json: bool,
    stages_only: bool,
    tracked_only: bool,
    remote_only: bool,
    remote: bool,
    check_imports: bool,
) -> None:
    """Show pipeline, tracked files, and remote status."""
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    stages_list = cli_helpers.stages_to_list(stages)
    pipeline = cli_decorators.get_pipeline_from_context()

    show_all = not (stages_only or tracked_only or remote_only)
    show_stages = show_all or stages_only
    show_tracked = show_all or tracked_only
    show_remote = remote_only or remote

    # Mapped stages expand first, as a run would expand them, so the stages shown hold their
    # instances and a target may name a mapped stage or one of its instances.
    expansion_result = expansion.Expansion(expanded={}, not_expanded={}, listings={})
    known = dict[str, StageExplanation]()
    shown_templates: list[str] | None = None
    given_targets = stages_list
    if pipeline is not None:
        if show_stages:
            expansion_result, known = cli_helpers.expand_mapped_stages()
        stages_list, shown_templates = cli_helpers.resolve_mapped_targets(
            stages_list, expansion_result
        )

    project_root = project.get_project_root()
    resolved_cache_dir = config.locate_cache_dir()
    if pipeline is not None and not quiet:
        # Warn, refusing nothing, about a tracked file saved in an output before track refused it
        for warning in ownership.saved_input_warnings(
            project_root,
            cli_helpers.get_all_stages(),
            cli_helpers.require_workspace_store(),
            config.get_state_dir(),
            resolved_cache_dir / "files",
        ):
            click.echo(warning, err=True)

    # A stage's own .pvt member of its output, in any pipeline, is no tracked file or import
    skip = (
        cli_helpers.own_member_skip(resolved_cache_dir / "files")
        if show_tracked or check_imports
        else None
    )

    # When --explain is used, fetch detailed explanations instead of basic status
    pipeline_status = list[PipelineStatusInfo]()
    pipeline_explanations = list[StageExplanation]()
    not_expanded = list[PipelineStatusInfo]()
    orphaned = list[PipelineStatusInfo]()
    tracked_status = list[TrackedFileInfo]()
    remote_status: RemoteSyncInfo | None = None

    orphaned_locks = list[str]()

    if show_stages and pipeline is not None:
        all_stages = cli_helpers.get_all_stages()
        graph = engine_graph.build_graph(all_stages)
        orphaned_locks = status_mod.find_orphaned_lock_files(all_stages, pipeline)
        not_expanded = status_mod.not_expanded_status(expansion_result, shown_templates)
        # An instance whose key left its producer's listing: of each mapped stage a run of the
        # same targets expands, by the run's own rule
        orphaned = status_mod.orphaned_status(
            pipeline,
            status_mod.orphaned_instances(
                pipeline, expansion_result, cli_helpers.run_templates(given_targets)
            ),
        )

        if explain:
            pipeline_explanations = status_mod.get_pipeline_explanations(
                stages_list,
                single_stage=False,
                all_stages=all_stages,
                pipeline=cli_helpers.get_pipeline(),
                graph=graph,
                known=known,
            )
        else:
            pipeline_status, _ = status_mod.get_pipeline_status(
                stages_list,
                single_stage=False,
                all_stages=all_stages,
                pipeline=cli_helpers.get_pipeline(),
                graph=graph,
                known=known,
            )

    if show_tracked:
        # Show progress bar only on TTY and not in quiet/JSON mode
        use_progress = sys.stderr.isatty() and not quiet and not output_json
        pbar: tqdm.tqdm[None] | None = (
            tqdm.tqdm(
                total=None, desc="Checking tracked files", file=sys.stderr, dynamic_ncols=True
            )
            if use_progress
            else None
        )

        def on_progress(_completed: int, total: int) -> None:
            if pbar is None:
                return
            if pbar.total is None:
                pbar.total = total
                pbar.refresh()
            pbar.update(1)

        try:
            tracked_status = status_mod.get_tracked_files_status(
                project_root, on_progress, skip=skip
            )
        finally:
            if pbar is not None:
                pbar.close()

    if show_remote:
        try:
            remote_status = status_mod.get_remote_status(None, resolved_cache_dir)
        except exceptions.RemoteNotConfiguredError:
            if remote_only:
                raise click.ClickException(
                    "No remotes configured. Use `pivot config set remotes.<name> <url>`."
                ) from None
            if not quiet:
                click.echo("No remotes configured", err=output_json)
        except exceptions.RemoteError as e:
            raise click.ClickException(f"Remote error: {e}") from e

    import_statuses = list[ImportStatusInfo]()
    if check_imports:
        import_statuses = status_mod.get_import_status(project_root, skip=skip)

    # Compute counts once for suggestions and output
    if explain and pipeline_explanations:
        stale_count = sum(1 for e in pipeline_explanations if e["will_run"])
        recoveries = [e["recovery"] for e in pipeline_explanations if "recovery" in e]
    else:
        stale_count = sum(1 for s in pipeline_status if s["status"] == PipelineStatus.STALE)
        recoveries = [s["recovery"] for s in pipeline_status if "recovery" in s]
    # Pull restores missing outputs, and fetches missing objects stages read by hash, without
    # recomputing; skip the count when there is no remote to pull from. A stage's recovery is set
    # only when it brings the stage up to date with everything upstream of it
    # (`status._compute_explanations_with_upstream`), so the count reads nothing else.
    missing_outputs_count = 0
    missing_reads_count = 0
    if remote_config.list_remotes():
        missing_outputs_count = recoveries.count(Recovery.PULL_OUTPUTS)
        missing_reads_count = recoveries.count(Recovery.PULL_READS)
    modified_count = sum(1 for f in tracked_status if f["status"] == TrackedFileStatus.MODIFIED)
    push_count = remote_status["push_count"] if remote_status else 0
    pull_count = remote_status["pull_count"] if remote_status else 0
    suggestions = status_mod.get_suggestions(
        stale_count,
        modified_count,
        push_count,
        pull_count,
        len(orphaned_locks),
        missing_outputs_count=missing_outputs_count,
        missing_reads_count=missing_reads_count,
        remote_configured=missing_outputs_count + missing_reads_count > 0,
        orphaned_instances=len(orphaned),
    )

    # Quiet mode: no output, exit 1 if there are issues needing attention
    if quiet and not output_json:
        if stale_count > 0 or modified_count > 0 or not_expanded or orphaned:
            raise SystemExit(1)
        return

    if output_json:
        imports = import_statuses if check_imports else None
        if explain:
            _output_explain_json(
                pipeline_explanations,
                not_expanded,
                orphaned,
                tracked_status,
                remote_status,
                imports,
                suggestions,
                show_stages,
                show_tracked,
                show_remote,
            )
        else:
            _output_json(
                [*pipeline_status, *not_expanded, *orphaned],
                tracked_status,
                remote_status,
                imports,
                suggestions,
                show_stages,
                show_tracked,
                show_remote,
            )
    else:
        if explain:
            output_explain_text(
                pipeline_explanations,
                tracked_status,
                remote_status,
                suggestions,
                show_tracked,
                not_expanded=not_expanded,
                orphaned=orphaned,
            )
        else:
            _output_text(
                [*pipeline_status, *not_expanded, *orphaned],
                tracked_status,
                remote_status,
                suggestions,
                verbose,
                show_stages,
                show_tracked,
            )

        if import_statuses:
            _print_import_section(import_statuses)


def _output_json(
    pipeline_status: list[PipelineStatusInfo],
    tracked_status: list[TrackedFileInfo],
    remote_status: RemoteSyncInfo | None,
    imports: list[ImportStatusInfo] | None,
    suggestions: list[str],
    show_stages: bool,
    show_tracked: bool,
    show_remote: bool,
) -> None:
    """Output status as JSON."""
    data = StatusOutput()

    if show_stages:
        data["stages"] = pipeline_status

    if show_tracked:
        data["tracked_files"] = tracked_status

    if show_remote and remote_status:
        data["remote"] = remote_status

    if imports is not None:
        data["imports"] = imports

    if suggestions:
        data["suggestions"] = suggestions

    click.echo(display.to_json(data, indent=2))


def _output_explain_json(
    explanations: list[StageExplanation],
    not_expanded: list[PipelineStatusInfo],
    orphaned: list[PipelineStatusInfo],
    tracked_status: list[TrackedFileInfo],
    remote_status: RemoteSyncInfo | None,
    imports: list[ImportStatusInfo] | None,
    suggestions: list[str],
    show_stages: bool,
    show_tracked: bool,
    show_remote: bool,
) -> None:
    """Output status with detailed explanations as JSON."""
    data = ExplainOutput()

    if show_stages:
        # Convert explanations to JSON-serializable format with name field
        stages = list[ExplainStageJson]()
        for exp in explanations:
            stage_data = ExplainStageJson(
                name=exp["stage_name"],
                status="stale" if exp["will_run"] else "cached",
                reason=exp["reason"],
                will_run=exp["will_run"],
                is_forced=exp["is_forced"],
                code_changes=exp["code_changes"],
                param_changes=exp["param_changes"],
                dep_changes=[dep_change_to_json(change) for change in exp["dep_changes"]],
                upstream_stale=exp["upstream_stale"],
            )
            if "recovery" in exp:
                stage_data["recovery"] = exp["recovery"]
            stages.append(stage_data)
        # A mapped stage not yet expanded: a run would expand it and run its instances
        stages.extend(
            ExplainStageJson(
                name=row["name"],
                status="not_expanded",
                reason=row["reason"],
                will_run=True,
                is_forced=False,
                code_changes=[],
                param_changes=[],
                dep_changes=[],
                upstream_stale=row["upstream_stale"],
            )
            for row in not_expanded
        )
        # An instance whose key left its producer's listing: the run removes it, runs nothing
        stages.extend(
            ExplainStageJson(
                name=row["name"],
                status="orphaned",
                reason=row["reason"],
                will_run=False,
                is_forced=False,
                code_changes=[],
                param_changes=[],
                dep_changes=[],
                upstream_stale=row["upstream_stale"],
            )
            for row in orphaned
        )
        data["stages"] = stages

    if show_tracked:
        data["tracked_files"] = tracked_status

    if show_remote and remote_status:
        data["remote"] = remote_status

    if imports is not None:
        data["imports"] = imports

    if suggestions:
        data["suggestions"] = suggestions

    click.echo(display.to_json(data, indent=2))


def _output_text(
    pipeline_status: list[PipelineStatusInfo],
    tracked_status: list[TrackedFileInfo],
    remote_status: RemoteSyncInfo | None,
    suggestions: list[str],
    verbose: bool,
    show_stages: bool,
    show_tracked: bool,
) -> None:
    """Output status as formatted text."""
    sections_printed = 0

    if show_stages:
        _print_pipeline_section(pipeline_status, verbose)
        sections_printed += 1

    if show_tracked:
        if sections_printed > 0:
            click.echo()
        _print_tracked_section(tracked_status, verbose)
        sections_printed += 1

    if remote_status:
        if sections_printed > 0:
            click.echo()
        _print_remote_section(remote_status)
        sections_printed += 1

    if suggestions:
        if sections_printed > 0:
            click.echo()
        _print_suggestions_section(suggestions)


def output_explain_text(
    explanations: list[StageExplanation],
    tracked_status: list[TrackedFileInfo] | None = None,
    remote_status: RemoteSyncInfo | None = None,
    suggestions: list[str] | None = None,
    show_tracked: bool = False,
    *,
    not_expanded: list[PipelineStatusInfo] | None = None,
    orphaned: list[PipelineStatusInfo] | None = None,
) -> None:
    """Output detailed stage explanations as formatted text, then each mapped stage not yet
    expanded (``not_expanded``) with what it waits for, then each orphaned instance of a mapped
    stage (``orphaned``), which the run removes."""
    sections_printed = 0

    # Always show stage explanations section
    if not explanations and not not_expanded and not orphaned:
        click.echo("No stages to run")
    else:
        con = console.Console()
        for exp in explanations:
            con.explain_stage(exp)
        for row in not_expanded or []:
            con.explain_not_expanded(row["name"], row["reason"])
        for row in orphaned or []:
            con.explain_orphaned(row["name"], row["reason"])

        will_run = sum(1 for e in explanations if e["will_run"])
        con.explain_summary(
            will_run,
            len(explanations) - will_run,
            len(not_expanded or []),
            orphaned=len(orphaned or []),
        )
    sections_printed += 1

    if show_tracked and tracked_status is not None:
        if sections_printed > 0:
            click.echo()
        _print_tracked_section(tracked_status, verbose=False)
        sections_printed += 1

    if remote_status:
        if sections_printed > 0:
            click.echo()
        _print_remote_section(remote_status)
        sections_printed += 1

    if suggestions:
        if sections_printed > 0:
            click.echo()
        _print_suggestions_section(suggestions)


def _print_pipeline_section(pipeline_status: list[PipelineStatusInfo], verbose: bool) -> None:
    """Print pipeline status section. A mapped stage not yet expanded, and an orphaned instance
    of one, is always shown."""
    click.echo("Pipeline Status")

    if not pipeline_status:
        click.echo("  No stages registered")
        return

    not_expanded = sum(1 for s in pipeline_status if s["status"] == PipelineStatus.NOT_EXPANDED)
    orphaned = sum(1 for s in pipeline_status if s["status"] == PipelineStatus.ORPHANED)
    total = len(pipeline_status) - not_expanded - orphaned
    stale = sum(1 for s in pipeline_status if s["status"] == PipelineStatus.STALE)
    click.echo(f"  {total} stages: {total - stale} cached, {stale} stale")
    if not_expanded:
        noun = "stage" if not_expanded == 1 else "stages"
        click.echo(f"  {not_expanded} mapped {noun} not yet expanded")
    if orphaned:
        noun = "instance" if orphaned == 1 else "instances"
        click.echo(f"  {orphaned} orphaned {noun} of mapped stages")
    click.echo()

    stages_to_show = (
        pipeline_status
        if verbose
        else [s for s in pipeline_status if s["status"] != PipelineStatus.CACHED]
    )

    icons = {
        PipelineStatus.CACHED: "\u2713",
        PipelineStatus.STALE: "\u26a0",
        PipelineStatus.NOT_EXPANDED: "\u2026",
        PipelineStatus.ORPHANED: "\u2212",
    }
    for stage in stages_to_show:
        reason = display.escape_control(stage["reason"], keep_newlines=True) or "-"
        click.echo(f"  {icons[stage['status']]} {stage['name']:<20} {reason}")


def _print_tracked_section(tracked_status: list[TrackedFileInfo], verbose: bool) -> None:
    """Print tracked files section."""
    click.echo("Tracked Files")

    if not tracked_status:
        click.echo("  No tracked files")
        return

    total = len(tracked_status)
    clean = sum(1 for f in tracked_status if f["status"] == TrackedFileStatus.CLEAN)
    modified = sum(1 for f in tracked_status if f["status"] == TrackedFileStatus.MODIFIED)
    missing = total - clean - modified

    parts = [f"{clean} clean", f"{modified} modified"]
    if missing > 0:
        parts.append(f"{missing} missing")
    click.echo(f"  {total} files: {', '.join(parts)}")
    click.echo()

    files_to_show = (
        tracked_status
        if verbose
        else [f for f in tracked_status if f["status"] != TrackedFileStatus.CLEAN]
    )

    for file in files_to_show:
        match file["status"]:
            case TrackedFileStatus.CLEAN:
                icon = "\u2713"
            case TrackedFileStatus.MODIFIED:
                icon = "\u26a0"
            case _:
                icon = "\u2717"
        click.echo(f"  {icon} {display.escape_control(file['path']):<30} {file['status']}")


def _print_remote_section(remote_status: RemoteSyncInfo) -> None:
    """Print remote status section."""
    name = display.escape_control(remote_status["name"])
    url = display.escape_control(remote_status["url"])
    click.echo(f"Remote Status ({name} \u2192 {url})")
    click.echo(f"  \u2191 {remote_status['push_count']} to push")
    click.echo(f"  \u2193 {remote_status['pull_count']} to pull")


def _print_import_section(import_statuses: list[ImportStatusInfo]) -> None:
    """Print import status section."""
    click.echo()
    click.echo("Imports")

    for info in import_statuses:
        path = display.escape_control(info["path"])
        if info["status"] is ImportCheckStatus.UPDATE_AVAILABLE:
            current = display.escape_control(info["current_rev"] or "")
            latest = display.escape_control(info["latest_rev"] or "")
            click.echo(f"  \u26a0 {path}: update available ({current}.. \u2192 {latest}..)")
        elif info["status"] is ImportCheckStatus.ERROR:
            click.echo(
                f"  \u2717 {path}: check failed ({display.escape_control(info['error'] or '')})"
            )
        else:
            click.echo(f"  \u2713 {path}: up to date")


def _print_suggestions_section(suggestions: list[str]) -> None:
    """Print suggestions section."""
    click.echo("Suggestions")
    for suggestion in suggestions:
        click.echo(f"  {suggestion}")
