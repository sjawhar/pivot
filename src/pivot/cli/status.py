from __future__ import annotations

import json
import pathlib

import click

from pivot import discovery, exceptions, project, registry
from pivot import status as status_mod
from pivot.cli import completion
from pivot.types import (
    PipelineStatusInfo,
    RemoteSyncInfo,
    StatusOutput,
    TrackedFileInfo,
)


def _ensure_stages_registered() -> None:
    """Auto-discover and register stages if none are registered."""
    if not discovery.has_registered_stages():
        try:
            discovery.discover_and_register()
        except discovery.DiscoveryError as e:
            raise click.ClickException(str(e)) from e


def _validate_stages(stages_list: list[str] | None) -> None:
    """Validate stage arguments."""
    if stages_list:
        graph = registry.REGISTRY.build_dag(validate=True)
        registered = set(graph.nodes())
        unknown = [s for s in stages_list if s not in registered]
        if unknown:
            raise click.ClickException(f"Unknown stage(s): {', '.join(unknown)}")


@click.command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--verbose", "-v", is_flag=True, help="Show all stages, not just stale")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--stages-only", is_flag=True, help="Show only pipeline status")
@click.option("--tracked-only", is_flag=True, help="Show only tracked files")
@click.option("--remote-only", is_flag=True, help="Show only remote status")
@click.option("--remote", "-r", is_flag=True, help="Include remote sync status")
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
def status(
    stages: tuple[str, ...],
    verbose: bool,
    output_json: bool,
    stages_only: bool,
    tracked_only: bool,
    remote_only: bool,
    remote: bool,
    cache_dir: pathlib.Path | None,
) -> None:
    """Show pipeline, tracked files, and remote status."""
    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list)

    project_root = project.get_project_root()
    resolved_cache_dir = cache_dir or project_root / ".pivot" / "cache"

    show_all = not (stages_only or tracked_only or remote_only)
    show_stages = show_all or stages_only
    show_tracked = show_all or tracked_only
    show_remote = remote_only or remote

    pipeline_status = list[PipelineStatusInfo]()
    tracked_status = list[TrackedFileInfo]()
    remote_status: RemoteSyncInfo | None = None

    if show_stages:
        pipeline_status, _ = status_mod.get_pipeline_status(
            stages_list, single_stage=False, cache_dir=resolved_cache_dir
        )

    if show_tracked:
        tracked_status = status_mod.get_tracked_files_status(project_root)

    if show_remote:
        try:
            remote_status = status_mod.get_remote_status(None, resolved_cache_dir)
        except exceptions.RemoteNotConfiguredError:
            if remote_only:
                raise click.ClickException(
                    "No remotes configured. Use `pivot remote add`."
                ) from None
            click.echo("No remotes configured")
        except exceptions.RemoteError as e:
            raise click.ClickException(f"Remote error: {e}") from e

    # Compute counts once for suggestions and output
    stale_count = sum(1 for s in pipeline_status if s["status"] == "stale")
    modified_count = sum(1 for f in tracked_status if f["status"] == "modified")
    push_count = remote_status["push_count"] if remote_status else 0
    pull_count = remote_status["pull_count"] if remote_status else 0
    suggestions = status_mod.get_suggestions(stale_count, modified_count, push_count, pull_count)

    if output_json:
        _output_json(
            pipeline_status,
            tracked_status,
            remote_status,
            suggestions,
            show_stages,
            show_tracked,
            show_remote,
        )
    else:
        _output_text(
            pipeline_status,
            tracked_status,
            remote_status,
            suggestions,
            verbose,
            show_stages,
            show_tracked,
        )


def _output_json(
    pipeline_status: list[PipelineStatusInfo],
    tracked_status: list[TrackedFileInfo],
    remote_status: RemoteSyncInfo | None,
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

    if suggestions:
        data["suggestions"] = suggestions

    click.echo(json.dumps(data, indent=2))


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


def _print_pipeline_section(pipeline_status: list[PipelineStatusInfo], verbose: bool) -> None:
    """Print pipeline status section."""
    click.echo("Pipeline Status")

    if not pipeline_status:
        click.echo("  No stages registered")
        return

    total = len(pipeline_status)
    stale = sum(1 for s in pipeline_status if s["status"] == "stale")
    click.echo(f"  {total} stages: {total - stale} cached, {stale} stale")
    click.echo()

    stages_to_show = (
        pipeline_status if verbose else [s for s in pipeline_status if s["status"] == "stale"]
    )

    for stage in stages_to_show:
        icon = "\u2713" if stage["status"] == "cached" else "\u26a0"
        click.echo(f"  {icon} {stage['name']:<20} {stage['reason'] or '-'}")


def _print_tracked_section(tracked_status: list[TrackedFileInfo], verbose: bool) -> None:
    """Print tracked files section."""
    click.echo("Tracked Files")

    if not tracked_status:
        click.echo("  No tracked files")
        return

    total = len(tracked_status)
    clean = sum(1 for f in tracked_status if f["status"] == "clean")
    modified = sum(1 for f in tracked_status if f["status"] == "modified")
    missing = total - clean - modified

    parts = [f"{clean} clean", f"{modified} modified"]
    if missing > 0:
        parts.append(f"{missing} missing")
    click.echo(f"  {total} files: {', '.join(parts)}")
    click.echo()

    files_to_show = (
        tracked_status if verbose else [f for f in tracked_status if f["status"] != "clean"]
    )

    for file in files_to_show:
        match file["status"]:
            case "clean":
                icon = "\u2713"
            case "modified":
                icon = "\u26a0"
            case _:
                icon = "\u2717"
        click.echo(f"  {icon} {file['path']:<30} {file['status']}")


def _print_remote_section(remote_status: RemoteSyncInfo) -> None:
    """Print remote status section."""
    click.echo(f"Remote Status ({remote_status['name']} \u2192 {remote_status['url']})")
    click.echo(f"  \u2191 {remote_status['push_count']} to push")
    click.echo(f"  \u2193 {remote_status['pull_count']} to pull")


def _print_suggestions_section(suggestions: list[str]) -> None:
    """Print suggestions section."""
    click.echo("Suggestions")
    for suggestion in suggestions:
        click.echo(f"  {suggestion}")
