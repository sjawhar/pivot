from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import click

from pivot import cache, config, discovery, executor, get, project, pvt, registry
from pivot.types import DataDiffResult, OutputHash, StageExplanation, StageStatus

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary
    from pivot.types import OutputFormat

logger = logging.getLogger(__name__)


def _ensure_stages_registered() -> None:
    """Auto-discover and register stages if none are registered."""
    if not discovery.has_registered_stages():
        try:
            discovered = discovery.discover_and_register()
            if discovered:
                logger.info(f"Loaded pipeline from {discovered}")
        except discovery.DiscoveryError as e:
            raise click.ClickException(str(e)) from e


def _validate_stages(stages_list: list[str] | None, single_stage: bool) -> None:
    """Validate stage arguments and options."""
    if single_stage and not stages_list:
        raise click.ClickException("--single-stage requires at least one stage name")

    if stages_list:
        graph = registry.REGISTRY.build_dag(validate=True)
        registered = set(graph.nodes())
        unknown = [s for s in stages_list if s not in registered]
        if unknown:
            raise click.ClickException(f"Unknown stage(s): {', '.join(unknown)}")


def _get_all_explanations(
    stages_list: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
) -> list[StageExplanation]:
    """Get explanations for all stages in execution order."""
    from pivot import dag, explain, parameters

    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=single_stage)

    if not execution_order:
        return []

    resolved_cache_dir = cache_dir or project.get_project_root() / ".pivot" / "cache"
    overrides = parameters.load_params_yaml()

    explanations = list[StageExplanation]()
    for stage_name in execution_order:
        stage_info = registry.REGISTRY.get(stage_name)
        explanation = explain.get_stage_explanation(
            stage_name,
            stage_info["fingerprint"],
            stage_info["deps"],
            stage_info["params"],
            overrides,
            resolved_cache_dir,
        )
        explanations.append(explanation)

    return explanations


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Fast pipeline execution with per-stage caching."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.argument("stages", nargs=-1)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would run without executing")
@click.option(
    "--explain", "-e", is_flag=True, help="Show detailed breakdown of why stages would run"
)
@click.option(
    "--watch",
    "-w",
    is_flag=False,
    flag_value="",
    default=None,
    metavar="GLOBS",
    help="Watch for file changes and re-run. Optionally specify comma-separated glob patterns.",
)
@click.option("--debounce", type=int, default=300, help="Debounce delay in milliseconds")
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    single_stage: bool,
    cache_dir: pathlib.Path | None,
    dry_run: bool,
    explain: bool,
    watch: str | None,
    debounce: int,
) -> None:
    """Execute pipeline stages.

    If STAGES are provided, runs those stages and their dependencies.
    Use --single-stage to run only the specified stages without dependencies.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    # Handle dry-run modes (with or without explain)
    if dry_run:
        if explain:
            # --dry-run --explain: detailed explanation without execution
            ctx.invoke(explain_cmd, stages=stages, single_stage=single_stage, cache_dir=cache_dir)
        else:
            # --dry-run only: terse output
            ctx.invoke(dry_run_cmd, stages=stages, single_stage=single_stage, cache_dir=cache_dir)
        return

    if watch is not None:
        from pivot import watch as watch_module

        # Parse comma-separated globs if provided
        watch_globs = [g.strip() for g in watch.split(",") if g.strip()] if watch else None

        watch_module.run_watch_loop(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            watch_globs=watch_globs,
            debounce_ms=debounce,
        )
        return

    # Normal execution (with optional explain mode)
    try:
        results = executor.run(
            stages=stages_list,
            single_stage=single_stage,
            cache_dir=cache_dir,
            explain_mode=explain,
        )
        if not results:
            click.echo("No stages to run")
        elif not explain:
            _print_results(results)
    except Exception as e:
        raise click.ClickException(repr(e)) from e


@cli.command("dry-run")
@click.argument("stages", nargs=-1)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
def dry_run_cmd(
    stages: tuple[str, ...], single_stage: bool, cache_dir: pathlib.Path | None
) -> None:
    """Show what would run without executing."""
    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    try:
        explanations = _get_all_explanations(stages_list, single_stage, cache_dir)

        if not explanations:
            click.echo("No stages to run")
            return

        click.echo("Would run:")
        for exp in explanations:
            status = "would run" if exp["will_run"] else "would skip"
            reason = exp["reason"] or "unchanged"
            click.echo(f"  {exp['stage_name']}: {status} ({reason})")

    except Exception as e:
        raise click.ClickException(repr(e)) from e


@cli.command("explain")
@click.argument("stages", nargs=-1)
@click.option(
    "--single-stage",
    "-s",
    is_flag=True,
    help="Run only the specified stages (in provided order), not their dependencies",
)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
def explain_cmd(
    stages: tuple[str, ...], single_stage: bool, cache_dir: pathlib.Path | None
) -> None:
    """Show detailed breakdown of why stages would run."""
    from pivot import console

    _ensure_stages_registered()
    stages_list = list(stages) if stages else None
    _validate_stages(stages_list, single_stage)

    try:
        explanations = _get_all_explanations(stages_list, single_stage, cache_dir)

        if not explanations:
            click.echo("No stages to run")
            return

        con = console.Console()
        for exp in explanations:
            con.explain_stage(exp)

        will_run = sum(1 for e in explanations if e["will_run"])
        con.explain_summary(will_run, len(explanations) - will_run)

    except Exception as e:
        raise click.ClickException(repr(e)) from e


@cli.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List registered stages."""
    verbose = ctx.obj.get("verbose", False)
    stage_list = registry.REGISTRY.list_stages()

    if not stage_list:
        click.echo("No stages registered")
        return

    click.echo(f"Registered stages ({len(stage_list)}):")
    for name in stage_list:
        info = registry.REGISTRY.get(name)
        deps = info["deps"]
        outs = info["outs_paths"]
        click.echo(f"  {name}")
        if verbose:
            click.echo(f"    deps: {deps}")
            click.echo(f"    outs: {outs}")


@cli.command()
@click.argument("stages", nargs=-1)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default="dvc.yaml",
    help="Output path for dvc.yaml (default: dvc.yaml)",
)
def export(stages: tuple[str, ...], output: pathlib.Path) -> None:
    """Export pipeline to DVC YAML format."""
    from pivot import dvc_compat

    stages_list = list(stages) if stages else None

    try:
        result = dvc_compat.export_dvc_yaml(output, stages=stages_list)
        click.echo(f"Exported {len(result['stages'])} stages to {output}")
    except Exception as e:
        raise click.ClickException(repr(e)) from e


@cli.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing .pvt files")
def track(paths: tuple[str, ...], force: bool) -> None:
    """Track files/directories for caching.

    Creates .pvt manifest files and caches content for reproducibility.
    """
    try:
        project_root = project.get_project_root()
        cache_dir = project_root / ".pivot" / "cache" / "files"

        # Get all stage outputs for overlap detection
        stage_outputs = _get_all_stage_outputs()

        # Discover existing .pvt files
        existing_tracked = pvt.discover_pvt_files(project_root)

        for path_str in paths:
            _track_single_path(path_str, cache_dir, stage_outputs, existing_tracked, force)
    except Exception as e:
        raise click.ClickException(repr(e)) from e


def _paths_overlap(path_a: pathlib.Path, path_b: pathlib.Path) -> bool:
    """Check if two paths overlap (same location or parent/child relationship)."""
    # Exact match - use samefile if both exist (handles hardlinks, case-insensitivity)
    if path_a.exists() and path_b.exists():
        try:
            if path_a.samefile(path_b):
                return True
        except OSError:
            pass

    if path_a == path_b:
        return True

    return path_a.is_relative_to(path_b) or path_b.is_relative_to(path_a)


def _get_all_stage_outputs() -> dict[str, pathlib.Path]:
    """Get stage outputs with resolved paths for overlap detection.

    Returns:
        Dict mapping normalized path -> resolved path

    Raises:
        click.ClickException: If any stage output has unresolvable issues
            (permission errors, circular symlinks, etc.)

    Note:
        Resolves all paths upfront and fails fast if any are problematic.
        This provides clear error messages showing ALL issues at once.
    """
    outputs_normalized = set[str]()
    for stage_name in registry.REGISTRY.list_stages():
        info = registry.REGISTRY.get(stage_name)
        outputs_normalized.update(info.get("outs_paths", []))

    # Pre-resolve all stage outputs with explicit error handling
    outputs_resolved = dict[str, pathlib.Path]()
    resolution_errors = list[str]()

    for norm_path in outputs_normalized:
        try:
            resolved = project.resolve_path_for_comparison(norm_path, "stage output")
            outputs_resolved[norm_path] = resolved
        except (PermissionError, RuntimeError, OSError) as e:
            # Collect errors to show all problems at once
            resolution_errors.append(f"  - {norm_path}: {e}")

    if resolution_errors:
        raise click.ClickException(
            "Cannot resolve stage outputs (fix these before tracking):\n"
            + "\n".join(resolution_errors)
        )

    return outputs_resolved


def _track_single_path(
    path_str: str,
    cache_dir: pathlib.Path,
    stage_outputs_resolved: dict[str, pathlib.Path],
    existing_tracked: dict[str, pvt.PvtData],
    force: bool,
) -> None:
    """Track a single file or directory.

    Args:
        path_str: Path to track (user input)
        cache_dir: Cache directory path
        stage_outputs_resolved: Dict mapping normalized output paths to resolved paths
        existing_tracked: Dict of already tracked files
        force: Whether to overwrite existing .pvt files
    """
    # Validate path doesn't escape project
    if pvt.has_path_traversal(path_str):
        raise click.ClickException(f"Path traversal not allowed: {path_str}")

    # Normalize path (preserve symlinks) for consistency with registry/pvt
    path = project.normalize_path(path_str)
    abs_path_str = str(path)

    # Check for broken symlinks (exist as symlinks but target doesn't exist)
    if path.is_symlink() and not path.exists():
        raise click.ClickException(f"Path '{path_str}' is a broken symlink (target does not exist)")

    # Check file exists
    if not path.exists():
        raise click.ClickException(f"Path not found: {path_str}")

    # Warn if tracking file inside symlinked directory
    project_root = project.get_project_root()
    if project.contains_symlink_in_path(path, project_root):
        click.echo(
            f"Warning: '{path_str}' is inside a symlinked directory. "
            + "Tracked path may not be portable across environments.",
            err=True,
        )

    # Check for overlap with stage outputs (resolve paths to detect symlink aliasing)
    try:
        user_resolved = project.resolve_path_for_comparison(path_str, "user path")
    except (PermissionError, RuntimeError, OSError) as e:
        raise click.ClickException(repr(e)) from e

    for out_norm, out_resolved in stage_outputs_resolved.items():
        if _paths_overlap(user_resolved, out_resolved):
            # Provide helpful error showing both normalized and resolved if different
            if str(user_resolved) != abs_path_str or str(out_resolved) != out_norm:
                # Symlink aliasing detected
                raise click.ClickException(
                    f"Cannot track '{path_str}' (resolves to '{user_resolved}'): "
                    + f"overlaps with stage output '{out_norm}' (resolves to '{out_resolved}')"
                )
            else:
                # Direct overlap
                raise click.ClickException(
                    f"Cannot track '{path_str}': overlaps with stage output '{out_norm}'"
                )

    # Check for duplicate tracking
    pvt_path = pvt.get_pvt_path(path)
    if abs_path_str in existing_tracked and not force:
        raise click.ClickException(f"'{path_str}' is already tracked. Use --force to update.")

    # Hash and cache
    if path.is_dir():
        tree_hash, manifest = cache.hash_directory(path)
        total_size = sum(e["size"] for e in manifest)
        num_files = len(manifest)

        # Save each file to cache
        for entry in manifest:
            file_path = path / entry["relpath"]
            file_cache_path = cache.get_cache_path(cache_dir, entry["hash"])
            if not file_cache_path.exists():
                cache.copy_to_cache(file_path, file_cache_path)

        pvt_data: pvt.PvtData = {
            "path": path.name,
            "hash": tree_hash,
            "size": total_size,
            "num_files": num_files,
            "manifest": manifest,
        }
    else:
        file_hash = cache.hash_file(path)
        file_size = path.stat().st_size
        file_cache_path = cache.get_cache_path(cache_dir, file_hash)
        if not file_cache_path.exists():
            cache.copy_to_cache(path, file_cache_path)

        pvt_data = {
            "path": path.name,
            "hash": file_hash,
            "size": file_size,
        }

    # Write .pvt file
    pvt.write_pvt_file(pvt_path, pvt_data)

    # Update existing_tracked for subsequent paths
    existing_tracked[abs_path_str] = pvt_data

    click.echo(f"Tracked: {path_str}")


@cli.command()
@click.argument("targets", nargs=-1)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def checkout(targets: tuple[str, ...], checkout_mode: str | None, force: bool) -> None:
    """Restore tracked files and stage outputs from cache.

    If no targets specified, restores all tracked files and stage outputs.
    """
    try:
        project_root = project.get_project_root()
        cache_dir = project_root / ".pivot" / "cache" / "files"

        # Determine checkout modes - CLI flag overrides config (single mode, no fallback)
        mode_strings = [checkout_mode] if checkout_mode else config.get_checkout_mode_order()
        checkout_modes = [cache.CheckoutMode(m) for m in mode_strings]

        # Discover tracked files
        tracked_files = pvt.discover_pvt_files(project_root)

        # Get stage output info from lock files
        stage_outputs = _get_stage_output_info(project_root)

        if not targets:
            # Restore everything
            _checkout_all_tracked(tracked_files, cache_dir, checkout_modes, force)
            _checkout_all_outputs(stage_outputs, cache_dir, checkout_modes, force)
        else:
            # Restore specific targets
            for target in targets:
                _checkout_target(
                    target, tracked_files, stage_outputs, cache_dir, checkout_modes, force
                )
    except Exception as e:
        raise click.ClickException(repr(e)) from e


def _get_stage_output_info(project_root: pathlib.Path) -> dict[str, OutputHash]:
    """Get output hash info from lock files for all stages."""
    from pivot import lock

    outputs = dict[str, OutputHash]()
    cache_dir = project_root / ".pivot" / "cache"

    for stage_name in registry.REGISTRY.list_stages():
        stage_lock = lock.StageLock(stage_name, cache_dir)
        lock_data = stage_lock.read()
        if lock_data and "output_hashes" in lock_data:
            for out_path, out_hash in lock_data["output_hashes"].items():
                # Normalize paths for backward compatibility with old lock files
                # (old lock files may have resolved paths, new ones have normalized)
                norm_path = str(project.normalize_path(out_path))
                outputs[norm_path] = out_hash

    return outputs


def _checkout_all_tracked(
    tracked_files: dict[str, pvt.PvtData],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore all tracked files."""
    for abs_path_str, pvt_data in tracked_files.items():
        path = pathlib.Path(abs_path_str)
        output_hash = _pvt_to_output_hash(pvt_data)
        _restore_path(path, output_hash, cache_dir, checkout_modes, force)


def _checkout_all_outputs(
    stage_outputs: dict[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore all stage outputs."""
    for abs_path_str, output_hash in stage_outputs.items():
        if output_hash is None:
            continue
        path = pathlib.Path(abs_path_str)
        _restore_path(path, output_hash, cache_dir, checkout_modes, force)


def _checkout_target(
    target: str,
    tracked_files: dict[str, pvt.PvtData],
    stage_outputs: dict[str, OutputHash],
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore a specific target."""
    # Validate path doesn't escape project
    if pvt.has_path_traversal(target):
        raise click.ClickException(f"Path traversal not allowed: {target}")

    # Use normalized path (preserve symlinks) to match keys in tracked_files/stage_outputs
    abs_path = project.normalize_path(target)
    abs_path_str = str(abs_path)

    # Check if it's a tracked file
    if abs_path_str in tracked_files:
        pvt_data = tracked_files[abs_path_str]
        output_hash = _pvt_to_output_hash(pvt_data)
        _restore_path(abs_path, output_hash, cache_dir, checkout_modes, force)
        return

    # Check if it's a stage output
    if abs_path_str in stage_outputs:
        output_hash = stage_outputs[abs_path_str]
        if output_hash is None:
            raise click.ClickException(f"'{target}' has no cached version")
        _restore_path(abs_path, output_hash, cache_dir, checkout_modes, force)
        return

    # Unknown target
    raise click.ClickException(f"'{target}' is not a tracked file or stage output")


def _pvt_to_output_hash(pvt_data: pvt.PvtData) -> OutputHash:
    """Convert PvtData to OutputHash format."""
    manifest = pvt_data.get("manifest")
    if manifest is not None:
        return {"hash": pvt_data["hash"], "manifest": manifest}
    return {"hash": pvt_data["hash"]}


def _restore_path(
    path: pathlib.Path,
    output_hash: OutputHash,
    cache_dir: pathlib.Path,
    checkout_modes: list[cache.CheckoutMode],
    force: bool,
) -> None:
    """Restore a file or directory from cache."""
    if path.exists() and not force:
        click.echo(f"Skipped: {path.name} (already exists)")
        return

    if force and path.exists():
        cache.remove_output(path)

    success = cache.restore_from_cache(path, output_hash, cache_dir, checkout_modes=checkout_modes)
    if not success:
        raise click.ClickException(f"Failed to restore '{path.name}': not found in cache")

    click.echo(f"Restored: {path.name}")


@cli.command("get")
@click.argument("targets", nargs=-1, required=True)
@click.option(
    "--rev",
    "-r",
    required=True,
    help="Git revision (SHA, branch, tag) to retrieve files from",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output path for single file target (incompatible with multiple targets or stage names)",
)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def get_cmd(
    targets: tuple[str, ...],
    rev: str,
    output: pathlib.Path | None,
    checkout_mode: str | None,
    force: bool,
) -> None:
    """Retrieve files or stage outputs from a specific git revision.

    TARGETS can be file paths or stage names.

    \b
    Examples:
      pivot get --rev v1.0 model.pkl              # Get file from tag
      pivot get --rev v1.0 model.pkl -o old.pkl   # Get file to alternate location
      pivot get --rev abc123 train                # Get all outputs from stage
    """
    try:
        project_root = project.get_project_root()
        cache_dir = project_root / ".pivot" / "cache"

        # Determine checkout modes
        mode_strings = [checkout_mode] if checkout_mode else config.get_checkout_mode_order()
        checkout_modes = [cache.CheckoutMode(m) for m in mode_strings]

        messages = get.restore_targets_from_revision(
            targets=list(targets),
            rev=rev,
            output=output,
            cache_dir=cache_dir,
            checkout_modes=checkout_modes,
            force=force,
        )

        for msg in messages:
            click.echo(msg)

    except Exception as e:
        raise click.ClickException(repr(e)) from e


def _print_results(results: dict[str, ExecutionSummary]) -> None:
    """Print execution results in a readable format."""
    ran = 0
    skipped = 0
    failed = 0

    for name, result in results.items():
        result_status = result["status"]
        reason = result["reason"]

        if result_status == StageStatus.RAN:
            ran += 1
            click.echo(f"{name}: ran ({reason})")
        elif result_status == StageStatus.FAILED:
            failed += 1
            click.echo(f"{name}: failed ({reason})")
        else:
            skipped += 1
            if reason:
                click.echo(f"{name}: skipped ({reason})")
            else:
                click.echo(f"{name}: skipped")

    parts = [f"{ran} ran", f"{skipped} skipped"]
    if failed > 0:
        parts.append(f"{failed} failed")
    click.echo(f"\nTotal: {', '.join(parts)}")


@cli.group()
def metrics() -> None:
    """Display and compare metrics."""


@metrics.command("show")
@click.argument("targets", nargs=-1)
@click.option("--json", "output_format", flag_value="json", default=None, help="Output as JSON")
@click.option("--md", "output_format", flag_value="md", help="Output as Markdown table")
@click.option("-R", "--recursive", is_flag=True, help="Search directories recursively")
@click.option("--precision", default=5, type=int, help="Decimal precision for floats")
def metrics_show(
    targets: tuple[str, ...],
    output_format: OutputFormat,
    recursive: bool,
    precision: int,
) -> None:
    """Display metric values in tabular format.

    If TARGETS are specified, parses those files/directories directly.
    Otherwise, shows metrics from all registered stages' Metric outputs.
    """
    from pivot import metrics as metrics_module

    try:
        if targets:
            all_metrics = metrics_module.collect_metrics_from_files(list(targets), recursive)
        else:
            all_metrics = metrics_module.collect_all_stage_metrics_flat()

        output = metrics_module.format_metrics_table(all_metrics, output_format, precision)
        click.echo(output)
    except metrics_module.MetricsError as e:
        raise click.ClickException(str(e)) from e


@metrics.command("diff")
@click.argument("targets", nargs=-1)
@click.option("--json", "output_format", flag_value="json", default=None, help="Output as JSON")
@click.option("--md", "output_format", flag_value="md", help="Output as Markdown table")
@click.option("-R", "--recursive", is_flag=True, help="Search directories recursively")
@click.option("--no-path", is_flag=True, help="Hide path column")
@click.option("--precision", default=5, type=int, help="Decimal precision for floats")
def metrics_diff(
    targets: tuple[str, ...],
    output_format: OutputFormat,
    recursive: bool,
    no_path: bool,
    precision: int,
) -> None:
    """Compare workspace metric files against git HEAD.

    If TARGETS are specified, compares those files/directories.
    Otherwise, compares all registered stages' Metric outputs.
    """
    from pivot import metrics as metrics_module

    try:
        # Get HEAD info (hashes from lock files)
        head_info = metrics_module.get_metric_info_from_head()

        if not head_info:
            click.echo("No metrics found in registered stages.")
            return

        # Filter to targets if specified
        if targets:
            proj_root = project.get_project_root()
            target_set = {
                project.to_relative_path(project.normalize_path(t), proj_root) for t in targets
            }
            head_info = {k: v for k, v in head_info.items() if k in target_set}
            paths = list(target_set)
        else:
            paths = list(head_info.keys())

        # Get metrics from HEAD (cache-first, git-fallback)
        head_metrics = metrics_module.collect_metrics_from_head(paths, head_info)

        # Get current workspace metrics
        workspace_metrics = metrics_module.collect_metrics_from_files(paths, recursive)

        diffs = metrics_module.diff_metrics(head_metrics, workspace_metrics)
        output = metrics_module.format_diff_table(
            diffs, output_format, precision, show_path=not no_path
        )
        click.echo(output)
    except metrics_module.MetricsError as e:
        raise click.ClickException(str(e)) from e


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
    )


# =============================================================================
# Plots Commands
# =============================================================================


@cli.group()
def plots() -> None:
    """Display and compare plots."""


@plots.command("show")
@click.argument("targets", nargs=-1)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=pathlib.Path),
    default="pivot_plots/index.html",
    help="Output HTML path (default: pivot_plots/index.html)",
)
@click.option("--open", "open_browser", is_flag=True, help="Open browser after rendering")
def plots_show(targets: tuple[str, ...], output: pathlib.Path, open_browser: bool) -> None:
    """Render plots as HTML image gallery."""
    from pivot import plots as plots_mod

    try:
        if targets:
            # Filter to specific targets
            all_plots = plots_mod.collect_plots_from_stages()
            target_set = set(targets)
            plot_list = [p for p in all_plots if p["path"] in target_set]
        else:
            plot_list = plots_mod.collect_plots_from_stages()

        if not plot_list:
            click.echo("No plots found.")
            return

        output_path = plots_mod.render_plots_html(plot_list, output)
        click.echo(f"Rendered {len(plot_list)} plot(s) to {output_path}")

        if open_browser:
            import webbrowser

            webbrowser.open(f"file://{output_path.resolve()}")
    except Exception as e:
        raise click.ClickException(repr(e)) from e


@plots.command("diff")
@click.argument("targets", nargs=-1)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--md", is_flag=True, help="Output as markdown table")
@click.option("--no-path", "no_path", is_flag=True, help="Hide path column")
def plots_diff(targets: tuple[str, ...], json_output: bool, md: bool, no_path: bool) -> None:
    """Show which plots changed since last commit."""
    from pivot import plots as plots_mod

    try:
        # Get old hashes from lock files at Git HEAD
        all_old_hashes = plots_mod.get_plot_hashes_from_head()

        if not all_old_hashes:
            click.echo("No plots found in registered stages.")
            return

        # Filter to targets if specified
        if targets:
            # Normalize user targets to relative paths from project root
            proj_root = project.get_project_root()
            target_set = {
                project.to_relative_path(project.normalize_path(t), proj_root) for t in targets
            }
            old_hashes = {k: v for k, v in all_old_hashes.items() if k in target_set}
            paths = list(target_set)
        else:
            old_hashes = all_old_hashes
            paths = list(old_hashes.keys())

        # Get current hashes from workspace
        new_hashes = plots_mod.get_plot_hashes_from_workspace(paths)

        # Compute diffs
        diffs = plots_mod.diff_plots(old_hashes, new_hashes)

        # Format output
        output_format: OutputFormat = "json" if json_output else ("md" if md else None)
        result = plots_mod.format_diff_table(diffs, output_format, show_path=not no_path)
        click.echo(result)
    except Exception as e:
        raise click.ClickException(repr(e)) from e


# =============================================================================
# Data Commands
# =============================================================================


@cli.group()
def data() -> None:
    """Inspect and compare data files."""


@data.command("diff")
@click.argument("targets", nargs=-1, required=True)
@click.option("--key", "key_cols", help="Comma-separated key columns for row matching")
@click.option("--positional", is_flag=True, help="Use positional (row-by-row) matching")
@click.option("--summary", is_flag=True, help="Show summary only (schema + counts)")
@click.option("--no-tui", is_flag=True, help="Print to stdout instead of launching TUI")
@click.option(
    "--json", "output_format", flag_value="json", help="Output as JSON (implies --no-tui)"
)
@click.option(
    "--md", "output_format", flag_value="md", help="Output as Markdown (implies --no-tui)"
)
@click.option("--max-rows", default=10000, help="Max rows for comparison (default: 10000)")
def data_diff(
    targets: tuple[str, ...],
    key_cols: str | None,
    positional: bool,
    summary: bool,
    no_tui: bool,
    output_format: OutputFormat,
    max_rows: int,
) -> None:
    """Compare data files in workspace against git HEAD.

    Compares CSV, JSON, and JSONL files showing schema changes, row additions,
    deletions, and modifications. Detects reorder-only changes.
    """
    from pivot import data as data_module

    try:
        # --json or --md implies --no-tui
        if output_format:
            no_tui = True

        # Parse key columns
        key_columns = [k.strip() for k in key_cols.split(",") if k.strip()] if key_cols else None

        # Validate conflicting options
        if key_columns and positional:
            raise click.ClickException("Cannot use both --key and --positional")

        # Get HEAD hashes from lock files
        head_hashes = data_module.get_data_hashes_from_head()
        if not head_hashes:
            click.echo("No data files found in registered stages.")
            return

        # Filter to targets
        proj_root = project.get_project_root()
        target_set = {
            project.to_relative_path(project.normalize_path(t), proj_root) for t in targets
        }
        filtered_head_hashes = {k: v for k, v in head_hashes.items() if k in target_set}

        # Get workspace hashes
        workspace_hashes = data_module.get_data_hashes_from_workspace(list(target_set))

        # Quick hash comparison to find changed files
        hash_diffs = data_module.diff_data_hashes(filtered_head_hashes, workspace_hashes)

        if not hash_diffs:
            click.echo("No data file changes detected.")
            return

        if no_tui or summary:
            # Non-interactive output
            diff_results = list[DataDiffResult]()
            temp_files = list[pathlib.Path]()
            try:
                for diff_entry in hash_diffs:
                    rel_path = diff_entry["path"]
                    abs_path = proj_root / rel_path
                    old_hash = diff_entry["old_hash"]

                    # Restore old file from cache if needed
                    old_path: pathlib.Path | None = None
                    if old_hash is not None:
                        old_path = data_module.restore_data_from_cache(rel_path, old_hash)
                        if old_path is not None:
                            temp_files.append(old_path)
                    new_path = abs_path if abs_path.exists() else None

                    # When --positional is set, don't use key columns
                    effective_keys = None if positional else key_columns
                    result = data_module.diff_data_files(
                        old_path=old_path,
                        new_path=new_path,
                        path_display=rel_path,
                        key_columns=effective_keys,
                        max_rows=max_rows,
                    )
                    diff_results.append(result)

                # Format output
                output = data_module.format_diff_table(
                    diff_results,
                    output_format,
                )
                click.echo(output)
            finally:
                for temp_file in temp_files:
                    temp_file.unlink(missing_ok=True)
        else:
            # Launch TUI
            from pivot import data_tui

            data_tui.run_diff_app(
                diff_entries=hash_diffs,
                key_cols=key_columns,
                max_rows=max_rows,
            )
    except data_module.DataError as e:
        raise click.ClickException(str(e)) from e
    except Exception as e:
        raise click.ClickException(repr(e)) from e


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
