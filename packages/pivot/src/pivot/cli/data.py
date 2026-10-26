from __future__ import annotations

import contextlib
import logging
import pathlib

import click

from pivot import config, discovery, exceptions, project, registry
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.cli import targets as cli_targets
from pivot.remote import sync as transfer
from pivot.storage import cache, restore
from pivot.types import DataDiffResult, OutputFormat

logger = logging.getLogger(__name__)


@cli_decorators.pivot_command(allow_all=True)
@click.argument("targets", nargs=-1, required=True)
@click.option("--key", "key_cols", help="Comma-separated key columns for row matching")
@click.option("--positional", is_flag=True, help="Use positional (row-by-row) matching")
@click.option("--summary", is_flag=True, help="Show summary only (schema + counts)")
@click.option("--no-tui", is_flag=True, help="Print to stdout instead of launching TUI")
@click.option(
    "--json",
    "output_format",
    flag_value=OutputFormat.JSON,
    help="Output as JSON (implies --no-tui)",
)
@click.option(
    "--md",
    "output_format",
    flag_value=OutputFormat.MD,
    help="Output as Markdown (implies --no-tui)",
)
@click.option(
    "--max-rows", default=None, type=click.IntRange(min=1), help="Max rows for comparison"
)
def diff(
    targets: tuple[str, ...],
    key_cols: str | None,
    positional: bool,
    summary: bool,
    no_tui: bool,
    output_format: OutputFormat | None,
    max_rows: int | None,
) -> None:
    """Compare data files in workspace against git HEAD.

    Compares CSV, JSON, and JSONL files showing schema changes, row additions,
    deletions, and modifications. Detects reorder-only changes.
    """
    from pivot.show import data as data_module
    from pivot.show import selection as selection_module

    max_rows = max_rows if max_rows is not None else config.get_diff_max_rows()

    # --json or --md implies --no-tui
    if output_format:
        no_tui = True

    # Parse key columns
    key_columns = [k.strip() for k in key_cols.split(",") if k.strip()] if key_cols else None

    # Validate conflicting options
    if key_columns and positional:
        raise click.ClickException("Cannot use both --key and --positional")

    # One selection decides, for HEAD and the workspace, the data files the targets name
    selection = selection_module.select_data_files(targets)
    if selection is None:
        click.echo("No data files found in registered stages.")
        return

    # Quick hash comparison to find changed files
    hash_diffs = data_module.diff_data_hashes(
        selection["head_hashes"], selection["workspace_hashes"]
    )

    if not hash_diffs:
        if output_format == OutputFormat.JSON:
            click.echo("[]")
        else:
            click.echo("No data file changes detected.")
        return

    run_app = None
    if not (no_tui or summary):
        try:
            from pivot_tui import diff as data_tui
        except ImportError as err:
            raise click.UsageError(
                "The TUI requires the 'pivot-tui' package. Install it with: pip install 'pivot[tui]' or: uv pip install pivot-tui"
            ) from err
        run_app = data_tui.run_diff_app

    # When --positional is set, don't use key columns
    effective_keys = None if positional else key_columns
    diff_results = list[DataDiffResult]()
    temp_files = list[pathlib.Path]()
    try:
        for diff_entry in hash_diffs:
            rel_path = diff_entry["path"]
            old_hash = diff_entry["old_hash"]

            # Restore old file from cache if needed
            old_path: pathlib.Path | None = None
            if old_hash is not None:
                old_path = data_module.restore_data_from_cache(rel_path, old_hash)
                if old_path is None:
                    logger.warning(
                        "%s: its git HEAD content is not in the local cache; run pivot fetch to compare it",
                        rel_path,
                    )
                else:
                    temp_files.append(old_path)
            result = data_module.diff_data_files(
                old_path=old_path,
                new_path=selection["workspace_paths"].get(rel_path),
                path_display=rel_path,
                change_type=diff_entry["change_type"],
                key_columns=effective_keys,
                max_rows=max_rows,
            )
            diff_results.append(result)

        if run_app is None:
            click.echo(data_module.format_diff_table(diff_results, output_format))
        else:
            run_app(diff_results)
    finally:
        for temp_file in temp_files:
            with contextlib.suppress(OSError):
                temp_file.unlink(missing_ok=True)


@cli_decorators.pivot_command(auto_discover=False)
@click.argument("targets", nargs=-1, required=True)
@click.option(
    "--rev",
    "-r",
    required=True,
    help="Git revision (SHA, branch, tag) to retrieve files from",
)
@click.option("--remote", "remote_name", help="Remote name (uses default if not specified)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output path when the targets name a single file (one target with one output)",
)
@click.option(
    "--checkout-mode",
    type=click.Choice(["symlink", "hardlink", "copy"]),
    default=None,
    help="Checkout mode for restoration (default: project config or hardlink)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def get(
    targets: tuple[str, ...],
    rev: str,
    remote_name: str | None,
    output: pathlib.Path | None,
    checkout_mode: str | None,
    force: bool,
) -> None:
    """Retrieve files or stage outputs from a specific git revision.

    TARGETS can be file paths or stage outputs (``stage`` or ``stage:key``).

    \b
    Examples:
      pivot get --rev v1.0 model.pkl              # Get file from tag
      pivot get --rev v1.0 model.pkl -o old.pkl   # Get file to alternate location
      pivot get --rev abc123 train                # Get all outputs from stage
    """
    cache_dir = config.get_cache_dir()

    checkout_modes = (
        [cache.CheckoutMode(checkout_mode)] if checkout_mode else config.get_checkout_mode_order()
    )

    # Stage targets need the pipeline; files and `.pvt` targets work from the revision alone,
    # so a pipeline.py that fails to import (pipeline.py itself is a target to restore) only
    # stops targets from resolving as stages.
    pipeline = cli_decorators.get_pipeline_from_context()
    if pipeline is None:
        try:
            pipeline = discovery.discover_pipeline()
        except (discovery.DiscoveryError, exceptions.PipelineConfigError) as e:
            logger.warning(f"Pipeline not loaded, so every target is read as a file: {e}")
        if pipeline is not None:
            cli_decorators.store_pipeline_in_context(pipeline)
    all_stages = {} if pipeline is None else cli_helpers.get_all_stages()
    proj_root = project.get_project_root()
    default_state_dir = config.get_state_dir()
    stage_targets = dict[str, restore.StageTarget]()
    for target in targets:
        resolved = cli_targets.resolve_cli_target(target, all_stages)
        if resolved is None:
            continue
        store = cli_helpers.require_workspace_store()
        stage_name = resolved["stage_name"]
        stage_targets[target] = restore.StageTarget(
            stage_name=stage_name,
            state_dir=registry.get_stage_state_dir(all_stages[stage_name], default_state_dir),
            paths_by_key={
                ref.identity.key: project.to_relative_path(
                    store.resolve_display_path(ref), proj_root
                )
                for ref in resolved["refs"]
            },
        )

    messages, success = restore.restore_targets_from_revision(
        targets=list(targets),
        rev=rev,
        output=output,
        cache_dir=cache_dir,
        stage_targets=stage_targets,
        checkout_modes=checkout_modes,
        force=force,
        remote_factory=lambda: transfer.create_remote_from_name(remote_name)[0],
    )

    for msg in messages:
        click.echo(msg)

    if not success:
        raise SystemExit(1)
