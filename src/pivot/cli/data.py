from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click

from pivot import project
from pivot.cli import decorators as cli_decorators
from pivot.types import DataDiffResult

if TYPE_CHECKING:
    from pivot.types import OutputFormat


@click.group()
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
@cli_decorators.with_error_handling
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
    target_set = {project.to_relative_path(project.normalize_path(t), proj_root) for t in targets}
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
