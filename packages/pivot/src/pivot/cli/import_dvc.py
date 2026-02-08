from __future__ import annotations

import pathlib

import click

from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers


@cli_decorators.pivot_command(auto_discover=False)
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=None,
    help="Path to dvc.yaml (default: auto-detect in current directory)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    default="pivot.yaml",
    help="Output path for pivot.yaml (default: pivot.yaml)",
)
@click.option(
    "--params",
    "-p",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=None,
    help="Path to params.yaml (default: auto-detect)",
)
@click.option(
    "--notes",
    "-n",
    type=click.Path(path_type=pathlib.Path),
    default=".pivot/migration-notes.md",
    help="Path for migration notes (default: .pivot/migration-notes.md)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be generated without writing files",
)
@click.pass_context
def import_dvc(
    ctx: click.Context,
    input_path: pathlib.Path | None,
    output: pathlib.Path,
    params: pathlib.Path | None,
    notes: pathlib.Path,
    force: bool,
    dry_run: bool,
) -> None:
    """Import DVC pipeline and convert to Pivot format.

    Reads dvc.yaml (and optionally dvc.lock, params.yaml) and generates
    pivot.yaml with migration notes for manual review.

    Shell commands in DVC stages cannot be automatically converted to Python
    functions. You must create corresponding Python functions and update
    the generated pivot.yaml manually.
    """
    from pivot import dvc_import, path_policy, project

    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]

    proj_root = project.get_project_root()

    # Auto-detect input file
    if input_path is None:
        input_path = _find_dvc_yaml(proj_root)
        if input_path is None:
            raise click.ClickException(
                "No dvc.yaml found in current directory. Use --input to specify the path."
            )

    # Auto-detect dvc.lock
    dvc_lock_path = input_path.parent / "dvc.lock"
    if not dvc_lock_path.exists():
        dvc_lock_path = None

    # Auto-detect params.yaml
    if params is None:
        params_path = input_path.parent / "params.yaml"
        if not params_path.exists():
            params_path = None
    else:
        params_path = params

    # Validate output paths
    path_policy.require_valid_path(
        str(output),
        path_policy.PathType.CLI_OUTPUT,
        proj_root,
        context="import-dvc --output",
    )
    path_policy.require_valid_path(
        str(notes),
        path_policy.PathType.CLI_OUTPUT,
        proj_root,
        context="import-dvc --notes",
    )

    # Convert pipeline
    if not quiet:
        click.echo(f"Importing from {input_path}...")

    result = dvc_import.convert_pipeline(
        dvc_yaml_path=input_path,
        dvc_lock_path=dvc_lock_path,
        params_yaml_path=params_path,
        project_root=proj_root,
    )

    stats = result["stats"]
    notes_list = result["notes"]

    # Report progress
    if not quiet:
        click.echo(f"  [OK] Converted {stats['stages_converted']} stages")
        if params_path:
            click.echo(f"  [OK] Inlined params from {params_path.name}")
        if stats["stages_with_shell_commands"] > 0:
            click.echo(
                f"  [WARN] {stats['stages_with_shell_commands']} stages use shell commands "
                + "(see migration notes)"
            )
        click.echo("  [INFO] First 'pivot run' will rebuild cache (DVC hashes not migrated)")

    if dry_run:
        if not quiet:
            click.echo("")
            click.echo("Dry run - no files written.")
            click.echo("")
            click.echo("Would create:")
            click.echo(f"  {output}")
            click.echo(f"  {notes}")
        return

    # Write output files
    dvc_import.write_pivot_yaml(result["stages"], output, force=force)
    dvc_import.write_migration_notes(notes_list, stats, notes, force=force)

    if not quiet:
        click.echo("")
        click.echo("Created:")
        click.echo(f"  {output} ({stats['stages_converted']} stages)")
        click.echo(f"  {notes}")
        click.echo("")
        click.echo("Next steps:")
        click.echo(f"  1. Review {notes} for required changes")
        click.echo("  2. Run 'pivot run --dry-run' to verify configuration")


def _find_dvc_yaml(root: pathlib.Path) -> pathlib.Path | None:
    """Find dvc.yaml in the given directory."""
    dvc_yaml = root / "dvc.yaml"
    if dvc_yaml.exists():
        return dvc_yaml
    return None
