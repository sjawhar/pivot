from __future__ import annotations

import json
import sys

import click

from pivot import ignore, project
from pivot.cli import completion
from pivot.cli import decorators as cli_decorators


@cli_decorators.pivot_command("check-ignore", auto_discover=False)
@click.argument("targets", nargs=-1, shell_complete=completion.complete_targets)
@click.option("--details", "-d", is_flag=True, help="Show matching pattern and source")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--show-defaults", is_flag=True, help="Show default patterns for starter .pivotignore"
)
def check_ignore(
    targets: tuple[str, ...],
    details: bool,
    as_json: bool,
    show_defaults: bool,
) -> None:
    """Check if paths are ignored by .pivotignore.

    Supports full gitignore syntax including negation (!pattern).
    Exit code 0 if any target is ignored, 1 if none are ignored.

    Examples:

        pivot check-ignore app.log

        pivot check-ignore --details *.pyc

        pivot check-ignore --json build/ temp.log

        pivot check-ignore --show-defaults
    """
    if show_defaults:
        _show_default_patterns(as_json)
        return

    if not targets:
        click.echo("No targets specified. Use --show-defaults to see default patterns.", err=True)
        sys.exit(2)

    project_root = project.get_project_root()
    ignore_filter = ignore.IgnoreFilter(project_root=project_root)

    results = [ignore_filter.check_ignore(target) for target in targets]

    if as_json:
        _output_json(results)
    else:
        _output_text(results, details)

    # Exit code 0 if any target is ignored, 1 if none are ignored
    any_ignored = any(r.ignored for r in results)
    sys.exit(0 if any_ignored else 1)


def _show_default_patterns(as_json: bool) -> None:
    """Show default patterns for starter .pivotignore."""
    patterns = ignore.get_default_patterns()

    if as_json:
        click.echo(json.dumps({"default_patterns": patterns}))
    else:
        click.echo("Default patterns for .pivotignore:\n")
        for pattern in patterns:
            click.echo(pattern)


def _output_json(results: list[ignore.CheckIgnoreResult]) -> None:
    """Output results as JSON."""
    data = [
        {
            "path": r.path,
            "ignored": r.ignored,
            "pattern": r.pattern,
            "source": r.source,
        }
        for r in results
    ]
    click.echo(json.dumps(data))


def _output_text(results: list[ignore.CheckIgnoreResult], details: bool) -> None:
    """Output results as text."""
    for result in results:
        if result.ignored:
            if details:
                click.echo(f"{result.source}\t{result.pattern}\t{result.path}")
            else:
                click.echo(result.path)
