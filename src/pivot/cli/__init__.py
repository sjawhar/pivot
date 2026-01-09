from __future__ import annotations

import logging

import click


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", force=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Fast pipeline execution with per-stage caching."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# Import and register commands - must be after cli definition to avoid circular imports
from pivot.cli import checkout as checkout_mod  # noqa: E402
from pivot.cli import data as data_mod  # noqa: E402
from pivot.cli import export as export_mod  # noqa: E402
from pivot.cli import get as get_mod  # noqa: E402
from pivot.cli import list as list_mod  # noqa: E402
from pivot.cli import metrics as metrics_mod  # noqa: E402
from pivot.cli import params as params_mod  # noqa: E402
from pivot.cli import plots as plots_mod  # noqa: E402
from pivot.cli import remote as remote_mod  # noqa: E402
from pivot.cli import run as run_mod  # noqa: E402
from pivot.cli import status as status_mod  # noqa: E402
from pivot.cli import track as track_mod  # noqa: E402

cli.add_command(run_mod.run)
cli.add_command(run_mod.dry_run_cmd, name="dry-run")
cli.add_command(run_mod.explain_cmd, name="explain")
cli.add_command(list_mod.list_cmd, name="list")
cli.add_command(export_mod.export)
cli.add_command(track_mod.track)
cli.add_command(status_mod.status)
cli.add_command(checkout_mod.checkout)
cli.add_command(get_mod.get_cmd, name="get")
cli.add_command(metrics_mod.metrics)
cli.add_command(plots_mod.plots)
cli.add_command(params_mod.params)
cli.add_command(remote_mod.remote)
cli.add_command(remote_mod.push)
cli.add_command(remote_mod.pull)
cli.add_command(data_mod.data)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
