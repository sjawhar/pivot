from __future__ import annotations

import builtins
import logging
from typing import override

import click

# Command categories for organized help output
COMMAND_CATEGORIES = {
    "Pipeline": ["run", "explain"],
    "Inspection": ["list", "metrics", "params", "plots", "data"],
    "Versioning": ["get", "track", "checkout"],
    "Remote": ["remote", "push", "pull"],
    "Other": ["export", "config", "completion"],
}


class PivotGroup(click.Group):
    """Custom Group that formats commands by category."""

    @override
    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format commands grouped by category."""
        commands = builtins.list[tuple[str, click.Command]]()
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            commands.append((subcommand, cmd))

        if not commands:
            return

        categorized: dict[str, list[tuple[str, click.Command]]] = {
            cat: [] for cat in COMMAND_CATEGORIES
        }
        uncategorized = builtins.list[tuple[str, click.Command]]()

        for name, cmd in commands:
            found = False
            for cat, cmd_names in COMMAND_CATEGORIES.items():
                if name in cmd_names:
                    categorized[cat].append((name, cmd))
                    found = True
                    break
            if not found:
                uncategorized.append((name, cmd))

        max_len = max(len(name) for name, _ in commands)
        limit = formatter.width - 6 - max_len

        for category, cmds in categorized.items():
            if not cmds:
                continue
            rows = [(name, cmd.get_short_help_str(limit)) for name, cmd in cmds]
            with formatter.section(f"{category} Commands"):
                formatter.write_dl(rows)

        if uncategorized:
            rows = [(name, cmd.get_short_help_str(limit)) for name, cmd in uncategorized]
            with formatter.section("Other Commands"):
                formatter.write_dl(rows)


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", force=True)


@click.group(cls=PivotGroup)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Fast pipeline execution with per-stage caching.

    Pivot accelerates ML pipelines with automatic change detection,
    parallel execution, and smart caching.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# Import and register commands - must be after cli definition to avoid circular imports
from pivot.cli import checkout as checkout_mod  # noqa: E402
from pivot.cli import completion as completion_mod  # noqa: E402
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
cli.add_command(completion_mod.completion_cmd, name="completion")


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
