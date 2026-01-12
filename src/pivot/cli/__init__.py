from __future__ import annotations

import importlib
import logging
from typing import TypedDict, override

import click

# Command categories for organized help output
COMMAND_CATEGORIES = {
    "Pipeline": ["run", "explain", "status", "commit"],
    "Inspection": ["list", "metrics", "params", "plots", "data", "history", "show"],
    "Versioning": ["track", "checkout"],
    "Remote": ["remote", "push", "pull"],
    "Other": ["init", "export", "config", "completion", "schema", "check-ignore", "doctor"],
}

# Lazy command registry: command_name -> (module_path, attr_name, help_text)
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "init": ("pivot.cli.init", "init", "Initialize a new Pivot project."),
    "run": ("pivot.cli.run", "run", "Execute pipeline stages."),
    "explain": ("pivot.cli.run", "explain_cmd", "Show detailed breakdown of why stages would run."),
    "list": ("pivot.cli.list", "list_cmd", "List registered stages."),
    "export": ("pivot.cli.export", "export", "Export pipeline to DVC YAML format."),
    "track": ("pivot.cli.track", "track", "Track files/directories for caching."),
    "status": ("pivot.cli.status", "status", "Show pipeline, tracked files, and remote status."),
    "checkout": (
        "pivot.cli.checkout",
        "checkout",
        "Restore tracked files and stage outputs from cache.",
    ),
    "metrics": ("pivot.cli.metrics", "metrics", "Display and compare metrics."),
    "plots": ("pivot.cli.plots", "plots", "Display and compare plots."),
    "params": ("pivot.cli.params", "params", "Display and compare parameters."),
    "remote": ("pivot.cli.remote", "remote", "Manage remote storage for cache synchronization."),
    "push": ("pivot.cli.remote", "push", "Push cached outputs to remote storage."),
    "pull": ("pivot.cli.remote", "pull", "Pull cached outputs from remote storage."),
    "data": ("pivot.cli.data", "data", "Inspect and compare data files."),
    "completion": ("pivot.cli.completion", "completion_cmd", "Generate shell completion script."),
    "config": ("pivot.cli.config", "config_cmd", "View and modify Pivot configuration."),
    "history": ("pivot.cli.history", "history", "List recent pipeline runs."),
    "show": ("pivot.cli.history", "show_cmd", "Show details of a specific run."),
    "schema": ("pivot.cli.schema", "schema", "Output JSON Schema for pivot.yaml configuration."),
    "commit": ("pivot.cli.commit", "commit_command", "Commit pending locks from --no-commit runs."),
    "check-ignore": (
        "pivot.cli.check_ignore",
        "check_ignore",
        "Check if paths are ignored by .pivotignore.",
    ),
    "doctor": ("pivot.cli.doctor", "doctor", "Check environment and configuration for issues."),
}


class CliContext(TypedDict):
    """Context object for CLI commands."""

    verbose: bool
    quiet: bool


class PivotGroup(click.Group):
    """Custom Group with lazy command loading and categorized help."""

    @override
    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return all available command names."""
        return sorted(_LAZY_COMMANDS.keys())

    @override
    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Lazily load and return a command by name."""
        if cmd_name not in _LAZY_COMMANDS:
            return None

        module_path, attr_name, _help = _LAZY_COMMANDS[cmd_name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    @override
    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format commands grouped by category using cached help strings."""
        commands: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            if name not in _LAZY_COMMANDS:
                continue
            _module, _attr, help_text = _LAZY_COMMANDS[name]
            commands.append((name, help_text))

        if not commands:
            return

        categorized: dict[str, list[tuple[str, str]]] = {cat: [] for cat in COMMAND_CATEGORIES}
        uncategorized: list[tuple[str, str]] = []

        for name, help_text in commands:
            found = False
            for cat, cmd_names in COMMAND_CATEGORIES.items():
                if name in cmd_names:
                    categorized[cat].append((name, help_text))
                    found = True
                    break
            if not found:
                uncategorized.append((name, help_text))

        for category, cmds in categorized.items():
            if not cmds:
                continue
            with formatter.section(f"{category} Commands"):
                formatter.write_dl(cmds)

        if uncategorized:
            with formatter.section("Other Commands"):
                formatter.write_dl(uncategorized)


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging for CLI output."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(message)s", force=True)


@click.group(cls=PivotGroup)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """Fast pipeline execution with per-stage caching.

    Pivot accelerates ML pipelines with automatic change detection,
    parallel execution, and smart caching.
    """
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")
    ctx.obj = CliContext(verbose=verbose, quiet=quiet)
    _setup_logging(verbose, quiet)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
