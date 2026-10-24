from __future__ import annotations

import contextvars
import importlib
import logging
import sys
from typing import TextIO, TypedDict, override

import click

# Command categories for organized help output
COMMAND_CATEGORIES = {
    "Pipeline": ["run", "repro", "status", "verify", "commit"],
    "Sync": ["checkout", "fetch", "get", "import", "pull", "push", "remote", "track", "update"],
    "Inspection": ["dag", "diff", "history", "list", "metrics", "params", "plots", "show"],
    "Other": [
        "init",
        "config",
        "completion",
        "check-ignore",
        "doctor",
        "fingerprint",
    ],
}

# Lazy command registry: command_name -> (module_path, attr_name, help_text)
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "init": ("pivot.cli.init", "init", "Initialize a new Pivot project."),
    "run": ("pivot.cli.run", "run", "Execute pipeline stages."),
    "repro": ("pivot.cli.repro", "repro", "Reproduce pipeline with full DAG resolution."),
    "dag": ("pivot.cli.dag", "dag_cmd", "Visualize pipeline DAG."),
    "list": ("pivot.cli.list", "list_cmd", "List registered stages."),
    "import": (
        "pivot.cli.import_cmd",
        "import_cmd",
        "Import artifact from a remote Pivot repo.",
    ),
    "track": ("pivot.cli.track", "track", "Track files/directories for caching."),
    "status": ("pivot.cli.status", "status", "Show pipeline, tracked files, and remote status."),
    "verify": (
        "pivot.cli.verify",
        "verify",
        "Verify pipeline was reproduced and outputs are available.",
    ),
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
    "fetch": ("pivot.cli.remote", "fetch", "Fetch cached outputs from remote to local cache."),
    "pull": ("pivot.cli.remote", "pull", "Pull and restore outputs from remote storage."),
    "diff": ("pivot.cli.data", "diff", "Compare data files against git HEAD."),
    "get": ("pivot.cli.data", "get", "Retrieve files from a specific git revision."),
    "completion": ("pivot.cli.completion", "completion_cmd", "Generate shell completion script."),
    "config": ("pivot.cli.config", "config_cmd", "View and modify Pivot configuration."),
    "history": ("pivot.cli.history", "history", "List recent pipeline runs."),
    "show": ("pivot.cli.history", "show_cmd", "Show details of a specific run."),
    "commit": ("pivot.cli.commit", "commit_command", "Commit current workspace state for stages."),
    "check-ignore": (
        "pivot.cli.check_ignore",
        "check_ignore",
        "Check if paths are ignored by .pivotignore.",
    ),
    "doctor": ("pivot.cli.doctor", "doctor", "Check environment and configuration for issues."),
    "fingerprint": (
        "pivot.cli.fingerprint",
        "fingerprint",
        "Manage function fingerprinting cache.",
    ),
    "update": (
        "pivot.cli.update",
        "update",
        "Update imported artifacts from source repos.",
    ),
}


_standalone_invocation = contextvars.ContextVar("standalone_invocation", default=False)


def is_standalone_invocation() -> bool:
    """Return whether the console-script entry point invoked the current command."""
    return _standalone_invocation.get()


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
        # Build reverse lookup: command name -> category
        cmd_to_category: dict[str, str] = {}
        for category, cmd_names in COMMAND_CATEGORIES.items():
            for cmd_name in cmd_names:
                cmd_to_category[cmd_name] = category

        # Group commands by category
        categorized: dict[str, list[tuple[str, str]]] = {cat: [] for cat in COMMAND_CATEGORIES}
        uncategorized: list[tuple[str, str]] = []

        for name in self.list_commands(ctx):
            if name not in _LAZY_COMMANDS:
                continue
            _module, _attr, help_text = _LAZY_COMMANDS[name]
            category = cmd_to_category.get(name)
            if category:
                categorized[category].append((name, help_text))
            else:
                uncategorized.append((name, help_text))

        # Output each category
        for category, cmds in categorized.items():
            if cmds:
                with formatter.section(f"{category} Commands"):
                    formatter.write_dl(cmds)

        if uncategorized:
            with formatter.section("Other Commands"):
                formatter.write_dl(uncategorized)


class _StderrHandler(logging.StreamHandler[TextIO]):
    """Log handler that writes to whatever ``sys.stderr`` is when a record is emitted.

    ``logging.basicConfig`` binds the stderr object present at startup, so records bypass a
    redirect installed later; rich's live console swaps ``sys.stderr`` to print stderr writes
    above its pinned area.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    @override
    def stream(self) -> TextIO:  # pyright: ignore[reportIncompatibleVariableOverride] - read-only view of the current sys.stderr
        return sys.stderr


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging for CLI output."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


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
    # One invocation reads the config once: its state-directory record and the merged config
    # both start afresh, so an embedder's next invocation sees edits made in between. Like the
    # CLI's other warnings about project files, the state-directory warnings give way to
    # --quiet, and each invocation reports each problem once.
    if not ctx.resilient_parsing:
        from pivot import state_dir as state_dir_mod
        from pivot.config import io as config_io

        token = state_dir_mod.begin_invocation(report=not quiet)
        ctx.call_on_close(lambda: state_dir_mod.end_invocation(token))
        config_io.clear_config_cache()


def main() -> None:
    """Public CLI entry point for embedders. Serve mode changes no signal state here: the
    caller's handlers decide what SIGTERM and Ctrl-C do, and the caller owns shutdown."""
    cli()


def _console_main() -> None:
    """Run the `pivot` command, which owns its process: serve mode takes over SIGTERM and Ctrl-C
    and keeps them ignored through process shutdown."""
    token = _standalone_invocation.set(True)
    try:
        cli()
    finally:
        _standalone_invocation.reset(token)


if __name__ == "__main__":
    # Run as a script, this file is a second copy of the module. Serve reads the flag of the
    # package, pivot.cli, so the script runs that copy's console entry point.
    import pivot.cli

    pivot.cli._console_main()  # pyright: ignore[reportPrivateUsage] - the same module, imported as the package
