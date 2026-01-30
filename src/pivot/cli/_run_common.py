"""Shared helpers for run and repro CLI commands."""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import TYPE_CHECKING, TextIO, TypedDict, cast

import click

from pivot import discovery
from pivot.cli import decorators as cli_decorators

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

    import networkx as nx

    from pivot.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stderr_logging() -> Generator[None]:
    """Suppress logging to stdout/stderr while TUI is active.

    Textual takes over the terminal, so stdout/stderr writes appear as garbage
    in the upper-left corner. This temporarily removes StreamHandlers
    that write to stdout or stderr and restores them on exit.
    """
    root = logging.getLogger()
    removed_handlers = list[logging.Handler]()

    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            # Cast to known type - stdlib handlers writing to stderr/stdout use TextIO
            # Use string literal because generic type isn't narrowed by isinstance
            stream_handler = cast("logging.StreamHandler[TextIO]", handler)
            if stream_handler.stream in (sys.stderr, sys.stdout):
                root.removeHandler(stream_handler)
                removed_handlers.append(stream_handler)
    try:
        yield
    finally:
        for handler in removed_handlers:
            root.addHandler(handler)


def compute_dag_levels(graph: nx.DiGraph[str]) -> dict[str, int]:
    """Compute DAG level for each stage.

    Level 0: stages with no dependencies
    Level N: stages whose dependencies are all at level < N

    Stages at the same level can run in parallel - there's no ordering between them.
    """
    import networkx as nx

    levels = dict[str, int]()
    # Process in topological order (dependencies before dependents)
    for stage in nx.dfs_postorder_nodes(graph):
        # successors = what this stage depends on (edges go consumer -> producer)
        dep_levels = [levels[dep] for dep in graph.successors(stage) if dep in levels]
        levels[stage] = max(dep_levels, default=-1) + 1
    return levels


def sort_for_display(execution_order: list[str], graph: nx.DiGraph[str]) -> list[str]:
    """Sort stages for TUI display: group matrix variants while respecting DAG structure.

    Uses DAG levels (not arbitrary execution order) so parallel-capable stages
    are treated as equals. Matrix variants are grouped at the level of their
    earliest member.
    """
    from pivot.tui.types import parse_stage_name

    levels = compute_dag_levels(graph)

    # Compute minimum level for each base_name (group position)
    group_min_level: dict[str, int] = {}
    for name in execution_order:
        base, _ = parse_stage_name(name)
        level = levels.get(name, 0)
        if base not in group_min_level or level < group_min_level[base]:
            group_min_level[base] = level

    def display_sort_key(name: str) -> tuple[int, str, int, str]:
        base, variant = parse_stage_name(name)
        individual_level = levels.get(name, 0)
        # Sort by: group level, then base_name (to keep groups together),
        # then individual level, then variant name
        return (group_min_level[base], base, individual_level, variant)

    return sorted(execution_order, key=display_sort_key)


def ensure_stages_registered() -> None:
    """Ensure a Pipeline is discovered and in context.

    If no Pipeline is in context, attempts discovery and stores the result.
    """
    if cli_decorators.get_pipeline_from_context() is not None:
        return
    try:
        pipeline: Pipeline | None = discovery.discover_pipeline()
        if pipeline is not None:
            cli_decorators.store_pipeline_in_context(pipeline)
            logger.info(f"Loaded pipeline: {pipeline.name}")
    except discovery.DiscoveryError as e:
        raise click.ClickException(str(e)) from e


def validate_tui_log(
    tui_log: pathlib.Path | None,
    as_json: bool,
    tui_flag: bool,
    dry_run: bool = False,
) -> pathlib.Path | None:
    """Validate --tui-log option and resolve path if valid."""
    if not tui_log:
        return None
    if as_json:
        raise click.ClickException("--tui-log cannot be used with --json")
    if not tui_flag:
        raise click.ClickException("--tui-log requires --tui")
    if dry_run:
        raise click.ClickException("--tui-log cannot be used with --dry-run")
    # Validate path upfront (fail fast)
    resolved = tui_log.expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.touch()  # Verify writable
    except OSError as e:
        raise click.ClickException(f"Cannot write to {resolved}: {e}") from e
    return resolved


class DryRunJsonStageOutput(TypedDict):
    """JSON output for a single stage in dry-run mode."""

    would_run: bool
    reason: str


class DryRunJsonOutput(TypedDict):
    """JSON output for pivot run --dry-run --json."""

    stages: dict[str, DryRunJsonStageOutput]
