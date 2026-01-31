"""Shared helpers for run and repro CLI commands."""

from __future__ import annotations

import contextlib
import datetime
import logging
import sys
import time
from typing import TYPE_CHECKING, TypedDict

import click

from pivot import discovery
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.types import (
    ExecutionResultEvent,
    OnError,
    RunEventType,
    SchemaVersionEvent,
    StageStatus,
)

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

    import networkx as nx

    from pivot.executor import ExecutionSummary


# JSONL schema version for forward compatibility
_JSONL_SCHEMA_VERSION = 1

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
            # StreamHandler is generic but handlers list is Handler[]
            stream = getattr(handler, "stream", None)  # pyright: ignore[reportUnknownArgumentType]
            if stream in (sys.stderr, sys.stdout):
                root.removeHandler(handler)  # pyright: ignore[reportUnknownArgumentType]
                removed_handlers.append(handler)  # pyright: ignore[reportUnknownArgumentType]
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

    levels: dict[str, int] = {}
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
    """Auto-discover and register stages if none are registered."""
    if not discovery.has_registered_stages():
        try:
            discovered = discovery.discover_and_register()
            if discovered:
                logger.info(f"Loaded pipeline from {discovered}")
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


def emit_execution_result(
    results: dict[str, ExecutionSummary],
    start_time: float,
) -> None:
    """Emit JSONL execution result event with summary counts."""
    total_duration_ms = (time.perf_counter() - start_time) * 1000
    ran = sum(1 for r in results.values() if r["status"] == StageStatus.RAN)
    skipped = sum(1 for r in results.values() if r["status"] == StageStatus.SKIPPED)
    failed = sum(1 for r in results.values() if r["status"] == StageStatus.FAILED)

    cli_helpers.emit_jsonl(
        ExecutionResultEvent(
            type=RunEventType.EXECUTION_RESULT,
            ran=ran,
            skipped=skipped,
            failed=failed,
            total_duration_ms=total_duration_ms,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
        )
    )


def emit_json_schema_version() -> None:
    """Emit the JSONL schema version event."""
    cli_helpers.emit_jsonl(
        SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
    )


def run_json_mode(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> dict[str, ExecutionSummary]:
    """Run pipeline in JSON streaming mode."""
    emit_json_schema_version()

    start_time = time.perf_counter()
    with engine.Engine() as eng:
        eng.add_sink(sinks.JsonlSink(callback=cli_helpers.emit_jsonl))
        results = eng.run_once(
            stages=stages_list,
            single_stage=single_stage,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )

    emit_execution_result(results, start_time)
    return results


def run_plain_mode(
    stages_list: list[str] | None,
    single_stage: bool,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
    quiet: bool,
) -> dict[str, ExecutionSummary]:
    """Run pipeline in plain (non-TUI) mode with optional console output."""
    from pivot.executor import core as executor_core
    from pivot.tui import console as tui_console

    console: tui_console.Console | None = None
    if not quiet:
        console = tui_console.Console()

    start_time = time.perf_counter()
    with engine.Engine() as eng:
        if console:
            eng.add_sink(sinks.ConsoleSink(console))
        results = eng.run_once(
            stages=stages_list,
            single_stage=single_stage,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )

    if console and results:
        ran, cached, blocked, failed = executor_core.count_results(results)
        total_duration = time.perf_counter() - start_time
        console.summary(ran, cached, blocked, failed, total_duration)

    return results


class DryRunJsonStageOutput(TypedDict):
    """JSON output for a single stage in dry-run mode."""

    would_run: bool
    reason: str


class DryRunJsonOutput(TypedDict):
    """JSON output for pivot run --dry-run --json."""

    stages: dict[str, DryRunJsonStageOutput]
