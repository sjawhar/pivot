# Split `run` and `repro` Commands Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the current `pivot run` command into two distinct commands: `pivot repro` (DAG-aware execution) and `pivot run` (single-stage execution).

**Architecture:** Extract shared helpers to `_run_common.py`. `repro.py` handles DAG-aware execution (current `run` default behavior). `run.py` handles single-stage execution (current `--single-stage` behavior). Both use the same Engine with different `single_stage` flag.

**Tech Stack:** Click CLI framework, Engine.run_once(), shared TUI/JSON output handling.

---

## Summary of Changes

| Current | New |
|---------|-----|
| `pivot run` | `pivot repro` (DAG-aware, runs dependencies) |
| `pivot run --single-stage` | `pivot run` (runs only named stages) |
| `pivot run --dry-run` | `pivot repro --dry-run` |
| `pivot dry-run` (standalone) | Remove (use `pivot repro -n`) |

### Key Behavioral Differences

| Aspect | `pivot repro` | `pivot run` |
|--------|---------------|-------------|
| No args | Runs entire pipeline | Error: "Missing argument STAGES..." |
| Dependencies | Resolved and run | Ignored |
| Default on error | Fail fast | Keep going |
| Order | Topological (DAG) | User-specified |

---

## Task 1: Create `_run_common.py` with Shared Helpers

**Files:**
- Create: `src/pivot/cli/_run_common.py`
- Test: `tests/cli/test_run_common.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_run_common.py
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pivot.cli import _run_common

if TYPE_CHECKING:
    import click.testing


def test_validate_stages_exist_passes_for_registered_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """validate_stages_exist passes when stages are registered."""
    from helpers import register_test_stage
    from pivot.types import StageParams

    class Params(StageParams):
        pass

    def _helper_noop(params: Params) -> None:
        pass

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_noop, name="my_stage")

        # Should not raise
        _run_common.validate_stages_exist(["my_stage"])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_run_common.py::test_validate_stages_exist_passes_for_registered_stages -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/pivot/cli/_run_common.py
"""Shared helpers for run and repro commands."""
from __future__ import annotations

import contextlib
import logging
import sys
from typing import TYPE_CHECKING

import click

from pivot import registry
from pivot.cli import helpers as cli_helpers

if TYPE_CHECKING:
    from collections.abc import Generator

    import networkx as nx


@contextlib.contextmanager
def suppress_stderr_logging() -> Generator[None]:
    """Suppress logging to stderr while TUI is active.

    Textual takes over the terminal, so stderr writes appear as garbage
    in the upper-left corner. This temporarily removes StreamHandlers
    that write to stderr and restores them on exit.
    """
    root = logging.getLogger()
    removed_handlers = list[logging.Handler]()

    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
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

    Stages at the same level can run in parallel.
    """
    import networkx as nx

    levels: dict[str, int] = {}
    for stage in nx.dfs_postorder_nodes(graph):
        dep_levels = [levels[dep] for dep in graph.successors(stage) if dep in levels]
        levels[stage] = max(dep_levels, default=-1) + 1
    return levels


def sort_for_display(execution_order: list[str], graph: nx.DiGraph[str]) -> list[str]:
    """Sort stages for TUI display: group matrix variants while respecting DAG structure."""
    from pivot.tui.types import parse_stage_name

    levels = compute_dag_levels(graph)

    group_min_level: dict[str, int] = {}
    for name in execution_order:
        base, _ = parse_stage_name(name)
        level = levels.get(name, 0)
        if base not in group_min_level or level < group_min_level[base]:
            group_min_level[base] = level

    def display_sort_key(name: str) -> tuple[int, str, int, str]:
        base, variant = parse_stage_name(name)
        individual_level = levels.get(name, 0)
        return (group_min_level[base], base, individual_level, variant)

    return sorted(execution_order, key=display_sort_key)


def validate_stages_exist(stages_list: list[str] | None) -> None:
    """Validate that all specified stages exist in the registry."""
    cli_helpers.validate_stages_exist(stages_list)


def ensure_stages_registered() -> None:
    """Auto-discover and register stages if none are registered."""
    from pivot import discovery

    logger = logging.getLogger(__name__)

    if not discovery.has_registered_stages():
        try:
            discovered = discovery.discover_and_register()
            if discovered:
                logger.info(f"Loaded pipeline from {discovered}")
        except discovery.DiscoveryError as e:
            raise click.ClickException(str(e)) from e
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_run_common.py::test_validate_stages_exist_passes_for_registered_stages -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "refactor(cli): extract shared helpers to _run_common.py"
```

---

## Task 2: Create `repro.py` Command

**Files:**
- Create: `src/pivot/cli/repro.py`
- Modify: `src/pivot/cli/__init__.py:10-11` (add to COMMAND_CATEGORIES)
- Modify: `src/pivot/cli/__init__.py:28-69` (add to _LAZY_COMMANDS)
- Test: `tests/cli/test_repro.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_repro.py
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Annotated, TypedDict

from helpers import register_test_stage
from pivot import cli, loaders, outputs

if TYPE_CHECKING:
    import click.testing


class _OutputTxtOutputs(TypedDict):
    output: Annotated[pathlib.Path, outputs.Out("output.txt", loaders.PathOnly())]


def _helper_process(
    input_file: Annotated[pathlib.Path, outputs.Dep("input.txt", loaders.PathOnly())],
) -> _OutputTxtOutputs:
    _ = input_file
    pathlib.Path("output.txt").write_text("done")
    return _OutputTxtOutputs(output=pathlib.Path("output.txt"))


def test_repro_runs_all_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot repro runs all stages by default."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert pathlib.Path("output.txt").exists()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_repro.py::test_repro_runs_all_stages -v`
Expected: FAIL with "No such command 'repro'"

**Step 3: Write the implementation**

Create `src/pivot/cli/repro.py`:

```python
# src/pivot/cli/repro.py
"""pivot repro - DAG-aware pipeline execution."""
from __future__ import annotations

import datetime
import pathlib
import time
from typing import TYPE_CHECKING

import click

from pivot import config, dag, registry
from pivot.cli import _run_common, completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.executor import prepare_workers
from pivot.types import (
    DisplayMode,
    ExecutionResultEvent,
    OnError,
    RunEventType,
    SchemaVersionEvent,
    StageStatus,
)

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary

# JSONL schema version for forward compatibility
_JSONL_SCHEMA_VERSION = 1


def _run_with_tui(
    stages_list: list[str] | None,
    cache_dir: pathlib.Path | None,
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    on_error: OnError = OnError.FAIL,
    allow_uncached_incremental: bool = False,
    checkout_missing: bool = False,
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with TUI display."""
    import queue as thread_queue
    import threading
    import uuid

    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=False)

    if not execution_order:
        return {}

    resolved_cache_dir = cache_dir or config.get_cache_dir()
    prepare_workers(len(execution_order))

    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()
    cancel_event = threading.Event()
    run_id = str(uuid.uuid4())[:8]

    def executor_func() -> dict[str, ExecutionSummary]:
        with engine.Engine() as eng:
            eng.set_cancel_event(cancel_event)
            eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
            return eng.run_once(
                stages=stages_list,
                single_stage=False,
                cache_dir=resolved_cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

    display_order = _run_common.sort_for_display(execution_order, graph)

    with _run_common.suppress_stderr_logging():
        return run_tui.run_with_tui(
            display_order, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


def _run_watch_with_tui(
    stages_list: list[str] | None,
    cache_dir: pathlib.Path | None,  # noqa: ARG001
    debounce: int,  # noqa: ARG001
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,  # noqa: ARG001
    no_cache: bool = False,  # noqa: ARG001
    on_error: OnError = OnError.FAIL,
    serve: bool = False,
) -> None:
    """Run watch mode with TUI display."""
    _ = cache_dir, debounce, no_commit, no_cache

    import queue as thread_queue
    import uuid

    from pivot.engine import graph as engine_graph
    from pivot.engine import sources
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    graph = registry.REGISTRY.build_dag(validate=True)
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=False)

    prepare_workers(len(execution_order) if execution_order else 1)

    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()
    run_id = str(uuid.uuid4())[:8]

    with engine.Engine() as eng:
        eng.set_keep_going(on_error == OnError.KEEP_GOING)

        all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
        bipartite_graph = engine_graph.build_graph(all_stages)
        watch_paths = engine_graph.get_watch_paths(bipartite_graph)

        filesystem_source = sources.FilesystemSource(watch_paths)
        eng.add_source(filesystem_source)

        if force:
            initial_source = sources.OneShotSource(
                stages=stages_list,
                force=True,
                reason="watch:initial:forced",
            )
            eng.add_source(initial_source)

        eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
        eng.add_sink(sinks.WatchSink(tui_queue=tui_queue))

        display_order = _run_common.sort_for_display(execution_order, graph) if execution_order else None

        with _run_common.suppress_stderr_logging():
            run_tui.run_watch_tui(
                eng,
                tui_queue,
                tui_log=tui_log,
                stage_names=display_order,
                no_commit=False,
                serve=serve,
            )


def _output_explain(
    stages_list: list[str] | None,
    force: bool = False,
    allow_missing: bool = False,
) -> None:
    """Output detailed stage explanations."""
    from pivot import status as status_mod
    from pivot.cli import status as status_cli
    from pivot.engine import graph as engine_graph

    if not allow_missing:
        registry.REGISTRY.build_dag(validate=True)

    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list,
        single_stage=False,
        force=force,
        allow_missing=allow_missing,
        graph=graph,
    )
    status_cli.output_explain_text(explanations)


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would run without executing")
@click.option(
    "--explain", "-e", is_flag=True, help="Show detailed breakdown of why stages would run"
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-run of stages, ignoring cache (in --watch mode, first run only)",
)
@click.option(
    "--watch",
    "-w",
    is_flag=True,
    help="Watch for file changes and re-run affected stages",
)
@click.option(
    "--debounce",
    type=click.IntRange(min=0),
    default=None,
    help="Debounce delay in milliseconds (for --watch mode)",
)
@click.option(
    "--display",
    type=click.Choice([e.value for e in DisplayMode]),
    default=None,
    help="Display mode: tui (interactive) or plain (streaming text). Auto-detects if not specified.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--tui-log",
    type=click.Path(path_type=pathlib.Path),
    help="Write TUI messages to JSONL file for monitoring",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Defer lock files to pending dir for faster iteration. Run 'pivot commit' to finalize.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip caching outputs entirely for maximum iteration speed. Outputs won't be cached.",
)
@click.option(
    "--keep-going",
    "-k",
    is_flag=True,
    help="Continue running stages after failures; skip only downstream dependents.",
)
@click.option(
    "--serve",
    is_flag=True,
    help="Start RPC server for agent control (requires --watch). Creates Unix socket at .pivot/agent.sock",
)
@click.option(
    "--allow-uncached-incremental",
    is_flag=True,
    help="Allow running stages with IncrementalOut files that exist but aren't in cache.",
)
@click.option(
    "--checkout-missing",
    is_flag=True,
    help="Restore tracked files that don't exist on disk from cache before running.",
)
@click.option(
    "--allow-missing",
    is_flag=True,
    help="Allow missing dep files if tracked (.pvt exists). Only affects --dry-run.",
)
@click.pass_context
def repro(
    ctx: click.Context,
    stages: tuple[str, ...],
    cache_dir: pathlib.Path | None,
    dry_run: bool,
    explain: bool,
    force: bool,
    watch: bool,
    debounce: int | None,
    display: str | None,
    as_json: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    keep_going: bool,
    serve: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
    allow_missing: bool,
) -> None:
    """Reproduce pipeline stages with dependencies.

    If STAGES are provided, runs those stages and everything they depend on.
    If no STAGES are provided, runs the entire pipeline.

    Auto-discovers pivot.yaml or pipeline.py if no stages are registered.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet
    debounce = debounce if debounce is not None else config.get_watch_debounce()

    stages_list = cli_helpers.stages_to_list(stages)
    _run_common.validate_stages_exist(stages_list)

    # Validate tui_log requires TUI mode
    if tui_log:
        if as_json:
            raise click.ClickException("--tui-log cannot be used with --json")
        if display == DisplayMode.PLAIN.value:
            raise click.ClickException("--tui-log cannot be used with --display=plain")
        if dry_run:
            raise click.ClickException("--tui-log cannot be used with --dry-run")
        tui_log = tui_log.expanduser().resolve()
        try:
            tui_log.parent.mkdir(parents=True, exist_ok=True)
            tui_log.touch()
        except OSError as e:
            raise click.ClickException(f"Cannot write to {tui_log}: {e}") from e

    if serve and not watch:
        raise click.ClickException("--serve requires --watch mode")

    if allow_missing and not dry_run:
        raise click.ClickException("--allow-missing can only be used with --dry-run")

    if explain:
        _output_explain(stages_list, force, allow_missing=allow_missing)
        return

    if dry_run:
        _dry_run(stages_list, force, as_json, allow_missing)
        return

    on_error = OnError.KEEP_GOING if keep_going else OnError.FAIL

    if watch:
        from pivot.tui import run as run_tui

        display_mode = DisplayMode(display) if display else None
        use_tui = run_tui.should_use_tui(display_mode) and not as_json

        if serve and not use_tui:
            raise click.ClickException(
                "--serve requires TUI mode (not compatible with --json or --display=plain)"
            )

        if use_tui:
            try:
                _run_watch_with_tui(
                    stages_list,
                    cache_dir,
                    debounce,
                    force,
                    tui_log=tui_log,
                    no_commit=no_commit,
                    no_cache=no_cache,
                    on_error=on_error,
                    serve=serve,
                )
            except KeyboardInterrupt:
                if show_human_output:
                    click.echo("\nWatch mode stopped.")
        else:
            from pivot.engine import graph as engine_graph
            from pivot.engine import sources

            with engine.Engine() as eng:
                eng.set_keep_going(on_error == OnError.KEEP_GOING)

                all_stages = {
                    name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()
                }
                bipartite_graph = engine_graph.build_graph(all_stages)
                watch_paths = engine_graph.get_watch_paths(bipartite_graph)

                filesystem_source = sources.FilesystemSource(watch_paths)
                eng.add_source(filesystem_source)

                if force:
                    initial_source = sources.OneShotSource(
                        stages=stages_list,
                        force=True,
                        reason="watch:initial:forced",
                    )
                    eng.add_source(initial_source)

                if not as_json:
                    from pivot.tui import console as tui_console

                    console = tui_console.Console()
                    eng.add_sink(sinks.ConsoleSink(console))

                try:
                    eng.run_loop()
                except KeyboardInterrupt:
                    pass
                finally:
                    eng.shutdown()
                    if show_human_output:
                        click.echo("\nWatch mode stopped.")
        return

    display_mode = DisplayMode(display) if display else None

    from pivot.tui import run as run_tui

    use_tui = run_tui.should_use_tui(display_mode) and not as_json
    if use_tui:
        results = _run_with_tui(
            stages_list,
            cache_dir,
            force=force,
            tui_log=tui_log,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    elif as_json:
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

        start_time = time.perf_counter()
        with engine.Engine() as eng:
            eng.add_sink(sinks.JsonlSink(callback=cli_helpers.emit_jsonl))
            results = eng.run_once(
                stages=stages_list,
                single_stage=False,
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
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
    else:
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
                single_stage=False,
                cache_dir=cache_dir,
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

    if not results and show_human_output and not use_tui:
        click.echo("No stages to run")


def _dry_run(
    stages_list: list[str] | None,
    force: bool,
    as_json: bool,
    allow_missing: bool,
) -> None:
    """Show what would run without executing."""
    import json

    from pivot import status as status_mod
    from pivot.engine import graph as engine_graph

    if not allow_missing:
        registry.REGISTRY.build_dag(validate=True)

    all_stages = {name: registry.REGISTRY.get(name) for name in registry.REGISTRY.list_stages()}
    graph = engine_graph.build_graph(all_stages)

    explanations = status_mod.get_pipeline_explanations(
        stages_list, single_stage=False, force=force, allow_missing=allow_missing, graph=graph
    )

    if not explanations:
        if as_json:
            click.echo(json.dumps({"stages": {}}))
        else:
            click.echo("No stages to run")
        return

    if as_json:
        output = {
            "stages": {
                exp["stage_name"]: {
                    "would_run": exp["will_run"],
                    "reason": exp["reason"] or "unchanged",
                }
                for exp in explanations
            }
        }
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo("Would run:")
        for exp in explanations:
            status = "would run" if exp["will_run"] else "would skip"
            reason = exp["reason"] or "unchanged"
            click.echo(f"  {exp['stage_name']}: {status} ({reason})")
```

**Step 4: Register in `__init__.py`**

Edit `src/pivot/cli/__init__.py`:

```python
# In COMMAND_CATEGORIES, change "run" to "repro" and add "run":
COMMAND_CATEGORIES = {
    "Pipeline": ["repro", "run", "status", "verify", "commit"],
    ...
}

# In _LAZY_COMMANDS, add repro and update run:
_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    ...
    "repro": ("pivot.cli.repro", "repro", "Reproduce pipeline stages with dependencies."),
    "run": ("pivot.cli.run", "run", "Execute specific stages directly."),
    ...
}
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_repro.py::test_repro_runs_all_stages -v`
Expected: PASS

**Step 6: Commit**

```bash
jj describe -m "feat(cli): add pivot repro command for DAG-aware execution"
```

---

## Task 3: Simplify `run.py` for Single-Stage Execution

**Files:**
- Modify: `src/pivot/cli/run.py`
- Test: `tests/cli/test_run.py` (update existing tests)

**Step 1: Write the failing test**

```python
# Add to tests/cli/test_run.py

def test_run_requires_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run without stages shows error."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["run"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "STAGES" in result.output


def test_run_executes_only_named_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run executes only the named stages, not dependencies."""
    from helpers import register_test_stage
    from pivot.types import StageParams

    class Params(StageParams):
        pass

    # Track which stages ran
    ran_stages: list[str] = []

    def _helper_stage_a(params: Params) -> None:
        pathlib.Path("a.txt").write_text("a")

    def _helper_stage_b(
        dep: Annotated[pathlib.Path, outputs.Dep("a.txt", loaders.PathOnly())],
    ) -> None:
        _ = dep
        pathlib.Path("b.txt").write_text("b")

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        # Create a.txt so stage_b doesn't fail due to missing dep
        pathlib.Path("a.txt").write_text("pre-existing")

        register_test_stage(_helper_stage_a, name="stage_a")
        register_test_stage(_helper_stage_b, name="stage_b")

        # Run only stage_b - should NOT run stage_a
        result = runner.invoke(cli.cli, ["run", "stage_b"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        # a.txt should still have pre-existing content (stage_a didn't run)
        assert pathlib.Path("a.txt").read_text() == "pre-existing"
        # b.txt should exist (stage_b ran)
        assert pathlib.Path("b.txt").exists()
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/cli/test_run.py::test_run_requires_stages -v`
Expected: FAIL (current `run` accepts no args and runs all)

**Step 3: Rewrite `run.py` for single-stage mode**

Replace `src/pivot/cli/run.py` with a simplified version:

```python
# src/pivot/cli/run.py
"""pivot run - Direct stage execution (no dependency resolution)."""
from __future__ import annotations

import datetime
import pathlib
import time
from typing import TYPE_CHECKING

import click

from pivot import config
from pivot.cli import _run_common, completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.executor import prepare_workers
from pivot.types import (
    DisplayMode,
    ExecutionResultEvent,
    OnError,
    RunEventType,
    SchemaVersionEvent,
    StageStatus,
)

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary

_JSONL_SCHEMA_VERSION = 1


def _run_with_tui(
    stages_list: list[str],
    cache_dir: pathlib.Path | None,
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    on_error: OnError = OnError.KEEP_GOING,
    allow_uncached_incremental: bool = False,
    checkout_missing: bool = False,
) -> dict[str, ExecutionSummary] | None:
    """Run stages with TUI display."""
    import queue as thread_queue
    import threading
    import uuid

    from pivot import dag, registry
    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    graph = registry.REGISTRY.build_dag(validate=True)
    # single_stage=True means run only the named stages
    execution_order = dag.get_execution_order(graph, stages_list, single_stage=True)

    if not execution_order:
        return {}

    resolved_cache_dir = cache_dir or config.get_cache_dir()
    prepare_workers(len(execution_order))

    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()
    cancel_event = threading.Event()
    run_id = str(uuid.uuid4())[:8]

    def executor_func() -> dict[str, ExecutionSummary]:
        with engine.Engine() as eng:
            eng.set_cancel_event(cancel_event)
            eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
            return eng.run_once(
                stages=stages_list,
                single_stage=True,
                cache_dir=resolved_cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

    display_order = _run_common.sort_for_display(execution_order, graph)

    with _run_common.suppress_stderr_logging():
        return run_tui.run_with_tui(
            display_order, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, required=True, shell_complete=completion.complete_stages)
@click.option("--cache-dir", type=click.Path(path_type=pathlib.Path), help="Cache directory")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-run of stages, ignoring cache",
)
@click.option(
    "--display",
    type=click.Choice([e.value for e in DisplayMode]),
    default=None,
    help="Display mode: tui (interactive) or plain (streaming text). Auto-detects if not specified.",
)
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
@click.option(
    "--tui-log",
    type=click.Path(path_type=pathlib.Path),
    help="Write TUI messages to JSONL file for monitoring",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Defer lock files to pending dir for faster iteration. Run 'pivot commit' to finalize.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip caching outputs entirely for maximum iteration speed. Outputs won't be cached.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure (default: keep going).",
)
@click.option(
    "--allow-uncached-incremental",
    is_flag=True,
    help="Allow running stages with IncrementalOut files that exist but aren't in cache.",
)
@click.option(
    "--checkout-missing",
    is_flag=True,
    help="Restore tracked files that don't exist on disk from cache before running.",
)
@click.pass_context
def run(
    ctx: click.Context,
    stages: tuple[str, ...],
    cache_dir: pathlib.Path | None,
    force: bool,
    display: str | None,
    as_json: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    fail_fast: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> None:
    """Execute specific stages directly (without dependency resolution).

    STAGES are required. Runs only the named stages in the order specified.
    Dependencies are NOT automatically run - use 'pivot repro' for DAG-aware execution.

    Unlike 'pivot repro', this command continues after failures by default.
    Use --fail-fast to stop on the first failure.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet

    stages_list = list(stages)
    _run_common.validate_stages_exist(stages_list)

    if tui_log:
        if as_json:
            raise click.ClickException("--tui-log cannot be used with --json")
        if display == DisplayMode.PLAIN.value:
            raise click.ClickException("--tui-log cannot be used with --display=plain")
        tui_log = tui_log.expanduser().resolve()
        try:
            tui_log.parent.mkdir(parents=True, exist_ok=True)
            tui_log.touch()
        except OSError as e:
            raise click.ClickException(f"Cannot write to {tui_log}: {e}") from e

    # Default is keep-going for run (opposite of repro)
    on_error = OnError.FAIL if fail_fast else OnError.KEEP_GOING

    display_mode = DisplayMode(display) if display else None

    from pivot.tui import run as run_tui

    use_tui = run_tui.should_use_tui(display_mode) and not as_json
    if use_tui:
        results = _run_with_tui(
            stages_list,
            cache_dir,
            force=force,
            tui_log=tui_log,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    elif as_json:
        cli_helpers.emit_jsonl(
            SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
        )

        start_time = time.perf_counter()
        with engine.Engine() as eng:
            eng.add_sink(sinks.JsonlSink(callback=cli_helpers.emit_jsonl))
            results = eng.run_once(
                stages=stages_list,
                single_stage=True,
                cache_dir=cache_dir,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
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
    else:
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
                single_stage=True,
                cache_dir=cache_dir,
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

    if not results and show_human_output and not use_tui:
        click.echo("No stages to run")
```

**Step 4: Run tests to verify**

Run: `uv run pytest tests/cli/test_run.py -v`
Expected: Some tests may need updating (tests that assumed DAG behavior)

**Step 5: Update existing tests**

Tests that use `pivot run` without `--single-stage` and expect DAG behavior need to change to `pivot repro`.

**Step 6: Commit**

```bash
jj describe -m "refactor(cli): simplify run.py for single-stage execution"
```

---

## Task 4: Remove Standalone `dry-run` Command

**Files:**
- Modify: `src/pivot/cli/__init__.py` (remove from _LAZY_COMMANDS)
- Delete: The `dry_run_cmd` function is in `run.py`, but we've already removed it in Task 3
- Test: Verify `pivot dry-run` no longer works

**Step 1: Write the failing test**

```python
# Add to tests/cli/test_repro.py

def test_dry_run_standalone_removed(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot dry-run standalone command no longer exists."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()

        result = runner.invoke(cli.cli, ["dry-run"])

        # Should fail with "No such command"
        assert result.exit_code != 0
        assert "No such command" in result.output
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/cli/test_repro.py::test_dry_run_standalone_removed -v`
Expected: FAIL (dry-run still exists)

**Step 3: Remove from CLI registration**

The `dry-run` command was defined in `run.py` with `@cli_decorators.pivot_command("dry-run")`. Since we rewrote `run.py` in Task 3, it's already gone.

Just verify it's not in `_LAZY_COMMANDS` in `__init__.py`.

**Step 4: Run test to verify**

Run: `uv run pytest tests/cli/test_repro.py::test_dry_run_standalone_removed -v`
Expected: PASS

**Step 5: Commit**

```bash
jj describe -m "refactor(cli): remove standalone dry-run command (use repro -n)"
```

---

## Task 5: Update CLI Tests

**Files:**
- Modify: `tests/cli/test_run.py`
- Create: `tests/cli/test_repro.py` (additional tests)

**Step 1: Add comprehensive tests for repro**

```python
# Add to tests/cli/test_repro.py

def test_repro_dry_run_shows_what_would_run(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot repro --dry-run shows what would run."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path("input.txt").write_text("data")
        register_test_stage(_helper_process, name="process")

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        assert result.exit_code == 0
        assert "Would run" in result.output or "would run" in result.output


def test_repro_keep_going_continues_after_failure(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot repro --keep-going continues after stage failure."""
    from pivot.types import StageParams

    class Params(StageParams):
        pass

    def _helper_failing(params: Params) -> None:
        raise RuntimeError("Intentional failure")

    def _helper_succeeding(params: Params) -> None:
        pathlib.Path("success.txt").write_text("ok")

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_failing, name="failing")
        register_test_stage(_helper_succeeding, name="succeeding")

        result = runner.invoke(cli.cli, ["repro", "--keep-going", "--display=plain"])

        # Should have run both (or at least tried)
        # The succeeding stage should have created its output
        assert pathlib.Path("success.txt").exists()
```

**Step 2: Add tests for run command**

```python
# Add to tests/cli/test_run.py

def test_run_keeps_going_by_default(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run continues after failure by default."""
    from pivot.types import StageParams

    class Params(StageParams):
        pass

    def _helper_failing(params: Params) -> None:
        raise RuntimeError("Intentional failure")

    def _helper_succeeding(params: Params) -> None:
        pathlib.Path("success.txt").write_text("ok")

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_failing, name="failing")
        register_test_stage(_helper_succeeding, name="succeeding")

        result = runner.invoke(cli.cli, ["run", "failing", "succeeding", "--display=plain"])

        # Should have tried both stages
        # succeeding should have created its output despite failing's failure
        assert pathlib.Path("success.txt").exists()


def test_run_fail_fast_stops_on_failure(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """pivot run --fail-fast stops on first failure."""
    from pivot.types import StageParams

    class Params(StageParams):
        pass

    def _helper_failing(params: Params) -> None:
        raise RuntimeError("Intentional failure")

    def _helper_succeeding(params: Params) -> None:
        pathlib.Path("success.txt").write_text("ok")

    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        register_test_stage(_helper_failing, name="failing")
        register_test_stage(_helper_succeeding, name="succeeding")

        result = runner.invoke(cli.cli, ["run", "failing", "succeeding", "--fail-fast", "--display=plain"])

        # Should have stopped after failing stage
        # succeeding should NOT have run
        assert not pathlib.Path("success.txt").exists()
```

**Step 3: Run all CLI tests**

Run: `uv run pytest tests/cli/test_run.py tests/cli/test_repro.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
jj describe -m "test(cli): add comprehensive tests for run and repro commands"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/cli/index.md`

**Step 1: Update documentation**

Replace the `pivot run` section with:

```markdown
## Pipeline Execution

### `pivot repro`

Reproduce pipeline stages with dependency resolution (DAG-aware execution).

```bash
pivot repro [STAGES...] [OPTIONS]
```

**Arguments:**

- `STAGES` - Stage names to run (optional, runs entire pipeline if not specified)

**Options:**

| Option | Description |
|--------|-------------|
| `--cache-dir PATH` | Custom cache directory |
| `--dry-run` / `-n` | Show what would run without executing |
| `--explain` / `-e` | Show detailed breakdown of why stages run |
| `--force` / `-f` | Force re-run of stages, ignoring cache |
| `--watch` / `-w` | Watch for file changes and re-run affected stages |
| `--debounce MS` | Debounce delay in milliseconds (default: 300) |
| `--display [tui\|plain]` | Display mode |
| `--json` | Output results as JSON |
| `--tui-log PATH` | Write TUI messages to JSONL file |
| `--no-commit` | Defer lock files to pending dir |
| `--no-cache` | Skip caching outputs entirely |
| `--keep-going` / `-k` | Continue running stages after failures |
| `--serve` | Start RPC server for agent control (requires --watch) |
| `--allow-uncached-incremental` | Allow IncrementalOut files not in cache |
| `--checkout-missing` | Restore tracked files from cache first |

**Examples:**

```bash
# Run entire pipeline
pivot repro

# Run specific stages and their dependencies
pivot repro train evaluate

# Dry run to see what would execute
pivot repro --dry-run

# Watch mode
pivot repro --watch
```

---

### `pivot run`

Execute specific stages directly without dependency resolution.

```bash
pivot run STAGES... [OPTIONS]
```

**Arguments:**

- `STAGES` - Stage names to run (required)

**Options:**

| Option | Description |
|--------|-------------|
| `--cache-dir PATH` | Custom cache directory |
| `--force` / `-f` | Force re-run of stages, ignoring cache |
| `--display [tui\|plain]` | Display mode |
| `--json` | Output results as JSON |
| `--tui-log PATH` | Write TUI messages to JSONL file |
| `--no-commit` | Defer lock files to pending dir |
| `--no-cache` | Skip caching outputs entirely |
| `--fail-fast` | Stop on first failure (default: keep going) |
| `--allow-uncached-incremental` | Allow IncrementalOut files not in cache |
| `--checkout-missing` | Restore tracked files from cache first |

**Examples:**

```bash
# Run single stage (no deps)
pivot run train

# Run multiple stages in order
pivot run preprocess train

# Force re-run
pivot run train --force
```

**When to use `run` vs `repro`:**

| Use `repro` when... | Use `run` when... |
|---------------------|-------------------|
| Running the pipeline normally | Debugging a specific stage |
| You want deps to run first | You know deps are already up-to-date |
| Starting fresh | Re-running a failed stage |
| Watching for changes | Quick iteration on one stage |
```

Also update the Quick Reference table:

```markdown
| Task | Command |
|------|---------|
| Run pipeline | `pivot repro` |
| Run specific stages + deps | `pivot repro stage1 stage2` |
| Run single stage (no deps) | `pivot run stage` |
| See what would run | `pivot repro -n` |
| Understand why stage runs | `pivot status --explain stage` |
| List all stages | `pivot list` |
| Show stage status | `pivot status` |
| Push outputs to remote | `pivot push` |
| Pull outputs from remote | `pivot pull` |
| Watch for changes | `pivot repro --watch` |
```

**Step 2: Commit**

```bash
jj describe -m "docs(cli): update documentation for run/repro split"
```

---

## Task 7: Update Existing Test Imports

**Files:**
- Modify: Various test files that import from `pivot.cli.run`

**Step 1: Search for affected tests**

```bash
rg "from pivot.cli.run import" tests/
rg "from pivot.cli import run" tests/
```

**Step 2: Update imports**

If any tests import helpers like `ensure_stages_registered` from `run.py`, update them to import from `_run_common.py`.

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -n auto`
Expected: All PASS

**Step 4: Commit**

```bash
jj describe -m "test: update imports after run/repro split"
```

---

## Task 8: Final Verification

**Step 1: Run quality checks**

```bash
uv run ruff format . && uv run ruff check . && uv run basedpyright .
```

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -n auto
```

**Step 3: Manual smoke test**

```bash
# Create a test pipeline and verify both commands work
cd /tmp && mkdir -p test-split && cd test-split
pivot init

# Create simple stages
cat > pivot.yaml << 'EOF'
stages:
  prepare:
    function: pipeline:prepare
  train:
    function: pipeline:train
EOF

cat > pipeline.py << 'EOF'
from pathlib import Path

def prepare():
    Path("data.txt").write_text("data")

def train():
    Path("model.txt").write_text("trained")
EOF

# Test repro (DAG-aware)
pivot repro --dry-run

# Test run (single-stage)
pivot run prepare
pivot run train
```

**Step 4: Push**

```bash
jj git push --named=split-run-repro=@
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Extract shared helpers to `_run_common.py` |
| 2 | Create `repro.py` command |
| 3 | Simplify `run.py` for single-stage execution |
| 4 | Remove standalone `dry-run` command |
| 5 | Update CLI tests |
| 6 | Update documentation |
| 7 | Update existing test imports |
| 8 | Final verification |
