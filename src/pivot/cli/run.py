"""Single-stage pipeline executor.

The `pivot run` command executes specified stages directly, without
resolving dependencies. Use `pivot repro` for DAG-aware execution.
"""

from __future__ import annotations

import contextlib
import datetime
import pathlib
import threading
import time
import uuid
from typing import TYPE_CHECKING

import anyio
import click

from pivot.cli import _run_common, completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine
from pivot.engine import sources as engine_sources
from pivot.executor import prepare_workers
from pivot.storage import project_lock
from pivot.types import (
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


def _configure_oneshot_source(
    eng: engine.Engine,
    stages: list[str],
    *,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> None:
    """Configure OneShotSource for single-stage mode."""
    eng.add_source(
        engine_sources.OneShotSource(
            stages=stages,
            force=force,
            reason="cli",
            single_stage=True,  # Always single-stage for run command
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    )


def _run_with_tui(
    stages_list: list[str],
    force: bool = False,
    tui_log: pathlib.Path | None = None,
    no_commit: bool = False,
    no_cache: bool = False,
    on_error: OnError = OnError.KEEP_GOING,
    allow_uncached_incremental: bool = False,
    checkout_missing: bool = False,
) -> dict[str, ExecutionSummary] | None:
    """Run pipeline with TUI display."""
    from pivot.executor import core as executor_core

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    prepare_workers(len(stages_list))

    # Cancel event allows TUI to signal executor to stop scheduling new stages
    cancel_event = threading.Event()

    # Generate run_id for TUI tracking
    run_id = str(uuid.uuid4())[:8]

    pipeline = cli_decorators.get_pipeline_from_context()

    async def tui_oneshot_main() -> dict[str, executor_core.ExecutionSummary]:
        from pivot.tui import run as tui_run

        # Create TUI app first (needed for TuiSink)
        def executor_wrapper() -> dict[str, ExecutionSummary]:
            return {}

        app = tui_run.PivotApp(
            stage_names=stages_list,
            tui_log=tui_log,
            executor_func=executor_wrapper,
            cancel_event=cancel_event,
        )

        async with engine.Engine(pipeline=pipeline) as eng:
            # Configure sinks - TuiSink for direct post_message
            result_sink = _run_common.configure_result_collector(eng)
            _run_common.configure_output_sink(
                eng,
                quiet=False,
                as_json=False,
                tui=True,
                app=app,
                run_id=run_id,
                use_console=False,
                jsonl_callback=None,
            )
            _configure_oneshot_source(
                eng,
                stages_list,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

            # Run engine and TUI concurrently
            async def engine_task() -> dict[str, executor_core.ExecutionSummary]:
                await eng.run(exit_on_completion=True)
                stage_results = await result_sink.get_results()
                return _run_common.convert_results(stage_results)

            # Suppress stderr logging while TUI is active
            with _run_common.suppress_stderr_logging():
                # Initialize results outside task group to avoid "possibly unbound" error
                engine_results: dict[str, executor_core.ExecutionSummary] = {}

                async def run_engine_and_signal() -> None:
                    nonlocal engine_results
                    engine_results = await engine_task()
                    # TuiSink.close() sends TuiShutdown when Engine closes sinks

                async with anyio.create_task_group() as tg:
                    tg.start_soon(run_engine_and_signal)
                    await anyio.to_thread.run_sync(app.run)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType] - anyio stub issue
                    # TUI exited - if engine is still running, cancel it
                    tg.cancel_scope.cancel()

                return engine_results

    # Acquire lock for oneshot mode when no_commit=True to prevent race conditions
    # with concurrent commit operations
    lock_context = project_lock.pending_state_lock() if no_commit else contextlib.nullcontext()
    with lock_context:
        return anyio.run(tui_oneshot_main)


def _run_json_mode(
    stages_list: list[str],
    *,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    on_error: OnError,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> dict[str, ExecutionSummary]:
    """Run pipeline in JSON streaming mode."""
    from pivot.executor import core as executor_core

    # Emit schema version first
    cli_helpers.emit_jsonl(
        SchemaVersionEvent(type=RunEventType.SCHEMA_VERSION, version=_JSONL_SCHEMA_VERSION)
    )

    start_time = time.perf_counter()
    pipeline = cli_decorators.get_pipeline_from_context()

    async def json_main() -> dict[str, executor_core.ExecutionSummary]:
        async with engine.Engine(pipeline=pipeline) as eng:
            result_sink = _run_common.configure_result_collector(eng)
            _run_common.configure_output_sink(
                eng,
                quiet=False,
                as_json=True,
                tui=False,
                app=None,
                run_id=None,
                use_console=False,
                jsonl_callback=cli_helpers.emit_jsonl,
            )
            _configure_oneshot_source(
                eng,
                stages_list,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
            await eng.run(exit_on_completion=True)
            stage_results = await result_sink.get_results()
            return _run_common.convert_results(stage_results)

    # Acquire lock for oneshot mode when no_commit=True to prevent race conditions
    # with concurrent commit operations
    lock_context = project_lock.pending_state_lock() if no_commit else contextlib.nullcontext()
    with lock_context:
        results = anyio.run(json_main)

    # Emit execution result
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

    return results


def _run_plain_mode(
    stages_list: list[str],
    *,
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
    pipeline = cli_decorators.get_pipeline_from_context()

    async def plain_main() -> dict[str, executor_core.ExecutionSummary]:
        async with engine.Engine(pipeline=pipeline) as eng:
            result_sink = _run_common.configure_result_collector(eng)
            _run_common.configure_output_sink(
                eng,
                quiet=quiet,
                as_json=False,
                tui=False,
                app=None,
                run_id=None,
                use_console=console is not None,
                jsonl_callback=None,
            )
            _configure_oneshot_source(
                eng,
                stages_list,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )
            await eng.run(exit_on_completion=True)
            stage_results = await result_sink.get_results()
            return _run_common.convert_results(stage_results)

    # Acquire lock for oneshot mode when no_commit=True to prevent race conditions
    # with concurrent commit operations. Watch mode handles this differently - the
    # TUI allows manual commit when ready.
    lock_context = project_lock.pending_state_lock() if no_commit else contextlib.nullcontext()
    with lock_context:
        results = anyio.run(plain_main)

    if console and results:
        ran, cached, blocked, failed = executor_core.count_results(results)
        total_duration = time.perf_counter() - start_time
        console.summary(ran, cached, blocked, failed, total_duration)

    return results


def _validate_stages_required(stages_list: list[str] | None) -> list[str]:
    """Validate that at least one stage was provided."""
    if not stages_list:
        raise click.UsageError("Missing argument 'STAGES...'.")
    return stages_list


@cli_decorators.pivot_command()
@click.argument("stages", nargs=-1, shell_complete=completion.complete_stages)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-run of stages, ignoring cache",
)
@click.option(
    "--tui",
    "tui_flag",
    is_flag=True,
    help="Use interactive TUI display (default: plain text)",
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
    help="Stop on first failure (default: keep going)",
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
    force: bool,
    tui_flag: bool,
    as_json: bool,
    tui_log: pathlib.Path | None,
    no_commit: bool,
    no_cache: bool,
    fail_fast: bool,
    allow_uncached_incremental: bool,
    checkout_missing: bool,
) -> None:
    """Execute specified pipeline stages directly.

    Runs STAGES in the order specified, without resolving dependencies.
    At least one stage name must be provided.

    Use 'pivot repro' to run stages with automatic dependency resolution.
    """
    cli_ctx = cli_helpers.get_cli_context(ctx)
    quiet = cli_ctx["quiet"]
    show_human_output = not as_json and not quiet

    # Convert tuple to list and validate at least one stage provided
    stages_list = _validate_stages_required(cli_helpers.stages_to_list(stages))

    # Validate stages exist in registry
    cli_helpers.validate_stages_exist(stages_list)

    # Validate --tui and --json are mutually exclusive
    if tui_flag and as_json:
        raise click.ClickException("--tui and --json are mutually exclusive")

    # Validate tui_log
    tui_log = _run_common.validate_tui_log(tui_log, as_json, tui_flag)

    # Default: keep going; --fail-fast stops on first failure
    on_error = OnError.FAIL if fail_fast else OnError.KEEP_GOING

    if tui_flag:
        results = _run_with_tui(
            stages_list,
            force=force,
            tui_log=tui_log,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
        # TUI returns None if user quit early - don't show "No stages" message
        if results is None:
            return
    elif as_json:
        results = _run_json_mode(
            stages_list,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    else:
        results = _run_plain_mode(
            stages_list,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
            quiet=quiet,
        )

    if not results and show_human_output:
        click.echo("No stages to run")
