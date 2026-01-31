"""Single-stage pipeline executor.

The `pivot run` command executes specified stages directly, without
resolving dependencies. Use `pivot repro` for DAG-aware execution.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import click

from pivot.cli import _run_common, completion
from pivot.cli import decorators as cli_decorators
from pivot.cli import helpers as cli_helpers
from pivot.engine import engine, sinks
from pivot.executor import prepare_workers
from pivot.types import OnError

if TYPE_CHECKING:
    from pivot.executor import ExecutionSummary


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
    import queue as thread_queue
    import threading
    import uuid

    from pivot.tui import run as run_tui
    from pivot.types import TuiMessage

    # Pre-warm loky executor before starting Textual TUI.
    # Textual manipulates terminal file descriptors which breaks loky's
    # resource tracker if spawned after Textual starts.
    prepare_workers(len(stages_list))

    # tui_queue is inter-thread only (executor -> TUI reader), no cross-process IPC needed.
    # Using stdlib queue.Queue avoids Manager subprocess dependency issues.
    tui_queue: thread_queue.Queue[TuiMessage] = thread_queue.Queue()

    # Cancel event allows TUI to signal executor to stop scheduling new stages
    cancel_event = threading.Event()

    # Generate run_id for TUI tracking
    run_id = str(uuid.uuid4())[:8]

    # Create executor function using Engine with TuiSink
    def executor_func() -> dict[str, ExecutionSummary]:
        with engine.Engine() as eng:
            # Share TUI's cancel_event so TUI can signal cancellation
            eng.set_cancel_event(cancel_event)
            eng.add_sink(sinks.TuiSink(tui_queue=tui_queue, run_id=run_id))
            return eng.run_once(
                stages=stages_list,
                single_stage=True,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
                on_error=on_error,
                allow_uncached_incremental=allow_uncached_incremental,
                checkout_missing=checkout_missing,
            )

    with _run_common.suppress_stderr_logging():
        return run_tui.run_with_tui(
            stages_list, tui_queue, executor_func, tui_log=tui_log, cancel_event=cancel_event
        )


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

    # Determine display mode from flags
    use_tui = tui_flag

    if use_tui:
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
        results = _run_common.run_json_mode(
            stages_list,
            single_stage=True,
            force=force,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )
    else:
        results = _run_common.run_plain_mode(
            stages_list,
            single_stage=True,
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
