"""Serve command for starting the coordinator.

The coordinator manages execution state and distributes work to workers.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import signal
import sys

import click

from pivot import project
from pivot.cli import decorators as cli_decorators
from pivot.executor import coordinator
from pivot.types import OnError

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = ".pivot/coordinator.sock"


@cli_decorators.pivot_command()
@click.option(
    "--socket",
    "socket_path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=f"Path for coordinator socket (default: {DEFAULT_SOCKET_PATH})",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=8,
    help="Maximum number of concurrent stages",
)
@click.option(
    "--on-error",
    type=click.Choice(["fail", "keep-going"]),
    default="fail",
    help="Error handling mode",
)
@click.option(
    "--run",
    "start_run",
    is_flag=True,
    help="Start a run immediately (otherwise wait for RPC)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-run all stages (with --run)",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Write to pending directory (with --run)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip caching entirely (with --run)",
)
@click.argument("stages", nargs=-1)
def serve(
    socket_path: pathlib.Path | None,
    max_workers: int,
    on_error: str,
    start_run: bool,
    force: bool,
    no_commit: bool,
    no_cache: bool,
    stages: tuple[str, ...],
) -> None:
    """Start the coordinator server for distributed execution.

    The coordinator manages pipeline state and distributes work to workers.
    Workers connect via 'pivot worker --coordinator <socket>'.

    Examples:

        # Start coordinator and wait for RPC commands
        pivot serve

        # Start coordinator and begin execution immediately
        pivot serve --run

        # Start with specific stages
        pivot serve --run train evaluate
    """
    proj_root = project.get_project_root()

    if socket_path is None:
        socket_path = proj_root / DEFAULT_SOCKET_PATH

    cache_dir = proj_root / ".pivot" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    error_mode = OnError.FAIL if on_error == "fail" else OnError.KEEP_GOING

    click.echo(f"Starting coordinator at {socket_path}")
    click.echo(f"Max workers: {max_workers}")

    try:
        asyncio.run(
            _run_coordinator(
                cache_dir=cache_dir,
                socket_path=socket_path,
                max_workers=max_workers,
                on_error=error_mode,
                start_run=start_run,
                stages=list(stages) if stages else None,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
            )
        )
    except KeyboardInterrupt:
        click.echo("\nCoordinator stopped")
        sys.exit(0)


async def _run_coordinator(
    cache_dir: pathlib.Path,
    socket_path: pathlib.Path,
    max_workers: int,
    on_error: OnError,
    start_run: bool,
    stages: list[str] | None,
    force: bool,
    no_commit: bool,
    no_cache: bool,
) -> None:
    """Run the coordinator event loop."""
    coord = coordinator.Coordinator(
        cache_dir=cache_dir,
        max_workers=max_workers,
        on_error=on_error,
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await coord.start(socket_path)
        click.echo(f"Coordinator listening on {socket_path}")
        click.echo("Waiting for workers to connect...")

        # Start run if requested
        if start_run:
            run_id = coord.start_run(
                stages=stages,
                force=force,
                no_commit=no_commit,
                no_cache=no_cache,
            )
            click.echo(f"Started run: {run_id}")

        # Main loop - wait for shutdown or completion
        while not stop_event.is_set():
            # Check if run is complete
            if start_run and coord.is_complete():
                summary = coord.get_run_summary()
                msg = (
                    "\nRun complete: "
                    + f"{summary['ran']} ran, "
                    + f"{summary['skipped']} skipped, "
                    + f"{summary['failed']} failed"
                )
                click.echo(msg)
                break

            # Small sleep to not busy-wait
            # contextlib.suppress doesn't work with async/await
            try:  # noqa: SIM105
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
            except TimeoutError:
                pass

    finally:
        await coord.stop()
        click.echo("Coordinator stopped")
