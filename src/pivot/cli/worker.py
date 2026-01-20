"""Worker command for distributed execution.

The worker connects to a coordinator and executes assigned stages.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
import uuid
from typing import TYPE_CHECKING

import click

from pivot import project
from pivot.cli import decorators as cli_decorators
from pivot.executor import protocol

if TYPE_CHECKING:
    from pivot.types import StageResult

logger = logging.getLogger(__name__)


@cli_decorators.pivot_command()
@click.option(
    "--coordinator",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Path to coordinator Unix socket",
)
@click.option(
    "--worker-id",
    type=str,
    default=None,
    help="Worker ID (defaults to generated UUID)",
)
def worker(coordinator: pathlib.Path, worker_id: str | None) -> None:
    """Run as a worker process, connecting to coordinator.

    Workers connect to a running coordinator (started via 'pivot serve'),
    request tasks, execute them, and report results.
    """
    if worker_id is None:
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"

    click.echo(f"Worker {worker_id} connecting to {coordinator}")

    try:
        asyncio.run(_run_worker(coordinator, worker_id))
    except KeyboardInterrupt:
        click.echo("\nWorker interrupted")
        sys.exit(0)


async def _run_worker(socket_path: pathlib.Path, worker_id: str) -> None:
    """Run the worker event loop."""
    client = protocol.WorkerClient(socket_path)
    pid = os.getpid()

    try:
        await client.connect()
        logger.info(f"Connected to coordinator at {socket_path}")

        # Register with coordinator
        result = await client.register(worker_id, pid)
        logger.info(f"Registered as worker: {result}")

        # Get project paths
        proj_root = project.get_project_root()
        cache_dir = proj_root / ".pivot" / "cache"

        # Main work loop
        while True:
            # Request next task
            response = await client.request_task()

            # Check for shutdown signal
            if response.get("shutdown"):
                logger.info("Coordinator is shutting down")
                break

            task = response.get("task")
            if task is None:
                # No work available, wait and retry
                await asyncio.sleep(0.1)
                continue

            stage_name = task["stage_name"]
            logger.info(f"Executing stage: {stage_name}")

            try:
                # Execute the stage
                result = await _execute_stage_async(
                    stage_name=stage_name,
                    cache_dir=cache_dir,
                    force=task["force"],
                    no_commit=task["no_commit"],
                    no_cache=task["no_cache"],
                )

                # Report result
                await client.report_result(stage_name, result)
                logger.info(f"Completed stage: {stage_name} ({result['status']})")

            except Exception as e:
                # Report failure
                from pivot.types import StageResult, StageStatus

                error_result = StageResult(
                    status=StageStatus.FAILED,
                    reason=str(e),
                    output_lines=[],
                )
                await client.report_result(stage_name, error_result)
                logger.error(f"Stage {stage_name} failed: {e}")

            # Send heartbeat periodically
            await client.heartbeat()

    except ConnectionRefusedError as e:
        logger.error(f"Could not connect to coordinator at {socket_path}")
        raise click.ClickException(f"Coordinator not running at {socket_path}") from e
    finally:
        await client.disconnect()
        logger.info("Disconnected from coordinator")


async def _execute_stage_async(
    stage_name: str,
    cache_dir: pathlib.Path,
    force: bool,
    no_commit: bool,
    no_cache: bool,
) -> StageResult:
    """Execute a stage asynchronously (runs in thread pool)."""
    from pivot import config, project, registry
    from pivot.executor import worker as worker_exec

    loop = asyncio.get_running_loop()

    # Prepare worker info
    info = registry.REGISTRY.get(stage_name)

    worker_info = worker_exec.WorkerStageInfo(
        func=info["func"],
        fingerprint=info["fingerprint"],
        deps=info["deps_paths"],
        outs=info["outs"],
        signature=info["signature"],
        params=info["params"],
        variant=info["variant"],
        overrides={},  # Could be passed from coordinator
        checkout_modes=[config.CheckoutMode.COPY],
        run_id="",  # Could be passed from coordinator
        force=force,
        no_commit=no_commit,
        no_cache=no_cache,
        dep_specs=info["dep_specs"],
        out_specs=info["out_specs"],
        params_arg_name=info["params_arg_name"],
        project_root=project.get_project_root(),
        state_dir=config.get_state_dir(),
    )

    # Create a dummy output queue (we're not forwarding output to TUI)
    import multiprocessing as mp

    from pivot.types import OutputMessage  # noqa: TC001 - used at runtime for type annotation

    spawn_ctx = mp.get_context("spawn")
    manager = spawn_ctx.Manager()
    # Manager().Queue() returns AutoProxy[Queue] which is typed as Queue[Any]
    output_queue: mp.Queue[OutputMessage] = manager.Queue()  # pyright: ignore[reportAssignmentType]

    try:
        # Run stage execution in thread pool to not block event loop
        result: StageResult = await loop.run_in_executor(
            None,
            worker_exec.execute_stage,
            stage_name,
            worker_info,
            cache_dir,
            output_queue,
        )
        return result
    finally:
        manager.shutdown()
