"""Integration tests for engine shutdown and drain thread behavior."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import threading
from typing import TYPE_CHECKING, Any, cast

import anyio
import pytest

from conftest import AsyncEventCaptureSink
from helpers import register_test_stage
from pivot.engine import engine as engine_mod
from pivot.engine import sinks, sources
from pivot.types import LogMessage, OutputMessage, OutputMessageKind, StageStatus

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pivot.pipeline.pipeline import Pipeline


def _helper_noop(params: None) -> dict[str, str]:
    """No-op stage for testing."""
    return {"result": "ok"}


def _helper_with_output() -> dict[str, str]:
    """Stage that writes to stdout/stderr for drain tests."""
    logger = logging.getLogger(__name__)
    print("Test output to stdout", flush=True)
    print("Test error to stderr", file=sys.stderr, flush=True)
    logger.warning("Test output to stdout")
    logger.error("Test error to stderr")
    return {"result": "ok"}


async def _helper_set_shutdown_event(delay: float, shutdown_event: threading.Event) -> None:
    """Set shutdown event after a short delay."""
    await anyio.sleep(delay)
    shutdown_event.set()


def _helper_flatten_exception_group(exc: BaseException) -> list[BaseException]:
    """Flatten nested ExceptionGroup instances."""
    if isinstance(exc, BaseExceptionGroup):
        flattened: list[BaseException] = []
        for sub_exc in exc.exceptions:
            flattened.extend(_helper_flatten_exception_group(sub_exc))
        return flattened
    return [exc]


async def _helper_drain_with_signal(
    eng: engine_mod.Engine,
    output_queue: mp.Queue[OutputMessage],
    shutdown_event: threading.Event,
    completed: anyio.Event,
) -> None:
    """Drain output queue and signal when complete."""
    try:
        await eng._drain_output_queue(output_queue, shutdown_event)
    finally:
        completed.set()


@pytest.mark.anyio
async def test_engine_shutdown_completes_without_hanging(minimal_pipeline: Pipeline) -> None:
    """Engine shutdown should complete without hanging."""
    register_test_stage(_helper_noop, name="test_stage")

    async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        collector = sinks.ResultCollectorSink()
        eng.add_sink(collector)
        eng.add_source(sources.OneShotSource(stages=["test_stage"], force=True, reason="test"))

        with anyio.fail_after(10.0):
            await eng.run(exit_on_completion=True)

        results = await collector.get_results()
        assert "test_stage" in results
        assert results["test_stage"]["status"] in (
            StageStatus.RAN,
            StageStatus.SKIPPED,
            StageStatus.FAILED,
        )


@pytest.mark.anyio
async def test_drain_thread_exits_on_sentinel(minimal_pipeline: Pipeline) -> None:
    """Drain thread should exit when sentinel is received."""
    shutdown_event = threading.Event()
    spawn_ctx = mp.get_context("spawn")

    with spawn_ctx.Manager() as manager:
        output_queue: mp.Queue[OutputMessage] = cast("Any", manager.Queue())

        async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
            with anyio.fail_after(7.0):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(eng._drain_output_queue, output_queue, shutdown_event)
                    output_queue.put(
                        LogMessage(
                            kind=OutputMessageKind.LOG,
                            stage="stage",
                            line="line-1",
                            is_stderr=False,
                        )
                    )
                    output_queue.put(
                        LogMessage(
                            kind=OutputMessageKind.LOG, stage="stage", line="line-2", is_stderr=True
                        )
                    )
                    output_queue.put(None)


@pytest.mark.anyio
async def test_drain_thread_exits_on_shutdown_event_without_sentinel(
    minimal_pipeline: Pipeline,
) -> None:
    """Drain thread should exit when shutdown_event is set without sentinel."""
    shutdown_event = threading.Event()
    spawn_ctx = mp.get_context("spawn")

    with spawn_ctx.Manager() as manager:
        output_queue: mp.Queue[OutputMessage] = cast("Any", manager.Queue())

        async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
            with anyio.fail_after(7.0):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(eng._drain_output_queue, output_queue, shutdown_event)
                    tg.start_soon(_helper_set_shutdown_event, 0.1, shutdown_event)


@pytest.mark.anyio
async def test_engine_shutdown_after_orchestration_exception(
    minimal_pipeline: Pipeline, mocker: MockerFixture
) -> None:
    """Engine should shut down cleanly even if orchestration raises."""
    register_test_stage(_helper_noop, name="test_stage")
    mocker.patch.object(
        engine_mod.Engine,
        "_start_ready_stages",
        autospec=True,
        side_effect=RuntimeError("boom"),
    )

    async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        eng.add_source(sources.OneShotSource(stages=["test_stage"], force=True, reason="test"))

        with anyio.fail_after(10.0):
            with pytest.raises(BaseExceptionGroup) as excinfo:
                await eng.run(exit_on_completion=True)
        flattened = _helper_flatten_exception_group(excinfo.value)
        assert any(isinstance(error, RuntimeError) and str(error) == "boom" for error in flattened)


@pytest.mark.anyio
async def test_engine_shutdown_forwards_output_messages(minimal_pipeline: Pipeline) -> None:
    """Engine shutdown should forward output messages through the drain thread."""
    register_test_stage(_helper_with_output, name="output_stage")

    async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
        collector = sinks.ResultCollectorSink()
        event_sink = AsyncEventCaptureSink()
        eng.add_sink(collector)
        eng.add_sink(event_sink)
        eng.add_source(sources.OneShotSource(stages=["output_stage"], force=True, reason="test"))

        with anyio.fail_after(10.0):
            await eng.run(exit_on_completion=True)

        results = await collector.get_results()
        assert "output_stage" in results

        log_lines = [event for event in event_sink.events if event["type"] == "log_line"]
        assert log_lines, f"Expected log_line events, got: {event_sink.events}"
        assert any("Test output to stdout" in event["line"] for event in log_lines), (
            f"Log lines: {log_lines}"
        )
        assert any("Test error to stderr" in event["line"] for event in log_lines), (
            f"Log lines: {log_lines}"
        )


@pytest.mark.anyio
async def test_engine_no_message_loss_on_normal_shutdown(minimal_pipeline: Pipeline) -> None:
    """Drain thread should forward all messages before exit on normal shutdown."""
    shutdown_event = threading.Event()
    spawn_ctx = mp.get_context("spawn")

    with spawn_ctx.Manager() as manager:
        output_queue: mp.Queue[OutputMessage] = cast("Any", manager.Queue())
        event_sink = AsyncEventCaptureSink()

        async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
            eng.add_sink(event_sink)
            drain_complete = anyio.Event()

            with anyio.fail_after(10.0):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(eng._dispatch_outputs)
                    tg.start_soon(
                        _helper_drain_with_signal,
                        eng,
                        output_queue,
                        shutdown_event,
                        drain_complete,
                    )

                    for i in range(100):
                        output_queue.put(
                            LogMessage(
                                kind=OutputMessageKind.LOG,
                                stage="stage",
                                line=f"message-{i}",
                                is_stderr=False,
                            )
                        )

                    output_queue.put(None)

                    await drain_complete.wait()
                    if eng._output_send is not None:
                        await eng._output_send.aclose()

        log_lines = [event for event in event_sink.events if event["type"] == "log_line"]
        assert len(log_lines) == 100, f"Expected 100 lines, got {len(log_lines)}"
        received = {event["line"] for event in log_lines}
        for i in range(100):
            assert f"message-{i}" in received, f"Missing message-{i}"


@pytest.mark.anyio
async def test_engine_cancel_during_active_drain(minimal_pipeline: Pipeline) -> None:
    """Engine should handle cancellation during active drain gracefully."""
    shutdown_event = threading.Event()
    spawn_ctx = mp.get_context("spawn")

    with spawn_ctx.Manager() as manager:
        output_queue: mp.Queue[OutputMessage] = cast("Any", manager.Queue())
        event_sink = AsyncEventCaptureSink()

        async with engine_mod.Engine(pipeline=minimal_pipeline) as eng:
            eng.add_sink(event_sink)
            with anyio.fail_after(10.0):
                try:
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(eng._drain_output_queue, output_queue, shutdown_event)

                        for i in range(50):
                            output_queue.put(
                                LogMessage(
                                    kind=OutputMessageKind.LOG,
                                    stage="stage",
                                    line=f"message-{i}",
                                    is_stderr=False,
                                )
                            )

                        await anyio.sleep(0)
                        tg.cancel_scope.cancel()
                except anyio.get_cancelled_exc_class():
                    pass
                finally:
                    shutdown_event.set()

            # Verify cleanup happened - worker pool should be None after context exit
            # (The key test is that we didn't hang; message processing is secondary)
            assert eng._worker_pool is None, "Worker pool should be cleaned up after shutdown"
