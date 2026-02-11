from __future__ import annotations

import multiprocessing as mp
import threading
import typing
from dataclasses import dataclass

from pivot.executor import core as executor_core

_T = typing.TypeVar("_T")

if typing.TYPE_CHECKING:
    import concurrent.futures
    from multiprocessing.managers import SyncManager

    import loky

    from pivot.types import OutputMessage


@dataclass(slots=True)
class WorkerPool:
    """Manage worker executor, manager queue, and shutdown signaling."""

    _executor: loky.ProcessPoolExecutor | None = None
    _manager: SyncManager | None = None
    _output_queue: mp.Queue[OutputMessage] | None = None
    _shutdown_event: threading.Event | None = None
    _accepting: bool = True

    def start(self, *, max_workers: int) -> None:
        """Initialize executor, manager, output queue, and shutdown event."""
        self._executor = typing.cast(
            "loky.ProcessPoolExecutor", executor_core.create_executor(max_workers)
        )
        spawn_ctx = mp.get_context("spawn")
        self._manager = spawn_ctx.Manager()
        raw_queue = self._manager.Queue()
        queue_obj = typing.cast("object", raw_queue)
        self._output_queue = typing.cast("mp.Queue[OutputMessage]", queue_obj)
        self._shutdown_event = threading.Event()
        self._accepting = True

    def output_queue(self) -> mp.Queue[OutputMessage]:
        """Return the worker output queue."""
        if self._output_queue is None:
            raise RuntimeError("WorkerPool not started")
        return self._output_queue

    def shutdown_event(self) -> threading.Event:
        """Return the shutdown event used by the drain thread."""
        if self._shutdown_event is None:
            raise RuntimeError("WorkerPool not started")
        return self._shutdown_event

    def submit(
        self,
        fn: typing.Callable[..., _T],
        /,
        *args: object,
        **kwargs: object,
    ) -> concurrent.futures.Future[_T]:
        """Submit work to the underlying executor."""
        if not self._accepting:
            raise RuntimeError("WorkerPool is not accepting new submissions")
        if self._executor is None:
            raise RuntimeError("WorkerPool not started")
        return self._executor.submit(fn, *args, **kwargs)

    def stop_accepting(self) -> None:
        """Stop accepting new submissions without shutting down workers."""
        self._accepting = False

    def shutdown(self) -> None:
        """Shut down executor and manager, waiting for completion."""
        self._accepting = False
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._manager is not None:
            self._manager.shutdown()

    def hard_cancel(self) -> None:
        """Forcefully shut down workers and signal drain thread to stop."""
        self._accepting = False
        if self._executor is not None:
            self._executor.shutdown(wait=False, kill_workers=True)
        if self._manager is not None:
            self._manager.shutdown()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
