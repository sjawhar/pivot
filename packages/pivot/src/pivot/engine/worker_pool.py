from __future__ import annotations

import multiprocessing as mp
import pathlib
import threading
import typing
from dataclasses import dataclass, field

from pivot.executor import core as executor_core

_T = typing.TypeVar("_T")

_FileStamp = tuple[int, int, int, int]


def _file_stamp(path: str) -> _FileStamp | None:
    """Return the stat fields that change when imported user code changes."""
    try:
        stat = pathlib.Path(path).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_ctime_ns


if typing.TYPE_CHECKING:
    import concurrent.futures
    from multiprocessing.managers import SyncManager

    import loky

    from pivot.types import OutputMessage


@dataclass(slots=True)
class WorkerPool:
    """Manage engine-lifetime workers and the per-run output queue."""

    _executor: loky.ProcessPoolExecutor | None = None
    _manager: SyncManager | None = None
    _output_queue: mp.Queue[OutputMessage] | None = None
    _shutdown_event: threading.Event | None = None
    _accepting: bool = True
    _in_run: bool = False
    _user_code_files: dict[str, _FileStamp] = field(default_factory=dict)
    _code_changed: bool = False

    def start(self, *, max_workers: int) -> None:
        """Start a run with the reusable executor and a fresh output queue."""
        if self._in_run:
            return
        executor = typing.cast(
            "loky.ProcessPoolExecutor", executor_core.create_executor(max_workers)
        )
        if self._executor is not None and executor is not self._executor:
            self._user_code_files.clear()
            self._code_changed = False
        if self._executor is executor and self._has_stale_user_code():
            executor_core.restart_workers(max_workers, max_workers)
            executor = typing.cast(
                "loky.ProcessPoolExecutor", executor_core.create_executor(max_workers)
            )
            self._user_code_files.clear()
            self._code_changed = False
        self._executor = executor
        if self._manager is None:
            spawn_ctx = mp.get_context("spawn")
            self._manager = spawn_ctx.Manager()
        raw_queue = self._manager.Queue()
        queue_obj = typing.cast("object", raw_queue)
        self._output_queue = typing.cast("mp.Queue[OutputMessage]", queue_obj)
        self._shutdown_event = threading.Event()
        self._accepting = True
        self._in_run = True

    @property
    def in_run(self) -> bool:
        """Whether this pool has a queue and accepts dispatches for the current run."""
        return self._in_run

    def record_user_code_files(self, paths: list[str], started_ns: int) -> None:
        """Record source files a worker imported while executing a stage.

        A first-seen file modified after the task started may have changed after that worker
        imported it, so the next run refreshes the pool rather than trusting its cached module.
        """
        for path in paths:
            stamp = _file_stamp(path)
            if stamp is None:
                continue
            existing = self._user_code_files.setdefault(path, stamp)
            if existing != stamp or (existing == stamp and stamp[3] >= started_ns):
                self._code_changed = True

    def _has_stale_user_code(self) -> bool:
        return self._code_changed or any(
            _file_stamp(path) != stamp for path, stamp in self._user_code_files.items()
        )

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

    def end_run(self) -> None:
        """Release this run's queue while keeping workers warm for the next run."""
        self._accepting = False
        self._output_queue = None
        self._shutdown_event = None
        self._in_run = False

    def shutdown(self) -> None:
        """Shut down workers and the manager when the owning Engine exits."""
        self._accepting = False
        self._in_run = False
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._output_queue = None
        self._shutdown_event = None

    def hard_cancel(self) -> None:
        """Forcefully shut down workers and signal drain thread to stop."""
        self._accepting = False
        self._in_run = False
        if self._executor is not None:
            self._executor.shutdown(wait=False, kill_workers=True)
            self._executor = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        self._output_queue = None
        self._shutdown_event = None
