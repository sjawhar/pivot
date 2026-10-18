from __future__ import annotations

import contextlib
import errno
import logging
import os
import pathlib
import signal
import threading
import time
import typing
from dataclasses import dataclass, field

import xxhash

from pivot.executor import core as executor_core
from pivot.executor import worker

_T = typing.TypeVar("_T")

_logger = logging.getLogger(__name__)

_FileStamp = tuple[int, int, int, int, int]


def _file_stamp(path: str) -> _FileStamp | None:
    """Return the stat signature and content digest of a user-code file, or None if unreadable.

    The digest catches an edit that leaves every timestamp unchanged: a filesystem with coarse
    timestamps, or one whose server clock lags, can stamp an edit with the time it already had.
    Running out of file descriptors is not an unreadable file, so it propagates.
    """
    try:
        stat = os.stat(path)
        digest = xxhash.xxh64_intdigest(pathlib.Path(path).read_bytes())
    except OSError as e:
        if e.errno in (errno.EMFILE, errno.ENFILE):
            raise
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_ctime_ns, digest


if typing.TYPE_CHECKING:
    import concurrent.futures
    import multiprocessing as mp
    from multiprocessing.managers import SyncManager

    import loky

    from pivot.types import OutputMessage


@dataclass(slots=True)
class WorkerPool:
    """Manage engine-lifetime workers and the per-run output queue.

    Workers are reused only while every user-code file they may have imported keeps the stat
    signature and content digest Pivot recorded before they imported it. Stamps are compared
    only with stamps, never with a clock, so a file timestamp that lags the coordinator's clock
    or does not move at all cannot hide an edit.
    """

    _executor: loky.ProcessPoolExecutor | None = None
    _manager: SyncManager | None = None
    _output_queue: mp.Queue[OutputMessage] | None = None
    _shutdown_event: threading.Event | None = None
    _accepting: bool = True
    _in_run: bool = False
    _max_workers: int = 0
    _user_code_files: dict[str, _FileStamp | None] = field(default_factory=dict)
    _code_changed: bool = False

    def start(self, *, max_workers: int) -> None:
        """Start a run without shrinking workers warmed by earlier runs."""
        if self._in_run:
            return
        self._max_workers = max(self._max_workers, max_workers)
        executor = typing.cast(
            "loky.ProcessPoolExecutor", executor_core.create_executor(self._max_workers)
        )
        fresh_workers = executor is not self._executor
        if self._executor is executor and self._has_stale_user_code():
            executor_core.restart_workers(self._max_workers, self._max_workers)
            executor = typing.cast(
                "loky.ProcessPoolExecutor",
                executor_core.create_executor(self._max_workers),
            )
            fresh_workers = True
        if fresh_workers:
            self._restamp_user_code_files()
        self._stamp_coordinator_user_code()
        self._executor = executor
        self._output_queue = typing.cast("mp.Queue[OutputMessage]", self._new_output_queue())
        self._shutdown_event = threading.Event()
        self._accepting = True
        self._in_run = True

    def _new_output_queue(self) -> object:
        """Return a fresh queue from the engine-lifetime manager, replacing it if unreachable."""
        if self._manager is not None:
            try:
                return self._manager.Queue()
            except (OSError, EOFError) as e:
                _logger.warning(
                    "Output-queue manager unreachable between runs (%r); starting a new one", e
                )
                self._manager.shutdown()
                self._manager = None
        self._manager = executor_core.create_manager()
        return self._manager.Queue()

    @property
    def in_run(self) -> bool:
        """Whether this pool has a queue and accepts dispatches for the current run."""
        return self._in_run

    def record_user_code_files(self, paths: list[str]) -> None:
        """Record the user-code files a worker reports having imported.

        A file Pivot had not stamped before the worker imported it may have changed in between,
        so its first report marks the workers stale; the next run replaces them after stamping
        it. Files already stamped are compared when the next run starts.
        """
        for path in paths:
            if path not in self._user_code_files:
                self._user_code_files[path] = None
                self._code_changed = True

    def mark_workers_unverified(self) -> None:
        """Replace the workers before the next run: a task ended without reporting its imports."""
        self._code_changed = True

    def _has_stale_user_code(self) -> bool:
        return self._code_changed or any(
            _file_stamp(path) != stamp for path, stamp in self._user_code_files.items()
        )

    def _restamp_user_code_files(self) -> None:
        """Stamp every known user-code file before fresh workers can import it."""
        self._user_code_files = {path: _file_stamp(path) for path in self._user_code_files}
        self._code_changed = False

    def _stamp_coordinator_user_code(self) -> None:
        """Stamp the user code this process imported, before this run's workers can import it."""
        for path in worker.user_code_files():
            if path not in self._user_code_files:
                self._user_code_files[path] = _file_stamp(path)

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
        """Stop workers mid-stage and signal the drain thread to stop.

        Each worker is sent SIGTERM, which unwinds its running stage
        (``worker.unwind_when_stopped``) so the stage's ``finally`` blocks run, and is given
        ``worker.TERMINATION_GRACE_S`` to exit; a worker still alive then is killed with its
        descendants.
        """
        self._accepting = False
        self._in_run = False
        if self._executor is not None:
            # loky exposes no public list of its worker processes.
            processes = list(self._executor._processes.values())
            for process in processes:
                if process.pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + worker.TERMINATION_GRACE_S
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            self._executor.shutdown(wait=False, kill_workers=True)
            self._executor = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        self._output_queue = None
        self._shutdown_event = None
