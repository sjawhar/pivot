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

# How long hard_cancel waits for a SIGKILLed worker to be reaped; it never blocks pivot's exit.
_KILLED_WORKER_JOIN_S = 5.0

_FileStamp = tuple[int, int, int, int, int]


# A ping round normally clears as soon as each idle worker has taken one ping; the timeout only
# bounds a round whose estimate of idle workers was off by a worker that exited meanwhile.
_KEEPALIVE_PING_TIMEOUT_S = 0.25


def _keepalive_ping(barrier: threading.Barrier) -> None:
    """Restart one idle worker's idle clock; the barrier spreads a round over distinct workers."""
    with contextlib.suppress(threading.BrokenBarrierError):
        _ = barrier.wait(_KEEPALIVE_PING_TIMEOUT_S)


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
    _idle_timeout_s: float | None = None
    # Parent -> (next ping deadline, consumer workers to keep warm). The Engine loop services
    # the deadlines, so a parent costs no thread.
    _keepalives: dict[str, tuple[float, int]] = field(default_factory=dict)
    # Ping futures are grouped by parent so completion and shutdown cancel queued pings.
    _pings: dict[str, list[concurrent.futures.Future[None]]] = field(default_factory=dict)
    _user_code_files: dict[str, _FileStamp | None] = field(default_factory=dict)
    _code_changed: bool = False

    def start(self, *, max_workers: int, idle_timeout_s: float | None = None) -> None:
        """Start a run without shrinking workers warmed by earlier runs."""
        if self._in_run:
            return
        self._max_workers = max(self._max_workers, max_workers)
        executor = self._executor
        fresh_workers = True
        if executor is None:
            executor = executor_core.create_executor(
                self._max_workers, idle_timeout_s=idle_timeout_s
            )
        elif (
            self._idle_timeout_s != idle_timeout_s
            or self._has_stale_user_code()
            or not executor_core.is_executor_usable(executor, self._max_workers)
        ):
            executor = self._replace_executor(idle_timeout_s)
        else:
            fresh_workers = False
        # An argument change can make loky replace a live reusable executor during first use.
        # Recheck every acquisition so this pool never keeps its narrowed call queue.
        if not executor_core.is_executor_usable(executor, self._max_workers):
            executor = self._replace_executor(idle_timeout_s)
        if fresh_workers:
            self._restamp_user_code_files()
        self._stamp_coordinator_user_code()
        self._executor = executor
        self._idle_timeout_s = idle_timeout_s
        self._output_queue = typing.cast("mp.Queue[OutputMessage]", self._new_output_queue())
        self._shutdown_event = threading.Event()
        self._accepting = True
        self._in_run = True

    def restart(self, *, max_workers: int) -> None:
        """Replace every worker before a reload starts its next run.

        ``max_workers`` is the reloaded run's width; that many fresh workers start together.
        If starting them fails, the pool holds no executor and the next ``start()`` creates
        one, as on first use.
        """
        self._max_workers = max(self._max_workers, max_workers)
        self._executor = self._replace_executor(self._idle_timeout_s, warm_workers=max_workers)
        self._restamp_user_code_files()

    def _replace_executor(
        self, idle_timeout_s: float | None, *, warm_workers: int = 0
    ) -> loky.ProcessPoolExecutor:
        """Stop this pool's workers, then return a fresh executor.

        The pool stops the executor it holds rather than asking loky for "the" reusable
        executor: loky 3.7 keeps one per thread, and a watch reload restarts from an anyio
        worker thread, so the executor this pool holds need not be the one loky finds on the
        calling thread. The pool forgets the old executor as soon as it is shut down, so a
        failure starting the fresh workers never leaves it holding one that rejects work.

        A reload warms ``warm_workers`` workers, all started together, before its next run. A
        replacement that ``start()`` makes belongs to that run's critical path, so it warms
        none and lets its submitted stages start every worker they need in parallel.
        """
        if self._executor is not None:
            executor = self._executor
            self._executor = None
            executor.shutdown(wait=True, kill_workers=True)
        if warm_workers:
            executor_core.restart_workers(
                self._max_workers, warm_workers=warm_workers, idle_timeout_s=idle_timeout_s
            )
            return executor_core.create_executor(self._max_workers, idle_timeout_s=idle_timeout_s)
        return executor_core.create_fresh_executor(self._max_workers, idle_timeout_s=idle_timeout_s)

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
        prestart_target: int = 0,
        **kwargs: object,
    ) -> concurrent.futures.Future[_T]:
        """Submit work to the underlying executor."""
        if not self._accepting:
            raise RuntimeError("WorkerPool is not accepting new submissions")
        if self._executor is None:
            raise RuntimeError("WorkerPool not started")
        self._prune_pings()
        return executor_core.submit(
            self._executor,
            self._max_workers,
            fn,
            *args,
            prestart_target=prestart_target,
            keepalive_pings=self._active_ping_count(),
            **kwargs,
        )

    def keep_consumers_warm(self, parent: str, worker_count: int) -> None:
        """Keep ``parent``'s pre-started consumer workers from idling out while it runs.

        ``run_due_keepalives`` pings up to ``worker_count`` idle workers every half idle timeout
        until ``release_consumers(parent)`` or the end of the run.
        """
        if worker_count <= 0 or self._idle_timeout_s is None:
            return
        if parent in self._keepalives:
            raise RuntimeError(f"workers already kept warm for {parent!r}")
        self._keepalives[parent] = (time.monotonic() + self._idle_timeout_s / 2, worker_count)

    def run_due_keepalives(self) -> None:
        """Ping pre-started idle workers whose parent has crossed the half-timeout deadline."""
        if not self._keepalives:
            return
        if self._executor is None or self._manager is None or self._idle_timeout_s is None:
            return
        now = time.monotonic()
        for parent, (deadline, worker_count) in list(self._keepalives.items()):
            if deadline > now:
                continue
            pings = min(worker_count, executor_core.idle_worker_count(self._executor))
            if pings > 0:
                barrier = self._manager.Barrier(pings)
                self._pings.setdefault(parent, []).extend(
                    executor_core.submit_to_live_workers(self._executor, _keepalive_ping, barrier)
                    for _ in range(pings)
                )
            self._keepalives[parent] = (now + self._idle_timeout_s / 2, worker_count)
        self._prune_pings()

    def release_consumers(self, parent: str) -> None:
        """Stop pinging a resolved parent's consumers and cancel queued ping work."""
        self._keepalives.pop(parent, None)
        for ping in self._pings.pop(parent, []):
            _ = ping.cancel()

    def _prune_pings(self) -> None:
        for parent, pings in list(self._pings.items()):
            active = [ping for ping in pings if not ping.done()]
            if active:
                self._pings[parent] = active
            else:
                del self._pings[parent]

    def _active_ping_count(self) -> int:
        return sum(len(pings) for pings in self._pings.values())

    def _stop_keepalives(self) -> None:
        self._keepalives.clear()
        for pings in self._pings.values():
            for ping in pings:
                _ = ping.cancel()
        self._pings.clear()

    def stop_accepting(self) -> None:
        """Stop accepting new submissions without shutting down workers."""
        self._accepting = False

    def end_run(self) -> None:
        """Release this run's queue while keeping workers warm for the next run."""
        self._accepting = False
        self._stop_keepalives()
        self._output_queue = None
        self._shutdown_event = None
        self._in_run = False

    def shutdown(self) -> None:
        """Shut down workers and the manager when the owning Engine exits."""
        self._accepting = False
        self._in_run = False
        self._stop_keepalives()
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

        Each worker's main thread, where its stage runs, is sent SIGTERM (``worker.terminate``),
        which unwinds its running stage (``worker.unwind_when_stopped``) so the stage's cleanup
        runs, and the worker is given ``worker.TERMINATION_GRACE_S`` to exit; a worker still
        alive then is killed with its descendants.

        The workers are first taken out of loky's executor. Its manager thread takes a worker
        that exits without being asked to for a broken pool and SIGKILLs every worker the
        executor still lists, so the first worker to exit would cut the others' cleanup off.
        """
        self._accepting = False
        self._in_run = False
        self._stop_keepalives()
        try:
            if self._executor is not None:
                executor = self._executor
                self._executor = None
                try:
                    self._stop_workers(executor)
                finally:
                    # Only shutdown wakes loky's manager thread once its workers are gone.
                    executor.shutdown(wait=False, kill_workers=True)
        finally:
            if self._manager is not None:
                self._manager.shutdown()
                self._manager = None
            if self._shutdown_event is not None:
                self._shutdown_event.set()
            self._output_queue = None
            self._shutdown_event = None

    @staticmethod
    def _stop_workers(executor: loky.ProcessPoolExecutor) -> None:
        """Take ``executor``'s workers out of it, send each SIGTERM (``worker.terminate``),
        and SIGKILL with its descendants any worker still alive after
        ``worker.TERMINATION_GRACE_S``."""
        # loky exposes no public list of its worker processes. A watch reload may have
        # replaced this executor already, which leaves it no lock and no workers.
        with executor._processes_management_lock or contextlib.nullcontext():
            processes = list(executor._processes.values())
            executor._processes.clear()
        for process in processes:
            if process.pid is not None:
                with contextlib.suppress(ProcessLookupError):
                    worker.terminate(process.pid)
        deadline = time.monotonic() + worker.TERMINATION_GRACE_S
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.pid is not None and process.exitcode is None:
                worker.kill_descendants(process.pid)
                # loky's process object has no kill(); signal the pid directly.
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process.pid, signal.SIGKILL)
                process.join(_KILLED_WORKER_JOIN_S)
