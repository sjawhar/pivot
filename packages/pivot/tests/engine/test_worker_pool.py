# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import concurrent.futures
import errno
import logging
import os
import pathlib
import signal
import threading
import time
from typing import TYPE_CHECKING, cast

import loky.reusable_executor
import psutil
import pytest

from pivot.engine import worker_pool
from pivot.engine.worker_pool import WorkerPool
from pivot.executor import core as executor_core
from pivot.executor import worker

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager
    from multiprocessing.process import BaseProcess


def _helper_identity(value: int) -> int:
    return value


def _helper_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _helper_pid() -> int:
    return os.getpid()


def _helper_wait_for_release(ready_dir: str, release: str) -> int:
    ready_path = pathlib.Path(ready_dir) / str(os.getpid())
    ready_path.write_text("")
    release_path = pathlib.Path(release)
    deadline = time.monotonic() + 60
    while not release_path.exists():
        assert time.monotonic() < deadline, "test did not release waiting workers"
        time.sleep(0.01)
    return os.getpid()


def _helper_fail_warm_up() -> None:
    raise OSError("injected warm-up failure")


def _helper_live_loky_workers() -> set[int]:
    found = set[int]()
    for child in psutil.Process().children(recursive=True):
        try:
            if "popen_loky_posix" in " ".join(child.cmdline()):
                found.add(child.pid)
        except psutil.NoSuchProcess:  # exited (or a zombie) between listing and inspecting
            continue
    return found


def test_unknown_loky_reusable_executor_layout_names_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(loky.reusable_executor, "_executor", raising=False)
    monkeypatch.delattr(loky.reusable_executor, "_executor_storage", raising=False)

    with pytest.raises(RuntimeError, match=r"Unsupported loky \d"):
        _ = executor_core._reusable_worker_target(1, kill_workers=False)


def test_short_parents_schedule_no_timer_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    started = list[object]()

    class _Timer:
        daemon: bool

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            started.append(self)
            self.daemon = False

        def start(self) -> None:
            pass

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(threading, "Timer", _Timer)
    pool = WorkerPool(_idle_timeout_s=10)
    for stage in range(100):
        name = str(stage)
        pool.keep_consumers_warm(name, 1)
        pool.release_consumers(name)

    assert started == []


def test_keepalive_completion_and_cancellation_leave_no_pending_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Executor:
        _processes: dict[int, object] = {1: object()}
        _pending_work_items: dict[int, object] = {}

    class _Manager:
        def Barrier(self, _parties: int) -> object:  # noqa: N802
            return object()

    submitted = list[concurrent.futures.Future[None]]()

    def submit_ping(*_args: object) -> concurrent.futures.Future[None]:
        future = concurrent.futures.Future[None]()
        submitted.append(future)
        return future

    monkeypatch.setattr(executor_core, "submit_to_live_workers", submit_ping)
    pool = WorkerPool(
        _executor=cast("loky.ProcessPoolExecutor", cast("object", _Executor())),
        _manager=cast("SyncManager", cast("object", _Manager())),
        _idle_timeout_s=10,
    )

    pool.keep_consumers_warm("completed", 1)
    pool._keepalives["completed"] = (0, 1)
    pool.run_due_keepalives()
    submitted[-1].set_result(None)
    pool.release_consumers("completed")
    assert pool._pings == {}

    pool.keep_consumers_warm("cancelled", 1)
    pool._keepalives["cancelled"] = (0, 1)
    pool.run_due_keepalives()
    pool._stop_keepalives()
    assert submitted[-1].cancelled()
    assert pool._pings == {}


@pytest.mark.slow
def test_idle_workers_exit_while_another_stage_runs(tmp_path: pathlib.Path) -> None:
    idle_timeout_s = executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    release = tmp_path / "release"
    pool = WorkerPool()
    try:
        pool.start(max_workers=4, idle_timeout_s=idle_timeout_s)
        # A worker that loses loky's process-lock race when its timeout fires waits one more
        # timeout (loky `_process_worker`); the slow stage outlives both.
        slow = pool.submit(_helper_sleep, 2 * idle_timeout_s + 5)
        waiting = [
            pool.submit(_helper_wait_for_release, str(ready_dir), str(release)) for _ in range(3)
        ]
        deadline = time.monotonic() + 60
        while len(list(ready_dir.iterdir())) < len(waiting):
            assert time.monotonic() < deadline, "workers did not start"
            time.sleep(0.01)
        release.write_text("")
        idle_pids = {future.result(timeout=60) for future in waiting}

        deadline = time.monotonic() + 2 * idle_timeout_s + 5
        while any(psutil.pid_exists(pid) for pid in idle_pids):
            assert not slow.done(), "the slow stage ended before idle workers exited"
            assert time.monotonic() < deadline, "idle workers stayed alive"
            time.sleep(0.01)
    finally:
        pool.hard_cancel()


def test_restart_workers_replaces_live_workers() -> None:
    """A code reload must replace workers rather than returning loky's live singleton."""
    executor_core.restart_workers(1, idle_timeout_s=None)
    first = executor_core.create_executor(1).submit(_helper_pid).result(timeout=60)

    executor_core.restart_workers(1, idle_timeout_s=None)
    second = executor_core.create_executor(1).submit(_helper_pid).result(timeout=60)

    assert second != first, "restart_workers should replace every live worker"
    executor_core._cleanup_worker_pool()


def test_restart_from_another_thread_stops_the_pools_previous_workers() -> None:
    """A watch reload restarts workers from an anyio worker thread; the pool's old workers
    must still be stopped even when loky keeps its reusable executor per thread."""
    pool = WorkerPool()
    try:
        pool.start(max_workers=2, idle_timeout_s=None)
        old_pids = {pool.submit(_helper_pid).result(timeout=60) for _ in range(2)}
        pool.end_run()

        restarted = threading.Thread(target=lambda: pool.restart(max_workers=2))
        restarted.start()
        restarted.join(timeout=120)
        assert not restarted.is_alive(), "restart did not finish"

        deadline = time.monotonic() + 10
        alive = {pid for pid in old_pids if psutil.pid_exists(pid)}
        while alive and time.monotonic() < deadline:
            time.sleep(0.05)
            alive = {pid for pid in old_pids if psutil.pid_exists(pid)}
        assert not alive, f"restart left the previous workers alive: {sorted(alive)}"
    finally:
        pool.hard_cancel()


def test_timeout_change_after_a_reload_restart_stops_the_restarts_worker() -> None:
    """A run with a new idle timeout replaces the executor the pool holds. After a reload
    restarted it from an anyio worker thread, which outlives the reload, loky 3.7's lookup on
    this thread never sees that executor, so the pool must stop it itself."""
    pool = WorkerPool()
    reload_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        pool.start(max_workers=2, idle_timeout_s=None)
        pool.end_run()
        reload_thread.submit(lambda: pool.restart(max_workers=2)).result(timeout=120)
        pool.start(max_workers=2, idle_timeout_s=None)
        restart_pid = pool.submit(_helper_pid).result(timeout=60)
        pool.end_run()

        pool.start(max_workers=2, idle_timeout_s=executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S)

        deadline = time.monotonic() + 10
        while psutil.pid_exists(restart_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not psutil.pid_exists(restart_pid), (
            f"the timeout change left the restart's worker alive: {restart_pid}"
        )
    finally:
        pool.hard_cancel()
        reload_thread.shutdown(wait=True)


# Loaded by every Python process started while PYTHONPATH names its directory; only a loky
# worker that finds the marker file deletes it and exits, so exactly one fresh worker dies.
# Once a test imports joblib, its vendored loky re-registers the "loky" multiprocessing
# context, and workers launch from joblib's copy of the launcher module instead.
_DIE_ONCE_HOOK = """\
import os
import sys

_LAUNCHERS = ("loky.backend.popen_loky_posix", "joblib.externals.loky.backend.popen_loky_posix")
if sys.orig_argv[1:2] == ["-m"] and sys.orig_argv[2:3] and sys.orig_argv[2] in _LAUNCHERS:
    try:
        os.unlink(os.environ["PIVOT_TEST_DIE_ONCE"])
    except (KeyError, FileNotFoundError):
        pass
    else:
        os._exit(1)
"""


def test_reload_restart_whose_fresh_worker_dies_leaves_the_pool_able_to_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watch reload restarts workers off the loop thread. When the restart's fresh worker
    dies at startup, the Engine logs the failed restart and re-runs
    (``Engine._reload_and_rerun``); that re-run, and every later one, must still run stages."""
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_DIE_ONCE_HOOK)
    marker = tmp_path / "die-once"
    pool = WorkerPool()
    try:
        pool.start(max_workers=2)
        assert pool.submit(_helper_pid).result(timeout=60) > 0
        pool.end_run()

        failures = list[BaseException]()

        def _reload_restart() -> None:
            try:
                _ = pool.restart(max_workers=2)
            except Exception as exc:  # the Engine catches and logs this
                failures.append(exc)

        marker.write_text("")
        with monkeypatch.context() as patched:
            patched.setenv("PIVOT_TEST_DIE_ONCE", str(marker))
            patched.setenv(
                "PYTHONPATH",
                os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")])),
            )
            restarted = threading.Thread(target=_reload_restart)
            restarted.start()
            restarted.join(timeout=120)
        assert not restarted.is_alive(), "restart did not finish"
        assert not marker.exists(), "no fresh worker started during the restart"
        assert failures, "the restart's dead worker never reached the caller"

        for run in range(2):
            pool.start(max_workers=2)
            assert pool.submit(_helper_pid).result(timeout=60) > 0, f"re-run {run} failed"
            pool.end_run()
    finally:
        pool.hard_cancel()


def test_failed_reload_restart_leaves_no_worker_beyond_the_cap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload restart whose warm-up task fails after loky started the fresh worker must stop
    that worker. On loky 3.7 it belongs to the reload thread's executor, which no pool holds,
    so every later run would keep one worker beyond the cap for the rest of the session."""
    cap = 2
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    release = tmp_path / "release"
    baseline = _helper_live_loky_workers()
    pool = WorkerPool()
    reload_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        pool.start(max_workers=cap, idle_timeout_s=None)
        assert pool.submit(_helper_pid).result(timeout=60) > 0
        pool.end_run()

        with monkeypatch.context() as patched:
            patched.setattr(executor_core, "_noop", _helper_fail_warm_up)
            restart = reload_thread.submit(lambda: pool.restart(max_workers=cap))
            with pytest.raises(OSError, match="injected warm-up failure"):
                _ = restart.result(timeout=120)

        pool.start(max_workers=cap, idle_timeout_s=None)
        running = [
            pool.submit(_helper_wait_for_release, str(ready_dir), str(release)) for _ in range(cap)
        ]
        deadline = time.monotonic() + 60
        while len(list(ready_dir.iterdir())) < cap:
            assert time.monotonic() < deadline, "the re-run's workers did not start"
            time.sleep(0.01)
        live = _helper_live_loky_workers() - baseline
        release.write_text("")
        for future in running:
            _ = future.result(timeout=60)
        pool.end_run()
        assert len(live) <= cap, f"{len(live)} workers live with a cap of {cap}: {sorted(live)}"
    finally:
        pool.hard_cancel()
        reload_thread.shutdown(wait=True)


def test_run_after_an_idle_worker_died_between_runs_still_runs() -> None:
    """An OS kill of an idle warm worker between watch runs (the OOM killer, a stray SIGKILL)
    leaves loky's executor broken. The next run's ``start()`` must hand it a working executor,
    as loky's own reuse test (``_flags.broken``) does, rather than keep the broken one."""
    pool = WorkerPool()
    try:
        pool.start(max_workers=2)
        assert pool.submit(_helper_pid).result(timeout=60) > 0
        pool.end_run()
        assert pool._executor is not None
        workers = [p.pid for p in pool._executor._processes.values() if p.pid is not None]
        assert workers, "the run left no warm worker to kill"
        os.kill(workers[0], signal.SIGKILL)
        deadline = time.monotonic() + 30
        while pool._executor._flags.broken is None:
            assert time.monotonic() < deadline, "loky never noticed the killed worker"
            time.sleep(0.05)

        for run in range(2):
            pool.start(max_workers=2)
            assert pool.submit(_helper_pid).result(timeout=60) > 0, f"run {run} after the kill"
            pool.end_run()
    finally:
        pool.hard_cancel()


# One more than a narrow executor's call queue (2 * max(cpu_count, 1) + 1), so the wide run
# needs a new executor.
_WIDER_THAN_A_NARROW_QUEUE = 2 * (loky.cpu_count() or 1) + 2
_STARTUP_S = 3.0
_STAGE_S = 4.0

# Loaded by every Python process started while PYTHONPATH names its directory; a loky worker
# sleeps at startup, standing in for a worker that imports a heavy pipeline.
_SLOW_STARTUP_HOOK = f"""\
import sys
import time

_LAUNCHERS = ("loky.backend.popen_loky_posix", "joblib.externals.loky.backend.popen_loky_posix")
if sys.orig_argv[1:2] == ["-m"] and sys.orig_argv[2:3] and sys.orig_argv[2] in _LAUNCHERS:
    time.sleep({_STARTUP_S!r})
"""


def _helper_started_at(seconds: float) -> float:
    started = time.time()
    time.sleep(seconds)
    return started


def _helper_hold_until_released(ready_dir: str, release: str) -> int:
    (pathlib.Path(ready_dir) / str(os.getpid())).write_text("")
    release_path = pathlib.Path(release)
    deadline = time.monotonic() + 600  # a hang guard only; the test always releases
    while not release_path.exists():
        assert time.monotonic() < deadline, "test did not release waiting workers"
        time.sleep(0.01)
    return os.getpid()


def test_wider_run_after_a_narrow_one_starts_its_stages_together(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watch session's wide run after a narrow one needs a new executor (its call queue is too
    small). The replacement must start no worker before the run submits its stages:
    a worker started and waited for there puts its start-up ahead of every other worker's, so
    one stage starts a whole worker start-up before the rest. Every stage must then hold a
    worker at once. The checks are counts, not times, so machine load cannot change them.

    loky sizes a call queue from the CPU count, so one CPU makes the narrow run's queue (3) too
    small for a run of 4. The width is the same on every machine."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    release = tmp_path / "release"
    width = 4
    executor_core._cleanup_worker_pool()
    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        assert pool.submit(_helper_identity, 1).result(timeout=60) == 1
        pool.end_run()
        narrow = pool._executor
        assert narrow is not None
        assert narrow.queue_size < width, "precondition: the narrow run's queue fits the wide run"

        pool.start(max_workers=width)
        executor = pool._executor
        assert executor is not None
        assert executor.queue_size >= width, (
            f"call queue {executor.queue_size} holds back stages of a {width}-wide run"
        )
        early = sorted(executor._processes)
        assert not early, (
            f"{len(early)} worker(s) started before the run's first submission: {early}; "
            + "a stage on one starts a whole worker start-up before the rest"
        )
        futures = [
            pool.submit(_helper_hold_until_released, str(ready_dir), str(release))
            for _ in range(width)
        ]
        deadline = time.monotonic() + 300  # a hang guard only
        while (running := len(list(ready_dir.iterdir()))) < width:
            assert time.monotonic() < deadline, f"only {running} of {width} stages ran at once"
            time.sleep(0.01)
        release.write_text("")
        assert len({future.result(timeout=60) for future in futures}) == width
        pool.end_run()
    finally:
        release.write_text("")
        pool.shutdown()


def test_run_after_a_reload_restart_starts_its_stages_together(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watch reload restarts workers before its re-run. That re-run must start every stage
    together: a restart that warmed fewer workers than the re-run can start at once leaves the
    rest a whole worker start-up behind the first stage."""
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_SLOW_STARTUP_HOOK)
    width = 4
    pool = WorkerPool()
    reload_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        pool.start(max_workers=width)
        assert pool.submit(_helper_identity, 1).result(timeout=60) == 1
        pool.end_run()

        with monkeypatch.context() as patched:
            patched.setenv(
                "PYTHONPATH",
                os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")])),
            )
            reload_thread.submit(lambda: pool.restart(max_workers=width)).result(timeout=120)
            pool.start(max_workers=width)
            futures = [pool.submit(_helper_started_at, _STAGE_S) for _ in range(width)]
            starts = sorted(future.result(timeout=120) for future in futures)
        pool.end_run()
        offsets = [round(start - starts[0], 2) for start in starts]
        late = [offset for offset in offsets if offset > _STAGE_S / 2]
        assert not late, f"{len(late)} of {width} stages started late: offsets {offsets}"
    finally:
        pool.shutdown()
        reload_thread.shutdown(wait=True)


def _helper_narrow_run(pool: WorkerPool) -> None:
    pool.start(max_workers=1, idle_timeout_s=None)
    assert pool.submit(_helper_identity, 1).result(timeout=60) == 1
    pool.end_run()


def _helper_wide_queue_size(pool: WorkerPool) -> int:
    pool.start(max_workers=_WIDER_THAN_A_NARROW_QUEUE, idle_timeout_s=None)
    assert pool._executor is not None
    return pool._executor.queue_size


def test_wider_run_after_a_narrow_one_gets_a_queue_that_fits() -> None:
    """A long-lived pool's narrow run must not leave a later wider run a call queue too small
    for its width, which would hold submitted stages back until a slot frees."""
    pool = WorkerPool()
    try:
        _helper_narrow_run(pool)
        assert _helper_wide_queue_size(pool) >= _WIDER_THAN_A_NARROW_QUEUE
    finally:
        pool.shutdown()


def test_second_pool_after_a_narrow_one_gets_a_queue_that_fits() -> None:
    """A second Engine in one process (two ``pivot.executor.run`` calls) must not inherit the
    narrow width a shut-down pool left in loky's reusable executor."""
    first = WorkerPool()
    _helper_narrow_run(first)
    first.shutdown()
    pool = WorkerPool()
    try:
        assert _helper_wide_queue_size(pool) >= _WIDER_THAN_A_NARROW_QUEUE
    finally:
        pool.shutdown()


def test_second_pool_after_a_wide_run_gets_a_queue_that_fits() -> None:
    """A wide run's last submission lowers loky's ``_max_workers``; a later pool must still get
    a queue that fits its own width."""
    first = WorkerPool()
    first.start(max_workers=_WIDER_THAN_A_NARROW_QUEUE, idle_timeout_s=None)
    assert first.submit(_helper_identity, 1).result(timeout=60) == 1
    first.end_run()
    first.shutdown()
    pool = WorkerPool()
    try:
        assert _helper_wide_queue_size(pool) >= _WIDER_THAN_A_NARROW_QUEUE
    finally:
        pool.shutdown()


def test_reusable_worker_target_uses_requested_width_when_live_queue_is_too_small() -> None:
    executor_core.restart_workers(1, idle_timeout_s=None)
    executor = executor_core.create_executor(1, idle_timeout_s=None)
    width = executor.queue_size + 1
    try:
        assert executor_core._reusable_worker_target(width, kill_workers=False) == width
    finally:
        executor_core._cleanup_worker_pool()


def test_reusable_worker_target_uses_requested_width_when_executor_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Flags:
        shutdown: bool = False
        broken: BaseException | None = RuntimeError("worker died")

    class _Executor:
        _max_workers: int = 1
        queue_size: int = 2
        _flags: _Flags = _Flags()

    executor = _Executor()
    if hasattr(loky.reusable_executor, "_executor"):
        monkeypatch.setattr(loky.reusable_executor, "_executor", executor)
    else:
        storage = loky.reusable_executor._executor_storage
        monkeypatch.setattr(storage, "executor", executor)

    assert executor_core._reusable_worker_target(2, kill_workers=False) == 2


def test_argument_changed_singleton_gets_a_queue_that_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live reusable executor narrowed by its last submission must still get the requested
    width when loky replaces it because only the idle timeout changed.

    loky sizes a call queue from the CPU count. With one CPU, a queue built for one worker holds
    3 calls, too few for a run of 16, so a wrongly sized replacement fails on any machine."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    width = 16
    executor_core.restart_workers(width, idle_timeout_s=None)
    before = executor_core.create_executor(width, idle_timeout_s=None)
    assert before._max_workers == 1
    pool = WorkerPool()
    try:
        pool.start(max_workers=width, idle_timeout_s=10.0)
        assert pool._executor is not None
        assert pool._executor is not before
        assert pool._executor.queue_size >= width
    finally:
        pool.shutdown()


@pytest.mark.slow
def test_warm_workers_survive_lokys_default_idle_timeout() -> None:
    """Long-lived workers stay warm after a pause longer than loky's default timeout."""
    executor_core.restart_workers(1, idle_timeout_s=None)
    executor = executor_core.create_executor(1)
    first = executor.submit(_helper_pid).result(timeout=60)

    time.sleep(11)

    second = executor.submit(_helper_pid).result(timeout=60)
    assert second == first, "the idle worker should remain ready until Engine shutdown"
    executor_core._cleanup_worker_pool()


def test_worker_pool_stop_accepting_rejects_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    future = pool.submit(_helper_identity, 1)
    # The first result includes spawning a fresh worker process (several seconds under load).
    assert future.result(timeout=60) == 1

    pool.stop_accepting()
    with pytest.raises(RuntimeError, match="accepting"):
        pool.submit(_helper_identity, 2)

    pool.shutdown()


def test_worker_pool_hard_cancel_blocks_new_submissions() -> None:
    pool = WorkerPool()
    pool.start(max_workers=1)

    pool.submit(_helper_sleep, 5)
    pool.hard_cancel()

    with pytest.raises(RuntimeError, match="accepting"):
        pool.submit(_helper_identity, 3)


def test_worker_pool_submit_requires_start() -> None:
    pool = WorkerPool()

    with pytest.raises(RuntimeError, match="not started"):
        pool.submit(_helper_identity, 1)


def test_worker_pool_output_queue_requires_start() -> None:
    pool = WorkerPool()

    with pytest.raises(RuntimeError, match="not started"):
        pool.output_queue()


def test_file_stamp_propagates_descriptor_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running out of descriptors must not read as an unreadable (changed) user-code file."""

    def _no_descriptors(*_args: object, **_kwargs: object) -> os.stat_result:
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(os, "stat", _no_descriptors)
    with pytest.raises(OSError, match="Too many open files"):
        worker_pool._file_stamp(__file__)


def test_failed_manager_replacement_does_not_poison_later_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose new manager fails to start leaves the pool able to start the next run."""
    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        pool.end_run()
        manager = pool._manager
        assert manager is not None
        process = cast("BaseProcess", manager._process)  # pyright: ignore[reportAttributeAccessIssue] - not in typeshed
        assert process.pid is not None
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=10)

        real_create = executor_core.create_manager
        calls = list[int]()

        def _fail_once() -> object:
            calls.append(1)
            if len(calls) == 1:
                raise OSError(errno.EAGAIN, "Resource temporarily unavailable")
            return real_create()

        monkeypatch.setattr(executor_core, "create_manager", _fail_once)
        with pytest.raises(OSError, match="Resource temporarily unavailable"):
            pool.start(max_workers=1)

        pool.start(max_workers=1)
        pool.output_queue().put_nowait(None)
        assert pool.output_queue().get(timeout=10) is None
    finally:
        pool.shutdown()


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EPERM])
def test_stop_workers_bounds_the_stop_when_one_send_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: int
) -> None:
    """A send that raises still leaves every other worker signalled and the stop bounded: the
    grace, the join and the SIGKILL below it run, and the failure names the worker. Without
    that, one failing send leaves the pool's workers alive and `pivot repro` never exits."""
    joined = list[int]()
    killed = list[int]()

    class _Process:
        pid: int
        exitcode: int | None

        def __init__(self, pid: int, exitcode: int | None) -> None:
            self.pid = pid
            self.exitcode = exitcode

        def join(self, _timeout: float | None = None) -> None:
            joined.append(self.pid)

    class _Executor:
        _processes_management_lock: None = None
        _processes: dict[int, _Process] = {0: _Process(101, None), 1: _Process(102, 0)}

    signalled = list[int]()

    def _terminate(pid: int) -> None:
        signalled.append(pid)
        if pid == 101:
            raise OSError(error, os.strerror(error))

    def _kill_descendants(pid: int) -> None:
        killed.append(pid)

    def _kill(pid: int, _signal: int) -> None:
        killed.append(pid)

    monkeypatch.setattr(worker, "terminate", _terminate)
    monkeypatch.setattr(worker, "kill_descendants", _kill_descendants)
    monkeypatch.setattr(os, "kill", _kill)

    with caplog.at_level(logging.ERROR, logger=worker_pool._logger.name):
        WorkerPool._stop_workers(cast("loky.ProcessPoolExecutor", cast("object", _Executor())))

    assert signalled == [101, 102]
    assert joined[:2] == [101, 102]
    # 101 never exited, so it and its descendants are killed; 102 exited within the grace.
    assert killed == [101, 101]
    assert [r.getMessage() for r in caplog.records] == ["Could not send SIGTERM to worker 101"]
