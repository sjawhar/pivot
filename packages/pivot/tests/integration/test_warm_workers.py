"""Worker processes across the runs of one long-lived Engine (watch mode, ``--serve``, the TUI).

Each test writes a project, discovers its pipeline the way the CLI does, and drives one Engine
through several runs, reading what its stages wrote: the worker's process id, or the value a
helper module returned.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

import anyio
import pytest

from pivot import discovery, project
from pivot.engine import engine as engine_mod
from pivot.engine import graph as engine_graph
from pivot.engine import sources
from pivot.engine.types import EngineState, OutputEvent, RunRequested
from pivot.engine.worker_pool import WorkerPool
from pivot.executor import core as executor_core
from pivot.storage import store as store_mod
from pivot.types import OnError, StageStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.process import BaseProcess

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    from pivot.engine.types import InputEvent
    from pivot.registry import PipelineLike


class _Recorder:
    """Sink that records every output event; tests wait on predicates over what it saw."""

    events: list[OutputEvent]
    _changed: anyio.Event

    def __init__(self) -> None:
        self.events = list[OutputEvent]()
        self._changed = anyio.Event()

    async def handle(self, event: OutputEvent) -> None:
        self.events.append(event)
        self._changed.set()
        self._changed = anyio.Event()

    async def close(self) -> None:
        pass

    async def wait_for(
        self, predicate: Callable[[OutputEvent], bool], *, start: int = 0, timeout: float = 60.0
    ) -> int:
        """Return the index of the first event at or after ``start`` matching ``predicate``."""
        with anyio.fail_after(timeout):
            while True:
                for index in range(start, len(self.events)):
                    if predicate(self.events[index]):
                        return index
                await self._changed.wait()


class _RunRequests:
    """Event source the test drives: each forced run request reaches the engine as it would
    from an RPC client."""

    _send: MemoryObjectSendStream[RunRequested]
    _recv: MemoryObjectReceiveStream[RunRequested]

    def __init__(self) -> None:
        self._send, self._recv = anyio.create_memory_object_stream[RunRequested](4)

    async def run(self, send: MemoryObjectSendStream[InputEvent]) -> None:
        async for event in self._recv:
            await send.send(event)

    async def request_forced_run(
        self,
        *,
        stages: list[str] | None = None,
        single_stage: bool = False,
        max_workers: int = 1,
    ) -> None:
        await self._send.send(
            RunRequested(
                type="run_requested",
                stages=stages,
                force=True,
                reason="test",
                single_stage=single_stage,
                parallel=True,
                max_workers=max_workers,
                resource_capacity={},
                no_commit=False,
                on_error=OnError.FAIL,
                cache_dir=None,
                allow_uncached_incremental=False,
                checkout_missing=False,
            )
        )


def _helper_engine_state(state: EngineState) -> Callable[[OutputEvent], bool]:
    def _predicate(event: OutputEvent) -> bool:
        return event["type"] == "engine_state_changed" and event["state"] == state

    return _predicate


async def _helper_forced_run(requests: _RunRequests, recorder: _Recorder, stage: str) -> None:
    """Request one forced run, wait until the engine is idle again, and check ``stage`` ran."""
    start = len(recorder.events)
    await requests.request_forced_run()
    active = await recorder.wait_for(_helper_engine_state(EngineState.ACTIVE), start=start)
    idle = await recorder.wait_for(_helper_engine_state(EngineState.IDLE), start=active)
    completions = [
        event
        for event in recorder.events[active:idle]
        if event["type"] == "stage_completed" and event["stage"] == stage
    ]
    assert len(completions) == 1, f"{stage} should complete once per run"
    completion = completions[0]
    assert completion["type"] == "stage_completed"
    assert completion["status"] == StageStatus.RAN, completion["reason"]


def _helper_project(
    root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, pipeline_source: str
) -> PipelineLike:
    """Write a project whose pipeline.py is ``pipeline_source`` and discover it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".pivot").mkdir()
    (root / ".git").mkdir()
    (root / "pipeline.py").write_text(pipeline_source)
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    pipeline = discovery.discover_pipeline(root)
    assert pipeline is not None, "pipeline.py should define a pipeline"
    return pipeline


def _helper_output_path(root: pathlib.Path, pipeline: PipelineLike, stage: str) -> pathlib.Path:
    store = store_mod.WorkspaceStore(
        project_root=root,
        pipeline_name=pipeline.name,
        input_bindings=pipeline.input_bindings,
    )
    return store.resolve_display_path(pipeline.get_stage(stage)["outs"][0])


_PID_PIPELINE = """\
import logging
import os

from pivot.compose import Pipeline, stage


@stage
def worker_pid() -> str:
    logging.warning("warm worker log")
    return str(os.getpid())


with Pipeline("warm") as pipeline:
    worker_pid()
"""


def _helper_pid_pipeline(version: str) -> str:
    """Return a pipeline whose output identifies the worker that executed its code."""
    return f"""\
import os

from pivot.compose import Pipeline, stage


@stage
def worker_pid() -> str:
    return "{version}:" + str(os.getpid())


with Pipeline("warm") as pipeline:
    worker_pid()
"""


_THREE_PID_PIPELINE = """\
import os

from pivot.compose import Pipeline, stage


@stage
def a() -> str:
    return str(os.getpid())


@stage
def b() -> str:
    return str(os.getpid())


@stage
def c() -> str:
    return str(os.getpid())


with Pipeline("warm") as pipeline:
    a()
    b()
    c()
"""


def _helper_call_user_function(func: Callable[[], str]) -> tuple[int, str, list[str]]:
    """Call a user function after loky deserializes it, then report resident imports."""
    from pivot.executor import worker

    return os.getpid(), func(), worker.user_code_files()


def _helper_import_user_module(module_name: str, marker: str) -> tuple[int, str, list[str]]:
    """Import a module in the worker, giving the coordinator time to edit or delete it."""
    from pivot.executor import worker

    module = importlib.import_module(module_name)
    pathlib.Path(marker).write_text("imported")
    time.sleep(0.5)
    return os.getpid(), module.value(), worker.user_code_files()


def _helper_wait_for_path(path: pathlib.Path) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} was not created"
        time.sleep(0.01)


def _helper_pid() -> int:
    return os.getpid()


def _helper_slow_pid() -> int:
    time.sleep(0.2)
    return os.getpid()


@pytest.mark.anyio
async def test_consecutive_runs_in_one_engine_reuse_the_same_worker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run in the same Engine executes in the worker the first run started."""
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(root, monkeypatch, _PID_PIPELINE)
    output = _helper_output_path(root, pipeline, "warm/worker_pid")
    recorder = _Recorder()
    requests = _RunRequests()
    first_pid = ""
    second_pid = ""

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(requests)
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            await _helper_forced_run(requests, recorder, "warm/worker_pid")
            first_pid = output.read_text()
            await _helper_forced_run(requests, recorder, "warm/worker_pid")
            second_pid = output.read_text()
            tg.cancel_scope.cancel()
    with pytest.raises(ProcessLookupError):
        os.kill(int(first_pid), 0)
    assert first_pid != str(os.getpid()), "the stage runs in a worker process"
    assert second_pid == first_pid, "the second run reuses the first run's worker"
    log_lines = [
        event
        for event in recorder.events
        if event["type"] == "log_line" and event["line"] == "WARNING: warm worker log"
    ]
    assert len(log_lines) == 2, "each run's fresh output queue forwards worker logs"


def test_smaller_run_does_not_shrink_warm_worker_pool() -> None:
    """A one-stage run retains the N warmed workers a later full run needs."""
    executor_core.restart_workers(1, idle_timeout_s=None)
    pool = WorkerPool()
    try:
        pool.start(max_workers=3)
        for future in [pool.submit(_helper_slow_pid) for _ in range(3)]:
            future.result(timeout=60)
        assert pool._executor is not None
        executor = cast("Any", pool._executor)
        process_pids = set(executor._processes)
        assert len(process_pids) == 3

        pool.end_run()
        pool.start(max_workers=1)
        assert pool.submit(_helper_slow_pid).result(timeout=60) in process_pids
        assert set(executor._processes) == process_pids

        pool.end_run()
        pool.start(max_workers=3)
        assert set(executor._processes) == process_pids
    finally:
        pool.shutdown()


def test_deserialization_import_change_restarts_before_next_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit during loky's callable deserialization cannot become a trusted first stamp."""
    module_name = f"warm_deserialize_{uuid.uuid4().hex}"
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    marker = helpers / "imported"
    helper = helpers / f"{module_name}.py"

    def write_source(value: str) -> None:
        helper.write_text(
            "import pathlib\n"
            + "import time\n"
            + f"pathlib.Path({str(marker)!r}).write_text('imported')\n"
            + "time.sleep(0.5)\n\n"
            + "def value() -> str:\n"
            + f"    return {value!r}\n"
        )

    write_source("alpha")
    monkeypatch.syspath_prepend(str(helpers))
    module = importlib.import_module(module_name)
    marker.unlink()
    executor_core.restart_workers(1, idle_timeout_s=None)
    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        first_future = pool.submit(_helper_call_user_function, module.value)
        _helper_wait_for_path(marker)
        write_source("bravo-two")
        first_pid, first_value, files = first_future.result(timeout=60)
        pool.record_user_code_files(files)
        assert first_value == "alpha"

        pool.end_run()
        pool.start(max_workers=1)
        second_pid, second_value, _ = pool.submit(_helper_call_user_function, module.value).result(
            timeout=60
        )
        assert second_pid != first_pid
        assert second_value == "bravo-two"
    finally:
        pool.shutdown()
        sys.modules.pop(module_name, None)


def test_deleted_worker_module_restarts_instead_of_reusing_cached_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstatable path is stale, so a fresh worker fails the deleted import."""
    module_name = f"warm_deleted_{uuid.uuid4().hex}"
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    marker = helpers / "imported"
    helper = helpers / f"{module_name}.py"
    helper.write_text('def value() -> str:\n    return "alpha"\n')
    monkeypatch.syspath_prepend(str(helpers))
    executor_core.restart_workers(1, idle_timeout_s=None)
    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        first_future = pool.submit(_helper_import_user_module, module_name, str(marker))
        _helper_wait_for_path(marker)
        helper.unlink()
        first_pid, first_value, files = first_future.result(timeout=60)
        pool.record_user_code_files(files)
        assert first_value == "alpha"

        pool.end_run()
        pool.start(max_workers=1)
        replacement_pid = pool.submit(_helper_pid).result(timeout=60)
        assert replacement_pid != first_pid
        with pytest.raises(ModuleNotFoundError):
            pool.submit(_helper_import_user_module, module_name, str(marker)).result(timeout=60)
    finally:
        pool.shutdown()
        sys.modules.pop(module_name, None)


def test_edit_that_keeps_file_timestamps_restarts_before_next_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit whose timestamps do not move, as on a coarse or lagging filesystem clock, still
    replaces the worker that imported the previous code."""
    module_name = f"warm_frozen_{uuid.uuid4().hex}"
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    helper = helpers / f"{module_name}.py"
    helper.write_text('def value() -> str:\n    return "alpha"\n')
    monkeypatch.syspath_prepend(str(helpers))
    module = importlib.import_module(module_name)
    executor_core.restart_workers(1, idle_timeout_s=None)
    target = str(helper.resolve())
    frozen = os.stat(target)
    real_stat = os.stat

    def stat_with_frozen_timestamps(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)
        return frozen if os.fspath(path) == target else result

    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        first_pid, first_value, files = pool.submit(
            _helper_call_user_function, module.value
        ).result(timeout=60)
        pool.record_user_code_files(files)
        assert first_value == "alpha"

        monkeypatch.setattr(os, "stat", stat_with_frozen_timestamps)
        helper.write_text('def value() -> str:\n    return "bravo"\n')
        assert os.stat(target) == frozen, "the edit must be invisible to stat"

        pool.end_run()
        pool.start(max_workers=1)
        second_pid, second_value, _ = pool.submit(_helper_call_user_function, module.value).result(
            timeout=60
        )
        assert second_pid != first_pid
        assert second_value == "bravo"
    finally:
        pool.shutdown()
        sys.modules.pop(module_name, None)


def _helper_process_table() -> dict[int, tuple[int, str]]:
    """Map every live, non-zombie pid to (parent pid, start time) from /proc."""
    table = dict[int, tuple[int, str]]()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        if fields[0] == "Z":
            continue
        table[int(entry.name)] = (int(fields[1]), fields[19])
    return table


def _helper_descendants(root_pid: int) -> dict[int, str]:
    """Return {pid: start time} for every live descendant of ``root_pid``."""
    table = _helper_process_table()
    found = dict[int, str]()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for pid, (ppid, started) in table.items():
            if ppid == parent and pid not in found:
                found[pid] = started
                frontier.append(pid)
    return found


def _helper_cmdline(pid: int) -> str:
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "<exited>"
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


@pytest.mark.parametrize("failure", ["killed", "socket_removed"])
def test_unreachable_manager_is_replaced_between_runs(failure: str) -> None:
    """A manager that died, or whose socket a temp cleaner removed, is replaced at the next run
    instead of ending the session."""
    executor_core.restart_workers(1, idle_timeout_s=None)
    pool = WorkerPool()
    try:
        pool.start(max_workers=1)
        pool.end_run()
        manager = pool._manager
        assert manager is not None
        if failure == "killed":
            process = cast("BaseProcess", manager._process)  # pyright: ignore[reportAttributeAccessIssue] - not in typeshed
            assert process.pid is not None
            os.kill(process.pid, signal.SIGKILL)
            process.join(timeout=10)
        else:
            os.unlink(cast("str", manager.address))

        pool.start(max_workers=1)
        pool.output_queue().put_nowait(None)
        assert pool.output_queue().get(timeout=10) is None
    finally:
        pool.shutdown()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the /proc process table")
def test_killed_pivot_leaves_no_child_processes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGKILLing a serving pivot leaves no worker, manager or tracker running without it."""
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(root, monkeypatch, _PID_PIPELINE)
    output = _helper_output_path(root, pipeline, "warm/worker_pid")
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    proc = subprocess.Popen(
        [str(pathlib.Path(sys.executable).parent / "pivot"), "repro", "--watch", "--serve"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    children = dict[int, str]()
    survivors = dict[int, str]()
    try:
        deadline = time.monotonic() + 60
        while not output.exists():
            assert proc.poll() is None, f"pivot exited early with {proc.returncode}"
            assert time.monotonic() < deadline, "the first watch run did not finish"
            time.sleep(0.1)
        children = _helper_descendants(proc.pid)
        assert any("loky" in _helper_cmdline(pid) for pid in children), (
            "the serving pivot should hold a warm worker when it is killed"
        )

        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)

        deadline = time.monotonic() + 15
        while True:
            table = _helper_process_table()
            survivors = {
                pid: started
                for pid, started in children.items()
                if pid in table and table[pid][1] == started
            }
            if not survivors or time.monotonic() > deadline:
                break
            time.sleep(0.25)
        assert not survivors, "child processes outlived pivot:\n" + "\n".join(
            f"{pid}: {_helper_cmdline(pid)}" for pid in survivors
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for pid in survivors:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.anyio
async def test_watch_reload_restarts_workers_after_user_code_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watcher reloads code in a new worker rather than a warm worker's old module."""
    root = tmp_path.resolve() / "project"
    pipeline_path = root / "pipeline.py"
    pipeline = _helper_project(root, monkeypatch, _helper_pid_pipeline("v1"))
    output = _helper_output_path(root, pipeline, "warm/worker_pid")
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[root], debounce_ms=50))
        eng.add_source(sources.OneShotSource(stages=None, force=True, reason="test", max_workers=1))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_engine_state(EngineState.IDLE))
            first_version, first_pid = output.read_text().split(":")
            assert first_version == "v1"

            pipeline_path.write_text(_helper_pid_pipeline("version-two"))
            reload = await recorder.wait_for(
                lambda event: event["type"] == "pipeline_reloaded" and event["error"] is None,
                start=first_idle,
            )
            await recorder.wait_for(_helper_engine_state(EngineState.IDLE), start=reload)
            second_version, second_pid = output.read_text().split(":")
            assert second_version == "version-two"
            assert second_pid != first_pid, "a code reload must replace live workers"
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_forced_run_restarts_workers_after_out_of_root_helper_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edited PYTHONPATH helper runs its new code, rather than a worker's cached module."""
    module_name = f"warm_external_{uuid.uuid4().hex}"
    helpers = tmp_path.resolve() / "helpers"
    helpers.mkdir()
    helper = helpers / f"{module_name}.py"
    helper.write_text('def value() -> str:\n    return "alpha"\n')
    monkeypatch.syspath_prepend(str(helpers))
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(
        root,
        monkeypatch,
        f"""\
import {module_name}

from pivot.compose import Pipeline, stage


@stage
def value() -> str:
    return {module_name}.value()


with Pipeline("warm") as pipeline:
    value()
""",
    )
    executor_core.restart_workers(1, idle_timeout_s=None)
    output = _helper_output_path(root, pipeline, "warm/value")
    recorder = _Recorder()
    requests = _RunRequests()

    try:
        async with engine_mod.Engine(pipeline=pipeline) as eng:
            eng.add_sink(recorder)
            eng.add_source(requests)
            async with anyio.create_task_group() as tg:
                tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
                await _helper_forced_run(requests, recorder, "warm/value")
                assert output.read_text() == "alpha"

                helper.write_text('def value() -> str:\n    return "bravo-two"\n')
                await _helper_forced_run(requests, recorder, "warm/value")
                assert output.read_text() == "bravo-two"
                tg.cancel_scope.cancel()
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.anyio
async def test_forced_run_restarts_workers_after_in_function_helper_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A helper first imported inside a stage is reported by the worker and invalidates it."""
    module_name = f"warm_in_function_{uuid.uuid4().hex}"
    root = tmp_path.resolve() / "project"
    root.mkdir()
    helper = root / f"{module_name}.py"
    helper.write_text('def value() -> str:\n    return "alpha"\n')
    pipeline = _helper_project(
        root,
        monkeypatch,
        f"""\
from pivot.compose import Pipeline, stage


@stage
def value() -> str:
    import {module_name}
    return {module_name}.value()


with Pipeline("warm") as pipeline:
    value()
""",
    )
    executor_core.restart_workers(1, idle_timeout_s=None)
    output = _helper_output_path(root, pipeline, "warm/value")
    recorder = _Recorder()
    requests = _RunRequests()

    try:
        async with engine_mod.Engine(pipeline=pipeline) as eng:
            eng.add_sink(recorder)
            eng.add_source(requests)
            async with anyio.create_task_group() as tg:
                tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
                await _helper_forced_run(requests, recorder, "warm/value")
                assert output.read_text() == "alpha"

                helper.write_text('def value() -> str:\n    return "bravo-two"\n')
                await _helper_forced_run(requests, recorder, "warm/value")
                assert output.read_text() == "bravo-two"
                tg.cancel_scope.cancel()
    finally:
        sys.modules.pop(module_name, None)


_CHAIN_PIPELINE = """\
import os

from pivot.compose import Pipeline, stage


@stage
def first() -> str:
    return str(os.getpid())


@stage
def step(previous: str) -> str:
    return str(os.getpid())


with Pipeline("chain") as pipeline:
    pid = first()
    for _ in range(11):
        pid = step(pid)
"""

# Levels {lone, source} and {left, right}: the widest level is 2, but lone, left and right can
# all run at once.
_WIDTH_PIPELINE = """\
from pivot.compose import Pipeline, stage


@stage
def lone() -> str:
    return "lone"


@stage
def source() -> str:
    return "source"


@stage
def left(value: str) -> str:
    return value + " left"


@stage
def right(value: str) -> str:
    return value + " right"


with Pipeline("width") as pipeline:
    lone()
    shared = source()
    left(shared)
    right(shared)
"""


_RESOURCE_LIMITED_PIPELINE = """\
import time

from pivot.compose import Pipeline, stage


@stage
def source() -> str:
    return "source"


@stage(resources={"local_grade": 1})
def grade(value: str) -> str:
    time.sleep(0.5)
    return value


with Pipeline("resources") as pipeline:
    shared = source()
    for index in range(4):
        with pipeline.variant(str(index)):
            grade(shared)
"""

# The parent sleeps longer than the one-shot idle timeout; the fan of four becomes ready only
# when it finishes.
_LONG_PARENT_FAN_PIPELINE = """\
import time

from pivot.compose import Pipeline, stage


@stage
def source() -> str:
    time.sleep({parent_s!r})
    return "source"


@stage
def fan(value: str) -> str:
    return value


with Pipeline("fan") as pipeline:
    shared = source()
    for index in range(4):
        with pipeline.variant(str(index)):
            fan(shared)
"""


# The parent fails while an independent stage outlives the idle timeout and then counts the
# live workers; workers reserved for the failed parent's fan must be gone by then. A worker
# that loses loky's process-lock race when its timeout fires waits one more timeout (loky
# `_process_worker`), so the bystander allows up to two.
_FAILED_PARENT_FAN_PIPELINE = """\
import os
import pathlib
import time

import psutil

from pivot.compose import Pipeline, stage


@stage
def source() -> str:
    time.sleep({parent_s!r})
    raise RuntimeError("parent fails")


@stage
def fan(value: str) -> str:
    return value


def _live_workers() -> int:
    count = 0
    for child in psutil.Process(os.getppid()).children():
        try:
            if "loky.backend.popen_loky_posix" in " ".join(child.cmdline()):
                count += 1
        except psutil.NoSuchProcess:  # exited (or a zombie) between listing and inspecting
            continue
    return count


@stage
def bystander() -> str:
    time.sleep({wait_s!r})
    deadline = time.monotonic() + {timeout_s!r}
    live = _live_workers()
    while live != 1 and time.monotonic() < deadline:
        time.sleep(0.1)
        live = _live_workers()
    pathlib.Path({live_path!r}).write_text(str(live))
    return "bystander"


with Pipeline("fan") as pipeline:
    shared = source()
    for index in range(4):
        with pipeline.variant(str(index)):
            fan(shared)
    bystander()
"""


# Imported by every Python process the command starts: each loky worker records when it started.
# With PIVOT_TEST_CHECKS_AFTER_TUI set, the pivot process's engine checks the tracked files only
# once the TUI has mounted, the order a busy machine sometimes gives.
_WORKER_LOG_HOOK = """\
import os
import sys
import time


def _log(event):
    with open(os.environ["PIVOT_TEST_WORKER_LOG"], "a") as log:
        log.write(f"{event} {time.time()}\\n")


if sys.orig_argv[1:3] == ["-m", "loky.backend.popen_loky_posix"]:
    _log("worker")
elif (
    len(sys.orig_argv) > 1
    and os.path.basename(sys.orig_argv[1]) == "pivot"
    and os.environ.get("PIVOT_TEST_CHECKS_AFTER_TUI")
):
    import threading

    import anyio
    import pivot_tui.run
    from pivot.engine import engine

    _mounted = threading.Event()
    _on_mount = pivot_tui.run.PivotApp.on_mount
    _orchestrate_execution = engine.Engine._orchestrate_execution

    async def _marked_on_mount(self):
        _mounted.set()
        await _on_mount(self)

    async def _orchestrate_execution_after_tui(self, *args, **kwargs):
        # The run checks the tracked files first, so it checks them after the TUI started.
        with anyio.fail_after(60):
            while not _mounted.is_set():
                await anyio.sleep(0.01)
        _log("held")
        return await _orchestrate_execution(self, *args, **kwargs)

    pivot_tui.run.PivotApp.on_mount = _marked_on_mount
    engine.Engine._orchestrate_execution = _orchestrate_execution_after_tui
"""


def _helper_logging_env(tmp_path: pathlib.Path) -> tuple[dict[str, str], pathlib.Path]:
    """Environment that runs the worker-log hook in every process the command starts."""
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    env["TERM"] = "xterm-256color"
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(_WORKER_LOG_HOOK)
    worker_log = tmp_path / "workers.log"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(hook), env.get("PYTHONPATH")]))
    env["PIVOT_TEST_WORKER_LOG"] = str(worker_log)
    return env, worker_log


def _helper_started_workers(worker_log: pathlib.Path) -> int:
    """Count workers that the command started."""
    events = [line.split() for line in worker_log.read_text().splitlines()]
    return sum(event == "worker" for event, _at in events)


def _helper_pivot(root: pathlib.Path, env: dict[str, str], *args: str) -> str:
    """Run the installed ``pivot`` command in ``root``, require it to succeed, return its output."""
    proc = subprocess.run(
        [str(pathlib.Path(sys.executable).parent / "pivot"), *args],
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    # Python reports a failing sitecustomize and goes on, which would leave a test's hook unset.
    assert "Error in sitecustomize" not in output, output
    return output


def _helper_configure_workers(root: pathlib.Path, max_workers: int) -> None:
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  max_workers: {max_workers}\n")


def _helper_assert_warning_outside_tui(output: str) -> None:
    """Require one tracked-file warning, printed outside the TUI's alternate screen."""
    assert output.count("has changed since tracking") == 1, output
    warning = output.find("has changed since tracking")
    enter, leave = output.find("\x1b[?1049h"), output.rfind("\x1b[?1049l")
    assert -1 < enter < leave, f"the TUI never took over the terminal:\n{output}"
    assert warning < enter or warning > leave, f"the warning landed on the TUI's screen:\n{output}"


def _helper_wait_for_running_pivot(
    proc: subprocess.Popen[bytes], log: pathlib.Path, done: Callable[[], bool], what: str
) -> None:
    """Wait up to 60 seconds for ``done`` while the pivot process keeps running."""
    deadline = time.monotonic() + 60
    while not done():
        assert proc.poll() is None, f"pivot exited before {what}:\n{log.read_text()}"
        assert time.monotonic() < deadline, f"no {what} after 60 s:\n{log.read_text()}"
        time.sleep(0.1)


def test_repro_runs_a_chain_of_stages_on_one_worker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chain runs one stage at a time, so `pivot repro` runs all twelve stages in one warm
    worker even when the configuration allows twelve."""
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(root, monkeypatch, _CHAIN_PIPELINE)
    _helper_configure_workers(root, 12)
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    _helper_pivot(root, env, "repro")

    stages = pipeline.list_stages()
    assert len(stages) == 12
    pids = {_helper_output_path(root, pipeline, stage).read_text() for stage in stages}
    assert len(pids) == 1, f"the chain ran on {len(pids)} workers: {sorted(pids)}"


def test_resource_capacity_bounds_workers_started(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve() / "project"
    _helper_project(root, monkeypatch, _RESOURCE_LIMITED_PIPELINE)
    _helper_configure_workers(root, 8)
    config_path = root / ".pivot" / "config.yaml"
    config_path.write_text(config_path.read_text() + "resources:\n  local_grade: 2\n")
    env, worker_log = _helper_logging_env(tmp_path)

    _helper_pivot(root, env, "repro")

    assert _helper_started_workers(worker_log) == 2


@pytest.mark.slow
def test_fan_after_parent_longer_than_idle_timeout_starts_each_worker_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker started for a fan while its parent runs must not exit idle before the fan is
    ready and then be started again: the run starts no more workers than the fan has stages."""
    root = tmp_path.resolve() / "project"
    parent_s = executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S + 5
    _helper_project(root, monkeypatch, _LONG_PARENT_FAN_PIPELINE.format(parent_s=parent_s))
    _helper_configure_workers(root, 8)
    env, worker_log = _helper_logging_env(tmp_path)

    _helper_pivot(root, env, "repro")

    started = _helper_started_workers(worker_log)
    assert started <= 4, (
        f"a four-stage fan after a {parent_s:g} s parent started {started} workers: workers "
        "started while the parent ran exited idle and were started again for the fan"
    )


@pytest.mark.slow
def test_workers_reserved_for_a_failed_parent_exit_within_the_idle_timeout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers pre-started for a fan whose parent fails become ordinary idle workers and exit."""
    root = tmp_path.resolve() / "project"
    live_path = tmp_path / "live-workers"
    parent_s = executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S / 2 + 1
    wait_s = parent_s + executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S + 5
    _helper_project(
        root,
        monkeypatch,
        _FAILED_PARENT_FAN_PIPELINE.format(
            parent_s=parent_s,
            wait_s=wait_s,
            timeout_s=executor_core.ONE_SHOT_WORKER_IDLE_TIMEOUT_S,
            live_path=str(live_path),
        ),
    )
    _helper_configure_workers(root, 8)
    env, _ = _helper_logging_env(tmp_path)

    proc = subprocess.run(
        [str(pathlib.Path(sys.executable).parent / "pivot"), "repro", "--keep-going"],
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert live_path.read_text() == "1", (
        f"{live_path.read_text()} workers were alive {wait_s:g} s after the parent failed; "
        "only the one running the bystander should be"
    )


@pytest.mark.parametrize(
    ("max_workers", "command", "worker_limit"),
    [
        pytest.param(8, ["repro"], 3, id="repro"),
        pytest.param(8, ["repro", "--tui"], 3, id="repro-tui"),
        pytest.param(2, ["repro"], 2, id="repro-configured-cap"),
        # `pivot run` orders left and right after source, so no more than two run at once.
        pytest.param(8, ["run", "--force", "source", "left", "right"], 2, id="run"),
        pytest.param(8, ["run", "--tui", "--force", "source", "left", "right"], 2, id="run-tui"),
    ],
)
def test_command_starts_no_more_workers_than_stages_that_can_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    max_workers: int,
    command: list[str],
    worker_limit: int,
) -> None:
    """A command starts no workers beyond the stages its scheduler admits."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    _helper_project(root, monkeypatch, _WIDTH_PIPELINE)
    _helper_configure_workers(root, max_workers)
    env, worker_log = _helper_logging_env(tmp_path)

    _helper_pivot(root, env, *command)

    started = _helper_started_workers(worker_log)
    assert started <= worker_limit, f"{' '.join(command)} started {started} workers"


# `lone` reads a tracked input, so the run can restore it from the cache.
_TRACKED_WIDTH_PIPELINE = _WIDTH_PIPELINE.replace(
    'def lone() -> str:\n    return "lone"', "def lone(raw: str) -> str:\n    return raw"
).replace("    lone()\n", '    lone(pipeline.input("x.txt", t=str))\n')


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["repro", "--tui", "--checkout-missing"], id="repro"),
        pytest.param(
            ["run", "--tui", "--checkout-missing", "--force", "lone", "source", "left", "right"],
            id="run",
        ),
    ],
)
def test_tui_restores_tracked_input_when_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """The engine restores a missing tracked input before the stage that consumes it runs."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    _helper_configure_workers(root, 8)
    env, _ = _helper_logging_env(tmp_path)
    tracked = root / "data" / "raw" / "x.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked")
    _helper_pivot(root, env, "track", "data/raw/x.txt")
    tracked.unlink()

    _helper_pivot(root, env, *command)

    assert (root / "data" / "width" / "lone.txt").read_text() == "tracked"


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(["repro", "--tui", "--checkout-missing"], id="repro-checkout-missing"),
        pytest.param(["repro", "--tui"], id="repro"),
        pytest.param(["run", "--tui", "--force", "lone"], id="run"),
    ],
)
@pytest.mark.parametrize(
    "order",
    [
        pytest.param({}, id="as-scheduled"),
        pytest.param({"PIVOT_TEST_CHECKS_AFTER_TUI": "1"}, id="checks-after-tui"),
    ],
)
def test_tui_warns_once_about_a_changed_tracked_file_whenever_the_engine_checks_it(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    order: dict[str, str],
) -> None:
    """A tracked file edited since `pivot track` is reported once, outside the TUI's screen,
    whether the engine checks it before the TUI starts or while the TUI runs."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    env, worker_log = _helper_logging_env(tmp_path)
    tracked = root / "data" / "raw" / "x.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked")
    _helper_pivot(root, env, "track", "data/raw/x.txt")
    tracked.write_text("edited")

    output = _helper_pivot(root, {**env, **order}, *command)

    held = [line for line in worker_log.read_text().splitlines() if line.startswith("held ")]
    assert len(held) == len(order), f"the order hook held {len(held)} runs"
    _helper_assert_warning_outside_tui(output)


# Appended to _WORKER_LOG_HOOK. With PIVOT_TEST_CHECK_LOG set, the pivot process records each
# tracked-file check there. With PIVOT_TEST_CHECKS_BEFORE_TUI set as well, the main thread starts
# the TUI only once the engine's first check has returned: the order a busy machine gives when
# the engine thread wins the race to the check.
_CHECK_LOG_HOOK = """
if (
    len(sys.orig_argv) > 1
    and os.path.basename(sys.orig_argv[1]) == "pivot"
    and os.environ.get("PIVOT_TEST_CHECK_LOG")
):
    import threading

    from pivot.cli import _run_common
    from pivot.executor import core as _core

    _checked = threading.Event()
    _verify = _core.verify_tracked_files

    def _recorded_verify(*args, **kwargs):
        try:
            return _verify(*args, **kwargs)
        finally:
            with open(os.environ["PIVOT_TEST_CHECK_LOG"], "a") as log:
                log.write("check\\n")
            _checked.set()

    _core.verify_tracked_files = _recorded_verify

    if os.environ.get("PIVOT_TEST_CHECKS_BEFORE_TUI"):
        _start = threading.Thread.start

        def _start_poller_after_first_check(self):
            # The main thread starts the event poller just before it starts the TUI.
            if self.name == _run_common.POLLER_THREAD_NAME and not _checked.wait(60):
                raise RuntimeError("the engine never checked the tracked files")
            return _start(self)

        threading.Thread.start = _start_poller_after_first_check
"""


@pytest.mark.parametrize(
    "order",
    [
        pytest.param({}, id="as-scheduled"),
        pytest.param({"PIVOT_TEST_CHECKS_AFTER_TUI": "1"}, id="checks-after-tui"),
        pytest.param({"PIVOT_TEST_CHECKS_BEFORE_TUI": "1"}, id="checks-before-tui"),
    ],
)
def test_watch_tui_warns_once_about_a_changed_tracked_file_across_its_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, order: dict[str, str]
) -> None:
    """`pivot repro --watch --tui` checks the tracked files on every run. A tracked file edited
    since `pivot track`, and edited again while the session runs, is reported once, outside the
    TUI's screen, whether the startup run checks it before the TUI starts or while it runs."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    env, _ = _helper_logging_env(tmp_path)
    hook = tmp_path / "hook" / "sitecustomize.py"
    hook.write_text(hook.read_text() + _CHECK_LOG_HOOK)
    checks = tmp_path / "checks.log"
    env["PIVOT_TEST_CHECK_LOG"] = str(checks)
    tracked = root / "data" / "raw" / "x.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked")
    _helper_pivot(root, env, "track", "data/raw/x.txt")
    tracked.write_text("edited")
    lone = root / "data" / "width" / "lone.txt"
    log = tmp_path / "pivot.log"
    with log.open("wb") as output:
        proc = subprocess.Popen(
            [str(pathlib.Path(sys.executable).parent / "pivot"), "repro", "--watch", "--tui"],
            cwd=root,
            env={**env, **order},
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )

    try:
        _helper_wait_for_running_pivot(
            proc,
            log,
            lambda: lone.exists() and lone.read_text().strip() == "edited",
            "startup run",
        )
        tracked.write_text("edited again")
        _helper_wait_for_running_pivot(
            proc,
            log,
            lambda: checks.exists() and len(checks.read_text().splitlines()) >= 2,
            "re-run's check of the tracked files",
        )
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    output = log.read_text()
    assert proc.returncode == 0, output
    assert "Error in sitecustomize" not in output, output
    _helper_assert_warning_outside_tui(output)


def test_watch_tui_prints_the_pipeline_error_it_shows_at_exit_after_a_second_break(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watch session whose pipeline breaks one way, then another, and is still broken when
    the user quits prints the error the TUI shows, in full, once: `Pipeline invalid: <error>`
    and `Pipeline invalid - waiting for fix`, as plain `pivot repro --watch` prints them."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")
    env["TERM"] = "xterm-256color"
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "x.txt").write_text("tracked")
    pipeline_file = root / "pipeline.py"
    lone = root / "data" / "width" / "lone.txt"
    tui_log = tmp_path / "tui.jsonl"
    log = tmp_path / "pivot.log"
    with log.open("wb") as output:
        proc = subprocess.Popen(
            [
                str(pathlib.Path(sys.executable).parent / "pivot"),
                "repro",
                "--watch",
                "--tui",
                "--tui-log",
                str(tui_log),
            ],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )

    def reload_errors() -> list[str]:
        if not tui_log.exists():
            return []
        messages = [json.loads(line) for line in tui_log.read_text().splitlines() if line]
        return [m["error"] for m in messages if m.get("type") == "reload" and m.get("error")]

    try:
        _helper_wait_for_running_pivot(proc, log, lone.exists, "startup run")
        pipeline_file.write_text(_TRACKED_WIDTH_PIPELINE + "def first_break(:\n")
        _helper_wait_for_running_pivot(
            proc, log, lambda: len(reload_errors()) >= 1, "TUI showing the first break"
        )
        pipeline_file.write_text(_TRACKED_WIDTH_PIPELINE + "\n\n\n\ndef second_break(:\n")
        _helper_wait_for_running_pivot(
            proc,
            log,
            lambda: len(set(reload_errors())) >= 2,  # noqa: PLR2004 - two distinct breaks
            "TUI showing the second break",
        )
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    output = log.read_text()
    assert proc.returncode == 0, output
    shown = reload_errors()[-1]
    # What follows the TUI's exit from the alternate screen, without its terminal resets.
    after_tui = re.sub(r"\x1b\[[0-9;?<]*[A-Za-z]", "", output[output.rfind("\x1b[?1049l") :])
    printed = [line for line in after_tui.splitlines() if "Pipeline invalid" in line]
    assert printed == [f"Pipeline invalid: {shown}", "Pipeline invalid - waiting for fix"], output


# Appended to _WORKER_LOG_HOOK. With PIVOT_TEST_LAG_RELOADS set to "failed" (or "good"), the TUI
# receives each reload result that rejected (or accepted) the edited pipeline, and so records it
# in --tui-log, but shows it only after the test has ended: the TUI's view of the pipeline lags
# the engine's, as it does for a poll interval or longer on a loaded machine.
_RELOAD_LAG_HOOK = """
if (
    len(sys.orig_argv) > 1
    and os.path.basename(sys.orig_argv[1]) == "pivot"
    and os.environ.get("PIVOT_TEST_LAG_RELOADS")
):
    import pivot_tui.run

    _handle_reload = pivot_tui.run.PivotApp._handle_reload
    _lag_failed = os.environ["PIVOT_TEST_LAG_RELOADS"] == "failed"

    def _lagging_handle_reload(self, msg):
        if (msg["error"] is not None) == _lag_failed:
            self.set_timer(600, lambda: _handle_reload(self, msg))
            return
        _handle_reload(self, msg)

    pivot_tui.run.PivotApp._handle_reload = _lagging_handle_reload
"""


@pytest.mark.parametrize(
    ("lagged", "repair"),
    [
        pytest.param("failed", False, id="engine-invalid-tui-not-showing-it-yet"),
        pytest.param("good", True, id="engine-repaired-tui-still-showing-the-error"),
    ],
)
def test_watch_tui_prints_a_pipeline_error_by_the_engines_state_not_the_tuis(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, lagged: str, repair: bool
) -> None:
    """After a watch session the terminal reports the pipeline error the engine still has when
    the user quits, even when the TUI has not yet shown the latest reload: a pipeline still
    broken prints its error, and one already repaired prints nothing stale."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    env, _ = _helper_logging_env(tmp_path)
    hook = tmp_path / "hook" / "sitecustomize.py"
    hook.write_text(hook.read_text() + _RELOAD_LAG_HOOK)
    env["PIVOT_TEST_LAG_RELOADS"] = lagged
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "x.txt").write_text("tracked")
    pipeline_file = root / "pipeline.py"
    lone = root / "data" / "width" / "lone.txt"
    tui_log = tmp_path / "tui.jsonl"
    log = tmp_path / "pivot.log"
    with log.open("wb") as output:
        proc = subprocess.Popen(
            [
                str(pathlib.Path(sys.executable).parent / "pivot"),
                "repro",
                "--watch",
                "--tui",
                "--tui-log",
                str(tui_log),
            ],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )

    def reload_results() -> list[str | None]:
        if not tui_log.exists():
            return []
        messages = [json.loads(line) for line in tui_log.read_text().splitlines() if line]
        return [m.get("error") for m in messages if m.get("type") == "reload"]

    try:
        _helper_wait_for_running_pivot(proc, log, lone.exists, "startup run")
        pipeline_file.write_text(_TRACKED_WIDTH_PIPELINE + "def first_break(:\n")
        _helper_wait_for_running_pivot(
            proc, log, lambda: any(reload_results()), "the engine rejecting the break"
        )
        if repair:
            pipeline_file.write_text(_TRACKED_WIDTH_PIPELINE)
            _helper_wait_for_running_pivot(
                proc, log, lambda: reload_results()[-1] is None, "the engine accepting the repair"
            )
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    output = log.read_text()
    assert proc.returncode == 0, output
    assert "Error in sitecustomize" not in output, output
    # What follows the TUI's exit from the alternate screen, without its terminal resets.
    after_tui = re.sub(r"\x1b\[[0-9;?<]*[A-Za-z]", "", output[output.rfind("\x1b[?1049l") :])
    printed = [line for line in after_tui.splitlines() if "Pipeline invalid" in line]
    if repair:
        assert printed == [], output
    else:
        error = next(result for result in reload_results() if result)
        assert printed == [f"Pipeline invalid: {error}", "Pipeline invalid - waiting for fix"], (
            output
        )


@pytest.mark.anyio
async def test_watch_reload_checks_dependencies_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload of a single pipeline leaves checking for missing dependencies to the run it
    starts, which checks them once."""
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(root, monkeypatch, _THREE_PID_PIPELINE)
    recorder = _Recorder()
    checks = list[int]()
    validate = engine_graph.validate_dependency_sources

    def counting_validate(*args: Any, **kwargs: Any) -> Any:
        checks.append(1)
        return validate(*args, **kwargs)

    monkeypatch.setattr(engine_graph, "validate_dependency_sources", counting_validate)
    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[root], debounce_ms=50))
        eng.add_source(sources.OneShotSource(stages=None, force=True, reason="test", max_workers=8))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_engine_state(EngineState.IDLE))
            checks.clear()
            (root / "pipeline.py").write_text(_CHAIN_OF_THREE_PIPELINE)
            reload = await recorder.wait_for(
                lambda event: event["type"] == "pipeline_reloaded" and event["error"] is None,
                start=first_idle,
            )
            await recorder.wait_for(_helper_engine_state(EngineState.IDLE), start=reload)
            tg.cancel_scope.cancel()

    assert len(checks) == 1, f"a reload and its run checked dependencies {len(checks)} times"


_BROKEN_PIPELINE = """\
from pivot.compose import Pipeline, stage


@stage
def first(raw: str) -> str:
    return raw


@stage
def second(raw: str) -> str:
    return raw


with Pipeline("broken") as pipeline:
    missing = pipeline.input("missing.txt", t=str)
    first(missing)
    second(missing)
"""


def test_repro_tui_runs_stages_available_with_keep_going(
    tmp_path: pathlib.Path,
) -> None:
    """Across all pipelines with keep-going, the run leaves out stages with a missing input."""
    # A short directory name keeps the TUI's .pivot/agent.sock within AF_UNIX's 107-byte limit.
    root = tmp_path.resolve() / "p"
    for name, source in [("width", _WIDTH_PIPELINE), ("broken", _BROKEN_PIPELINE)]:
        (root / name).mkdir(parents=True)
        (root / name / "pipeline.py").write_text(source)
    (root / ".pivot").mkdir()
    (root / ".git").mkdir()
    _helper_configure_workers(root, 8)
    env, worker_log = _helper_logging_env(tmp_path)
    command = ["repro", "--all", "--keep-going", "--tui"]

    _helper_pivot(root, env, *command)

    assert (root / "data" / "width" / "left.txt").read_text() == "source left"
    assert _helper_started_workers(worker_log) <= 3


_CHAIN_OF_THREE_PIPELINE = """\
import os

from pivot.compose import Pipeline, stage


@stage
def a() -> str:
    return str(os.getpid())


@stage
def b(previous: str) -> str:
    return str(os.getpid())


@stage
def c(previous: str) -> str:
    return str(os.getpid())


with Pipeline("warm") as pipeline:
    c(b(a()))
"""


def _helper_pool_worker_pids(pool: WorkerPool | None) -> set[int]:
    """The worker processes of the executor ``pool`` holds."""
    assert pool is not None and pool._executor is not None, "the pool holds no executor"
    return set(pool._executor._processes)


@pytest.mark.anyio
async def test_watch_reload_restarts_one_warm_worker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload replaces one worker before its narrowed run begins."""
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(root, monkeypatch, _THREE_PID_PIPELINE)
    recorder = _Recorder()
    restarted = list[set[int]]()
    after_run = set[int]()
    restart = WorkerPool.restart

    def recording_restart(self: WorkerPool, *, max_workers: int) -> int:
        workers = restart(self, max_workers=max_workers)
        restarted.append(_helper_pool_worker_pids(self))
        return workers

    monkeypatch.setattr(WorkerPool, "restart", recording_restart)
    async with engine_mod.Engine(pipeline=pipeline) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[root], debounce_ms=50))
        eng.add_source(sources.OneShotSource(stages=None, force=True, reason="test", max_workers=8))
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_engine_state(EngineState.IDLE))
            (root / "pipeline.py").write_text(_CHAIN_OF_THREE_PIPELINE)
            reload = await recorder.wait_for(
                lambda event: event["type"] == "pipeline_reloaded" and event["error"] is None,
                start=first_idle,
            )
            await recorder.wait_for(_helper_engine_state(EngineState.IDLE), start=reload)
            after_run = _helper_pool_worker_pids(eng._worker_pool)
            tg.cancel_scope.cancel()

    assert [len(pids) for pids in restarted] == [1], "the reload starts one warm worker"
    assert after_run == restarted[0], "the narrowed run should reuse the restarted worker"


@pytest.mark.anyio
async def test_watch_reload_keeps_going_past_a_pipeline_with_a_missing_input(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across all pipelines with keep-going, a reload re-runs the stages the first run could
    run and leaves out the ones a missing input breaks."""
    root = tmp_path.resolve() / "p"
    for name, source in [("width", _WIDTH_PIPELINE), ("broken", _BROKEN_PIPELINE)]:
        (root / name).mkdir(parents=True)
        (root / name / "pipeline.py").write_text(source)
    (root / ".pivot").mkdir()
    (root / ".git").mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    pipeline = discovery.discover_pipeline(root, all_pipelines=True)
    assert pipeline is not None, "the project should define pipelines"
    left = root / "data" / "width" / "left.txt"
    recorder = _Recorder()

    async with engine_mod.Engine(pipeline=pipeline, all_pipelines=True) as eng:
        eng.add_sink(recorder)
        eng.add_source(sources.FilesystemSource(watch_paths=[root], debounce_ms=50))
        eng.add_source(
            sources.OneShotSource(
                stages=None, force=True, reason="test", max_workers=8, on_error=OnError.KEEP_GOING
            )
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(functools.partial(eng.run, exit_on_completion=False))
            first_idle = await recorder.wait_for(_helper_engine_state(EngineState.IDLE))
            assert left.read_text() == "source left"
            (root / "width" / "pipeline.py").write_text(
                _WIDTH_PIPELINE.replace('" left"', '" LEFT"')
            )
            reload = await recorder.wait_for(
                lambda event: event["type"] == "pipeline_reloaded" and event["error"] is None,
                start=first_idle,
            )
            await recorder.wait_for(_helper_engine_state(EngineState.IDLE), start=reload)
            assert left.read_text() == "source LEFT"
            tg.cancel_scope.cancel()
