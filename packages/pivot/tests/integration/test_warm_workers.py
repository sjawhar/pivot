"""Worker processes across the runs of one long-lived Engine (watch mode, ``--serve``, the TUI).

Each test writes a project, discovers its pipeline the way the CLI does, and drives one Engine
through several runs, reading what its stages wrote: the worker's process id, or the value a
helper module returned.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
import pathlib
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
    executor_core.restart_workers(1, 1)
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
    executor_core.restart_workers(1, 1)
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
    executor_core.restart_workers(1, 1)
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
    executor_core.restart_workers(1, 1)
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
    executor_core.restart_workers(1, 1)
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
    executor_core.restart_workers(1, 1)
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
    executor_core.restart_workers(1, 1)
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

# Imported by every Python process the command starts: each loky worker records when it started,
# and the pivot process records when it starts the Textual TUI.
_WORKER_LOG_HOOK = """\
import os
import sys
import time


def _log(event):
    with open(os.environ["PIVOT_TEST_WORKER_LOG"], "a") as log:
        log.write(f"{event} {time.time()}\\n")


if sys.orig_argv[1:3] == ["-m", "loky.backend.popen_loky_posix"]:
    _log("worker")
elif len(sys.orig_argv) > 1 and os.path.basename(sys.orig_argv[1]) == "pivot":
    from pivot.cli import _run_common

    _run_tui_with_engine = _run_common.run_tui_with_engine

    def _logged_run_tui_with_engine(*args, **kwargs):
        _log("tui")
        return _run_tui_with_engine(*args, **kwargs)

    _run_common.run_tui_with_engine = _logged_run_tui_with_engine
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


def _helper_started_workers(worker_log: pathlib.Path, command: list[str]) -> int:
    """Count the workers the command started, requiring each to start before any TUI did."""
    events = [line.split() for line in worker_log.read_text().splitlines()]
    workers = [float(at) for event, at in events if event == "worker"]
    tuis = [float(at) for event, at in events if event == "tui"]
    late = [at for at in workers if tuis and at > tuis[0]]
    assert not late, f"{' '.join(command)}: {len(late)} of {len(workers)} workers started late"
    return len(workers)


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
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


def _helper_configure_workers(root: pathlib.Path, max_workers: int) -> None:
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  max_workers: {max_workers}\n")


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


@pytest.mark.parametrize(
    ("max_workers", "command", "workers"),
    [
        pytest.param(8, ["repro"], 3, id="repro"),
        pytest.param(8, ["repro", "--tui"], 3, id="repro-tui-prewarm"),
        pytest.param(2, ["repro"], 2, id="repro-configured-cap"),
        # `pivot run` orders left and right after source, so two of the three run at once.
        pytest.param(8, ["run", "--force", "source", "left", "right"], 2, id="run"),
        pytest.param(
            8, ["run", "--tui", "--force", "source", "left", "right"], 2, id="run-tui-prewarm"
        ),
    ],
)
def test_command_starts_as_many_workers_as_stages_can_run_at_once(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    max_workers: int,
    command: list[str],
    workers: int,
) -> None:
    """A command starts one worker per stage that can run at once, up to the configured worker
    count; a TUI's pre-warm starts the same workers its engine then uses."""
    root = tmp_path.resolve() / "project"
    _helper_project(root, monkeypatch, _WIDTH_PIPELINE)
    _helper_configure_workers(root, max_workers)
    env, worker_log = _helper_logging_env(tmp_path)

    _helper_pivot(root, env, *command)

    started = _helper_started_workers(worker_log, command)
    assert started == workers, f"{' '.join(command)} started {started} workers"


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
def test_tui_starts_its_workers_before_the_tui_when_a_tracked_input_is_restored(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """The pre-warm restores a missing tracked input before it sizes the pool, as the engine
    does, so no worker starts after Textual does."""
    root = tmp_path.resolve() / "project"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    _helper_configure_workers(root, 8)
    env, worker_log = _helper_logging_env(tmp_path)
    tracked = root / "data" / "raw" / "x.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked")
    _helper_pivot(root, env, "track", "data/raw/x.txt")
    tracked.unlink()
    worker_log.unlink(missing_ok=True)

    _helper_pivot(root, env, *command)

    assert (root / "data" / "width" / "lone.txt").read_text() == "tracked"
    started = _helper_started_workers(worker_log, command)
    assert started == 3, f"{' '.join(command)} started {started} workers"


def test_tui_checkout_missing_warns_once_about_a_changed_tracked_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-warm only restores missing tracked files; checking the others stays the
    engine's, so a tracked file edited since `pivot track` is reported once."""
    root = tmp_path.resolve() / "project"
    _helper_project(root, monkeypatch, _TRACKED_WIDTH_PIPELINE)
    env, _ = _helper_logging_env(tmp_path)
    tracked = root / "data" / "raw" / "x.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked")
    _helper_pivot(root, env, "track", "data/raw/x.txt")
    tracked.write_text("edited")

    output = _helper_pivot(root, env, "repro", "--tui", "--checkout-missing")

    assert output.count("has changed since tracking") == 1, output


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


def test_repro_tui_prewarm_leaves_out_the_stages_a_keep_going_run_skips(
    tmp_path: pathlib.Path,
) -> None:
    """Across all pipelines with keep-going, the engine leaves out stages with a missing
    dependency; the TUI's pre-warm leaves them out too, so it starts only the workers the
    engine then uses."""
    root = tmp_path.resolve() / "project"
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
    started = _helper_started_workers(worker_log, command)
    assert started == 3, f"{' '.join(command)} started {started} workers"


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


def _helper_live_worker_pids() -> set[int]:
    """The worker processes of Pivot's reusable loky executor."""
    reusable_executor = importlib.import_module("loky.reusable_executor")
    return set(cast("Any", reusable_executor)._executor._processes)


@pytest.mark.anyio
async def test_watch_reload_restarts_the_workers_the_next_run_uses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload that narrows the pipeline restarts as many workers as the warm pool holds, so
    the run after it starts no other worker."""
    root = tmp_path.resolve() / "project"
    pipeline = _helper_project(root, monkeypatch, _THREE_PID_PIPELINE)
    recorder = _Recorder()
    restarted = list[set[int]]()
    after_run = set[int]()
    restart_workers = executor_core.restart_workers

    def recording_restart(width: int, max_workers: int | None = None) -> int:
        workers = restart_workers(width, max_workers)
        restarted.append(_helper_live_worker_pids())
        return workers

    monkeypatch.setattr(executor_core, "restart_workers", recording_restart)
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
            after_run = _helper_live_worker_pids()
            tg.cancel_scope.cancel()

    assert [len(pids) for pids in restarted] == [3], "the reload restarts the warm pool's size"
    assert after_run == restarted[0], "the run after the reload started other workers"


@pytest.mark.anyio
async def test_watch_reload_keeps_going_past_a_pipeline_with_a_missing_input(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across all pipelines with keep-going, a reload re-runs the stages the first run could
    run and leaves out the ones a missing input breaks."""
    root = tmp_path.resolve() / "project"
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
