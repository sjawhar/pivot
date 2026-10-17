"""Worker processes across the runs of one long-lived Engine (watch mode, ``--serve``, the TUI).

Each test writes a project, discovers its pipeline the way the CLI does, and drives one Engine
through several runs, reading what its stages wrote: the worker's process id, or the value a
helper module returned.
"""

from __future__ import annotations

import contextlib
import functools
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING

import anyio
import pytest

from pivot import discovery, project
from pivot.engine import engine as engine_mod
from pivot.engine import sources
from pivot.engine.types import EngineState, OutputEvent, RunRequested
from pivot.executor import core as executor_core
from pivot.storage import store as store_mod
from pivot.types import OnError, StageStatus

if TYPE_CHECKING:
    from collections.abc import Callable

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

    async def request_forced_run(self) -> None:
        await self._send.send(
            RunRequested(
                type="run_requested",
                stages=None,
                force=True,
                reason="test",
                single_stage=False,
                parallel=True,
                max_workers=1,
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
