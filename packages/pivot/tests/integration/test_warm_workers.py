"""Worker processes across the runs of one long-lived Engine (watch mode, ``--serve``, the TUI).

Each test writes a project, discovers its pipeline the way the CLI does, and drives one Engine
through several runs, reading what its stages wrote: the worker's process id, or the value a
helper module returned.
"""

from __future__ import annotations

import functools
import os
import sys
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
    import pathlib
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
